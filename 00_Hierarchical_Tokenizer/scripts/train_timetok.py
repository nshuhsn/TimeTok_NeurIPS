import os, argparse, math, copy
import torch
import torch.nn.functional as F
import sys
from types import SimpleNamespace
from hydra.utils import instantiate
import pickle
from tqdm import tqdm

# --- PATHS ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # TimeTok_Repo (for data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 00_Hierarchical_Tokenizer (higher priority)

from data.UCR_data_factory import ucr_data_provider 
from data.UTSD_data_factory import utsd_data_provider
from data.Nasdaq_factory import nasdaq_data_provider
from data.ETTh1_factory import etth1_data_provider
from TimeTok_wrapper import TimeTok
from config.TimeTok import cfg as FT_CFG

import json
import numpy as np

def save_checkpoint(model, cfg, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"TimeTok_{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    # Save config for reproducibility (JSON-serializable values only)
    with open(os.path.join(save_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"💾 saved: {ckpt_path}")

def unpad_L(x, unpad_right: int):
    if unpad_right == 0:
        return x
    return x[..., :-unpad_right]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ECG5000")
    _default_ucr_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "UCRConverted")
    p.add_argument("--ucr_root", default=_default_ucr_root, help="Path to UCR dataset root") 
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    _default_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "03_Shared", "tokenizer_weights")
    p.add_argument("--save_dir", type=str, default=_default_save_dir)
    p.add_argument("--pz", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)    # 2048 for UTSD
    p.add_argument("--timesteps", type=int, default=25)      # Decoder denoising steps
    return p.parse_args()

# Calculate minimum padding
def pad_to_multiple_L(x, multiple):
    # x: [B,C,L] -> pad right to multiple of `multiple`
    L = x.shape[-1]
    need = (multiple - (L % multiple)) % multiple
    if need == 0: return x, 0
    return F.pad(x, (0, need), mode="reflect"), need

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_root = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_root, exist_ok=True)

    # 1) UCR Individual data
    ns = SimpleNamespace(dataset=args.dataset, data_root=args.ucr_root,
                         root_path=args.ucr_root, num_workers=0, seq_len=None, seed=args.seed)
    train_ls, val_ls, _, param = ucr_data_provider(ns)
    train_loader, val_loader = train_ls[0], val_ls[0]

    # 2) Nasdaq data (alternative option)
    # train_loader, val_loader, _, _ = nasdaq_data_provider(args.ucr_root, args.batch_size)

    # 3) ETTh1 data (alternative option)
    # train_loader, val_loader, _, _ = etth1_data_provider(args.ucr_root, args.batch_size)

    # 4) UTSD data (alternative option)
    # train_loader, val_loader = utsd_data_provider(args, "train"), utsd_data_provider(args, "val")
    
    Pz = args.pz

    # Config override (deep copy then modify)
    cfg = copy.deepcopy(FT_CFG)

    cfg["regularizer"]["levels"] = [4, 4, 4, 4, 4, 4]
    cfg["encoder"]["module_dict"]["enc_patch_emb"]["patch_sizes"] = [Pz]       
    cfg["decoder"]["module_dict"]["dec_noise_patch_emb"]["patch_sizes"] = [Pz]  
    cfg["decoder"]["module_dict"]["dec_to_patches"]["patch_sizes"] = [Pz]

    # Model instantiate 
    timetok: TimeTok = instantiate(cfg)
    timetok = timetok.to(device)

    # Optimizer (encoder/decoder/regularizer only)
    params = [p for p in timetok.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    best_val = float("inf")
    best_tag = None

    def loss_step(batch):
        # Input (always 3D: [B,C,L])
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        # Minimum padding (patch safe: multiple of Pz)
        x_pad, need = pad_to_multiple_L(x, multiple=Pz)
        
        data_dict = {timetok.encoder.module_dict.enc_channels_to_last.read_key: list(x_pad.split(1, dim=0))}

        out = timetok(data_dict)  # Internal: VAE.encode -> enc -> FSQ -> noise -> dec -> pipeline

        v_pred = torch.cat(out["ts_reconst"], dim=0)  # [B,C,L]
        x0    = torch.cat(out["ts"],          dim=0)  # [B,C,L]
        eps   = torch.cat(out["flow_noise"],  dim=0)  # [B,C,L]

        # Rectified Flow velocity target and MSE
        v_tgt = eps - x0  # [B,C,L]

        # Remove padding before loss calculation
        v_pred_unpad = unpad_L(v_pred, need)
        v_tgt_unpad  = unpad_L(v_tgt,  need)

        loss = torch.nn.functional.mse_loss(v_pred_unpad, v_tgt_unpad)

        return loss, {"fm_mse": float(loss.detach().item())}
    
    def loss_step_smoothing_gaussian(batch):
        # Input (always 3D: [B,C,L])
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        
        # Minimum padding (patch safe: multiple of Pz)
        x_pad, need = pad_to_multiple_L(x, multiple=Pz)

        # Construct input dict for pipeline (encoder first module read_key)
        data_dict = {timetok.encoder.module_dict.enc_channels_to_last.read_key: list(x_pad.split(1, dim=0))}

        # Forward pass (raw TS path: enc -> (register/nested) -> noise -> dec -> pipeline)
        out = timetok(data_dict)

        v_pred  = torch.cat(out["ts_reconst"], dim=0)  # Predicted velocity
        x_clean = torch.cat(out["ts"],         dim=0)  # Clean
        x_noise = torch.cat(out["flow_noise"], dim=0)  # Noise
        keep_ks = out["train_keep_k"]                  # [B] list(int)

        v_tgt = x_noise - x_clean

        # Remove padding before loss calculation
        v_pred_unpad = unpad_L(v_pred, need)
        v_tgt_unpad  = unpad_L(v_tgt,  need)

        loss = torch.nn.functional.mse_loss(v_pred_unpad, v_tgt_unpad)
        return loss, {"fm_mse": float(loss.detach().item())}

    # Training loop
    for epoch in range(args.epochs):
        timetok.train()
        tr_loss = 0.0
        
        # Add tqdm for training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs-1} [Train]")
        for batch in train_pbar:
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_step_smoothing_gaussian(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += loss.item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % 5 == 0:
            # Validation
            timetok.eval()
            with torch.no_grad():
                val_loss = 0.0
                
                # Add tqdm for validation loop
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs-1} [Val]")
                for batch in val_pbar:
                    loss, _ = loss_step(batch)
                    val_loss += loss.item()
                    
                    # Update progress bar with current loss
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            steps = max(1, len(train_loader))
            vsteps = max(1, len(val_loader))
            tr = tr_loss/steps
            va = val_loss/vsteps
            print(f"epoch {epoch}: train={tr:.4f} | val={va:.4f}")

            # Save best model
            if va < best_val:
                best_val = va
                best_tag = f"best_val_epoch{epoch:04d}"
                save_checkpoint(timetok, cfg, save_root, best_tag)
                best_path = os.path.join(save_root, f"timetok_{best_tag}.pt")
                print(f"✓ New best model saved: {best_tag} (val_loss: {va:.4f})")

if __name__ == "__main__":
    main()