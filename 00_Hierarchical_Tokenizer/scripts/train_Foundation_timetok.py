import os, argparse, math, copy, pickle


import torch
import torch.nn.functional as F
import sys
from types import SimpleNamespace
from hydra.utils import instantiate
from tqdm import tqdm

# --- PATHS ---
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # TimeTok_Repo (for data/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 00_Hierarchical_Tokenizer (higher priority)

from data.UCR_data_factory import ucr_data_provider
from TimeTok_wrapper import TimeTok
from config.Foundation_TimeTok import cfg as FT_CFG
from data.UTSD_data_factory import utsd_data_provider

import json
import numpy as np

def save_checkpoint(model, cfg, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"TimeTok_{tag}.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(save_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"💾 saved: {ckpt_path}")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="UTSD")
    _default_utsd_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "UTSD-full-npy")
    p.add_argument("--utsd_root", default=_default_utsd_root, help="Path to UTSD dataset root") 
    _default_ucr_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "UCRConverted")
    p.add_argument("--ucr_root", default=_default_ucr_root, help="Path to UCR dataset root") 
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2.5e-4)  # 1e-3 for UTSD
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    _default_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "03_Shared", "tokenizer_weights")
    p.add_argument("--save_dir", type=str, default=_default_save_dir)
    p.add_argument("--pz", type=int, default=3)
    p.add_argument("--subset_ratio", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=512)    # 2048 for UTSD
    p.add_argument("--val_batch_size", type=int, default=64)  # Needed for UTSD to avoid recompilation issues
    p.add_argument("--timesteps", type=int, default=25)       # Decoder denoising steps
    return p.parse_args()

def pad_to_multiple_L(x, multiple):
    # x: [B,C,L] -> pad right to multiple of `multiple`
    L = x.shape[-1]
    need = (multiple - (L % multiple)) % multiple
    if need == 0: return x, 0
    return F.pad(x, (0, need), mode="reflect"), need

def unpad_L(x, unpad_right: int):
    if unpad_right == 0:
        return x
    return x[..., :-unpad_right]

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_root = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_root, exist_ok=True)

    # 1) UTSD data
    train_loader, val_loader = utsd_data_provider(args, "train"), utsd_data_provider(args, "val")

    # 2) UCR Individual data (alternative option)
    # ns = SimpleNamespace(dataset=args.dataset, data_root=args.ucr_root,
    #                      root_path=args.ucr_root, num_workers=0, seq_len=None, seed=args.seed)
    # train_ls, val_ls, _, param = ucr_data_provider(ns)
    # train_loader, val_loader = train_ls[0], val_ls[0]

    print(f"Successfully loaded train and val dataloaders")
    
    # Config setup
    cfg = copy.deepcopy(FT_CFG)

    if args.dataset == "UTSD":
        cfg["encoder"]["module_dict"]["sigma_level_controller"]["alpha_edges"] = [1.6049, 1.7895, 1.912, 1.9751, 2.0152, 2.0437, 2.0666]
    elif args.dataset == "ECG5000":
        cfg["encoder"]["module_dict"]["sigma_level_controller"]["alpha_edges"] = [1.553, 1.7421, 1.8511, 1.9343, 1.9968, 2.035, 2.0532]
    else:
        # If you want to use other datasets, please run the following notebook to get the alpha_edges:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please calculate alpha_edges using DFA_Binning.ipynb and add it here.")
    
    Pz = cfg["encoder"]["module_dict"]["enc_patch_emb"]["patch_sizes"][0]

    # Model instantiate 
    timetok: TimeTok = instantiate(cfg)
    timetok = timetok.to(device)

    # Optimizer
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

        out = timetok(data_dict)

        v_pred = torch.cat(out["ts_reconst"], dim=0)
        x0 = torch.cat(out["ts"], dim=0)
        eps = torch.cat(out["flow_noise"], dim=0)

        v_tgt = eps - x0

        v_pred_unpad = unpad_L(v_pred, need)
        v_tgt_unpad = unpad_L(v_tgt, need)

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
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs-1} [Train]")
        for batch in train_pbar:
            opt.zero_grad(set_to_none=True)
            loss, _ = loss_step_smoothing_gaussian(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += loss.item()

        if epoch % 2 == 0:
            # Validation
            timetok.eval()
            
            # Clear GPU cache before validation
            torch.cuda.empty_cache()
            
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs-1} [Val]")
            
            with torch.no_grad():
                for batch in val_pbar:
                    loss, _ = loss_step(batch)
                    val_loss += loss.item()
                    
                    # Clear batch from memory immediately
                    del batch, loss
                
            # Clear cache after validation
            torch.cuda.empty_cache()

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
                best_path = os.path.join(save_root, f"TimeTok_{best_tag}.pt")

    # Save final snapshot
    save_checkpoint(timetok, cfg, save_root, "last")

    print(f"✅ training done. best_val={best_val:.4f} ({best_tag})")

if __name__ == "__main__":
    main()
