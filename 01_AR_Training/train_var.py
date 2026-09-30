import hydra
from omegaconf import DictConfig
import sys
import random
import math
import torch
import os
import glob
import json
from torch.utils.data import DataLoader
import tqdm
from data.dataset import CustomTokenDataset, CustomForecastDataset
from ar_pytorch.var_pytorch import MultiScaleVARTransformer
from ar_pytorch.visualautoregressive_wrapper import VARAutoregressiveWrapper
from utils import decode_tokens, get_lr, bcolors, make_dir, get_next_batch, get_prompt_length, remove_existing_dir
from logger.wandb_logger import WandbLogger
from einops import rearrange
import torch.nn.functional as F
from generate_samples import generate_samples_fixed_length, generate_samples_mix_prompt_length
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="confs", config_name="config")
def main(cfg: DictConfig) -> None:
    remove_existing_dir(cfg.save_output_path)
    make_dir(cfg.save_output_path)
    logger = WandbLogger(cfg)
    
    # Training constants
    NUM_BATCHES = int(1e5)
    BATCH_SIZE = cfg.trainer.dataset.batch_size
    GRADIENT_ACCUMULATE_EVERY = cfg.trainer.optimizer.gradient_accumulate_every
    LEARNING_RATE = cfg.trainer.optimizer.lr
    VALIDATE_EVERY = cfg.trainer.callbacks.check_val_every_n_steps
    GENERATE_EVERY = cfg.trainer.callbacks.generate_every_n_steps
    GENERATE_LENGTH = cfg.data.max_seq_len + 1  # +1 for BOS
    WARMUP_STEPS = cfg.trainer.optimizer.warmup_steps
    
    
    # Device setup
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"{bcolors.OKGREEN}Using device: {device}{bcolors.ENDC}")
    
    # Initialize datasets
    print("Loading datasets...")
    if cfg.task.task_type == "classification":
        print("Loading classification datasets...")
        train_dataset = CustomTokenDataset(cfg, mode="train")
        val_dataset = CustomTokenDataset(cfg, mode="val")
        test_dataset = CustomTokenDataset(cfg, mode="test")
    elif cfg.task.task_type == "forecasting":
        print("Loading forecasting datasets...")
        train_dataset = CustomForecastDataset(cfg, mode="train")
        val_dataset = CustomForecastDataset(cfg, mode="val")
        test_dataset = CustomForecastDataset(cfg, mode="test")
    else:
        raise ValueError(f"Invalid task type: {cfg.task.task_type}")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Token sequence length: {len(train_dataset[0])} (including BOS)")
    print(f"Vocabulary size: {cfg.data.total_vocab_size}")

    # Calculate batches per epoch for epoch tracking
    batches_per_epoch = len(train_dataset) // BATCH_SIZE
    total_epochs = NUM_BATCHES / batches_per_epoch
    print(f"Batches per epoch: {batches_per_epoch}")
    print(f"Total epochs (approx): {total_epochs:.2f}")

    # Get special token IDs
    bos_id = train_dataset.bos_id
    mask_id = train_dataset.mask_id

    print(f"Special tokens - BOS: {bos_id}, Mask: {mask_id}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True if device.type == 'cuda' else False)

    # Initialize model
    print("Initializing model...")
    model = MultiScaleVARTransformer(
        num_tokens=cfg.data.total_vocab_size,
        dim=cfg.ar_model.dim,
        depth=cfg.ar_model.depth,
        heads=cfg.ar_model.heads,
        dim_head=cfg.ar_model.dim_head,
        max_seq_len=GENERATE_LENGTH,
        max_class_num=cfg.data.max_class_num,
        dropout=cfg.ar_model.dropout,
        mask_id=cfg.data.mask_id,
        condition_on_class=cfg.ar_model.condition_on_class,
        condition_on_dataset=cfg.ar_model.condition_on_dataset,
        classifier_free_guidance_prob=cfg.ar_model.classifier_free_guidance_prob
    )
    model = VARAutoregressiveWrapper(model)
    model = model.to(device)
    
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                                 weight_decay=cfg.trainer.optimizer.weight_decay)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    best_val_loss = float('inf')
    patience_counter = 0
    best_i = -1
    current_val_loss = float('inf')

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
        # Calculate current epoch
        current_epoch = i // batches_per_epoch
        batch_in_epoch = i % batches_per_epoch
        
        # Update learning rate
        lr = get_lr(i, WARMUP_STEPS, LEARNING_RATE, total_steps=NUM_BATCHES)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()
        total_loss = 0
        
        # Gradient accumulation
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            train_iter, batch, y_true, dataset_id = get_next_batch(train_iter, train_loader, device)
            
            # VAR next-scale prediction happens here automatically
            loss = model(batch, y_true, dataset_id)
            loss = loss / GRADIENT_ACCUMULATE_EVERY
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Log metrics
        logger.log_metric({
            "train_loss": total_loss, 
            "lr": lr, 
            "epoch": current_epoch,
        })
        
        if i % 10 == 0:  
            print(f"Epoch {current_epoch}, Step {i} ({batch_in_epoch}/{batches_per_epoch}), "
                  f"LR: {lr:.2e}, Training loss: {total_loss:.4f}")

        # Generation examples
        if i % GENERATE_EVERY == 0 and i > 10:
            model.eval()
            with torch.no_grad():
                # Get a validation sample for prompt
                val_iter, sample_batch, sample_y_true, sample_dataset_id = get_next_batch(val_iter, val_loader, device)
                
                # Randomly choose prompt length from the list
                prompt_length_options = [0, 1, 2, 4, 8, 16, 32, 64]
                prompt_length = random.choice(prompt_length_options) + 1
                prompt = sample_batch[0, : prompt_length].unsqueeze(0)
                sample_y_true = sample_y_true[0].unsqueeze(0)
                sample_dataset_id = sample_dataset_id[0].unsqueeze(0)
                
                # Generate until full sequence length
                target_length = GENERATE_LENGTH - prompt_length 

                print("\n" + "="*80)
                print("GENERATION EXAMPLE (VAR Style - Coarse-to-Fine):")
                print(f"Prompt length: {prompt_length} tokens (Including BOS)")
                print(f"Will generate: {target_length} tokens")
                print(f"Prompt: {decode_tokens(prompt[0].cpu().numpy(), cfg.data.total_vocab_size, bos_id)}")
        
                # VAR-style generation (scale-by-scale)
                generated = model.generate_var_style(
                    start_tokens=prompt, 
                    y_true=sample_y_true,
                    dataset_id=sample_dataset_id,
                    temperature=cfg.ar_model.generation.inference_temperature,
                    top_k=50,
                )
                print(f"Generated length: {len(generated[0])}")
                

                output_str = decode_tokens(generated[0].cpu().numpy(), cfg.data.total_vocab_size, bos_id, color_print="cyan")
                print(f"Generated: {output_str}")
                print("="*80)
                
                # Print true sequences for comparison
                true_sequence = sample_batch[0].cpu().numpy()
                true_sequence = true_sequence[prompt_length:prompt_length+target_length]
                true_output_str = decode_tokens(true_sequence, cfg.data.total_vocab_size, bos_id, color_print="green")
                print(f"True sequence: {true_output_str}")


        if i > cfg.trainer.callbacks.max_steps:
            print(f"{bcolors.WARNING}Reached max steps: {cfg.trainer.callbacks.max_steps}{bcolors.ENDC}")
            best_val_loss = current_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'train_loss': total_loss,
                'val_loss': current_val_loss,
                'config': {
                    'vocab_size': cfg.data.total_vocab_size,
                    'max_token_size': cfg.data.max_token_size,
                    'bos_id': bos_id,
                }
            }
            torch.save(checkpoint, f'{cfg.save_output_path}/checkpoint_step_{i}.pt')
            best_i = i
            
            # Save config as snapshot
            with open(f'{cfg.save_output_path}/config_snapshot.yaml', 'w') as f:
                OmegaConf.save(cfg, f)
            break
    
    print(f"{bcolors.OKGREEN}Training completed!{bcolors.ENDC}")

    # Generate synthetic samples
    if cfg.generate_samples:
        print("Generating synthetic samples...This may take a while.")
        
        # Reinitialize data loaders for generation
        del train_loader, val_loader, test_loader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, 
                                num_workers=4, pin_memory=True if device.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                              num_workers=4, pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                               num_workers=4, pin_memory=True if device.type == 'cuda' else False)
        
        # Load best model
        best_checkpoint_path_ar = f'{cfg.save_output_path}/checkpoint_step_{best_i}.pt'
        
        # Remove all other checkpoints
        for checkpoint_path in glob.glob(f'{cfg.save_output_path}/checkpoint_step_*.pt'):
            if checkpoint_path != best_checkpoint_path_ar:
                os.remove(checkpoint_path)
        
        checkpoint = torch.load(best_checkpoint_path_ar, map_location=device)
        print(f"Loaded checkpoint: {best_checkpoint_path_ar}")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Generate samples
        splits_info = [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader)
        ]
        synthetic_samples_output_path = f"../03_Shared/AR_outputs/{cfg.data.data_name}/EXP{cfg.exp_num}/synthetic_tokens/"
        
        if cfg.generate_mix_prompt_length:
            # This is just used for checking. 
            prompt_len_list = [0, 1, 2, 4, 8, 16, 32, 64]
            generate_samples_mix_prompt_length(cfg, model, device, prompt_len_list, 
                                              splits_info, synthetic_samples_output_path, generate_type='var')
        else:
            # This is used for final generation.
            prompt_len_list = [0]
            generate_samples_fixed_length(cfg, model, device, prompt_len_list, 
                                         splits_info, synthetic_samples_output_path, generate_type='var')


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    main()
