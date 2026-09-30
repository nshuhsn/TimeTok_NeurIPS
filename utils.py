import math
import os
import random
import shutil

# Color codes for terminal output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_lr(step, warmup_steps, max_lr, min_lr=1e-5, total_steps=100000):
    """Learning rate scheduler with warmup and cosine decay"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

def decode_tokens(tokens, vocab_size, bos_id, skip_special_tokens=True, color_print=None):
    """Decode tokens back to readable format"""

    result = []
    for token in tokens:
        if skip_special_tokens:
            if token == bos_id:
                result.append("<BOS>")
                continue
        
        # For custom tokens, just show the token ID
        result.append(f"[{token}]")

    if color_print == "green":
        return f"{bcolors.OKGREEN} {result} {bcolors.ENDC}"
    elif color_print == "cyan":
        return f"{bcolors.OKCYAN} {result} {bcolors.ENDC}"
    else:
        return " ".join(result) 


def make_dir(path):
    if not os.path.exists(path):
        print(f"{bcolors.OKGREEN}Making directory: {path}{bcolors.ENDC}")
        os.makedirs(path)


def remove_existing_dir(path):
    if os.path.exists(path):
        print(f"{bcolors.WARNING}Removing existing directory: {path}{bcolors.ENDC}")
        shutil.rmtree(path)

def get_next_batch(data_iter, data_loader, device):
    """
    Get next batch from iterator, reinitializing if needed.
    
    Args:
        data_iter: Current data iterator
        data_loader: Data loader
        device: Device to move data to
    
    Returns:
        tuple: (updated iterator, batch)
    """
    try:
        batch = next(data_iter)
        full_sequence = batch["full_sequence"].to(device)
        y_true = batch["y_true"].to(device)
        dataset_id = batch["dataset_id"].to(device)

    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
        full_sequence = batch["full_sequence"].to(device)
        y_true = batch["y_true"].to(device)
        dataset_id = batch["dataset_id"].to(device)
    return data_iter, full_sequence, y_true, dataset_id

def get_prompt_length(cfg):
    if cfg.validation.validation_strategy == "fix_prompt_length":
        prompt_length = cfg.validation.prompt_length
        assert prompt_length in [1, 2, 4, 8, 16, 32, 64], "Prompt length must be in [1, 2, 4, 8, 16, 32, 64]"
    elif cfg.validation.validation_strategy == "random_prompt_length":
        prompt_length = random.choice([1, 2, 4, 8, 16, 32, 64])
    elif cfg.validation.validation_strategy == "bos_only":
        prompt_length = 0
    else:
        raise ValueError(f"Invalid validation strategy: {cfg.validation.validation_strategy}")
    return prompt_length + 1 # +1 for BOS token