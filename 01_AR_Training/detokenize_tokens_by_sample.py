import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from types import SimpleNamespace
from hydra.utils import instantiate
from collections import defaultdict
import gc

# --- PATHS ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # Parent of 01_AR_Training = TimeTok_Repo

# Add paths for module imports
# IMPORTANT: Order matters! 00_Hierarchical_Tokenizer must be first (index 0)
# because it has a 'utils' package, and REPO_DIR has a 'utils.py' file.
sys.path.insert(0, REPO_DIR)  # For data.UCR_data_factory, data.UTSD_data_factory, etc.
sys.path.insert(0, os.path.join(REPO_DIR, "00_Hierarchical_Tokenizer"))  # For TimeTok_wrapper (higher priority)


from data.UCR_data_factory import ucr_data_provider 
from data.UTSD_data_factory import utsd_data_provider
from data.Nasdaq_factory import nasdaq_data_provider
from data.ETTh1_factory import etth1_data_provider
from TimeTok_wrapper import TimeTok

# Setup
from utils.misc import detect_bf16_support, get_bf16_context, get_generator
enable_bf16 = detect_bf16_support()

# ----------------------------------------------------------------
# 2. BASE CONFIGURATION (via argparse)
# ----------------------------------------------------------------

parser = argparse.ArgumentParser(description="Detokenize token sequences to time series")
parser.add_argument("--dataset_names", type=str, nargs="+", default=["ECG5000"],
                    help="List of dataset names to process (e.g., ECG5000)")
parser.add_argument("--tokenizer_name", type=str, default="ECG5000_tokenizer",
                    help="Name of the tokenizer checkpoint directory")
parser.add_argument("--ckpt_root", type=str, default="../03_Shared/tokenized_sequences",
                    help="Root directory for tokenizer checkpoints")
parser.add_argument("--exp_num", type=int, default=9999,
                    help="Experiment number to load token sequences from AR Training")
parser.add_argument("--mixed_prompt", action="store_true", default=False,
                    help="Use mixed prompt length")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="GPU device ID")
parser.add_argument("--patch_size", type=int, default=3,
                    help="Patch size used for tokenization")
parser.add_argument("--denormalize", action="store_true", default=False,
                    help="Denormalize output data")

args = parser.parse_args()

# Assign parsed arguments to variables
DATASET_NAME_ONLY_LIST = args.dataset_names
TOKENIZER_NAME = args.tokenizer_name
ABS_CKPT_ROOT = args.ckpt_root
EXP_NUM = args.exp_num
MIXED_PROMPT = args.mixed_prompt
gpu_id = args.gpu_id
PATCH_SIZE = args.patch_size
denormalize = args.denormalize

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")




REPO_DIR = os.path.dirname("../")  # TimeTok_Repo
DATA_DIR = os.path.join(REPO_DIR, "data")
UCR_ROOT = os.path.join(DATA_DIR, "UCRConverted")

CKPT_ROOT = f"{ABS_CKPT_ROOT}/{TOKENIZER_NAME}"
PROMPT_LEN = 0 # This is usually set to 0. 

# ----------------------------------------------------------------
# 3. HELPER FUNCTIONS AND DATA MAPPINGS
# ----------------------------------------------------------------
data_id_2_tslen = {
    0: 140,   # ECG5000
    1: 24,   # ItalyPowerDemand
    2: 244,   # Nasdaq
    3: 144,   # ETTh1
}

data_id_2_dataset_name = {
    0: "ECG5000",
    1: "ItalyPowerDemand",
    2: "Nasdaq",
    3: "ETTh1"
}
datasetname_2_id = {val: key for key, val in data_id_2_dataset_name.items()}

def load_model(dataset_name, patch_size, ckpt_dir):
    """Load TSgen model from checkpoint for specific dataset.""" 
    # Find best checkpoint
    best_checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "TimeTok_best_val_epoch*.pt")))
    assert len(best_checkpoints) > 0, f"No best checkpoint found in: {ckpt_dir}"
    best_ckpt_path = max(best_checkpoints, key=os.path.getmtime)
    
    # Load configuration
    config_path = os.path.join(ckpt_dir, "config_snapshot.json")
    assert os.path.isfile(config_path), f"Config file not found: {config_path}"
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Update patch sizes in config
    config["encoder"]["module_dict"]["enc_patch_emb"]["patch_sizes"] = [patch_size]
    config["decoder"]["module_dict"]["dec_noise_patch_emb"]["patch_sizes"] = [patch_size]
    config["decoder"]["module_dict"]["dec_to_patches"]["patch_sizes"] = [patch_size]
    
    # Instantiate and load model
    tsgen = instantiate(config)
    state_dict = torch.load(best_ckpt_path, map_location=device)
    tsgen.load_state_dict(state_dict)
    tsgen = tsgen.to(device).eval()
    
    print(f"✓ Loaded model from: {best_ckpt_path}")
    return tsgen

def load_tokens_from_jsonl(jsonl_path, dataset_id):
    """
    Load pre-computed tokens from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file containing tokens
        dataset_id: Dataset ID
    
    Returns:
        List of dictionaries containing token data
    """
    token_data = []
    # check if jsonl_path exists
    if not os.path.exists(jsonl_path):
        print(f"JSONL file not found: {jsonl_path}")
        return None
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['dataset_id'] != dataset_id:
                continue
            token_data.append(data)
    
    print(f"✓ Loaded {len(token_data)} token sequences from {jsonl_path}")
    return token_data

def detokenize_tokens(token_data_list, model, 
                        patch_size: int = 16, 
                        limit_input_token_len: int = None,
                        seed: int = 42):
    """
    Detokenize pre-computed tokens using the TSgen model.
    
    Args:
        token_data_list: List of dictionaries containing token data from JSONL
                        Each dict should have 'tokens' key with list of token integers
        model: TSgen model for detokenization
        patch_size: Patch size used for tokenization training.
        limit_input_token_len: Limit the input token number. This is used for Coarse-to-fine generation.
        seed: Random seed for generation
    Returns:
        reconstructed_data: Tensor of shape [B, C, L] containing reconstructed time series
        label_list: List of labels
    """   
    device = next(model.parameters()).device
    enable_bf16 = True  # Adjust based on your setup
    
    # Convert token data to format expected by model
    token_sequences = []
    label_list = []
    
    def pad_to_multiple_L(seq_len, multiple):
        """ Pad to multiple of L"""
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return 0
        return pad_len

    def unpad(x, unpad_right):
        """Remove right padding from tensor."""
        print("Unpad right: ", unpad_right)
        if unpad_right == 0:
            return x
        return x[..., :-unpad_right]

    for token_data in token_data_list:
        tokens = torch.tensor(token_data['tokens'], dtype=torch.long, device=device)
        # remove BOS token
        tokens = tokens[1:]
        label_list.append(token_data['label'])
        # Reshape to [1, token_len] format expected by model
        tokens = tokens.unsqueeze(0)
        if limit_input_token_len is not None:
            tokens = tokens[..., :limit_input_token_len]
        else:
            tokens = tokens[..., :]
        # assert len(tokens[0]) == 128, f"Tokens length should be 128, but got {len(tokens)}"
        token_sequences.append(tokens)

    
    original_ts_len = token_data_list[0]['original_ts_len']
    print(original_ts_len)
    pad_len = pad_to_multiple_L(original_ts_len, patch_size)
    print("Pad length: ", pad_len)
    padded_ts_len = original_ts_len + pad_len
    print("Padded TS length: ", padded_ts_len)
    
    # Perform detokenization
    with get_bf16_context(enable_bf16):
        generator = get_generator(seed=seed, device=device)
        reconstructed_data = model.detokenize(
            token_sequences,
            timesteps=25,
            guidance_scale=1,
            perform_norm_guidance=False,
            ts_image_sizes=padded_ts_len,
            generator=generator,
            verbose=False,
        )
    reconstructed_data = unpad(reconstructed_data, pad_len)
    return reconstructed_data, label_list

def denormalize_output(data, min_val, max_val):
    return (data * max_val) + min_val

def clear_gpu_memory():
    """Helper function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ----------------------------------------------------------------
# 4. MAIN PROCESSING LOOP
# ----------------------------------------------------------------
print(f"Processing {len(DATASET_NAME_ONLY_LIST)} datasets: {DATASET_NAME_ONLY_LIST}")

for dataset_idx, DATASET_NAME_ONLY in enumerate(DATASET_NAME_ONLY_LIST):
    print(f"\n{'='*60}")
    print(f"Processing dataset {dataset_idx + 1}/{len(DATASET_NAME_ONLY_LIST)}: {DATASET_NAME_ONLY}")
    print(f"{'='*60}")
    
    # Dataset-specific configuration
    DATASET = f"{DATASET_NAME_ONLY}_pz{PATCH_SIZE}"
    CKPT_DIR = f"{CKPT_ROOT}/{DATASET}"


    SYNTH_DIR = f"../03_Shared/AR_outputs/{DATASET}/EXP{EXP_NUM}/synthetic_tokens"
    

    if MIXED_PROMPT:
        print("Using mixed prompt length!")
        TRAIN_JSONL_PATH = f"{DATASET}_train_synth_tokens_promptlenmixed.jsonl"
        VAL_JSONL_PATH = f"{DATASET}_val_synth_tokens_promptlenmixed.jsonl"
        TEST_JSONL_PATH = f"{DATASET}_test_synth_tokens_promptlenmixed.jsonl"
    else:
        print("Using fixed prompt length!")
        TRAIN_JSONL_PATH = f"{DATASET}_train_synth_tokens00.jsonl"
        VAL_JSONL_PATH = f"{DATASET}_val_synth_tokens00.jsonl"
        TEST_JSONL_PATH = f"{DATASET}_test_synth_tokens00.jsonl"


    train_synth_jsonl_path = f"{SYNTH_DIR}/{TRAIN_JSONL_PATH}"
    val_synth_jsonl_path = f"{SYNTH_DIR}/{VAL_JSONL_PATH}"
    test_synth_jsonl_path = f"{SYNTH_DIR}/{TEST_JSONL_PATH}"
    
    print(f"Dataset: {DATASET}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print(f"Synthetic dir: {SYNTH_DIR}")
    print(f"Train JSONL: {train_synth_jsonl_path}")
    print(f"Val JSONL: {val_synth_jsonl_path}")
    print(f"Test JSONL: {test_synth_jsonl_path}")
    
    # Check if required files exist
    required_files = [train_synth_jsonl_path, val_synth_jsonl_path, test_synth_jsonl_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("*"*50)
        print(f"Missing files for {DATASET_NAME_ONLY}: {missing_files}")
        print(f"Skipping {DATASET_NAME_ONLY}...")
        print("*"*50)
        continue
    
    ############## UCR data ##############
    #Get dataset parameters
    ns = SimpleNamespace(dataset=DATASET_NAME_ONLY, data_root=UCR_ROOT, root_path=UCR_ROOT,
                         num_workers=0, seq_len=None, seed=42)
    _, _, _, param = ucr_data_provider(ns)


    print(f"Dataset parameters: {param}")
    
    # normalization factor
    MIN_VALUE_SCALER = param['min_val']
    MAX_VALUE_SCALER = param['max_val']
    print(f"Normalization - Min: {MIN_VALUE_SCALER}, Max: {MAX_VALUE_SCALER}")
    
    # Load model for this dataset
    print(f"Loading model for {DATASET_NAME_ONLY}...")
    model = load_model(DATASET_NAME_ONLY, PATCH_SIZE, CKPT_DIR)
    
    # Setup synthetic data paths
    synth_jsonl_dataset = {
        "train": {"prompt_len": 0, "jsonl_path": train_synth_jsonl_path},
        "val": {"prompt_len": 0, "jsonl_path": val_synth_jsonl_path},
        "test": {"prompt_len": 0, "jsonl_path": test_synth_jsonl_path},
    }
    synth_jsonl_dataset = defaultdict(dict, synth_jsonl_dataset)
    
    data_id = datasetname_2_id[DATASET_NAME_ONLY]
    data_name = data_id_2_dataset_name[data_id]
    print(f"Processing dataset ID {data_id}: {data_name}")


    # Create output directory
    synthetic_data_dir = f"../03_Shared/AR_outputs/{DATASET}/EXP{EXP_NUM}/synthetic_data"
    if not os.path.exists(synthetic_data_dir):
        print(f"Creating directory: {synthetic_data_dir}")
        os.makedirs(synthetic_data_dir)
    
    # Process each split separately to save memory
    for split_key in ['train', 'val', 'test']:
        print(f"\n  Processing {split_key} split for {DATASET_NAME_ONLY}")
        
        # Load token data for this split only
        token_data = load_tokens_from_jsonl(
            synth_jsonl_dataset[split_key]["jsonl_path"], 
            dataset_id=data_id
        )
        
        if not token_data:
            print(f"No token data found for {split_key} split, skipping...")
            continue
        
        # Store generated data temporarily for combination creation
        generated_data = {}
        label_list = None
        
        # Generate synthetic data with different seeds and token limits
        for limit_input_token_len in [128]: # Generate with full token.
            print(f"Generating data: token_len={limit_input_token_len}")
            
            # Add chunking to prevent GPU OOM
            chunk_size = 512  # Adjust based on GPU memory
            synth_data_chunks = []
            label_list = []
            
            # Process token_data in chunks
            for i in range(0, len(token_data), chunk_size):
                chunk_end = min(i + chunk_size, len(token_data))
                token_data_chunk = token_data[i:chunk_end]
                
                print(f" Processing chunk {i//chunk_size + 1}/{(len(token_data) + chunk_size - 1)//chunk_size}")
                
                chunk_synth_data, chunk_label_list = detokenize_tokens(
                    token_data_chunk, 
                    model, 
                    patch_size=PATCH_SIZE, 
                    limit_input_token_len=limit_input_token_len,
                    seed=42
                )
                # De-normalize 
                if denormalize:
                    chunk_synth_data = denormalize_output(chunk_synth_data, MIN_VALUE_SCALER, MAX_VALUE_SCALER)
                # Move to CPU immediately to save GPU memory
                synth_data_chunks.append(chunk_synth_data.to('cpu'))
                label_list.extend(chunk_label_list)
                
                # Clear GPU memory after each chunk
                del chunk_synth_data, chunk_label_list
                clear_gpu_memory()
            
            # Concatenate all chunks
            synth_data = torch.cat(synth_data_chunks, dim=0)
            
            # Clean up chunk data
            del synth_data_chunks
            clear_gpu_memory()
            
            # Store temporarily (already on CPU)
            key = f"token{limit_input_token_len}"
            generated_data[key] = synth_data.squeeze()
            
            # Clear GPU memory after each generation
            clear_gpu_memory()
        
        # Clear token data from memory
        del token_data
        clear_gpu_memory()
        
        # Create key combinations
        data_keys = list(generated_data.keys())
        print(f"Generated keys: {data_keys}")
        
    
        
        # Convert label_list to numpy once
        data_label = np.array(label_list)
        
        # Process and save each combination type immediately
        for data_level_type in generated_data.keys():
                
            print(f"    Processing combination: {data_level_type}")
            
            detokenized_data = []
            detokenized_data_label = []
            
            # Collect data for this combination
            # for combination_key in combination_keys:
            detokenized_data.append(generated_data[data_level_type].numpy())
            detokenized_data_label.append(data_label)
            
            # Concatenate and validate
            final_data = np.concatenate(detokenized_data, axis=0)
            final_labels = np.concatenate(detokenized_data_label, axis=0)
            
            expected_length = data_id_2_tslen[data_id]
            actual_length = final_data.shape[-1]
            assert actual_length == expected_length, \
                f"Sequence length mismatch for {data_name}: {actual_length} != {expected_length}"
            
            # Save immediately
            output_file = os.path.join(synthetic_data_dir, f"synthetic_{data_level_type}_{split_key}.npy")
            np.save(output_file, {
                'X': final_data,
                'y_true': final_labels
            })
            # save params
            params_file = os.path.join(synthetic_data_dir, f"params.json")
            with open(params_file, "w") as f:
                json.dump({
                    "MIN_VALUE_SCALER": MIN_VALUE_SCALER,
                    "MAX_VALUE_SCALER": MAX_VALUE_SCALER
                }, f)
            print(f"Saved: {output_file}")
            
            # Clean up intermediate data
            del detokenized_data, detokenized_data_label, final_data, final_labels
            clear_gpu_memory()
        
        # Clean up all generated data for this split
        del generated_data, data_label, label_list
        clear_gpu_memory()
        
        print(f"Completed {split_key} split for {DATASET_NAME_ONLY}")
    
    # Clean up model after processing this dataset
    del model
    clear_gpu_memory()
    
    print(f"Completed dataset: {DATASET_NAME_ONLY}")

print(f" All datasets processed successfully!")
print(f"Processed datasets: {DATASET_NAME_ONLY_LIST}")