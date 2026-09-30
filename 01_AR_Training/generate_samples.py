import os
import json
import shutil
from tqdm import tqdm
from utils import bcolors
import torch
import random

def generate_samples_fixed_length(cfg, model, device, prompt_len_list, splits_info, output_dir, generate_type='var'):
    GENERATE_LENGTH = cfg.data.max_seq_len + 1 # +1 for BOS

    # Remove existing directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"{bcolors.WARNING}Removing existing directory: {output_dir}{bcolors.ENDC}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{bcolors.OKGREEN}Created directory to save synthetic samples: {output_dir}{bcolors.ENDC}")

    # add +1 for BOS token to prompt_len
    prompt_len_list_with_bos = [prompt_len + 1 for prompt_len in prompt_len_list]
    
    # Add tqdm progress bar for prompt length processing
    for prompt_len in tqdm(prompt_len_list_with_bos, 
                        desc="Processing prompt lengths", 
                        unit="length",
                        colour="green"):
        sample_id = 0
        target_length = GENERATE_LENGTH - prompt_len
        
        # Add progress bar for splits within each prompt length
        for split, loader in tqdm(splits_info, 
                                desc=f"Prompt len {prompt_len-1}", 
                                unit="split", 
                                leave=False):
            output_file_path = f'{output_dir}/{cfg.data.data_name}_{split}_synth_tokens{str(prompt_len-1).zfill(2)}.jsonl'
            with open(output_file_path, 'a') as f:
                # Add progress bar for batches
                for batch in tqdm(loader, 
                                desc=f"Generating {split}", 
                                unit="batch", 
                                leave=False):
                    full_sequence = batch["full_sequence"].to(device)
                    y_true = batch["y_true"].to(device)
                    dataset_id = batch["dataset_id"].to(device)

                    if target_length == 0:
                        # This is to simply save the Ground Truth set.
                        generated = full_sequence[:, 1:prompt_len]
                    else: 
                        prompt = full_sequence[:, :prompt_len]
                        if generate_type == 'var':
                            generated = model.generate_var_style(
                                start_tokens=prompt, 
                                y_true=y_true,
                                dataset_id=dataset_id,
                                temperature=cfg.ar_model.generation.inference_temperature
                            )
                        else:
                            raise ValueError(f"Invalid generate type: {generate_type}")
                    # Vectorized approach - convert tensors to CPU once
                    generated = torch.cat([prompt, generated], dim=1)
                    generated_cpu = generated.to("cpu").long().tolist()
                    dataset_ids = dataset_id.cpu().tolist()
                    labels = y_true.cpu().tolist()
                    
                    # Create base record template
                    base_record = {
                        "dataset": cfg.data.data_name,
                        "split": split,
                    }
                    
                    # Create all records at once using list comprehension
                    records = [
                        {
                            **base_record,
                            "sample_id": sample_id + i,
                            "dataset_id": ds_id,
                            "label": label,
                            "token_len": len(tokens),
                            "tokens": tokens,
                            "prompt_len": prompt_len,
                            "original_ts_len": data_id_2_tslen[ds_id],
                        }
                        for i, (tokens, ds_id, label) in enumerate(zip(generated_cpu, dataset_ids, labels))
                    ]
                    
                    # Batch write - single write operation
                    f.write('\n'.join(json.dumps(record) for record in records) + '\n')
                    sample_id += len(records)
                    print(f"Saved {sample_id} samples")

                    
            print(f"Done. Saved {split} JSONL: {output_file_path}")




def generate_samples_mix_prompt_length(cfg, model, device, prompt_len_list, splits_info, output_dir, generate_type='var'):
    GENERATE_LENGTH = cfg.data.max_seq_len + 1 # +1 for BOS

    # Remove existing directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"{bcolors.WARNING}Removing existing directory: {output_dir}{bcolors.ENDC}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{bcolors.OKGREEN}Created directory to save synthetic samples: {output_dir}{bcolors.ENDC}")

    # add +1 for BOS token to prompt_len
    prompt_len_list_with_bos = [prompt_len + 1 for prompt_len in prompt_len_list]

    sample_id = 0
    for split, loader in tqdm(splits_info,
                            desc="Processing splits",
                            unit="split", 
                            leave=False):
        output_file_path = f'{output_dir}/{cfg.data.data_name}_{split}_synth_tokens_promptlenmixed.jsonl'
        with open(output_file_path, 'a') as f:
            # Add progress bar for batches
            for batch in tqdm(loader, 
                            desc=f"Generating {split}", 
                            unit="batch", 
                            leave=False):
                full_sequence = batch["full_sequence"].to(device)
                y_true = batch["y_true"].to(device)
                dataset_id = batch["dataset_id"].to(device)
                prompt_len = random.choice(prompt_len_list_with_bos)
                target_length = GENERATE_LENGTH - prompt_len
                prompt = full_sequence[:, :prompt_len]
                if generate_type == 'var':
                    generated = model.generate_var_style(
                        start_tokens=prompt, 
                        y_true=y_true,
                        dataset_id=dataset_id,
                        temperature=cfg.ar_model.generation.inference_temperature
                    )
                else:
                    raise ValueError(f"Invalid generate type: {generate_type}")
                # Vectorized approach - convert tensors to CPU once
                # concat prompt and generated
                generated = torch.cat([prompt, generated], dim=1)
                generated_cpu = generated.to("cpu").long().tolist()
                dataset_ids = dataset_id.cpu().tolist()
                labels = y_true.cpu().tolist()
                
                # Create base record template
                base_record = {
                    "dataset": cfg.data.data_name,
                    "split": split,
                }
                
                # Create all records at once using list comprehension
                records = [
                    {
                        **base_record,
                        "sample_id": sample_id + i,
                        "dataset_id": ds_id,
                        "label": label,
                        "token_len": len(tokens),
                        "tokens": tokens,
                        "prompt_len": prompt_len,
                        "original_ts_len": data_id_2_tslen[ds_id],
                    }
                    for i, (tokens, ds_id, label) in enumerate(zip(generated_cpu, dataset_ids, labels))
                ]
                
                # Batch write - single write operation
                f.write('\n'.join(json.dumps(record) for record in records) + '\n')
                sample_id += len(records)
                print(f"Saved {sample_id} samples")

                
        print(f"Done. Saved {split} JSONL: {output_file_path}")


data_id_2_tslen = {
    0: 140,   # ECG5000
    1: 24,   # ItalyPowerDemand
    2: 244,   # Nasdaq
    3: 144,   # ETTh1
}

data_dict = {
    'ECG5000': 0,
    'ItalyPowerDemand': 1,
    'Nasdaq': 2,
    'ETTh1': 3
}
