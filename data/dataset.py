import json
import torch
from torch.utils.data import Dataset
import os
import importlib.util

_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils.py")
_spec = importlib.util.spec_from_file_location("repo_utils", _utils_path)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)
bcolors = _utils_module.bcolors

class CustomTokenDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg 
        self.mode = mode
        max_token_size = cfg.data.max_token_size
        self.data, self.meta = self.load_data()
        self.bos_id = cfg.data.bos_id
        self.mask_id = cfg.data.mask_id

    def load_data(self):

        with open(f"{self.cfg.data.abs_data_root}/{self.cfg.data.tokenizer_name}/{self.cfg.data.data_name}/{self.cfg.data.data_name}_{self.mode}_fsq_tokens.jsonl", "r") as f:
            print(f"{bcolors.OKGREEN}=====> Loading Token data: {self.cfg.data.data_name}_{self.mode}_fsq_tokens.jsonl {bcolors.ENDC}")
            token_data = [json.loads(line.strip()) for line in f]
        with open(f"{self.cfg.data.abs_data_root}/{self.cfg.data.tokenizer_name}/{self.cfg.data.data_name}/{self.cfg.data.data_name}_{self.mode}_fsq_tokens.jsonl.meta.json", "r") as f:
            print(f"{bcolors.OKGREEN}=====> Loading Token meta: {self.cfg.data.data_name}_{self.mode}_fsq_tokens.jsonl.meta.json {bcolors.ENDC}")
            token_meta = json.load(f)
        return token_data, token_meta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        dataset_id = self.data[idx]["dataset_id"]
        y_true = self.data[idx]["label"]
        
        # Convert to tensor and add BOS tokens
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Add BOS at the beginning 
        full_sequence = torch.cat([
            torch.tensor([self.bos_id], dtype=torch.long),
            tokens_tensor,
        ])
        y_true_tensor = torch.tensor(y_true, dtype=torch.long)
        dataset_id_tensor = torch.tensor(dataset_id, dtype=torch.long)
        return {
            "full_sequence":full_sequence, 
            "y_true":y_true_tensor,
            "dataset_id":dataset_id_tensor, 
        }



class CustomForecastDataset(CustomTokenDataset):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg, mode)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        dataset_id = self.data[idx]["dataset_id"]
        y_true = self.data[idx]["label"]
        
        # Convert to tensor and add BOS tokens
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        # Add BOS at the beginning 
        full_sequence = torch.cat([
            torch.tensor([self.bos_id], dtype=torch.long),
            tokens_tensor,
        ])
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        dataset_id_tensor = torch.tensor(dataset_id, dtype=torch.long)
        return {
            "full_sequence":full_sequence, 
            "y_true":y_true_tensor,
            "dataset_id":dataset_id_tensor, 
        }
