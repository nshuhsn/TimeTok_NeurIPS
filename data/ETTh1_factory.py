
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from einops import rearrange

def etth1_data_loading(data_name, type, data_root):

    assert data_name == 'Etth1'
    assert type in ['train', 'val', 'test']
    if type == 'train':
        X = np.load(f"{data_root}/train_full.npy")
        y = np.load(f"{data_root}/train_y.npy")
        meta_data = pd.read_csv(f"{data_root}/train_meta_data.csv")
    elif type == 'val':
        X = np.load(f"{data_root}/val_full.npy")
        y = np.load(f"{data_root}/val_y.npy")
        meta_data = pd.read_csv(f"{data_root}/val_meta_data.csv")
    elif type == 'test':
        X = np.load(f"{data_root}/test_full.npy")
        y = np.load(f"{data_root}/test_y.npy")
        meta_data = pd.read_csv(f"{data_root}/test_meta_data.csv")

    return X, y, meta_data

def etth1_data_provider(data_root, batch_size):
    ## Following TimeGAN .
    input_len = 96 
    pred_len = 48

    seq_len =  input_len + pred_len
    train_data, train_y, train_meta_data = etth1_data_loading("Etth1", "train", data_root)
    val_data, val_y, val_meta_data = etth1_data_loading("Etth1", "val", data_root)
    test_data, test_y, test_meta_data = etth1_data_loading("Etth1", "test", data_root)
    train_dataset = Etth1Dataset(train_data, train_y, train_meta_data)
    val_dataset = Etth1Dataset(val_data, val_y, val_meta_data)
    test_dataset = Etth1Dataset(test_data, test_y, test_meta_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    param_dict = {
      "seq_len": seq_len,
      "input_len": input_len,
      "pred_len": pred_len,
      "min_val": 0, 
      "max_val": 0, 
    }
    return train_loader, val_loader, test_loader, param_dict

class Etth1Dataset(Dataset):
    def __init__(self, data, y, meta_data):
        self.data = data
        self.y = y
        self.meta_data = meta_data
    
    def __len__(self):
        return len(self.data)

    def get_meta_data(self):
        return self.meta_data
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        # Input: all timesteps except last, all features except last
        X = torch.tensor(seq, dtype=torch.float32)
        X = rearrange(X, 'l -> 1 l')
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        y = rearrange(y, 'l -> 1 l')
        return X, y, torch.tensor(-50) # dummy