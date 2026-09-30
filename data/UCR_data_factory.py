from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class CustomUCRDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # [B, 1, L]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], torch.tensor(-50)  # dummy

    def __len__(self):
        return len(self.data)

import os

def debug_dataset_paths(dataset_name, data_root):
    dataset_dir = os.path.join(data_root, dataset_name)
    train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")

    print("\n🔍 [DEBUG] Dataset Path Check")
    print(f"📁 Dataset Dir: {dataset_dir} - Exists: {os.path.exists(dataset_dir)}")
    print(f"📄 Train File : {train_file} - Exists: {os.path.exists(train_file)}")
    print(f"📄 Test File  : {test_file} - Exists: {os.path.exists(test_file)}")

    # Check files exist in list
    if os.path.exists(dataset_dir):
        print("\n📂 [DEBUG] Contents of Dataset Dir:")
        for fname in os.listdir(dataset_dir):
            print(" -", fname)
    else:
        print("❌ Dataset directory does not exist.")


def transfer_labels(labels):
    unique = np.unique(labels)
    label_map = {v: i for i, v in enumerate(unique)}
    return np.array([label_map[y] for y in labels])

def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    
    return train_set, val_set, test_set

def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_

def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
def ucr_data_provider(args, n_splits=5, shuffle_train=True):
    print(args.dataset, args.data_root)
    debug_dataset_paths(args.dataset, args.data_root)

    X_train, y_train = load_UCR_UEA_dataset(
        name=args.dataset, split="train", return_type="numpy2d", extract_path=args.data_root)
    X_test, y_test = load_UCR_UEA_dataset(
        name=args.dataset, split="test", return_type="numpy2d", extract_path=args.data_root)

    X_all = np.concatenate([X_train, X_test], axis=0)

    _, min_val, max_val = MinMaxScaler(X_all)

    y_all = np.concatenate([y_train, y_test], axis=0).astype(int)
    y_all = transfer_labels(y_all)

    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    train_loaders, val_loaders, test_loaders = [], [], []

    for fold_idx, (raw_idx, test_idx) in enumerate(skf_outer.split(X_all, y_all)):
        X_raw, y_raw = X_all[raw_idx], y_all[raw_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        # Now split raw (80%) into 4-fold: 1 for val, 3 for train
        skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
        inner_train_idx, val_idx = next(skf_inner.split(X_raw, y_raw))
        X_train, y_train_ = X_raw[inner_train_idx], y_raw[inner_train_idx]
        X_val, y_val = X_raw[val_idx], y_raw[val_idx]

        X_train, X_val, X_test = fill_nan_value(X_train, X_val, X_test)

        X_train = normalize_per_series(X_train)
        X_val = normalize_per_series(X_val)
        X_test = normalize_per_series(X_test)

        # print(f"[Fold {fold_idx}] (train): {Counter(y_train_)}")

        # min(sample * 0.6 /10, 16)
        # batch_size = int(min(X_train.shape[0] / 10, 16))

        args.batch_size = 512
        if X_train[0].shape[0] < args.batch_size:
            args.batch_size = X_train[0].shape[0]

        # Dataset wrapping
        train_set = CustomUCRDataset(X_train, y_train_)
        val_set = CustomUCRDataset(X_val, y_val)
        test_set = CustomUCRDataset(X_test, y_test)

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle_train, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    # Shape info
    X_all = np.expand_dims(X_all, axis=1) if X_all.ndim == 2 else X_all
    enc_in = X_all.shape[1]
    seq_len = X_all.shape[2]

    param_dict = {
        "seq_len": seq_len,
        "enc_in": enc_in,
        "num_classes": len(np.unique(y_all)),
        'min_val': min_val,
        'max_val': max_val,
    }

    return train_loaders, val_loaders, test_loaders, param_dict