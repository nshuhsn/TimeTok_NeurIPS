from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from einops import rearrange

def utsd_data_provider(args, flag):
    Data = UTSD_Npy
    subset_ratio = args.subset_ratio
    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.val_batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.utsd_root,
            flag=flag,
            size=[256, 1, 1],
            subset_ratio=subset_ratio  # Add this parameter
        )
    else:
        data_set = Data(
            root_path=args.utsd_root,
            flag=flag,
            size=[256, 1, 1],
            subset_ratio=subset_ratio  # Add this parameter
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=drop_last
    )
    return data_loader

# Download link: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
class UTSD_Npy(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', 
                 scale=True, nonautoregressive=False, stride=1, split=0.9, 
                 test_flag='T', subset_ratio=1.0):  # Add subset_ratio parameter
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.subset_ratio = subset_ratio  # Store the subset ratio
        self.data_list = []
        self.n_window_list = []
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    data = np.load(dataset_path)
                    
                    # Apply subset ratio to the original data
                    original_len = len(data)
                    subset_len = int(original_len * self.subset_ratio)
                    if subset_len < self.context_len:
                        continue
                    
                    # Take the first subset_ratio of the data
                    data = data[:subset_len]

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        
        # Remove the hard limit of 500 files or adjust as needed
        # self.n_window_list = self.n_window_list[:500]
        
        if len(self.n_window_list) > 0:
            print(f"Total number of windows in merged dataset: {self.n_window_list[-1]}")
        else:
            print("Warning: No valid data found!")

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        # seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        # seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        # print(seq_x.shape, seq_y.shape)
        # rearrange to [B,C,L]
        seq_x = rearrange(seq_x, 'l c -> c l')
        seq_y = rearrange(seq_y, 'l c -> c l')
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32) 

    def __len__(self):
        return self.n_window_list[-1]

data_dict = {
    'Utsd_Npy': UTSD_Npy
}