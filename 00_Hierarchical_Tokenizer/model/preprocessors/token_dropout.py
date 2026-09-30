import math
import random
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .registers import is_power_of_two, powers_of_two

__all__ = ["MaskedNestedDropout"]
import pdb

class MaskedNestedDropout(nn.Module):
    """
    Module that randomly drops tokens of the given tensors in a nested fashion, i.e.
    performs nested dropout / Matryoshka sampling. Simply replaces dropped tokens with
    a learnable mask token.

    Args:
        read_write_key: Key to apply the nested dropout on.
        dim: Dimension size of the mask token.
        eval_keep_k_read_key: During inference, by default nested dropout is disabled.
            This key allows to optionally choose the number of tokens to keep per tensor.
        train_keep_k_write_key: During training, write the number of kept tokens to this key.
        size_sampling_mode: Method to sample the number of tokens to randomly drop.
    """

    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_keep_k_read_key: Optional[str] = "eval_keep_k",
        train_keep_k_write_key: Optional[str] = "train_keep_k",
        size_sampling_mode: Literal["uniform", "pow2", "uniform_pow2"] = "uniform",
        drop_mode: Literal["truncate", "replace"] = "replace",  # ← NEW
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.train_keep_k_write_key = train_keep_k_write_key
        self.size_sampling_mode = size_sampling_mode
        self.drop_mode = drop_mode

        self.dropout_mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.dropout_mask_token, std=0.02)
        # self.dropout_mask_token = torch.zeros(self.dim)
        
    def sample_keep_k(self, N):
        if self.size_sampling_mode == "uniform":
            keep_k = np.random.randint(low=1, high=N + 1)
        elif self.size_sampling_mode == "pow2":
            assert is_power_of_two(N)
            keep_k = np.random.choice(powers_of_two(1, N))
        elif self.size_sampling_mode == "uniform_pow2":
            k = np.random.randint(low=1, high=N + 1)
            keep_k = k if is_power_of_two(k) else 1 << k.bit_length()
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} is not defined.")
        return keep_k

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            if self.eval_keep_k_read_key is None:
                return data_dict

            # Sequence list to read (each element: [1, N, D])
            xs = data_dict[self.read_write_key]

            # If eval_keep_k is missing, auto-set to N (total tokens) for each sample
            if self.eval_keep_k_read_key not in data_dict:
                data_dict[self.eval_keep_k_read_key] = [x.shape[1] for x in xs]

            keep_ks = data_dict[self.eval_keep_k_read_key]

            # Apply actual dropout with the configured keep_ks
            for i in range(len(xs)):
                k = int(keep_ks[i])
                x = xs[i]
                if self.drop_mode == "truncate":
                    xs[i] = x[:, :k]
                else:
                    # Broadcasts from [D] to fill [1, N-k, D] in [1, N, D]
                    x = x.clone()
                    x[:, k:] = self.dropout_mask_token
                    xs[i] = x
            return data_dict
        
        if self.train_keep_k_write_key in data_dict:
            for i in range(len(data_dict[self.read_write_key])):
                keep_ks = data_dict[self.train_keep_k_write_key]
                if self.drop_mode == "truncate":
                    data_dict[self.read_write_key][i] = data_dict[self.read_write_key][i][:, :keep_ks[i]]
                else:
                    data_dict[self.read_write_key][i][:, keep_ks[i]:] = self.dropout_mask_token
        else:
            keep_ks = []
            for i in range(len(data_dict[self.read_write_key])):
                N = data_dict[self.read_write_key][i].shape[1]
                keep_k = self.sample_keep_k(N)
                keep_ks.append(keep_k)
                if self.drop_mode == "truncate":
                    data_dict[self.read_write_key][i] = data_dict[self.read_write_key][i][:, :keep_k]
                else:
                    data_dict[self.read_write_key][i][:, keep_k:] = self.dropout_mask_token
                
            data_dict[self.train_keep_k_write_key] = keep_ks

        return data_dict