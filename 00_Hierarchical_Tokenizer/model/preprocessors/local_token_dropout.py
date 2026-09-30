import math
import random
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .registers import is_power_of_two, powers_of_two

__all__ = ["MaskedDropout"]

class MaskedDropout(nn.Module):
    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_keep_k_read_key: Optional[str] = "local_eval_keep_k",
        train_keep_k_write_key: Optional[str] = "local_train_keep_k",
        size_sampling_mode: Literal["uniform", "pow2", "uniform_pow2"] = "uniform",
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.size_sampling_mode = size_sampling_mode
        self.train_keep_k_write_key = train_keep_k_write_key

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
        xs = data_dict[self.read_write_key]  # list of [1, N, D]
        B = len(xs)

        if not self.training:
            use_keep_idx = ("eval_keep_idx" in data_dict)
            keep_k_list = []

            for i in range(B):
                x = xs[i]  # [1, N, D]
                assert x.dim() == 3 and x.shape[0] == 1, f"expected [1,N,D], got {tuple(x.shape)}"
                N = x.shape[1]

                if use_keep_idx:
                    idx = torch.as_tensor(data_dict["eval_keep_idx"][i], device=x.device, dtype=torch.long)
                    if idx.numel() == 0:
                        idx = torch.zeros(1, dtype=torch.long, device=x.device)
                    idx = idx.clamp_(0, N - 1)
                    xs[i] = x.index_select(dim=1, index=idx)   # [1, k, D]
                    keep_k_list.append(int(xs[i].shape[1]))

                elif (self.eval_keep_k_read_key is not None) and (self.eval_keep_k_read_key in data_dict):
                    keep_k = int(data_dict[self.eval_keep_k_read_key][i])
                    keep_k = max(0, min(N, keep_k))  #
                    if keep_k < N:
                        perm = torch.randperm(N, device=x.device)
                        keep_idx = perm[:keep_k]
                        keep_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
                        keep_mask[keep_idx] = True

                        m = self.dropout_mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, -1)  # [1,1,D]
                        x_new = torch.where(keep_mask.view(1, N, 1), x, m.expand(1, N, -1))            # [1,N,D]
                        xs[i] = x_new
                    else:
                        xs[i] = x
                else:
                    xs[i] = x

            if use_keep_idx and (self.eval_keep_k_read_key is not None):
                data_dict[self.eval_keep_k_read_key] = keep_k_list

        else:
            keep_ks = []
            for i in range(B):
                x = xs[i]
                N = x.shape[1]
                keep_k = int(self.sample_keep_k(N))
                keep_k = max(0, min(N, keep_k))  # 

                if keep_k < N:
                    perm = torch.randperm(N, device=x.device)
                    keep_idx = perm[:keep_k]
                    keep_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
                    keep_mask[keep_idx] = True

                    m = self.dropout_mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
                    x_new = torch.where(keep_mask.view(1, N, 1), x, m.expand(1, N, -1))
                    xs[i] = x_new

                keep_ks.append(keep_k)

            if self.train_keep_k_write_key is not None:
                data_dict[self.train_keep_k_write_key] = keep_ks

        return data_dict