import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

class TimeWindowScaler(nn.Module):
    """
    Per-sample time windowing:
      - Given max token count via 'level' (e.g., 128), internally generates [1,2,4,...,level] levels
      - When keep_k == 2^i, scales each sample's t to 0~(i+1)/S (S = total number of levels)
      - k=1 -> t in [0,1/S], k=2 -> t in [0,2/S], ..., k=level -> t in [0,1]
    """

    def __init__(
        self,
        timesteps_read_key: str,
        keepk_read_key: str,
        write_key: Optional[str] = None,
        level: int = 128,
        eval_keep_k_read_key: Optional[str] = None,   # keep_k key to use during eval
    ):
        super().__init__()
        self.timesteps_read_key = timesteps_read_key
        self.keepk_read_key = keepk_read_key
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.write_key = write_key or timesteps_read_key

        # level must be a power of two
        if level < 1 or (level & (level - 1)) != 0:
            raise ValueError(f"'level' must be a power of two (got {level}).")
        self.level = int(level)

        # Level list: [1,2,4,...,level]
        lv = []
        v = 1
        while v <= self.level:
            lv.append(v)
            v *= 2
        self.register_buffer("_levels", torch.tensor(lv, dtype=torch.long), persistent=False)
        self.S = len(lv)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # During eval, prefer eval_keep_k if available, otherwise use default keepk
        keepk_key = None
        if (not self.training) and (self.eval_keep_k_read_key is not None) and (self.eval_keep_k_read_key in data_dict):
            keepk_key = self.eval_keep_k_read_key
        elif self.keepk_read_key in data_dict:
            keepk_key = self.keepk_read_key

        t_in = data_dict[self.timesteps_read_key]
        keepks = data_dict[keepk_key]

        # Convert keep_k list/tensor to Python int list
        if isinstance(keepks, torch.Tensor):
            keep_list = keepks.view(-1).tolist()
        else:
            keep_list = [int(k) for k in keepks]
        B = len(keep_list)

        # Normalize input t to list[float]
        if isinstance(t_in, torch.Tensor):
            if t_in.numel() == 1:
                t_list: List[float] = [float(t_in.item())] * B
            else:
                t_list = t_in.view(-1).float().tolist()
        elif isinstance(t_in, (list, tuple)):
            if len(t_in) == 1 and B > 1:
                t_list = [float(t_in[0])] * B
            else:
                t_list = [float(x) for x in t_in]
        else:  # scalar
            t_list = [float(t_in)] * B

        # Length check / only allow single-value broadcast
        if len(t_list) != B:
            if len(t_list) == 1:
                t_list = [t_list[0]] * B
            else:
                raise ValueError(f"timesteps length {len(t_list)} != batch size {B}")

        # Strict validation: each keep_k must be a power of two in [1..level]
        for k in keep_list:
            if k < 1 or k > self.level or (k & (k - 1)) != 0:
                raise ValueError(
                    f"invalid keep_k {k}; must be a power of two in [1, {self.level}]"
                )

        # Per-sample scale factor f = (idx+1)/S, idx = log2(keep_k)
        f_list: List[float] = []
        for k in keep_list:
            idx = int(round(math.log2(k)))  # k is validated here
            f = float((idx + 1) / self.S)
            f_list.append(f)

        # t_out = t_in * f
        t_out = [float(ti * fi) for ti, fi in zip(t_list, f_list)]
        device = t_in.device if isinstance(t_in, torch.Tensor) else self._levels.device
        data_dict[self.write_key] = torch.tensor(t_out).to(device)
        return data_dict
