import random
from typing import Any, Dict, Optional

import einops

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

__all__ = ["ZeroNullCond", "LearnedNullCond"]


class ZeroNullCond(nn.Module):
    """
    Module that randomly overwrites elements of a list of tensors with all zeros.
    Useful for implementing the null condition for classifier-free guidance dropout.

    Args:
        read_write_key: Key to indicate which tensor list to modify in place.
        eval_drop_flag_read_key: During inference, dropout is disabled. This key
            allows to optionally override individual tensors with the null condition.
            Useful to run the classifier-free guidance unconditional forward pass.
        dropout_prob: Probability of setting each tensor to the null condition.
    """

    def __init__(
        self,
        read_write_key: str,
        eval_drop_flag_read_key: Optional[str] = "eval_dropout_mask",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.eval_drop_flag_read_key = eval_drop_flag_read_key
        self.dropout_prob = dropout_prob

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            if self.eval_drop_flag_read_key is None:
                return data_dict
            if self.eval_drop_flag_read_key not in data_dict:
                return data_dict

            for i in range(len(data_dict[self.read_write_key])):
                # Drop if the i-th element is True
                if data_dict[self.eval_drop_flag_read_key][i]:
                    data_dict[self.read_write_key][i][:] = 0.0
        else:
            for i in range(len(data_dict[self.read_write_key])):
                if random.random() < self.dropout_prob:
                    data_dict[self.read_write_key][i][:] = 0.0

        return data_dict


class LearnedNullCond(nn.Module):
    """
    Module that randomly overwrites elements of a list of tensors with a learned embedding.
    Useful for implementing the null condition for classifier-free guidance dropout.

    Args:
        read_write_key: Key to indicate which tensor list to modify in place.
        dim: Dimension of the learned embedding.
        eval_drop_flag_read_key: During inference, dropout is disabled. This key
            allows to optionally override individual tensors with the null condition.
            Useful to run the classifier-free guidance unconditional forward pass.
        dropout_prob: Probability of setting each tensor to the null condition.
        replace_with_single_token: If True, the entire dropped sequence will be replaced
            by a single [B,1,D] sized token, rather than overriding the entire [B,N,D]
            sized sequence with the same null condition.
    """

    def __init__(
        self,
        read_write_key: str,
        dim: int,
        eval_drop_flag_read_key: Optional[str] = "eval_dropout_mask",
        dropout_prob: float = 0.0,
        replace_with_single_token: bool = False,
    ):
        super().__init__()
        self.read_write_key = read_write_key
        self.dim = dim
        self.eval_drop_flag_read_key = eval_drop_flag_read_key
        self.dropout_prob = dropout_prob
        self.replace_with_single_token = replace_with_single_token

        self.nullcond = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.nullcond, std=0.02)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.training:
            if self.eval_drop_flag_read_key is None:
                return data_dict
            if self.eval_drop_flag_read_key not in data_dict:
                return data_dict

            for i in range(len(data_dict[self.read_write_key])):
                # Drop if the i-th element is True
                if data_dict[self.eval_drop_flag_read_key][i]:
                    if self.replace_with_single_token:
                        B = data_dict[self.read_write_key][i].shape[0]
                        nullcond_token = einops.repeat(self.nullcond, "d -> b 1 d", b=B)
                        data_dict[self.read_write_key][i] = nullcond_token
                    else:
                        data_dict[self.read_write_key][i][:] = self.nullcond
        else:
            for i in range(len(data_dict[self.read_write_key])):
                if random.random() < self.dropout_prob:
                    if self.replace_with_single_token:
                        B = data_dict[self.read_write_key][i].shape[0]
                        nullcond_token = einops.repeat(self.nullcond, "d -> b 1 d", b=B)
                        data_dict[self.read_write_key][i] = nullcond_token
                    else:
                        data_dict[self.read_write_key][i][:] = self.nullcond

        return data_dict
