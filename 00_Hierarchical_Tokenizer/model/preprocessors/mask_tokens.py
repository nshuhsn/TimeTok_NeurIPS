
from typing import Any, Dict, List, Optional

import einops

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

__all__ = ["MaskTokenModule"]


class MaskTokenModule(nn.Module):
    """
    Module that generates and expands learnable mask tokens based on input size information,
    useful for creating masked tokens to decode.

    Args:
        mask_tokens_write_key: Key to write the generated mask tokens into the output dictionary.
        dim: Dimension size of the mask token.
        sizes_read_key: Key to read the list of sizes for generating mask tokens.
        read_sizes_from_key: Key to read tensors from which mask sizes will be determined.

    Note:
        Only one of `sizes_read_key` or `read_sizes_from_key` should be provided; specifying both raises an error.
    """

    def __init__(
        self,
        mask_tokens_write_key: str,
        dim: int,
        sizes_read_key: Optional[str] = None,
        read_sizes_from_key: Optional[str] = None,
    ):
        super().__init__()
        self.mask_tokens_write_key = mask_tokens_write_key
        self.sizes_read_key = sizes_read_key
        self.read_sizes_from_key = read_sizes_from_key
        if sizes_read_key is not None and read_sizes_from_key is not None:
            raise ValueError(
                "Only one of sizes_read_key or read_sizes_from_key should be provided."
            )

        self.dim = dim

        self.mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
        trunc_normal_(self.mask_token, std=0.02)

    def expand_mask_tokens(self, sizes):
        dim_names = [f"s{i}" for i in range(len(sizes))]
        pattern = "d -> 1 " + " ".join(dim_names) + " d"
        kwargs = dict(zip(dim_names, sizes))
        return einops.repeat(self.mask_token, pattern, **kwargs)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Get [n0, n1, ..., nN] shapes (without batch size, nor dimension)
        if self.sizes_read_key is not None:
            sizes_list = data_dict[self.sizes_read_key]
        elif self.read_sizes_from_key is not None:
            sizes_list = [
                list(tensor.shape[1:-1]) for tensor in data_dict[self.read_sizes_from_key]
            ]

        mask_token_list = [self.expand_mask_tokens(sizes) for sizes in sizes_list]

        data_dict[self.mask_tokens_write_key] = mask_token_list

        return data_dict
