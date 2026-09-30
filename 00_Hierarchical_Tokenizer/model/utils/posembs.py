import operator
from typing import Any, Dict, List, Optional

import einops
import mup

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

__all__ = ["build_2d_sincos_posemb", "PositionalEmbedding", "PositionalEmbeddingAdder"]


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings as used in MoCo-v3

    Returns positional embedding of shape (1, N, D) where N = W*H
    """
    grid_w = torch.arange(w, dtype=torch.float32)  # Shape (W,)
    grid_h = torch.arange(h, dtype=torch.float32)  # Shape (H, )
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")  # Shapes (W, H)
    assert (
        embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  # Shape (D/4,)
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("n,d->nd", [grid_w.reshape(-1), omega])  # Outer product, shape (W*H, D/4)
    out_h = torch.einsum("n,d->nd", [grid_h.reshape(-1), omega])  # Outer product, shape (W*H, D/4)
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    ).unsqueeze(
        0
    )  # Shape (1, W*H, D)
    pos_emb = einops.rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)
    return pos_emb


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding module that generates learnable or sinusoidal positional embeddings
    for n-dimensional inputs, with support for scaling to target sizes via truncation or interpolation.
    Returned positional embeddings are always [B, D, ...] shaped.

    Args:
        dim: Dimension size of the positional embeddings.
        max_sizes: Maximum sizes for positional embeddings per dimension (specified in patches or tokens).
        posemb_type: Type of positional embedding; options include 'learnable_sum', 'learnable_prod', and 'sincos'.
        posemb_scaling: Method to scale embeddings to target sizes; options are 'absolute' (truncation) or 'interpolate'.
    """

    def __init__(
        self,
        dim: int,
        max_sizes: List[int],  # Maximum posemb sizes per dimension (in patches/tokens, not pixels)
        posemb_type: str = "sincos",  # learnable_sum, learnable_prod, sincos (choice of posemb type)
        posemb_scaling: str = "absolute",  # absolute or interpolate (how to scale to different image sizes)
    ):
        super().__init__()
        self.dim = dim
        self.max_sizes = max_sizes
        self.n_dim = len(max_sizes)
        self.posemb_type = posemb_type
        self.posemb_scaling = posemb_scaling

        self.setup_posembs()

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    def setup_posembs(self):
        if self.posemb_type in ["learnable_sum", "learnable_prod"]:
            assert len(np.unique(self.max_sizes)) == 1, "All max_sizes must be the same."
            self.posembs = nn.Parameter(
                torch.randn(self.n_dim, 1, self.dim, self.max_sizes[0]),
                requires_grad=True,
            )
            trunc_normal_(self.posembs, std=0.02)
        elif self.posemb_type == "sincos":
            assert self.n_dim == 2, "Only 2D sincos positional embeddings are supported for now."
            self.posembs = build_2d_sincos_posemb(*self.max_sizes, embed_dim=self.dim)
            self.posembs = nn.Parameter(self.posembs, requires_grad=False)
        else:
            raise NotImplementedError(f"posemb_type '{self.posemb_type}' is not supported.")

    def init_weights_muP(self):
        """μP weight initialization scheme"""
        if self.posemb_type in ["learnable_sum", "learnable_prod"]:
            mup.init.trunc_normal_(self.posembs, std=0.02)

    @torch.compiler.disable
    def forward(self, shape: List[int]) -> torch.Tensor:
        # Get [1, D, n1, n2, ..., nN] shaped positional embeddings
        if self.posemb_type in ["learnable_sum", "learnable_prod"]:
            if self.n_dim == 1:
                posembs = self.posembs[0]
            elif self.n_dim == 2:
                op = operator.mul if "prod" in self.posemb_type else operator.add
                posembs_h, posembs_w = self.posembs
                # Broadcasted sum or prod from [1, D, N_H] and [1, D, N_W] to [1, D, N_H, N_W]
                posembs = op(posembs_h.unsqueeze(-1), posembs_w.unsqueeze(-2))
            else:
                raise NotImplementedError()
        elif self.posemb_type == "sincos":
            posembs = self.posembs
        else:
            raise NotImplementedError(f"posemb_type '{self.posemb_type}' is not supported.")

        # Adapt posembs to various tensor sizes
        if self.posemb_scaling == "absolute":
            # Crop the posemb according to the tensor sizes
            slices = [slice(None), slice(None)] + [slice(None, n) for n in shape]
            posembs = posembs[tuple(slices)]
        elif self.posemb_scaling == "interpolate":
            # Adapt to specific tensor size by interpolating entire posemb
            posembs = F.interpolate(posembs, shape, mode="bilinear", align_corners=True)
        else:
            raise NotImplementedError(f"posemb_scaling '{self.posemb_scaling}' is not supported.")

        return posembs


class PositionalEmbeddingAdder(PositionalEmbedding):
    """
    Positional Embedding Adder module that sums positional embeddings with input tensors
    and updates the data_dict accordingly.

    Args:
        read_key: Key to read the input tensor(s) from the input dictionary.
        write_key: Key to write the modified tensor(s) (input tensor + positional embeddings) into the output dictionary.
        dim: Dimension size of the positional embeddings.
        max_sizes: Maximum sizes for positional embeddings per dimension (specified in patches or tokens, not pixels).
        posemb_type: Type of positional embedding; options include 'learnable_sum', 'learnable_prod', and 'sincos'.
        posemb_scaling: Method to scale embeddings to target sizes; options are 'absolute' (truncation) or 'interpolate'.
        posemb_write_key: Optional key to write the generated positional embeddings into the output dictionary.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        dim: int,
        max_sizes: List[int],
        posemb_type: str = "sincos",
        posemb_scaling: str = "absolute",
        posemb_write_key: Optional[str] = None,
    ):
        super().__init__(
            dim=dim,
            max_sizes=max_sizes,
            posemb_type=posemb_type,
            posemb_scaling=posemb_scaling,
        )
        self.read_key = read_key
        self.write_key = write_key
        self.posemb_write_key = posemb_write_key

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    def get_posembs_for_tensor_list(
        self, x_list: List[torch.Tensor], channels_last: bool = True
    ) -> List[torch.Tensor]:
        shapes_list = [list(x.shape[1:-1]) for x in x_list]
        return self.get_posembs_for_shapes_list(shapes_list, channels_last=channels_last)

    def get_posembs_for_shapes_list(
        self, shapes_list: List[List[int]], channels_last: bool = True
    ) -> List[torch.Tensor]:
        posembs_list = [
            super(PositionalEmbeddingAdder, self).forward(shape) for shape in shapes_list
        ]
        if channels_last:
            posembs_list = [
                einops.rearrange(posemb_i, "B D ... -> B ... D") for posemb_i in posembs_list
            ]
        return posembs_list

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict[self.read_key]

        if isinstance(x, List):
            posembs = self.get_posembs_for_tensor_list(x, channels_last=True)
            x = [x_i + posemb_i for x_i, posemb_i in zip(x, posembs)]
        else:
            posembs = super(PositionalEmbeddingAdder, self).forward(list(x.shape[1:-1]))
            posembs = einops.rearrange(posembs, "B D ... -> B ... D")
            x = x + posembs

        data_dict[self.write_key] = x
        if self.posemb_write_key is not None:
            data_dict[self.posemb_write_key] = posembs

        return data_dict
