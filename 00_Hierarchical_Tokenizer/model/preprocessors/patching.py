
from typing import Any, Dict, List, Optional, Tuple

import einops
import mup
import math
import numpy as np
import torch
import torch.nn as nn

from model.utils.packed_ops import packed_proj

__all__ = ["PatchEmbedder"]


def get_num_patches(
    image_sizes: List[int],
    patch_sizes: List[int],
) -> List[int]:
    """Determine the number of patch tokens along each dim of a image.

    Args:
        image_sizes: List of image sizes for each spatial dimension [s1, s2, ..., sN].
        patch_sizes: List of patch sizes for each spatial dimension [p1, p2, ..., pN].

    Returns:
        List of number of patches in each spatial dimension [n1, n2, ..., nN].
    """

    assert len(image_sizes) == len(
        patch_sizes
    ), "patch_size must match the number of spatial dimensions"

    # Ensure each spatial dimension is divisible by the corresponding patch size
    for i, (s, p) in enumerate(zip(image_sizes, patch_sizes)):
        assert s % p == 0, f"Spatial dimension {i} (size {s}) is not divisible by patch size {p}"

    # Compute number of patches in each dimension
    n_list = [s_i // p_i for s_i, p_i in zip(image_sizes, patch_sizes)]

    return n_list


def get_num_patches_list(
    image_sizes_list: List[List[int]],
    patch_sizes: List[int],
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """
    Applies the patchify function to a list of tensors.

    Args:
        list_image_sizes: List of image sizes, for each item there is a int value for spatial
            dimension [s1, s2, ..., sN].
        patch_sizes: List of patch sizes for each spatial dimension [p1, p2, ..., pN]

    Returns:
        List of lists, where each inner list contains the number of patches in each spatial
            dimension for the corresponding tensor.
    """
    n_lists = []
    for image_sizes in image_sizes_list:
        n_list = get_num_patches(image_sizes, patch_sizes)
        n_lists.append(n_list)
    return n_lists


@torch.compiler.disable
def patchify(
    x: torch.Tensor, patch_sizes: List[int], flatten_patches: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Divides the input tensor into patches for arbitrary spatial dimensions.

    Args:
        x: Input tensor of shape (b, s1, s2, ..., sN, d)
        patch_sizes: List of patch sizes for each spatial dimension [p1, p2, ..., pN]
        flatten_patches: If True, patches are flattened into one n1*n2*...*nN sequence.

    Returns:
        Tuple[torch.Tensor, List[int]]: A tuple containing:
            - Patchified tensor of shape (b, n1*n2*...*nN, p1*p2*...*pN*d) if patches
              are to be flattened, (b, n1, n2, ..., nN, p1*p2*...*pN*d) otherwise.
            - List of number of patches in each spatial dimension [n1, n2, ..., nN].
    """
    # Validate input dimensions
    assert x.dim() >= 3, "Input must have at least batch, one spatial, and channel dimension"
    # Extract dimensions
    b, *spatial, d = x.shape

    N = len(spatial) # 1D TS => 1
    assert len(patch_sizes) == N, "patch_size must match the number of spatial dimensions"

    # Ensure each spatial dimension is divisible by the corresponding patch size
    for i, (s, p) in enumerate(zip(spatial, patch_sizes)):
        assert s % p == 0, f"Spatial dimension {i} (size {s}) is not divisible by patch size {p}"

    # Compute number of patches in each dimension
    n_list = [s_i // p_i for s_i, p_i in zip(spatial, patch_sizes)]

    # Construct the input pattern for einops
    input_pattern = "b " + " ".join([f"(n{i} p{i})" for i in range(N)]) + " d"

    # Construct the output pattern for einops
    pleft, pright = ("(", ")") if flatten_patches else ("", "")
    output_pattern = (
        f"b {pleft}"
        + " ".join([f"n{i}" for i in range(N)])
        + f"{pright} ("
        + " ".join([f"p{i}" for i in range(N)])
        + " d)"
    )

    # Create a dictionary for patch sizes and number of patches to pass as keyword arguments
    patch_kwargs = {f"p{i}": patch_sizes[i] for i in range(N)}
    patch_kwargs.update({f"n{i}": n_list[i] for i in range(N)})

    # Perform the rearrangement
    patches = einops.rearrange(x, f"{input_pattern} -> {output_pattern}", **patch_kwargs)

    return patches, n_list

import torch

def _get_patch_size_1d(patch_sizes):
    if isinstance(patch_sizes, (list, tuple)):
        return int(patch_sizes[0])
    try:
        return int(patch_sizes[0])
    except Exception:
        return int(patch_sizes)

def patchify_overlapping(
    x: torch.Tensor,
    patch_sizes,
    *,
    stride: int | None = None,      # 
    flatten_patches: bool = False,  # True → [B, N, P*D], False → [B, N, P, D]
):
    assert x.dim() == 3, f"expected [B,L,D], got {tuple(x.shape)}"
    B, L, D = x.shape

    P = _get_patch_size_1d(patch_sizes)
    S = (P + 1) // 2 if stride is None else int(stride)
    assert P <= L, f"patch_size({P}) must be <= L({L})"
    assert (L - P) % S == 0, f"(L - P) % stride must be 0; got (L={L}, P={P}, stride={S})"
    patches = x.unfold(dimension=1, size=P, step=S)  # [B, N, P, D]

    if flatten_patches:
        patches = patches.reshape(B, patches.size(1), P * D)  # [B, N, P*D]

    n_patches = patches.size(1)
    return patches.contiguous(), [n_patches]

def patchify_tensor_list_unfold(
    x_list: list[torch.Tensor],
    patch_sizes,
    *,
    stride: int | None = None,
    flatten_patches: bool = False,
):
    x = torch.cat(x_list, dim=0)  # [B, L, D]
    patches, n_list = patchify_overlapping(
        x, patch_sizes, stride=stride, flatten_patches=flatten_patches
    )  

    patches_list = list(patches.split(1, dim=0))  
    n_patches_list = [n_list[0]] * len(patches_list)
    return patches_list, n_patches_list

@torch.compiler.disable
def patchify_tensor_list(
    x_list: List[torch.Tensor], patch_sizes: List[int], flatten_patches: bool = False
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """
    Applies the patchify function to a list of tensors.

    Args:
        x_list: List of input tensors, each of shape (b, s1, s2, ..., sN, d)
        patch_sizes: List of patch sizes for each spatial dimension [p1, p2, ..., pN]
        flatten_patches: If True, patches are flattened into one n1*n2*...*nN sequence.

    Returns:
        Tuple containing:
            - List of patchified tensors.
            - List of lists, where each inner list contains the number of patches
              in each spatial dimension for the corresponding tensor.
    """
    patches_list = []
    n_lists = []
    for x in x_list:
        patches, n_list = patchify(x, patch_sizes, flatten_patches=flatten_patches)
        patches_list.append(patches)
        n_lists.append(n_list)
    return patches_list, n_lists


class PatchEmbedder(nn.Module):
    """
    Module for patch embedding of tensors across arbitrary dimensions, projecting
    patches to a desired output dimension.

    Args:
        input_tensor_list_read_key: Key to read the input list of tensors from the input dictionary.
        patches_list_write_key: Key to write the list of embedded patches into the output dictionary.
        n_patches_write_key: Key to write the number of patches for each input tensor.
        patch_sizes: List of patch sizes for each dimension of the input tensors.
        channels_in: Input feature dimension of each tensor.
        dim: Optional output feature dimension after projection; if None, no projection is applied.
        flatten_patches: Whether to flatten all patch dimensions except the first (batch) and last (feature) dimensions.
        weight_init_style: Initialization style for weights ('xavier', 'zero', or 'trunc_normal').
    """

    def __init__(
        self,
        input_tensor_list_read_key: str,
        patches_list_write_key: str,
        n_patches_write_key: str,
        patch_sizes: List[int],
        channels_in: int,
        dim: Optional[int] = None,
        flatten_patches: bool = True,
        weight_init_style: str = "xavier",
    ):
        super().__init__()
        self.input_tensor_list_read_key = input_tensor_list_read_key
        self.patches_list_write_key = patches_list_write_key
        self.n_patches_write_key = n_patches_write_key

        # Set up patching & input projection
        self.patch_sizes = patch_sizes
        self.dim_in = channels_in
        self.dim_out = dim
        self.flatten_patches = flatten_patches
        if self.dim_out is not None:
            dim_patch = channels_in * np.prod(patch_sizes)
            self.patch_proj = nn.Linear(dim_patch, self.dim_out, bias=True)
        else:
            self.dim_out = channels_in
            self.patch_proj = nn.Identity()

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        if self.dim_out is None:
            return

        if self.weight_init_style == "zero":
            nn.init.constant_(self.patch_proj.weight, 0)
        elif self.weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.patch_proj.weight)
        elif self.weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        # Bias
        if self.patch_proj.bias is not None:
            nn.init.constant_(self.patch_proj.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme"""
        if self.dim_out is None:
            return

        if self.weight_init_style == "zero":
            nn.init.constant_(self.patch_proj.weight, 0)
        elif self.weight_init_style == "xavier":
            mup.init.xavier_uniform_(self.patch_proj.weight)
        elif self.weight_init_style == "trunc_normal":
            mup.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        # Bias
        if self.patch_proj.bias is not None:
            nn.init.constant_(self.patch_proj.bias, 0)

    def get_n_patches(self, image_sizes_list: List[List[int]]) -> List[List[int]]:
        n_patches = get_num_patches_list(
            image_sizes_list=image_sizes_list,
            patch_sizes=self.patch_sizes,
        )
        return n_patches

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # List of [B, s1, s2, ..., sN, D] tensors
        # x_list.shape == (B, H, W, C(=D))
        x_list = data_dict[self.input_tensor_list_read_key] # [B,L,C]

        # List of [B, s1//p1, s2//p2, ..., sN//pN, D*p1*p2*...*pN] tensors
        # If flatten_patches=True, all except the first and last dimension are flattened
        patches_list, n_patches_list = patchify_tensor_list(
            x_list, self.patch_sizes, flatten_patches=self.flatten_patches
        )

        # Project each patch of dim D*p1*p2*...*pN to dim_out
        patches_proj_list = packed_proj(patches_list, self.patch_proj)

        data_dict[self.patches_list_write_key] = patches_proj_list
        data_dict[self.n_patches_write_key] = n_patches_list

        return data_dict
