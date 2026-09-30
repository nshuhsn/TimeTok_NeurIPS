import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from einops import pack, rearrange, repeat, unpack

import torch
import torch.nn as nn
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module

from model.utils.packed_ops import packed_call

__all__ = ["FSQ"]

# helper functions

def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with straight through gradients, randomly skip quantization per sample."""
    zhat = z.round()
    batch_size = z.shape[0]
    device = z.device
    # Create a mask where each sample has a probability `drop_quant_p` to not be quantized
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    # Reshape mask to broadcast over the remaining dimensions
    mask = mask.view(batch_size, *([1] * (z.ndim - 1)))
    # Apply the mask: if mask=1, keep the original value; if mask=0, use the quantized value
    output = z + ((1 - mask) * (zhat - z)).detach()
    return output


class FSQ(nn.Module):
    """Minimal FSQ (https://arxiv.org/abs/2309.15505) implementation. Except when using
    packed_call, expects channel dimension to be last.

    Args:
        latents_read_key: Dictionary key to read input latents from.
        quants_write_key: Dictionary key to write quantized latents.
        tokens_write_key: Dictionary key to write discrete token ids.
        levels: List of FSQ levels. See https://arxiv.org/abs/2309.15505 for suggestions.
        drop_quant_p: During training, pass the non-rounded values with this probability for each
            sample in the batch/list.
        corrupt_tokens_p: During training, optionally corrupt tokens by setting a percentage of them
            to random other tokens indices.
        min_corrupt_tokens_p: Optional argument specifying a minimum percentage of tokens to be
            corrupted. The actual percentage is sampled uniformly between the min and max per sample.
        apply_corrupt_tokens_p: Probability of activating token corruption per sample. Only active if
            corrupt_tokens_p > 0.
        packed_call: Set to True to pack list of examples and quantize them jointly. Might be slighly
            more efficient.
    """

    def __init__(
        self,
        latents_read_key: str,
        quants_write_key: str,
        tokens_write_key: str,
        levels: List[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
    ):
        super().__init__()
        self.latents_read_key = latents_read_key
        self.quants_write_key = quants_write_key
        self.tokens_write_key = tokens_write_key

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(levels)
        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p
        self.packed_call = packed_call

    def __repr__(self):
        return f"FSQ(levels={self._levels.tolist()}, codebook_size={self.codebook_size})"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  levels={self._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  drop_quant_p={self.drop_quant_p!r},\n"
            ")"
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        bounded = self.bound(z)
        drop_quant_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded, drop_quant_p)
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_embedding(
        self,
        indices: Tensor,
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        "Randomly corrupt some entries of the quantized Tensor"
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_tokens_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_tokens_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        quant = self.quantize(z.float())
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            # Optionally corrupt a random percentage of tokens during training
            quant = self.corrupt_quant(quant)
        tokens = self.codes_to_indices(quant)

        # Incompatible return value type (got "Tensor", expected "LongTensor")  [return-value]
        return quant, tokens.long()  # type: ignore

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        z = data_dict[self.latents_read_key]

        if self.packed_call:
            fsq_fn = partial(self.forward_z)
            quant, tokens = packed_call(fsq_fn, z)
        elif isinstance(z, list):
            quant, tokens = [], []
            for z_i in z:
                quant_i, tokens_i = self.forward_z(z_i)
                quant.append(quant_i)
                tokens.append(tokens_i)
        else:
            quant, tokens = self.forward_z(z)

        data_dict[self.quants_write_key] = quant
        data_dict[self.tokens_write_key] = tokens

        return data_dict
