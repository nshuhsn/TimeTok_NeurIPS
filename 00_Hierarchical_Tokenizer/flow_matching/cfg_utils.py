from typing import Tuple

import torch

__all__ = ["MomentumBuffer", "normalized_guidance", "classifier_free_guidance"]


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,
    v1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
) -> torch.Tensor:
    """
    Normalized classifier-free guidance from:
    "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"
    https://arxiv.org/abs/2410.02416

    Args:
        pred_cond: Conditional prediction [B, C, H, W].
        pred_uncond: Unconditional prediction [B, C, H, W].
        guidance_scale: CFG scale.
        momentum_buffer: Momentum (beta).
        eta: Parallel component.
        norm_threshold: Rescaling threshold (r).
    """

    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def classifier_free_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Standard classifier-free guidance.

    Args:
        pred_cond: Conditional prediction [B, C, H, W].
        pred_uncond: Unconditional prediction [B, C, H, W].
        guidance_scale: CFG scale.
    """

    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
