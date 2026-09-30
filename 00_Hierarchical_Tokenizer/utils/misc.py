
import collections.abc
import hashlib
from contextlib import nullcontext
from itertools import repeat
from typing import List, Optional, Union

import torch

import torchvision.transforms.functional as TF

__all__ = [
    "str_to_dtype",
    "detect_bf16_support",
    "get_bf16_context",
    "get_autocast_context",
    "get_generator",
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_ntuple",
]


def str_to_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    elif dtype_str in ["float16", "fp16"]:
        return torch.float16
    elif dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    elif dtype_str in ["float32", "fp32"]:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype string representation: {dtype_str}")


def detect_bf16_support():
    """
    Checks if the current GPU supports BF16 precision.

    Returns:
        bool: True if BF16 is supported, False otherwise.
    """
    if torch.cuda.is_available():
        # For NVIDIA GPUs, BF16 support typically requires compute capability 8.0 or higher.
        cc_major, _ = torch.cuda.get_device_capability(0)
        return cc_major >= 8
    return False


def get_bf16_context(enable_bf16: bool = detect_bf16_support(), device_type: str = "cuda"):
    """
    Returns an autocast context that uses BF16 precision if enable_bf16 is True,
    otherwise returns a no-op context.
    """
    if enable_bf16:
        # When BF16 is enabled, we use torch.cuda.amp.autocast.
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)
    else:
        # Otherwise, we use a no-op context.
        return nullcontext()


def get_autocast_context(x: Union[torch.Tensor, List[torch.Tensor]], dtype_override: torch.dtype):
    device_type = x[0].device.type
    if dtype_override is None:
        auto_cast_context = nullcontext()
    else:
        auto_cast_context = torch.amp.autocast(
            device_type, dtype=dtype_override, enabled=dtype_override != torch.float32
        )
    return auto_cast_context


def get_generator(seed: int = None, device: str = "cuda") -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is None:
        seed = torch.seed()
    gen.manual_seed(seed)
    return gen


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
