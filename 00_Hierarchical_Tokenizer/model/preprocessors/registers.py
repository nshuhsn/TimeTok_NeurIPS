
from functools import lru_cache
from typing import Any, Dict, List, Optional

import einops
import mup

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

__all__ = ["Registers1D"]


@lru_cache
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


@lru_cache
def powers_of_two(min_val, max_val):
    return [2**i for i in range(int(min_val).bit_length() - 1, int(max_val).bit_length())]


class Registers1D(nn.Module):
    """
    1D register token module. Given a data_dict, this module creates registers for a given
    list of tensors. The registers may be of a fixed size, or downsampled/ordered depending
    on the given size_sampling_mode and ordering_mode.

    Args:
        input_tensor_list_read_key: Key to read the input list of tensors from the input dictionary.
        register_sizes_read_write_key: Key to read or write the sizes of the registers in the data dictionary.
        registers_write_key: Key to write the generated registers into the output dictionary.
        dim: Dimension of each register token.
        n_min: Minimum number of register tokens to sample.
        n_max: Maximum number of register tokens available.
        n_eval: Optional fixed number of register tokens to use during evaluation; defaults to n_max if not specified.
        size_sampling_mode: Method to sample the number of register tokens; choices include 'uniform', 'powers_of_two',
            or specific sizes with 'k={sizes}' (e.g., 'k=1-2-4-8').
        ordering_mode: Method to select and order register tokens; options are 'nested' or 'avg_pool'.
        extra_registers_write_key: Optional key to write extra registers (e.g., positional embeddings) into the output dictionary.
        extra_registers_dim: Dimension of the extra registers if extra_registers_write_key is specified.
    """

    def __init__(
        self,
        input_tensor_list_read_key: str,
        register_sizes_read_write_key: str,
        registers_write_key: str,
        dim: int,
        n_min: int,
        n_max: int,
        n_eval: Optional[int] = None,
        size_sampling_mode: str = "uniform",
        ordering_mode: str = "nested",
        extra_registers_write_key: Optional[str] = None,
        extra_registers_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_tensor_list_read_key = input_tensor_list_read_key
        self.register_sizes_read_write_key = register_sizes_read_write_key
        self.registers_write_key = registers_write_key
        self.extra_registers_write_key = extra_registers_write_key

        self.n_min, self.n_max, self.n_eval = n_min, n_max, n_eval or n_max
        self.size_sampling_mode, self.ordering_mode = size_sampling_mode, ordering_mode
        if size_sampling_mode == "powers_of_two":
            assert is_power_of_two(
                n_min
            ), f"n_min must be a power of 2 for exponential size_sampling_mode, but is {n_min}."
            assert is_power_of_two(
                n_max
            ), f"n_max must be a power of 2 for exponential size_sampling_mode, but is {n_max}."
            self.valid_sizes = powers_of_two(n_min, n_max)
        elif size_sampling_mode == "uniform":
            assert n_min <= n_max, f"n_min={n_min} must be smaller or equal to n_max={n_max}"
            self.valid_sizes = list(range(n_min, n_max + 1))
        elif size_sampling_mode.startswith("k="):
            self.valid_sizes = [int(k) for k in size_sampling_mode.replace("k=", "").split("-")]
        else:
            raise NotImplementedError()

        # Learnable register tokens / queries
        self.registers = nn.Parameter(torch.randn(n_max, dim), requires_grad=True)
        trunc_normal_(self.registers, std=0.02)

        # Extra learnable tokens, e.g. to use as register positional embeddings for a decoder
        if extra_registers_write_key is not None:
            self.extra_registers = nn.Parameter(
                torch.randn(n_max, extra_registers_dim), requires_grad=True
            )
            trunc_normal_(self.registers, std=0.02)

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    def sample_register_sizes(self, N):
        if self.training:
            return np.random.choice(self.valid_sizes, N)
        else:
            return self.n_eval * np.ones(N, dtype=np.int64)

    def get_registers(self, register_sizes_list, batch_sizes_list, registers):
        registers_list = []
        for size, batch_size in zip(register_sizes_list, batch_sizes_list):
            if self.ordering_mode == "nested":
                reg = registers[:size]
            elif self.ordering_mode == "avg_pool":
                reg = F.adaptive_avg_pool1d(registers.T, (size)).T
            else:
                raise NotImplementedError(f"Ordering mode {self.ordering_mode} not implemented.")
            registers_list.append(einops.repeat(reg, "n d -> b n d", b=batch_size))
        return registers_list

    def get_extra_registers_from_registers(
        self, registers_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        register_sizes_list = [reg.shape[1] for reg in registers_list]
        batch_sizes_list = [reg.shape[0] for reg in registers_list]
        return self.get_registers(register_sizes_list, batch_sizes_list, self.extra_registers)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # List of [B, n1, n2, ..., nN, D] tensors
        x_list = data_dict[self.input_tensor_list_read_key]
        batch_sizes = [x.shape[0] for x in x_list]

        # If the data_dict already contains the register sizes, use those
        if self.register_sizes_read_write_key in data_dict:
            register_sizes_list = data_dict[self.register_sizes_read_write_key]
        else:
            register_sizes_list = self.sample_register_sizes(len(x_list))
            data_dict[self.register_sizes_read_write_key] = register_sizes_list

        data_dict[self.registers_write_key] = self.get_registers(
            register_sizes_list, batch_sizes, self.registers
        )
        if self.extra_registers_write_key is not None:
            data_dict[self.extra_registers_write_key] = self.get_registers(
                register_sizes_list, batch_sizes, self.extra_registers
            )

        return data_dict