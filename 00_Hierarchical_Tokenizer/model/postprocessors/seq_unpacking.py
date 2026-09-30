from typing import Any, Dict, List

import einops

import torch
import torch.nn as nn

__all__ = ["SequenceUnpacker"]


class SequenceUnpacker(nn.Module):
    """
    Module for unpacking sequences from nested packed formats, designed to manage hierarchical
    tensor unpacking within a dictionary-based pipeline.

    Args:
        packed_seq_read_key: Key to read the packed outer sequence from the input dictionary.
        inner_seq_write_keys: List of keys to write the unpacked inner sequences into the output dictionary.
        inner_packed_shapes_read_key: Key to read the packed shapes of inner sequences.
        outer_packed_shapes_read_key: Key to read the packed shapes of the outer sequence.
    """

    def __init__(
        self,
        packed_seq_read_key: str,
        inner_seq_write_keys: List[str],
        inner_packed_shapes_read_key: str,
        outer_packed_shapes_read_key: str,
    ):
        super().__init__()
        self.packed_seq_read_key = packed_seq_read_key
        self.inner_seq_write_keys = inner_seq_write_keys
        self.inner_packed_shapes_read_key = inner_packed_shapes_read_key
        self.outer_packed_shapes_read_key = outer_packed_shapes_read_key

        # For pretty printing
        self._init_args = locals().copy()
        self._init_args.pop("self")
        self._init_args.pop("__class__")

    def __repr__(self):
        cls_name = self.__class__.__name__
        args_str = ",\n  ".join(f"{k}={v!r}" for k, v in self._init_args.items())
        return f"{cls_name}(\n  {args_str}\n)"

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        outer_seq_packed = data_dict[self.packed_seq_read_key]
        outer_packed_shapes = data_dict[self.outer_packed_shapes_read_key]
        inner_packed_shapes = data_dict[self.inner_packed_shapes_read_key]

        # Remove padding tokens, if there are any
        B, N, D = outer_seq_packed.shape
        N_orig = sum([shape.numel() for shape in outer_packed_shapes])
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            outer_seq_packed = outer_seq_packed[:, :-num_padding_tokens]

        # Unpack outer sequence, i.e. concatenation of all documents
        inner_seqs_packed = einops.unpack(outer_seq_packed, outer_packed_shapes, "b * d")

        # Unpack inner sequences
        inner_seqs_unpacked = []
        for inner_seq_packed, inner_ps in zip(inner_seqs_packed, inner_packed_shapes):
            inner_seq_unpacked = einops.unpack(inner_seq_packed, inner_ps, "b * d")
            inner_seqs_unpacked.append(inner_seq_unpacked)

        # Split by keys and add to dictionary
        for key_idx, write_key in enumerate(self.inner_seq_write_keys):
            data_dict[write_key] = [seq[key_idx] for seq in inner_seqs_unpacked]

        return data_dict
