import math
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, create_mask

__all__ = ["BlockWiseSequencePacker"]


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@lru_cache
def generate_seq_ids(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = torch.cat(
        [torch.full((size.numel(),), fill_value=i, device=device) for i, size in enumerate(ps)]
    )
    if max_seq_len is None:
        return seq_ids
    seq_len = len(seq_ids)
    assert max_seq_len >= seq_len
    return F.pad(seq_ids, (0, max_seq_len - seq_len), mode="constant", value=padding_id)

@lru_cache
def generate_packed_seq_mask(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    def packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]
        return same_seq_mask & not_padded_mask

    return packed_seq_masking, seq_ids

@lru_cache
def generate_causal_packed_seq_mask(ps, max_seq_len=None, device="cuda", padding_id=-1):
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)

    def causal_packed_seq_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]
        causal = kv_idx <= q_idx
        return same_seq_mask & not_padded_mask & causal

    return causal_packed_seq_masking, seq_ids

@lru_cache
def generate_prefix_packed_seq_mask(
    ps, prefix_lens, max_seq_len=None, device="cuda", padding_id=-1
):
    # Create tensor of sequence IDs
    seq_ids = generate_seq_ids(ps, max_seq_len=max_seq_len, device=device, padding_id=padding_id)
    # Get unique sequence IDs and their counts
    _, counts = torch.unique_consecutive(seq_ids, return_counts=True)
    # Create cumulative counts (offsets)
    offsets = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)[:-1]])
    # Convert prefix_lens tuple to tensor. Needs to be predictably hashable.
    prefix_lens = torch.tensor(prefix_lens, device=device)

    def prefix_packed_seq_mask(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (seq_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == seq_ids[kv_idx]

        q_logical = q_idx - offsets[seq_ids[q_idx]]
        kv_logical = kv_idx - offsets[seq_ids[kv_idx]]
        inner_causal_mask = causal(b, h, q_logical, kv_logical)
        inner_prefix_mask = kv_logical < prefix_lens[seq_ids[kv_idx]]

        return same_seq_mask & (inner_causal_mask | inner_prefix_mask) & not_padded_mask

    return prefix_packed_seq_mask, seq_ids

@lru_cache
def generate_packed_xattn_mask(
    ps_seq, ps_ctx, max_seq_len=None, max_ctx_len=None, device="cuda", padding_id=-1
):
    seq_ids = generate_seq_ids(
        ps_seq, max_seq_len=max_seq_len, device=device, padding_id=padding_id
    )
    ctx_ids = generate_seq_ids(
        ps_ctx, max_seq_len=max_ctx_len, device=device, padding_id=padding_id
    )

    def packed_xattn_masking(b, h, q_idx, kv_idx):
        not_padded_mask = (seq_ids[q_idx] != padding_id) | (ctx_ids[kv_idx] != padding_id)
        same_seq_mask = seq_ids[q_idx] == ctx_ids[kv_idx]
        return same_seq_mask & not_padded_mask

    return packed_xattn_masking, seq_ids, ctx_ids

def strict_zip(*iterables):
    lengths = [len(iterable) for iterable in iterables]
    if len(set(lengths)) != 1:
        raise ValueError("All input iterables must be of the same length")
    return zip(*iterables)

def next_highest_multiple(N, multiple):
    return multiple * math.ceil(N / multiple)

def expand_emb(emb, seq_lens):
    return torch.cat(
        [einops.repeat(emb_i, "d -> 1 n d", n=n) for emb_i, n in zip(emb, seq_lens)],
        dim=1,
    )


def expand_emb_per_subseq(emb_packed, packed_shapes_list):
    adaLN_expansion = len(packed_shapes_list[0])
    emb_packed = einops.rearrange(emb_packed, "b (n d) -> b n d", n=adaLN_expansion)

    # Compute the repeats for each embedding
    repeats = torch.tensor(
        [[shape.numel() for shape in ps_list_i] for ps_list_i in packed_shapes_list],
        dtype=torch.long,
        device=emb_packed.device,
    )

    # Flatten embeddings and repeats
    emb_packed_flat = emb_packed.reshape(-1, emb_packed.shape[-1])  # Shape: [b * n, d]
    repeats_flat = repeats.flatten()  # Shape: [b * n]

    # Repeat embeddings according to repeats
    emb_expanded_flat = emb_packed_flat.repeat_interleave(repeats_flat, dim=0)

    # Return the expanded embeddings
    return emb_expanded_flat.unsqueeze(0)  # Shape: [1, total_n, d]


class BlockWiseSequencePacker(nn.Module):
    """
    Module for packing sequences from multiple input lists of tensors, creating a block-wise
    self-attention, causal, or prefix LM masks for FlexAttention. Sequences will be concatenated
    in order of the input_list_read_keys.

    Args:
        input_list_read_keys: List of keys to read input lists of tensors from the input dictionary.
        packed_seq_write_key: Key to write the packed sequence into the output dictionary.
        block_mask_write_key: Key to write the block-wise attention mask into the output dictionary.
        inner_packed_shapes_write_key: Key to write the packed shapes of the inner sequences.
        outer_packed_shapes_write_key: Key to write the packed shape of the outer sequence.
        mask_mode: Block-wise attention mask mode, 'full' (full self-attention over all inner sequences),
            'causal' (causal across all), or 'causal_last' (full self-attention for all but the
            last inner sequences which is causal, e.g. prefix LM). Attention will always be
            block-wise, i.e. outer sequences cannot attend to each other.
        max_seq_len: Optionally pads packed token sequence to the given length. Useful for
            FlexAttention's requirement that sequence lengths must be multiples of 128.
        pad_to_multiple: While max_seq_len specifies a fixed seq_len, pad_to_multiple can be used
            to pad the sequence to the next multiple of the given value, e.g. 128. Cannot be set
            simultaneously with max_seq_len.
        emb_packing_fn_write_key: Optional embedding packing function that specifies how elements
            of a tensor of shape (B, L, ...) should be expanded to sequences of shape
            (B, l1, ...), (B, l2, ...), ..., (B, lL, ...).
        per_subseq_embs: If True, applies the emb_packing_fn to each subsequence of the packed sequences.
            Useful for applying different embeddings to each subsequence, e.g. for AdaLN applied differently
            to image patches and registers.
        compile_block_mask: Whether or not to compile FlexAttention's create_block_mask.
        return_materialized_mask: If True, returns the materialized mask instead of the block mask.
    """

    def __init__(
        self,
        input_list_read_keys: List[str],
        packed_seq_write_key: str,
        block_mask_write_key: str,
        inner_packed_shapes_write_key: str,
        outer_packed_shapes_write_key: str,
        mask_mode: str = "full",
        max_seq_len: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        emb_packing_fn_write_key: Optional[str] = None,
        per_subseq_embs: bool = False,
        compile_block_mask: bool = True,
        return_materialized_mask: bool = False,
    ):
        super().__init__()
        self.input_list_read_keys = input_list_read_keys
        self.packed_seq_write_key = packed_seq_write_key
        self.block_mask_write_key = block_mask_write_key
        self.inner_packed_shapes_write_key = inner_packed_shapes_write_key
        self.outer_packed_shapes_write_key = outer_packed_shapes_write_key
        self.emb_packing_fn_write_key = emb_packing_fn_write_key

        self.mask_mode = mask_mode
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        if max_seq_len is not None and pad_to_multiple is not None:
            raise ValueError("Only one of max_seq_len or pad_to_multiple should be provided.")
        self.per_subseq_embs = per_subseq_embs

        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)
        self.return_materialized_mask = return_materialized_mask

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
        # From the data_dict, get the lists containing the aligned tensors,
        # e.g., a list of image patches and a list of registers.
        list_of_tensor_lists = [data_dict[key] for key in self.input_list_read_keys]

        # Concatenate each sample across the lists, e.g., images[0] | registers[0], images[1] | registers[1], ...
        # Keep track of the shape of these inner tensors
        tensors_concat_list = []
        packed_shapes_list = []
        for tensors in strict_zip(*list_of_tensor_lists):
            # tensors contains the i-th entries of each of the lists in list_of_tensor_lists
            sample_packed, ps = einops.pack(tensors, "b * d")
            tensors_concat_list.append(sample_packed)
            packed_shapes_list.append(ps)

        # Pack tensors into one large sequence
        tensors_packed, ps = einops.pack(tensors_concat_list, "b * d")
        B, N_orig, D = tensors_packed.shape

        # Only supporting B=1 until https://github.com/pytorch/pytorch/issues/134560 is resolved
        assert B == 1

        device = str(tensors_packed.device)

        # Create full or causal block-wise self-attention mask using FlexAttention. Optionally pad sequences.
        if self.pad_to_multiple is not None:
            max_seq_len = next_highest_multiple(N_orig, self.pad_to_multiple)
        else:
            max_seq_len = self.max_seq_len
            
        if self.mask_mode == "full":
            mask_fn, seq_ids = generate_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "causal":
            mask_fn, seq_ids = generate_causal_packed_seq_mask(
                tuple(ps),
                max_seq_len=max_seq_len,
                device=device,
            )
        elif self.mask_mode == "causal_last":
            prefix_lens = [
                sum([shape.numel() for shape in ps_i[:-1]]) for ps_i in packed_shapes_list
            ]
            mask_fn, seq_ids = generate_prefix_packed_seq_mask(
                tuple(ps),
                tuple(prefix_lens),
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            raise ValueError(f"Invalid mask mode {self.mask_mode}")

        N = len(seq_ids)
        assert (
            N % 128 == 0
        ), f"flex_attention sequence length must be a multiple of 128, but current is {N}."
        if self.return_materialized_mask:
            block_mask = create_mask(mask_fn, None, None, N, N, device=device)
        else:
            block_mask = self.create_block_mask(
                mask_fn, None, None, N, N, device=device, _compile=self.compile_block_mask
            )

        # Optionally zero-pad packed sequence
        # Outer packed shapes can be used to remove padding
        num_padding_tokens = N - N_orig
        if num_padding_tokens > 0:
            tensors_packed = F.pad(tensors_packed, (0, 0, 0, num_padding_tokens))

        # Optionally add an embedding packing function that specifies how elements
        # of a tensor of shape (B, L, ...) should be expanded to sequences of shape
        # (B, l1, ...), (B, l2, ...), ..., (B, lL, ...). Does not take into account
        # any padding.
        if self.emb_packing_fn_write_key is not None:
            seq_lens = [shape.numel() for shape in ps]
            if self.per_subseq_embs:
                emb_packing_fn = partial(
                    expand_emb_per_subseq, packed_shapes_list=packed_shapes_list
                )
            else:
                emb_packing_fn = partial(expand_emb, seq_lens=seq_lens)
            data_dict[self.emb_packing_fn_write_key] = emb_packing_fn

        data_dict[self.packed_seq_write_key] = tensors_packed
        data_dict[self.block_mask_write_key] = block_mask
        data_dict[self.inner_packed_shapes_write_key] = packed_shapes_list
        data_dict[self.outer_packed_shapes_write_key] = ps

        return data_dict
