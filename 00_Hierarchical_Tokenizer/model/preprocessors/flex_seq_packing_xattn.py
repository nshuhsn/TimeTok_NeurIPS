import torch, torch.nn as nn, torch.nn.functional as F
import einops
from typing import Any, Dict

from .flex_seq_packing import (
    create_block_mask_cached, generate_packed_xattn_mask, next_highest_multiple
)

class ContextSequencePacker(nn.Module):
    def __init__(
        self,
        input_list_read_key: str,              # "local_tokens_proj"
        context_packed_seq_write_key: str,     # "ctx_packed_seq"
        context_packed_shapes_write_key: str,  # "ctx_ps_outer"
        pad_to_multiple: int = 256,
    ):
        super().__init__()
        self.input_list_read_key = input_list_read_key
        self.context_packed_seq_write_key = context_packed_seq_write_key
        self.context_packed_shapes_write_key = context_packed_shapes_write_key
        self.pad_to_multiple = pad_to_multiple

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        xs = data_dict[self.input_list_read_key]          # list of [1, n, D]
        ctx_packed, ps_ctx = einops.pack(xs, "b * d")     # [B, Nctx, D], ps_ctx: einops shapes
        B, Nctx, D = ctx_packed.shape
        assert B == 1, "FlexAttention"

        Nctx_pad = next_highest_multiple(Nctx, self.pad_to_multiple)
        if Nctx_pad > Nctx:
            ctx_packed = F.pad(ctx_packed, (0, 0, 0, Nctx_pad - Nctx))

        data_dict[self.context_packed_seq_write_key] = ctx_packed
        data_dict[self.context_packed_shapes_write_key] = ps_ctx
        data_dict["__ctx_len_orig"] = int(Nctx)
        data_dict["__ctx_len_pad"]  = int(Nctx_pad)
        data_dict["__ctx_device"]   = str(ctx_packed.device)
        return data_dict


class CrossAttnMaskBuilder(nn.Module):
    def __init__(
        self,
        query_shapes_read_key: str,       # "dec_ps_outer"
        ctx_shapes_read_key: str,         # "ctx_ps_outer"
        xattn_block_mask_write_key: str,  # "xattn_block_mask"
        pad_to_multiple: int = 256,
        compile_block_mask: bool = False,
    ):
        super().__init__()
        self.query_shapes_read_key = query_shapes_read_key
        self.ctx_shapes_read_key = ctx_shapes_read_key
        self.xattn_block_mask_write_key = xattn_block_mask_write_key
        self.pad_to_multiple = pad_to_multiple
        self.compile_block_mask = compile_block_mask
        self.create_block_mask = torch.compiler.disable(create_block_mask_cached)

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        ps_q = tuple(data_dict[self.query_shapes_read_key])   # dec_ps_outer
        ps_k = tuple(data_dict[self.ctx_shapes_read_key])     # ctx_ps_outer

        M_orig = sum([s.numel() for s in ps_q])
        N_orig = sum([s.numel() for s in ps_k])

        M = next_highest_multiple(M_orig, self.pad_to_multiple)
        N = next_highest_multiple(N_orig, self.pad_to_multiple)


        if "ctx_packed_seq" in data_dict:
            dev = str(data_dict["ctx_packed_seq"].device)
        elif "dec_packed_seq" in data_dict:
            dev = str(data_dict["dec_packed_seq"].device)
        else:
            dev = str(torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else "cpu")


        mask_fn, seq_ids_q, seq_ids_k = generate_packed_xattn_mask(
            ps_q, ps_k, max_seq_len=M, max_ctx_len=N, device=dev
        )

        block_mask = self.create_block_mask(
            mask_fn, None, None, M, N, device=dev, _compile=self.compile_block_mask
        )
        data_dict[self.xattn_block_mask_write_key] = block_mask

        data_dict["__xattn_M_orig"] = int(M_orig)
        data_dict["__xattn_N_orig"] = int(N_orig)
        data_dict["__xattn_M"]      = int(M)
        data_dict["__xattn_N"]      = int(N)
        data_dict["__xattn_device"] = dev
        return data_dict