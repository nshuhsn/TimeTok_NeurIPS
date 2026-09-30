from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from .attention import FlexCrossAttention, FlexSelfAttention
from .drop_path import DropPath
from .mlp import GatedMlp, Mlp
from .norm import Fp32LayerNorm

__all__ = ["FlexBlock", "FlexBlockAdaLN", "FlexDecoderBlock", "FlexDecoderBlockAdaLN"]


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def expand_to_padded_seq(emb, padded_seq):
    N_emb, N_seq = emb.shape[1], padded_seq.shape[1]
    num_padding_tokens = N_seq - N_emb
    if num_padding_tokens == 0:
        return emb
    else:
        return F.pad(emb, (0, 0, 0, num_padding_tokens))


class FlexBlock(nn.Module):
    """Flexible Transformer block.

    Args:
        dim: Transformer dimension size.
        num_heads: Number of attention heads (overrides head_dim if specified).
        head_dim: Dimension size per attention head.
        mlp_ratio: Ratio of hidden dimension size to input dimension size in the MLP.
        qkv_bias: Whether to use bias in Q, K, V projections.
        proj_bias: Whether to use bias in the output projection layer.
        mlp_bias: Whether to use bias in the MLP layer.
        drop: Dropout rate for attention and MLP linear layers.
        drop_path: Stochastic depth drop rate.
        act_layer: Activation layer used in the MLP.
        norm_layer: Pre-normalization layer.
        gated_mlp: Whether to use a gated MLP layer, e.g. for SwiGLU.
        qk_norm: Whether to apply QK normalization.
        muP_scale: Whether to use μP-compatible attention scaling.
    """

    def __init__(
        self,
        dim,
        num_heads=None,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
        gated_mlp=False,
        qk_norm=False,
        muP_scale=True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        num_heads = num_heads or dim // head_dim

        self.attn = FlexSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            muP_scale=muP_scale,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=mlp_bias,
            drop=drop,
        )

    def forward(self, x, score_mod=None, block_mask=None, rope_forward=None, **kwargs):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                score_mod=score_mod,
                block_mask=block_mask,
                rope_forward=rope_forward,
            )
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlexBlockAdaLN(FlexBlock):
    """Flexible Transformer block with adaLN-zero modulation.
    See FlexBlock for arguments.

    Args:
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
    """

    def __init__(self, adaLN_expansion: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.adaLN_expansion = adaLN_expansion
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, adaLN_expansion * 6 * self.dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x,
        score_mod=None,
        block_mask=None,
        adaLN_emb=None,
        adaLN_packing_fn=None,
        rope_forward=None,
        **kwargs,
    ):
        # Embed and expand adaLN_embs: B x (exp*6*D) -> sum(N_i) x (6*D)
        adaLN_emb_packed = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb))
        adaLN_emb_packed = expand_to_padded_seq(adaLN_emb_packed, x)
        gate_msa, gate_mlp, shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_emb_packed.chunk(
            6, dim=-1
        )

        x = x + gate_msa * self.drop_path(
            self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                score_mod=score_mod,
                block_mask=block_mask,
                rope_forward=rope_forward,
            )
        )
        x = x + gate_mlp * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class FlexDecoderBlock(nn.Module):
    """Flexible Transformer Decoder block, i.e. interleaved self- and cross-attention.

    Args:
        dim: Transformer dimension size.
        num_heads: Number of attention heads (overrides head_dim if specified).
        head_dim: Dimension size per attention head.
        mlp_ratio: Ratio of hidden dimension size to input dimension size in the MLP.
        qkv_bias: Whether to use bias in Q, K, V projections.
        proj_bias: Whether to use bias in the output projection layer.
        mlp_bias: Whether to use bias in the MLP layer.
        drop: Dropout rate for attention and MLP linear layers.
        drop_path: Stochastic depth drop rate.
        act_layer: Activation layer used in the MLP.
        norm_layer: Pre-normalization layer.
        gated_mlp: Whether to use a gated MLP layer, e.g. for SwiGLU.
        qk_norm: Whether to apply QK normalization.
        muP_scale: Whether to use μP-compatible attention scaling.
    """

    def __init__(
        self,
        dim,
        num_heads=None,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
        gated_mlp=False,
        qk_norm=False,
        muP_scale=True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        num_heads = num_heads or dim // head_dim

        self.self_attn = FlexSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            muP_scale=muP_scale,
            norm_layer=norm_layer,
        )
        self.cross_attn = FlexCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            muP_scale=muP_scale,
            norm_layer=norm_layer,
        )

        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=mlp_bias,
            drop=drop,
        )

    def forward(
        self,
        x,
        context,
        sa_score_mod=None,
        sa_block_mask=None,
        xa_score_mod=None,
        xa_block_mask=None,
        rope_forward=None,
        **kwargs,
    ):
        x = x + self.drop_path(
            self.self_attn(
                self.norm1(x),
                score_mod=sa_score_mod,
                block_mask=sa_block_mask,
                rope_forward=rope_forward,
            )
        )
        x = x + self.drop_path(
            self.cross_attn(
                self.query_norm(x),
                self.context_norm(context),
                score_mod=xa_score_mod,
                block_mask=xa_block_mask,
            )
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FlexDecoderBlockAdaLN(FlexDecoderBlock):
    """Flexible Transformer Decoder block with adaLN-zero modulation.
    See FlexDecoderBlock for arguments.
    """

    def __init__(self, adaLN_expansion: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, adaLN_expansion * 11 * self.dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x,
        context,
        sa_score_mod=None,
        sa_block_mask=None,
        xa_score_mod=None,
        xa_block_mask=None,
        adaLN_emb=None,
        adaLN_packing_fn=None,
        rope_forward=None,
        **kwargs,
    ):
        # Embed and expand adaLN_embs: B x (exp*11*D) -> sum(N_i) x (11*D)
        adaLN_emb_packed = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb))
        adaLN_emb_packed = expand_to_padded_seq(adaLN_emb_packed, x)
        gate_shift_scale = adaLN_emb_packed.chunk(11, dim=-1)
        gate_msa, gate_mxa, gate_mlp = gate_shift_scale[:3]
        (
            shift_msa,
            scale_msa,
            shift_mxa_q,
            scale_mxa_q,
            shift_mxa_c,
            scale_mxa_c,
            shift_mlp,
            scale_mlp,
        ) = gate_shift_scale[3:]

        x = x + gate_msa * self.drop_path(
            self.self_attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                score_mod=sa_score_mod,
                block_mask=sa_block_mask,
                rope_forward=rope_forward,
            )
        )
        x = x + gate_mxa * self.drop_path(
            self.cross_attn(
                modulate(self.query_norm(x), shift_mxa_q, scale_mxa_q),
                modulate(self.context_norm(context), shift_mxa_c, scale_mxa_c),
                score_mod=xa_score_mod,
                block_mask=xa_block_mask,
            )
        )
        x = x + gate_mlp * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x
