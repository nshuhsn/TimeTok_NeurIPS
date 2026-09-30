from functools import partial

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .norm import Fp32LayerNorm

__all__ = ["FlexAttention", "FlexSelfAttention", "FlexCrossAttention"]


class FlexAttention(nn.Module):
    """Flexible multi-head attention module with optional normalization, scaling, and dropout.

    Args:
        dim: Transformer dimension size.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in Q, K, V projections.
        proj_bias: Whether to use bias in the output projection layer.
        proj_drop: Dropout rate for the projection layer. 0.0 = no dropout.
        qk_norm: Whether to apply QK normalization.
        muP_scale: Whether to use μP-compatible attention scaling.
        norm_layer: Normalization layer when using QK-norm.
        use_flex_attention: Set to False to fall back to standard SDPA.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=False,
        proj_drop=0.0,
        qk_norm=True,
        muP_scale=True,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
        use_flex_attention=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        # μP compatible scaling. Equals standard 1.0 / head_dim ** 0.5 for commonly used head_dim=64.
        self.scale = 8.0 / self.head_dim if muP_scale else None

        self.use_flex_attention = use_flex_attention
        if use_flex_attention:
            # If used, should always be compiled
            self.flex_attention = torch.compile(flex_attention, dynamic=False)

    def forward(self, xq, xk, xv, score_mod=None, block_mask=None, rope_forward=None):
        """Forward pass of the FlexAttention module.

        Args:
            xq: Input tensor of shape [B, N_q, D] to compute queries.
            xk: Input tensor of shape [B, N_kv, D] to compute keys.
            xv: Input tensor of shape [B, N_kv, D] to compute values.
            score_mod: Optional flex_attention function to modify attention scores.
            block_mask: Optional flex_attention mask for efficient sparse attention computation.
            rope_forward: Optional function to apply Rotary Position Embeddings (RoPE) to Q, K, V.

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """

        # q, k, v each of shape [batch_size, sequence_length, num_heads*head_dim]
        q, k, v = self.wq(xq), self.wk(xk), self.wv(xv)

        # Separate heads of q, k, v into [batch_size, num_heads, sequence_length, head_dim]
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Optional QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply optional RoPE
        if rope_forward is not None:
            q, k, v = rope_forward(q, k, v)

        if self.use_flex_attention:
            # Run FlexAttention with score_mod and/or block_mask
            x = self.flex_attention(
                q, k, v, score_mod=score_mod, block_mask=block_mask, scale=self.scale
            )
        else:
            # When falling back to SDPA, we use block_mask as the attention mask
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=block_mask, scale=self.scale
            )

        # Combine heads back to [batch_size, sequence_length, num_heads*head_dim]
        x = einops.rearrange(x, "b h n d -> b n (h d)")

        # Project and apply optional dropout
        x = self.proj_drop(self.proj(x))
        return x


class FlexSelfAttention(FlexAttention):
    """Flexible multi-head self-attention module with optional normalization, scaling, and dropout.
    See FlexAttention for arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, score_mod=None, block_mask=None, rope_forward=None, **kwargs):
        """Forward pass of the FlexSelfAttention module.

        Args:
            x: Input tensor of shape [B, ..., D] to compute queries, keys, and values from.
            score_mod: Optional flex_attention function to modify attention scores.
            block_mask: Optional flex_attention mask for efficient sparse attention computation.
            rope_forward: Optional function to apply Rotary Position Embeddings (RoPE) to Q, K, V.

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """
        return super().forward(
            xq=x,
            xk=x,
            xv=x,
            score_mod=score_mod,
            block_mask=block_mask,
            rope_forward=rope_forward,
        )


class FlexCrossAttention(FlexAttention):
    """Flexible multi-head cross-attention module with optional normalization, scaling, and dropout.
    See FlexAttention for arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context, score_mod=None, block_mask=None, rope_forward=None, **kwargs):
        """Forward pass of the FlexSelfAttention module.

        Args:
            x: Input tensor of shape [B, ..., D] to compute queries from.
            context: Context tensor of shape [B, ..., D] to compute keys, and values from.
            score_mod: Optional flex_attention function to modify attention scores.
            block_mask: Optional flex_attention mask for efficient sparse attention computation.
            rope_forward: Optional function to apply Rotary Position Embeddings (RoPE) to Q, K, V.

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """
        return super().forward(
            xq=x,
            xk=context,
            xv=context,
            score_mod=score_mod,
            block_mask=block_mask,
            rope_forward=rope_forward,
        )
