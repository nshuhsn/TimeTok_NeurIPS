from functools import partial
from typing import Any, Dict, List, Optional, Union

import mup

import torch
import torch.nn as nn

try:
    from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
except:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

from model.layers.norm import Fp32LayerNorm
from model.layers.transformer_blocks import (
    FlexBlock,
    FlexBlockAdaLN,
    FlexDecoderBlock,
    FlexDecoderBlockAdaLN,
)

__all__ = ["FlexTransformer", "FlexTransformerDecoder"]


def get_from_dict(data_dict, key, default=None):
    return data_dict[key] if key is not None else default




class FlexTransformer(nn.Module):
    """
    Transformer module using FlexAttention.

    Args:
        input_seq_read_key: Key to read the input sequence from the input dictionary.
        output_seq_write_key: Key to write the output sequence into the output dictionary.
        dim: Dimension of the input and output features.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        gated_mlp: Whether to use gated MLP layers in the transformer blocks.
        qk_norm: Whether to apply normalization to the Q and K projections.
        use_act_checkpoint: Whether to use activation checkpointing to save memory.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        zero_init_query_proj: Whether to zero-initialize the query projection layer.
        block_mask_read_key: Optional key to read the block-wise attention mask from the input dictionary.
        score_mod_read_key: Optional key to read score modification functions from the input dictionary.
        adaLN_emb_read_key: Optional key to read embeddings for adaptive LayerNorm (adaLN).
        adaLN_packing_fn_read_key: Optional key to read packing functions for adaLN modulation.
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
        rope_forward_read_key: Optional key to read Rotary Position Embedding (RoPE) forward functions.
        intermediate_layer_write_key: Optional key to write features of intermediate layers into the
            output dictionary. Useful for REPA-style training.
        intermediate_layers: List of layer indices to return features of intermediate layers.
    """

    def __init__(
        self,
        input_seq_read_key: str,
        output_seq_write_key: str,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: Union[nn.Module, partial] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        gated_mlp: bool = True,
        qk_norm: bool = True,
        use_act_checkpoint: bool = False,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        block_mask_read_key: Optional[str] = None,
        score_mod_read_key: Optional[str] = None,
        adaLN_emb_read_key: Optional[str] = None,
        adaLN_packing_fn_read_key: Optional[str] = None,
        adaLN_expansion: int = 1,
        rope_forward_read_key: Optional[str] = None,
        intermediate_layer_write_key: Optional[str] = None,
        intermediate_layers: Optional[List[int]] = [],
    ):
        super().__init__()
        self.input_seq_read_key = input_seq_read_key
        self.output_seq_write_key = output_seq_write_key
        self.block_mask_read_key = block_mask_read_key
        self.score_mod_read_key = score_mod_read_key
        self.adaLN_emb_read_key = adaLN_emb_read_key
        self.adaLN_packing_fn_read_key = adaLN_packing_fn_read_key
        self.rope_forward_read_key = rope_forward_read_key
        self.intermediate_layer_write_key = intermediate_layer_write_key
        self.intermediate_layers = intermediate_layers

        block_fn = FlexBlockAdaLN if adaLN_emb_read_key is not None else FlexBlock

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    adaLN_expansion=adaLN_expansion,
                )
                for i in range(depth)
            ]
        )
        if use_act_checkpoint:
            self.blocks = nn.ModuleList([checkpoint_wrapper(blk) for blk in self.blocks])

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = zero_init_query_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif "wq" in name and self.zero_init_query_proj:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif "wq" in name and self.zero_init_query_proj:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict[self.input_seq_read_key]
        block_mask = get_from_dict(data_dict, self.block_mask_read_key, default=None)
        score_mod = get_from_dict(data_dict, self.score_mod_read_key, default=None)
        adaLN_emb = get_from_dict(data_dict, self.adaLN_emb_read_key, default=None)
        adaLN_packing_fn = get_from_dict(data_dict, self.adaLN_packing_fn_read_key, default=None)
        rope_forward = get_from_dict(data_dict, self.rope_forward_read_key, default=None)

        intermediate_layers = []

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                block_mask=block_mask,
                score_mod=score_mod,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
                rope_forward=rope_forward,
            )
            if layer_idx in self.intermediate_layers:
                intermediate_layers.append(x)

        # Return Transformer output
        data_dict[self.output_seq_write_key] = x

        # Optionally return features of intermediate layers
        if self.intermediate_layer_write_key is not None:
            if len(intermediate_layers) == 1:
                intermediate_layers = intermediate_layers[0]
            data_dict[self.intermediate_layer_write_key] = intermediate_layers

        return data_dict



class FlexTransformerDecoder(nn.Module):
    """
    Transformer decoder module using FlexAttention. Ever layer is interleaved with a cross-attention
    layer, used to cross-attend to a context.

    Args:
        input_seq_read_key: Key to read the input sequence from the input dictionary.
        output_seq_write_key: Key to write the output sequence into the output dictionary.
        dim: Dimension of the input and output features.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        gated_mlp: Whether to use gated MLP layers in the transformer blocks.
        qk_norm: Whether to apply normalization to the Q and K projections.
        use_act_checkpoint: Whether to use activation checkpointing to save memory.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        zero_init_query_proj: Whether to zero-initialize the query projection layer.
        block_mask_read_key: Optional key to read the block-wise attention mask from the input dictionary.
        score_mod_read_key: Optional key to read score modification functions from the input dictionary.
        adaLN_emb_read_key: Optional key to read embeddings for adaptive LayerNorm (adaLN).
        adaLN_packing_fn_read_key: Optional key to read packing functions for adaLN modulation.
        rope_forward_read_key: Optional key to read Rotary Position Embedding (RoPE) forward functions.
        intermediate_layer_write_key: Optional key to write features of intermediate layers into the output dictionary.
            Useful for REPA-style training.
        intermediate_layers: List of layer indices to return features of intermediate layers.
    """

    def __init__(
        self,
        input_seq_read_key: str,
        context_seq_read_key: str,
        output_seq_write_key: str,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: Union[nn.Module, partial] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        gated_mlp: bool = True,
        qk_norm: bool = True,
        use_act_checkpoint: bool = False,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        sa_block_mask_read_key: Optional[str] = None,
        xa_block_mask_read_key: Optional[str] = None,
        sa_score_mod_read_key: Optional[str] = None,
        xa_score_mod_read_key: Optional[str] = None,
        adaLN_emb_read_key: Optional[str] = None,
        adaLN_packing_fn_read_key: Optional[str] = None,
        adaLN_expansion: int = 1,
        rope_forward_read_key: Optional[str] = None,
        intermediate_layer_write_key: Optional[str] = None,
        intermediate_layers: Optional[List[int]] = [],
    ):
        super().__init__()
        self.input_seq_read_key = input_seq_read_key
        self.context_seq_read_key = context_seq_read_key
        self.output_seq_write_key = output_seq_write_key
        self.sa_block_mask_read_key = sa_block_mask_read_key
        self.xa_block_mask_read_key = xa_block_mask_read_key
        self.sa_score_mod_read_key = sa_score_mod_read_key
        self.xa_score_mod_read_key = xa_score_mod_read_key
        self.adaLN_emb_read_key = adaLN_emb_read_key
        self.adaLN_packing_fn_read_key = adaLN_packing_fn_read_key
        self.rope_forward_read_key = rope_forward_read_key
        self.intermediate_layer_write_key = intermediate_layer_write_key
        self.intermediate_layers = intermediate_layers

        block_fn = FlexDecoderBlockAdaLN if adaLN_emb_read_key is not None else FlexDecoderBlock

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    adaLN_expansion=adaLN_expansion,
                )
                for i in range(depth)
            ]
        )
        if use_act_checkpoint:
            self.blocks = nn.ModuleList([checkpoint_wrapper(blk) for blk in self.blocks])

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = zero_init_query_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif "wq" in name and self.zero_init_query_proj:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif "wq" in name and self.zero_init_query_proj:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = data_dict[self.input_seq_read_key]
        context = data_dict[self.context_seq_read_key]
        sa_block_mask = get_from_dict(data_dict, self.sa_block_mask_read_key, default=None)
        xa_block_mask = get_from_dict(data_dict, self.xa_block_mask_read_key, default=None)
        sa_score_mod = get_from_dict(data_dict, self.sa_score_mod_read_key, default=None)
        xa_score_mod = get_from_dict(data_dict, self.xa_score_mod_read_key, default=None)
        adaLN_emb = get_from_dict(data_dict, self.adaLN_emb_read_key, default=None)
        adaLN_packing_fn = get_from_dict(data_dict, self.adaLN_packing_fn_read_key, default=None)
        rope_forward = get_from_dict(data_dict, self.rope_forward_read_key, default=None)

        intermediate_layers = []

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                context,
                sa_block_mask=sa_block_mask,
                xa_block_mask=xa_block_mask,
                sa_score_mod=sa_score_mod,
                xa_score_mod=xa_score_mod,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
                rope_forward=rope_forward,
            )
            if layer_idx in self.intermediate_layers:
                intermediate_layers.append(x)

        # Return Transformer output
        data_dict[self.output_seq_write_key] = x

        # Optionally return features of intermediate layers
        if self.intermediate_layer_write_key is not None:
            if len(intermediate_layers) == 1:
                intermediate_layers = intermediate_layers[0]
            data_dict[self.intermediate_layer_write_key] = intermediate_layers

        return data_dict
