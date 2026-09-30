
from typing import Any, Union
from yaml import safe_load, YAMLError
import io
import json
import os
from pathlib import Path
from typing import Optional, Union

from safetensors.torch import load as load_st
from safetensors.torch import save_file

import torch

from hydra.utils import instantiate

__all__ = ["save_safetensors", "load_safetensors", "load_model_from_safetensors"]

ALLOWED_TARGETS = [
    # Wrapper targets
    "flextok.flextok_wrapper.FlexTok",
    "flextok.flextok_wrapper.FlexTokFromHub",
    "flextok.vae_wrapper.StableDiffusionVAE",
    # Flow Matching targets
    "flextok.flow_matching.pipelines.MinRFPipeline",
    "flextok.flow_matching.noise_modules.MinRFNoiseModule",
    "flextok.flow_matching.cfg_utils.MomentumBuffer",
    # Model Utils targets
    "flextok.model.utils.wrappers.SequentialModuleDictWrapper",
    "flextok.model.utils.posembs.PositionalEmbedding",
    "flextok.model.utils.posembs.PositionalEmbeddingAdder",
    "flextok.model.utils.dict_ops.DictKeyFilter",
    "flextok.model.utils.dict_ops.PerSampleOp",
    "flextok.model.utils.dict_ops.PerSampleReducer",
    "flextok.model.utils.dict_ops.channels_first_to_last",
    "flextok.model.utils.dict_ops.channels_last_to_first",
    # Model Trunks targets
    "flextok.model.trunks.transformers.FlexTransformer",
    "flextok.model.trunks.transformers.FlexTransformerDecoder",
    # Model Preprocessors targets
    "flextok.model.preprocessors.token_dropout.MaskedNestedDropout",
    "flextok.model.preprocessors.time_embedding.TimestepEmbedder",
    "flextok.model.preprocessors.registers.Registers1D",
    "flextok.model.preprocessors.patching.PatchEmbedder",
    "flextok.model.preprocessors.nullcond.ZeroNullCond",
    "flextok.model.preprocessors.nullcond.LearnedNullCond",
    "flextok.model.preprocessors.mask_tokens.MaskTokenModule",
    "flextok.model.preprocessors.linear.LinearLayer",
    "flextok.model.preprocessors.flex_seq_packing.BlockWiseSequencePacker",
    # Model Postprocessors targets
    "flextok.model.postprocessors.seq_unpacking.SequenceUnpacker",
    "flextok.model.postprocessors.heads.LinearHead",
    "flextok.model.postprocessors.heads.ToPatchesLinearHead",
    "flextok.model.postprocessors.heads.MLPHead",
    # Model Layers targets
    "flextok.model.layers.transformer_blocks.FlexBlock",
    "flextok.model.layers.transformer_blocks.FlexBlockAdaLN",
    "flextok.model.layers.transformer_blocks.FlexDecoderBlock",
    "flextok.model.layers.transformer_blocks.FlexDecoderBlockAdaLN",
    "flextok.model.layers.norm.Fp32LayerNorm",
    "flextok.model.layers.mup_readout.MuReadoutFSDP",
    "flextok.model.layers.mlp.Mlp",
    "flextok.model.layers.mlp.GatedMlp",
    "flextok.model.layers.drop_path.DropPath",
    "flextok.model.layers.attention.FlexAttention",
    "flextok.model.layers.attention.FlexSelfAttention",
    "flextok.model.layers.attention.FlexCrossAttention",
    # Regularizers targets
    "flextok.regularizers.quantize_fsq.FSQ",
]

MAX_LEN_YAML_PARSE = 10_000


def save_safetensors(state_dict, ckpt_path, metadata_dict=None):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
    if metadata_dict is not None:
        metadata = {k: str(v) for k, v in metadata_dict.items()}
    else:
        metadata = None
    save_file(state_dict, ckpt_path, metadata=metadata)


def _sanitize_hydra_config(cfg: Union[dict, list, tuple]) -> None:
    """
    Recursively scan a Hydra-style config for any `_target_` entries and ensure 
    they only point inside a known list of classes and functions. Raises if any 
    disallowed target is found.
    """
    if isinstance(cfg, dict):
        for key, val in cfg.items():
            if key == "_target_":
                if not isinstance(val, str) or not val in ALLOWED_TARGETS:
                    raise ValueError(f"Potentially unsafe _target_ in Hydra config: {val!r}")
            else:
                _sanitize_hydra_config(val)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            _sanitize_hydra_config(item)


def safe_parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        if not isinstance(v, str) or len(v) > MAX_LEN_YAML_PARSE:
            metadata[k] = v
            continue
        try:
            parsed = safe_load(v.replace('None', 'null'))
            metadata[k] = parsed
        except YAMLError:
            metadata[k] = v
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, "rb") as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors

    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})

    # Safely parse and sanitize before handing it off to hydra.utils.instantiate
    metadata = safe_parse_metadata(metadata)
    _sanitize_hydra_config(metadata)
    
    return tensors, metadata


def load_model_from_safetensors(
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None,
    to_eval: bool = True,
) -> torch.nn.Module:
    """Loads a safetensors checkpoint from the given path and instantiates
    a model from the config safed in it.

    Args:
        ckpt_path: Path to .safetensors checkpoint
        device: Optional torch device
        to_eval: Set to call .eval() on model

    Returns:
        Model with loaded weights.
    """
    ckpt, config = load_safetensors(ckpt_path)
    model = instantiate(config)
    model.load_state_dict(ckpt)

    if device is not None:
        model = model.to(device)

    if to_eval:
        model = model.eval()

    return model
