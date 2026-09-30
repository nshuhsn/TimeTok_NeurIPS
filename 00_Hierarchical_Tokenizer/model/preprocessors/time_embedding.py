import math
from typing import Any, Dict

import einops
import mup

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """TimestepEmbedder embeds scalar timesteps into vector representations using an MLP
    and sinusoidal embeddings. Adapted from DiT.

    Args:
        timesteps_read_key: Key to read timesteps from the input data dictionary.
        time_embedding_write_key: Key to write the generated time embeddings back into
            the data dictionary.
        dim: The size of the hidden layers in the MLP.
        frequency_embedding_size: Controls the minimum frequency of the embeddings.
        weight_init_style: The initialization style for weights, default is "xavier".
        max_timestep: Usually 1000.0 for timesteps in [0,1000] or 1.0 for timesteps in [0.0,1.0].
        temb_as_tokens: If True, tembs are returned as a list of [1,1,dim]-sized tokens, such
            that they can be easily concatenated in the sequence packing modules.
    """

    def __init__(
        self,
        timesteps_read_key: str,
        time_embedding_write_key: str,
        dim: int,
        frequency_embedding_size: int = 256,
        weight_init_style: str = "xavier",
        max_timestep: float = 1000.0,
        temb_as_tokens: bool = False,
    ):
        super().__init__()
        self.timesteps_read_key = timesteps_read_key
        self.time_embedding_write_key = time_embedding_write_key
        self.frequency_embedding_size = frequency_embedding_size
        self.max_timestep = max_timestep
        self.temb_as_tokens = temb_as_tokens

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights()

    def init_weights(self):
        """Weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_weights_muP(self):
        """μP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if self.weight_init_style == "xavier":
                    mup.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    mup.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def timestep_embedding(self, t, dim, max_period=256):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        assert torch.all(t <= self.max_timestep)
        half = dim // 2
        freq_scale_factor = 1000.0 / self.max_timestep
        freqs = freq_scale_factor * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        t = data_dict[self.timesteps_read_key]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        if self.temb_as_tokens:
            # t_emb is [batch_size, dim], but we can split it into lists of [1, 1, dim] to use as tokens
            t_emb = t_emb.unsqueeze(1).split(1, dim=0)
        data_dict[self.time_embedding_write_key] = t_emb
        return data_dict
