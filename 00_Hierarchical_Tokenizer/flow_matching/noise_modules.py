import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["MinRFNoiseModule"]


class MinRFNoiseModule(nn.Module):
    """
    Minimal Rectified Flow (RF) noise module, adapted from https://github.com/cloneofsimo/minRF.

    Args:
        clean_images_read_key: Key for reading clean images from data_dict.
        noised_images_write_key: Key for writing noised images to data_dict.
        timesteps_write_key: Key for writing timesteps to data_dict.
        sigmas_write_key: Key for writing sigmas to data_dict.
        ln: Whether to use logit-normal noise.
        stratisfied: Whether to use stratified sampling for logit-normal noise.
        mode_scale: Mode scale, used if ln is False. See SD3: https://arxiv.org/abs/2403.03206.
        noise_write_key: Key for writing noise to data_dict.
        noise_read_key: Key for reading noise from data_dict
    """

    def __init__(
        self,
        clean_images_read_key: str,
        noised_images_write_key: str,
        timesteps_write_key: str,
        sigmas_write_key: str,
        ln: bool = True,
        stratisfied: bool = False,
        mode_scale: float = 0.0,
        noise_write_key: str = "flow_noise",
        noise_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.clean_images_read_key = clean_images_read_key
        self.noised_images_write_key = noised_images_write_key
        self.timesteps_write_key = timesteps_write_key
        self.sigmas_write_key = sigmas_write_key
        self.noise_write_key = noise_write_key
        self.noise_read_key = noise_read_key

        self.ln = ln
        self.stratisfied = stratisfied
        self.mode_scale = mode_scale

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.clean_images_read_key not in data_dict:
            return data_dict
        
        if self.noised_images_write_key in data_dict:
            return data_dict

        clean_images = data_dict[self.clean_images_read_key]
        device = clean_images[0].device
        batch_size = len(clean_images)

        if self.noise_read_key is not None:
            noises = data_dict[self.noise_read_key]
        else:
            noises = [torch.randn_like(img) for img in clean_images]

        if self.ln:
            if self.stratisfied:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, batch_size + 1).to(device)
                z = quantiles[:-1] + torch.rand((batch_size,)).to(device) / batch_size
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                sigmas = torch.sigmoid(z)
            else:
                nt = torch.randn((batch_size,)).to(device)
                sigmas = torch.sigmoid(nt)
        else:
            sigmas = torch.rand((batch_size,)).to(device)
            if self.mode_scale != 0.0:
                # See SD3 paper, https://arxiv.org/abs/2403.03206.
                sigmas = (
                    1
                    - sigmas
                    - self.mode_scale * (torch.cos(math.pi * sigmas / 2) ** 2 - 1 + sigmas)
                )

        noised_images_list = [
            sigma * noise + (1.0 - sigma) * clean_img
            for sigma, noise, clean_img in zip(sigmas, noises, clean_images)
        ]

        data_dict[self.noised_images_write_key] = noised_images_list
        data_dict[self.noise_write_key] = noises
        data_dict[self.sigmas_write_key] = sigmas
        data_dict[self.timesteps_write_key] = sigmas

        return data_dict