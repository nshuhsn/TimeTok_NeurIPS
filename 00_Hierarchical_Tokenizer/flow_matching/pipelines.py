import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from tqdm import tqdm

from utils.misc import to_2tuple

from .cfg_utils import MomentumBuffer, classifier_free_guidance, normalized_guidance

__all__ = ["MinRFPipeline"]


class MinRFPipeline:
    """
    Minimal Rectified Flow (RF) inference pipeline, adapted from https://github.com/cloneofsimo/minRF.

    Args:
        model: Flow model (e.g. TimeTok decoder).
        noise_read_key: Key for reading noise from data_dict.
        target_sizes_read_key: Key for reading target sizes from data_dict.
            Needs to be given in terms of latent space dimensions.
        latents_read_key: Key for reading latents from data_dict.
        timesteps_read_key: Key for reading timesteps from data_dict.
        noised_images_read_key: Key for reading noised images from data_dict.
        reconst_write_key: Key for writing reconstructed images to data_dict.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        noise_read_key: Optional[str] = None,
        target_sizes_read_key: Optional[str] = None,
        latents_read_key: Optional[str] = None,
        timesteps_read_key: Optional[str] = None,
        noised_images_read_key: Optional[str] = None,
        reconst_write_key: Optional[str] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.noise_read_key = noise_read_key
        self.target_sizes_read_key = target_sizes_read_key
        self.latents_read_key = latents_read_key
        self.timesteps_read_key = timesteps_read_key
        self.noised_images_read_key = noised_images_read_key
        self.reconst_write_key = reconst_write_key
        self.out_channels = out_channels

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: int = 25,
        vae_image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        perform_norm_guidance: bool = False,
    ) -> Dict[str, Any]:
        """
        Inference pipeline forward function, performing the denoising.

        Args:
            data_dict: Data dictionary.
            generator: Optional torch.Generator to set seed for noise sampling.
            timesteps: Number of inference steps.
            vae_image_sizes: Image sizes, needs to be given in terms of latent space dimensions.
                E.g. a 256x256 image has VAE latent size 32.
            verbose: Whether to show progress bar.
            guidance_scale: Guidance scale.
            perform_norm_guidance: Whether to perform APG (Sadat et al., 2024),
                https://arxiv.org/abs/2410.02416. If False, uses classifier-free guidance.
        """

        do_cfg = callable(guidance_scale) or guidance_scale != 1.0

        if vae_image_sizes is None:
            vae_image_sizes = data_dict[self.target_sizes_read_key]
        elif isinstance(vae_image_sizes, int):
            batch_size = len(data_dict[self.latents_read_key])
            vae_image_sizes = [to_2tuple(vae_image_sizes) for _ in range(batch_size)]
        assert isinstance(vae_image_sizes, list)
        batch_size = len(vae_image_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            images_list = data_dict[self.noise_read_key]
        else:
            images_list = [
                torch.randn(
                    (1, self.out_channels, w),
                    generator=generator,
                    device=self.model.device,
                )
                for h, w in vae_image_sizes
            ]

        # Set step values
        dt = 1.0 / timesteps

        if perform_norm_guidance:
            momentum_buffers = [MomentumBuffer(-0.5) for _ in range(batch_size)]

        if verbose:
            pbar = tqdm(total=timesteps)

        for i in range(timesteps, 0, -1):
            t = i / timesteps
            timesteps_tensor = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.timesteps_read_key] = timesteps_tensor
            data_dict[self.noised_images_read_key] = images_list

            # 1.1 Conditional forward pass
            data_dict_cond = copy.deepcopy(data_dict)

            data_dict_cond = self.model(data_dict_cond)
            model_output_list = data_dict_cond[self.reconst_write_key]

            # shapes = [t.shape for t in model_output_list]
            # print(len(model_output_list))
            # print(shapes)

            # 1.2 (Optional) unconditional forward pass
            if do_cfg:
                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(t)
                else:
                    guidance_scale_value = guidance_scale

                data_dict_uncond = copy.deepcopy(data_dict)
                data_dict_uncond["eval_dropout_mask"] = [True] * len(model_output_list)
                data_dict_uncond = self.model(data_dict_uncond)
                model_output_list_uncond = data_dict_uncond[self.reconst_write_key]

                model_output_list_cfg = []
                for j, (output_cond, output_uncond) in enumerate(
                    zip(model_output_list, model_output_list_uncond)
                ):
                    if not perform_norm_guidance:
                        output_cfg = classifier_free_guidance(
                            output_cond, output_uncond, guidance_scale_value
                        )
                    else:
                        output_cfg = normalized_guidance(
                            output_cond,
                            output_uncond,
                            guidance_scale_value,
                            momentum_buffers[j],
                            eta=0.0,
                            norm_threshold=2.5,
                        )
                    model_output_list_cfg.append(output_cfg)

                model_output_list = model_output_list_cfg

            # shapes = [t.shape for t in images_list]
            # print(len(images_list))
            # print(shapes)

            # shapes = [t.shape for t in model_output_list]
            # print(len(model_output_list))
            # print(shapes)

            # [[1, 16, 1, 34], [], [] ,...]

            
            # 2. Compute previous image: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                images_list_next = []
                for model_output, image in zip(model_output_list, images_list):
                    image_next = image - dt * model_output
                    images_list_next.append(image_next)
                images_list = images_list_next


            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        data_dict[self.reconst_write_key] = images_list
        return data_dict
