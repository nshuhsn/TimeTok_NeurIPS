
from io import BytesIO
from typing import List

import einops
import requests

import torch

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

__all__ = ["img_from_url", "imgs_from_urls", "denormalize", "batch_to_pil"]


def img_from_url(
    url: str,
    img_size: int = 256,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> torch.Tensor:
    """
    Download an image from a URL, apply preprocessing, and return a tensor.

    Parameters:
        url (str): URL of the image.
        img_size (int): The size to which the image is resized and cropped.
        mean (List[float]): Mean for normalization.
        std (List[float]): Standard deviation for normalization.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download image from {url}") from e

    try:
        img_pil = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ValueError("Failed to open image from downloaded data.") from e

    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transform(img_pil).unsqueeze(0)


def imgs_from_urls(
    urls: List[str],
    img_size: int = 256,
    mean: List[float] = [0.5, 0.5, 0.5],
    std: List[float] = [0.5, 0.5, 0.5],
) -> torch.Tensor:
    """
    Download and preprocess a batch of images from a list of URLs.

    Parameters:
        urls (List[str]): List of image URLs.
        img_size (int): The size to which each image is resized and cropped.
        mean (List[float]): Mean for normalization.
        std (List[float]): Standard deviation for normalization.

    Returns:
        torch.Tensor: A batch tensor of shape (N, C, H, W), where N is the number of images.
    """
    images = [img_from_url(url, img_size, mean, std) for url in urls]
    return torch.cat(images, dim=0)


def denormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Denormalizes an image.

    Args:
        img (torch.Tensor): Image to denormalize.
        mean (tuple): Mean to use for denormalization.
        std (tuple): Standard deviation to use for denormalization.
    """
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def batch_to_pil(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], stack_horizontal=True):
    """
    Converts a batched tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Tensor to convert.
        mean (tuple): Mean to use for denormalization.
        std (tuple): Standard deviation to use for denormalization.
    """
    if stack_horizontal:
        tensor_stacked = einops.rearrange(tensor, "b c h w -> c h (b w)")
    else:
        tensor_stacked = einops.rearrange(tensor, "b c h w -> c (b h) w")
    return TF.to_pil_image(denormalize(tensor_stacked.detach().cpu(), mean, std).clamp(0, 1))
