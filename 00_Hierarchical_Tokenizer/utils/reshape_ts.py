

from __future__ import annotations
import torch


def to_fake_image(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [B, C, L]
    Returns:
        [B, C, 1, L]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected [B,C,L], got {tuple(x.shape)}")
    return x.unsqueeze(2)


def from_fake_image(x_img: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x_img: [B, C, 1, L]
    Returns:
        [B, C, L]
    """
    if x_img.dim() != 4 or x_img.shape[2] != 1:
        raise ValueError(f"Expected [B,C,1,L], got {tuple(x_img.shape)}")
    return x_img.squeeze(2)


# quick self-test
if __name__ == "__main__":
    x = torch.randn(2, 3, 1024)
    xi = to_fake_image(x)
    xo = from_fake_image(xi)
    assert torch.allclose(x, xo), "Round-trip reshape failed"
    print("OK")