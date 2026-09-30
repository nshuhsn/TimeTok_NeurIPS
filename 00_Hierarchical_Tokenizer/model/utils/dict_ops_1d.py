import torch

def channels_first_to_last_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Allowed input: [B,C,L] or [B,C,1,L]
    Output: [B,L,C]
    """
    if x.dim() == 4:
        # Expected: [B,C,1,L]
        if x.shape[2] != 1:
            raise ValueError(f"Expected [B,C,1,L] for 4D input, got {tuple(x.shape)}")
        x = x.squeeze(2)  # [B,C,L]
    if x.dim() != 3:
        raise ValueError(f"Expected [B,C,L], got {tuple(x.shape)}")
    return x.transpose(1, 2)  # [B,L,C]


def channels_last_to_first_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Allowed input: [B,L,C] or [B,1,L,C]
    Output: [B,C,L]
    """
    if x.dim() == 4:
        # Expected: [B,1,L,C]
        if x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,L,C] for 4D input, got {tuple(x.shape)}")
        x = x.squeeze(1)  # [B,L,C]
    if x.dim() != 3:
        raise ValueError(f"Expected [B,L,C], got {tuple(x.shape)}")
    return x.transpose(1, 2)  # [B,C,L]