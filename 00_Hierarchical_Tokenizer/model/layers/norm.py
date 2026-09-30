import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Fp32LayerNorm"]


class Fp32LayerNorm(nn.LayerNorm):
    """
    Mixed precision friendly LayerNorm.
    From torchtune.modules.Fp32LayerNorm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
