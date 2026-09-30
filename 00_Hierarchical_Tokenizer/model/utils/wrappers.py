
from typing import Any, Dict

import torch.nn as nn

__all__ = ["SequentialModuleDictWrapper"]


class SequentialModuleDictWrapper(nn.Module):
    """
    A wrapper for sequentially applying a dictionary of modules to a data_dict.

    Args:
        module_dict: A dictionary of module_name: nn.Module pairs.
    """

    def __init__(self, module_dict: Dict[str, nn.Module]):
        super().__init__()
        self.module_dict = nn.ModuleDict(module_dict)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights_muP(self, strict=False, verbose=True):
        """μP weight initialization scheme, applied to every submodule.

        Args:
            strict: If True, throws an Error if a submodule does not implement init_weights_muP()
        """
        for module_name, module in self.module_dict.items():
            if hasattr(module, "init_weights_muP"):
                module.init_weights_muP()
            elif strict:
                raise AttributeError(
                    f"Module '{module_name}' does not implement 'init_weights_muP'."
                )
            elif verbose:
                print(f"Warning: Module '{module_name}' does not implement 'init_weights_muP'.")

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        for module in self.module_dict.values():
            data_dict = module(data_dict)
        return data_dict
