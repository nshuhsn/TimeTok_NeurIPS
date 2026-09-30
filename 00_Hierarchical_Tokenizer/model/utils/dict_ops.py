from typing import Any, Callable, Dict, List, Optional

import einops

import torch
import torch.nn as nn

__all__ = [
    "DictKeyFilter",
    "channels_first_to_last",
    "channels_last_to_first",
    "PerSampleOp",
    "sum_tensors",
    "multiply_tensors",
    "concat_tensors",
    "PerSampleReducer",
]


class DictKeyFilter(nn.Module):
    """
    Filters keys in a data dictionary based on allowed or blocked keys.

    If `allowed_keys` is provided, only the keys in `allowed_keys` are kept in `data_dict`.
    If `blocked_keys` is provided, keys in `blocked_keys` are removed from `data_dict`.
    If both are `None`, all keys are kept.

    Args:
        allowed_keys: List of keys to allow.
        blocked_keys: List of keys to block.

    Raises:
        ValueError: If both `allowed_keys` and `blocked_keys` are provided.
    """

    def __init__(
        self,
        allowed_keys: Optional[List[str]] = None,
        blocked_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        if allowed_keys is not None and blocked_keys is not None:
            raise ValueError("Only one of allowed_keys or blocked_keys should be provided.")
        self.allowed_keys = allowed_keys
        self.blocked_keys = blocked_keys

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  allowed_keys={self.allowed_keys!r},\n"
            f"  blocked_keys={self.blocked_keys!r},\n"
            ")"
        )

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.allowed_keys is not None:
            filtered_dict = {k: data_dict[k] for k in self.allowed_keys if k in data_dict}
        elif self.blocked_keys is not None:
            filtered_dict = {k: v for k, v in data_dict.items() if k not in self.blocked_keys}
        else:
            filtered_dict = data_dict
        return filtered_dict


def channels_first_to_last(tensor: torch.Tensor):
    return einops.rearrange(tensor, "b d ... -> b ... d")


def channels_last_to_first(tensor: torch.Tensor):
    return einops.rearrange(tensor, "b ... d -> b d ...")


class PerSampleOp(nn.Module):
    """
    Applies a specified operation to each element of a list in a data dictionary.

    Args:
        read_key: The key in the data dictionary that contains the list to transform.
        write_key: The key under which to store the transformed list.
        per_sample_op: The operation to apply to each element of the list.
    """

    def __init__(
        self,
        read_key: str,
        write_key: str,
        per_sample_op: Callable[[Any], Any],
    ):
        super().__init__()
        self.read_key = read_key
        self.write_key = write_key
        self.per_sample_op = per_sample_op

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  read_key={self.read_key!r},\n"
            f"  write_key={self.write_key!r},\n"
            f"  per_sample_op={self.per_sample_op!r}\n"
            ")"
        )

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        x_list = data_dict[self.read_key]

        # Apply the operation to each element in the list
        x_list = [self.per_sample_op(element) for element in x_list]

        data_dict[self.write_key] = x_list

        return data_dict


def sum_tensors(*tensors):
    return sum(tensors)


def multiply_tensors(*tensors):
    return torch.stack(tensors).prod(dim=0)


def concat_tensors(*tensors, dim=-1):
    return torch.cat(tensors, dim=dim)


class PerSampleReducer(nn.Module):
    """
    Applies a per-sample operation across multiple lists in a data dictionary.

    Assumes that `data_dict` contains lists of elements for different `read_keys`.
    The forward pass applies `per_sample_op` on the i-th element from each list.
    The results are collected into a new list, which is stored under `write_key`.

    Args:
        read_keys: List of keys to read from `data_dict`.
        write_key: Key under which to store the result in `data_dict`.
        per_sample_op: Function to apply to the i-th elements from each list.

    Raises:
        ValueError: If the lists corresponding to `read_keys` are not all the same length.
    """

    def __init__(
        self,
        read_keys: List[str],
        write_key: str,
        per_sample_op: Callable[..., Any],
    ):
        super().__init__()
        self.read_keys = read_keys
        self.write_key = write_key
        self.per_sample_op = per_sample_op

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  read_keys={self.read_keys!r},\n"
            f"  write_key={self.write_key!r},\n"
            f"  per_sample_op={self.per_sample_op!r}\n"
            ")"
        )

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Extract lists from data_dict for each read_key
        lists = []
        for key in self.read_keys:
            if key not in data_dict:
                raise KeyError(f"Key '{key}' not found in data_dict.")
            if not isinstance(data_dict[key], list):
                raise TypeError(f"Value for key '{key}' must be a list.")
            lists.append(data_dict[key])

        # Check that all lists have the same length
        list_lengths = [len(lst) for lst in lists]
        if len(set(list_lengths)) != 1:
            raise ValueError("All lists corresponding to read_keys must have the same length.")

        length = list_lengths[0]
        result_list = []
        for i in range(length):
            elements = [lst[i] for lst in lists]
            combined_element = self.per_sample_op(*elements)
            result_list.append(combined_element)

        data_dict[self.write_key] = result_list
        return data_dict
