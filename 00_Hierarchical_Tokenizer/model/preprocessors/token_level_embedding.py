import math
import torch
import torch.nn as nn

def _to_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if torch.is_tensor(v):
        if v.dim() == 0:
            return [int(v.item())]
        return [int(x) for x in v.flatten().tolist()]
    return [v]

class TokenLevelEmbedding(nn.Module):
    """
    Reads keep_k (token count) from data_dict, converts via log2 to index, then Embedding -> [B, dim].
    - During training (self.training=True): uses train_keep_k_read_key
    - During eval/inference (self.training=False): prefers eval_keep_k_read_key
    - Falls back to default_n_tokens if key not found
    - idx range is clamped to [0, num_levels-1]
    """
    def __init__(
        self,
        train_keep_k_read_key: str = "train_keep_k",
        eval_keep_k_read_key: str = "eval_keep_k",
        write_key: str = "dec_toklev_emb",
        num_levels: int = 8,          # {1,2,4,8,16,32,64,128} → {0..7}
        dim: int = 256,
        default_n_tokens: int = 128,  # k to use when key is missing (-> idx=7)
        weight_init_style: str = "xavier",
    ):
        super().__init__()
        self.train_keep_k_read_key = train_keep_k_read_key
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.write_key = write_key
        self.num_levels = num_levels
        self.default_n_tokens = default_n_tokens
        self.embed = nn.Embedding(num_levels, dim)

        self.weight_init_style = weight_init_style
        self.init_weights()

    @staticmethod
    def _idx_from_k(k: int) -> int:
        k = max(1, int(k))
        return int(round(math.log2(k)))

    def _get_batch_size_fallback(self, data_dict):
        # If batch size is hard to estimate, default to 1
        if "dec_temb" in data_dict and torch.is_tensor(data_dict["dec_temb"]):
            return int(data_dict["dec_temb"].shape[0])
        # Use length of list-type input if available
        for v in data_dict.values():
            if isinstance(v, (list, tuple)):
                return len(v)
        return 1

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

    def forward(self, data_dict):
        # 1) Select read_key based on current mode
        prefer_key = self.eval_keep_k_read_key if (not self.training) else self.train_keep_k_read_key

        if prefer_key in data_dict:
            ks = _to_list(data_dict[prefer_key])
        else:
            # Fall back to default if key not found
            B = self._get_batch_size_fallback(data_dict)
            ks = [self.default_n_tokens] * B

        # 2) log2(k) -> idx (clamped)
        idxs = [self._idx_from_k(k) for k in ks]
        # Range correction: [0, num_levels-1]
        idxs = [i for i in idxs]
        idxs = torch.tensor(idxs, dtype=torch.long, device=self.embed.weight.device)

        # 3) Record embedding [B, dim]
        z = self.embed(idxs)
        data_dict[self.write_key] = z
        return data_dict


class CondFuserConcatProj(nn.Module):
    """
    Concatenates time embedding and token-level embedding ([B,dim]x2), then Linear(2D->D) to write_key.
    """
    def __init__(
        self,
        time_read_key: str = "dec_temb",
        toklev_read_key: str = "dec_toklev_emb",
        write_key: str = "dec_temb_fused",
        dim: int = 256,
    ):
        super().__init__()
        self.time_read_key = time_read_key
        self.toklev_read_key = toklev_read_key
        self.write_key = write_key
        self.proj = nn.Linear(2 * dim, dim)

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

    def forward(self, data_dict):
        if self.time_read_key not in data_dict or self.toklev_read_key not in data_dict:
            return data_dict
        t = data_dict[self.time_read_key]     # [B, D]
        z = data_dict[self.toklev_read_key]   # [B, D]
        data_dict[self.write_key] = self.proj(torch.cat([t, z], dim=-1))
        return data_dict
        