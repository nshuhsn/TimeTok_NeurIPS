import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAxisProjector1D(nn.Module):
    """
    Input  (list): [ [1, T', in_dim], ... ]
    Steps:
      1) Linear(in_dim -> n_tokens): [1, T', K]
      2) Transpose to token-axis:    [1, K, T']
      3) (optional) add token-pos-emb: [1, K, T']
      4) AdaptiveAvgPool1d(T' -> out_dim): [1, K, out_dim]
    Output (list): [ [1, K, out_dim], ... ]  -> FSQ(latents)
    """
    def __init__(
        self,
        input_tensor_list_read_key: str,
        tokens_write_key: str,
        in_dim: int,
        n_tokens: int,
        out_dim: int,
        temporal_pool: str = "adaptive_max",
        add_pos_emb: bool = False,
    ):
        super().__init__()
        self.read_key = input_tensor_list_read_key
        self.write_key = tokens_write_key

        self.in_dim = in_dim
        self.n_tokens = n_tokens
        self.out_dim = out_dim
        self.temporal_pool = temporal_pool
        self.add_pos_emb = add_pos_emb

        # (1) feature -> token-axis projection
        self.proj = nn.Linear(in_dim, n_tokens, bias=True)

        # Expose max length for detokenization
        self.n_max = n_tokens

        # (optional) token-axis positional embedding (broadcast along time)
        if add_pos_emb:
            # shape: [1, K, 1], broadcast along time axis (T')
            self.token_pos = nn.Parameter(torch.zeros(1, n_tokens, 1))
        else:
            self.register_parameter("token_pos", None)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [1, K, T'] -> [1, K, out_dim]
        """
        if self.temporal_pool == "adaptive_avg":
            return F.adaptive_avg_pool1d(x, self.out_dim)
        elif self.temporal_pool == "adaptive_max":
            return F.adaptive_max_pool1d(x, self.out_dim)
        elif self.temporal_pool == "avg":
            assert self.out_dim == 1, "temporal_pool='avg' can only be used with out_dim=1."
            return x.mean(dim=-1, keepdim=True)
        elif self.temporal_pool == "max":
            assert self.out_dim == 1, "temporal_pool='max' can only be used with out_dim=1."
            return x.max(dim=-1, keepdim=True).values
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")

    def _project_and_pool_one(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [1, T', in_dim] -> [1, K, out_dim]
        """
        assert t.ndim == 3 and t.shape[0] == 1 and t.shape[-1] == self.in_dim, \
            f"Expected [1, T', {self.in_dim}], got {list(t.shape)}"

        # (1) proj: [1, T', K]
        if self.in_dim > self.out_dim:
            y = self.proj(t)

        # (2) Transpose to token-axis: [1, K, T']
        y = y.transpose(1, 2)

        # (3) Add token-pos-emb (broadcast along time axis)
        if self.token_pos is not None:
            y = y + self.token_pos

        # (4) Temporal axis pooling: [1, K, out_dim]
        y = self._pool(y)
        return y

    def forward(self, data_dict: dict) -> dict:
        x_list = data_dict[self.read_key]       # List[Tensor(1, T', in_dim)]
        out_list = [self._project_and_pool_one(x) for x in x_list]
        data_dict[self.write_key] = out_list    # List[Tensor(1, K, out_dim)]
        return data_dict