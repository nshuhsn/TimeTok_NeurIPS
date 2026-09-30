import math, torch
import torch.nn.functional as F 
import torch.nn as nn
from scipy.signal import savgol_coeffs

class GaussianSmooth1D(nn.Module):
    def __init__(
        self,
        read_key: str = "ts",
        write_key: str = "ts_smooth",
        Kmax: int = 128,
        eval_keep_k_read_key: str = "eval_keep_k",
        train_keep_k_read_key: str = "train_keep_k",
        gamma: float = 1.0,
        kernel_mix: str = "gauss_only",   # "gauss_only" | "ma_only" | "sg_only"
        padding_mode: str = "reflect",     # "reflect" | "replicate" | "circular"
        sigma_override_read_key: str | None = None,
    ):
        super().__init__()
        self.read_key = read_key
        self.write_key = write_key
        self.Kmax = Kmax
        self.eval_keep_k_read_key = eval_keep_k_read_key
        self.train_keep_k_read_key = train_keep_k_read_key
        self.gamma = gamma
        self.kernel_mix = kernel_mix
        self.padding_mode = padding_mode
        self.sigma_override_read_key = sigma_override_read_key

    @staticmethod
    def _as_list(v):
        return v if isinstance(v, (list, tuple)) else [v]

    def _sigma_from_k(self, k: int, L: int):
        """
        """

        m = math.log2(self.Kmax)
        i = math.log2(max(1, int(k)))
        s = (m - i) / m  # in [0,1]


        W_min = 1.0
        W_max = 0.5 * float(L)
        W = W_min + (W_max - W_min) * (s ** self.gamma)


        sigma = max(0.0, (W - 1.0) / 6.0)
        return sigma

    def _get_polyorder(self, K: int):
        """
        """
        if K <= 11:
            polyorder = 2
        else:
            polyorder = 3
        

        polyorder = min(polyorder, K - 1)
        return max(1, polyorder)  

    def forward(self, data_dict):
        if self.read_key not in data_dict:
            return data_dict

        xs = data_dict[self.read_key]  # list of [1,C,L]
        B = len(xs)

        # 1) Obtain keep_k
        keep_ks = None
        if (not self.training) and (self.eval_keep_k_read_key in data_dict):
            keep_ks = self._as_list(data_dict[self.eval_keep_k_read_key])
            assert len(keep_ks) == B, "eval_keep_k"
        elif self.training:
            if self.train_keep_k_read_key in data_dict:
                keep_ks = self._as_list(data_dict[self.train_keep_k_read_key])
                assert len(keep_ks) == B, "train_keep_k"
            else:
                keep_ks = [self.Kmax] * B

        sigma_overrides = None
        if (self.sigma_override_read_key is not None) and (self.sigma_override_read_key in data_dict):
            sigma_overrides = self._as_list(data_dict[self.sigma_override_read_key])
            assert len(sigma_overrides) == B, "sigma_override "

        # 2) Per-sample smoothing
        outs = []
        for i, x in enumerate(xs):
            C, L = x.shape[1], x.shape[-1]

            if sigma_overrides is not None:
                sigma = float(sigma_overrides[i])
            else:
                k = keep_ks[i] if keep_ks is not None else self.Kmax
                sigma = self._sigma_from_k(k, L)

            # Determine kernel size
            half = int(3 * sigma + 0.5)
            half = min(half, max(0, (L - 1) // 2))

            if half == 0:
                outs.append(x)
                continue

            K = 2 * half + 1
            if self.kernel_mix == "ma_only": 
                ker = torch.full((K,), 1.0 / K, device=x.device, dtype=x.dtype)
            
            elif self.kernel_mix == "sg_only":
                polyorder = self._get_polyorder(K)
                coeffs = savgol_coeffs(K, polyorder, deriv=0, delta=1.0)
                ker = torch.tensor(coeffs, device=x.device, dtype=x.dtype)
                ker = ker / ker.sum()  # 
            
            else:  # gauss_only (default)
                grid = torch.arange(-half, half + 1, device=x.device, dtype=x.dtype)
                ker = torch.exp(-0.5 * (grid / sigma) ** 2)
                ker = ker / ker.sum()  # 

            # Depthwise convolution weight [C, 1, K]
            w = ker.view(1, 1, -1).repeat(C, 1, 1)

            x_padded = F.pad(x, (half, half), mode=self.padding_mode)
            yi = F.conv1d(x_padded, w, groups=C)

            outs.append(yi)

        data_dict[self.write_key] = outs
        return data_dict