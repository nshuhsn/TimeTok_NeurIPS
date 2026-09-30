import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

def default_dfa_scales(L: int, s_min: int = 4, s_max_frac: float = 0.25, num_scales: int = 12) -> List[int]:
    s_max = max(s_min+1, int(L * s_max_frac))
    # Generate approximately num_scales points on logspace, ensure unique/sorted integers
    raw = np.unique(np.clip(np.round(np.logspace(np.log10(s_min), np.log10(s_max), num_scales)), s_min, s_max).astype(int))
    # Remove duplicates with interval <= 1
    scales = []
    last = -10**9
    for v in raw:
        if v - last >= 1:
            scales.append(int(v))
            last = v

    # before: s <= L//2
    scales = [s for s in scales if s >= 4 and s <= L//4]    
    return scales

def dfa_alpha_batch_torch(
    x_bl: torch.Tensor,  
    scales: List[int],
    order: int = 1
) -> torch.Tensor:
    """
    Batch DFA (order 1 only). Returns: alphas [B]
    """
    assert x_bl.ndim == 2, "x_bl must be [B, L]"
    assert order == 1, "batch DFA only supports order 1 with minimal modifications."
    B, L = x_bl.shape
    device = x_bl.device

    x = x_bl - x_bl.mean(dim=1, keepdim=True)
    y = torch.cumsum(x, dim=1)  # [B, L]

    Fs = []     # F(s) for each s, shape [B]
    s_kept = [] # List of actually used s values (only those meeting conditions)

    for s in scales:
        if s < 4:
            continue
        Ns = L // s
        if Ns < 4:
            continue

        # --- forward windows: [B, Ns, s]
        fw = y.unfold(dimension=1, size=s, step=s)  # [B, Ns, s]


        if (L % s) != 0:
            y_rev = torch.flip(y, dims=[1])
            bw = y_rev.unfold(dimension=1, size=s, step=s)[:, :Ns, :]  # [B, Ns, s]
            windows = torch.cat([fw, bw], dim=1)  # [B, 2*Ns, s]
        else:
            windows = fw  # [B, Ns, s]

        t = torch.arange(s, device=device, dtype=x.dtype).view(1, 1, s)  # [1,1,s]
        N = float(s)
        sum_t  = (s - 1) * s * 0.5                        # Σ t
        sum_t2 = (s - 1) * s * (2 * s - 1) / 6.0          # Σ t^2

        y_sum  = windows.sum(dim=2)                       # [B, M]
        ty_sum = (windows * t).sum(dim=2)                 # [B, M]

        denom = (N * sum_t2 - sum_t * sum_t)             # scalar
        a1 = (N * ty_sum - sum_t * y_sum) / denom        # slope [B, M]
        a0 = (y_sum - a1 * sum_t) / N                    # intercept [B, M]

        trend = a1.unsqueeze(-1) * t + a0.unsqueeze(-1)  # [B, M, s]
        resid = windows - trend                          # [B, M, s]
        rms = torch.sqrt((resid ** 2).mean(dim=(2)))  # [B,M]
        F_s = torch.sqrt((rms**2).mean(dim=1))  # [B]
        Fs.append(F_s)
        s_kept.append(s)

    if len(Fs) < 2:
        return torch.full((B,), float('nan'), device=device)

    F_mat = torch.stack(Fs, dim=1)              # [B, S’]
    s_arr = torch.tensor(s_kept, dtype=x.dtype) # 
    s_arr = s_arr.to(device)
    # mask: F>0
    mask = F_mat > 0
    xs = torch.log(s_arr).view(1, -1).expand(B, -1)      # [B, S’]
    ys = torch.log(torch.clamp(F_mat, min=1e-12))        # [B, S’]

    # Use only valid scales
    # (For simple handling of per-sample different masks, calculate mean-covariance formula with mask)
    def masked_mean(t, m):
        w = m.float()
        return (t * w).sum(dim=1) / torch.clamp(w.sum(dim=1), min=1.0)

    mx = masked_mean(xs, mask)            # [B]
    my = masked_mean(ys, mask)            # [B]
    vx = masked_mean((xs - mx.unsqueeze(1))**2, mask)     # [B]
    cxy = masked_mean((xs - mx.unsqueeze(1))*(ys - my.unsqueeze(1)), mask)  # [B]

    # slope = cov/var
    alphas = cxy / torch.clamp(vx, min=1e-12)            # [B]
    return alphas

class SigmaLevelController(torch.nn.Module):
    """
    Takes ts (list of [1,C,L]) as input and:
      1) Measures alpha_raw -> determines raw bin (cap_raw)
      2) Samples window W ~ {1..L//2} -> sigma=(W-1)/6 -> temporary smoothing -> measures alpha_s
      3) If bin(alpha_s) != bin(alpha_raw): sigma_override=sigma, cap_step=bin(alpha_s)
         Else: sigma_override=0, cap_step=cap_raw
      4) keep_k = cap_step (use this step's coarse level as-is)

    Records:
      - sigma_override: [float] * B
      - train_keep_k: [int] * B
      - cap_step: [int] * B

    Config parameters:
      - read_ts_key / write_*_key: input/output keys
      - alpha_edges: list of thresholds with length len(levels)-1 (ascending)
      - levels: e.g., [1,2,4,8,16,32,64,128] (num bins == num levels)
      - dfa_*: DFA settings
    """

    def __init__(
        self,
        read_ts_key: str = "ts",
        write_sigma_key: str = "sigma_override",
        write_keep_k_key: str = "train_keep_k",
        write_eval_keep_k_key: str = "eval_keep_k",
        alpha_edges: Optional[List[float]] = None,   # ← edges, len = len(levels)-1
        levels: Optional[List[int]] = None,
        dfa_order: int = 1,
        dfa_num_scales: int = 12,
        dfa_smin_frac: float = 0.02,
        dfa_smax_frac: float = 0.25,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.read_ts_key = read_ts_key
        self.write_sigma_key = write_sigma_key
        self.write_keep_k_key = write_keep_k_key
        self.write_eval_keep_k_key = write_eval_keep_k_key
        
        self.levels = levels if levels is not None else [1,2,4,8,16,32,64,128]
        # Default edge example (replace in config if needed): len(levels)-1 must be 7
        default_edges = [1, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0][:max(0, len(self.levels)-1)]
        self.alpha_edges = alpha_edges if alpha_edges is not None else default_edges

        # --- Validation ---
        assert len(self.levels) >= 1, "levels cannot be empty."
        assert all(self.levels[i] < self.levels[i+1] for i in range(len(self.levels)-1)), "levels must be ascending."
        assert len(self.alpha_edges) == len(self.levels) - 1, "alpha_edges length must be len(levels)-1."
        assert all(self.alpha_edges[i] < self.alpha_edges[i+1] for i in range(len(self.alpha_edges)-1)), "alpha_edges must be ascending."

        self.dfa_order = int(dfa_order)
        self.dfa_num_scales = int(dfa_num_scales)
        self.dfa_smin_frac = float(dfa_smin_frac)
        self.dfa_smax_frac = float(dfa_smax_frac)

        self.rng = np.random.default_rng(seed if seed is not None else 0)

    def _alpha_to_cap(self, alpha: float) -> int:
        """
        If alpha_edges is t[0..M-2] and levels is L[0..M-1], then:
          alpha < t0           -> 128
        t0 <= alpha < t1       -> 64
        ...
        t_{M-2} <= alpha       -> 1
        """
        edges = self.alpha_edges
        levels = self.levels

        # Get bin index with np.searchsorted
        idx = int(np.searchsorted(edges, alpha, side="left"))
        
        idx = max(0, min(idx, len(levels)-1))
        cap = levels[::-1][idx]  # Reverse indexing
        return int(cap)

    # def _sample_sigma_from_window(self, L: int) -> float:
    #     """
    #     Generate 8 evenly-spaced W candidates in [1, L//2] range,
    #     randomly select one, and convert to sigma.
    #     sigma = (W - 1) / 6
    #     """
    #     W_min = L//10
    #     W_max = max(1, L // 2)
    #     # 8 evenly-spaced candidates
    #     W_candidates = np.linspace(W_min, W_max, num=7)
    #     W_candidates = np.concatenate(([1.0], W_candidates))
    #     W = float(self.rng.choice(W_candidates))
    #     sigma = max(0.0, (W - 1.0) / 6.0)
    #     return float(sigma)
    def _sample_sigma_from_window(self, L: int) -> float:
        """
        Generate 8 evenly-spaced W candidates in [1, L//2] range,
        randomly select one, and convert to sigma.
        sigma = (W - 1) / 6
        """
        W_min = 1
        W_max = max(1, L // 2)
        # 8 evenly-spaced candidates
        W_candidates = np.linspace(W_min, W_max, num=8)
        W = float(self.rng.choice(W_candidates))
        sigma = max(0.0, (W - 1.0) / 6.0)
        return float(sigma)

    @staticmethod
    def _gauss_smooth_1d(x: torch.Tensor, sigma: float, padding_mode: str = "reflect") -> torch.Tensor:
        # x: [1, C, L]
        if sigma <= 0:
            return x
        L = x.shape[-1]
        half = int(3 * sigma + 0.5)
        half = min(half, max(0, (L - 1) // 2))
        if half == 0:
            return x
        K = 2 * half + 1
        device = x.device
        dtype = x.dtype
        grid = torch.arange(-half, half + 1, device=device, dtype=dtype)
        ker = torch.exp(-0.5 * (grid / sigma) ** 2)
        ker = ker / ker.sum()
        w = ker.view(1, 1, -1).repeat(x.shape[1], 1, 1)  # [C,1,K]
        x_pad = F.pad(x, (half, half), mode=padding_mode)
        y = F.conv1d(x_pad, w, groups=x.shape[1])
        return y

    def forward(self, data_dict):
        if self.read_ts_key not in data_dict:
            return data_dict

        xs = data_dict[self.read_ts_key]  # list of [1,C,L]
        B = len(xs)
        assert B > 0

        C, L = xs[0].shape[1], xs[0].shape[-1]
        assert C == 1, "Only perform univariate"

        smin = max(4, int(self.dfa_smin_frac * L))
        scales = default_dfa_scales(L=L, s_min=smin,
                                    s_max_frac=self.dfa_smax_frac,
                                    num_scales=self.dfa_num_scales)

        # ----- Common: calculate raw alpha in batch at once -----
        x_bl = torch.stack([x[0,0] for x in xs], dim=0)   # [B, L]
        alpha_raw_b = dfa_alpha_batch_torch(x_bl, scales, order=self.dfa_order)  # [B]
        cap_raw_list = [self._alpha_to_cap(float(a)) for a in alpha_raw_b.tolist()]

        # ===== eval mode: no smoothing applied =====
        if not self.training:
            sigma_overrides = [0.0 for _ in range(B)]
            keep_ks         = [int(c) for c in cap_raw_list]   # keep_k = cap_raw

            data_dict[self.write_sigma_key] = sigma_overrides
            data_dict[self.write_eval_keep_k_key] = keep_ks
            return data_dict
        # =====================================

        x_tmp_list = []
        sigma_list = []
        for i in range(B):
            sigma = self._sample_sigma_from_window(L)
            xi_s  = self._gauss_smooth_1d(xs[i], sigma)   # [1,1,L]
            x_tmp_list.append(xi_s)
            sigma_list.append(float(sigma))

        x_tmp_bl = torch.stack([xt[0,0] for xt in x_tmp_list], dim=0)  # [B,L]
        alpha_s_b = dfa_alpha_batch_torch(x_tmp_bl, scales, order=self.dfa_order)  # [B]
        cap_s_list = [self._alpha_to_cap(float(a)) for a in alpha_s_b.tolist()]

        sigma_overrides, keep_ks = [], []
        for cap_raw, cap_s, sigma in zip(cap_raw_list, cap_s_list, sigma_list):
            if cap_s == cap_raw:
                sigma_final = 0.0
                cap_step = cap_raw
            else:
                sigma_final = sigma
                cap_step = cap_s
            sigma_overrides.append(float(sigma_final))
            keep_ks.append(int(cap_step))     # This step: keep_k = cap_step

        data_dict[self.write_sigma_key] = sigma_overrides
        data_dict[self.write_keep_k_key] = keep_ks

        return data_dict