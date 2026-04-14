"""AttackGenerator: lightweight EEGNet-style encoder → perturbation coefficient head.

Architecture:
    Input  (N, 1, C, T)
    Block1  Conv2d(1, 16, (1,64)) → BN → ELU → AvgPool(1,4)  → (N, 16, C, T/4)
    Block2  DepthwiseConv(16->32, (C,1)) → BN → ELU → AvgPool(1,4) → (N, 32, 1, T/16)
    Flatten → FC(1024→128) → ELU → FC(128→K*r)
    Reshape (N, K, r) → Tanh × max_coeff_abs   (coefficient-range constraint)

At test time, generate() assembles the full delta (C, T) from predicted coefficients
using the fixed universal support atoms — zero model queries needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .basis import build_basis_matrix
from .support import make_window_partition


class AttackGenerator(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        support_budget_k: int,
        basis_rank_r: int,
        max_coeff_abs: float = 0.75,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.support_budget_k = support_budget_k
        self.basis_rank_r = basis_rank_r
        self.max_coeff_abs = max_coeff_abs

        # set by setup_basis(); used in forward_delta()
        self._universal_support: list[tuple[int, int]] | None = None
        self._n_windows: int | None = None
        self._basis_cfg: dict | None = None

        kern = 64
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern // 2 - 1, kern - kern // 2, 0, 0)),
            nn.Conv2d(1, 16, (1, kern), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )
        flat_size = 32 * (n_times // 16)
        self.head = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ELU(),
            nn.Linear(128, support_budget_k * basis_rank_r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T) → coeffs: (N, K, r)"""
        if x.ndim == 2:
            x = x.unsqueeze(0)          # (1, C, T)
        h = x.unsqueeze(1)              # (N, 1, C, T)
        h = self.block1(h)              # (N, 16, C, T/4)
        h = self.block2(h)              # (N, 32, 1, T/16)
        h = h.reshape(h.size(0), -1)   # (N, flat_size)
        out = self.head(h)              # (N, K*r)
        coeffs = out.reshape(h.size(0), self.support_budget_k, self.basis_rank_r)
        coeffs = torch.tanh(coeffs) * self.max_coeff_abs
        return coeffs

    def setup_basis(
        self,
        universal_support: list[tuple[int, int]],
        n_windows: int,
        basis_min_hz: float,
        basis_max_hz: float,
        basis_mode: str,
        basis_phase_count: int,
        sfreq: float,
    ) -> None:
        """Precompute and register basis matrices as buffers for differentiable delta assembly."""
        self._universal_support = universal_support
        self._n_windows = n_windows
        self._basis_cfg = dict(
            basis_min_hz=basis_min_hz, basis_max_hz=basis_max_hz,
            basis_mode=basis_mode, basis_phase_count=basis_phase_count, sfreq=sfreq,
        )
        partition = make_window_partition(self.n_times, n_windows)
        # one buffer per support atom: (r, window_len)
        for k, (c, w) in enumerate(universal_support):
            s, e = partition.boundaries[w]
            B = build_basis_matrix(
                basis_mode=basis_mode, window_length=e - s, rank=self.basis_rank_r,
                f_min_hz=basis_min_hz, f_max_hz=basis_max_hz, sfreq=sfreq,
                phase_count=basis_phase_count,
            )
            self.register_buffer(f"_basis_{k}", torch.as_tensor(B, dtype=torch.float32))
        self._partition_boundaries = partition.boundaries  # list of (s,e) tuples

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable path: x (N,C,T) → delta (N,C,T). Requires setup_basis() first."""
        assert self._universal_support is not None, "Call setup_basis() before forward_delta()."
        coeffs = self(x)                   # (N, K, r)
        N = coeffs.size(0)
        delta = torch.zeros(N, self.n_channels, self.n_times,
                            dtype=coeffs.dtype, device=coeffs.device)
        for k, (c, w) in enumerate(self._universal_support):
            s, e = self._partition_boundaries[w]
            B = getattr(self, f"_basis_{k}").to(coeffs.device)   # (r, win_len)
            # coeffs[:, k, :] @ B  →  (N, win_len)
            delta[:, c, s:e] = delta[:, c, s:e] + coeffs[:, k, :] @ B
        return delta

    def generate(
        self,
        x_np: np.ndarray,
        universal_support: list[tuple[int, int]],
        n_windows: int,
        basis_min_hz: float,
        basis_max_hz: float,
        basis_mode: str,
        basis_phase_count: int,
        sfreq: float,
        device: torch.device | str | None = None,
    ) -> np.ndarray:
        """Generate delta (C, T) for a single trial x_np (C, T). Zero model queries."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        x_t = torch.as_tensor(x_np[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            coeffs_np = self(x_t).squeeze(0).cpu().numpy()   # (K, r)

        n_channels, n_samples = x_np.shape
        partition = make_window_partition(n_samples, n_windows)
        delta = np.zeros((n_channels, n_samples), dtype=np.float32)

        window_ids = sorted({w for _, w in universal_support})
        basis_by_window: dict[int, np.ndarray] = {}
        for w in window_ids:
            s, e = partition.boundaries[w]
            basis_by_window[w] = build_basis_matrix(
                basis_mode=basis_mode,
                window_length=e - s,
                rank=self.basis_rank_r,
                f_min_hz=basis_min_hz,
                f_max_hz=basis_max_hz,
                sfreq=sfreq,
                phase_count=basis_phase_count,
            )

        for k, (c, w) in enumerate(universal_support):
            s, e = partition.boundaries[w]
            B = basis_by_window[w]       # (r, window_len)
            delta[c, s:e] += (coeffs_np[k] @ B).astype(np.float32)

        return delta
