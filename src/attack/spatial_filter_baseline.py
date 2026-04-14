"""Spatial-filter evasion baseline, adapted from Filter_Attack (lbinmeng/Filter_Attack).

The original attack learns a single spatial filter over a training set (white-box,
gradient-based). Here we adapt it to per-sample evasion so it sits alongside our
greedy black-box attack in the same evaluation pipeline.

For each sample (C, T):
    W  ← zeros (C, C), learnable
    filter_matrix = W + I
    x_adv = filter_matrix @ x        # spatial channel remix
    loss  = -CE(model(x_adv), y) + alpha * ||W||_F^2
    minimise over W with Adam

alpha trades off fooling power vs. closeness to identity (stealthiness).
A large alpha → W stays small → less distortion but potentially lower ASR.
We run a short binary-search on alpha (3 candidates) and take the best successful
result, or the closest-to-success result if none succeed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .greedy_attack import AttackResult
from .losses import untargeted_margin


class SpatialFilterBaseline:
    """Per-sample white-box spatial-filter evasion (Filter_Attack style)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        sfreq: float,
        n_steps: int = 200,
        lr: float = 5e-3,
        alpha: float = 1e1,
    ):
        self.model = model
        self.device = torch.device(device)
        self.sfreq = sfreq
        self.n_steps = n_steps
        self.lr = lr
        self.alpha = alpha

    def _run_with_alpha(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        n_channels: int,
        alpha: float,
    ) -> tuple[np.ndarray, float, int]:
        """Returns (best_W_np, best_margin, steps_run)."""
        W = torch.zeros(n_channels, n_channels, device=self.device, requires_grad=True)
        optimizer = optim.Adam([W], lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_margin = float("inf")
        best_W_np = np.zeros((n_channels, n_channels), dtype=np.float32)

        self.model.eval()
        eye = torch.eye(n_channels, device=self.device)

        for step in range(self.n_steps):
            optimizer.zero_grad()
            x_adv = (W + eye) @ x_tensor          # (C, C) @ (1, C, T) → (1, C, T)
            logits = self.model(x_adv)
            ce = criterion(logits, y_tensor)
            reg = torch.norm(W, p="fro") ** 2
            loss = -ce + alpha * reg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                margin = untargeted_margin(
                    logits.detach().cpu().numpy().squeeze(0), int(y_tensor.item())
                )
                if margin < best_margin:
                    best_margin = margin
                    best_W_np = W.detach().cpu().numpy().copy()

            if best_margin < 0.0:
                break

        return best_W_np, best_margin, self.n_steps

    def run(self, x: np.ndarray, y: int) -> AttackResult:
        """Craft an adversarial spatial filter for a single sample x (C, T)."""
        n_channels = x.shape[0]
        x_tensor = torch.as_tensor(x[None], dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor([y], dtype=torch.long, device=self.device)

        # Binary-search over alpha: small = more powerful, large = stealthier
        alphas = [self.alpha * 0.1, self.alpha, self.alpha * 10.0]
        best_W_np = np.zeros((n_channels, n_channels), dtype=np.float32)
        best_margin = float("inf")

        for a in alphas:
            W_np, margin, _ = self._run_with_alpha(x_tensor, y_tensor, n_channels, a)
            # prefer successful; among successful, prefer smallest ||W||
            if margin < best_margin or (margin < 0.0 and best_margin >= 0.0):
                best_margin = margin
                best_W_np = W_np

        # Build final adversarial sample
        eye = np.eye(n_channels, dtype=np.float32)
        filter_matrix = best_W_np + eye
        x_adv = (filter_matrix @ x).astype(np.float32)
        delta = x_adv - x

        return AttackResult(
            x_adv=x_adv,
            delta=delta,
            support=list(range(n_channels)),  # all channels touched
            coeffs=best_W_np,
            margin=best_margin,
            success=bool(best_margin < 0.0),
            queries_used=self.n_steps * len(alphas),
            budget_exhausted=False,
        )
