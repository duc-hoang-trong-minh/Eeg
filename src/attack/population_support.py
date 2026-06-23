"""Population (universal-order) channel selection for the score-only sparse-channel attack.

This module implements Option A from the theory refactor: instead of ranking channels
per trial, we estimate a *population* leverage score for each channel as a sample mean
over a probe set D_probe, then attack every test trial by adding channels in that single
fixed (universal) order while refining the waveform per trial with SPSA.

The split is deliberate and matches the claim card:
  - the channel RANKING is population-level  -> Claim 1 (ranking stability, n* ~ nu^2 log C / Delta^2)
  - the waveform REFINEMENT stays per-trial   -> SPSA (unchanged)

Leverage definition (matches the theory's g_c):
    g_c(X) = M(X, y0) - mean_{u in P} M(X + E_{{c}, probe_abs * u}, y0)
i.e. the AVERAGE single-channel margin drop over a fixed bank of probe directions P.
We use the mean (an unbiased sample statistic), NOT the best-case min the old per-trial
greedy used -- that min was exactly why the old code did not match a sample-mean law.

The population estimate is
    L_hat_c = (1/n) sum_{X in D_probe} g_c(X),
a sample mean over n probe trials, so the ranking-stability theorem applies directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .greedy_attack import (
    ChannelFirstScoreAttack,
    QueryBudgetExhausted,
    _apply_peak_ratio_constraint,
    _build_full_trial_basis_matrix,
)
from .losses import untargeted_margin


@dataclass
class PopulationLeverage:
    """Result of population leverage estimation.

    order:    channel indices sorted by descending estimated leverage L_hat
    leverage: L_hat_c per channel, shape (C,)
    std:      per-channel empirical std of g_c(X) over probe trials, shape (C,)
              (an estimate of the sub-Gaussian proxy nu; needed to CONFIRM Claim 1)
    n_probe:  number of probe trials used (the n in the n* law)
    gap:      Delta, the leverage gap at the selection frontier (top-`frontier_k`)
    queries_used: total score queries spent on ranking (amortized once over all trials)
    """

    order: np.ndarray
    leverage: np.ndarray
    std: np.ndarray
    n_probe: int
    gap: float
    queries_used: int


def _make_probe_directions(rank: int, n_dirs: int, seed: int) -> np.ndarray:
    """Fixed bank of +-1 probe directions, shape (n_dirs, rank), shared across channels/trials."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(n_dirs, rank)).astype(np.float32)


def estimate_population_leverage(
    score_fn,
    probe_trials: list[tuple[np.ndarray, int]],
    *,
    sfreq: float,
    basis_mode: str,
    basis_rank_r: int,
    basis_min_hz: float,
    basis_max_hz: float,
    basis_phase_count: int,
    channel_waveform_rank: int | None,
    candidate_probe_scale: float,
    max_coeff_abs: float,
    max_perturbation_peak_ratio: float | None,
    n_windows: int = 1,
    n_probe_dirs: int = 8,
    frontier_k: int | None = None,
    seed: int = 0,
) -> PopulationLeverage:
    """Estimate per-channel population leverage L_hat_c over a probe set.

    Args:
        score_fn: black-box score oracle, X -> logits (R^K).
        probe_trials: list of (x, y0) pairs; y0 is the clean predicted identity for x.
                      For Claim 1's independence assumption, draw these from DISTINCT
                      recordings (one window per recording).
        frontier_k: if given, Delta is the leverage gap between ranks frontier_k and
                    frontier_k+1 (the gap that actually governs top-k selection); else
                    the global minimum adjacent gap.
    Returns:
        PopulationLeverage with the universal channel order and the quantities (n, nu, Delta)
        needed to evaluate n* = 8 nu^2 log(2C/delta) / Delta^2.
    """
    if not probe_trials:
        raise ValueError("probe_trials must be non-empty")

    n_channels = probe_trials[0][0].shape[0]
    n_samples = probe_trials[0][0].shape[1]
    coeff_rank = (
        int(channel_waveform_rank)
        if channel_waveform_rank is not None
        else max(int(basis_rank_r * n_windows), int(basis_rank_r))
    )
    basis_matrix = _build_full_trial_basis_matrix(
        basis_mode=basis_mode,
        n_samples=n_samples,
        coeff_rank=coeff_rank,
        basis_min_hz=basis_min_hz,
        basis_max_hz=basis_max_hz,
        sfreq=sfreq,
        basis_phase_count=basis_phase_count,
    )  # (coeff_rank, T)

    probe_dirs = _make_probe_directions(coeff_rank, n_probe_dirs, seed)
    probe_abs = float(candidate_probe_scale * max_coeff_abs)

    n = len(probe_trials)
    # per-trial, per-channel leverage g_c(X_i): shape (n, C)
    g = np.zeros((n, n_channels), dtype=np.float64)
    queries = 0

    for i, (x, y0) in enumerate(probe_trials):
        base_margin = untargeted_margin(score_fn(x), int(y0))
        queries += 1
        for c in range(n_channels):
            drops = np.empty(n_probe_dirs, dtype=np.float64)
            for d in range(n_probe_dirs):
                waveform = (probe_abs * probe_dirs[d]) @ basis_matrix  # (T,)
                delta = np.zeros((n_channels, n_samples), dtype=np.float32)
                delta[c, :] = waveform.astype(np.float32)
                delta = _apply_peak_ratio_constraint(
                    x=x, delta=delta, max_perturbation_peak_ratio=max_perturbation_peak_ratio
                )
                m = untargeted_margin(score_fn(x + delta), int(y0))
                queries += 1
                drops[d] = base_margin - m
            g[i, c] = float(np.mean(drops))

    leverage = g.mean(axis=0)                       # L_hat_c
    std = g.std(axis=0, ddof=1) if n > 1 else np.zeros(n_channels)
    order = np.argsort(leverage)[::-1].astype(int)  # descending

    sorted_lev = leverage[order]
    if frontier_k is not None and 0 < frontier_k < n_channels:
        gap = float(sorted_lev[frontier_k - 1] - sorted_lev[frontier_k])
    else:
        diffs = -np.diff(sorted_lev)                # adjacent gaps (>=0)
        gap = float(diffs.min()) if diffs.size else 0.0

    return PopulationLeverage(
        order=order,
        leverage=leverage,
        std=std,
        n_probe=n,
        gap=gap,
        queries_used=queries,
    )


class PopulationOrderAttack(ChannelFirstScoreAttack):
    """Attack a trial by adding channels in a FIXED (population) order, refining waveform per trial.

    Identical waveform machinery as ChannelFirstScoreAttack; only channel selection is
    replaced by the precomputed universal order, so k_star measures how deep into the
    universal channel order a given trial needs to go before it flips.
    """

    def run_fixed_order(self, x: np.ndarray, y: int, channel_order) -> "AttackResult":  # noqa: F821
        self.queries_used = 0
        n_channels, n_samples = x.shape
        coeff_rank = self._coeff_rank()
        basis_matrix = _build_full_trial_basis_matrix(
            basis_mode=self.basis_mode,
            n_samples=n_samples,
            coeff_rank=coeff_rank,
            basis_min_hz=self.basis_min_hz,
            basis_max_hz=self.basis_max_hz,
            sfreq=self.sfreq,
            basis_phase_count=self.basis_phase_count,
        )

        support: list[int] = []
        coeffs = np.zeros((0, coeff_rank), dtype=np.float32)

        current_margin = untargeted_margin(self._query_scores(x), y)
        if current_margin < 0.0:
            return self._build_result(
                x=x, y=y, basis_matrix=basis_matrix,
                support=support, coeffs=coeffs,
                margin=current_margin, budget_exhausted=False,
            )

        order = [int(c) for c in channel_order][: self.support_budget_k]
        try:
            for channel in order:
                if channel in support:
                    continue
                support.append(channel)
                init_coeffs = np.vstack([coeffs, np.zeros((1, coeff_rank), dtype=np.float32)])
                coeffs, _ = self._refine_coeffs(
                    x=x, y=y, support=support,
                    init_coeffs=init_coeffs, basis_matrix=basis_matrix,
                )
                delta = self._assemble_delta(
                    n_channels=n_channels, n_samples=n_samples,
                    basis_matrix=basis_matrix, support=support, coeffs=coeffs,
                )
                delta = _apply_peak_ratio_constraint(
                    x=x, delta=delta, max_perturbation_peak_ratio=self.max_perturbation_peak_ratio,
                )
                current_margin = untargeted_margin(self._query_scores(x + delta), y)
                if current_margin < 0.0 and self.stop_on_success:
                    return self._build_result(
                        x=x, y=y, basis_matrix=basis_matrix,
                        support=support, coeffs=coeffs,
                        margin=current_margin, budget_exhausted=False,
                    )
        except QueryBudgetExhausted:
            return self._build_result(
                x=x, y=y, basis_matrix=basis_matrix,
                support=support, coeffs=coeffs,
                margin=current_margin, budget_exhausted=True,
            )

        return self._build_result(
            x=x, y=y, basis_matrix=basis_matrix,
            support=support, coeffs=coeffs,
            margin=current_margin, budget_exhausted=False,
        )
