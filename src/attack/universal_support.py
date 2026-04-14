"""Universal support discovery and delta projection utilities.

Given a set of sample-wise attack results (.npz files from collect_attack_patterns),
this module:
  1. Counts how frequently each (channel, window) atom was selected across attacks
  2. Returns the top-K most common atoms as the universal support
  3. Projects any delta (C, T) onto those atoms' basis vectors to produce
     target coefficients for generator training
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .basis import build_basis_matrix
from .support import make_window_partition


def discover_universal_support(
    pattern_paths: list[str | Path],
    n_channels: int,
    n_windows: int,
    support_budget_k: int,
    successful_only: bool = True,
) -> list[tuple[int, int]]:
    """Return the top-K (channel, window) atoms by selection frequency.

    Args:
        pattern_paths: paths to .npz files from collect_attack_patterns.py
        n_channels: C (e.g. 22)
        n_windows: number of windows used during attack (e.g. 8)
        support_budget_k: K atoms to return
        successful_only: if True, only count atoms from successful attacks
    """
    counts = np.zeros((n_channels, n_windows), dtype=np.int64)

    for path in pattern_paths:
        data = np.load(str(path))
        supports = data["supports"]   # (N, K_max) with -1 padding
        success = data["success"]     # (N,)

        mask = success if successful_only else np.ones(len(success), dtype=bool)
        for atom_indices in supports[mask]:
            for flat_idx in atom_indices:
                if flat_idx < 0:
                    continue
                c = int(flat_idx) // n_windows
                w = int(flat_idx) % n_windows
                if 0 <= c < n_channels and 0 <= w < n_windows:
                    counts[c, w] += 1

    # flatten and sort
    flat_counts = counts.reshape(-1)
    top_k_flat = np.argsort(flat_counts)[::-1][:support_budget_k]
    universal_support = []
    for flat_idx in top_k_flat:
        c = int(flat_idx) // n_windows
        w = int(flat_idx) % n_windows
        universal_support.append((c, w))

    return universal_support


def save_universal_support(
    universal_support: list[tuple[int, int]],
    out_path: str | Path,
    meta: dict | None = None,
) -> None:
    payload = {"universal_support": universal_support, "meta": meta or {}}
    with Path(out_path).open("w") as f:
        json.dump(payload, f, indent=2)


def load_universal_support(path: str | Path) -> list[tuple[int, int]]:
    with Path(path).open() as f:
        payload = json.load(f)
    return [tuple(atom) for atom in payload["universal_support"]]


def project_delta_onto_support(
    delta: np.ndarray,
    universal_support: list[tuple[int, int]],
    n_samples: int,
    basis_rank_r: int,
    basis_min_hz: float,
    basis_max_hz: float,
    basis_mode: str,
    basis_phase_count: int,
    sfreq: float,
    n_windows: int,
) -> np.ndarray:
    """Project a delta (C, T) onto the basis vectors of each universal support atom.

    Returns coefficients of shape (K, r) where K = len(universal_support).
    Uses least-squares projection: coeffs[k] = B_w^+ @ delta[c, s:e]
    where B_w^+ is the pseudo-inverse of the basis matrix for window w.
    """
    partition = make_window_partition(n_samples, n_windows)

    # pre-build basis per unique window
    window_ids = sorted({w for _, w in universal_support})
    basis_by_window: dict[int, np.ndarray] = {}
    for w in window_ids:
        s, e = partition.boundaries[w]
        basis_by_window[w] = build_basis_matrix(
            basis_mode=basis_mode,
            window_length=e - s,
            rank=basis_rank_r,
            f_min_hz=basis_min_hz,
            f_max_hz=basis_max_hz,
            sfreq=sfreq,
            phase_count=basis_phase_count,
        )  # shape (r, window_len)

    coeffs = np.zeros((len(universal_support), basis_rank_r), dtype=np.float32)
    for k, (c, w) in enumerate(universal_support):
        s, e = partition.boundaries[w]
        B = basis_by_window[w]          # (r, window_len)
        segment = delta[c, s:e]        # (window_len,)
        # least-squares: coeffs = (B B^T)^{-1} B @ segment
        coeffs[k] = (np.linalg.lstsq(B.T, segment, rcond=None)[0]).astype(np.float32)

    return coeffs


def build_target_coeffs_dataset(
    pattern_paths: list[str | Path],
    universal_support: list[tuple[int, int]],
    n_samples: int,
    basis_rank_r: int,
    basis_min_hz: float,
    basis_max_hz: float,
    basis_mode: str,
    basis_phase_count: int,
    sfreq: float,
    n_windows: int,
    successful_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all .npz pattern files and return (X, target_coeffs) arrays.

    X:             (N_total, C, T)
    target_coeffs: (N_total, K, r)
    """
    all_X, all_coeffs = [], []

    for path in pattern_paths:
        data = np.load(str(path))
        X = data["X"]          # (N, C, T)
        Delta = data["Delta"]  # (N, C, T)
        success = data["success"]

        mask = success if successful_only else np.ones(len(success), dtype=bool)
        for i in np.where(mask)[0]:
            x = X[i]
            delta = Delta[i]
            coeffs = project_delta_onto_support(
                delta=delta,
                universal_support=universal_support,
                n_samples=n_samples,
                basis_rank_r=basis_rank_r,
                basis_min_hz=basis_min_hz,
                basis_max_hz=basis_max_hz,
                basis_mode=basis_mode,
                basis_phase_count=basis_phase_count,
                sfreq=sfreq,
                n_windows=n_windows,
            )
            all_X.append(x)
            all_coeffs.append(coeffs)

    if not all_X:
        raise ValueError("No successful attacks found in the provided pattern files.")

    return np.stack(all_X, axis=0), np.stack(all_coeffs, axis=0)
