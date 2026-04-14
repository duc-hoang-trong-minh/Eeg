from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter_defense(
    x: np.ndarray,
    low_hz: float = 4.0,
    high_hz: float = 40.0,
    sfreq: float = 128.0,
    order: int = 5,
) -> np.ndarray:
    """Butterworth bandpass preprocessing defense (mirrors Filter_Attack's FilterLayer).

    Retains only frequencies in [low_hz, high_hz]. Triggers living inside this
    band survive; out-of-band perturbations are suppressed.
    """
    nyq = sfreq / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def localized_denoise(x: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return x.copy()
    pad = kernel_size // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        out[:, t] = np.mean(x_pad[:, t : t + kernel_size], axis=1)
    return out


def suspicious_window_score(delta: np.ndarray, n_windows: int) -> np.ndarray:
    c, t = delta.shape
    edges = np.linspace(0, t, n_windows + 1, dtype=int)
    scores = np.zeros((c, n_windows), dtype=np.float32)
    for w in range(n_windows):
        s, e = edges[w], edges[w + 1]
        scores[:, w] = np.mean(np.abs(delta[:, s:e]), axis=1)
    return scores


def flag_suspicious_atoms(delta: np.ndarray, n_windows: int, threshold: float) -> list[tuple[int, int]]:
    scores = suspicious_window_score(delta, n_windows=n_windows)
    flagged = []
    for c in range(scores.shape[0]):
        for w in range(scores.shape[1]):
            if float(scores[c, w]) >= threshold:
                flagged.append((c, w))
    return flagged


def suspicious_residual_score(
    x: np.ndarray,
    n_windows: int,
    residual_kernel_size: int = 5,
) -> np.ndarray:
    residual = np.abs(x - localized_denoise(x, kernel_size=residual_kernel_size))
    return suspicious_window_score(residual, n_windows=n_windows)


def flag_suspicious_atoms_from_signal(
    x: np.ndarray,
    n_windows: int,
    z_threshold: float = 2.5,
    residual_kernel_size: int = 5,
) -> list[tuple[int, int]]:
    scores = suspicious_residual_score(
        x=x,
        n_windows=n_windows,
        residual_kernel_size=residual_kernel_size,
    )
    flagged = []
    for c in range(scores.shape[0]):
        row = scores[c]
        median = float(np.median(row))
        mad = float(np.median(np.abs(row - median))) + 1e-6
        robust_z = 0.6745 * (row - median) / mad
        for w, value in enumerate(robust_z):
            if float(value) >= z_threshold:
                flagged.append((c, w))
    return flagged


def suppress_flagged_atoms(
    x: np.ndarray,
    flagged_atoms: list[tuple[int, int]],
    n_windows: int,
    kernel_size: int = 5,
) -> np.ndarray:
    if not flagged_atoms:
        return x.copy()
    repaired = x.copy()
    smoothed = localized_denoise(x, kernel_size=kernel_size)
    edges = np.linspace(0, x.shape[1], n_windows + 1, dtype=int)
    for c, w in flagged_atoms:
        s, e = int(edges[w]), int(edges[w + 1])
        repaired[c, s:e] = smoothed[c, s:e]
    return repaired
