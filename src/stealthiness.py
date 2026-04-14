from __future__ import annotations

import numpy as np


def covariance_frob_distance(x_clean: np.ndarray, x_adv: np.ndarray) -> float:
    """Frobenius distance between channel covariance matrices.

    Filter_Attack's spatial remix rotates the full covariance matrix — this will
    be large. A channel-sparse waveform perturbation shifts far fewer off-diagonal
    entries, so this score should be much lower for our attack.
    """
    cov_clean = np.cov(x_clean)  # (C, C)
    cov_adv = np.cov(x_adv)     # (C, C)
    return float(np.linalg.norm(cov_adv - cov_clean, ord="fro"))


def channel_sparsity(delta: np.ndarray, threshold: float = 1e-4) -> float:
    """Fraction of channels with meaningful perturbation.

    max|delta[c, :]| > threshold counts channel c as touched.
    Ours touches k channels; Filter_Attack touches all C.
    """
    n_channels = delta.shape[0]
    touched = int(np.sum(np.max(np.abs(delta), axis=-1) > threshold))
    return float(touched) / float(n_channels)


def psd_deviation(x_clean: np.ndarray, x_adv: np.ndarray, sfreq: float) -> float:
    """Mean absolute deviation of per-channel power spectral density (dB).

    A spatially-global filter changes the PSD on every channel.
    A band-constrained waveform perturbation only injects energy in [f_min, f_max]
    on a few channels, so the channel-average PSD deviation should be smaller.
    """
    psd_clean = 10.0 * np.log10(np.abs(np.fft.rfft(x_clean, axis=-1)) ** 2 + 1e-12)
    psd_adv   = 10.0 * np.log10(np.abs(np.fft.rfft(x_adv,   axis=-1)) ** 2 + 1e-12)
    return float(np.mean(np.abs(psd_adv - psd_clean)))
