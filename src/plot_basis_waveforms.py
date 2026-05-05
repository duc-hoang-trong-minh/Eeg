from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .attack.basis import RaisedCosineBasis
    from .config import BaselineConfig, OutputConfig
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.attack.basis import RaisedCosineBasis
    from src.config import BaselineConfig, OutputConfig


def plot_basis_waveforms(
    window_length: int | None = None,
    rank: int = 5,
    f_min_hz: float = 4.0,
    f_max_hz: float = 30.0,
    sfreq: float | None = None,
) -> dict[str, str]:
    baseline_cfg = BaselineConfig()
    sfreq = float(baseline_cfg.sfreq if sfreq is None else sfreq)
    if window_length is None:
        window_length = int(round(baseline_cfg.window_size_seconds * sfreq))

    basis = RaisedCosineBasis(
        window_length=window_length,
        rank=rank,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
        sfreq=sfreq,
    ).matrix

    t = np.arange(window_length, dtype=np.float32) / sfreq
    envelope = np.hanning(window_length).astype(np.float32)
    freqs = np.linspace(f_min_hz, f_max_hz, rank, dtype=np.float32)
    phases = np.linspace(0.0, np.pi / 2.0, rank, dtype=np.float32)

    out_cfg = OutputConfig()
    out_cfg.root.mkdir(parents=True, exist_ok=True)
    out_path = out_cfg.root / "basis_waveform_examples.png"

    fig, axes = plt.subplots(rank, 1, figsize=(8.4, 1.2 * rank + 0.8), sharex=True)
    if rank == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        wave = basis[idx]
        peak = float(np.max(np.abs(wave)) + 1e-8)
        envelope_scaled = peak * envelope / float(np.max(envelope) + 1e-8)

        ax.plot(t, wave, color="#1f77b4", linewidth=2.0, label="basis atom" if idx == 0 else None)
        ax.fill_between(t, 0.0, wave, color="#1f77b4", alpha=0.12)
        ax.plot(
            t,
            envelope_scaled,
            color="#6a6a6a",
            linewidth=1.2,
            linestyle="--",
            label="window $h(t)$" if idx == 0 else None,
        )
        ax.plot(t, -envelope_scaled, color="#6a6a6a", linewidth=1.2, linestyle="--")
        ax.axhline(0.0, color="#bfbfbf", linewidth=0.8)
        ax.set_ylabel(f"$\\phi_{idx + 1}$", rotation=0, labelpad=18)
        ax.text(
            0.99,
            0.86,
            f"{freqs[idx]:.1f} Hz, phase {phases[idx] / np.pi:.2f}$\\pi$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlim(float(t[0]), float(t[-1]))

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Sample Basis Waveforms With Window $h(t)$", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"figure_path": str(out_path)}


if __name__ == "__main__":
    print(plot_basis_waveforms())
