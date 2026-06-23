from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_oscillatory_atom_taper(
    out_dir: str | Path = "Figure",
    sfreq: float = 100.0,
    window_seconds: float = 4.0,
    atom_hz: float = 10.0,
) -> dict[str, str]:
    """Plot one oscillatory atom with and without the Hanning taper."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(window_seconds * sfreq))
    pad_samples = int(round(0.25 * sfreq))
    t = np.arange(window_samples, dtype=np.float32) / sfreq
    t_ext = (np.arange(window_samples + 2 * pad_samples, dtype=np.float32) - pad_samples) / sfreq

    raw = np.cos(2.0 * np.pi * atom_hz * t)
    envelope = np.hanning(window_samples).astype(np.float32)
    tapered = envelope * raw

    raw_ext = np.zeros_like(t_ext)
    tapered_ext = np.zeros_like(t_ext)
    env_ext = np.zeros_like(t_ext)
    sl = slice(pad_samples, pad_samples + window_samples)
    raw_ext[sl] = raw
    tapered_ext[sl] = tapered
    env_ext[sl] = envelope

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 3.9), sharex=True)
    panels = [
        (raw_ext, None, "Without taper: abrupt finite-window edges", "#b24c3f"),
        (tapered_ext, env_ext, "With Hanning taper: smooth entry and exit", "#1f6f8b"),
    ]
    for ax, (wave, env, title, color) in zip(axes, panels):
        ax.plot(t_ext, wave, color=color, linewidth=1.7)
        ax.fill_between(t_ext, 0.0, wave, color=color, alpha=0.16)
        if env is not None:
            ax.plot(t_ext, env, color="#555555", linestyle="--", linewidth=1.0, label="Hanning envelope")
            ax.plot(t_ext, -env, color="#555555", linestyle="--", linewidth=1.0)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.axhline(0.0, color="#b8b8b8", linewidth=0.8)
        ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle=":")
        ax.axvline(window_seconds, color="#777777", linewidth=0.8, linestyle=":")
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.18)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(float(t_ext[0]), float(t_ext[-1]))
    fig.suptitle(f"Example {atom_hz:.0f} Hz Oscillatory Atom in a {window_seconds:.0f} s EEG Window", fontsize=12)
    fig.tight_layout()

    pdf_path = out_dir / "oscillatory_atom_taper_example.pdf"
    png_path = out_dir / "oscillatory_atom_taper_example.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {"pdf_path": str(pdf_path), "png_path": str(png_path)}


if __name__ == "__main__":
    print(plot_oscillatory_atom_taper())
