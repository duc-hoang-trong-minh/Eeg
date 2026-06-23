from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATH = Path("outputs/hbn_subject_sweep/hbn_subject_count_channel_control_summary.json")
OUT_DIR = Path("outputs/paper_figures")


def _load_rows(summary_path: Path = SUMMARY_PATH) -> list[dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    rows.sort(key=lambda row: int(row["subject_count"]))
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def plot_scaling_dashboard(rows: list[dict]) -> dict[str, str]:
    _style()
    subjects = np.asarray([row["subject_count"] for row in rows], dtype=float)
    acc = np.asarray([100.0 * row["best_val_acc"] for row in rows], dtype=float)
    asr = np.asarray([100.0 * row["attack_success_rate"] for row in rows], dtype=float)
    mean_k = np.asarray([row["mean_first_success_k"] for row in rows], dtype=float)
    median_k = np.asarray([row["median_first_success_k"] for row in rows], dtype=float)
    k95 = np.asarray([row["k95"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(subjects, acc, marker="o", linewidth=2.0, color="#1f77b4", label="Accuracy")
    ax.set_title("(a) Clean recognizer")
    ax.set_xlabel("Enrolled subjects")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(72, 100)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(subjects, asr, marker="D", linewidth=2.0, color="#2ca02c")
    ax.fill_between(subjects, 95, 100, color="#2ca02c", alpha=0.10, linewidth=0)
    ax.set_title("(b) Attack reliability")
    ax.set_xlabel("Enrolled subjects")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(90, 101)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2]
    ax.plot(subjects, mean_k, marker="o", linewidth=2.0, color="#1f77b4", label="Mean")
    ax.plot(subjects, median_k, marker="s", linewidth=1.8, color="#9467bd", label="Median")
    ax.plot(subjects, k95, marker="^", linewidth=1.8, color="#ff7f0e", label="K95")
    ax.set_title("(c) Channel-control cost")
    ax.set_xlabel("Enrolled subjects")
    ax.set_ylabel("Channels")
    ax.set_ylim(0, 13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.set_xticks(subjects)

    return _save(fig, "hbn_scaling_dashboard")


def _first_success_array(row: dict, max_k: int = 12) -> np.ndarray:
    counts = row.get("first_success_counts", {})
    values = np.zeros(max_k, dtype=float)
    for key, value in counts.items():
        k = int(key)
        if 1 <= k <= max_k:
            values[k - 1] = float(value)
    return values


def plot_prefix_success(rows: list[dict], max_k: int = 12) -> dict[str, str]:
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))
    ks = np.arange(1, max_k + 1)

    for color, row in zip(colors, rows):
        counts = _first_success_array(row, max_k=max_k)
        n_attacked = float(row["n_attack_samples"])
        cumulative = 100.0 * np.cumsum(counts) / n_attacked
        ax.plot(
            ks,
            cumulative,
            marker="o",
            markersize=4,
            linewidth=1.7,
            color=color,
            label=f"{row['subject_count']} IDs",
        )

    ax.axhline(95, color="#d62728", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.text(12.15, 95, "95%", va="center", fontsize=8, color="#d62728")
    ax.set_xlabel("Controlled channels K")
    ax.set_ylabel("Cumulative attack success (%)")
    ax.set_title("Prefix Success: Most Trials Flip Before the 12-Channel Budget")
    ax.set_xticks(ks)
    ax.set_ylim(0, 104)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False, loc="lower right")
    return _save(fig, "hbn_prefix_success_curves")


def plot_first_success_heatmap(rows: list[dict], max_k: int = 12) -> dict[str, str]:
    _style()
    matrix = []
    labels = []
    for row in rows:
        counts = _first_success_array(row, max_k=max_k)
        n_attacked = float(row["n_attack_samples"])
        matrix.append(100.0 * counts / n_attacked)
        labels.append(str(row["subject_count"]))
    arr = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1.0, float(arr.max())))
    ax.set_xlabel("First-success channel count")
    ax.set_ylabel("Enrolled subjects")
    ax.set_title("Where the First Successful Control Occurs")
    ax.set_xticks(np.arange(max_k))
    ax.set_xticklabels([str(k) for k in range(1, max_k + 1)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if value >= 4.0:
                ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Attacked trials (%)")
    return _save(fig, "hbn_first_success_heatmap")


def main() -> None:
    rows = _load_rows()
    outputs = {
        "scaling_dashboard": plot_scaling_dashboard(rows),
        "prefix_success": plot_prefix_success(rows),
        "first_success_heatmap": plot_first_success_heatmap(rows),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
