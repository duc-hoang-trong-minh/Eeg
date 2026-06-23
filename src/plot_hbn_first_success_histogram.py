from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REPORT_PATH = Path("outputs/hbn_r1_l100_human_recognition/human_recognition_attack_basis_comparison_report.json")
DEFAULT_OUT_PATH = Path("outputs/hbn_r1_l100_human_recognition/hbn_k12_first_success_channel_histogram.png")
DEFAULT_SUMMARY_PATH = Path("outputs/hbn_r1_l100_human_recognition/hbn_k12_first_success_channel_summary.json")


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _load_sparse_variant(report: dict, variant_name: str) -> dict:
    variants = report.get("variants", [])
    for variant in variants:
        if variant.get("variant_name") == variant_name:
            return variant
    if len(variants) == 1:
        return variants[0]
    raise ValueError(f"No variant named {variant_name!r} found in report")


def summarize_first_success(report_path: Path, variant_name: str) -> dict:
    report = json.loads(report_path.read_text())
    variant = _load_sparse_variant(report, variant_name)
    rows = variant.get("per_sample", [])
    first_success_ks = [int(row["first_success_k"]) for row in rows if row.get("first_success_k") is not None]
    failed_indices = [int(row["idx"]) for row in rows if row.get("first_success_k") is None]
    counts = Counter(first_success_ks)
    attack_config = variant.get("attack_config", {})
    support_budget = int(attack_config.get("support_budget_k") or (max(counts) if counts else 0))

    return {
        "source_report_path": str(report_path),
        "dataset_name": report.get("dataset_name"),
        "model_name": report.get("model_name"),
        "target_mode": report.get("target_mode"),
        "evaluation_protocol": report.get("evaluation_protocol"),
        "n_clean_correct_total": report.get("n_clean_correct_total"),
        "n_clean_correct_attacked": report.get("n_clean_correct_attacked"),
        "max_samples": report.get("max_samples"),
        "score_device": report.get("score_device"),
        "worker_device": report.get("worker_device"),
        "n_workers": report.get("n_workers"),
        "variant_name": variant.get("variant_name"),
        "attack_config": {
            key: attack_config.get(key)
            for key in [
                "support_mode",
                "support_budget_k",
                "max_outer_iters",
                "max_query_budget",
                "max_perturbation_peak_ratio",
                "max_coeff_abs",
                "l2_weight",
                "tv_weight",
                "band_weight",
            ]
        },
        "attack_success_rate": variant.get("attack_success_rate"),
        "attacked_accuracy": variant.get("attacked_accuracy"),
        "first_success_channel_summary": {
            "n_success": len(first_success_ks),
            "n_missing": len(failed_indices),
            "mean": None if not first_success_ks else float(st.mean(first_success_ks)),
            "median": None if not first_success_ks else float(st.median(first_success_ks)),
            "min": None if not first_success_ks else int(min(first_success_ks)),
            "max": None if not first_success_ks else int(max(first_success_ks)),
            "counts": {str(k): int(counts[k]) for k in range(1, support_budget + 1) if counts.get(k, 0) > 0},
            "missing_sample_indices": failed_indices,
        },
        "prefix_success_rate": [
            {
                "k": int(row["k"]),
                "success_rate": float(row["success_rate"]),
                "attacked_accuracy": float(row["attacked_accuracy"]),
            }
            for row in variant.get("prefix_summary", [])
        ],
    }


def plot_histogram(summary: dict, out_path: Path) -> None:
    _configure_plot_style()
    channel_summary = summary["first_success_channel_summary"]
    counts = {int(k): int(v) for k, v in channel_summary["counts"].items()}
    support_budget = int(summary["attack_config"]["support_budget_k"] or max(counts, default=1))
    xs = list(range(1, support_budget + 1))
    ys = [counts.get(k, 0) for k in xs]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars = ax.bar(xs, ys, width=0.76, color="#2f6f9f", alpha=0.9)
    for bar, count in zip(bars, ys):
        if count:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.35,
                str(count),
                ha="center",
                va="bottom",
                fontsize=11,
            )

    n = int(summary["n_clean_correct_attacked"])
    mean = channel_summary["mean"]
    median = channel_summary["median"]
    success_pct = 100.0 * float(summary["attack_success_rate"])
    ax.set_title(f"HBN K={support_budget} First-Success Channel Histogram (n={n})")
    ax.set_xlabel("Channels at First Successful Attack")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(xs)
    ax.set_ylim(top=max(ys + [1]) * 1.22)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.96,
        f"ASR {success_pct:.1f}%\nmean {mean:.2f}\nmedian {median:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.95},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HBN first-success channel-count histogram from an attack report.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--variant", default="human_sparse_channel_hybrid")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    summary = summarize_first_success(args.report, args.variant)
    plot_histogram(summary, args.out)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    summary["histogram_path"] = str(args.out)
    summary["summary_path"] = str(args.summary_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
