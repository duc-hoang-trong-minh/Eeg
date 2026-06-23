from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCHIVE_DIR = Path("outputs/hbn_r1_l100_human_recognition/scaleup_archive")


REPORTS = [
    {
        "label": "Plain 10% cap",
        "path": ARCHIVE_DIR / "diag10_human_recognition_attack_basis_comparison_report.json",
        "color": "#7f7f7f",
        "marker": "o",
    },
    {
        "label": "Tuned K=8",
        "path": ARCHIVE_DIR / "aggr10_max64_sparse_human_recognition_attack_basis_comparison_report.json",
        "color": "#1f77b4",
        "marker": "s",
    },
    {
        "label": "Tuned+ K=10",
        "path": ARCHIVE_DIR / "aggr10plus_max64_sparse_human_recognition_attack_basis_comparison_report.json",
        "color": "#d62728",
        "marker": "^",
    },
    {
        "label": "Tuned+ K=12",
        "path": ARCHIVE_DIR / "aggr10plus_k12_max64_sparse_human_recognition_attack_basis_comparison_report.json",
        "color": "#9467bd",
        "marker": "D",
    },
]


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
        }
    )


def _load_sparse_variant(path: Path) -> dict:
    report = json.loads(path.read_text())
    variants = report.get("variants", [])
    for variant in variants:
        if variant.get("variant_name") == "human_sparse_channel_hybrid":
            return variant
    if len(variants) == 1:
        return variants[0]
    raise ValueError(f"No sparse-channel variant found in {path}")


def _power_ratio_percent(signal_l2: float, delta_l2: float) -> float:
    return 100.0 * (float(delta_l2) ** 2) / max(float(signal_l2) ** 2, 1e-12)


def _delivered_power(row: dict) -> float | None:
    if row.get("delivered_delta_power_ratio_pct") is not None:
        return float(row["delivered_delta_power_ratio_pct"])
    if row.get("first_success_min_l2") is None:
        return None
    return _power_ratio_percent(row["signal_l2"], row["first_success_min_l2"])


def _adaptive_channel_summary(variant: dict) -> dict:
    if variant.get("adaptive_channel_summary"):
        return variant["adaptive_channel_summary"]
    ks = [int(row["first_success_k"]) for row in variant.get("per_sample", []) if row.get("first_success_k") is not None]
    return {
        "mean": None if not ks else float(st.mean(ks)),
        "median": None if not ks else float(st.median(ks)),
        "max": None if not ks else int(max(ks)),
        "counts": {str(k): int(v) for k, v in sorted(Counter(ks).items())},
    }


def _delivered_power_summary(variant: dict) -> dict:
    if variant.get("delivered_power_ratio_pct_summary"):
        return variant["delivered_power_ratio_pct_summary"]
    powers = [power for row in variant.get("per_sample", []) if (power := _delivered_power(row)) is not None]
    return {
        "mean": None if not powers else float(st.mean(powers)),
        "median": None if not powers else float(st.median(powers)),
        "max": None if not powers else float(max(powers)),
    }


def _summarize_variant(entry: dict, variant: dict) -> dict:
    prefix_rows = variant["prefix_summary"]
    return {
        "label": entry["label"],
        "report_path": str(entry["path"]),
        "attack_config": {
            "support_budget_k": variant["attack_config"]["support_budget_k"],
            "max_outer_iters": variant["attack_config"]["max_outer_iters"],
            "max_query_budget": variant["attack_config"]["max_query_budget"],
            "max_perturbation_peak_ratio": variant["attack_config"]["max_perturbation_peak_ratio"],
            "max_coeff_abs": variant["attack_config"]["max_coeff_abs"],
            "l2_weight": variant["attack_config"]["l2_weight"],
            "tv_weight": variant["attack_config"]["tv_weight"],
            "band_weight": variant["attack_config"]["band_weight"],
        },
        "n_clean_correct_attacked": variant["n_clean_correct_attacked"],
        "attack_success_rate": variant["attack_success_rate"],
        "attacked_accuracy": variant["attacked_accuracy"],
        "adaptive_channel_summary": _adaptive_channel_summary(variant),
        "delivered_power_ratio_pct_summary": _delivered_power_summary(variant),
        "prefix": [
            {
                "k": row["k"],
                "attack_success_rate": row["success_rate"],
                "attacked_accuracy": row["attacked_accuracy"],
                "raw_avg_delta_power_ratio_pct": row["avg_delta_power_ratio_pct"],
                "delivered_avg_power_ratio_pct_success_only": row["avg_min_power_ratio_pct_success_only"],
                "avg_min_power_ratio_pct_success_only": row["avg_min_power_ratio_pct_success_only"],
            }
            for row in prefix_rows
        ],
    }


def _plot_accuracy(series: list[dict], out_path: Path) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for item in series:
        prefix = item["summary"]["prefix"]
        ks = [row["k"] for row in prefix]
        attacked_accuracy_pct = [100.0 * row["attacked_accuracy"] for row in prefix]
        ax.plot(
            ks,
            attacked_accuracy_pct,
            marker=item["marker"],
            markersize=7.5,
            linewidth=2.4,
            color=item["color"],
            label=(
                f"{item['label']} "
                f"(n={item['summary']['n_clean_correct_attacked']}, "
                f"{100.0 * item['summary']['attacked_accuracy']:.1f}% final)"
            ),
        )

    max_k = max(row["k"] for item in series for row in item["summary"]["prefix"])
    ax.set_xlabel("Sparsity Budget K")
    ax.set_ylabel("Attacked Accuracy (%)")
    ax.set_title("HBN EEGConformer Under Sparse Attacks")
    ax.set_xticks(list(range(1, max_k + 1)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_power(series: list[dict], out_path: Path) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for item in series:
        prefix = item["summary"]["prefix"]
        ks = [row["k"] for row in prefix]
        power = [
            float("nan")
            if row["delivered_avg_power_ratio_pct_success_only"] is None
            else row["delivered_avg_power_ratio_pct_success_only"]
            for row in prefix
        ]
        ax.plot(
            ks,
            power,
            marker=item["marker"],
            markersize=7.5,
            linewidth=2.4,
            color=item["color"],
            label=item["label"],
        )

    max_k = max(row["k"] for item in series for row in item["summary"]["prefix"])
    ax.set_xlabel("Sparsity Budget K")
    ax.set_ylabel("Delivered Power (% of Signal Power)")
    ax.set_title("Min-Scaled Delivered Power for HBN Sparse Attacks")
    ax.set_xticks(list(range(1, max_k + 1)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_adaptive_channels(series: list[dict], out_path: Path) -> None:
    _configure_plot_style()
    k12 = next((item for item in reversed(series) if item["summary"]["label"] == "Tuned+ K=12"), series[-1])
    counts = {
        int(k): int(v)
        for k, v in k12["summary"]["adaptive_channel_summary"]["counts"].items()
    }
    max_k = max(counts) if counts else 1
    xs = list(range(1, max_k + 1))
    ys = [counts.get(k, 0) for k in xs]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(xs, ys, color="#9467bd", alpha=0.88)
    ax.set_xlabel("Delivered Channels at First Success")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Adaptive Channel Use for HBN K=12 Attack")
    ax.set_xticks(xs)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hbn_attack_sparsity_tuning() -> dict:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    series = []
    for entry in REPORTS:
        variant = _load_sparse_variant(entry["path"])
        item = dict(entry)
        item["summary"] = _summarize_variant(entry, variant)
        series.append(item)

    accuracy_path = ARCHIVE_DIR / "hbn_attack_accuracy_vs_sparsity_tuning.png"
    power_path = ARCHIVE_DIR / "hbn_attack_power_vs_sparsity_tuning.png"
    adaptive_channels_path = ARCHIVE_DIR / "hbn_attack_adaptive_channels_k12.png"
    _plot_accuracy(series, accuracy_path)
    _plot_power(series, power_path)
    _plot_adaptive_channels(series, adaptive_channels_path)

    summary = {
        "accuracy_plot_path": str(accuracy_path),
        "power_plot_path": str(power_path),
        "adaptive_channels_plot_path": str(adaptive_channels_path),
        "series": [item["summary"] for item in series],
    }
    summary_path = ARCHIVE_DIR / "hbn_attack_sparsity_tuning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    print(json.dumps(plot_hbn_attack_sparsity_tuning(), indent=2))
