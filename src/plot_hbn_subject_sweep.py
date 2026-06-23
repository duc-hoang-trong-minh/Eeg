from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_ROOT = Path("outputs/hbn_subject_sweep")
DEFAULT_SOURCE_URL = "https://neuromechanist.github.io/data/hbn/"


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("subject_recognition_metrics.json")})


def _expand_counts(counts: dict[str, int]) -> list[int]:
    values = []
    for key, value in counts.items():
        values.extend([int(key)] * int(value))
    return sorted(values)


def _find_sparse_variant(report: dict) -> dict | None:
    variants = report.get("variants", [])
    for variant in variants:
        if variant.get("variant_name") == "human_sparse_channel_hybrid":
            return variant
    if len(variants) == 1:
        return variants[0]
    return None


def _first_success_from_report(report_path: Path) -> tuple[list[int], list[dict]]:
    report = _read_json(report_path)
    variant = _find_sparse_variant(report)
    if variant is None:
        return [], []
    rows = variant.get("per_sample", [])
    first_success = [int(row["first_success_k"]) for row in rows if row.get("first_success_k") is not None]
    return first_success, variant.get("prefix_summary", [])


def _first_success_from_summary(summary_path: Path) -> tuple[list[int], list[dict]]:
    summary = _read_json(summary_path)
    channel_summary = summary.get("first_success_channel_summary", {})
    counts = channel_summary.get("counts", {})
    first_success = _expand_counts(counts)
    prefix = summary.get("prefix_success_rate", [])
    return first_success, prefix


def _bootstrap_ci(values: list[int], seed: int = 7, n_bootstrap: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = float(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _k_for_success(prefix: list[dict], target: float) -> int | None:
    for row in sorted(prefix, key=lambda item: int(item["k"])):
        success = row.get("success_rate")
        if success is None:
            success = row.get("attack_success_rate")
        if success is not None and float(success) >= target:
            return int(row["k"])
    return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "subject_recognition_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = _read_json(metrics_path)
    if metrics.get("target_mode") != "subject":
        return None
    split = metrics.get("split_summary", {})
    if split.get("dataset_name") != "HBN":
        return None

    model_name = str(metrics.get("model_name", "unknown"))
    selected_subjects = int(split["selected_subjects"])
    n_common_channels = split.get("n_common_eeg_channels")
    best_val_acc = _to_float(metrics.get("best_val_acc"))

    biometric_path = run_dir / "subject_recognition_biometric_metrics.json"
    biometric = _read_json(biometric_path) if biometric_path.exists() else {}

    summary_path = run_dir / "hbn_k12_first_success_channel_summary.json"
    report_path = run_dir / "human_recognition_attack_basis_comparison_report.json"
    if summary_path.exists():
        first_success, prefix = _first_success_from_summary(summary_path)
        attack_summary = _read_json(summary_path)
        attack_success_rate = _to_float(attack_summary.get("attack_success_rate"))
        attacked_accuracy = _to_float(attack_summary.get("attacked_accuracy"))
        n_attacked = attack_summary.get("n_clean_correct_attacked")
    elif report_path.exists():
        first_success, prefix = _first_success_from_report(report_path)
        report = _read_json(report_path)
        variant = _find_sparse_variant(report)
        attack_success_rate = None if variant is None else _to_float(variant.get("attack_success_rate"))
        attacked_accuracy = None if variant is None else _to_float(variant.get("attacked_accuracy"))
        n_attacked = report.get("n_clean_correct_attacked")
    else:
        first_success, prefix = [], []
        attack_success_rate = None
        attacked_accuracy = None
        n_attacked = None

    mean_ci_low, mean_ci_high = _bootstrap_ci(first_success, seed=selected_subjects)
    row = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "subject_count": selected_subjects,
        "n_common_eeg_channels": None if n_common_channels is None else int(n_common_channels),
        "best_val_acc": best_val_acc,
        "pooled_eer_pct": _to_float(biometric.get("pooled_eer_pct")),
        "macro_subject_eer_pct": _to_float(biometric.get("macro_subject_eer_pct")),
        "n_attack_samples": None if n_attacked is None else int(n_attacked),
        "attack_success_rate": attack_success_rate,
        "attacked_accuracy": attacked_accuracy,
        "mean_first_success_k": None if not first_success else float(st.mean(first_success)),
        "mean_first_success_k_ci_low": mean_ci_low,
        "mean_first_success_k_ci_high": mean_ci_high,
        "median_first_success_k": None if not first_success else float(st.median(first_success)),
        "max_first_success_k": None if not first_success else int(max(first_success)),
        "k90": _k_for_success(prefix, 0.90),
        "k95": _k_for_success(prefix, 0.95),
        "k100": _k_for_success(prefix, 1.00),
        "first_success_counts": {str(k): int(first_success.count(k)) for k in sorted(set(first_success))},
    }
    return row


def _spearman(rows: list[dict], y_key: str) -> dict[str, float | None]:
    pairs = [
        (float(row["subject_count"]), float(row[y_key]))
        for row in rows
        if row.get(y_key) is not None
    ]
    if len(pairs) < 2:
        return {"rho": None, "pvalue": None, "n": len(pairs)}
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"rho": None, "pvalue": None, "n": len(pairs)}
    result = stats.spearmanr(xs, ys)
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "n": len(pairs)}


def _correlations(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    return {
        "mean_first_success_k": _spearman(rows, "mean_first_success_k"),
        "median_first_success_k": _spearman(rows, "median_first_success_k"),
        "k90": _spearman(rows, "k90"),
        "k95": _spearman(rows, "k95"),
        "k100": _spearman(rows, "k100"),
    }


def _correlations_by_model(rows: list[dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_name"]), []).append(row)
    return {model_name: _correlations(model_rows) for model_name, model_rows in sorted(grouped.items())}


def _write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "model_name",
        "subject_count",
        "n_common_eeg_channels",
        "best_val_acc",
        "pooled_eer_pct",
        "macro_subject_eer_pct",
        "n_attack_samples",
        "attack_success_rate",
        "attacked_accuracy",
        "mean_first_success_k",
        "mean_first_success_k_ci_low",
        "mean_first_success_k_ci_high",
        "median_first_success_k",
        "max_first_success_k",
        "k90",
        "k95",
        "k100",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot_channel_control(rows: list[dict], out_path: Path) -> None:
    _configure_plot_style()
    rows = [row for row in rows if row.get("mean_first_success_k") is not None]
    if not rows:
        raise ValueError("No completed attack summaries with first-success channels were found")

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    model_names = sorted({str(row["model_name"]) for row in rows})
    if len(model_names) == 1:
        xs = np.asarray([row["subject_count"] for row in rows], dtype=np.float64)
        mean = np.asarray([row["mean_first_success_k"] for row in rows], dtype=np.float64)
        ci_low = np.asarray([row["mean_first_success_k_ci_low"] for row in rows], dtype=np.float64)
        ci_high = np.asarray([row["mean_first_success_k_ci_high"] for row in rows], dtype=np.float64)
        yerr = np.vstack([mean - ci_low, ci_high - mean])
        ax.errorbar(
            xs,
            mean,
            yerr=yerr,
            marker="o",
            markersize=7,
            linewidth=2.4,
            capsize=4,
            color="#1f77b4",
            label="Mean first-success K",
        )

        optional_series = [
            ("median_first_success_k", "Median first-success K", "#2ca02c", "s"),
            ("k90", "K90", "#9467bd", "^"),
            ("k95", "K95", "#d62728", "D"),
            ("k100", "K100", "#7f7f7f", "v"),
        ]
        for key, label, color, marker in optional_series:
            series_rows = [row for row in rows if row.get(key) is not None]
            if not series_rows:
                continue
            ax.plot(
                [row["subject_count"] for row in series_rows],
                [row[key] for row in series_rows],
                marker=marker,
                linewidth=1.8,
                color=color,
                label=label,
            )
    else:
        colors = {
            "EEGConformer": "#1f77b4",
            "EEGNet": "#2ca02c",
            "ShallowFBCSPNet": "#d62728",
        }
        markers = {
            "EEGConformer": "o",
            "EEGNet": "s",
            "ShallowFBCSPNet": "^",
        }
        for model_name in model_names:
            model_rows = [row for row in rows if row["model_name"] == model_name and row.get("mean_first_success_k") is not None]
            model_rows.sort(key=lambda row: row["subject_count"])
            xs = np.asarray([row["subject_count"] for row in model_rows], dtype=np.float64)
            mean = np.asarray([row["mean_first_success_k"] for row in model_rows], dtype=np.float64)
            ci_low = np.asarray([row["mean_first_success_k_ci_low"] for row in model_rows], dtype=np.float64)
            ci_high = np.asarray([row["mean_first_success_k_ci_high"] for row in model_rows], dtype=np.float64)
            yerr = np.vstack([mean - ci_low, ci_high - mean])
            ax.errorbar(
                xs,
                mean,
                yerr=yerr,
                marker=markers.get(model_name, "o"),
                markersize=7,
                linewidth=2.2,
                capsize=4,
                color=colors.get(model_name),
                label=model_name,
            )

    ax.set_xlabel("HBN R1 Subject Count")
    ax.set_ylabel("Channels Needed for Control")
    ax.set_title("Subject-Set Size vs Sparse-Channel Control")
    ax.set_xticks(sorted({row["subject_count"] for row in rows}))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_hbn_subject_sweep(root: Path = DEFAULT_ROOT, out_dir: Path | None = None, model_names: list[str] | None = None) -> dict:
    out_dir = root if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in _run_dirs(root):
        row = _summarize_run(run_dir)
        if row is not None:
            rows.append(row)
    if model_names:
        wanted = set(model_names)
        rows = [row for row in rows if row["model_name"] in wanted]
    rows.sort(key=lambda row: (row["model_name"], row["subject_count"]))

    csv_path = out_dir / "hbn_subject_count_channel_control_summary.csv"
    json_path = out_dir / "hbn_subject_count_channel_control_summary.json"
    plot_path = out_dir / "hbn_subject_count_channel_control.png"

    _write_csv(rows, csv_path)
    if any(row.get("mean_first_success_k") is not None for row in rows):
        _plot_channel_control(rows, plot_path)

    summary = {
        "source_url": DEFAULT_SOURCE_URL,
        "root": str(root),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path) if plot_path.exists() else None,
        "summary_path": str(json_path),
        "n_completed_rows": len(rows),
        "model_names": sorted({row["model_name"] for row in rows}),
        "correlations": _correlations(rows),
        "correlations_by_model": _correlations_by_model(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and plot the HBN subject-count attack sweep.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-names", nargs="*", default=None)
    args = parser.parse_args()
    print(json.dumps(plot_hbn_subject_sweep(root=args.root, out_dir=args.out_dir, model_names=args.model_names), indent=2))


if __name__ == "__main__":
    main()
