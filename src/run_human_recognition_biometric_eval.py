from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .human_recognition_config import build_bnci2014_001_human_recognition_output_config
    from .config import OutputConfig
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.human_recognition_config import build_bnci2014_001_human_recognition_output_config
    from src.config import OutputConfig


def _build_threshold_grid(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.sort(np.unique(scores.astype(np.float64, copy=False)))
    if unique_scores.size == 0:
        raise ValueError("Cannot build a biometric threshold grid from empty scores")
    if unique_scores.size == 1:
        value = float(unique_scores[0])
        eps = max(1e-9, np.finfo(np.float64).eps * max(1.0, abs(value)))
        return np.asarray([value - eps, value, value + eps], dtype=np.float64)

    eps = max(1e-9, np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(unique_scores)))))
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    return np.concatenate(([unique_scores[0] - eps], midpoints, [unique_scores[-1] + eps]))


def _compute_far_frr(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> dict[str, np.ndarray | float]:
    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError("Both genuine and impostor score sets must be non-empty")

    thresholds = _build_threshold_grid(np.concatenate([genuine_scores, impostor_scores]))
    genuine_sorted = np.sort(genuine_scores.astype(np.float64, copy=False))
    impostor_sorted = np.sort(impostor_scores.astype(np.float64, copy=False))

    far = 1.0 - (
        np.searchsorted(impostor_sorted, thresholds, side="left").astype(np.float64) / float(impostor_sorted.size)
    )
    frr = (
        np.searchsorted(genuine_sorted, thresholds, side="left").astype(np.float64) / float(genuine_sorted.size)
    )

    diff = far - frr
    crossing = np.flatnonzero(diff <= 0)
    if crossing.size:
        idx_hi = int(crossing[0])
        if idx_hi == 0:
            eer_threshold = float(thresholds[0])
            far_at_eer = float(far[0])
            frr_at_eer = float(frr[0])
        else:
            idx_lo = idx_hi - 1
            diff_lo = float(diff[idx_lo])
            diff_hi = float(diff[idx_hi])
            alpha = 0.0 if diff_lo == diff_hi else float(diff_lo / (diff_lo - diff_hi))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            eer_threshold = float(thresholds[idx_lo] + alpha * (thresholds[idx_hi] - thresholds[idx_lo]))
            far_at_eer = float(far[idx_lo] + alpha * (far[idx_hi] - far[idx_lo]))
            frr_at_eer = float(frr[idx_lo] + alpha * (frr[idx_hi] - frr[idx_lo]))
    else:
        idx_best = int(np.argmin(np.abs(diff)))
        eer_threshold = float(thresholds[idx_best])
        far_at_eer = float(far[idx_best])
        frr_at_eer = float(frr[idx_best])

    eer = 0.5 * (far_at_eer + frr_at_eer)
    return {
        "thresholds": thresholds,
        "far": far,
        "frr": frr,
        "eer": float(eer),
        "eer_threshold": eer_threshold,
        "far_at_eer": far_at_eer,
        "frr_at_eer": frr_at_eer,
    }


def _compute_multiclass_verification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    if scores.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {scores.shape}")
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels must be a 1D array aligned with the score matrix")

    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    n_samples, n_classes = scores.shape
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("labels contain values outside the score matrix class range")

    genuine_scores = scores[np.arange(n_samples), labels]
    impostor_mask = np.ones(scores.shape, dtype=bool)
    impostor_mask[np.arange(n_samples), labels] = False
    impostor_scores = scores[impostor_mask]

    pooled = _compute_far_frr(genuine_scores=genuine_scores, impostor_scores=impostor_scores)
    per_subject = []
    for subject_label in sorted(np.unique(labels).tolist()):
        subject_curve = _compute_far_frr(
            genuine_scores=scores[labels == subject_label, subject_label],
            impostor_scores=scores[labels != subject_label, subject_label],
        )
        per_subject.append(
            {
                "subject_label": int(subject_label),
                "eer": float(subject_curve["eer"]),
                "eer_pct": float(100.0 * subject_curve["eer"]),
                "eer_threshold": float(subject_curve["eer_threshold"]),
                "far_at_eer": float(subject_curve["far_at_eer"]),
                "frr_at_eer": float(subject_curve["frr_at_eer"]),
            }
        )

    return {
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "n_genuine_scores": int(genuine_scores.size),
        "n_impostor_scores": int(impostor_scores.size),
        "pooled": pooled,
        "per_subject": per_subject,
        "macro_subject_eer": float(np.mean([row["eer"] for row in per_subject])),
    }


def _plot_far_frr_curve(curve: dict, out_path, title: str) -> None:
    thresholds = np.asarray(curve["thresholds"], dtype=np.float64)
    far = 100.0 * np.asarray(curve["far"], dtype=np.float64)
    frr = 100.0 * np.asarray(curve["frr"], dtype=np.float64)
    eer_threshold = float(curve["eer_threshold"])
    eer_pct = float(100.0 * curve["eer"])

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(thresholds, far, color="#d62728", linewidth=2.4, label="FAR")
    ax.plot(thresholds, frr, color="#1f77b4", linewidth=2.4, label="FRR")
    ax.axvline(eer_threshold, color="#4d4d4d", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.scatter([eer_threshold], [eer_pct], color="#111111", s=54, zorder=4, label=f"EER = {eer_pct:.2f}%")
    ax.annotate(
        f"EER = {eer_pct:.2f}%\n$\\tau^*$ = {eer_threshold:.3f}",
        xy=(eer_threshold, eer_pct),
        xytext=(12, 10),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bfbfbf", "alpha": 0.95},
    )
    ax.set_title(title)
    ax.set_xlabel("Acceptance Threshold")
    ax.set_ylabel("Error Rate (%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    inset = ax.inset_axes([0.52, 0.16, 0.38, 0.34])
    inset.plot(thresholds, far, color="#d62728", linewidth=1.8)
    inset.plot(thresholds, frr, color="#1f77b4", linewidth=1.8)
    inset.axvline(eer_threshold, color="#4d4d4d", linewidth=1.0, linestyle="--", alpha=0.8)
    inset.scatter([eer_threshold], [eer_pct], color="#111111", s=20, zorder=4)
    zoom_x_low = max(float(thresholds.min()), eer_threshold - 0.08)
    zoom_x_high = min(float(thresholds.max()), eer_threshold + 0.08)
    in_zoom = (thresholds >= zoom_x_low) & (thresholds <= zoom_x_high)
    if np.any(in_zoom):
        zoom_y_high = max(2.0, 1.25 * float(np.max(np.concatenate([far[in_zoom], frr[in_zoom], [eer_pct]]))))
    else:
        zoom_y_high = max(2.0, 1.5 * eer_pct)
    inset.set_xlim(zoom_x_low, zoom_x_high)
    inset.set_ylim(0.0, zoom_y_high)
    inset.set_title("EER zoom", fontsize=9)
    inset.grid(True, alpha=0.2)
    inset.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_biometric_eval(out_cfg: OutputConfig, title: str) -> dict:
    out_cfg.root.mkdir(parents=True, exist_ok=True)

    scores_payload = np.load(out_cfg.baseline_scores_path)
    scores = np.asarray(scores_payload["logits"], dtype=np.float64)
    labels = np.asarray(scores_payload["labels"], dtype=np.int64)

    metrics = _compute_multiclass_verification_metrics(scores=scores, labels=labels)

    curve_path = out_cfg.root / "subject_recognition_biometric_curve.npz"
    np.savez(
        curve_path,
        thresholds=np.asarray(metrics["pooled"]["thresholds"], dtype=np.float64),
        far=np.asarray(metrics["pooled"]["far"], dtype=np.float64),
        frr=np.asarray(metrics["pooled"]["frr"], dtype=np.float64),
    )

    figure_path = out_cfg.root / "subject_recognition_biometric_far_frr.png"
    _plot_far_frr_curve(metrics["pooled"], figure_path, title=title)

    summary = {
        "scores_path": str(out_cfg.baseline_scores_path),
        "curve_path": str(curve_path),
        "figure_path": str(figure_path),
        "n_samples": metrics["n_samples"],
        "n_classes": metrics["n_classes"],
        "n_genuine_scores": metrics["n_genuine_scores"],
        "n_impostor_scores": metrics["n_impostor_scores"],
        "pooled_eer": float(metrics["pooled"]["eer"]),
        "pooled_eer_pct": float(100.0 * metrics["pooled"]["eer"]),
        "eer_threshold": float(metrics["pooled"]["eer_threshold"]),
        "far_at_eer": float(metrics["pooled"]["far_at_eer"]),
        "frr_at_eer": float(metrics["pooled"]["frr_at_eer"]),
        "macro_subject_eer": float(metrics["macro_subject_eer"]),
        "macro_subject_eer_pct": float(100.0 * metrics["macro_subject_eer"]),
        "per_subject": metrics["per_subject"],
    }

    summary_path = out_cfg.root / "subject_recognition_biometric_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def run_human_recognition_biometric_eval() -> dict:
    return run_biometric_eval(
        out_cfg=build_bnci2014_001_human_recognition_output_config(),
        title="BNCI2014-001 Subject Verification",
    )


if __name__ == "__main__":
    result = run_human_recognition_biometric_eval()
    compact = {
        "summary_path": result["summary_path"],
        "figure_path": result["figure_path"],
        "pooled_eer_pct": result["pooled_eer_pct"],
        "eer_threshold": result["eer_threshold"],
        "macro_subject_eer_pct": result["macro_subject_eer_pct"],
    }
    print(compact)
