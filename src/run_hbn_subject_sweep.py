from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator

if __package__:
    from .human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )
    from .plot_hbn_first_success_histogram import plot_histogram, summarize_first_success
    from .plot_hbn_subject_sweep import DEFAULT_SOURCE_URL, plot_hbn_subject_sweep
    from .run_human_recognition_attack_comparison import run_human_recognition_attack_comparison
    from .run_human_recognition_biometric_eval import run_biometric_eval
    from .train_baseline import train_and_save_baseline
else:
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )
    from src.plot_hbn_first_success_histogram import plot_histogram, summarize_first_success
    from src.plot_hbn_subject_sweep import DEFAULT_SOURCE_URL, plot_hbn_subject_sweep
    from src.run_human_recognition_attack_comparison import run_human_recognition_attack_comparison
    from src.run_human_recognition_biometric_eval import run_biometric_eval
    from src.train_baseline import train_and_save_baseline


DEFAULT_SUBJECT_COUNTS = (20, 35, 50, 75, 100, 122)
DEFAULT_MODEL_NAMES = ("EEGConformer",)
MODEL_ALIASES = {
    "eegconformer": "EEGConformer",
    "conformer": "EEGConformer",
    "eegnet": "EEGNet",
    "eegnetv4": "EEGNet",
    "shallowfbcspnet": "ShallowFBCSPNet",
    "shallow": "ShallowFBCSPNet",
}


@contextmanager
def _patched_environ(updates: dict[str, str | None]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_counts(raw: list[str] | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SUBJECT_COUNTS)
    counts = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                counts.append(int(part))
    return counts


def _parse_model_names(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MODEL_NAMES)
    model_names = []
    for item in raw:
        for part in item.split(","):
            key = part.strip()
            if not key:
                continue
            canonical = MODEL_ALIASES.get(key.lower(), key)
            if canonical not in model_names:
                model_names.append(canonical)
    return model_names


def _model_slug(model_name: str) -> str:
    return model_name.lower().replace("_", "").replace("-", "")


def _stage_enabled(stages: set[str], stage: str) -> bool:
    return "all" in stages or stage in stages


def _attack_sample_count(subject_count: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    return min(128, max(64, int(subject_count)))


def _run_name(subject_count: int) -> str:
    return f"n{int(subject_count):03d}"


def _summarize_artifacts(run_dir: Path, intended_subject_count: int, intended_model_name: str) -> dict:
    metrics_path = run_dir / "subject_recognition_metrics.json"
    biometric_path = run_dir / "subject_recognition_biometric_metrics.json"
    attack_summary_path = run_dir / "hbn_k12_first_success_channel_summary.json"
    report_path = run_dir / "human_recognition_attack_basis_comparison_report.json"

    summary = {
        "run_dir": str(run_dir),
        "intended_model_name": intended_model_name,
        "intended_subject_count": int(intended_subject_count),
        "baseline_metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "biometric_metrics_path": str(biometric_path) if biometric_path.exists() else None,
        "attack_report_path": str(report_path) if report_path.exists() else None,
        "attack_summary_path": str(attack_summary_path) if attack_summary_path.exists() else None,
    }
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        split = metrics.get("split_summary", {})
        summary.update(
            {
                "selected_subjects": split.get("selected_subjects"),
                "model_name": metrics.get("model_name"),
                "candidate_subjects_with_train_and_valid": split.get("candidate_subjects_with_train_and_valid"),
                "n_common_eeg_channels": split.get("n_common_eeg_channels"),
                "n_train_samples": metrics.get("n_train_samples"),
                "n_valid_samples": metrics.get("n_valid_samples"),
                "best_val_acc": metrics.get("best_val_acc"),
                "model_name_ok": metrics.get("model_name") == intended_model_name,
                "subject_count_ok": split.get("selected_subjects") == int(intended_subject_count),
                "channel_count_ok": split.get("n_common_eeg_channels") == 64,
            }
        )
    if biometric_path.exists():
        biometric = json.loads(biometric_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "pooled_eer_pct": biometric.get("pooled_eer_pct"),
                "macro_subject_eer_pct": biometric.get("macro_subject_eer_pct"),
            }
        )
    if attack_summary_path.exists():
        attack = json.loads(attack_summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "n_clean_correct_attacked": attack.get("n_clean_correct_attacked"),
                "attack_success_rate": attack.get("attack_success_rate"),
                "attacked_accuracy": attack.get("attacked_accuracy"),
                "first_success_channel_summary": attack.get("first_success_channel_summary"),
            }
        )
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "attack_report_model_name": report.get("model_name"),
                "attack_model_name_ok": report.get("model_name") == intended_model_name,
                "attack_used_checkpoint_config": report.get("used_checkpoint_config"),
                "attack_checkpoint_selected_subjects": report.get("checkpoint_split_summary", {}).get("selected_subjects"),
                "attack_checkpoint_n_common_eeg_channels": report.get("checkpoint_split_summary", {}).get("n_common_eeg_channels"),
            }
        )
    return summary


def _write_run_manifest(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "hbn_subject_sweep_run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_subject_count(args: argparse.Namespace, subject_count: int, model_name: str, nested_model_root: bool) -> dict:
    run_dir = args.root / _model_slug(model_name) / _run_name(subject_count) if nested_model_root else args.root / _run_name(subject_count)
    attack_max_samples = _attack_sample_count(subject_count, args.attack_max_samples)
    stages = set(args.stages)
    run_env = {
        "EEG_HBN_OUTPUT_ROOT": str(run_dir),
        "EEG_HBN_MAX_SUBJECTS": str(subject_count),
        "EEG_HBN_CHANNEL_LIMIT": "64",
    }
    if args.hbn_path:
        run_env["EEG_HBN_PATH"] = str(args.hbn_path)
    if args.epochs is not None:
        run_env["EEG_HBN_EPOCHS"] = str(args.epochs)

    attack_env = {
        "EEG_ATTACK_VARIANTS": "human_sparse_channel_hybrid",
        "EEG_ATTACK_MAX_SAMPLES": str(attack_max_samples),
        "EEG_ATTACK_SUPPORT_BUDGET_K": "12",
        "EEG_ATTACK_MAX_OUTER_ITERS": "12",
        "EEG_ATTACK_MAX_QUERY_BUDGET": "45000",
        "EEG_ATTACK_MAX_PEAK_RATIO": "0.1",
        "EEG_ATTACK_MAX_COEFF_ABS": "7.5",
        "EEG_ATTACK_L2_WEIGHT": "1e-8",
        "EEG_ATTACK_TV_WEIGHT": "1e-8",
        "EEG_ATTACK_BAND_WEIGHT": "1e-8",
        "EEG_ATTACK_USE_CHECKPOINT_CONFIG": "1",
    }
    if args.attack_device:
        attack_env["EEG_ATTACK_DEVICE"] = args.attack_device
    if args.attack_workers is not None:
        attack_env["EEG_ATTACK_WORKERS"] = str(args.attack_workers)

    manifest = {
        "source_url": DEFAULT_SOURCE_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "subject_count": int(subject_count),
        "attack_max_samples": int(attack_max_samples),
        "stages": sorted(stages),
        "env": {**run_env, **attack_env},
        "dry_run": bool(args.dry_run),
        "steps": [],
    }
    print(
        json.dumps(
            {"run_dir": str(run_dir), "model_name": model_name, "subject_count": subject_count, "dry_run": args.dry_run},
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        _write_run_manifest(run_dir, manifest) if args.write_dry_run_manifest else None
        return manifest

    with _patched_environ(run_env):
        baseline_cfg = build_hbn_r1_l100_human_recognition_config(model_name=model_name)
        out_cfg = build_hbn_r1_l100_human_recognition_output_config()
        out_cfg.root.mkdir(parents=True, exist_ok=True)

        baseline_ready = out_cfg.baseline_model_path.exists() and out_cfg.baseline_metrics_path.exists()
        if _stage_enabled(stages, "baseline"):
            if baseline_ready and args.skip_existing:
                manifest["steps"].append({"stage": "baseline", "status": "skipped_existing"})
            else:
                result = train_and_save_baseline(baseline_cfg, out_cfg)
                manifest["steps"].append({"stage": "baseline", "status": "completed", "result": result})

        if _stage_enabled(stages, "biometric"):
            biometric_path = out_cfg.root / "subject_recognition_biometric_metrics.json"
            if biometric_path.exists() and args.skip_existing:
                manifest["steps"].append({"stage": "biometric", "status": "skipped_existing"})
            else:
                result = run_biometric_eval(
                    out_cfg=out_cfg,
                    title=f"HBN R1-L100 {model_name} {subject_count}-Subject Verification",
                )
                manifest["steps"].append({"stage": "biometric", "status": "completed", "result": result})

        if _stage_enabled(stages, "attack"):
            report_path = out_cfg.root / "human_recognition_attack_basis_comparison_report.json"
            if report_path.exists() and args.skip_existing:
                manifest["steps"].append({"stage": "attack", "status": "skipped_existing"})
            else:
                with _patched_environ(attack_env):
                    result = run_human_recognition_attack_comparison(baseline_cfg=baseline_cfg, out_cfg=out_cfg)
                manifest["steps"].append(
                    {
                        "stage": "attack",
                        "status": "completed",
                        "result": {
                            "report_path": result.get("report_path"),
                            "n_clean_correct_total": result.get("n_clean_correct_total"),
                            "n_clean_correct_attacked": result.get("n_clean_correct_attacked"),
                        },
                    }
                )

        if _stage_enabled(stages, "histogram"):
            report_path = out_cfg.root / "human_recognition_attack_basis_comparison_report.json"
            histogram_path = out_cfg.root / "hbn_k12_first_success_channel_histogram.png"
            summary_path = out_cfg.root / "hbn_k12_first_success_channel_summary.json"
            if summary_path.exists() and histogram_path.exists() and args.skip_existing:
                manifest["steps"].append({"stage": "histogram", "status": "skipped_existing"})
            elif report_path.exists():
                summary = summarize_first_success(report_path, "human_sparse_channel_hybrid")
                plot_histogram(summary, histogram_path)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                manifest["steps"].append(
                    {
                        "stage": "histogram",
                        "status": "completed",
                        "summary_path": str(summary_path),
                        "histogram_path": str(histogram_path),
                    }
                )
            else:
                manifest["steps"].append(
                    {
                        "stage": "histogram",
                        "status": "missing_attack_report",
                        "report_path": str(report_path),
                    }
                )

    manifest["artifact_summary"] = _summarize_artifacts(run_dir, subject_count, model_name)
    _write_run_manifest(run_dir, manifest)
    print(json.dumps(manifest["artifact_summary"], indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nested HBN R1 subject-count baseline/attack sweep.")
    parser.add_argument("--root", type=Path, default=Path("outputs/hbn_subject_sweep"))
    parser.add_argument("--subject-counts", nargs="*", default=None, help="Counts such as '20 35 50' or '20,35,50'.")
    parser.add_argument("--model-names", nargs="*", default=None, help="Models such as 'EEGConformer EEGNet ShallowFBCSPNet'.")
    parser.add_argument("--hbn-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--attack-max-samples", type=int, default=None, help="Default is min(128, max(64, subject_count)).")
    parser.add_argument("--attack-device", default=os.environ.get("EEG_ATTACK_DEVICE", "cuda"))
    parser.add_argument("--attack-workers", type=int, default=1)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", "baseline", "biometric", "attack", "histogram"],
        default=["all"],
    )
    parser.add_argument("--rerun-existing", action="store_true", help="Recompute artifacts even when outputs exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-dry-run-manifest", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Skip the final cross-run summary plot.")
    args = parser.parse_args()
    args.skip_existing = not args.rerun_existing

    args.root.mkdir(parents=True, exist_ok=True)
    counts = _parse_counts(args.subject_counts)
    model_names = _parse_model_names(args.model_names)
    nested_model_root = bool(args.model_names) or model_names != list(DEFAULT_MODEL_NAMES)
    manifests = []
    for model_name in model_names:
        for count in counts:
            manifests.append(run_subject_count(args, count, model_name, nested_model_root=nested_model_root))

    sweep_manifest = {
        "source_url": DEFAULT_SOURCE_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "model_names": model_names,
        "subject_counts": counts,
        "dry_run": bool(args.dry_run),
        "runs": manifests,
    }
    manifest_path = args.root / "hbn_subject_sweep_manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2), encoding="utf-8")
    sweep_manifest["manifest_path"] = str(manifest_path)

    if not args.dry_run and not args.no_plot:
        sweep_manifest["summary"] = plot_hbn_subject_sweep(root=args.root, model_names=model_names)

    print(json.dumps(sweep_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
