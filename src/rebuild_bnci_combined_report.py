"""Rebuild the BNCI combined attack-comparison report + plots from saved per-variant
reports, without re-running the (expensive) attacks.

Use this when the per-variant ``*_full_report.json`` files are already current but the
final combined report / plots failed to regenerate (e.g. a transient CUDA error in the
post-hoc waveform-example rerun). The waveform-example reruns here are forced onto CPU.
"""
from __future__ import annotations

import json
from dataclasses import fields

import numpy as np

from .config import AttackConfig
from .data import load_windows
from .human_recognition_config import (
    build_bnci2014_001_human_recognition_config,
    build_bnci2014_001_human_recognition_output_config,
)
from .model_oracle import load_eegnet_checkpoint, make_score_fn
from .run_attack_basis_comparison import (
    _plot_power_ratio_comparison,
    _plot_waveform_example,
    _plot_zero_accuracy_channel_budget,
    _rerun_attack_for_sample,
)

# Canonical order, matching _build_variants() in run_human_recognition_attack_comparison.
VARIANT_FILES = [
    "human_sparse_channel_hybrid_full_report.json",
    "human_saga_pgd_full_report.json",
    "human_sparse_channel_time_hybrid_full_report.json",
    "human_qeldba_full_report.json",
]


def _attack_config_from_report(report: dict) -> AttackConfig:
    allowed = {f.name for f in fields(AttackConfig)}
    cfg_kwargs = {k: v for k, v in report["attack_config"].items() if k in allowed}
    return AttackConfig(**cfg_kwargs)


def _baseline_config_from_checkpoint(checkpoint: dict):
    from .config import BaselineConfig

    cfg_payload = checkpoint.get("config")
    if not isinstance(cfg_payload, dict):
        return None
    allowed = {f.name for f in fields(BaselineConfig)}
    return BaselineConfig(**{k: v for k, v in cfg_payload.items() if k in allowed})


def main() -> None:
    baseline_cfg = build_bnci2014_001_human_recognition_config()
    out_cfg = build_bnci2014_001_human_recognition_output_config()

    variant_reports = [json.loads((out_cfg.root / f).read_text()) for f in VARIANT_FILES]

    # CPU model for the (cheap) waveform-example reruns, to avoid GPU flakiness.
    model, device, checkpoint = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device="cpu")
    checkpoint_cfg = _baseline_config_from_checkpoint(checkpoint)
    if checkpoint_cfg is not None:
        baseline_cfg = checkpoint_cfg
    score_fn = make_score_fn(model, device)
    bundle = load_windows(baseline_cfg)

    n_attacked = variant_reports[0]["n_clean_correct_attacked"]
    comparison = {
        "dataset_name": baseline_cfg.dataset_name,
        "model_name": baseline_cfg.model_name,
        "target_mode": baseline_cfg.target_mode,
        "evaluation_protocol": baseline_cfg.evaluation_protocol,
        "score_device": str(device),
        "worker_device": str(device),
        "n_clean_correct_attacked": n_attacked,
        "variants": variant_reports,
    }

    power_plot_path = out_cfg.root / "human_recognition_attack_basis_power_ratio_comparison.png"
    channel_budget_plot_path = out_cfg.root / "human_recognition_attack_basis_zero_accuracy_channel_budget.png"
    _plot_power_ratio_comparison(comparison, power_plot_path)
    _plot_zero_accuracy_channel_budget(comparison, channel_budget_plot_path)

    success_sets = [{int(r["idx"]) for r in v["per_sample"] if r["success"]} for v in variant_reports]
    common_success = set.intersection(*success_sets) if success_sets and all(success_sets) else set()
    example_sample_idx = None if not common_success else min(common_success)

    variant_examples = []
    for report in variant_reports:
        if example_sample_idx is not None:
            sample_idx = int(example_sample_idx)
        else:
            success_row = next((r for r in report["per_sample"] if r["success"]), None)
            sample_idx = int((success_row or report["per_sample"][0])["idx"])

        x, y, _ = bundle.valid_set[sample_idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        rerun = _rerun_attack_for_sample(
            sample_idx=sample_idx,
            x_np=x_np,
            y_int=y_int,
            attack_cfg=_attack_config_from_report(report),
            baseline_cfg=baseline_cfg,
            score_fn=score_fn,
            model=model,
            device=device,
        )
        variant_examples.append(
            {
                "sample_idx": sample_idx,
                "x_np": x_np,
                "display_name": report["display_name"],
                "color": report["color"],
                "result": rerun["result"],
                "adv_pred": rerun["adv_pred"],
                "true_label": y_int,
                "power_ratio_pct": rerun["power_ratio_pct"],
            }
        )

    waveform_plot_path = out_cfg.root / "human_recognition_attack_basis_waveform_example.png"
    _plot_waveform_example(baseline_cfg=baseline_cfg, variant_examples=variant_examples, out_path=waveform_plot_path)

    comparison["comparison_plot_path"] = str(power_plot_path)
    comparison["channel_budget_plot_path"] = str(channel_budget_plot_path)
    comparison["waveform_plot_path"] = str(waveform_plot_path)
    comparison["example_sample_idx"] = None if example_sample_idx is None else int(example_sample_idx)

    report_path = out_cfg.root / "human_recognition_attack_basis_comparison_report.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    print(f"Wrote combined report with {len(variant_reports)} variants -> {report_path}")
    for v in variant_reports:
        cs = v["adaptive_channel_summary"]
        print(
            f"  {v['display_name']:42s} ASR={v['attack_success_rate'] * 100:6.2f}%  "
            f"mean_k={cs['mean']}  median_k={cs['median']}"
        )


if __name__ == "__main__":
    main()
