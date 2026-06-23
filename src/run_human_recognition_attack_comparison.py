from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import fields

import numpy as np

if __package__:
    from .config import BaselineConfig
    from .data import load_windows
    from .human_recognition_config import (
        build_bnci2014_001_human_recognition_config,
        build_bnci2014_001_human_recognition_output_config,
    )
    from .model_oracle import load_eegnet_checkpoint, make_score_fn
    from .run_attack_basis_comparison import (
        _make_config,
        _plot_power_ratio_comparison,
        _plot_waveform_example,
        _plot_zero_accuracy_channel_budget,
        _rerun_attack_for_sample,
        _run_variant,
    )
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import BaselineConfig
    from src.data import load_windows
    from src.human_recognition_config import (
        build_bnci2014_001_human_recognition_config,
        build_bnci2014_001_human_recognition_output_config,
    )
    from src.model_oracle import load_eegnet_checkpoint, make_score_fn
    from src.run_attack_basis_comparison import (
        _make_config,
        _plot_power_ratio_comparison,
        _plot_waveform_example,
        _plot_zero_accuracy_channel_budget,
        _rerun_attack_for_sample,
        _run_variant,
    )


def _build_variants() -> list[dict]:
    shortlist_size = int(os.environ.get("EEG_ATTACK_CHANNEL_SHORTLIST_SIZE", "11"))
    config_updates = {}

    int_env_overrides = {
        "EEG_ATTACK_SUPPORT_BUDGET_K": "support_budget_k",
        "EEG_ATTACK_MAX_OUTER_ITERS": "max_outer_iters",
        "EEG_ATTACK_MAX_QUERY_BUDGET": "max_query_budget",
        "EEG_ATTACK_CANDIDATE_PROBE_RESTARTS": "candidate_probe_restarts",
        "EEG_ATTACK_SPSA_STEPS": "spsa_steps",
        "EEG_ATTACK_SPSA_RESTARTS": "spsa_restarts",
    }
    for env_name, field_name in int_env_overrides.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            config_updates[field_name] = int(env_value)
    if "support_budget_k" in config_updates and "max_outer_iters" not in config_updates:
        config_updates["max_outer_iters"] = int(config_updates["support_budget_k"])

    float_env_overrides = {
        "EEG_ATTACK_MAX_PEAK_RATIO": "max_perturbation_peak_ratio",
        "EEG_ATTACK_MAX_COEFF_ABS": "max_coeff_abs",
        "EEG_ATTACK_L2_WEIGHT": "l2_weight",
        "EEG_ATTACK_TV_WEIGHT": "tv_weight",
        "EEG_ATTACK_BAND_WEIGHT": "band_weight",
    }
    for env_name, field_name in float_env_overrides.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            config_updates[field_name] = float(env_value)

    saga_config_updates = {
        "support_mode": "saga_pgd",
        "basis_mode": "hybrid",
        "basis_phase_count": 2,
        "support_budget_k": 8,
        "max_outer_iters": 8,
        "spsa_steps": 60,
        "spsa_step_size": 0.05,
        "spsa_restarts": 2,
        "spsa_init_scale": 0.25,
        "candidate_probe_restarts": 2,
        "candidate_probe_scale": 0.75,
        "spsa_perturb_scale": 0.02,
        "max_coeff_abs": 0.75,
        "l2_weight": 1e-5,
        "tv_weight": 1e-5,
        "band_weight": 1e-5,
    }
    saga_config_updates.update(config_updates)

    # QELDBA shares SCHS's strong score-only optimizer (greedy channel selection +
    # SPSA refinement, inherited from the base config). The deliberate difference is
    # the high-frequency perturbation unit: a sparse bank of high-frequency atoms.
    qeldba_config_updates = {
        "support_mode": "qeldba",
        "basis_mode": "freq_atom_bank",
        "basis_phase_count": 2,
        "basis_min_hz": float(os.environ.get("QELDBA_BASIS_MIN_HZ", "20.0")),
        "basis_max_hz": float(os.environ.get("QELDBA_BASIS_MAX_HZ", "38.0")),
    }
    qeldba_config_updates.update(config_updates)

    return [
        {
            "name": "human_sparse_channel_hybrid",
            "display_name": "Sparse channel + hybrid waveform",
            "color": "#1f77b4",
            "config": _make_config(
                support_mode="channel_first",
                basis_mode="hybrid",
                basis_phase_count=2,
                **config_updates,
            ),
        },
        {
            "name": "human_saga_pgd",
            "display_name": "SAGA-style sparse channel-time PGD",
            "color": "#9467bd",
            "config": _make_config(**saga_config_updates),
        },
        {
            "name": "human_sparse_channel_time_hybrid",
            "display_name": "Sparse channel-time + hybrid waveform",
            "color": "#2ca02c",
            "config": _make_config(
                support_mode="channel_then_window",
                basis_mode="hybrid",
                basis_phase_count=2,
                channel_shortlist_size=shortlist_size,
                **config_updates,
            ),
        },
        {
            "name": "human_qeldba",
            "display_name": "QELDBA-style high-frequency black-box",
            "color": "#ff7f0e",
            "config": _make_config(**qeldba_config_updates),
        },
    ]


def _balanced_sample_candidates(
    candidate_payloads: list[tuple[int, np.ndarray, int]],
    max_samples: int,
    seed: int,
) -> list[tuple[int, np.ndarray, int]]:
    if max_samples <= 0 or len(candidate_payloads) <= max_samples:
        return candidate_payloads

    rng = np.random.default_rng(seed)
    buckets: dict[int, list[tuple[int, np.ndarray, int]]] = defaultdict(list)
    for payload in candidate_payloads:
        buckets[int(payload[2])].append(payload)

    ordered_labels = sorted(buckets)
    queues: dict[int, deque[tuple[int, np.ndarray, int]]] = {}
    for label in ordered_labels:
        items = list(buckets[label])
        rng.shuffle(items)
        queues[label] = deque(items)

    selected: list[tuple[int, np.ndarray, int]] = []
    while len(selected) < max_samples:
        made_progress = False
        for label in ordered_labels:
            queue = queues[label]
            if not queue:
                continue
            selected.append(queue.popleft())
            made_progress = True
            if len(selected) >= max_samples:
                break
        if not made_progress:
            break

    selected.sort(key=lambda item: int(item[0]))
    return selected


def _baseline_config_from_checkpoint(checkpoint: dict) -> BaselineConfig | None:
    cfg_payload = checkpoint.get("config")
    if not isinstance(cfg_payload, dict):
        return None

    allowed_fields = {field.name for field in fields(BaselineConfig)}
    cfg_kwargs = {key: value for key, value in cfg_payload.items() if key in allowed_fields}
    return BaselineConfig(**cfg_kwargs)


def run_human_recognition_attack_comparison(baseline_cfg=None, out_cfg=None) -> dict:
    if baseline_cfg is None:
        baseline_cfg = build_bnci2014_001_human_recognition_config()
    if out_cfg is None:
        out_cfg = build_bnci2014_001_human_recognition_output_config()
    out_cfg.root.mkdir(parents=True, exist_ok=True)

    requested_device = os.environ.get("EEG_ATTACK_DEVICE")
    max_samples = int(os.environ.get("EEG_ATTACK_MAX_SAMPLES", "128"))

    model, device, checkpoint = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device=requested_device)
    checkpoint_cfg = _baseline_config_from_checkpoint(checkpoint)
    use_checkpoint_config = os.environ.get("EEG_ATTACK_USE_CHECKPOINT_CONFIG", "1") != "0"
    if use_checkpoint_config and checkpoint_cfg is not None:
        baseline_cfg = checkpoint_cfg
    score_fn = make_score_fn(model, device)
    bundle = load_windows(baseline_cfg)

    candidate_payloads: list[tuple[int, np.ndarray, int]] = []
    for idx in range(len(bundle.valid_set)):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        if int(np.argmax(score_fn(x_np))) == y_int:
            candidate_payloads.append((idx, x_np, y_int))

    total_clean_correct = len(candidate_payloads)
    candidate_payloads = _balanced_sample_candidates(
        candidate_payloads=candidate_payloads,
        max_samples=max_samples,
        seed=baseline_cfg.random_seed,
    )
    if not candidate_payloads:
        raise RuntimeError("No clean-correct validation trials available for human-recognition attack evaluation")

    cpu_count = os.cpu_count() or 1
    n_workers = max(1, min(8, cpu_count // 2 if cpu_count > 1 else 1))
    env_workers = os.environ.get("EEG_ATTACK_WORKERS")
    if env_workers is not None:
        n_workers = max(1, int(env_workers))
    worker_device = str(device) if n_workers == 1 else "cpu"

    print(
        f"Comparing human-recognition attacks on {len(candidate_payloads)} sampled clean-correct validation trials "
        f"(total_clean_correct={total_clean_correct}, score_device={device}, worker_device={worker_device}, workers={n_workers})",
        flush=True,
    )

    variants = _build_variants()
    requested_variants = os.environ.get("EEG_ATTACK_VARIANTS")
    if requested_variants:
        requested_names = {name.strip() for name in requested_variants.split(",") if name.strip()}
        variants = [variant for variant in variants if variant["name"] in requested_names]
        if not variants:
            raise ValueError(f"No attack variants matched EEG_ATTACK_VARIANTS={requested_variants!r}")

    variant_reports = []
    for variant in variants:
        print(f"Running variant: {variant['display_name']}", flush=True)
        variant_reports.append(
            _run_variant(
                variant=variant,
                candidate_payloads=candidate_payloads,
                baseline_cfg=baseline_cfg,
                out_cfg=out_cfg,
                score_device=str(device),
                worker_device=worker_device,
                n_workers=n_workers,
                n_channels=bundle.n_chans,
            )
        )

    comparison = {
        "dataset_name": baseline_cfg.dataset_name,
        "model_name": baseline_cfg.model_name,
        "target_mode": baseline_cfg.target_mode,
        "evaluation_protocol": baseline_cfg.evaluation_protocol,
        "used_checkpoint_config": bool(use_checkpoint_config and checkpoint_cfg is not None),
        "checkpoint_split_summary": checkpoint.get("split_summary", {}),
        "score_device": str(device),
        "worker_device": worker_device,
        "n_workers": n_workers,
        "n_clean_correct_total": total_clean_correct,
        "n_clean_correct_attacked": len(candidate_payloads),
        "max_samples": max_samples,
        "variants": variant_reports,
    }

    power_plot_path = out_cfg.root / "human_recognition_attack_basis_power_ratio_comparison.png"
    channel_budget_plot_path = out_cfg.root / "human_recognition_attack_basis_zero_accuracy_channel_budget.png"
    _plot_power_ratio_comparison(comparison, power_plot_path)
    _plot_zero_accuracy_channel_budget(comparison, channel_budget_plot_path)

    success_sets = [{int(row["idx"]) for row in variant["per_sample"] if row["success"]} for variant in comparison["variants"]]
    common_success = set.intersection(*success_sets) if success_sets and all(success_sets) else set()
    example_sample_idx = None if not common_success else min(common_success)

    variant_examples = []
    for variant, report in zip(variants, comparison["variants"]):
        if example_sample_idx is not None:
            sample_idx = int(example_sample_idx)
        else:
            success_row = next((row for row in report["per_sample"] if row["success"]), None)
            fallback_row = report["per_sample"][0]
            sample_idx = int(fallback_row["idx"] if success_row is None else success_row["idx"])

        x, y, _ = bundle.valid_set[sample_idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        rerun = _rerun_attack_for_sample(
            sample_idx=sample_idx,
            x_np=x_np,
            y_int=y_int,
            attack_cfg=variant["config"],
            baseline_cfg=baseline_cfg,
            score_fn=score_fn,
            model=model,
            device=device,
        )
        variant_examples.append(
            {
                "sample_idx": sample_idx,
                "x_np": x_np,
                "display_name": variant["display_name"],
                "color": variant["color"],
                "result": rerun["result"],
                "adv_pred": rerun["adv_pred"],
                "true_label": y_int,
                "power_ratio_pct": rerun["power_ratio_pct"],
            }
        )

    waveform_plot_path = out_cfg.root / "human_recognition_attack_basis_waveform_example.png"
    _plot_waveform_example(
        baseline_cfg=baseline_cfg,
        variant_examples=variant_examples,
        out_path=waveform_plot_path,
    )

    comparison["comparison_plot_path"] = str(power_plot_path)
    comparison["channel_budget_plot_path"] = str(channel_budget_plot_path)
    comparison["waveform_plot_path"] = str(waveform_plot_path)
    comparison["example_sample_idx"] = None if example_sample_idx is None else int(example_sample_idx)

    report_path = out_cfg.root / "human_recognition_attack_basis_comparison_report.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    comparison["report_path"] = str(report_path)
    return comparison


if __name__ == "__main__":
    summary = run_human_recognition_attack_comparison()
    compact = {
        "report_path": summary["report_path"],
        "comparison_plot_path": summary["comparison_plot_path"],
        "channel_budget_plot_path": summary["channel_budget_plot_path"],
        "waveform_plot_path": summary["waveform_plot_path"],
        "n_clean_correct_total": summary["n_clean_correct_total"],
        "n_clean_correct_attacked": summary["n_clean_correct_attacked"],
        "variants": [
            {
                "name": variant["variant_name"],
                "k_zero_accuracy": variant["k_zero_accuracy"],
                "power_ratio_zero_accuracy_pct": variant["power_ratio_zero_accuracy_pct"],
                "attacked_accuracy": variant["attacked_accuracy"],
                "attack_success_rate": variant["attack_success_rate"],
            }
            for variant in summary["variants"]
        ],
    }
    print(compact)
