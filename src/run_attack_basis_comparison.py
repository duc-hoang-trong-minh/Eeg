from __future__ import annotations

import json
import os
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .attack.greedy_attack import _apply_peak_ratio_constraint, build_score_attack
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn
from .run_full_freqbank_report import (
    _assemble_channel_prefix_delta,
    _assemble_freqbank_prefix_delta,
    _binary_search_min_scale,
    _channel_index_from_atom,
    _compute_prefix_metrics,
    _make_channel_basis_matrix,
    _make_window_basis_cache,
    _power_ratio_percent,
    _serialize_support,
)

_GLOBAL_SCORE_FN = None
_GLOBAL_BASELINE_CFG = None
_GLOBAL_ATTACK_CFG = None
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None


def _base_config() -> AttackConfig:
    return AttackConfig(
        support_mode="channel_first",
        n_windows=8,
        support_budget_k=8,
        basis_rank_r=8,
        channel_waveform_rank=32,
        basis_min_hz=2.0,
        basis_max_hz=30.0,
        basis_phase_count=2,
        candidate_probe_restarts=4,
        candidate_probe_scale=0.90,
        max_outer_iters=8,
        max_query_budget=25000,
        spsa_steps=180,
        spsa_step_size=0.10,
        spsa_perturb_scale=0.05,
        spsa_restarts=4,
        spsa_init_scale=0.35,
        l2_weight=5e-5,
        tv_weight=5e-5,
        band_weight=5e-5,
        max_coeff_abs=1.0,
        max_perturbation_peak_ratio=0.05,
    )


def _make_config(**updates) -> AttackConfig:
    cfg = _base_config().__dict__.copy()
    cfg.update(updates)
    return AttackConfig(**cfg)


def _build_variants() -> list[dict]:
    return [
        {
            "name": "unrestricted_hybrid",
            "display_name": "Unrestricted hybrid basis",
            "color": "#1f77b4",
            "config": _make_config(
                basis_mode="hybrid",
                basis_phase_count=2,
            ),
        },
        {
            "name": "restricted_freq_bank",
            "display_name": "Restricted frequency bank",
            "color": "#d62728",
            "config": _make_config(
                basis_mode="freq_atom_bank",
                basis_phase_count=4,
            ),
        },
        {
            "name": "saga_pgd",
            "display_name": "SAGA-style sparse channel-time PGD",
            "color": "#9467bd",
            "config": _make_config(
                support_mode="saga_pgd",
                basis_mode="hybrid",
                basis_phase_count=2,
            ),
        },
        {
            "name": "channel_then_window_hybrid",
            "display_name": "Channel-then-window hybrid basis",
            "color": "#2ca02c",
            "config": _make_config(
                support_mode="channel_then_window",
                basis_mode="hybrid",
                basis_phase_count=2,
                channel_shortlist_size=6,
            ),
        },
    ]


def _init_worker(checkpoint_path: str, baseline_cfg: BaselineConfig, attack_cfg: AttackConfig, worker_device: str | None) -> None:
    global _GLOBAL_ATTACK_CFG, _GLOBAL_BASELINE_CFG, _GLOBAL_DEVICE, _GLOBAL_MODEL, _GLOBAL_SCORE_FN
    torch.set_num_threads(1)
    model, device, _ = load_eegnet_checkpoint(checkpoint_path, device=worker_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _GLOBAL_SCORE_FN = make_score_fn(model, device)
    _GLOBAL_BASELINE_CFG = baseline_cfg
    _GLOBAL_ATTACK_CFG = attack_cfg
    _GLOBAL_MODEL = model
    _GLOBAL_DEVICE = device


def _attack_one_sample(task: tuple[int, np.ndarray, int]) -> dict:
    score_fn = _GLOBAL_SCORE_FN
    baseline_cfg = _GLOBAL_BASELINE_CFG
    attack_cfg = _GLOBAL_ATTACK_CFG
    model = _GLOBAL_MODEL
    device = _GLOBAL_DEVICE
    if score_fn is None or baseline_cfg is None or attack_cfg is None:
        raise RuntimeError("Worker globals are not initialized")

    idx, x_np, y_int = task
    attack = build_score_attack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=attack_cfg.n_windows,
        support_budget_k=attack_cfg.support_budget_k,
        basis_rank_r=attack_cfg.basis_rank_r,
        basis_min_hz=attack_cfg.basis_min_hz,
        basis_max_hz=attack_cfg.basis_max_hz,
        basis_mode=attack_cfg.basis_mode,
        basis_phase_count=attack_cfg.basis_phase_count,
        candidate_probe_restarts=attack_cfg.candidate_probe_restarts,
        candidate_probe_scale=attack_cfg.candidate_probe_scale,
        max_outer_iters=attack_cfg.max_outer_iters,
        max_query_budget=attack_cfg.max_query_budget,
        spsa_steps=attack_cfg.spsa_steps,
        spsa_step_size=attack_cfg.spsa_step_size,
        spsa_perturb_scale=attack_cfg.spsa_perturb_scale,
        spsa_restarts=attack_cfg.spsa_restarts,
        spsa_init_scale=attack_cfg.spsa_init_scale,
        l2_weight=attack_cfg.l2_weight,
        tv_weight=attack_cfg.tv_weight,
        band_weight=attack_cfg.band_weight,
        max_coeff_abs=attack_cfg.max_coeff_abs,
        max_perturbation_peak_ratio=attack_cfg.max_perturbation_peak_ratio,
        support_mode=attack_cfg.support_mode,
        channel_waveform_rank=attack_cfg.channel_waveform_rank,
        channel_shortlist_size=attack_cfg.channel_shortlist_size,
        enforce_unique_channels=attack_cfg.enforce_unique_channels,
        stop_on_success=attack_cfg.stop_on_success,
        seed=baseline_cfg.random_seed + idx,
        model=model,
        device=device,
    )
    result = attack.run(x_np, y_int)
    adv_scores = score_fn(result.x_adv)
    adv_pred = int(np.argmax(adv_scores))
    success = adv_pred != y_int
    if result.support and isinstance(result.support[0], tuple):
        boundaries, basis_by_window = _make_window_basis_cache(
            cfg=attack_cfg,
            n_samples=x_np.shape[1],
            sfreq=baseline_cfg.sfreq,
        )
        basis_matrix = None
    else:
        boundaries, basis_by_window = [], {}
        basis_matrix = _make_channel_basis_matrix(
            cfg=attack_cfg,
            n_samples=x_np.shape[1],
            sfreq=baseline_cfg.sfreq,
        )

    prefix_rows = _compute_prefix_metrics(
        x=x_np,
        y=y_int,
        result=result,
        score_fn=score_fn,
        cfg=attack_cfg,
        baseline_cfg=baseline_cfg,
    )
    for row in prefix_rows:
        if not row["success"]:
            row["min_scale"] = None
            row["min_delta_l2"] = None
            row["min_delta_linf"] = None
            continue
        if result.support and isinstance(result.support[0], tuple):
            delta_k = _assemble_freqbank_prefix_delta(
                support=result.support,
                coeffs=result.coeffs,
                k=int(row["k"]),
                n_channels=x_np.shape[0],
                n_samples=x_np.shape[1],
                boundaries=boundaries,
                basis_by_window=basis_by_window,
            )
        else:
            if basis_matrix is None:
                raise RuntimeError("Missing channel basis matrix for channel-first prefix scaling")
            delta_k = _assemble_channel_prefix_delta(
                support=result.support,
                coeffs=result.coeffs,
                k=int(row["k"]),
                n_channels=x_np.shape[0],
                n_samples=x_np.shape[1],
                basis_matrix=basis_matrix,
            )
        delta_k = _apply_peak_ratio_constraint(
            x=x_np,
            delta=delta_k,
            max_perturbation_peak_ratio=attack_cfg.max_perturbation_peak_ratio,
        )
        _, min_scale, min_l2, min_linf = _binary_search_min_scale(
            x=x_np,
            y=y_int,
            delta=delta_k,
            score_fn=score_fn,
        )
        row["min_scale"] = float(min_scale)
        row["min_delta_l2"] = float(min_l2)
        row["min_delta_linf"] = float(min_linf)
        row["min_delta_power_ratio_pct"] = _power_ratio_percent(float(np.linalg.norm(x_np.reshape(-1))), float(min_l2))

    first_success = next((row for row in prefix_rows if row["success"]), None)
    serialized_support = _serialize_support(result.support)
    signal_l2 = float(np.linalg.norm(x_np.reshape(-1)))
    raw_delta_l2 = float(np.linalg.norm(result.delta.reshape(-1)))
    raw_delta_linf = float(np.max(np.abs(result.delta)))
    delivered_support_k = None if first_success is None else int(first_success["k"])
    delivered_delta_l2 = None if first_success is None else float(first_success["min_delta_l2"])
    delivered_delta_linf = None if first_success is None else float(first_success["min_delta_linf"])
    delivered_power_ratio_pct = None if delivered_delta_l2 is None else _power_ratio_percent(signal_l2, delivered_delta_l2)
    delivered_support = [] if delivered_support_k is None else serialized_support[:delivered_support_k]
    delta_channel_energy = np.linalg.norm(result.delta, axis=1)
    return {
        "idx": idx,
        "true_label": y_int,
        "adv_pred": adv_pred,
        "success": bool(success),
        "final_margin": float(result.margin),
        "queries_used": int(result.queries_used),
        "budget_exhausted": bool(result.budget_exhausted),
        "support": serialized_support,
        "signal_l2": signal_l2,
        "delta_l2": raw_delta_l2,
        "delta_linf": raw_delta_linf,
        "raw_delta_l2": raw_delta_l2,
        "raw_delta_linf": raw_delta_linf,
        "raw_delta_power_ratio_pct": _power_ratio_percent(signal_l2, raw_delta_l2),
        "delivered_support": delivered_support,
        "delivered_support_k": delivered_support_k,
        "delivered_min_scale": None if first_success is None else float(first_success["min_scale"]),
        "delivered_delta_l2": delivered_delta_l2,
        "delivered_delta_linf": delivered_delta_linf,
        "delivered_delta_power_ratio_pct": delivered_power_ratio_pct,
        "dominant_channel": int(np.argmax(delta_channel_energy)),
        "first_success_k": None if first_success is None else int(first_success["k"]),
        "first_success_min_l2": None if first_success is None else float(first_success["min_delta_l2"]),
        "first_success_min_power_ratio_pct": delivered_power_ratio_pct,
        "prefix_rows": prefix_rows,
    }


def _summarize_variant(
    variant_name: str,
    display_name: str,
    color: str,
    attack_cfg: AttackConfig,
    baseline_cfg: BaselineConfig,
    report_rows: list[dict],
    n_channels: int,
    score_device: str,
    worker_device: str,
) -> dict:
    n_candidates = len(report_rows)
    k_max = attack_cfg.support_budget_k
    n_success = int(sum(int(row["success"]) for row in report_rows))

    channel_counts = np.zeros((n_channels,), dtype=np.int64)
    delivered_channel_counts = np.zeros((n_channels,), dtype=np.int64)
    prefix_correct_counts = np.zeros((k_max,), dtype=np.int64)
    prefix_l2_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_linf_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_power_ratio_pct_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_l2_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_linf_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_min_power_ratio_pct_success_sums = np.zeros((k_max,), dtype=np.float64)
    prefix_success_counts = np.zeros((k_max,), dtype=np.int64)
    delivered_support_ks = []
    delivered_power_ratio_pcts = []
    raw_power_ratio_pcts = []

    for result_row in report_rows:
        for atom in result_row["support"]:
            channel = _channel_index_from_atom(tuple(atom) if isinstance(atom, list) else atom)
            channel_counts[channel] += 1
        for atom in result_row.get("delivered_support", []):
            channel = _channel_index_from_atom(tuple(atom) if isinstance(atom, list) else atom)
            delivered_channel_counts[channel] += 1
        if result_row.get("delivered_support_k") is not None:
            delivered_support_ks.append(int(result_row["delivered_support_k"]))
        if result_row.get("delivered_delta_power_ratio_pct") is not None:
            delivered_power_ratio_pcts.append(float(result_row["delivered_delta_power_ratio_pct"]))
        raw_power_ratio_pcts.append(
            float(
                result_row.get(
                    "raw_delta_power_ratio_pct",
                    _power_ratio_percent(float(result_row["signal_l2"]), float(result_row["delta_l2"])),
                )
            )
        )
        prefix_rows = result_row["prefix_rows"]
        signal_l2 = float(result_row["signal_l2"])
        if not prefix_rows:
            raise RuntimeError("Expected at least one prefix row per attacked sample")
        for k_idx in range(k_max):
            effective_row = prefix_rows[min(k_idx, len(prefix_rows) - 1)]
            prefix_correct_counts[k_idx] += int(not effective_row["success"])
            prefix_l2_sums[k_idx] += float(effective_row["delta_l2"])
            prefix_linf_sums[k_idx] += float(effective_row["delta_linf"])
            prefix_power_ratio_pct_sums[k_idx] += _power_ratio_percent(signal_l2, float(effective_row["delta_l2"]))
            if effective_row["success"] and effective_row["min_delta_l2"] is not None:
                prefix_success_counts[k_idx] += 1
                prefix_min_l2_success_sums[k_idx] += float(effective_row["min_delta_l2"])
                prefix_min_linf_success_sums[k_idx] += float(effective_row["min_delta_linf"])
                prefix_min_power_ratio_pct_success_sums[k_idx] += _power_ratio_percent(
                    signal_l2,
                    float(effective_row["min_delta_l2"]),
                )

    prefix_summary = []
    for k in range(1, k_max + 1):
        denom = max(n_candidates, 1)
        success_denom = int(prefix_success_counts[k - 1])
        prefix_summary.append(
            {
                "k": k,
                "success_rate": float(1.0 - (prefix_correct_counts[k - 1] / denom)),
                "attacked_accuracy": float(prefix_correct_counts[k - 1] / denom),
                "avg_delta_l2": float(prefix_l2_sums[k - 1] / denom),
                "avg_delta_linf": float(prefix_linf_sums[k - 1] / denom),
                "avg_delta_power_ratio_pct": float(prefix_power_ratio_pct_sums[k - 1] / denom),
                "avg_min_delta_l2_success_only": None
                if success_denom == 0
                else float(prefix_min_l2_success_sums[k - 1] / success_denom),
                "avg_min_delta_linf_success_only": None
                if success_denom == 0
                else float(prefix_min_linf_success_sums[k - 1] / success_denom),
                "avg_min_power_ratio_pct_success_only": None
                if success_denom == 0
                else float(prefix_min_power_ratio_pct_success_sums[k - 1] / success_denom),
                "avg_min_delta_l2_zero_accuracy": None
                if prefix_correct_counts[k - 1] != 0
                else float(prefix_min_l2_success_sums[k - 1] / denom),
                "avg_min_power_ratio_pct_zero_accuracy": None
                if prefix_correct_counts[k - 1] != 0
                else float(prefix_min_power_ratio_pct_success_sums[k - 1] / denom),
            }
        )

    zero_accuracy_rows = [row for row in prefix_summary if row["attacked_accuracy"] == 0.0]
    k_zero_accuracy = None if not zero_accuracy_rows else int(zero_accuracy_rows[0]["k"])
    power_ratio_zero_accuracy = None
    if zero_accuracy_rows:
        power_ratio_zero_accuracy = float(zero_accuracy_rows[0]["avg_min_power_ratio_pct_zero_accuracy"])

    delivered_support_counts = {
        str(k): int(sum(1 for value in delivered_support_ks if value == k))
        for k in sorted(set(delivered_support_ks))
    }
    adaptive_channel_summary = {
        "mean": None if not delivered_support_ks else float(np.mean(delivered_support_ks)),
        "median": None if not delivered_support_ks else float(np.median(delivered_support_ks)),
        "max": None if not delivered_support_ks else int(max(delivered_support_ks)),
        "counts": delivered_support_counts,
    }
    delivered_power_ratio_summary = {
        "mean": None if not delivered_power_ratio_pcts else float(np.mean(delivered_power_ratio_pcts)),
        "median": None if not delivered_power_ratio_pcts else float(np.median(delivered_power_ratio_pcts)),
        "max": None if not delivered_power_ratio_pcts else float(max(delivered_power_ratio_pcts)),
    }
    raw_power_ratio_summary = {
        "mean": None if not raw_power_ratio_pcts else float(np.mean(raw_power_ratio_pcts)),
        "median": None if not raw_power_ratio_pcts else float(np.median(raw_power_ratio_pcts)),
        "max": None if not raw_power_ratio_pcts else float(max(raw_power_ratio_pcts)),
    }

    report_rows.sort(key=lambda row: int(row["idx"]))
    return {
        "variant_name": variant_name,
        "display_name": display_name,
        "color": color,
        "dataset_name": baseline_cfg.dataset_name,
        "model_name": baseline_cfg.model_name,
        "score_device": score_device,
        "worker_device": worker_device,
        "n_clean_correct_attacked": n_candidates,
        "attack_config": attack_cfg.__dict__,
        "attack_success_rate": float(n_success / max(n_candidates, 1)),
        "attacked_accuracy": float((n_candidates - n_success) / max(n_candidates, 1)),
        "k_zero_accuracy": k_zero_accuracy,
        "power_ratio_zero_accuracy_pct": power_ratio_zero_accuracy,
        "adaptive_channel_summary": adaptive_channel_summary,
        "delivered_power_ratio_pct_summary": delivered_power_ratio_summary,
        "raw_power_ratio_pct_summary": raw_power_ratio_summary,
        "prefix_summary": prefix_summary,
        "selected_channel_counts": channel_counts.tolist(),
        "delivered_channel_counts": delivered_channel_counts.tolist(),
        "per_sample": report_rows,
    }


def _run_variant(
    variant: dict,
    candidate_payloads: list[tuple[int, np.ndarray, int]],
    baseline_cfg: BaselineConfig,
    out_cfg: OutputConfig,
    score_device: str,
    worker_device: str,
    n_workers: int,
    n_channels: int,
) -> dict:
    report_rows: list[dict] = []

    def _consume(result_row: dict, order: int) -> None:
        report_rows.append(result_row)
        if order % 10 == 0 or order == len(candidate_payloads):
            success_count = sum(int(row["success"]) for row in report_rows)
            attacked_accuracy = (order - success_count) / max(order, 1)
            print(
                f"{variant['name']} [{order}/{len(candidate_payloads)}] "
                f"success_rate={success_count / order:.4f} attacked_accuracy={attacked_accuracy:.4f}",
                flush=True,
            )

    if n_workers == 1:
        model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device=worker_device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        try:
            global _GLOBAL_SCORE_FN, _GLOBAL_BASELINE_CFG, _GLOBAL_ATTACK_CFG, _GLOBAL_MODEL, _GLOBAL_DEVICE
            _GLOBAL_SCORE_FN = make_score_fn(model, device)
            _GLOBAL_BASELINE_CFG = baseline_cfg
            _GLOBAL_ATTACK_CFG = variant["config"]
            _GLOBAL_MODEL = model
            _GLOBAL_DEVICE = device
            for order, task in enumerate(candidate_payloads, start=1):
                _consume(_attack_one_sample(task), order)
        finally:
            del model
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(str(out_cfg.baseline_model_path), baseline_cfg, variant["config"], worker_device),
        ) as pool:
            for order, result_row in enumerate(pool.imap_unordered(_attack_one_sample, candidate_payloads, chunksize=1), start=1):
                _consume(result_row, order)

    summary = _summarize_variant(
        variant_name=variant["name"],
        display_name=variant["display_name"],
        color=variant["color"],
        attack_cfg=variant["config"],
        baseline_cfg=baseline_cfg,
        report_rows=report_rows,
        n_channels=n_channels,
        score_device=score_device,
        worker_device=worker_device,
    )
    report_path = out_cfg.root / f"{variant['name']}_full_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    summary["report_path"] = str(report_path)
    return summary


def _plot_power_ratio_comparison(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for variant in summary["variants"]:
        ks = [row["k"] for row in variant["prefix_summary"]]
        ys = [
            np.nan if row["avg_min_power_ratio_pct_zero_accuracy"] is None else row["avg_min_power_ratio_pct_zero_accuracy"]
            for row in variant["prefix_summary"]
        ]
        ax.plot(
            ks,
            ys,
            marker="o",
            linewidth=2.2,
            color=variant["color"],
            label=variant["display_name"],
        )
        if variant["k_zero_accuracy"] is not None and variant["power_ratio_zero_accuracy_pct"] is not None:
            ax.scatter(
                [variant["k_zero_accuracy"]],
                [variant["power_ratio_zero_accuracy_pct"]],
                color=variant["color"],
                s=80,
                zorder=4,
            )
    ax.set_xlabel("Channel Budget K")
    ax.set_ylabel("Minimum Added Perturbation Power (% of Signal Power)")
    ax.set_title("Restricted vs Unrestricted Attack Power Needed for 0% Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_zero_accuracy_channel_budget(summary: dict, out_path: Path) -> None:
    variants = summary["variants"]
    labels = [variant["display_name"] for variant in variants]
    xs = np.arange(len(variants))
    heights = [0 if variant["k_zero_accuracy"] is None else variant["k_zero_accuracy"] for variant in variants]
    colors = [variant["color"] for variant in variants]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(xs, heights, color=colors, alpha=0.88)
    ax.set_xticks(xs, labels, rotation=0)
    ax.set_ylabel("Channels Needed for 0% Attacked Accuracy")
    ax.set_title("Channel Budget Needed to Drive EEGConformer to 0% Accuracy")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, variant in zip(bars, variants):
        if variant["k_zero_accuracy"] is None:
            ax.text(bar.get_x() + bar.get_width() / 2.0, 0.15, "Not reached", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.08,
                str(variant["k_zero_accuracy"]),
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _rerun_attack_for_sample(
    sample_idx: int,
    x_np: np.ndarray,
    y_int: int,
    attack_cfg: AttackConfig,
    baseline_cfg: BaselineConfig,
    score_fn,
    model=None,
    device=None,
) -> dict:
    attack = build_score_attack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=attack_cfg.n_windows,
        support_budget_k=attack_cfg.support_budget_k,
        basis_rank_r=attack_cfg.basis_rank_r,
        basis_min_hz=attack_cfg.basis_min_hz,
        basis_max_hz=attack_cfg.basis_max_hz,
        basis_mode=attack_cfg.basis_mode,
        basis_phase_count=attack_cfg.basis_phase_count,
        candidate_probe_restarts=attack_cfg.candidate_probe_restarts,
        candidate_probe_scale=attack_cfg.candidate_probe_scale,
        max_outer_iters=attack_cfg.max_outer_iters,
        max_query_budget=attack_cfg.max_query_budget,
        spsa_steps=attack_cfg.spsa_steps,
        spsa_step_size=attack_cfg.spsa_step_size,
        spsa_perturb_scale=attack_cfg.spsa_perturb_scale,
        spsa_restarts=attack_cfg.spsa_restarts,
        spsa_init_scale=attack_cfg.spsa_init_scale,
        l2_weight=attack_cfg.l2_weight,
        tv_weight=attack_cfg.tv_weight,
        band_weight=attack_cfg.band_weight,
        max_coeff_abs=attack_cfg.max_coeff_abs,
        max_perturbation_peak_ratio=attack_cfg.max_perturbation_peak_ratio,
        support_mode=attack_cfg.support_mode,
        channel_waveform_rank=attack_cfg.channel_waveform_rank,
        channel_shortlist_size=attack_cfg.channel_shortlist_size,
        enforce_unique_channels=attack_cfg.enforce_unique_channels,
        stop_on_success=attack_cfg.stop_on_success,
        seed=baseline_cfg.random_seed + sample_idx,
        model=model,
        device=device,
    )
    result = attack.run(x_np, y_int)
    adv_scores = score_fn(result.x_adv)
    return {
        "result": result,
        "adv_pred": int(np.argmax(adv_scores)),
        "power_ratio_pct": _power_ratio_percent(
            float(np.linalg.norm(x_np.reshape(-1))),
            float(np.linalg.norm(result.delta.reshape(-1))),
        ),
    }


def _plot_waveform_example(
    baseline_cfg: BaselineConfig,
    variant_examples: list[dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(variant_examples), 1, figsize=(10.5, 6.4), sharex=True)
    if len(variant_examples) == 1:
        axes = [axes]

    for ax, example in zip(axes, variant_examples):
        x_np = example["x_np"]
        t = np.arange(x_np.shape[1], dtype=np.float32) / baseline_cfg.sfreq
        result = example["result"]
        channel_energy = np.linalg.norm(result.delta, axis=1)
        channel_idx = int(np.argmax(channel_energy))
        ax.plot(t, x_np[channel_idx], color="#4d4d4d", linewidth=1.4, label="Original")
        ax.plot(t, result.x_adv[channel_idx], color=example["color"], linewidth=1.4, label="Attacked")
        ax.plot(t, result.delta[channel_idx], color="#d62728", linewidth=1.0, linestyle="--", label="Added perturbation")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{example['display_name']} | sample {example['sample_idx']} | channel {channel_idx} | "
            f"power {example['power_ratio_pct']:.4f}% | pred {example['true_label']} -> {example['adv_pred']}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_attack_basis_comparison() -> dict:
    out_cfg = OutputConfig()
    out_cfg.root.mkdir(parents=True, exist_ok=True)
    baseline_cfg = BaselineConfig()
    requested_device = os.environ.get("EEG_ATTACK_DEVICE")

    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device=requested_device)
    score_fn = make_score_fn(model, device)
    bundle = load_moabb_windows(baseline_cfg)

    candidate_payloads = []
    for idx in range(len(bundle.valid_set)):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        if int(np.argmax(score_fn(x_np))) == y_int:
            candidate_payloads.append((idx, x_np, y_int))

    cpu_count = os.cpu_count() or 1
    n_workers = max(1, min(8, cpu_count // 2 if cpu_count > 1 else 1))
    env_workers = os.environ.get("EEG_ATTACK_WORKERS")
    if env_workers is not None:
        n_workers = max(1, int(env_workers))
    worker_device = str(device) if n_workers == 1 else "cpu"

    print(
        f"Comparing attacks on {len(candidate_payloads)} clean-correct validation trials "
        f"(score_device={device}, worker_device={worker_device}, workers={n_workers})",
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
        "score_device": str(device),
        "worker_device": worker_device,
        "n_workers": n_workers,
        "n_clean_correct_attacked": len(candidate_payloads),
        "variants": variant_reports,
    }

    power_plot_path = out_cfg.root / "attack_basis_power_ratio_comparison.png"
    channel_budget_plot_path = out_cfg.root / "attack_basis_zero_accuracy_channel_budget.png"

    _plot_power_ratio_comparison(comparison, power_plot_path)
    _plot_zero_accuracy_channel_budget(comparison, channel_budget_plot_path)

    successful_idx_sets = [{int(row["idx"]) for row in variant["per_sample"] if row["success"]} for variant in comparison["variants"]]
    common_success = set.intersection(*successful_idx_sets) if successful_idx_sets else set()
    example_sample_idx = None if not common_success else min(common_success)

    variant_examples = []
    for variant, report in zip(variants, comparison["variants"]):
        if example_sample_idx is not None:
            sample_idx = int(example_sample_idx)
        else:
            sample_idx = next(int(row["idx"]) for row in report["per_sample"] if row["success"])
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

    waveform_plot_path = out_cfg.root / "attack_basis_waveform_example.png"
    _plot_waveform_example(
        baseline_cfg=baseline_cfg,
        variant_examples=variant_examples,
        out_path=waveform_plot_path,
    )

    comparison["comparison_plot_path"] = str(power_plot_path)
    comparison["channel_budget_plot_path"] = str(channel_budget_plot_path)
    comparison["waveform_plot_path"] = str(waveform_plot_path)
    comparison["example_sample_idx"] = None if example_sample_idx is None else int(example_sample_idx)

    report_path = out_cfg.root / "attack_basis_comparison_report.json"
    report_path.write_text(json.dumps(comparison, indent=2))
    comparison["report_path"] = str(report_path)
    return comparison


if __name__ == "__main__":
    summary = run_attack_basis_comparison()
    compact = {
        "report_path": summary["report_path"],
        "comparison_plot_path": summary["comparison_plot_path"],
        "channel_budget_plot_path": summary["channel_budget_plot_path"],
        "waveform_plot_path": summary["waveform_plot_path"],
        "variants": [
            {
                "name": variant["variant_name"],
                "k_zero_accuracy": variant["k_zero_accuracy"],
                "power_ratio_zero_accuracy_pct": variant["power_ratio_zero_accuracy_pct"],
                "attacked_accuracy": variant["attacked_accuracy"],
            }
            for variant in summary["variants"]
        ],
    }
    print(compact)
