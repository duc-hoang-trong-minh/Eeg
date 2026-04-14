from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attack.greedy_attack import build_score_attack
from .attack.losses import untargeted_margin
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .defense.lightweight import (
    bandpass_filter_defense,
    flag_suspicious_atoms_from_signal,
    localized_denoise,
    suppress_flagged_atoms,
)
from .model_oracle import load_eegnet_checkpoint, make_score_fn


def _build_attack_from_config(score_fn, baseline_cfg: BaselineConfig, cfg: AttackConfig, seed: int):
    return build_score_attack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=cfg.n_windows,
        support_budget_k=cfg.support_budget_k,
        basis_rank_r=cfg.basis_rank_r,
        basis_min_hz=cfg.basis_min_hz,
        basis_max_hz=cfg.basis_max_hz,
        basis_mode=cfg.basis_mode,
        basis_phase_count=cfg.basis_phase_count,
        candidate_probe_restarts=cfg.candidate_probe_restarts,
        candidate_probe_scale=cfg.candidate_probe_scale,
        max_outer_iters=cfg.max_outer_iters,
        max_query_budget=cfg.max_query_budget,
        spsa_steps=cfg.spsa_steps,
        spsa_step_size=cfg.spsa_step_size,
        spsa_perturb_scale=cfg.spsa_perturb_scale,
        spsa_restarts=cfg.spsa_restarts,
        spsa_init_scale=cfg.spsa_init_scale,
        l2_weight=cfg.l2_weight,
        tv_weight=cfg.tv_weight,
        band_weight=cfg.band_weight,
        max_coeff_abs=cfg.max_coeff_abs,
        max_perturbation_peak_ratio=cfg.max_perturbation_peak_ratio,
        support_mode=cfg.support_mode,
        channel_waveform_rank=cfg.channel_waveform_rank,
        channel_shortlist_size=cfg.channel_shortlist_size,
        enforce_unique_channels=cfg.enforce_unique_channels,
        stop_on_success=cfg.stop_on_success,
        seed=seed,
    )


def _build_eval_configs(mode: str) -> list[tuple[str, AttackConfig]]:
    presets = {
        "default": [
            (
                "cf_k2_q6000_r16_a0.50",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=2,
                    basis_rank_r=4,
                    channel_waveform_rank=16,
                    max_outer_iters=2,
                    max_query_budget=6000,
                    spsa_steps=100,
                    spsa_step_size=0.06,
                    spsa_perturb_scale=0.03,
                    spsa_restarts=2,
                    spsa_init_scale=0.20,
                    max_coeff_abs=0.50,
                    candidate_probe_restarts=3,
                    candidate_probe_scale=0.75,
                    l2_weight=1e-4,
                    tv_weight=1e-4,
                    band_weight=1e-4,
                ),
            ),
            (
                "cf_k4_q12000_r24_a0.75",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=4,
                    basis_rank_r=4,
                    channel_waveform_rank=24,
                    max_outer_iters=4,
                    max_query_budget=12000,
                    spsa_steps=140,
                    spsa_step_size=0.08,
                    spsa_perturb_scale=0.04,
                    spsa_restarts=3,
                    spsa_init_scale=0.25,
                    max_coeff_abs=0.75,
                    candidate_probe_restarts=3,
                    candidate_probe_scale=0.75,
                    l2_weight=1e-4,
                    tv_weight=1e-4,
                    band_weight=1e-4,
                ),
            ),
        ],
        "aggressive": [
            (
                "cf_k2_q8000_r16_a0.50",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=2,
                    basis_rank_r=4,
                    channel_waveform_rank=16,
                    max_outer_iters=2,
                    max_query_budget=8000,
                    spsa_steps=100,
                    spsa_step_size=0.06,
                    spsa_perturb_scale=0.03,
                    spsa_restarts=2,
                    spsa_init_scale=0.20,
                    max_coeff_abs=0.50,
                    candidate_probe_restarts=3,
                    candidate_probe_scale=0.75,
                    l2_weight=1e-4,
                    tv_weight=1e-4,
                    band_weight=1e-4,
                ),
            ),
            (
                "cf_k3_q12000_r24_a0.75",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=3,
                    basis_rank_r=4,
                    channel_waveform_rank=24,
                    max_outer_iters=3,
                    max_query_budget=12000,
                    spsa_steps=140,
                    spsa_step_size=0.10,
                    spsa_perturb_scale=0.05,
                    spsa_restarts=3,
                    spsa_init_scale=0.25,
                    max_coeff_abs=0.75,
                    candidate_probe_restarts=4,
                    candidate_probe_scale=0.80,
                    l2_weight=1e-4,
                    tv_weight=1e-4,
                    band_weight=1e-4,
                ),
            ),
            (
                "cf_k4_q16000_r32_a0.75",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=4,
                    basis_rank_r=6,
                    channel_waveform_rank=32,
                    max_outer_iters=4,
                    max_query_budget=16000,
                    spsa_steps=160,
                    spsa_step_size=0.10,
                    spsa_perturb_scale=0.05,
                    spsa_restarts=3,
                    spsa_init_scale=0.30,
                    max_coeff_abs=0.75,
                    candidate_probe_restarts=4,
                    candidate_probe_scale=0.80,
                    l2_weight=5e-5,
                    tv_weight=5e-5,
                    band_weight=5e-5,
                ),
            ),
            (
                "cf_k5_q20000_r32_a1.00",
                AttackConfig(
                    support_mode="channel_first",
                    n_windows=8,
                    support_budget_k=5,
                    basis_rank_r=6,
                    channel_waveform_rank=32,
                    max_outer_iters=5,
                    max_query_budget=20000,
                    spsa_steps=180,
                    spsa_step_size=0.12,
                    spsa_perturb_scale=0.06,
                    spsa_restarts=4,
                    spsa_init_scale=0.35,
                    max_coeff_abs=1.00,
                    candidate_probe_restarts=4,
                    candidate_probe_scale=0.90,
                    l2_weight=5e-5,
                    tv_weight=5e-5,
                    band_weight=5e-5,
                ),
            ),
        ],
        "freq_bank": [
            (
                "fb_w8_k4_q8000_b8",
                AttackConfig(
                    support_mode="channel_window_freq_bank",
                    basis_mode="freq_atom_bank",
                    basis_phase_count=2,
                    n_windows=8,
                    support_budget_k=4,
                    basis_rank_r=8,
                    max_outer_iters=4,
                    max_query_budget=8000,
                    spsa_steps=120,
                    spsa_step_size=0.08,
                    spsa_perturb_scale=0.04,
                    spsa_restarts=3,
                    spsa_init_scale=0.20,
                    max_coeff_abs=0.75,
                    candidate_probe_restarts=3,
                    candidate_probe_scale=0.75,
                    l2_weight=1e-4,
                    tv_weight=1e-4,
                    band_weight=1e-4,
                ),
            ),
            (
                "fb_w8_k6_q12000_b12",
                AttackConfig(
                    support_mode="channel_window_freq_bank",
                    basis_mode="freq_atom_bank",
                    basis_phase_count=3,
                    n_windows=8,
                    support_budget_k=6,
                    basis_rank_r=12,
                    max_outer_iters=6,
                    max_query_budget=12000,
                    spsa_steps=140,
                    spsa_step_size=0.10,
                    spsa_perturb_scale=0.05,
                    spsa_restarts=3,
                    spsa_init_scale=0.25,
                    max_coeff_abs=0.90,
                    candidate_probe_restarts=4,
                    candidate_probe_scale=0.80,
                    l2_weight=5e-5,
                    tv_weight=5e-5,
                    band_weight=5e-5,
                ),
            ),
            (
                "fb_w12_k6_q16000_b12",
                AttackConfig(
                    support_mode="channel_window_freq_bank",
                    basis_mode="freq_atom_bank",
                    basis_phase_count=3,
                    n_windows=12,
                    support_budget_k=6,
                    basis_rank_r=12,
                    max_outer_iters=6,
                    max_query_budget=16000,
                    spsa_steps=160,
                    spsa_step_size=0.10,
                    spsa_perturb_scale=0.05,
                    spsa_restarts=4,
                    spsa_init_scale=0.30,
                    max_coeff_abs=1.00,
                    candidate_probe_restarts=4,
                    candidate_probe_scale=0.85,
                    l2_weight=5e-5,
                    tv_weight=5e-5,
                    band_weight=5e-5,
                ),
            ),
        ],
    }
    if mode not in presets:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    return presets[mode]


def _collect_clean_candidates(bundle, score_fn, sample_pool_size: int | None) -> list[dict]:
    n_pool = len(bundle.valid_set) if sample_pool_size is None else min(sample_pool_size, len(bundle.valid_set))
    candidates = []
    for idx in range(n_pool):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        y_int = int(y)
        clean_scores = score_fn(x_np)
        clean_pred = int(np.argmax(clean_scores))
        clean_margin = float(untargeted_margin(clean_scores, y_int))
        if clean_pred != y_int:
            continue
        candidates.append(
            {
                "idx": idx,
                "true_label": y_int,
                "clean_pred": clean_pred,
                "clean_margin": clean_margin,
            }
        )
    candidates.sort(key=lambda row: row["clean_margin"])
    return candidates


def _evaluate_single_config(
    config_name: str,
    cfg: AttackConfig,
    candidate_rows: list[dict],
    bundle,
    score_fn,
    baseline_cfg: BaselineConfig,
) -> dict:
    per_sample = []
    n_success = 0
    n_success_after_denoise = 0
    n_success_after_filter = 0
    n_success_after_bandpass = 0
    n_budget_exhausted = 0
    margins = []
    margin_deltas = []
    queries = []
    flagged_counts = []

    for offset, row in enumerate(candidate_rows):
        idx = int(row["idx"])
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        y_int = int(y)

        attack = _build_attack_from_config(
            score_fn=score_fn,
            baseline_cfg=baseline_cfg,
            cfg=cfg,
            seed=baseline_cfg.random_seed + idx,
        )
        result = attack.run(x_np, y_int)
        adv_scores = score_fn(result.x_adv)
        adv_pred = int(np.argmax(adv_scores))

        denoised_pred = int(np.argmax(score_fn(localized_denoise(result.x_adv))))
        flagged_atoms = flag_suspicious_atoms_from_signal(
            result.x_adv,
            n_windows=cfg.n_windows,
        )
        filtered_x = suppress_flagged_atoms(
            result.x_adv,
            flagged_atoms=flagged_atoms,
            n_windows=cfg.n_windows,
        )
        filtered_pred = int(np.argmax(score_fn(filtered_x)))

        sample_row = {
            "rank": offset,
            "idx": idx,
            "true_label": y_int,
            "clean_pred": int(row["clean_pred"]),
            "clean_margin": float(row["clean_margin"]),
            "adv_pred": adv_pred,
            "success": bool(adv_pred != y_int),
            "final_margin": float(result.margin),
            "margin_delta": float(row["clean_margin"] - result.margin),
            "queries_used": int(result.queries_used),
            "budget_exhausted": bool(result.budget_exhausted),
            "support": result.support,
            "delta_l2": float(np.linalg.norm(result.delta.reshape(-1))),
            "delta_linf": float(np.max(np.abs(result.delta))),
            "post_denoise_success": bool(denoised_pred != y_int),
            "post_suspicious_filter_success": bool(filtered_pred != y_int),
            "flagged_atoms": len(flagged_atoms),
        }
        per_sample.append(sample_row)

        n_success += int(sample_row["success"])
        n_success_after_denoise += int(sample_row["post_denoise_success"])
        n_success_after_filter += int(sample_row["post_suspicious_filter_success"])
        n_budget_exhausted += int(sample_row["budget_exhausted"])
        margins.append(sample_row["final_margin"])
        margin_deltas.append(sample_row["margin_delta"])
        queries.append(sample_row["queries_used"])
        flagged_counts.append(sample_row["flagged_atoms"])

    denom = max(len(candidate_rows), 1)
    best_attack = None
    if per_sample:
        best_attack = min(
            per_sample,
            key=lambda row: (
                not row["success"],
                row["final_margin"],
                row["queries_used"],
            ),
        )

    return {
        "config_name": config_name,
        "config": cfg.__dict__,
        "n_clean_correct_attacked": len(candidate_rows),
        "attack_success_rate": n_success / denom,
        "post_denoise_attack_success_rate": n_success_after_denoise / denom,
        "post_suspicious_filter_attack_success_rate": n_success_after_filter / denom,
        "budget_exhaustion_rate": n_budget_exhausted / denom,
        "avg_final_margin": float(np.mean(margins)) if margins else 0.0,
        "avg_margin_reduction": float(np.mean(margin_deltas)) if margin_deltas else 0.0,
        "avg_queries": float(np.mean(queries)) if queries else 0.0,
        "avg_flagged_atoms": float(np.mean(flagged_counts)) if flagged_counts else 0.0,
        "best_attack": best_attack,
        "per_sample": per_sample,
    }


def run_eval(
    n_samples: int = 8,
    mode: str = "aggressive",
    sample_pool_size: int | None = 64,
) -> dict:
    out_cfg = OutputConfig()
    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, device)

    baseline_cfg = BaselineConfig()
    bundle = load_moabb_windows(baseline_cfg)
    clean_candidates = _collect_clean_candidates(
        bundle=bundle,
        score_fn=score_fn,
        sample_pool_size=sample_pool_size,
    )
    selected_candidates = clean_candidates[: min(n_samples, len(clean_candidates))]

    rows = []
    for config_name, cfg in _build_eval_configs(mode):
        rows.append(
            _evaluate_single_config(
                config_name=config_name,
                cfg=cfg,
                candidate_rows=selected_candidates,
                bundle=bundle,
                score_fn=score_fn,
                baseline_cfg=baseline_cfg,
            )
        )

    best_config = None
    if rows:
        best_config = max(
            rows,
            key=lambda row: (
                row["attack_success_rate"],
                row["avg_margin_reduction"],
                -row["avg_final_margin"],
            ),
        )

    return {
        "mode": mode,
        "n_eval_samples_requested": n_samples,
        "sample_pool_size": sample_pool_size,
        "n_clean_correct_candidates": len(clean_candidates),
        "attacked_sample_indices": [int(row["idx"]) for row in selected_candidates],
        "attacked_clean_margins": [float(row["clean_margin"]) for row in selected_candidates],
        "results": rows,
        "best_config": best_config,
    }


if __name__ == "__main__":
    report = run_eval(n_samples=8, mode="freq_bank", sample_pool_size=64)
    out_cfg = OutputConfig()
    out_path = Path(out_cfg.root) / "attack_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(report)
    print(f"Saved: {out_path}")
