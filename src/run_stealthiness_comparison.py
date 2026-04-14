"""Stealthiness comparison: our greedy black-box attack vs Filter_Attack spatial filter.

Runs both attacks on the same N correctly-classified test samples and prints a
side-by-side table of:
  - ASR
  - delta L2 / Linf
  - covariance Frobenius distance  (spatial-filter attacks score badly here)
  - channel sparsity               (fraction of channels touched)
  - PSD deviation (dB)             (mean per-channel spectral change)

Usage:
    python -m src.run_stealthiness_comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .attack.greedy_attack import build_score_attack
from .attack.losses import untargeted_margin
from .attack.spatial_filter_baseline import SpatialFilterBaseline
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn
from .stealthiness import channel_sparsity, covariance_frob_distance, psd_deviation


# ---------------------------------------------------------------------------
# Attack configs to use for our greedy attacker (pick one representative)
# ---------------------------------------------------------------------------
_OUR_CFG = AttackConfig(
    support_mode="channel_first",
    basis_mode="hybrid",
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
)


def _collect_candidates(bundle, score_fn, pool_size: int, n_samples: int) -> list[dict]:
    candidates = []
    for idx in range(min(pool_size, len(bundle.valid_set))):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        scores = score_fn(x_np)
        pred = int(np.argmax(scores))
        if pred != int(y):
            continue
        margin = untargeted_margin(scores, int(y))
        candidates.append({"idx": idx, "true_label": int(y), "clean_margin": margin})
    candidates.sort(key=lambda r: r["clean_margin"])
    return candidates[:n_samples]


def _run_our_attack(row, bundle, score_fn, baseline_cfg) -> dict:
    idx = row["idx"]
    x, y, _ = bundle.valid_set[idx]
    x_np = x.astype(np.float32)
    y_int = int(y)

    attack = build_score_attack(
        score_fn=score_fn,
        sfreq=baseline_cfg.sfreq,
        n_windows=_OUR_CFG.n_windows,
        support_budget_k=_OUR_CFG.support_budget_k,
        basis_rank_r=_OUR_CFG.basis_rank_r,
        basis_min_hz=_OUR_CFG.basis_min_hz,
        basis_max_hz=_OUR_CFG.basis_max_hz,
        basis_mode=_OUR_CFG.basis_mode,
        basis_phase_count=_OUR_CFG.basis_phase_count,
        candidate_probe_restarts=_OUR_CFG.candidate_probe_restarts,
        candidate_probe_scale=_OUR_CFG.candidate_probe_scale,
        max_outer_iters=_OUR_CFG.max_outer_iters,
        max_query_budget=_OUR_CFG.max_query_budget,
        spsa_steps=_OUR_CFG.spsa_steps,
        spsa_step_size=_OUR_CFG.spsa_step_size,
        spsa_perturb_scale=_OUR_CFG.spsa_perturb_scale,
        spsa_restarts=_OUR_CFG.spsa_restarts,
        spsa_init_scale=_OUR_CFG.spsa_init_scale,
        l2_weight=_OUR_CFG.l2_weight,
        tv_weight=_OUR_CFG.tv_weight,
        band_weight=_OUR_CFG.band_weight,
        max_coeff_abs=_OUR_CFG.max_coeff_abs,
        max_perturbation_peak_ratio=_OUR_CFG.max_perturbation_peak_ratio,
        support_mode=_OUR_CFG.support_mode,
        channel_waveform_rank=_OUR_CFG.channel_waveform_rank,
        channel_shortlist_size=_OUR_CFG.channel_shortlist_size,
        enforce_unique_channels=_OUR_CFG.enforce_unique_channels,
        stop_on_success=_OUR_CFG.stop_on_success,
        seed=baseline_cfg.random_seed + idx,
    )
    result = attack.run(x_np, y_int)

    return {
        "idx": idx,
        "success": result.success,
        "delta_l2": float(np.linalg.norm(result.delta)),
        "delta_linf": float(np.max(np.abs(result.delta))),
        "cov_frob": covariance_frob_distance(x_np, result.x_adv),
        "channel_sparsity": channel_sparsity(result.delta),
        "psd_deviation": psd_deviation(x_np, result.x_adv, baseline_cfg.sfreq),
        "queries_used": result.queries_used,
    }


def _run_filter_attack(row, bundle, model, device, baseline_cfg) -> dict:
    idx = row["idx"]
    x, y, _ = bundle.valid_set[idx]
    x_np = x.astype(np.float32)
    y_int = int(y)

    attacker = SpatialFilterBaseline(
        model=model,
        device=device,
        sfreq=baseline_cfg.sfreq,
        n_steps=200,
        lr=5e-3,
        alpha=1e1,
    )
    result = attacker.run(x_np, y_int)

    return {
        "idx": idx,
        "success": result.success,
        "delta_l2": float(np.linalg.norm(result.delta)),
        "delta_linf": float(np.max(np.abs(result.delta))),
        "cov_frob": covariance_frob_distance(x_np, result.x_adv),
        "channel_sparsity": channel_sparsity(result.delta),
        "psd_deviation": psd_deviation(x_np, result.x_adv, baseline_cfg.sfreq),
        "queries_used": result.queries_used,
    }


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")


def _asr(rows: list[dict]) -> float:
    return float(np.mean([r["success"] for r in rows])) if rows else float("nan")


def _print_table(our_rows: list[dict], fa_rows: list[dict]) -> None:
    metrics = [
        ("ASR",              lambda rows: f"{_asr(rows):.2%}"),
        ("delta L2",         lambda rows: f"{_mean(rows, 'delta_l2'):.4f}"),
        ("delta Linf",       lambda rows: f"{_mean(rows, 'delta_linf'):.4f}"),
        ("cov_frob ↓",       lambda rows: f"{_mean(rows, 'cov_frob'):.4f}"),
        ("ch_sparsity ↓",    lambda rows: f"{_mean(rows, 'channel_sparsity'):.2%}"),
        ("psd_dev (dB) ↓",   lambda rows: f"{_mean(rows, 'psd_deviation'):.4f}"),
        ("queries",          lambda rows: f"{_mean(rows, 'queries_used'):.0f}"),
    ]

    col_w = 20
    header = f"{'Metric':<20}{'Ours (greedy BB)':>{col_w}}{'Filter_Attack (WB)':>{col_w}}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, fn in metrics:
        print(f"{name:<20}{fn(our_rows):>{col_w}}{fn(fa_rows):>{col_w}}")
    print(sep)
    print("↓ = lower is stealthier")
    print()


def run_comparison(n_samples: int = 10, pool_size: int = 64) -> dict:
    out_cfg = OutputConfig()
    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, device)

    baseline_cfg = BaselineConfig()
    bundle = load_moabb_windows(baseline_cfg)

    print(f"Collecting up to {n_samples} correctly-classified candidates...")
    candidates = _collect_candidates(bundle, score_fn, pool_size, n_samples)
    print(f"  → {len(candidates)} candidates found\n")

    our_rows, fa_rows = [], []
    for i, row in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] sample idx={row['idx']}  label={row['true_label']}")

        print("  Running our greedy attack...")
        our_rows.append(_run_our_attack(row, bundle, score_fn, baseline_cfg))
        print(f"    success={our_rows[-1]['success']}  cov_frob={our_rows[-1]['cov_frob']:.4f}")

        print("  Running Filter_Attack spatial filter...")
        fa_rows.append(_run_filter_attack(row, bundle, model, device, baseline_cfg))
        print(f"    success={fa_rows[-1]['success']}  cov_frob={fa_rows[-1]['cov_frob']:.4f}")

    _print_table(our_rows, fa_rows)

    result = {
        "n_samples": len(candidates),
        "ours": our_rows,
        "filter_attack": fa_rows,
        "summary": {
            "ours": {
                "asr": _asr(our_rows),
                "avg_delta_l2": _mean(our_rows, "delta_l2"),
                "avg_delta_linf": _mean(our_rows, "delta_linf"),
                "avg_cov_frob": _mean(our_rows, "cov_frob"),
                "avg_channel_sparsity": _mean(our_rows, "channel_sparsity"),
                "avg_psd_deviation": _mean(our_rows, "psd_deviation"),
            },
            "filter_attack": {
                "asr": _asr(fa_rows),
                "avg_delta_l2": _mean(fa_rows, "delta_l2"),
                "avg_delta_linf": _mean(fa_rows, "delta_linf"),
                "avg_cov_frob": _mean(fa_rows, "cov_frob"),
                "avg_channel_sparsity": _mean(fa_rows, "channel_sparsity"),
                "avg_psd_deviation": _mean(fa_rows, "psd_deviation"),
            },
        },
    }

    out_path = Path(out_cfg.root) / "stealthiness_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")

    return result


if __name__ == "__main__":
    run_comparison(n_samples=10, pool_size=64)
