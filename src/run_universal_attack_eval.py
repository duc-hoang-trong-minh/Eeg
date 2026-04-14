"""Evaluate trained AttackGenerator against sample-wise SPSA baseline.

Loads a trained generator checkpoint, applies it to test trials (zero model queries),
and prints a comparison table:

    Metric            | sample-wise SPSA | subject-wise G | model-wise G
    ------------------|-----------------|----------------|-------------
    ASR               | ...             | ...            | ...
    delta L2          | ...             | ...            | ...
    ...

Usage:
    python -m src.run_universal_attack_eval
    python -m src.run_universal_attack_eval --n_samples 20 --scope subject
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attack.generator import AttackGenerator
from .attack.greedy_attack import build_score_attack
from .attack.losses import untargeted_margin
from .config import AttackConfig, BaselineConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn
from .stealthiness import channel_sparsity, covariance_frob_distance, psd_deviation


_SPSA_CFG = AttackConfig(
    support_mode="channel_window",
    basis_mode="hybrid",
    n_windows=8,
    support_budget_k=5,
    basis_rank_r=4,
    basis_min_hz=2.0,
    basis_max_hz=30.0,
    max_outer_iters=4,
    max_query_budget=10000,
    spsa_steps=120,
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
    stop_on_success=True,
)


def _load_generator(ckpt_path: Path, device: torch.device) -> tuple[AttackGenerator, list, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = AttackGenerator(
        n_channels=ckpt["n_channels"],
        n_times=ckpt["n_times"],
        support_budget_k=ckpt["support_budget_k"],
        basis_rank_r=ckpt["basis_rank_r"],
        max_coeff_abs=ckpt["max_coeff_abs"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    universal_support = [tuple(a) for a in ckpt["universal_support"]]
    gen_cfg = ckpt.get("gen_cfg", {})
    return model, universal_support, gen_cfg


def _stealthiness_row(x_np: np.ndarray, delta: np.ndarray, sfreq: float, success: bool, queries: int) -> dict:
    x_adv = x_np + delta
    return {
        "success": success,
        "delta_l2": float(np.linalg.norm(delta)),
        "delta_linf": float(np.max(np.abs(delta))),
        "cov_frob": covariance_frob_distance(x_np, x_adv),
        "channel_sparsity": channel_sparsity(delta),
        "psd_deviation": psd_deviation(x_np, x_adv, sfreq),
        "queries": queries,
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _asr(rows: list[dict]) -> float:
    return float(np.mean([r["success"] for r in rows])) if rows else float("nan")


def _print_table(columns: list[tuple[str, list[dict]]]) -> None:
    metrics = [
        ("ASR",            lambda rows: f"{_asr(rows):.2%}"),
        ("delta L2",       lambda rows: f"{_mean(rows, 'delta_l2'):.4f}"),
        ("delta Linf",     lambda rows: f"{_mean(rows, 'delta_linf'):.4f}"),
        ("cov_frob ↓",     lambda rows: f"{_mean(rows, 'cov_frob'):.4f}"),
        ("ch_sparsity ↓",  lambda rows: f"{_mean(rows, 'channel_sparsity'):.2%}"),
        ("psd_dev(dB) ↓",  lambda rows: f"{_mean(rows, 'psd_deviation'):.4f}"),
        ("queries",        lambda rows: f"{_mean(rows, 'queries'):.0f}"),
    ]
    col_w = 18
    header = f"{'Metric':<20}" + "".join(f"{name:>{col_w}}" for name, _ in columns)
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for m_name, m_fn in metrics:
        row = f"{m_name:<20}" + "".join(f"{m_fn(rows):>{col_w}}" for _, rows in columns)
        print(row)
    print(sep)
    print("↓ = lower is stealthier  |  queries=0 means generator (no model access at test time)")
    print()


def evaluate(n_samples: int = 10, pool_size: int = 64, scope: str = "subject") -> None:
    out_cfg = OutputConfig()
    baseline_cfg = BaselineConfig()
    out_dir = Path(out_cfg.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, torch_device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, torch_device)
    bundle = load_moabb_windows(baseline_cfg)

    # collect correctly-classified candidates
    candidates = []
    for idx in range(min(pool_size, len(bundle.valid_set))):
        x, y, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        scores = score_fn(x_np)
        if int(np.argmax(scores)) != int(y):
            continue
        candidates.append({"idx": idx, "y": int(y), "margin": untargeted_margin(scores, int(y))})
    candidates.sort(key=lambda r: r["margin"])
    candidates = candidates[:n_samples]
    print(f"Evaluating {len(candidates)} samples  (pool_size={pool_size})")

    # find generator checkpoints
    gen_ckpts: list[tuple[str, Path]] = []
    if scope in ("subject", "both"):
        for p in sorted(out_dir.glob("attack_generator_subject_*.pt")):
            subject = p.stem.replace("attack_generator_subject_", "")
            gen_ckpts.append((f"subject-wise G (s={subject})", p))
    if scope in ("model", "both"):
        mw_path = out_dir / "attack_generator_model_wise.pt"
        if mw_path.exists():
            gen_ckpts.append(("model-wise G", mw_path))

    if not gen_ckpts:
        print("No generator checkpoints found. Run train_attack_generator first.")
        return

    # --- sample-wise SPSA baseline ---
    spsa_rows = []
    print("\nRunning sample-wise SPSA attack (baseline)...")
    for i, cand in enumerate(candidates):
        idx, y_int = cand["idx"], cand["y"]
        x, _, _ = bundle.valid_set[idx]
        x_np = x.astype(np.float32)
        cfg = _SPSA_CFG
        attack = build_score_attack(
            score_fn=score_fn, sfreq=baseline_cfg.sfreq,
            n_windows=cfg.n_windows, support_budget_k=cfg.support_budget_k,
            basis_rank_r=cfg.basis_rank_r, basis_min_hz=cfg.basis_min_hz,
            basis_max_hz=cfg.basis_max_hz, basis_mode=cfg.basis_mode,
            basis_phase_count=cfg.basis_phase_count,
            candidate_probe_restarts=cfg.candidate_probe_restarts,
            candidate_probe_scale=cfg.candidate_probe_scale,
            max_outer_iters=cfg.max_outer_iters, max_query_budget=cfg.max_query_budget,
            spsa_steps=cfg.spsa_steps, spsa_step_size=cfg.spsa_step_size,
            spsa_perturb_scale=cfg.spsa_perturb_scale, spsa_restarts=cfg.spsa_restarts,
            spsa_init_scale=cfg.spsa_init_scale, l2_weight=cfg.l2_weight,
            tv_weight=cfg.tv_weight, band_weight=cfg.band_weight,
            max_coeff_abs=cfg.max_coeff_abs,
            max_perturbation_peak_ratio=cfg.max_perturbation_peak_ratio,
            support_mode=cfg.support_mode,
            enforce_unique_channels=cfg.enforce_unique_channels,
            stop_on_success=cfg.stop_on_success,
            seed=baseline_cfg.random_seed + idx,
        )
        result = attack.run(x_np, y_int)
        spsa_rows.append(_stealthiness_row(x_np, result.delta, baseline_cfg.sfreq,
                                           result.success, result.queries_used))
        print(f"  [{i+1}/{len(candidates)}] success={result.success} queries={result.queries_used}")

    # --- generator attacks ---
    gen_columns: list[tuple[str, list[dict]]] = []
    for label, ckpt_path in gen_ckpts:
        print(f"\nRunning {label} ...")
        gen_model, universal_support, gen_cfg_dict = _load_generator(ckpt_path, device)
        rows = []
        for i, cand in enumerate(candidates):
            idx, y_int = cand["idx"], cand["y"]
            x, _, _ = bundle.valid_set[idx]
            x_np = x.astype(np.float32)

            delta = gen_model.generate(
                x_np=x_np,
                universal_support=universal_support,
                n_windows=gen_cfg_dict.get("n_windows", 8),
                basis_min_hz=gen_cfg_dict.get("basis_min_hz", 2.0),
                basis_max_hz=gen_cfg_dict.get("basis_max_hz", 30.0),
                basis_mode=gen_cfg_dict.get("basis_mode", "hybrid"),
                basis_phase_count=gen_cfg_dict.get("basis_phase_count", 2),
                sfreq=baseline_cfg.sfreq,
                device=device,
            )
            x_adv = x_np + delta
            adv_scores = score_fn(x_adv)
            success = int(np.argmax(adv_scores)) != y_int
            rows.append(_stealthiness_row(x_np, delta, baseline_cfg.sfreq, success, queries=0))
            print(f"  [{i+1}/{len(candidates)}] success={success}")
        gen_columns.append((label, rows))

    columns = [("sample-wise SPSA", spsa_rows)] + gen_columns
    _print_table(columns)

    # save
    out_path = out_dir / "universal_attack_eval.json"
    result_data = {
        "n_samples": len(candidates),
        "columns": [
            {"name": name, "rows": rows} for name, rows in columns
        ],
    }
    with out_path.open("w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=64)
    parser.add_argument("--scope", choices=["subject", "model", "both"], default="both")
    args = parser.parse_args()
    evaluate(n_samples=args.n_samples, pool_size=args.pool_size, scope=args.scope)
