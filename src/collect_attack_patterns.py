"""Collect sample-wise attack patterns for generator training.

Runs the greedy black-box SPSA attack on every correctly-classified trial in
valid_set (per subject) and saves the resulting (x, delta, support, success)
dataset to disk.  This is the expensive offline phase — run once.

Output per subject S:
    outputs/attack_patterns_subject_{S}.npz
        X          (N, C, T)  — original EEG trials
        Delta      (N, C, T)  — perturbation deltas (zeros for failed attacks)
        success    (N,)       — bool: attack succeeded
        margin     (N,)       — final adversarial margin
        supports   (N, K)     — selected atom indices into all_atoms list (-1 = unused)

Usage:
    python -m src.collect_attack_patterns
    python -m src.collect_attack_patterns --max_per_subject 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .attack.greedy_attack import build_score_attack
from .attack.losses import untargeted_margin
from .attack.support import all_atoms, make_window_partition
from .config import AttackConfig, BaselineConfig, GeneratorConfig, OutputConfig
from .data import load_moabb_windows
from .model_oracle import load_eegnet_checkpoint, make_score_fn


_COLLECTION_CFG = AttackConfig(
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


def _atom_index(atom: tuple[int, int], n_channels: int, n_windows: int) -> int:
    c, w = atom
    return c * n_windows + w


def _collect_subject(
    subject: object,
    subject_indices: list[int],
    bundle,
    score_fn,
    baseline_cfg: BaselineConfig,
    gen_cfg: GeneratorConfig,
    max_per_subject: int | None,
    seed_offset: int,
) -> dict:
    """Run attacks on all correctly-classified trials for one subject."""
    cfg = _COLLECTION_CFG
    n_windows = gen_cfg.n_windows

    xs, deltas, successes, margins, supports_list = [], [], [], [], []

    limit = max_per_subject if max_per_subject is not None else len(subject_indices)
    processed = 0

    for pos, local_idx in enumerate(subject_indices):
        if processed >= limit:
            break

        x, y, _ = bundle.valid_set[local_idx]
        x_np = x.astype(np.float32)
        y_int = int(y)

        # skip already-misclassified
        scores = score_fn(x_np)
        if int(np.argmax(scores)) != y_int:
            continue

        attack = build_score_attack(
            score_fn=score_fn,
            sfreq=baseline_cfg.sfreq,
            n_windows=n_windows,
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
            enforce_unique_channels=cfg.enforce_unique_channels,
            stop_on_success=cfg.stop_on_success,
            seed=baseline_cfg.random_seed + seed_offset + pos,
        )
        result = attack.run(x_np, y_int)

        n_channels = x_np.shape[0]
        # encode support as flat atom indices (pad with -1 to fixed length K)
        atom_indices = np.full(cfg.support_budget_k, -1, dtype=np.int32)
        for i, atom in enumerate(result.support[: cfg.support_budget_k]):
            atom_indices[i] = _atom_index(atom, n_channels, n_windows)

        xs.append(x_np)
        deltas.append(result.delta)
        successes.append(result.success)
        margins.append(result.margin)
        supports_list.append(atom_indices)
        processed += 1

        status = "✓" if result.success else "✗"
        print(
            f"  [{processed}/{limit}] idx={local_idx} label={y_int} "
            f"{status} margin={result.margin:.3f} queries={result.queries_used}"
        )

    if not xs:
        return {}

    return {
        "X": np.stack(xs, axis=0),
        "Delta": np.stack(deltas, axis=0),
        "success": np.array(successes, dtype=bool),
        "margin": np.array(margins, dtype=np.float32),
        "supports": np.stack(supports_list, axis=0),
    }


def collect_patterns(max_per_subject: int | None = None) -> None:
    out_cfg = OutputConfig()
    baseline_cfg = BaselineConfig()
    gen_cfg = GeneratorConfig()

    model, device, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path))
    score_fn = make_score_fn(model, device)
    bundle = load_moabb_windows(baseline_cfg)

    if bundle.valid_subjects is None:
        raise RuntimeError("DatasetBundle.valid_subjects is None — subject metadata missing.")

    unique_subjects = np.unique(bundle.valid_subjects)
    print(f"Subjects: {unique_subjects.tolist()}  |  total valid trials: {len(bundle.valid_set)}")

    out_dir = Path(out_cfg.root)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for sid, subject in enumerate(unique_subjects):
        subject_mask = bundle.valid_subjects == subject
        subject_local_indices = np.where(subject_mask)[0].tolist()
        print(f"\nSubject {subject}  ({len(subject_local_indices)} valid trials)")

        data = _collect_subject(
            subject=subject,
            subject_indices=subject_local_indices,
            bundle=bundle,
            score_fn=score_fn,
            baseline_cfg=baseline_cfg,
            gen_cfg=gen_cfg,
            max_per_subject=max_per_subject,
            seed_offset=sid * 10000,
        )
        if not data:
            print(f"  No data collected for subject {subject}, skipping.")
            continue

        out_path = out_dir / f"attack_patterns_subject_{subject}.npz"
        np.savez_compressed(str(out_path), **data)

        n_success = int(data["success"].sum())
        n_total = len(data["success"])
        asr = n_success / n_total if n_total > 0 else 0.0
        print(f"  Saved {n_total} trials  ASR={asr:.1%}  → {out_path}")
        summary[str(subject)] = {"n_total": n_total, "n_success": n_success, "asr": asr}

    summary_path = out_dir / "attack_patterns_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_subject", type=int, default=None,
                        help="Max trials to attack per subject (default: all)")
    args = parser.parse_args()
    collect_patterns(max_per_subject=args.max_per_subject)
