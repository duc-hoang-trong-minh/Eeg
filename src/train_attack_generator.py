"""Train the AttackGenerator from collected sample-wise attack patterns.

Two scopes:
  --scope subject   Train one generator per subject using only that subject's patterns.
  --scope model     Train a single generator across all subjects' patterns.

Checkpoints saved to:
  outputs/attack_generator_subject_{S}.pt
  outputs/attack_generator_model_wise.pt

Universal support saved to:
  outputs/universal_support_subject_{S}.json
  outputs/universal_support_model_wise.json

Usage:
    python -m src.train_attack_generator --scope subject
    python -m src.train_attack_generator --scope model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

from .attack.generator import AttackGenerator
from .attack.universal_support import (
    build_target_coeffs_dataset,
    discover_universal_support,
    load_universal_support,
    save_universal_support,
)
from .config import BaselineConfig, GeneratorConfig, OutputConfig
from .model_oracle import load_eegnet_checkpoint


def _find_pattern_files(out_dir: Path, subject: object | None) -> list[Path]:
    if subject is not None:
        paths = [out_dir / f"attack_patterns_subject_{subject}.npz"]
    else:
        paths = sorted(out_dir.glob("attack_patterns_subject_*.npz"))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Pattern files not found: {missing}\n"
            "Run `python -m src.collect_attack_patterns` first."
        )
    return paths


def _train_one(
    X: np.ndarray,
    universal_support: list[tuple[int, int]],
    victim_model: nn.Module,
    gen_cfg: GeneratorConfig,
    baseline_cfg: BaselineConfig,
    device: torch.device,
    delta_weight: float = 1e-3,
) -> AttackGenerator:
    """Train generator with adversarial loss: fool the victim model, keep delta small.

    Loss = -CE(victim(x + delta), y)   ← maximise misclassification
         + delta_weight * ||delta||^2   ← keep perturbation small

    Victim model is white-box during training; zero queries at test time.
    """
    n_total = len(X)
    # load labels from the victim model (we treat it as an oracle)
    victim_model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits_all = victim_model(X_t)
    Y_t = logits_all.argmax(dim=1).cpu()  # predicted labels used as "true" labels

    # train/val split
    if n_total < 2:
        print(f"  Warning: only {n_total} sample(s), training on all data.")
        train_idx = np.arange(n_total)
        val_idx = np.array([], dtype=int)
    else:
        n_train = max(1, int(n_total * gen_cfg.train_fraction))
        idx = np.random.default_rng(baseline_cfg.random_seed).permutation(n_total)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = TensorDataset(X_t.cpu()[train_idx], Y_t[train_idx])
    val_ds   = TensorDataset(X_t.cpu()[val_idx],   Y_t[val_idx]) if len(val_idx) > 0 else None
    train_loader = DataLoader(train_ds, batch_size=min(gen_cfg.batch_size, len(train_ds)),
                              shuffle=True, drop_last=False)
    val_loader   = (DataLoader(val_ds, batch_size=min(gen_cfg.batch_size, len(val_ds)),
                               shuffle=False, drop_last=False)
                    if val_ds is not None else None)

    n_channels, n_times = X.shape[1], X.shape[2]
    generator = AttackGenerator(
        n_channels=n_channels,
        n_times=n_times,
        support_budget_k=gen_cfg.support_budget_k,
        basis_rank_r=gen_cfg.basis_rank_r,
        max_coeff_abs=gen_cfg.max_coeff_abs,
    ).to(device)
    generator.setup_basis(
        universal_support=universal_support,
        n_windows=gen_cfg.n_windows,
        basis_min_hz=gen_cfg.basis_min_hz,
        basis_max_hz=gen_cfg.basis_max_hz,
        basis_mode=gen_cfg.basis_mode,
        basis_phase_count=gen_cfg.basis_phase_count,
        sfreq=baseline_cfg.sfreq,
    )

    optimizer = torch.optim.Adam(generator.parameters(), lr=gen_cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    def _adv_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        delta = generator.forward_delta(xb)             # (N, C, T)
        x_adv = xb + delta
        logits = victim_model(x_adv)
        # maximise misclassification = minimise CE w/ flipped sign
        ce = F.cross_entropy(logits, yb)
        reg = delta.pow(2).mean()
        return -ce + delta_weight * reg, ce.item()

    best_val_loss = float("inf")
    best_state = None

    victim_model.eval()
    for epoch in range(gen_cfg.n_epochs):
        generator.train()
        train_loss, train_ce = 0.0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss, ce = _adv_loss(xb, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_ce   += ce * len(xb)
        train_loss /= len(train_ds)
        train_ce   /= len(train_ds)

        if val_loader is not None:
            generator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss, _ = _adv_loss(xb, yb)
                    val_loss += loss.item() * len(xb)
            val_loss /= len(val_ds)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            val_loss = float("nan")

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{gen_cfg.n_epochs}  "
                  f"adv_loss={train_loss:.4f}  CE={train_ce:.4f}  val={val_loss:.4f}")

    if best_state is not None:
        generator.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return generator


def train(scope: str) -> None:
    gen_cfg = GeneratorConfig(scope=scope)
    baseline_cfg = BaselineConfig()
    out_cfg = OutputConfig()
    out_dir = Path(out_cfg.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  scope: {scope}")

    victim_model, _, _ = load_eegnet_checkpoint(str(out_cfg.baseline_model_path), device=str(device))
    victim_model.eval()

    if scope == "subject":
        pattern_files = sorted(out_dir.glob("attack_patterns_subject_*.npz"))
        if not pattern_files:
            raise FileNotFoundError("No pattern files found. Run collect_attack_patterns first.")

        for pf in pattern_files:
            subject = pf.stem.replace("attack_patterns_subject_", "")
            print(f"\n{'='*50}")
            print(f"Training subject-wise generator for subject {subject}")

            support = discover_universal_support(
                pattern_paths=[pf],
                n_channels=22,
                n_windows=gen_cfg.n_windows,
                support_budget_k=gen_cfg.support_budget_k,
            )
            support_path = out_dir / f"universal_support_subject_{subject}.json"
            save_universal_support(support, support_path, meta={"subject": subject, "scope": "subject"})
            print(f"  Universal support: {support}")

            data = np.load(str(pf))
            X = data["X"]           # all trials, not just successes
            print(f"  Training samples: {len(X)}")

            model = _train_one(X, support, victim_model, gen_cfg, baseline_cfg, device)
            ckpt_path = out_dir / f"attack_generator_subject_{subject}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "n_channels": model.n_channels,
                "n_times": model.n_times,
                "support_budget_k": gen_cfg.support_budget_k,
                "basis_rank_r": gen_cfg.basis_rank_r,
                "max_coeff_abs": gen_cfg.max_coeff_abs,
                "universal_support": support,
                "gen_cfg": gen_cfg.__dict__,
                "scope": "subject",
                "subject": subject,
            }, str(ckpt_path))
            print(f"  Saved: {ckpt_path}")

    elif scope == "model":
        pattern_files = sorted(out_dir.glob("attack_patterns_subject_*.npz"))
        if not pattern_files:
            raise FileNotFoundError("No pattern files found. Run collect_attack_patterns first.")
        print(f"\nTraining model-wise generator on {len(pattern_files)} subjects")

        support = discover_universal_support(
            pattern_paths=pattern_files,
            n_channels=22,
            n_windows=gen_cfg.n_windows,
            support_budget_k=gen_cfg.support_budget_k,
        )
        support_path = out_dir / "universal_support_model_wise.json"
        save_universal_support(support, support_path, meta={"scope": "model"})
        print(f"Universal support: {support}")

        X = np.concatenate([np.load(str(pf))["X"] for pf in pattern_files], axis=0)
        print(f"Training samples: {len(X)}")

        model = _train_one(X, support, victim_model, gen_cfg, baseline_cfg, device)
        ckpt_path = out_dir / "attack_generator_model_wise.pt"
        torch.save({
            "model_state": model.state_dict(),
            "n_channels": model.n_channels,
            "n_times": model.n_times,
            "support_budget_k": gen_cfg.support_budget_k,
            "basis_rank_r": gen_cfg.basis_rank_r,
            "max_coeff_abs": gen_cfg.max_coeff_abs,
            "universal_support": support,
            "gen_cfg": gen_cfg.__dict__,
            "scope": "model",
        }, str(ckpt_path))
        print(f"Saved: {ckpt_path}")

    else:
        raise ValueError(f"Unknown scope: {scope!r}. Use 'subject' or 'model'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["subject", "model"], default="subject")
    args = parser.parse_args()
    train(scope=args.scope)
