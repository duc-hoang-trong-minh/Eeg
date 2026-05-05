from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode import EEGClassifier
from braindecode.models import EEGConformer
try:
    from braindecode.models import EEGNet
except ImportError:  # pragma: no cover - compatibility with older braindecode releases
    from braindecode.models import EEGNetv4 as EEGNet
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.helper import predefined_split

from .config import BaselineConfig, OutputConfig
from .data import load_windows
from .plot_training import generate_training_plots


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _monitor_prefers_lower(name: str) -> bool:
    lowered = name.lower()
    return "loss" in lowered or lowered.endswith("error")


def _resolve_paths(out_cfg: OutputConfig, seed: int, use_seed_suffix: bool) -> tuple[Path, Path, Path]:
    if use_seed_suffix:
        return (
            out_cfg.baseline_model_path_for_seed(seed),
            out_cfg.baseline_metrics_path_for_seed(seed),
            out_cfg.baseline_scores_path_for_seed(seed),
        )
    return (
        out_cfg.baseline_model_path,
        out_cfg.baseline_metrics_path,
        out_cfg.baseline_scores_path,
    )


def _build_model(cfg: BaselineConfig, n_chans: int, n_classes: int, n_times: int) -> nn.Module:
    if cfg.model_name in {"EEGNet", "EEGNetv4"}:
        return EEGNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
        )
    if cfg.model_name == "EEGConformer":
        return EEGConformer(
            n_chans=n_chans,
            n_outputs=n_classes,
            n_times=n_times,
        )
    raise ValueError(f"Unsupported baseline model_name: {cfg.model_name}")


def _build_classifier(
    cfg: BaselineConfig,
    model: nn.Module,
    valid_set,
    classes: list[int],
    device: str,
) -> EEGClassifier:
    if cfg.optimizer_name != "Adam":
        raise ValueError(f"Unsupported optimizer_name: {cfg.optimizer_name}")
    if cfg.criterion_name != "CrossEntropyLoss":
        raise ValueError(f"Unsupported criterion_name: {cfg.criterion_name}")
    if cfg.lr_scheduler_name != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported lr_scheduler_name: {cfg.lr_scheduler_name}")

    monitor_lower_is_better = _monitor_prefers_lower(cfg.early_stopping_monitor)
    callbacks = [
        (
            "train_accuracy",
            EpochScoring(
                scoring="accuracy",
                on_train=True,
                name="train_accuracy",
                lower_is_better=False,
            ),
        ),
        (
            "valid_accuracy",
            EpochScoring(
                scoring="accuracy",
                on_train=False,
                name="valid_accuracy",
                lower_is_better=False,
            ),
        ),
        (
            "lr_scheduler",
            LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                monitor=cfg.lr_scheduler_monitor,
                factor=cfg.lr_scheduler_factor,
                patience=cfg.lr_scheduler_patience,
            ),
        ),
        (
            "early_stopping",
            EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                patience=cfg.early_stopping_patience,
                lower_is_better=monitor_lower_is_better,
                load_best=True,
            ),
        ),
    ]

    return EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__label_smoothing=cfg.label_smoothing,
        optimizer=torch.optim.Adam,
        train_split=predefined_split(valid_set),
        optimizer__lr=cfg.learning_rate,
        optimizer__weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        max_epochs=cfg.n_epochs,
        iterator_train__shuffle=True,
        callbacks=callbacks,
        device=device,
        classes=classes,
    )


def _history_to_rows(clf: EEGClassifier, cfg: BaselineConfig) -> list[dict]:
    rows = []
    current_lr = float(cfg.learning_rate)
    for epoch_idx, row in enumerate(clf.history, start=1):
        current_lr = float(row.get("event_lr", row.get("lr", current_lr)))
        rows.append(
            {
                "epoch": int(row.get("epoch", epoch_idx)),
                "lr": current_lr,
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row.get("train_accuracy", row.get("train_acc", 0.0))),
                "val_loss": float(row["valid_loss"]),
                "val_acc": float(row.get("valid_accuracy", row.get("valid_acc", row.get("valid_acc_best", 0.0)))),
            }
        )
    return rows


def _collect_scores(clf: EEGClassifier, dataset) -> tuple[np.ndarray, np.ndarray]:
    logits = clf.predict_proba(dataset).astype(np.float32)
    labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)
    return logits, labels


def train_and_save_baseline(
    cfg: BaselineConfig,
    out_cfg: OutputConfig,
    use_seed_suffix: bool = False,
) -> dict:
    _set_seed(cfg.random_seed)
    out_cfg.root.mkdir(parents=True, exist_ok=True)
    model_path, metrics_path, scores_path = _resolve_paths(
        out_cfg=out_cfg,
        seed=cfg.random_seed,
        use_seed_suffix=use_seed_suffix,
    )

    bundle = load_windows(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(
        cfg=cfg,
        n_chans=bundle.n_chans,
        n_classes=bundle.n_classes,
        n_times=bundle.input_window_samples,
    )
    classes = list(range(bundle.n_classes))
    clf = _build_classifier(
        cfg=cfg,
        model=model,
        valid_set=bundle.valid_set,
        classes=classes,
        device=device,
    )
    clf.fit(bundle.train_set, y=None)

    history = _history_to_rows(clf, cfg)
    best_val_acc = max((row["val_acc"] for row in history), default=0.0)
    best_state = {k: v.detach().cpu() for k, v in clf.module_.state_dict().items()}

    checkpoint = {
        "model_state": best_state,
        "model_name": cfg.model_name,
        "n_chans": bundle.n_chans,
        "n_classes": bundle.n_classes,
        "input_window_samples": bundle.input_window_samples,
        "target_mode": cfg.target_mode,
        "class_names": list(bundle.class_names),
        "config": cfg.__dict__,
        "history": history,
        "split_summary": bundle.split_summary,
    }
    torch.save(checkpoint, model_path)

    valid_scores, valid_labels = _collect_scores(clf, bundle.valid_set)
    np.savez_compressed(
        scores_path,
        logits=valid_scores,
        labels=valid_labels,
        predictions=valid_scores.argmax(axis=1) if valid_scores.size else np.zeros((0,), dtype=np.int64),
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "history": history,
                "model_path": str(model_path),
                "scores_path": str(scores_path),
                "model_name": cfg.model_name,
                "target_mode": cfg.target_mode,
                "class_names": list(bundle.class_names),
                "evaluation_protocol": cfg.evaluation_protocol,
                "train_session_name": cfg.train_session_name,
                "valid_session_name": cfg.valid_session_name,
                "split_summary": bundle.split_summary,
                "n_train_samples": len(bundle.train_set),
                "n_valid_samples": len(bundle.valid_set),
                "random_seed": cfg.random_seed,
            },
            f,
            indent=2,
        )
    plot_paths = generate_training_plots(metrics_path)

    return {
        "best_val_acc": best_val_acc,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "scores_path": str(scores_path),
        **plot_paths,
        "model_name": cfg.model_name,
        "evaluation_protocol": cfg.evaluation_protocol,
        "random_seed": cfg.random_seed,
    }


def train_multiseed_baselines(cfg: BaselineConfig, out_cfg: OutputConfig) -> dict:
    runs = []
    for seed in cfg.stability_seeds:
        seed_cfg = BaselineConfig(**{**cfg.__dict__, "random_seed": seed})
        runs.append(train_and_save_baseline(seed_cfg, out_cfg, use_seed_suffix=True))

    best_val_accs = np.asarray([run["best_val_acc"] for run in runs], dtype=np.float32)
    summary = {
        "dataset_name": cfg.dataset_name,
        "seeds": list(cfg.stability_seeds),
        "mean_best_val_acc": float(best_val_accs.mean()) if len(best_val_accs) else 0.0,
        "std_best_val_acc": float(best_val_accs.std()) if len(best_val_accs) else 0.0,
        "runs": runs,
    }
    with out_cfg.baseline_multiseed_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
