from __future__ import annotations

import os
from pathlib import Path

from .config import BaselineConfig, OutputConfig


def build_bnci2014_001_human_recognition_config(model_name: str = "EEGConformer") -> BaselineConfig:
    return BaselineConfig(
        model_name=model_name,
        dataset_name="BNCI2014_001",
        subject_ids=tuple(range(1, 10)),
        target_mode="subject",
        evaluation_protocol="cross_session",
        train_session_name="0train",
        valid_session_name="1test",
        train_fraction=0.5,
    )


def build_bnci2014_001_human_recognition_output_config() -> OutputConfig:
    return OutputConfig(
        root=Path("outputs/bnci2014_001_human_recognition"),
        baseline_model_name="subject_recognition_baseline.pt",
        baseline_metrics_name="subject_recognition_metrics.json",
        baseline_scores_name="subject_recognition_scores.npz",
        baseline_multiseed_summary_name="subject_recognition_multiseed_summary.json",
    )


def build_tuh_abnormal_human_recognition_config(model_name: str = "EEGConformer") -> BaselineConfig:
    return BaselineConfig(
        model_name=model_name,
        dataset_name="TUHAbnormal",
        dataset_path=os.environ.get("EEG_TUHABNORMAL_PATH") or os.environ.get("EEG_TUH_PATH"),
        subject_ids=None,
        max_subjects=2000,
        min_sessions_per_subject=2,
        target_mode="subject",
        evaluation_protocol="subject_session_holdout",
        trial_start_offset_seconds=0.0,
        window_preload=False,
        n_epochs=30,
        batch_size=64,
        learning_rate=5e-4,
    )


def build_tuh_abnormal_human_recognition_output_config() -> OutputConfig:
    return OutputConfig(
        root=Path("outputs/tuh_abnormal_human_recognition"),
        baseline_model_name="subject_recognition_baseline.pt",
        baseline_metrics_name="subject_recognition_metrics.json",
        baseline_scores_name="subject_recognition_scores.npz",
        baseline_multiseed_summary_name="subject_recognition_multiseed_summary.json",
    )


def build_hbn_r1_l100_human_recognition_config(model_name: str = "EEGConformer") -> BaselineConfig:
    max_subjects_env = os.environ.get("EEG_HBN_MAX_SUBJECTS")
    n_epochs_env = os.environ.get("EEG_HBN_EPOCHS")
    channel_limit_env = os.environ.get("EEG_HBN_CHANNEL_LIMIT")
    common_eeg_channels = None
    if channel_limit_env:
        channel_limit = int(channel_limit_env)
        common_eeg_channels = tuple(f"E{idx}" for idx in range(1, channel_limit + 1))

    return BaselineConfig(
        model_name=model_name,
        dataset_name="HBN",
        dataset_path=os.environ.get("EEG_HBN_PATH") or "/home/necphy/data/hbn/R1_L100_bdf",
        subject_ids=None,
        max_subjects=None if max_subjects_env is None else int(max_subjects_env),
        min_sessions_per_subject=2,
        common_eeg_channels=common_eeg_channels,
        target_mode="subject",
        evaluation_protocol="task_holdout",
        train_session_name="train",
        valid_session_name="valid",
        sfreq=100.0,
        filter_low_hz=4.0,
        filter_high_hz=38.0,
        trial_start_offset_seconds=0.0,
        window_size_seconds=4.0,
        window_stride_seconds=4.0,
        trialwise_decoding=True,
        window_preload=False,
        use_data_augmentation=True,
        n_epochs=30 if n_epochs_env is None else int(n_epochs_env),
        batch_size=64,
        learning_rate=5e-4,
    )


def build_hbn_r1_l100_human_recognition_output_config() -> OutputConfig:
    return OutputConfig(
        root=Path("outputs/hbn_r1_l100_human_recognition"),
        baseline_model_name="subject_recognition_baseline.pt",
        baseline_metrics_name="subject_recognition_metrics.json",
        baseline_scores_name="subject_recognition_scores.npz",
        baseline_multiseed_summary_name="subject_recognition_multiseed_summary.json",
    )
