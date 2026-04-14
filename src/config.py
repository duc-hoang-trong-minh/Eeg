from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineConfig:
    model_name: str = "EEGConformer"
    dataset_name: str = "BNCI2014_001"
    subject_ids: tuple[int, ...] = (1, 2, 3)
    target_mode: str = "task"
    stability_seeds: tuple[int, ...] = (7, 11, 19)
    evaluation_protocol: str = "within_session"
    train_session_name: str = "0train"
    valid_session_name: str = "1test"
    sfreq: float = 128.0
    filter_low_hz: float = 4.0
    filter_high_hz: float = 38.0
    use_exponential_moving_standardize: bool = True
    standardize_factor_new: float = 1e-3
    standardize_init_block_size: int = 1000

    use_euclidean_alignment: bool = True
    ea_group_by_subject: bool = True
    ea_eps: float = 1e-6

    trial_start_offset_seconds: float = -0.5
    window_size_seconds: float = 4.0
    window_stride_seconds: float = 4.0
    trialwise_decoding: bool = True

    use_data_augmentation: bool = True

    aug_time_shift_prob: float = 0.5
    aug_time_shift_max_samples: int = 16
    aug_amplitude_jitter_prob: float = 0.5
    aug_amplitude_jitter_std: float = 0.1
    aug_gaussian_noise_prob: float = 0.5
    aug_gaussian_noise_std: float = 0.01
    aug_channel_dropout_prob: float = 0.1
    n_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "Adam"
    criterion_name: str = "CrossEntropyLoss"
    label_smoothing: float = 0.1
    early_stopping_patience: int = 12
    early_stopping_monitor: str = "valid_accuracy"
    lr_scheduler_name: str = "ReduceLROnPlateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    lr_scheduler_monitor: str = "valid_loss"
    random_seed: int = 7
    train_fraction: float = 0.8


@dataclass
class AttackConfig:
    support_mode: str = "channel_first"
    basis_mode: str = "hybrid"
    n_windows: int = 8
    support_budget_k: int = 5
    channel_shortlist_size: int | None = None
    enforce_unique_channels: bool = False
    stop_on_success: bool = True
    basis_rank_r: int = 4
    channel_waveform_rank: int | None = None
    basis_min_hz: float = 2.0
    basis_max_hz: float = 30.0
    basis_phase_count: int = 2
    candidate_probe_restarts: int = 2
    candidate_probe_scale: float = 0.5
    max_outer_iters: int = 5
    max_query_budget: int | None = 5000
    spsa_steps: int = 80
    spsa_step_size: float = 0.04
    spsa_perturb_scale: float = 0.02
    spsa_restarts: int = 2
    spsa_init_scale: float = 0.2
    l2_weight: float = 1e-3
    tv_weight: float = 1e-3
    band_weight: float = 1e-3
    max_coeff_abs: float = 0.2
    max_perturbation_peak_ratio: float | None = None


@dataclass
class GeneratorConfig:
    support_budget_k: int = 5
    basis_rank_r: int = 4
    n_windows: int = 8
    basis_min_hz: float = 2.0
    basis_max_hz: float = 30.0
    basis_mode: str = "hybrid"
    basis_phase_count: int = 2
    max_coeff_abs: float = 0.75
    n_epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    train_fraction: float = 0.8
    scope: str = "subject"  # "subject" or "model"


@dataclass
class OutputConfig:
    root: Path = Path("outputs")
    baseline_model_name: str = "eegconformer_baseline.pt"
    baseline_metrics_name: str = "baseline_metrics.json"
    baseline_scores_name: str = "baseline_scores.npz"
    baseline_multiseed_summary_name: str = "baseline_multiseed_summary.json"

    @property
    def baseline_model_path(self) -> Path:
        return self.root / self.baseline_model_name

    @property
    def baseline_metrics_path(self) -> Path:
        return self.root / self.baseline_metrics_name

    @property
    def baseline_scores_path(self) -> Path:
        return self.root / self.baseline_scores_name

    @property
    def baseline_multiseed_summary_path(self) -> Path:
        return self.root / self.baseline_multiseed_summary_name

    def _seeded_name(self, filename: str, seed: int) -> str:
        path = Path(filename)
        suffix = "".join(path.suffixes)
        stem = path.name[: -len(suffix)] if suffix else path.name
        return f"{stem}_seed{seed}{suffix}"

    def baseline_model_path_for_seed(self, seed: int) -> Path:
        return self.root / self._seeded_name(self.baseline_model_name, seed)

    def baseline_metrics_path_for_seed(self, seed: int) -> Path:
        return self.root / self._seeded_name(self.baseline_metrics_name, seed)

    def baseline_scores_path_for_seed(self, seed: int) -> Path:
        return self.root / self._seeded_name(self.baseline_scores_name, seed)
