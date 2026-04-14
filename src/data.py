from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import BaselineConfig


@dataclass
class DatasetBundle:
    train_set: Dataset
    valid_set: Dataset
    n_chans: int
    n_classes: int
    input_window_samples: int
    class_names: tuple[str, ...]
    split_summary: dict[str, object]
    valid_subjects: np.ndarray = None  # subject ID per valid trial (len == len(valid_set))


def _extract_shape_and_classes(windows_dataset, targets: np.ndarray) -> tuple[int, int, int]:
    x0, y0, _ = windows_dataset[0]
    n_chans = int(x0.shape[0])
    input_window_samples = int(x0.shape[1])
    n_classes = len(np.unique(targets))
    return n_chans, n_classes, input_window_samples


def _resolve_targets(metadata, cfg: BaselineConfig) -> tuple[np.ndarray, tuple[str, ...]]:
    target_mode = cfg.target_mode.lower()

    if target_mode in {"task", "class", "event"}:
        targets = metadata["target"].to_numpy(dtype=np.int64, copy=False)
        class_names = tuple(str(label) for label in np.unique(targets))
        return targets, class_names

    if target_mode == "subject":
        if "subject" not in metadata.columns:
            raise ValueError("subject target_mode requires a 'subject' column in metadata")
        subject_values = metadata["subject"].to_numpy(copy=False)
        unique_subjects = np.sort(np.unique(subject_values))
        subject_to_index = {subject: idx for idx, subject in enumerate(unique_subjects.tolist())}
        targets = np.asarray([subject_to_index[subject] for subject in subject_values], dtype=np.int64)
        class_names = tuple(f"subject_{int(subject)}" for subject in unique_subjects.tolist())
        return targets, class_names

    raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")


def _inverse_symmetric_matrix_sqrt(matrix: np.ndarray, eps: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return inv_sqrt.astype(np.float32, copy=False)


def _unpack_sample(sample):
    if len(sample) == 3:
        return sample[0], sample[1], sample[2]
    if len(sample) == 2:
        return sample[0], sample[1], None
    raise ValueError(f"Unexpected dataset sample structure with length {len(sample)}")


class EuclideanAlignedSubset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices: list[int],
        alignment_mats: dict[object, np.ndarray],
        groups: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.alignment_mats = alignment_mats
        self.groups = groups
        self.labels = labels
        self.default_group = next(iter(alignment_mats))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        x, _, extra = _unpack_sample(self.base_dataset[base_index])
        x_np = np.asarray(x, dtype=np.float32)
        group = self.groups[base_index]
        align = self.alignment_mats.get(group, self.alignment_mats[self.default_group])
        aligned = (align @ x_np).astype(np.float32, copy=False)
        y = int(self.labels[base_index])
        if extra is None:
            return aligned, y
        return aligned, y, extra


class LabelRemappedSubset(Dataset):
    def __init__(self, base_dataset, indices: list[int], labels: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        base_index = self.indices[item]
        x, _, extra = _unpack_sample(self.base_dataset[base_index])
        y = int(self.labels[base_index])
        if extra is None:
            return x, y
        return x, y, extra


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, cfg: BaselineConfig) -> None:
        self.base_dataset = base_dataset
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        max_shift = int(self.cfg.aug_time_shift_max_samples)
        if max_shift <= 0 or self.rng.random() >= self.cfg.aug_time_shift_prob:
            return x
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return x
        shifted = np.zeros_like(x)
        if shift > 0:
            shifted[:, shift:] = x[:, :-shift]
        else:
            shifted[:, :shift] = x[:, -shift:]
        return shifted

    def _amplitude_jitter(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.cfg.aug_amplitude_jitter_prob:
            return x
        scale = float(self.rng.normal(loc=1.0, scale=self.cfg.aug_amplitude_jitter_std))
        return x * scale

    def _gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() >= self.cfg.aug_gaussian_noise_prob:
            return x
        signal_scale = max(float(np.std(x)), 1e-6)
        noise = self.rng.normal(
            loc=0.0,
            scale=self.cfg.aug_gaussian_noise_std * signal_scale,
            size=x.shape,
        )
        return x + noise.astype(np.float32)

    def _channel_dropout(self, x: np.ndarray) -> np.ndarray:
        dropout_prob = float(self.cfg.aug_channel_dropout_prob)
        if dropout_prob <= 0.0:
            return x
        mask = self.rng.random(x.shape[0]) >= dropout_prob
        if mask.all():
            return x
        dropped = x.copy()
        dropped[~mask, :] = 0.0
        return dropped

    def __getitem__(self, item: int):
        sample = self.base_dataset[item]
        x, y, extra = _unpack_sample(sample)
        x_aug = np.asarray(x, dtype=np.float32).copy()
        x_aug = self._time_shift(x_aug)
        x_aug = self._amplitude_jitter(x_aug)
        x_aug = self._gaussian_noise(x_aug)
        x_aug = self._channel_dropout(x_aug)
        if extra is None:
            return x_aug, y
        return x_aug, y, extra


def _compute_alignment_mats(
    windows_dataset,
    train_indices: np.ndarray,
    groups: np.ndarray,
    eps: float,
) -> dict[object, np.ndarray]:
    cov_sums: dict[object, np.ndarray] = {}
    counts: dict[object, int] = {}

    for index in train_indices.tolist():
        x, _, _ = _unpack_sample(windows_dataset[index])
        x_np = np.asarray(x, dtype=np.float32)
        cov = x_np @ x_np.T
        trace = float(np.trace(cov))
        if trace > eps:
            cov = cov / trace
        group = groups[index]
        if group not in cov_sums:
            cov_sums[group] = cov.astype(np.float64, copy=True)
            counts[group] = 1
        else:
            cov_sums[group] += cov
            counts[group] += 1

    return {
        group: _inverse_symmetric_matrix_sqrt(cov_sums[group] / counts[group], eps=eps)
        for group in cov_sums
    }


def _split_indices(
    metadata,
    targets: np.ndarray,
    cfg: BaselineConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    indices = np.arange(len(metadata))
    protocol = cfg.evaluation_protocol.lower()

    if protocol == "random":
        train_idx, valid_idx = train_test_split(
            indices,
            train_size=cfg.train_fraction,
            random_state=cfg.random_seed,
            stratify=targets,
        )
        return train_idx, valid_idx, {"protocol": protocol, "train_fraction": float(cfg.train_fraction)}

    if protocol == "cross_session":
        if "session" not in metadata.columns:
            raise ValueError("cross_session evaluation requires a 'session' column in metadata")
        session_values = metadata["session"].astype(str).to_numpy()
        train_idx = indices[session_values == cfg.train_session_name]
        valid_idx = indices[session_values == cfg.valid_session_name]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(
                "cross_session split produced an empty partition: "
                f"train_session_name={cfg.train_session_name!r}, valid_session_name={cfg.valid_session_name!r}"
            )
        return train_idx, valid_idx, {
            "protocol": protocol,
            "train_session_name": cfg.train_session_name,
            "valid_session_name": cfg.valid_session_name,
        }

    if protocol == "within_session":
        required_columns = {"subject", "session"}
        missing = required_columns.difference(metadata.columns)
        if missing:
            raise ValueError(
                "within_session evaluation requires metadata columns: "
                + ", ".join(sorted(missing))
            )

        train_parts = []
        valid_parts = []
        subject_values = metadata["subject"].to_numpy()
        session_values = metadata["session"].astype(str).to_numpy()

        for subject in np.unique(subject_values):
            for session in np.unique(session_values[subject_values == subject]):
                group_mask = (subject_values == subject) & (session_values == session)
                group_idx = indices[group_mask]
                group_targets = targets[group_mask]
                unique_targets, counts = np.unique(group_targets, return_counts=True)
                can_stratify = len(unique_targets) > 1 and np.all(counts >= 2)
                split_kwargs = {
                    "train_size": cfg.train_fraction,
                    "random_state": cfg.random_seed,
                }
                if can_stratify:
                    split_kwargs["stratify"] = group_targets
                group_train_idx, group_valid_idx = train_test_split(group_idx, **split_kwargs)
                train_parts.append(np.sort(group_train_idx))
                valid_parts.append(np.sort(group_valid_idx))

        if not train_parts or not valid_parts:
            raise ValueError("within_session split produced an empty partition")

        train_idx = np.concatenate(train_parts)
        valid_idx = np.concatenate(valid_parts)
        return np.sort(train_idx), np.sort(valid_idx), {
            "protocol": protocol,
            "train_fraction": float(cfg.train_fraction),
        }

    if protocol == "session_run_holdout":
        required_columns = {"subject", "session", "run"}
        missing = required_columns.difference(metadata.columns)
        if missing:
            raise ValueError(
                "session_run_holdout evaluation requires metadata columns: "
                + ", ".join(sorted(missing))
            )

        subject_values = metadata["subject"].to_numpy()
        session_values = metadata["session"].astype(str).to_numpy()
        run_values = metadata["run"].astype(str).to_numpy()
        rng = np.random.default_rng(cfg.random_seed)

        subject_specs = []
        for subject in np.unique(subject_values):
            subject_mask = subject_values == subject
            train_session_mask = subject_mask & (session_values == cfg.train_session_name)
            valid_session_mask = subject_mask & (session_values == cfg.valid_session_name)
            if not train_session_mask.any() or not valid_session_mask.any():
                raise ValueError(
                    "session_run_holdout requires both configured sessions for every subject: "
                    f"subject={subject!r}, train_session_name={cfg.train_session_name!r}, "
                    f"valid_session_name={cfg.valid_session_name!r}"
                )

            train_runs = sorted(np.unique(run_values[train_session_mask]).tolist())
            valid_runs = sorted(np.unique(run_values[valid_session_mask]).tolist())
            total_runs = len(train_runs) + len(valid_runs)
            desired_holdout = max(
                0.0,
                min(float(len(valid_runs)), float((1.0 - cfg.train_fraction) * total_runs)),
            )
            floor_holdout = int(np.floor(desired_holdout))
            frac_holdout = float(desired_holdout - floor_holdout)
            subject_specs.append(
                {
                    "subject": int(subject),
                    "train_runs": train_runs,
                    "valid_runs": valid_runs,
                    "desired_holdout": desired_holdout,
                    "floor_holdout": floor_holdout,
                    "frac_holdout": frac_holdout,
                }
            )

        base_total = sum(int(spec["floor_holdout"]) for spec in subject_specs)
        desired_total = sum(float(spec["desired_holdout"]) for spec in subject_specs)
        target_total = int(np.round(desired_total))
        extra_needed = max(0, min(target_total - base_total, len(subject_specs)))
        ranked_subjects = sorted(
            range(len(subject_specs)),
            key=lambda idx: (
                float(subject_specs[idx]["frac_holdout"]),
                -int(subject_specs[idx]["subject"]),
            ),
            reverse=True,
        )
        extra_subject_ids = set(ranked_subjects[:extra_needed])

        train_parts = []
        valid_parts = []
        heldout_runs_by_subject: dict[str, list[str]] = {}
        train_runs_by_subject: dict[str, list[str]] = {}
        holdout_run_count_by_subject: dict[str, int] = {}

        for spec_idx, spec in enumerate(subject_specs):
            subject = int(spec["subject"])
            valid_runs = list(spec["valid_runs"])
            n_holdout = int(spec["floor_holdout"]) + int(spec_idx in extra_subject_ids)
            n_holdout = max(0, min(n_holdout, len(valid_runs)))
            permuted_valid_runs = list(rng.permutation(valid_runs))
            heldout_runs = [str(run) for run in sorted(permuted_valid_runs[:n_holdout])]
            extra_train_runs = [str(run) for run in sorted(permuted_valid_runs[n_holdout:])]

            subject_mask = subject_values == subject
            subject_train_mask = subject_mask & (session_values == cfg.train_session_name)
            subject_valid_mask = subject_mask & (session_values == cfg.valid_session_name)
            subject_holdout_mask = subject_valid_mask & np.isin(run_values, heldout_runs)
            subject_extra_train_mask = subject_valid_mask & np.isin(run_values, extra_train_runs)

            train_idx_subject = indices[subject_train_mask | subject_extra_train_mask]
            valid_idx_subject = indices[subject_holdout_mask]
            if len(train_idx_subject) == 0 or len(valid_idx_subject) == 0:
                raise ValueError(
                    "session_run_holdout produced an empty partition for subject "
                    f"{subject}; heldout_runs={heldout_runs!r}"
                )

            train_parts.append(np.sort(train_idx_subject))
            valid_parts.append(np.sort(valid_idx_subject))
            heldout_runs_by_subject[str(subject)] = heldout_runs
            train_runs_by_subject[str(subject)] = [str(run) for run in spec["train_runs"]] + extra_train_runs
            holdout_run_count_by_subject[str(subject)] = n_holdout

        train_idx = np.concatenate(train_parts)
        valid_idx = np.concatenate(valid_parts)
        actual_train_fraction = float(len(train_idx) / max(len(train_idx) + len(valid_idx), 1))
        return np.sort(train_idx), np.sort(valid_idx), {
            "protocol": protocol,
            "train_session_name": cfg.train_session_name,
            "valid_session_name": cfg.valid_session_name,
            "requested_train_fraction": float(cfg.train_fraction),
            "actual_train_fraction": actual_train_fraction,
            "heldout_runs_by_subject": heldout_runs_by_subject,
            "train_runs_by_subject": train_runs_by_subject,
            "holdout_run_count_by_subject": holdout_run_count_by_subject,
        }

    raise ValueError(f"Unsupported evaluation_protocol: {cfg.evaluation_protocol}")


def load_moabb_windows(cfg: BaselineConfig) -> DatasetBundle:
    dataset = MOABBDataset(dataset_name=cfg.dataset_name, subject_ids=list(cfg.subject_ids))

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda x: x * 1e6, apply_on_array=True),
        Preprocessor("filter", l_freq=cfg.filter_low_hz, h_freq=cfg.filter_high_hz),
        Preprocessor("resample", sfreq=cfg.sfreq),
    ]
    if cfg.use_exponential_moving_standardize:
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=cfg.standardize_factor_new,
                init_block_size=cfg.standardize_init_block_size,
            )
        )
    preprocess(dataset, preprocessors)

    trial_start_offset_samples = int(cfg.trial_start_offset_seconds * cfg.sfreq)
    window_size_samples = int(cfg.window_size_seconds * cfg.sfreq)
    if cfg.trialwise_decoding:
        window_stride_samples = window_size_samples
    else:
        window_stride_samples = int(cfg.window_stride_seconds * cfg.sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        preload=True,
    )
    metadata = windows_dataset.get_metadata()
    targets, class_names = _resolve_targets(metadata, cfg)
    if cfg.ea_group_by_subject and "subject" in metadata.columns:
        groups = metadata["subject"].to_numpy(copy=False)
    else:
        groups = np.asarray(["global"] * len(metadata), dtype=object)

    train_idx, valid_idx, split_summary = _split_indices(
        metadata=metadata,
        targets=targets,
        cfg=cfg,
    )

    train_set: Dataset = LabelRemappedSubset(windows_dataset, train_idx.tolist(), targets)
    valid_set: Dataset = LabelRemappedSubset(windows_dataset, valid_idx.tolist(), targets)

    if cfg.use_euclidean_alignment:
        alignment_mats = _compute_alignment_mats(
            windows_dataset=windows_dataset,
            train_indices=train_idx,
            groups=groups,
            eps=cfg.ea_eps,
        )
        train_set = EuclideanAlignedSubset(
            base_dataset=windows_dataset,
            indices=train_idx.tolist(),
            alignment_mats=alignment_mats,
            groups=groups,
            labels=targets,
        )
        valid_set = EuclideanAlignedSubset(
            base_dataset=windows_dataset,
            indices=valid_idx.tolist(),
            alignment_mats=alignment_mats,
            groups=groups,
            labels=targets,
        )

    if cfg.use_data_augmentation:
        train_set = AugmentedDataset(train_set, cfg)

    n_chans, n_classes, input_window_samples = _extract_shape_and_classes(windows_dataset, targets)
    subject_values = metadata["subject"].to_numpy(copy=False) if "subject" in metadata.columns else None
    valid_subjects = subject_values[valid_idx] if subject_values is not None else None
    return DatasetBundle(
        train_set=train_set,
        valid_set=valid_set,
        n_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        class_names=class_names,
        split_summary=split_summary,
        valid_subjects=valid_subjects,
    )
