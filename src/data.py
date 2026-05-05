from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, BaseDataset, MOABBDataset, TUH, TUHAbnormal
from braindecode.datasets import tuh as tuh_module
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import BaselineConfig


HBN_PASSIVE_TASKS = (
    "RestingState",
    "DespicableMe",
    "FunwithFractals",
    "ThePresent",
    "DiaryOfAWimpyKid",
    "surroundSupp",
)
HBN_ACTIVE_TASKS = (
    "contrastChangeDetection",
    "seqLearning6target",
    "seqLearning8target",
    "symbolSearch",
)
HBN_EEG_EXTENSIONS = (".bdf", ".set", ".edf")
_HBN_EEG_RE = re.compile(
    r"^sub-(?P<subject>[^_]+)_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?_eeg(?P<extension>\.[^.]+)$"
)


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


def _value_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _stringify_value(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


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
        unique_subjects = sorted(np.unique(subject_values).tolist(), key=_value_sort_key)
        subject_to_index = {subject: idx for idx, subject in enumerate(unique_subjects)}
        targets = np.asarray([subject_to_index[subject] for subject in subject_values], dtype=np.int64)
        class_names = tuple(f"subject_{_stringify_value(subject)}" for subject in unique_subjects)
        return targets, class_names

    raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")


def _normalize_preprocessors(cfg: BaselineConfig) -> list[Preprocessor]:
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose="error"),
        Preprocessor(lambda x: x * 1e6, apply_on_array=True),
        Preprocessor("filter", l_freq=cfg.filter_low_hz, h_freq=cfg.filter_high_hz, verbose="error"),
        Preprocessor("resample", sfreq=cfg.sfreq, verbose="error"),
    ]
    if cfg.use_exponential_moving_standardize:
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=cfg.standardize_factor_new,
                init_block_size=cfg.standardize_init_block_size,
            )
        )
    return preprocessors


def _pick_and_order_channels(raw, ch_names: tuple[str, ...]):
    raw.pick(list(ch_names))
    raw.reorder_channels(list(ch_names))
    return raw


def _resolve_tuh_path(cfg: BaselineConfig) -> str:
    candidates: list[str] = []
    if cfg.dataset_path:
        candidates.append(cfg.dataset_path)

    dataset_key = cfg.dataset_name.strip().lower()
    env_names: list[str]
    if dataset_key in {"tuhabnormal", "tuh_abnormal", "tuab"}:
        env_names = ["EEG_TUHABNORMAL_PATH", "EEG_TUH_PATH"]
    elif dataset_key == "tuh":
        env_names = ["EEG_TUH_PATH", "EEG_TUHABNORMAL_PATH"]
    else:
        env_names = []
    env_names.extend(["EEG_DATA_PATH", "EEG_DATA_ROOT"])

    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(env_value)

    checked_paths: list[str] = []
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser())
        checked_paths.append(resolved)
        if Path(resolved).exists():
            return resolved

    if checked_paths:
        raise FileNotFoundError(
            f"Could not find a local path for {cfg.dataset_name}. Checked: {', '.join(checked_paths)}"
        )

    raise FileNotFoundError(
        f"{cfg.dataset_name} requires a local dataset path. "
        "Set BaselineConfig.dataset_path or an environment variable such as "
        "EEG_TUHABNORMAL_PATH or EEG_TUH_PATH."
    )


def _select_tuh_recording_ids(cfg: BaselineConfig, dataset_path: str) -> tuple[list[int] | None, dict[str, object]]:
    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.edf"), recursive=True)
    if not file_paths:
        raise FileNotFoundError(f"No EDF files were found under {dataset_path!r}")

    descriptions = tuh_module._create_description(file_paths)
    descriptions = tuh_module._sort_chronologically(descriptions)
    description_df = descriptions.T.reset_index(drop=True)
    required_columns = {"subject", "session"}
    missing = required_columns.difference(description_df.columns)
    if missing:
        raise ValueError(
            f"{cfg.dataset_name} descriptions are missing required columns: {', '.join(sorted(missing))}"
        )

    subject_keys = np.asarray(
        [_stringify_value(subject) for subject in description_df["subject"].to_numpy(copy=False)],
        dtype=object,
    )
    description_df = description_df.copy()
    description_df["subject_key"] = subject_keys

    required_sessions = max(
        int(cfg.min_sessions_per_subject),
        2 if cfg.evaluation_protocol.lower() == "subject_session_holdout" else 1,
    )
    subject_specs: list[tuple[str, int, int]] = []
    for subject_key in np.unique(subject_keys):
        subject_mask = subject_keys == subject_key
        subject_rows = description_df.loc[subject_mask]
        n_sessions = int(subject_rows["session"].nunique())
        if n_sessions < required_sessions:
            continue
        subject_specs.append((subject_key, n_sessions, int(len(subject_rows))))

    subject_specs.sort(key=lambda item: (-item[1], -item[2], _value_sort_key(item[0])))
    if not subject_specs:
        raise ValueError(
            f"No {cfg.dataset_name} subjects satisfy min_sessions_per_subject={required_sessions}."
        )

    if cfg.max_subjects is None:
        selected_subjects = {subject_key for subject_key, _, _ in subject_specs}
    else:
        capped_specs = subject_specs[: max(0, int(cfg.max_subjects))]
        selected_subjects = {subject_key for subject_key, _, _ in capped_specs}

    if len(selected_subjects) == len(subject_specs):
        recording_ids = None
        selected_rows = description_df
    else:
        selected_rows = description_df.loc[description_df["subject_key"].isin(selected_subjects)]
        recording_ids = selected_rows.index.to_list()

    summary = {
        "dataset_name": cfg.dataset_name,
        "dataset_path": dataset_path,
        "available_recordings": int(len(description_df)),
        "available_subjects_with_required_sessions": int(len(subject_specs)),
        "selected_subjects": int(len(selected_subjects)),
        "selected_recordings": int(len(selected_rows)),
        "min_sessions_per_subject": int(required_sessions),
    }
    return recording_ids, summary


def _find_hbn_bids_root(path: Path) -> Path | None:
    if (path / "participants.tsv").exists() and (path / "dataset_description.json").exists():
        return path

    if path.is_dir():
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "participants.tsv").exists() and (child / "dataset_description.json").exists():
                return child

        for participants_path in sorted(path.glob("**/participants.tsv")):
            candidate = participants_path.parent
            if (candidate / "dataset_description.json").exists():
                return candidate

    return None


def _resolve_hbn_path(cfg: BaselineConfig) -> str:
    candidates: list[str] = []
    if cfg.dataset_path:
        candidates.append(cfg.dataset_path)

    for env_name in ("EEG_HBN_PATH", "EEG_DATA_PATH", "EEG_DATA_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(env_value)

    checked_paths: list[str] = []
    for candidate in candidates:
        resolved = Path(candidate).expanduser()
        checked_paths.append(str(resolved))
        if not resolved.exists():
            continue
        bids_root = _find_hbn_bids_root(resolved)
        if bids_root is not None:
            return str(bids_root)

    if checked_paths:
        raise FileNotFoundError(
            "Could not find an HBN BIDS root containing participants.tsv and "
            f"dataset_description.json. Checked: {', '.join(checked_paths)}"
        )

    raise FileNotFoundError(
        "HBN requires a local BIDS path. Set BaselineConfig.dataset_path or EEG_HBN_PATH, "
        "for example EEG_HBN_PATH=/home/necphy/data/hbn/R1_L100_bdf."
    )


def _normalize_hbn_subject_id(subject: object) -> str:
    text = _stringify_value(subject)
    return text if text.startswith("sub-") else f"sub-{text}"


def _parse_hbn_eeg_path(path: Path) -> dict[str, object] | None:
    match = _HBN_EEG_RE.match(path.name)
    if match is None:
        return None

    extension = match.group("extension").lower()
    if extension not in HBN_EEG_EXTENSIONS:
        return None

    run = match.group("run")
    task = match.group("task")
    task_split = "train" if task in HBN_PASSIVE_TASKS else "valid" if task in HBN_ACTIVE_TASKS else "unused"
    return {
        "path": str(path),
        "subject": _normalize_hbn_subject_id(match.group("subject")),
        "session": task_split,
        "task_split": task_split,
        "task": task,
        "run": "" if run is None else str(run),
        "extension": extension,
    }


def _hbn_qc_column(task: str, run: str) -> str:
    if task in {"contrastChangeDetection", "surroundSupp"} and run:
        return f"{task}_{run}"
    return task


def _load_hbn_participants(bids_root: Path) -> pd.DataFrame:
    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        return pd.DataFrame()

    participants = pd.read_csv(participants_path, sep="\t", dtype=str)
    if "participant_id" not in participants.columns:
        raise ValueError(f"{participants_path} is missing the participant_id column")
    participants["participant_id"] = participants["participant_id"].map(_normalize_hbn_subject_id)
    return participants.set_index("participant_id", drop=False)


def _discover_hbn_records(bids_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for extension in HBN_EEG_EXTENSIONS:
        for path in sorted(bids_root.glob(f"sub-*/eeg/*_eeg{extension}")):
            record = _parse_hbn_eeg_path(path)
            if record is not None:
                records.append(record)

    if not records:
        raise FileNotFoundError(f"No HBN EEG files with extensions {HBN_EEG_EXTENSIONS!r} found under {bids_root}")
    return records


def _hbn_record_status(record: dict[str, object], participants: pd.DataFrame) -> str:
    subject = str(record["subject"])
    if participants.empty or subject not in participants.index:
        return "unknown"

    column = _hbn_qc_column(str(record["task"]), str(record["run"]))
    if column not in participants.columns:
        return "unknown"
    status = participants.loc[subject, column]
    if pd.isna(status):
        return "unknown"
    return str(status).strip().lower()


def _hbn_status_allowed(status: str, policy: str) -> bool:
    policy = policy.lower()
    if policy == "available_only":
        return status == "available"
    if policy == "include_caution":
        return status in {"available", "caution"}
    if policy == "all":
        return True
    raise ValueError(f"Unsupported hbn_qc_policy: {policy!r}")


def _select_hbn_records(cfg: BaselineConfig, bids_root: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = Path(bids_root)
    participants = _load_hbn_participants(root)
    records = _discover_hbn_records(root)

    status_counts: dict[str, int] = {}
    for record in records:
        status = _hbn_record_status(record, participants)
        record["qc_status"] = status
        status_counts[status] = status_counts.get(status, 0) + 1

    train_tasks = set(cfg.hbn_train_tasks or HBN_PASSIVE_TASKS)
    valid_tasks = set(cfg.hbn_valid_tasks or HBN_ACTIVE_TASKS)
    allowed_tasks = train_tasks | valid_tasks

    requested_subjects = None
    if cfg.subject_ids is not None:
        requested_subjects = {_normalize_hbn_subject_id(subject) for subject in cfg.subject_ids}

    candidate_records = [
        record
        for record in records
        if str(record["task"]) in allowed_tasks
        and str(record["task_split"]) in {"train", "valid"}
        and _hbn_status_allowed(str(record["qc_status"]), cfg.hbn_qc_policy)
        and (requested_subjects is None or str(record["subject"]) in requested_subjects)
    ]

    records_by_subject: dict[str, list[dict[str, object]]] = {}
    for record in candidate_records:
        records_by_subject.setdefault(str(record["subject"]), []).append(record)

    subject_specs: list[tuple[str, int, int, int]] = []
    required_recordings = max(2, int(cfg.min_sessions_per_subject))
    for subject, subject_records in records_by_subject.items():
        n_train = sum(1 for record in subject_records if record["task_split"] == "train")
        n_valid = sum(1 for record in subject_records if record["task_split"] == "valid")
        if n_train == 0 or n_valid == 0 or len(subject_records) < required_recordings:
            continue
        subject_specs.append((subject, n_train, n_valid, len(subject_records)))

    subject_specs.sort(key=lambda item: (-item[3], -item[1], -item[2], _value_sort_key(item[0])))
    if not subject_specs:
        raise ValueError(
            "No HBN subjects have both train/passive and validation/active recordings "
            f"after applying hbn_qc_policy={cfg.hbn_qc_policy!r}."
        )

    if cfg.max_subjects is None:
        selected_subjects = {subject for subject, _, _, _ in subject_specs}
    else:
        selected_subjects = {
            subject for subject, _, _, _ in subject_specs[: max(0, int(cfg.max_subjects))]
        }

    selected_records = [
        record for record in candidate_records if str(record["subject"]) in selected_subjects
    ]
    selected_records.sort(
        key=lambda record: (
            _value_sort_key(record["subject"]),
            str(record["task_split"]),
            str(record["task"]),
            _value_sort_key(record["run"]),
        )
    )

    selected_train = sum(1 for record in selected_records if record["task_split"] == "train")
    selected_valid = sum(1 for record in selected_records if record["task_split"] == "valid")
    summary = {
        "dataset_name": cfg.dataset_name,
        "dataset_path": bids_root,
        "dataset_backend": "hbn_bids",
        "hbn_qc_policy": cfg.hbn_qc_policy,
        "available_subjects_in_participants_tsv": int(len(participants)) if not participants.empty else None,
        "discovered_eeg_files": int(len(records)),
        "qc_status_counts": status_counts,
        "candidate_subjects_with_train_and_valid": int(len(subject_specs)),
        "selected_subjects": int(len(selected_subjects)),
        "selected_recordings": int(len(selected_records)),
        "selected_train_recordings": int(selected_train),
        "selected_valid_recordings": int(selected_valid),
        "train_tasks": sorted(train_tasks),
        "valid_tasks": sorted(valid_tasks),
        "file_extensions": sorted({str(record["extension"]) for record in selected_records}),
    }
    return selected_records, summary


def _read_hbn_raw(path: str):
    raw_path = Path(path)
    extension = raw_path.suffix.lower()
    if extension == ".bdf":
        return mne.io.read_raw_bdf(raw_path, preload=False, infer_types=True, verbose="error")
    if extension == ".set":
        return mne.io.read_raw_eeglab(raw_path, preload=False, verbose="error")
    if extension == ".edf":
        return mne.io.read_raw_edf(raw_path, preload=False, infer_types=True, verbose="error")
    raise ValueError(f"Unsupported HBN raw file extension: {extension!r}")


def inspect_hbn_dataset(cfg: BaselineConfig, read_sample_raw: bool = True) -> dict[str, object]:
    bids_root = _resolve_hbn_path(cfg)
    records, summary = _select_hbn_records(cfg, bids_root)
    summary["example_records"] = [
        {
            key: record[key]
            for key in ("subject", "task", "run", "task_split", "qc_status", "path")
        }
        for record in records[:5]
    ]

    if read_sample_raw and records:
        raw = _read_hbn_raw(str(records[0]["path"]))
        eeg_channels = [
            ch_name
            for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
            if ch_type == "eeg"
        ]
        summary["sample_raw"] = {
            "path": str(records[0]["path"]),
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "n_eeg_channels": int(len(eeg_channels)),
            "first_eeg_channels": eeg_channels[:10],
            "duration_seconds": float(raw.times[-1]) if raw.n_times else 0.0,
        }

    return summary


def _resolve_common_eeg_channels(dataset, cfg: BaselineConfig) -> tuple[str, ...]:
    desired_channels = tuple(cfg.common_eeg_channels) if cfg.common_eeg_channels else None

    ordered_common: list[str] | None = list(desired_channels) if desired_channels is not None else None
    common_set: set[str] | None = set(desired_channels) if desired_channels is not None else None

    for base_dataset in dataset.datasets:
        raw = base_dataset.raw
        eeg_channels = [
            ch_name
            for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
            if ch_type == "eeg"
        ]
        eeg_set = set(eeg_channels)
        if common_set is None:
            common_set = set(eeg_channels)
            ordered_common = list(eeg_channels)
        else:
            common_set &= eeg_set
            ordered_common = [ch_name for ch_name in ordered_common if ch_name in eeg_set]

    common_channels = tuple(ch_name for ch_name in ordered_common if ch_name in common_set)
    if not common_channels:
        raise ValueError(f"Could not find a non-empty common EEG channel set for {cfg.dataset_name}.")

    if desired_channels is not None and common_channels != desired_channels:
        missing = [ch_name for ch_name in desired_channels if ch_name not in common_channels]
        raise ValueError(
            f"Configured common_eeg_channels are not available in every recording. Missing: {missing!r}"
        )

    return common_channels


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

    if protocol == "task_holdout":
        if "task_split" not in metadata.columns:
            raise ValueError("task_holdout evaluation requires a 'task_split' column in metadata")
        split_values = metadata["task_split"].astype(str).to_numpy()
        train_idx = indices[split_values == cfg.train_session_name]
        valid_idx = indices[split_values == cfg.valid_session_name]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(
                "task_holdout split produced an empty partition: "
                f"train_session_name={cfg.train_session_name!r}, valid_session_name={cfg.valid_session_name!r}"
            )
        return train_idx, valid_idx, {
            "protocol": protocol,
            "train_task_split": cfg.train_session_name,
            "valid_task_split": cfg.valid_session_name,
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

    if protocol == "subject_session_holdout":
        required_columns = {"subject", "session"}
        missing = required_columns.difference(metadata.columns)
        if missing:
            raise ValueError(
                "subject_session_holdout evaluation requires metadata columns: "
                + ", ".join(sorted(missing))
            )

        subject_values = metadata["subject"].to_numpy(copy=False)
        session_values = metadata["session"].to_numpy(copy=False)
        train_parts = []
        valid_parts = []
        heldout_session_by_subject: dict[str, str] = {}

        for subject in sorted(np.unique(subject_values).tolist(), key=_value_sort_key):
            subject_mask = subject_values == subject
            subject_sessions = sorted(
                np.unique(session_values[subject_mask]).tolist(),
                key=_value_sort_key,
            )
            if len(subject_sessions) < 2:
                continue

            heldout_session = subject_sessions[-1]
            train_mask = subject_mask & (session_values != heldout_session)
            valid_mask = subject_mask & (session_values == heldout_session)
            train_idx_subject = indices[train_mask]
            valid_idx_subject = indices[valid_mask]
            if len(train_idx_subject) == 0 or len(valid_idx_subject) == 0:
                continue

            train_parts.append(np.sort(train_idx_subject))
            valid_parts.append(np.sort(valid_idx_subject))
            heldout_session_by_subject[_stringify_value(subject)] = _stringify_value(heldout_session)

        if not train_parts or not valid_parts:
            raise ValueError("subject_session_holdout produced an empty partition")

        train_idx = np.concatenate(train_parts)
        valid_idx = np.concatenate(valid_parts)
        return np.sort(train_idx), np.sort(valid_idx), {
            "protocol": protocol,
            "heldout_session_by_subject": heldout_session_by_subject,
            "n_subjects": int(len(heldout_session_by_subject)),
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
                    "subject": subject,
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
                _value_sort_key(subject_specs[idx]["subject"]),
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
            subject = spec["subject"]
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
            subject_key = _stringify_value(subject)
            heldout_runs_by_subject[subject_key] = heldout_runs
            train_runs_by_subject[subject_key] = [str(run) for run in spec["train_runs"]] + extra_train_runs
            holdout_run_count_by_subject[subject_key] = n_holdout

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


def _finalize_bundle(windows_dataset, metadata, targets: np.ndarray, class_names: tuple[str, ...], cfg: BaselineConfig, split_summary: dict[str, object]) -> DatasetBundle:
    if cfg.ea_group_by_subject and "subject" in metadata.columns:
        groups = metadata["subject"].to_numpy(copy=False)
    else:
        groups = np.asarray(["global"] * len(metadata), dtype=object)

    train_idx, valid_idx, split_summary_core = _split_indices(
        metadata=metadata,
        targets=targets,
        cfg=cfg,
    )
    split_summary = {**split_summary, **split_summary_core}

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


def _load_moabb_windows(cfg: BaselineConfig) -> DatasetBundle:
    dataset = MOABBDataset(
        dataset_name=cfg.dataset_name,
        subject_ids=None if cfg.subject_ids is None else list(cfg.subject_ids),
    )

    preprocess(dataset, _normalize_preprocessors(cfg))

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
        preload=cfg.window_preload,
    )
    metadata = windows_dataset.get_metadata()
    targets, class_names = _resolve_targets(metadata, cfg)
    return _finalize_bundle(
        windows_dataset=windows_dataset,
        metadata=metadata,
        targets=targets,
        class_names=class_names,
        cfg=cfg,
        split_summary={"dataset_backend": "moabb"},
    )


def _load_tuh_windows(cfg: BaselineConfig) -> DatasetBundle:
    dataset_path = _resolve_tuh_path(cfg)
    dataset_key = cfg.dataset_name.strip().lower()
    dataset_cls = TUHAbnormal if dataset_key in {"tuhabnormal", "tuh_abnormal", "tuab"} else TUH
    target_name = "pathological" if dataset_cls is TUHAbnormal else None
    recording_ids, selection_summary = _select_tuh_recording_ids(cfg, dataset_path)
    dataset = dataset_cls(
        path=dataset_path,
        recording_ids=recording_ids,
        target_name=target_name,
        preload=False,
        rename_channels=True,
        set_montage=True,
        n_jobs=1,
    )
    common_channels = _resolve_common_eeg_channels(dataset, cfg)

    preprocessors = [
        Preprocessor(_pick_and_order_channels, apply_on_array=False, ch_names=common_channels),
        *_normalize_preprocessors(cfg),
    ]
    preprocess(dataset, preprocessors)

    window_size_samples = int(cfg.window_size_seconds * cfg.sfreq)
    if cfg.trialwise_decoding:
        window_stride_samples = window_size_samples
    else:
        window_stride_samples = int(cfg.window_stride_seconds * cfg.sfreq)

    windows_dataset = create_fixed_length_windows(
        dataset,
        start_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        preload=cfg.window_preload,
    )
    metadata = windows_dataset.get_metadata()
    targets, class_names = _resolve_targets(metadata, cfg)
    selection_summary = {
        **selection_summary,
        "dataset_backend": "braindecode_tuh",
        "n_common_eeg_channels": int(len(common_channels)),
        "common_eeg_channels": list(common_channels),
    }
    return _finalize_bundle(
        windows_dataset=windows_dataset,
        metadata=metadata,
        targets=targets,
        class_names=class_names,
        cfg=cfg,
        split_summary=selection_summary,
    )


def _load_hbn_windows(cfg: BaselineConfig) -> DatasetBundle:
    bids_root = _resolve_hbn_path(cfg)
    selected_records, selection_summary = _select_hbn_records(cfg, bids_root)

    base_datasets = []
    for record in selected_records:
        raw = _read_hbn_raw(str(record["path"]))
        description = {
            key: record[key]
            for key in (
                "subject",
                "session",
                "task_split",
                "task",
                "run",
                "qc_status",
                "extension",
                "path",
            )
        }
        base_datasets.append(BaseDataset(raw=raw, description=description, target_name=None))

    dataset = BaseConcatDataset(base_datasets)
    common_channels = _resolve_common_eeg_channels(dataset, cfg)
    if len(common_channels) < int(cfg.hbn_min_eeg_channels):
        raise ValueError(
            f"HBN common EEG channel count is {len(common_channels)}, below "
            f"hbn_min_eeg_channels={cfg.hbn_min_eeg_channels}."
        )

    preprocessors = [
        Preprocessor(_pick_and_order_channels, apply_on_array=False, ch_names=common_channels),
        *_normalize_preprocessors(cfg),
    ]
    preprocess(dataset, preprocessors)

    window_size_samples = int(cfg.window_size_seconds * cfg.sfreq)
    if cfg.trialwise_decoding:
        window_stride_samples = window_size_samples
    else:
        window_stride_samples = int(cfg.window_stride_seconds * cfg.sfreq)

    windows_dataset = create_fixed_length_windows(
        dataset,
        start_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=cfg.window_preload,
    )
    metadata = windows_dataset.get_metadata()
    targets, class_names = _resolve_targets(metadata, cfg)
    selection_summary = {
        **selection_summary,
        "n_common_eeg_channels": int(len(common_channels)),
        "common_eeg_channels": list(common_channels),
    }
    return _finalize_bundle(
        windows_dataset=windows_dataset,
        metadata=metadata,
        targets=targets,
        class_names=class_names,
        cfg=cfg,
        split_summary=selection_summary,
    )


def load_windows(cfg: BaselineConfig) -> DatasetBundle:
    dataset_key = cfg.dataset_name.strip().lower()
    if dataset_key in {"hbn", "hbn_eeg", "hbn-r1", "hbn_r1"}:
        return _load_hbn_windows(cfg)
    if dataset_key in {"tuh", "tuhabnormal", "tuh_abnormal", "tuab"}:
        return _load_tuh_windows(cfg)
    return _load_moabb_windows(cfg)


def load_moabb_windows(cfg: BaselineConfig) -> DatasetBundle:
    return load_windows(cfg)
