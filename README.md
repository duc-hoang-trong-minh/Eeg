# EEG Sparse Score-Based Attack (Research Scaffold)

This repository provides a staged implementation scaffold for developing and evaluating a sparse score-based black-box perturbation pipeline on EEG classification models built with Braindecode.

## Scope and ethics

Use this code only for authorized security research in controlled environments.
Do not use it against real systems or data without explicit permission.

## Project structure

- `src/run_baseline.py`: Stage 1 baseline EEGNet training.
- `src/attack/support.py`: Stage 2 channel-window support representation.
- `src/attack/basis.py`: Stage 3 smooth raised-cosine basis.
- `src/attack/greedy_attack.py`: Stage 4 greedy score-based support search + Stage 5 SPSA refinement.
- `src/evaluate_attack.py`: Stage 6 attack evaluation over budget settings.
- `src/defense/lightweight.py`: Lightweight defense primitives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Stage 1 baseline

```bash
python -m src.run_baseline
```

Outputs:
- `outputs/eegconformer_baseline.pt`
- `outputs/baseline_metrics.json`
- `outputs/baseline_scores.npz`
- `outputs/train_val_loss_curve.png`
- `outputs/train_val_accuracy_curve.png`
- `outputs/learning_rate_curve.png`

## Run BNCI2014_001 human-recognition baseline

```bash
python -m src.run_human_recognition_baseline
```

Outputs:
- `outputs/bnci2014_001_human_recognition/subject_recognition_baseline.pt`
- `outputs/bnci2014_001_human_recognition/subject_recognition_metrics.json`
- `outputs/bnci2014_001_human_recognition/subject_recognition_scores.npz`
Notes:
- Uses subject-ID recognition labels.
- Uses a full cross-session split: trains on `0train` and validates on `1test`.

## Run TUHAbnormal human-recognition baseline

```bash
export EEG_TUHABNORMAL_PATH=/path/to/tuh_abnormal_root
python -m src.run_tuh_abnormal_human_recognition_baseline
```

Outputs:
- `outputs/tuh_abnormal_human_recognition/subject_recognition_baseline.pt`
- `outputs/tuh_abnormal_human_recognition/subject_recognition_metrics.json`
- `outputs/tuh_abnormal_human_recognition/subject_recognition_scores.npz`
Notes:
- Uses subject-ID recognition labels on the TUH Abnormal EEG corpus.
- Selects up to `2000` subjects with at least `2` sessions each.
- Uses a subject-session holdout split: earlier sessions for training, last session for validation.
- Requires a local TUH Abnormal EEG download path via `EEG_TUHABNORMAL_PATH` or `EEG_TUH_PATH`.

## Download HBN R1-L100-BDF

```bash
python scripts/download_hbn_release.py
export EEG_HBN_PATH=/home/necphy/data/hbn/R1_L100_bdf
```

Notes:
- Downloads the 100 Hz BDF version of HBN Release 1 from `https://sccn.ucsd.edu/download/eeg2025/R1_L100_bdf.zip`.
- Stores the zip at `/home/necphy/data/hbn/R1_L100_bdf.zip` and extracts under `/home/necphy/data/hbn/R1_L100_bdf`.
- The downloader resumes partial zip downloads when rerun.

## Inspect HBN R1-L100-BDF

```bash
export EEG_HBN_PATH=/home/necphy/data/hbn/R1_L100_bdf
python -m src.inspect_hbn_dataset
```

Output:
- `outputs/hbn_r1_l100_human_recognition/hbn_dataset_inspection.json`

Notes:
- Uses BIDS metadata and HBN task availability columns from `participants.tsv`.
- Defaults to `available_only` records.
- Uses passive tasks for training and active tasks for validation.

## Run HBN R1-L100 human-recognition baseline

```bash
export EEG_HBN_PATH=/home/necphy/data/hbn/R1_L100_bdf
EEG_HBN_MAX_SUBJECTS=20 EEG_HBN_EPOCHS=1 python -m src.run_hbn_human_recognition_baseline
python -m src.run_hbn_human_recognition_baseline
```

Outputs:
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_baseline.pt`
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_metrics.json`
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_scores.npz`

Notes:
- The first command is a smoke run on 20 subjects and 1 epoch.
- The second command uses all eligible R1 subjects unless `EEG_HBN_MAX_SUBJECTS` is set.
- HBN is used for subject-ID recognition with a passive-task train split and active-task validation split.
- Set `EEG_HBN_CHANNEL_LIMIT=64` to train on channels `E1`-`E64` when you want a smaller feature surface.

## Run HBN R1-L100 biometric evaluation

```bash
python -m src.run_hbn_human_recognition_biometric_eval
```

Outputs:
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_biometric_metrics.json`
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_biometric_curve.npz`
- `outputs/hbn_r1_l100_human_recognition/subject_recognition_biometric_far_frr.png`

## Run human-recognition attack comparison

```bash
python -m src.run_human_recognition_attack_comparison
```

Notes:
- Reuses the saved human-recognition checkpoint in `outputs/bnci2014_001_human_recognition/`.
- Compares sparse channel + hybrid waveform against sparse channel-time + hybrid waveform.
- Defaults to a balanced capped subset of clean-correct validation trials for runtime; override with `EEG_ATTACK_MAX_SAMPLES`.

For the HBN checkpoint:

```bash
python -m src.run_hbn_human_recognition_attack_comparison
```

## Run human-recognition biometric evaluation

```bash
python -m src.run_human_recognition_biometric_eval
```

Outputs:
- `outputs/bnci2014_001_human_recognition/subject_recognition_biometric_metrics.json`
- `outputs/bnci2014_001_human_recognition/subject_recognition_biometric_curve.npz`
- `outputs/bnci2014_001_human_recognition/subject_recognition_biometric_far_frr.png`
Notes:
- Reuses the saved human-recognition score file in `outputs/bnci2014_001_human_recognition/`.
- Computes pooled one-vs-rest biometric verification metrics: FAR, FRR, and EER.

## Run multi-seed baseline stability sweep

```bash
python -m src.run_multiseed_baseline
```

Outputs:
- `outputs/eegconformer_baseline_seed*.pt`
- `outputs/baseline_metrics_seed*.json`
- `outputs/baseline_scores_seed*.npz`
- `outputs/baseline_multiseed_summary.json`

## Run a single attack demo

```bash
python -m src.run_attack_demo
```

Output:
- `outputs/attack_demo_result.json`

## Run evaluation grid

```bash
python -m src.evaluate_attack
```

Output:
- `outputs/attack_eval_report.json`

## Notes

- Default data source is `BNCI2014_001` via MOABB/Braindecode.
- Default baseline preprocessing now combines a 4-38 Hz bandpass, exponential moving standardization, optional Euclidean Alignment, and training-only augmentation.
- Hyperparameters are defined in `src/config.py`.
- Attack evaluation now reports pre-defense ASR, post-denoising ASR, post suspicious-window filtering ASR, and query-budget exhaustion rate.
- This is a development scaffold with clean module boundaries so you can iterate each stage independently.
