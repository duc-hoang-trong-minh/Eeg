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
- Set `EEG_HBN_OUTPUT_ROOT=/path/to/output` to keep a run isolated from the default `outputs/hbn_r1_l100_human_recognition` folder.

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
- Compares sparse channel + hybrid waveform, sparse channel-time + hybrid waveform, and a SAGA-style sparse channel-time PGD baseline.
- Defaults to a balanced capped subset of clean-correct validation trials for runtime; override with `EEG_ATTACK_MAX_SAMPLES`.
- To run only the SAGA-style baseline, set `EEG_ATTACK_VARIANTS=human_saga_pgd`.
- The SAGA variant is a paper-guided white-box reconstruction from the public abstract/metadata, not a vendored upstream implementation.

## Run attack basis comparison

```bash
python -m src.run_attack_basis_comparison
```

Notes:
- Compares unrestricted hybrid basis, restricted frequency bank, channel-then-window hybrid basis, and a SAGA-style sparse channel-time PGD baseline.
- To run only the SAGA-style baseline, set `EEG_ATTACK_VARIANTS=saga_pgd`.

## Run full SAGA comparison

```bash
python -m src.run_full_saga_comparison
```

Or use the shell wrapper:

```bash
bash run_full_saga_comparison.sh
```

Notes:
- Runs the basis comparison, the BNCI human-recognition comparison, and the HBN comparison when `EEG_HBN_PATH` is set.
- Do not append `.sh` to the `python -m` command; that name is the Python module entrypoint.

For the HBN checkpoint:

```bash
python -m src.run_hbn_human_recognition_attack_comparison
```

Recent sparse-channel HBN result:
- Fresh HBN R1-L100 sparse-channel hybrid rerun: `outputs/hbn_r1_l100_human_recognition/hbn_k12_first_success_channel_summary.json`.
- Model/task: EEGConformer subject recognition, task-holdout HBN split, 50 subjects, 64 EEG channels.
- Attack setting: channel-first hybrid waveform, `support_budget_k=12`, `max_outer_iters=12`, `max_query_budget=45000`, 10% peak-ratio cap.
- Sample set: 64 balanced clean-correct validation samples, drawn from 19,993 clean-correct HBN validation windows.
- Result: 64/64 attack success. The average first successful channel count was 3.671875 channels, median 3, min 1, max 12.
- First-success channel distribution: 1 channel: 14 samples; 2: 14; 3: 12; 4: 5; 5: 6; 6: 3; 7: 1; 8: 5; 9: 2; 12: 2.
- Prefix success: K=8 flipped 93.75%, K=9 flipped 96.875%, and K=12 flipped 100%.

Comparison to the older 9-subject BNCI2014_001 run:
- BNCI2014_001 artifact: `outputs/bnci2014_001_human_recognition/human_recognition_attack_basis_comparison_report.json`.
- BNCI setting: EEGConformer subject recognition, cross-session split, 9 subjects, 22 EEG channels, 10 attacked clean-correct samples.
- BNCI attack setting: channel-first hybrid waveform, `support_budget_k=8`, 5% peak-ratio cap, `max_query_budget=25000`, default regularization.
- BNCI result: 8/10 attack success by K=8. Among successful samples, first success averaged 2.5 channels with median 2; two samples did not flip within the K=8 budget.
- The HBN result is therefore stronger in final success rate, but it is not a pure dataset-only comparison: HBN used a larger search/perturbation budget, more available channels, and weaker regularization. It also attacks a 50-subject task-holdout identity model where subject-specific cues can be spread across many electrodes; the greedy search has more possible channels to exploit. BNCI2014_001 has only 22 channels and was attacked under a stricter K=8/5% cap, so the older result should be treated as a stricter, smaller-dataset reference point rather than a direct control.

## Run HBN subject-count sweep

```bash
python -m src.run_hbn_subject_sweep
```

Default sweep:
- Uses HBN R1-L100-BDF only; no R2-R11 download is required.
- Subject counts: `20, 35, 50, 75, 100, 122`.
- Output root: `outputs/hbn_subject_sweep/`, with one isolated folder per count (`n020`, `n035`, ...).
- Fixed channels: `E1`-`E64`.
- Attack setting: sparse channel + hybrid waveform, `K=12`, 10% peak-ratio cap, `max_query_budget=45000`.
- Attack sample count defaults to `min(128, max(64, subject_count))`.

Useful variants:

```bash
# Verify commands and output roots without running training or attack.
python -m src.run_hbn_subject_sweep --dry-run --subject-counts 20

# Run only the 122-subject count.
python -m src.run_hbn_subject_sweep --subject-counts 122

# Rebuild only the cross-run summary plot/table from completed run folders.
python -m src.plot_hbn_subject_sweep --root outputs/hbn_subject_sweep

# Run added architecture-validation sweeps in isolated model folders.
python -m src.run_hbn_subject_sweep \
  --root outputs/hbn_model_sweep \
  --model-names EEGNet ShallowFBCSPNet

# Rebuild a multi-model summary plot/table.
python -m src.plot_hbn_subject_sweep --root outputs/hbn_model_sweep
```

Outputs:
- `outputs/hbn_subject_sweep/hbn_subject_count_channel_control_summary.csv`
- `outputs/hbn_subject_sweep/hbn_subject_count_channel_control_summary.json`
- `outputs/hbn_subject_sweep/hbn_subject_count_channel_control.png`

The summary reports clean validation accuracy, biometric EER, attack success, mean/median first-success channel count, bootstrap confidence intervals, and `K90`/`K95`/`K100`. Treat the subject-count correlation as exploratory because it has six nested HBN R1 points, not independent datasets.
For multi-model runs, the summary CSV includes `model_name`, and the plot overlays mean first-success channels by architecture.

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
