#!/usr/bin/env bash
# Run HBN attack-side baselines against an existing HBN subject-recognition checkpoint.
#
# Defaults target the hardest paper sweep point:
#   source: outputs/hbn_subject_sweep/n122
#   output: outputs/hbn_sota_attack_comparison/n122
#
# Override examples:
#   HBN_ATTACK_SOURCE_ROOT=outputs/hbn_r1_l100_human_recognition bash run_hbn_sota_attack_comparison.sh
#   HBN_ATTACK_VARIANTS=human_saga_pgd,human_qeldba bash run_hbn_sota_attack_comparison.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SOURCE_ROOT="${HBN_ATTACK_SOURCE_ROOT:-outputs/hbn_subject_sweep/n122}"
TARGET_ROOT="${HBN_ATTACK_OUTPUT_ROOT:-outputs/hbn_sota_attack_comparison/n122}"

if [[ ! -f "$SOURCE_ROOT/subject_recognition_baseline.pt" ]]; then
    echo "Missing HBN checkpoint: $SOURCE_ROOT/subject_recognition_baseline.pt" >&2
    echo "Set HBN_ATTACK_SOURCE_ROOT to a completed HBN output folder." >&2
    exit 1
fi

mkdir -p "$TARGET_ROOT"

copy_if_present() {
    local filename="$1"
    if [[ -f "$SOURCE_ROOT/$filename" ]]; then
        cp -p "$SOURCE_ROOT/$filename" "$TARGET_ROOT/$filename"
    fi
}

# The attack runner expects the baseline checkpoint under EEG_HBN_OUTPUT_ROOT.
copy_if_present subject_recognition_baseline.pt
copy_if_present subject_recognition_metrics.json
copy_if_present subject_recognition_scores.npz
copy_if_present subject_recognition_biometric_metrics.json
copy_if_present hbn_subject_sweep_run_manifest.json

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
fi

export EEG_HBN_OUTPUT_ROOT="$TARGET_ROOT"
export EEG_ATTACK_USE_CHECKPOINT_CONFIG="${EEG_ATTACK_USE_CHECKPOINT_CONFIG:-1}"
export EEG_ATTACK_VARIANTS="${HBN_ATTACK_VARIANTS:-human_sparse_channel_hybrid,human_saga_pgd,human_qeldba}"

# Match the n122 paper sweep budget unless the caller overrides a value.
export EEG_ATTACK_MAX_SAMPLES="${EEG_ATTACK_MAX_SAMPLES:-122}"
export EEG_ATTACK_SUPPORT_BUDGET_K="${EEG_ATTACK_SUPPORT_BUDGET_K:-12}"
export EEG_ATTACK_MAX_OUTER_ITERS="${EEG_ATTACK_MAX_OUTER_ITERS:-12}"
export EEG_ATTACK_MAX_QUERY_BUDGET="${EEG_ATTACK_MAX_QUERY_BUDGET:-45000}"
export EEG_ATTACK_MAX_PEAK_RATIO="${EEG_ATTACK_MAX_PEAK_RATIO:-0.10}"
export EEG_ATTACK_MAX_COEFF_ABS="${EEG_ATTACK_MAX_COEFF_ABS:-7.5}"
export EEG_ATTACK_CANDIDATE_PROBE_RESTARTS="${EEG_ATTACK_CANDIDATE_PROBE_RESTARTS:-4}"
export EEG_ATTACK_SPSA_STEPS="${EEG_ATTACK_SPSA_STEPS:-180}"
export EEG_ATTACK_SPSA_RESTARTS="${EEG_ATTACK_SPSA_RESTARTS:-4}"
export EEG_ATTACK_L2_WEIGHT="${EEG_ATTACK_L2_WEIGHT:-1e-8}"
export EEG_ATTACK_TV_WEIGHT="${EEG_ATTACK_TV_WEIGHT:-1e-8}"
export EEG_ATTACK_BAND_WEIGHT="${EEG_ATTACK_BAND_WEIGHT:-1e-8}"

echo "================================================================"
echo " HBN SOTA Attack Comparison"
echo "================================================================"
echo "Source checkpoint root: $SOURCE_ROOT"
echo "Output root:            $TARGET_ROOT"
echo "Variants:               $EEG_ATTACK_VARIANTS"
echo "Samples:                $EEG_ATTACK_MAX_SAMPLES"
echo "Budget:                 K=$EEG_ATTACK_SUPPORT_BUDGET_K, queries=$EEG_ATTACK_MAX_QUERY_BUDGET, peak=$EEG_ATTACK_MAX_PEAK_RATIO"
echo "================================================================"

python -m src.run_hbn_human_recognition_attack_comparison

echo ""
echo "Wrote report:"
echo "  $TARGET_ROOT/human_recognition_attack_basis_comparison_report.json"
echo ""
if command -v jq >/dev/null 2>&1; then
    jq -r '.variants[] | "\(.display_name): ASR=\((.attack_success_rate * 100))%, attacked_acc=\((.attacked_accuracy * 100))%, mean_k=\(.adaptive_channel_summary.mean), median_k=\(.adaptive_channel_summary.median)"' \
        "$TARGET_ROOT/human_recognition_attack_basis_comparison_report.json"
fi
