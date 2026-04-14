#!/usr/bin/env bash
# Full pipeline: collect attack patterns → train generators → evaluate
# Run from the project root: bash run_generator_pipeline.sh
set -euo pipefail

MAX_PER_SUBJECT=${1:-""} # optional: pass a number to limit trials per subject
N_EVAL_SAMPLES=${2:-10}  # samples used in the final comparison table

echo "================================================================"
echo " EEG Attack Generator Pipeline"
echo "================================================================"

# --- Step 1: Collect sample-wise attack patterns ---
echo ""
echo "[1/4] Collecting attack patterns (sample-wise SPSA)..."
if [ -n "$MAX_PER_SUBJECT" ]; then
    python -m src.collect_attack_patterns --max_per_subject "$MAX_PER_SUBJECT"
else
    python -m src.collect_attack_patterns
fi

# --- Step 2: Train subject-wise generators ---
echo ""
echo "[2/4] Training subject-wise attack generators..."
python -m src.train_attack_generator --scope subject

# --- Step 3: Train model-wise generator ---
echo ""
echo "[3/4] Training model-wise attack generator..."
python -m src.train_attack_generator --scope model

# --- Step 4: Evaluate and compare ---
echo ""
echo "[4/4] Evaluating: sample-wise SPSA vs subject-wise G vs model-wise G..."
python -m src.run_universal_attack_eval --scope both --n_samples "$N_EVAL_SAMPLES"

echo ""
echo "================================================================"
echo " Done. Results saved to outputs/universal_attack_eval.json"
echo "================================================================"
