#!/usr/bin/env bash
# Full SAGA comparison run: basis comparison + human-recognition comparison + optional HBN pass.
# Run from anywhere: bash run_full_saga_comparison.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/.venv/bin/activate"

echo "================================================================"
echo " EEG Backdoor: Full SAGA Comparison Run"
echo "================================================================"

echo ""
echo "[1/3] Running general attack basis comparison (includes saga_pgd)..."
python -m src.run_attack_basis_comparison

echo ""
echo "[2/3] Running BNCI human-recognition comparison (includes human_saga_pgd)..."
python -m src.run_human_recognition_attack_comparison

echo ""
if [[ -n "${EEG_HBN_PATH:-}" ]]; then
    echo "[3/3] Running HBN human-recognition comparison (includes human_saga_pgd)..."
    python -m src.run_hbn_human_recognition_attack_comparison
else
    echo "[3/3] Skipping HBN human-recognition comparison because EEG_HBN_PATH is not set."
    echo "      Set EEG_HBN_PATH to enable this step."
fi

echo ""
echo "================================================================"
echo " Done. Results are in outputs/."
echo "================================================================"