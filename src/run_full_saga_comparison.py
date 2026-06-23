from __future__ import annotations

import os

from .run_attack_basis_comparison import run_attack_basis_comparison
from .run_hbn_human_recognition_attack_comparison import run_human_recognition_attack_comparison as run_hbn_human_recognition_attack_comparison
from .run_human_recognition_attack_comparison import run_human_recognition_attack_comparison


def run_full_saga_comparison() -> dict:
    print("================================================================")
    print(" EEG Backdoor: Full SAGA Comparison Run")
    print("================================================================")

    print("")
    print("[1/3] Running general attack basis comparison (includes saga_pgd)...")
    basis_summary = run_attack_basis_comparison()

    print("")
    print("[2/3] Running BNCI human-recognition comparison (includes human_saga_pgd)...")
    human_summary = run_human_recognition_attack_comparison()

    hbn_summary = None
    print("")
    if os.environ.get("EEG_HBN_PATH"):
        print("[3/3] Running HBN human-recognition comparison (includes human_saga_pgd)...")
        hbn_summary = run_hbn_human_recognition_attack_comparison()
    else:
        print("[3/3] Skipping HBN human-recognition comparison because EEG_HBN_PATH is not set.")
        print("      Set EEG_HBN_PATH to enable this step.")

    print("")
    print("================================================================")
    print(" Done. Results are in outputs/.")
    print("================================================================")

    return {
        "basis_summary": basis_summary,
        "human_summary": human_summary,
        "hbn_summary": hbn_summary,
    }


if __name__ == "__main__":
    run_full_saga_comparison()