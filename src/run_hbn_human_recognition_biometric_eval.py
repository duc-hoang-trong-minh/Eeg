from __future__ import annotations

if __package__:
    from .human_recognition_config import build_hbn_r1_l100_human_recognition_output_config
    from .run_human_recognition_biometric_eval import run_biometric_eval
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.human_recognition_config import build_hbn_r1_l100_human_recognition_output_config
    from src.run_human_recognition_biometric_eval import run_biometric_eval


if __name__ == "__main__":
    result = run_biometric_eval(
        out_cfg=build_hbn_r1_l100_human_recognition_output_config(),
        title="HBN R1-L100 Subject Verification",
    )
    compact = {
        "summary_path": result["summary_path"],
        "figure_path": result["figure_path"],
        "pooled_eer_pct": result["pooled_eer_pct"],
        "eer_threshold": result["eer_threshold"],
        "macro_subject_eer_pct": result["macro_subject_eer_pct"],
    }
    print(compact)
