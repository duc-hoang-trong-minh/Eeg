from __future__ import annotations

if __package__:
    from .human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )
    from .run_human_recognition_attack_comparison import run_human_recognition_attack_comparison
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )
    from src.run_human_recognition_attack_comparison import run_human_recognition_attack_comparison


if __name__ == "__main__":
    summary = run_human_recognition_attack_comparison(
        baseline_cfg=build_hbn_r1_l100_human_recognition_config(),
        out_cfg=build_hbn_r1_l100_human_recognition_output_config(),
    )
    compact = {
        "report_path": summary["report_path"],
        "comparison_plot_path": summary["comparison_plot_path"],
        "channel_budget_plot_path": summary["channel_budget_plot_path"],
        "waveform_plot_path": summary["waveform_plot_path"],
        "n_clean_correct_total": summary["n_clean_correct_total"],
        "n_clean_correct_attacked": summary["n_clean_correct_attacked"],
    }
    print(compact)
