from __future__ import annotations

if __package__:
    from .human_recognition_config import (
        build_tuh_abnormal_human_recognition_config,
        build_tuh_abnormal_human_recognition_output_config,
    )
    from .train_baseline import train_and_save_baseline
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.human_recognition_config import (
        build_tuh_abnormal_human_recognition_config,
        build_tuh_abnormal_human_recognition_output_config,
    )
    from src.train_baseline import train_and_save_baseline


if __name__ == "__main__":
    baseline_cfg = build_tuh_abnormal_human_recognition_config()
    output_cfg = build_tuh_abnormal_human_recognition_output_config()
    result = train_and_save_baseline(baseline_cfg, output_cfg)
    print(result)
