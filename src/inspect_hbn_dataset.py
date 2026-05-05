from __future__ import annotations

import json

if __package__:
    from .data import inspect_hbn_dataset
    from .human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )
else:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data import inspect_hbn_dataset
    from src.human_recognition_config import (
        build_hbn_r1_l100_human_recognition_config,
        build_hbn_r1_l100_human_recognition_output_config,
    )


def main() -> dict:
    cfg = build_hbn_r1_l100_human_recognition_config()
    out_cfg = build_hbn_r1_l100_human_recognition_output_config()
    out_cfg.root.mkdir(parents=True, exist_ok=True)

    summary = inspect_hbn_dataset(cfg)
    summary_path = out_cfg.root / "hbn_dataset_inspection.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    result = main()
    compact = {
        "summary_path": result["summary_path"],
        "dataset_path": result["dataset_path"],
        "selected_subjects": result["selected_subjects"],
        "selected_recordings": result["selected_recordings"],
        "selected_train_recordings": result["selected_train_recordings"],
        "selected_valid_recordings": result["selected_valid_recordings"],
        "sample_raw": result.get("sample_raw"),
    }
    print(json.dumps(compact, indent=2))
