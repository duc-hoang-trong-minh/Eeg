from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request
import zipfile


DEFAULT_URL = "https://sccn.ucsd.edu/download/eeg2025/R1_L100_bdf.zip"
DEFAULT_ROOT = Path("/home/necphy/data/hbn/R1_L100_bdf")
CHUNK_SIZE = 8 * 1024 * 1024


def _format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def _download(url: str, zip_path: Path, force: bool) -> dict[str, object]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = 0 if force or not zip_path.exists() else zip_path.stat().st_size
    mode = "wb" if force or existing_size == 0 else "ab"

    headers = {}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = int(getattr(response, "status", 200))
        if existing_size and status != 206:
            existing_size = 0
            mode = "wb"

        content_length = int(response.headers.get("Content-Length", "0") or 0)
        expected_total = existing_size + content_length if status == 206 else content_length
        downloaded = existing_size
        last_reported_gib = downloaded // (1024 ** 3)

        print(
            f"Downloading {url}\n  -> {zip_path}\n"
            f"  starting at {_format_gib(existing_size)}"
            + (f" of {_format_gib(expected_total)}" if expected_total else ""),
            flush=True,
        )
        with zip_path.open(mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                current_gib = downloaded // (1024 ** 3)
                if current_gib > last_reported_gib:
                    last_reported_gib = current_gib
                    if expected_total:
                        pct = 100.0 * downloaded / expected_total
                        print(f"  downloaded {_format_gib(downloaded)} ({pct:.1f}%)", flush=True)
                    else:
                        print(f"  downloaded {_format_gib(downloaded)}", flush=True)

    final_size = zip_path.stat().st_size
    return {
        "url": url,
        "zip_path": str(zip_path),
        "bytes": int(final_size),
        "size_gib": final_size / (1024 ** 3),
    }


def _extract(zip_path: Path, root: Path, force: bool) -> dict[str, object]:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Downloaded file is not a valid zip archive: {zip_path}")

    root.mkdir(parents=True, exist_ok=True)
    participants_paths = list(root.glob("**/participants.tsv"))
    if participants_paths and not force:
        print(f"Extraction already appears present under {root}; skipping.", flush=True)
        return {"root": str(root), "skipped": True, "participants_paths": [str(p) for p in participants_paths]}

    print(f"Extracting {zip_path} -> {root}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)

    participants_paths = list(root.glob("**/participants.tsv"))
    return {"root": str(root), "skipped": False, "participants_paths": [str(p) for p in participants_paths]}


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Download and optionally extract HBN R1-L100-BDF.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args(argv)

    zip_path = args.zip_path or args.root.with_suffix(".zip")
    download = _download(args.url, zip_path, force=args.force_download)

    extract = None
    if not args.no_extract:
        extract = _extract(zip_path, args.root, force=args.force_extract)

    manifest = {
        "download": download,
        "extract": extract,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.root.parent / "R1_L100_bdf_download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume the download.", file=sys.stderr)
        raise
