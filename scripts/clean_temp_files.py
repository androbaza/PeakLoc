from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove PeakLoc per-recording temporary localization arrays"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing per-recording output directories",
    )
    return parser.parse_args()


def remove_temp_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    removed = []
    for recording_dir in input_dir.iterdir():
        temp_dir = recording_dir / "temp_files"
        if not temp_dir.is_dir():
            continue
        for path in temp_dir.iterdir():
            if path.name.startswith(("localizations", "localization_qc", "rois")):
                path.unlink()
                removed.append(path)
    return removed


def main() -> None:
    args = parse_args()
    removed = remove_temp_files(args.input_dir)
    print(f"Removed {len(removed)} temporary files")


if __name__ == "__main__":
    main()
