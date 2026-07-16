from __future__ import annotations

import os
from pathlib import Path


RECORDING_SUFFIXES = frozenset({".raw", ".npy"})


def find_recording_files(input_folder: Path, *, recursive: bool) -> list[Path]:
    """Find input recordings while avoiding PeakLoc output directories."""
    if not recursive:
        return [
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix in RECORDING_SUFFIXES
        ]

    recordings: list[Path] = []
    for root, directory_names, file_names in os.walk(input_folder):
        root_path = Path(root)
        root_recordings = [
            root_path / file_name
            for file_name in file_names
            if Path(file_name).suffix in RECORDING_SUFFIXES
        ]
        recordings.extend(root_recordings)

        # Each recording writes into a sibling directory named after its stem.
        # Pruning those directories prevents subsequent recursive runs from
        # treating generated localization arrays as new input recordings.
        output_directory_names = {path.stem for path in root_recordings}
        directory_names[:] = [
            name for name in directory_names if name not in output_directory_names
        ]

    return recordings
