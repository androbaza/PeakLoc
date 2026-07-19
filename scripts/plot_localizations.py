from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from localization_scripts.artifact_layout import ArtifactLayout
from localization_scripts.pipeline_config import load_peakloc_config
from localization_scripts.smlm_visualization import save_smlm_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PeakLoc localizations as an SMLM result image"
    )
    parser.add_argument(
        "localizations",
        nargs="?",
        type=Path,
        help="Path to a PeakLoc localizations .npy file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON PeakLoc configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for rendered PNG and TIFF outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_peakloc_config(args.config)
    localizations_path = args.localizations or ask_for_localizations_path()
    localizations = np.load(localizations_path)
    default_output_dir, crop_to_data, output_stem = _default_output_settings(
        localizations_path
    )
    output_dir = args.output_dir or default_output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = save_smlm_visualization(
        localizations,
        localizations_path,
        output_dir,
        optical_pixel_size_nm=config.optical_pixel_size_nm,
        output_stem=output_stem,
        crop_to_data=crop_to_data,
        timestamp=timestamp,
        sensor_shape=config.sensor_shape,
    )
    if result is None:
        logger.info("No localizations to render in {}", localizations_path)
        return

    logger.info("Saved 8-bit PNG SMLM preview to {}", result.png_path)
    logger.info("Saved 12-bit TIFF SMLM render to {}", result.tiff_path)
    logger.info(
        "Rendered {} localizations at {:.2f} nm/pixel",
        result.localization_count,
        result.render_pixel_size_nm,
    )


def _default_output_settings(
    localizations_path: Path,
) -> tuple[Path, bool, str | None]:
    """Choose a shareable output location for arrays from a structured run."""
    arrays_directory = localizations_path.parent
    if arrays_directory.name != "arrays" or arrays_directory.parent.name != "debug":
        return arrays_directory / "figures", False, None
    layout = ArtifactLayout.from_run_directory(arrays_directory.parent.parent)
    if (
        arrays_directory == layout.debug_arrays_dir
        and (layout.share_metadata_dir / "run_metadata.json").is_file()
    ):
        return layout.share_figures_dir, True, None
    return arrays_directory / "figures", False, None


def ask_for_localizations_path() -> Path:
    path = Path(input("Path to localizations .npy file: ").strip()).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Localization file does not exist: {path}")
    return path


if __name__ == "__main__":
    main()
