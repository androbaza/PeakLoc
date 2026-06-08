from __future__ import annotations

import argparse
import multiprocessing
import time
from pathlib import Path

import numpy as np
from loguru import logger

from localization_scripts.calibration import NullCalibration
from localization_scripts.event_array_processing import (
    array_to_time_map,
    load_dict,
    raw_events_to_array,
)
from localization_scripts.localization_fitting import localize_rois
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.roi_generation import generate_coord_lists, generate_rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROIs and localizations from a saved PeakLoc peaks dict"
    )
    parser.add_argument("input_file", type=Path, help="Raw or .npy event recording")
    parser.add_argument("peaks_file", type=Path, help="Pickled unique peaks dictionary")
    parser.add_argument("--prominence", type=float, default=12.0)
    parser.add_argument("--dataset-fwhm", type=float, default=7.0)
    parser.add_argument("--peak-time-threshold", type=float, default=50e3)
    parser.add_argument("--peak-neighbors", type=int, default=8)
    parser.add_argument("--roi-radius", type=int, default=8)
    parser.add_argument("--num-cores", type=int, default=multiprocessing.cpu_count())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    localize_from_peaks(
        input_file=args.input_file,
        peaks_file=args.peaks_file,
        prominence=args.prominence,
        dataset_fwhm=args.dataset_fwhm,
        peak_time_threshold=args.peak_time_threshold,
        peak_neighbors=args.peak_neighbors,
        roi_radius=args.roi_radius,
        num_cores=args.num_cores,
    )


def localize_from_peaks(
    *,
    input_file: Path,
    peaks_file: Path,
    prominence: float,
    dataset_fwhm: float,
    peak_time_threshold: float,
    peak_neighbors: int,
    roi_radius: int,
    num_cores: int,
) -> tuple[Path, Path]:
    start_time = time.time()
    events = _load_events(input_file)
    min_x = int(events["x"].min())
    min_y = int(events["y"].min())
    max_x = int(events["x"].max())
    max_y = int(events["y"].max())

    coords = generate_coord_lists(min_y, max_y, min_x, max_x)
    logger.info("Generated {} coordinates", len(coords))
    logger.info("Converting events to dictionaries")
    events_t_p_dict = array_to_time_map(events)
    unique_peaks = load_dict(str(peaks_file))

    output_folder = input_file.with_suffix("")
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating ROIs; elapsed time: {:.2f} seconds", time.time() - start_time
    )
    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=roi_radius,
        min_x=min_x,
        min_y=min_y,
        num_cores=num_cores,
        max_x=max_x,
        max_y=max_y,
    )

    config = PeakLocConfig(
        input_folder=str(input_file.parent),
        num_cores=num_cores,
        prominence=prominence,
        dataset_fwhm=dataset_fwhm,
        peak_time_threshold=peak_time_threshold,
        peak_neighbors=peak_neighbors,
        roi_radius=roi_radius,
        sensor_height=max_y + 1,
        sensor_width=max_x + 1,
    )
    calibration = NullCalibration(config.sensor_shape)

    logger.info("Performing localization")
    localizations = localize_rois(rois, config, calibration)
    logger.info(
        "Finished; total elapsed time: {:.2f} seconds", time.time() - start_time
    )

    localizations_path = output_folder / (
        f"localizations_prominence_fwhm_{dataset_fwhm:g}_prominence_{prominence:g}.npy"
    )
    rois_path = output_folder / (
        f"rois_prominence_fwhm_{dataset_fwhm:g}_prominence_{prominence:g}.npy"
    )
    np.save(localizations_path, localizations)
    np.save(rois_path, rois)
    return localizations_path, rois_path


def _load_events(input_file: Path) -> np.ndarray:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    if input_file.suffix == ".raw":
        return raw_events_to_array(str(input_file)).astype(
            [("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")]
        )
    if input_file.suffix == ".npy":
        return np.load(input_file)
    raise ValueError(f"Unsupported input file suffix: {input_file.suffix}")


if __name__ == "__main__":
    main()
