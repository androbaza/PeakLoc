from __future__ import annotations

import gc
from dataclasses import dataclass, field
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger
from natsort import natsorted

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.calibration import EventCalibration, load_calibration
from localization_scripts.event_array_processing import (
    array_to_polarity_map,
    array_to_time_map,
    create_convolved_signals,
    raw_events_to_array,
    save_dict,
)
from localization_scripts.localization_fitting import (
    localization_uncertainty_px,
    localize_rois_with_attempts,
)
from localization_scripts.peak_finding import (
    create_peak_lists,
    find_local_max_peak,
    find_peaks_parallel,
    group_timestamps_by_coordinate,
)
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plotting_functions import plot_rois_from_locs
from localization_scripts.roi_generation import generate_coord_lists, generate_rois
from localization_scripts.smlm_visualization import save_smlm_visualization


@dataclass
class SliceResult:
    time_slice: int
    event_count: int
    unique_peak_count: int
    roi_count: int
    localization_count: int
    elapsed_seconds: float
    fit_success_fraction: float | None = None
    median_uncertainty_px: float | None = None
    median_nll_per_event: float | None = None
    hot_pixel_fraction: float | None = None
    rejected_localization_count: int = 0
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class RecordingResult:
    input_file: Path
    output_folder: Path
    event_count: int
    time_min: int | None
    time_max: int | None
    slice_results: list[SliceResult] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    calibration_metadata: dict[str, object] = field(default_factory=dict)


def run_batch(config: PeakLocConfig) -> list[RecordingResult]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(config.input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}. Set PEAKLOC_INPUT_FOLDER "
            "or provide input_folder in the JSON config."
        )

    results = []
    for filename in natsorted(folder.iterdir()):
        if filename.is_dir() or filename.suffix == ".bias":
            continue
        if filename.suffix not in {".raw", ".npy"}:
            continue

        logger.info("Processing {}", filename)
        recording = process_recording(filename, config, run_timestamp)
        report_folder = recording.output_folder / "reports"
        settings_path = report_folder / f"peakloc_settings_{run_timestamp}.json"
        write_effective_run_settings(
            config, recording.calibration_metadata, settings_path
        )
        recording.artifacts.append(settings_path)
        write_run_report(recording, config, run_timestamp)
        results.append(recording)
    return results


def process_time_slice(
    event_slice: np.ndarray,
    time_slice: int,
    filename: Path,
    config: PeakLocConfig,
    calibration: EventCalibration,
) -> SliceResult | None:
    events = event_slice
    if events.size == 0:
        logger.info(
            "No events found in time slice ending at {} for {}", time_slice, filename
        )
        return None

    start_time = time.time()

    min_x = events["x"].min()
    min_y = events["y"].min()
    max_x = events["x"].max()
    max_y = events["y"].max()

    coords = generate_coord_lists(min_y, max_y, min_x, max_x)

    logger.info("Analyzing the data using {} cores", config.num_cores)
    logger.info(
        "Converting events to dictionaries; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    dict_events, max_len = array_to_polarity_map(events, coords)
    events_t_p_dict = array_to_time_map(events)
    del events
    gc.collect()

    logger.info(
        "Creating convolved signals; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    max_len = int(max_len * 2 * (config.convolution_roi_radius * 2 + 1) ** 2)
    times, cumsum, coordinates = create_convolved_signals(
        dict_events, coords, max_len, config.num_cores
    )
    del dict_events, max_len

    logger.info("Finding peaks; elapsed time: {:.2f} seconds", time.time() - start_time)
    peak_list = find_peaks_parallel(
        times,
        cumsum,
        coordinates,
        config.num_cores,
        prominence=config.prominence,
        interpolation_coefficient=config.interpolation_coefficient,
        cutoff_event_count=config.peak_min_event_count,
        spline_smooth=config.spline_smooth,
    )
    peaks, prominences, on_times, coordinates_peaks = create_peak_lists(peak_list)
    peaks_dict = group_timestamps_by_coordinate(
        coordinates_peaks, peaks, prominences, on_times
    )

    logger.info(
        "Filtering peaks; elapsed time: {:.2f} seconds", time.time() - start_time
    )
    unique_peaks = find_local_max_peak(
        peaks_dict,
        threshold=config.peak_time_threshold,
        neighbors=config.peak_neighbors,
    )

    out_folder_localizations = filename.with_suffix("")
    temp_files_localization = out_folder_localizations / "temp_files"
    out_folder_localizations.mkdir(parents=True, exist_ok=True)
    temp_files_localization.mkdir(parents=True, exist_ok=True)

    unique_peaks_path = (
        temp_files_localization
        / f"unique_peaks_fwhm_{config.dataset_fwhm:g}_prominence_{config.prominence:g}"
        f"_time_slice_{time_slice}.pkl"
    )
    save_dict(unique_peaks, str(unique_peaks_path))

    logger.info(
        "Generating ROIs; elapsed time: {:.2f} seconds", time.time() - start_time
    )
    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=config.roi_radius,
        min_x=min_x,
        min_y=min_y,
        num_cores=config.num_cores,
        max_x=config.sensor_width - 1,
        max_y=config.sensor_height - 1,
        polarity_time_gate_us=config.polarity_time_gate_us,
    )

    logger.info(
        "Performing localization; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    localization_tables = localize_rois_with_attempts(rois, config, calibration)
    attempted_localizations = localization_tables.attempted
    localizations = localization_tables.filtered

    logger.info(
        "Finished; total elapsed time: {:.2f} seconds", time.time() - start_time
    )
    attempted_localizations_path = (
        temp_files_localization
        / f"attempted_localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    localizations_path = (
        temp_files_localization
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    rois_path = (
        temp_files_localization / f"rois_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    np.save(attempted_localizations_path, attempted_localizations)
    np.save(localizations_path, localizations)
    np.save(rois_path, rois)

    unique_peak_count = sum(len(values) for values in unique_peaks.values())
    fit_qc = summarize_fit_qc(
        attempted_localizations,
        roi_count=len(rois),
        filtered_localization_count=len(localizations),
    )
    rejected_localization_count = fit_qc["rejected_localization_count"]
    if not isinstance(rejected_localization_count, int):
        rejected_localization_count = 0
    return SliceResult(
        time_slice=time_slice,
        event_count=len(event_slice),
        unique_peak_count=unique_peak_count,
        roi_count=len(rois),
        localization_count=len(localizations),
        elapsed_seconds=time.time() - start_time,
        fit_success_fraction=fit_qc["fit_success_fraction"],
        median_uncertainty_px=fit_qc["median_uncertainty_px"],
        median_nll_per_event=fit_qc["median_nll_per_event"],
        hot_pixel_fraction=fit_qc["hot_pixel_fraction"],
        rejected_localization_count=rejected_localization_count,
        artifacts=[
            unique_peaks_path,
            attempted_localizations_path,
            localizations_path,
            rois_path,
        ],
    )


def process_recording(
    filename: Path, config: PeakLocConfig, run_timestamp: str
) -> RecordingResult:
    recording_start = time.time()
    events = load_events(filename, config)
    if events is None:
        raise ValueError(f"Unsupported input file: {filename}")

    out_folder_localizations = filename.with_suffix("")
    out_folder_localizations.mkdir(parents=True, exist_ok=True)
    recording = RecordingResult(
        input_file=filename,
        output_folder=out_folder_localizations,
        event_count=len(events),
        time_min=int(events["t"].min()) if events.size else None,
        time_max=int(events["t"].max()) if events.size else None,
    )

    if events.size == 0:
        logger.info("No events found for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    calibration = load_calibration(
        config.calibration_path,
        config.sensor_shape,
        allow_uncalibrated=config.allow_uncalibrated,
    )
    recording.calibration_metadata = calibration_to_metadata(calibration)

    time_slices = range(
        config.slice_start + config.slice_duration,
        int(events["t"].max()) + config.slice_duration + 1,
        config.slice_duration,
    )
    if len(time_slices) == 0:
        logger.info("No time slices to process for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    for time_slice in time_slices:
        event_slice = events[
            (events["t"] >= time_slice - config.slice_duration)
            & (events["t"] < time_slice)
        ]
        slice_result = process_time_slice(
            event_slice,
            time_slice,
            filename,
            config,
            calibration,
        )
        if slice_result is not None:
            recording.slice_results.append(slice_result)
            recording.artifacts.extend(slice_result.artifacts)

    temp_files_localization = out_folder_localizations / "temp_files"
    if not temp_files_localization.is_dir():
        logger.info("No temporary localization folder found for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    sorted_names = natsorted(path.name for path in temp_files_localization.iterdir())
    loc_names = [name for name in sorted_names if name.startswith("localizations")]
    attempted_loc_names = [
        name for name in sorted_names if name.startswith("attempted_localizations")
    ]
    roi_names = [name for name in sorted_names if name.startswith("rois")]
    if not loc_names or not roi_names:
        logger.info("No localization outputs found for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    localizations_full_list = concatenate_localization_slices(
        temp_files_localization, loc_names
    )
    attempted_localizations_full_list = concatenate_localization_slices(
        temp_files_localization, attempted_loc_names
    )
    rois_full_list = None

    for roi_name in roi_names:
        roi_path = temp_files_localization / roi_name
        rois_slice = np.load(roi_path)

        if rois_slice.size == 0:
            logger.info("Skipping empty ROI slice {}", roi_path)
            continue

        rois_full_list = (
            np.concatenate((rois_full_list, rois_slice))
            if rois_full_list is not None
            else rois_slice
        )

    if localizations_full_list is None or rois_full_list is None:
        logger.info("No localization outputs found for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    localizations_path = (
        out_folder_localizations
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    attempted_localizations_path = (
        out_folder_localizations
        / f"attempted_localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    rois_path = (
        out_folder_localizations / f"rois_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    if attempted_localizations_full_list is not None:
        np.save(attempted_localizations_path, attempted_localizations_full_list)
        recording.artifacts.append(attempted_localizations_path)
    np.save(localizations_path, localizations_full_list)
    np.save(rois_path, rois_full_list)
    recording.artifacts.extend([localizations_path, rois_path])
    recording.artifacts.extend(
        save_processed_plots(
            localizations_full_list,
            out_folder_localizations,
            localizations_path,
            config,
            run_timestamp,
        )
    )

    if config.cleanup_temp_outputs:
        remove_temp_artifacts(recording, temp_files_localization, sorted_names)

    recording.elapsed_seconds = time.time() - recording_start
    return recording


def concatenate_localization_slices(
    temp_folder: Path, localization_names: list[str]
) -> np.ndarray | None:
    localizations_full_list = None
    next_id = 0

    for localization_name in localization_names:
        localization_path = temp_folder / localization_name
        localizations_slice = np.load(localization_path)

        if (
            localizations_slice.dtype.names is None
            or "id" not in localizations_slice.dtype.names
        ):
            raise ValueError(
                f"Localization file has no structured 'id' field: {localization_path}"
            )

        localizations_slice = localizations_slice.copy()
        if localizations_slice.size > 0:
            localizations_slice["id"] += next_id
            next_id = int(np.max(localizations_slice["id"])) + 1
        else:
            logger.info("Including empty localization slice {}", localization_path)

        localizations_full_list = (
            np.concatenate((localizations_full_list, localizations_slice))
            if localizations_full_list is not None
            else localizations_slice
        )

    return localizations_full_list


def load_events(filename: Path, config: PeakLocConfig) -> np.ndarray | None:
    basename = filename.name
    if basename.endswith(".raw"):
        return raw_events_to_array(
            str(filename), max_events=config.max_raw_events
        ).astype([("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")])
    if basename.endswith(".npy"):
        return np.load(filename)
    return None


def write_effective_run_settings(
    config: PeakLocConfig,
    calibration_metadata: dict[str, object],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["calibration"] = calibration_metadata
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def calibration_to_metadata(calibration: EventCalibration) -> dict[str, object]:
    return {
        "calibration_id": calibration.calibration_id,
        "calibrated": calibration.calibrated,
        "sensor_shape": list(calibration.sensor_shape),
        "sensor_model": calibration.sensor_model,
        "pixel_size_nm": calibration.pixel_size_nm,
    }


def summarize_fit_qc(
    localizations: np.ndarray,
    *,
    roi_count: int,
    filtered_localization_count: int | None = None,
) -> dict[str, float | int | None]:
    accepted_count = (
        int(localizations.size)
        if filtered_localization_count is None
        else filtered_localization_count
    )
    if localizations.size == 0 or localizations.dtype.names is None:
        return {
            "fit_success_fraction": None,
            "median_uncertainty_px": None,
            "median_nll_per_event": None,
            "hot_pixel_fraction": None,
            "rejected_localization_count": max(roi_count - accepted_count, 0),
        }
    names = set(localizations.dtype.names)
    fit_success_fraction = None
    median_uncertainty_px = None
    median_nll_per_event = None
    hot_pixel_fraction = None
    if "fit_success" in names:
        fit_success_fraction = float(np.mean(localizations["fit_success"]))
    if {"sigma_x", "sigma_y", "cov_xy"}.issubset(names):
        uncertainty = localization_uncertainty_px(localizations)
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if finite_uncertainty.size:
            median_uncertainty_px = float(np.median(finite_uncertainty))
    elif {"sigma_x", "sigma_y"}.issubset(names):
        uncertainty = np.sqrt(
            np.maximum(localizations["sigma_x"], 0) ** 2
            + np.maximum(localizations["sigma_y"], 0) ** 2
        )
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if finite_uncertainty.size:
            median_uncertainty_px = float(np.median(finite_uncertainty))
    if "nll_per_event" in names:
        finite_nll = localizations["nll_per_event"][
            np.isfinite(localizations["nll_per_event"])
        ]
        if finite_nll.size:
            median_nll_per_event = float(np.median(finite_nll))
    if {"hot_pixel_count", "valid_pixel_count"}.issubset(names):
        valid_count = int(np.sum(localizations["valid_pixel_count"]))
        if valid_count > 0:
            hot_pixel_fraction = float(
                np.sum(localizations["hot_pixel_count"]) / valid_count
            )
    return {
        "fit_success_fraction": fit_success_fraction,
        "median_uncertainty_px": median_uncertainty_px,
        "median_nll_per_event": median_nll_per_event,
        "hot_pixel_fraction": hot_pixel_fraction,
        "rejected_localization_count": max(roi_count - accepted_count, 0),
    }


def save_processed_plots(
    localizations: np.ndarray,
    out_folder: Path,
    localizations_path: Path,
    config: PeakLocConfig,
    timestamp: str,
) -> list[Path]:
    if localizations.size == 0:
        logger.info("Skipping plots because no localizations were produced")
        return []

    figure_folder = out_folder / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    artifacts = []

    roi_fit_figure = plot_rois_from_locs(
        localizations,
        subplotsize=config.plot_subplotsize,
        dataset_FWHM=config.dataset_fwhm,
    )
    if roi_fit_figure is not None:
        roi_fit_path = figure_folder / f"roi_fits_{timestamp}.png"
        roi_fit_figure.savefig(roi_fit_path, dpi=300, bbox_inches="tight")
        plt.close(roi_fit_figure)
        artifacts.append(roi_fit_path)
        logger.info("Saved ROI fit plot to {}", roi_fit_path)

    if config.plot_result:
        result = save_smlm_visualization(
            localizations,
            localizations_path,
            figure_folder,
            optical_pixel_size_nm=config.optical_pixel_size_nm,
            timestamp=timestamp,
        )
        if result is not None:
            artifacts.extend([result.png_path, result.tiff_path])
            logger.info("Saved SMLM result PNG to {}", result.png_path)
            logger.info("Saved SMLM result TIFF to {}", result.tiff_path)

    return artifacts


def remove_temp_artifacts(
    recording: RecordingResult, temp_folder: Path, sorted_names: list[str]
) -> None:
    removed_artifacts = set()
    for loc_file in sorted_names:
        if (
            loc_file.startswith("attempted_localizations")
            or loc_file.startswith("localizations")
            or loc_file.startswith("rois")
        ):
            temp_artifact = temp_folder / loc_file
            temp_artifact.unlink()
            removed_artifacts.add(temp_artifact)
    recording.artifacts = [
        artifact
        for artifact in recording.artifacts
        if artifact not in removed_artifacts
    ]


def write_run_report(
    recording: RecordingResult, config: PeakLocConfig, timestamp: str
) -> Path:
    report_folder = recording.output_folder / "reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / f"peakloc_report_{timestamp}.md"
    if report_path not in recording.artifacts:
        recording.artifacts.append(report_path)

    total_unique_peaks = sum(
        result.unique_peak_count for result in recording.slice_results
    )
    total_rois = sum(result.roi_count for result in recording.slice_results)
    total_localizations = sum(
        result.localization_count for result in recording.slice_results
    )

    lines = [
        "# PeakLoc Run Report",
        "",
        f"- Run timestamp: `{timestamp}`",
        f"- Input file: `{recording.input_file}`",
        f"- Output folder: `{recording.output_folder}`",
        f"- Input events: `{recording.event_count}`",
        f"- Event time range: `{recording.time_min}` to `{recording.time_max}`",
        f"- Processed slices: `{len(recording.slice_results)}`",
        f"- Total unique peaks: `{total_unique_peaks}`",
        f"- Total ROIs: `{total_rois}`",
        f"- Total localizations: `{total_localizations}`",
        f"- Elapsed time: `{recording.elapsed_seconds:.2f} s`",
        f"- Peak interpolation min events: `{config.peak_min_event_count}`",
        f"- Calibration ID: `{recording.calibration_metadata.get('calibration_id')}`",
        f"- Calibrated background: `{recording.calibration_metadata.get('calibrated')}`",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
        "## Slice Results",
        "",
    ]

    if recording.slice_results:
        lines.extend(
            [
                "| Time slice | Events | Unique peaks | ROIs | Localizations | "
                "Success | Unc. px | NLL/event | Hot px | Rejected | Seconds |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in recording.slice_results:
            lines.append(
                f"| {result.time_slice} | {result.event_count} | "
                f"{result.unique_peak_count} | {result.roi_count} | "
                f"{result.localization_count} | "
                f"{_format_optional_float(result.fit_success_fraction)} | "
                f"{_format_optional_float(result.median_uncertainty_px)} | "
                f"{_format_optional_float(result.median_nll_per_event)} | "
                f"{_format_optional_float(result.hot_pixel_fraction)} | "
                f"{result.rejected_localization_count} | {result.elapsed_seconds:.2f} |"
            )
    else:
        lines.append("No time slices produced localizations.")

    lines.extend(["", "## Artifacts", ""])
    if recording.artifacts:
        for artifact in recording.artifacts:
            lines.append(f"- `{artifact}`")
    else:
        lines.append("No output artifacts were generated.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved run report to {}", report_path)
    return report_path


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"
