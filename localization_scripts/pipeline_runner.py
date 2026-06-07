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

from localization_scripts.event_array_processing import (
    array_to_polarity_map,
    array_to_time_map,
    create_convolved_signals,
    raw_events_to_array,
    save_dict,
)
from localization_scripts.localization_fitting import perfrom_localization_parallel
from localization_scripts.peak_finding import (
    create_peak_lists,
    find_local_max_peak,
    find_peaks_parallel,
    group_timestamps_by_coordinate,
)
from localization_scripts.pipeline_config import PeakLocConfig, write_effective_config
from localization_scripts.plotting_functions import plot_3d_time, plot_rois_from_locs
from localization_scripts.roi_generation import generate_coord_lists, generate_rois


@dataclass
class SliceResult:
    time_slice: int
    event_count: int
    unique_peak_count: int
    roi_count: int
    localization_count: int
    elapsed_seconds: float
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
        write_effective_config(config, settings_path)
        recording.artifacts.append(settings_path)
        write_run_report(recording, config, run_timestamp)
        results.append(recording)
    return results


def process_time_slice(
    event_slice: np.ndarray, time_slice: int, filename: Path, config: PeakLocConfig
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
        max_x=max_x,
        max_y=max_y,
    )

    logger.info(
        "Performing localization; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    localizations = perfrom_localization_parallel(
        rois, dataset_FWHM=config.dataset_fwhm, num_cores=config.num_cores
    )

    logger.info(
        "Finished; total elapsed time: {:.2f} seconds", time.time() - start_time
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
    np.save(localizations_path, localizations)
    np.save(rois_path, rois)

    unique_peak_count = sum(len(values) for values in unique_peaks.values())
    return SliceResult(
        time_slice=time_slice,
        event_count=len(event_slice),
        unique_peak_count=unique_peak_count,
        roi_count=len(rois),
        localization_count=len(localizations),
        elapsed_seconds=time.time() - start_time,
        artifacts=[unique_peaks_path, localizations_path, rois_path],
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
        slice_result = process_time_slice(event_slice, time_slice, filename, config)
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
    roi_names = [name for name in sorted_names if name.startswith("rois")]
    if not loc_names or not roi_names:
        logger.info("No localization outputs found for {}", filename)
        recording.elapsed_seconds = time.time() - recording_start
        return recording

    localizations_full_list = None
    rois_full_list = None
    for loc_file in sorted_names:
        loc_path = temp_files_localization / loc_file
        if loc_file.startswith("localizations"):
            locs_slice = np.load(loc_path)
            if localizations_full_list is not None:
                locs_slice["id"] += np.max(localizations_full_list["id"])
            localizations_full_list = (
                np.concatenate((localizations_full_list, locs_slice))
                if localizations_full_list is not None
                else locs_slice
            )
        elif loc_file.startswith("rois"):
            rois_slice = np.load(loc_path)
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
    rois_path = (
        out_folder_localizations / f"rois_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    np.save(localizations_path, localizations_full_list)
    np.save(rois_path, rois_full_list)
    recording.artifacts.extend([localizations_path, rois_path])
    recording.artifacts.extend(
        save_processed_plots(
            localizations_full_list,
            out_folder_localizations,
            config,
            run_timestamp,
        )
    )

    if config.cleanup_temp_outputs:
        remove_temp_artifacts(recording, temp_files_localization, sorted_names)

    recording.elapsed_seconds = time.time() - recording_start
    return recording


def load_events(filename: Path, config: PeakLocConfig) -> np.ndarray | None:
    basename = filename.name
    if basename.endswith(".raw"):
        return raw_events_to_array(
            str(filename), max_events=config.max_raw_events
        ).astype([("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")])
    if basename.endswith(".npy"):
        return np.load(filename)
    return None


def save_processed_plots(
    localizations: np.ndarray, out_folder: Path, config: PeakLocConfig, timestamp: str
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

    localization = next(
        (
            loc
            for loc in localizations
            if np.any(loc["roi_event_times"]) and np.any(loc["roi_event_times_n"])
        ),
        None,
    )
    if localization is None:
        logger.info("Skipping ROI event-time plot because no timed ROI data was found")
        return artifacts

    roi_time_figure = plot_3d_time(
        localization["roi_event_times"],
        localization["roi_event_times_n"],
    )
    if roi_time_figure is None:
        return artifacts
    roi_time_path = figure_folder / f"roi_event_times_{timestamp}.png"
    roi_time_figure.savefig(roi_time_path, dpi=300, bbox_inches="tight")
    plt.close(roi_time_figure)
    artifacts.append(roi_time_path)
    logger.info("Saved ROI event-time plot to {}", roi_time_path)
    return artifacts


def remove_temp_artifacts(
    recording: RecordingResult, temp_folder: Path, sorted_names: list[str]
) -> None:
    removed_artifacts = set()
    for loc_file in sorted_names:
        if loc_file.startswith("localizations") or loc_file.startswith("rois"):
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
                "| Time slice | Events | Unique peaks | ROIs | Localizations | Seconds |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in recording.slice_results:
            lines.append(
                f"| {result.time_slice} | {result.event_count} | "
                f"{result.unique_peak_count} | {result.roi_count} | "
                f"{result.localization_count} | {result.elapsed_seconds:.2f} |"
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
