"""Summarize PeakLoc dead-time calibration runs against a 50 Hz square wave."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.plot_style import (
    DOUBLE_COLUMN_WIDTH_IN,
    PANEL_LABEL_SIZE,
    PLOT_COLORS,
    PUBLICATION_DPI,
    save_publication_figure,
    style_publication_axis,
)


PERIOD_US = 20_000
BRIGHT_DURATION_US = 10_000
SLICE_START_US = 1_000_000
SLICE_STOP_US = 5_000_000
EXPECTED_CYCLE_COUNT = (SLICE_STOP_US - SLICE_START_US) // PERIOD_US
BEAD_ASSIGNMENT_RADIUS_PX = 10.0
CALIBRATED_PEAK_THRESHOLD_US = 10_000.0
CALIBRATED_PEAK_NEIGHBORS = 9


def numeric_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected a numeric value, got {type(value).__name__}")


@dataclass(frozen=True)
class DeadTimeSetting:
    name: str
    bias_refr: int
    recording_name: str


SETTINGS = (
    DeadTimeSetting("Maximum", -20, "recording_2026-07-24_10-20-24"),
    DeadTimeSetting("Default", 0, "recording_2026-07-24_10-18-39"),
    DeadTimeSetting("Setting 127", 127, "recording_2026-07-24_10-21-47"),
    DeadTimeSetting("Minimum", 235, "recording_2026-07-24_10-23-03"),
)


@dataclass(frozen=True)
class RunData:
    setting: DeadTimeSetting
    method: str
    run_directory: Path
    rois: np.ndarray
    localization_count: int
    slice_metrics: dict[str, Any]


@dataclass(frozen=True)
class ReferenceGrid:
    phase_us: float
    cycle_indices: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibrated legacy and temporal PeakLoc runs against a "
            "50 Hz, 50% duty-cycle reference."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/media/angelina/data1/andrey/peakloc_data/2026_07_23_Dead_Time_Calibration"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dead-time-calibration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.dataset_root)
    default_legacy = runs[("Default", "legacy")]
    bead_centers = infer_bead_centers(default_legacy.rois["peak"])
    references = {
        setting.name: reference_grid(runs[(setting.name, "legacy")].rois["t_peak"])
        for setting in SETTINGS
    }

    summary_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        reference = references[setting.name]
        for method in ("legacy", "transition_train"):
            run = runs[(setting.name, method)]
            summary, measurements = summarize_run(run, bead_centers, reference)
            summary_rows.append(summary)
            measurement_rows.extend(measurements)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dead_time_calibration_summary.csv", summary_rows)
    write_csv(output_dir / "dead_time_calibration_measurements.csv", measurement_rows)
    save_figure(summary_rows, measurement_rows, output_dir)
    write_manifest(runs, bead_centers, references, output_dir)


def discover_runs(dataset_root: Path) -> dict[tuple[str, str], RunData]:
    runs = {}
    for setting in SETTINGS:
        recording_dir = (
            dataset_root / setting.name.replace(" ", "_") / setting.recording_name
        )
        if not recording_dir.is_dir():
            raise FileNotFoundError(
                f"Recording output directory not found: {recording_dir}"
            )
        for method in ("legacy", "transition_train"):
            run_directory = find_latest_matching_run(recording_dir, method)
            runs[(setting.name, method)] = load_run(setting, method, run_directory)
    return runs


def find_latest_matching_run(recording_dir: Path, method: str) -> Path:
    matches = []
    for candidate in recording_dir.iterdir():
        config_path = candidate / "share" / "metadata" / "effective_config.json"
        if not config_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as input_file:
            config = json.load(input_file)
        if run_matches(config, method):
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"No calibrated {method} run found below {recording_dir}"
        )
    return max(matches, key=lambda path: path.name)


def run_matches(config: dict[str, object], method: str) -> bool:
    if config.get("slice_start") != SLICE_START_US:
        return False
    if config.get("slice_end") != SLICE_STOP_US:
        return False
    if (
        numeric_value(config.get("peak_time_threshold", -1.0))
        != CALIBRATED_PEAK_THRESHOLD_US
    ):
        return False
    if config.get("peak_neighbors") != CALIBRATED_PEAK_NEIGHBORS:
        return False
    temporal = bool(config.get("temporal_segmentation_enabled"))
    if method == "legacy":
        return not temporal
    return (
        temporal
        and config.get("temporal_context_pre_us") == 30_000
        and config.get("temporal_context_post_us") == 30_000
    )


def load_run(setting: DeadTimeSetting, method: str, run_directory: Path) -> RunData:
    array_dir = run_directory / "debug" / "arrays"
    report_dir = run_directory / "debug" / "reports"
    rois = np.load(single_path(array_dir, "rois*.npy"), allow_pickle=False)
    localizations = np.load(
        single_path(array_dir, "localizations*.npy"), allow_pickle=False
    )
    with single_path(report_dir, "slice_metrics*.json").open(
        encoding="utf-8"
    ) as input_file:
        metrics = json.load(input_file)
    if len(metrics) != 1:
        raise ValueError(f"Expected one 1–5 s slice in {run_directory}")
    return RunData(
        setting=setting,
        method=method,
        run_directory=run_directory,
        rois=rois,
        localization_count=len(localizations),
        slice_metrics=metrics[0],
    )


def single_path(directory: Path, pattern: str) -> Path:
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one path matching {pattern!r} in {directory}, found {len(paths)}"
        )
    return paths[0]


def infer_bead_centers(peak_coordinates: np.ndarray) -> np.ndarray:
    unique_coordinates = np.unique(
        np.asarray(peak_coordinates, dtype=np.float64), axis=0
    )
    unused = set(range(len(unique_coordinates)))
    components = []
    while unused:
        seed = unused.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            candidates = list(unused)
            distances = np.linalg.norm(
                unique_coordinates[candidates] - unique_coordinates[current], axis=1
            )
            neighbors = [
                candidate
                for candidate, distance in zip(candidates, distances)
                if distance <= BEAD_ASSIGNMENT_RADIUS_PX
            ]
            for neighbor in neighbors:
                unused.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(sorted(component))
    centers = np.asarray(
        [unique_coordinates[component].mean(axis=0) for component in components]
    )
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def reference_grid(peak_times: np.ndarray) -> ReferenceGrid:
    phase_us = circular_phase(np.asarray(peak_times, dtype=np.float64), PERIOD_US)
    first_index = math.ceil((SLICE_START_US - phase_us) / PERIOD_US)
    last_index = math.floor((SLICE_STOP_US - 1 - phase_us) / PERIOD_US)
    cycle_indices = frozenset(range(first_index, last_index + 1))
    if len(cycle_indices) != EXPECTED_CYCLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CYCLE_COUNT} reference cycles, got {len(cycle_indices)}"
        )
    return ReferenceGrid(phase_us=phase_us, cycle_indices=cycle_indices)


def circular_phase(times_us: np.ndarray, period_us: int) -> float:
    angles = times_us / period_us * 2.0 * np.pi
    mean_vector = np.mean(np.exp(1j * angles))
    phase = np.angle(mean_vector) / (2.0 * np.pi) * period_us
    return float(phase % period_us)


def wrapped_residual(
    times_us: np.ndarray, phase_us: float, period_us: int
) -> np.ndarray:
    return (times_us - phase_us + period_us / 2) % period_us - period_us / 2


def summarize_run(
    run: RunData,
    bead_centers: np.ndarray,
    reference: ReferenceGrid,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rois = run.rois
    peak_times = rois["t_peak"].astype(np.int64)
    bead_indices, spatial_distances = assign_beads(rois["peak"], bead_centers)
    cycle_indices = np.rint((peak_times - reference.phase_us) / PERIOD_US).astype(int)
    marker_residuals = wrapped_residual(
        peak_times.astype(np.float64), reference.phase_us, PERIOD_US
    )
    valid = (
        (bead_indices >= 0)
        & np.isin(cycle_indices, list(reference.cycle_indices))
        & (np.abs(marker_residuals) <= CALIBRATED_PEAK_THRESHOLD_US)
    )
    keys = [
        (int(bead_index), int(cycle_index))
        for bead_index, cycle_index, keep in zip(bead_indices, cycle_indices, valid)
        if keep
    ]
    matched_count = len(set(keys))
    duplicate_count = len(keys) - matched_count
    expected_detection_count = len(bead_centers) * len(reference.cycle_indices)

    starts, stops, boundary_definition = boundary_times(run)
    durations = stops - starts
    duration_errors = durations - BRIGHT_DURATION_US
    start_phase = circular_phase(starts.astype(np.float64), PERIOD_US)
    stop_phase = circular_phase(stops.astype(np.float64), PERIOD_US)
    start_residuals = wrapped_residual(
        starts.astype(np.float64), start_phase, PERIOD_US
    )
    stop_residuals = wrapped_residual(stops.astype(np.float64), stop_phase, PERIOD_US)
    total_roi_events = rois["total_events_roi"].astype(np.int64) + rois[
        "total_neg_events_roi"
    ].astype(np.int64)

    summary: dict[str, object] = {
        "setting": run.setting.name,
        "bias_refr": run.setting.bias_refr,
        "method": run.method,
        "run_directory": str(run.run_directory),
        "boundary_definition": boundary_definition,
        "event_count_1_to_5_s": int(run.slice_metrics["event_count"]),
        "candidate_count": int(run.slice_metrics["unique_peak_count"]),
        "roi_count": len(rois),
        "localization_count": run.localization_count,
        "expected_bead_cycle_count": expected_detection_count,
        "matched_bead_cycle_count": matched_count,
        "bead_cycle_recall_percent": matched_count / expected_detection_count * 100.0,
        "duplicate_count": duplicate_count,
        "unassigned_count": int(np.count_nonzero(~valid)),
        "marker_absolute_error_median_us": float(np.median(np.abs(marker_residuals))),
        "marker_absolute_error_p90_us": float(
            np.quantile(np.abs(marker_residuals), 0.9)
        ),
        "start_jitter_median_us": float(np.median(np.abs(start_residuals))),
        "stop_jitter_median_us": float(np.median(np.abs(stop_residuals))),
        "duration_median_ms": float(np.median(durations) * 1e-3),
        "duration_p10_ms": float(np.quantile(durations, 0.1) * 1e-3),
        "duration_p90_ms": float(np.quantile(durations, 0.9) * 1e-3),
        "duration_absolute_error_median_us": float(np.median(np.abs(duration_errors))),
        "duration_absolute_error_p90_us": float(
            np.quantile(np.abs(duration_errors), 0.9)
        ),
        "roi_event_count_median": float(np.median(total_roi_events)),
        "fit_yield_percent": run.localization_count / max(len(rois), 1) * 100.0,
        "processing_seconds": float(run.slice_metrics["elapsed_seconds"]),
    }

    measurements: list[dict[str, object]] = []
    for index in range(len(rois)):
        measurements.append(
            {
                "setting": run.setting.name,
                "bias_refr": run.setting.bias_refr,
                "method": run.method,
                "roi_index": index,
                "bead_index": int(bead_indices[index]),
                "spatial_distance_px": float(spatial_distances[index]),
                "cycle_index": int(cycle_indices[index]),
                "matched_reference": bool(valid[index]),
                "peak_time_us": int(peak_times[index]),
                "marker_residual_us": float(marker_residuals[index]),
                "start_time_us": int(starts[index]),
                "stop_time_us": int(stops[index]),
                "duration_us": int(durations[index]),
                "duration_error_us": int(duration_errors[index]),
                "start_phase_residual_us": float(start_residuals[index]),
                "stop_phase_residual_us": float(stop_residuals[index]),
                "roi_event_count": int(total_roi_events[index]),
            }
        )
    return summary, measurements


def assign_beads(
    peak_coordinates: np.ndarray, bead_centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(peak_coordinates, dtype=np.float64)
    distances = np.linalg.norm(
        coordinates[:, np.newaxis, :] - bead_centers[np.newaxis, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(coordinates)), nearest]
    nearest[nearest_distances > BEAD_ASSIGNMENT_RADIUS_PX] = -1
    return nearest, nearest_distances


def boundary_times(run: RunData) -> tuple[np.ndarray, np.ndarray, str]:
    if run.method == "transition_train":
        return (
            run.rois["t_on_first"].astype(np.int64),
            run.rois["t_off_first"].astype(np.int64),
            "first positive event to first negative event",
        )
    return (
        run.rois["t_1st"].astype(np.int64),
        run.rois["t_last"].astype(np.int64),
        "legacy gated ROI first to last event",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    summary_rows: list[dict[str, object]],
    measurement_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.4),
        constrained_layout=True,
    )
    method_colors = {
        "legacy": PLOT_COLORS["sky_blue"],
        "transition_train": PLOT_COLORS["vermillion"],
    }
    setting_names = [setting.name for setting in SETTINGS]
    x_positions = np.arange(len(setting_names), dtype=float)

    legacy_rows = [row for row in summary_rows if row["method"] == "legacy"]
    event_counts = np.asarray(
        [numeric_value(row["event_count_1_to_5_s"]) for row in legacy_rows]
    )
    axes[0, 0].bar(
        x_positions,
        event_counts / 1e6,
        color=PLOT_COLORS["blue"],
        width=0.68,
        zorder=2,
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Events in 1–5 s (millions; log scale)")
    axes[0, 0].set_title("Acquisition load")
    for x_position, count in zip(x_positions, event_counts):
        axes[0, 0].text(
            x_position,
            count / 1e6 * 1.12,
            f"{count / 1e6:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    bar_width = 0.36
    for method_index, method in enumerate(("legacy", "transition_train")):
        rows = [row for row in summary_rows if row["method"] == method]
        recalls = [numeric_value(row["bead_cycle_recall_percent"]) for row in rows]
        offset = (method_index - 0.5) * bar_width
        axes[0, 1].bar(
            x_positions + offset,
            recalls,
            width=bar_width,
            color=method_colors[method],
            label=method.replace("_", " ").title(),
            zorder=2,
        )
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("Matched bead-cycles (%)")
    axes[0, 1].set_title("50 Hz cycle recovery")
    axes[0, 1].legend(frameon=False, fontsize=6.5, loc="lower right")

    for method_index, method in enumerate(("legacy", "transition_train")):
        rows = [row for row in summary_rows if row["method"] == method]
        medians = np.asarray([numeric_value(row["duration_median_ms"]) for row in rows])
        low = np.asarray([numeric_value(row["duration_p10_ms"]) for row in rows])
        high = np.asarray([numeric_value(row["duration_p90_ms"]) for row in rows])
        offset = (method_index - 0.5) * 0.18
        axes[1, 0].errorbar(
            x_positions + offset,
            medians,
            yerr=np.vstack((medians - low, high - medians)),
            fmt="o",
            color=method_colors[method],
            markersize=4,
            capsize=2.5,
            linewidth=1.0,
            label=method.replace("_", " ").title(),
            zorder=3,
        )
    axes[1, 0].axhline(
        BRIGHT_DURATION_US * 1e-3,
        color=PLOT_COLORS["black"],
        linestyle="--",
        linewidth=0.9,
        label="10 ms reference",
        zorder=1,
    )
    axes[1, 0].set_ylabel("Detected bright interval (ms)")
    axes[1, 0].set_title("Bright-interval estimate")

    box_data = []
    box_positions = []
    box_colors = []
    for setting_index, setting_name in enumerate(setting_names):
        for method_index, method in enumerate(("legacy", "transition_train")):
            residuals = [
                abs(numeric_value(row["marker_residual_us"]))
                for row in measurement_rows
                if row["setting"] == setting_name and row["method"] == method
            ]
            box_data.append(np.asarray(residuals) * 1e-3)
            box_positions.append(setting_index + (method_index - 0.5) * 0.22)
            box_colors.append(method_colors[method])
    boxplot = axes[1, 1].boxplot(
        box_data,
        positions=box_positions,
        widths=0.18,
        showfliers=False,
        whis=(10, 90),
        patch_artist=True,
        medianprops={"color": PLOT_COLORS["black"], "linewidth": 0.9},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(0.8)
    axes[1, 1].set_ylabel("Absolute 50 Hz phase error (ms)")
    axes[1, 1].set_title("Peak timing repeatability")

    for panel_label, axis in zip("ABCD", axes.flat):
        axis.set_xticks(x_positions, setting_names, rotation=18, ha="right")
        style_publication_axis(axis, show_grid=True)
        axis.text(
            -0.16,
            1.06,
            panel_label,
            transform=axis.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
        )
    save_publication_figure(
        figure,
        output_dir / "dead_time_calibration.png",
        dpi=PUBLICATION_DPI,
        save_vector=True,
    )
    plt.close(figure)


def write_manifest(
    runs: dict[tuple[str, str], RunData],
    bead_centers: np.ndarray,
    references: dict[str, ReferenceGrid],
    output_dir: Path,
) -> None:
    payload = {
        "ground_truth": {
            "frequency_hz": 50,
            "duty_cycle_percent": 50,
            "period_us": PERIOD_US,
            "bright_duration_us": BRIGHT_DURATION_US,
            "slice_start_us": SLICE_START_US,
            "slice_stop_us": SLICE_STOP_US,
            "cycles_per_recording": EXPECTED_CYCLE_COUNT,
        },
        "calibrated_peak_time_threshold_us": CALIBRATED_PEAK_THRESHOLD_US,
        "calibrated_peak_neighbors_px": CALIBRATED_PEAK_NEIGHBORS,
        "bead_assignment_radius_px": BEAD_ASSIGNMENT_RADIUS_PX,
        "bead_centers_yx_px": bead_centers.tolist(),
        "reference_phase_us": {
            name: reference.phase_us for name, reference in references.items()
        },
        "runs": {
            f"{setting}/{method}": str(run.run_directory)
            for (setting, method), run in sorted(runs.items())
        },
    }
    with (output_dir / "dead_time_calibration_manifest.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
