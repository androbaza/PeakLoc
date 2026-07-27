"""Compare full-minute Maximum and Default dead-time recordings with legacy extraction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.photophysics_deconstruction import (
    BlinkSample,
    DetectionTrace,
    _load_exclusions,
    _load_spatial_masks,
    _read_sample_regions,
    _reconstruct_detection,
)
from localization_scripts.plot_style import (
    DOUBLE_COLUMN_WIDTH_IN,
    PANEL_LABEL_SIZE,
    PLOT_COLORS,
    PUBLICATION_DPI,
    save_publication_figure,
    style_publication_axis,
)
from localization_scripts.temporal_segmentation import TemporalSegmentationSettings
from scripts.dead_time_calibration import (
    assign_beads,
    circular_phase,
    infer_bead_centers,
    wrapped_residual,
)


PERIOD_US = 20_000
BRIGHT_DURATION_US = 10_000
RECORDING_START_US = 0
RECORDING_STOP_US = 60_000_000
EXPECTED_CYCLE_COUNT = 3_000
EXPECTED_BEAD_COUNT = 8
BEAD_ASSIGNMENT_RADIUS_PX = 10.0
PEAK_TIME_THRESHOLD_US = 10_000.0
PEAK_NEIGHBORS_PX = 9
TRACE_SAMPLE_SIZE = 5
TRACE_RANDOM_SEED = 50_647
TRACE_EDGE_MARGIN_US = 500_000
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
Scalar = str | int | float | bool
Row = dict[str, Scalar]


@dataclass(frozen=True)
class Setting:
    name: str
    bias_refr: int
    recording_name: str


SETTINGS = (
    Setting("Default", 0, "recording_2026-07-24_10-18-39"),
    Setting("Maximum", -20, "recording_2026-07-24_10-20-24"),
)


@dataclass(frozen=True)
class RunData:
    setting: Setting
    run_directory: Path
    raw_path: Path
    config: dict[str, Any]
    rois: np.ndarray
    localization_count: int
    slice_metrics: dict[str, Any]


@dataclass(frozen=True)
class TraceReplay:
    setting: Setting
    sample: BlinkSample
    trace: DetectionTrace
    count_window_start_us: float
    count_window_stop_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-minute Maximum and Default dead-time recordings using "
            "only PeakLoc's calibrated legacy blink extraction."
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
        default=Path("reports/dead-time-full-recording-comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.dataset_root)
    bead_centers = infer_dominant_bead_centers(
        runs["Default"].rois["peak"], EXPECTED_BEAD_COUNT
    )

    summary_rows: list[Row] = []
    roi_rows: list[Row] = []
    for setting in SETTINGS:
        summary, measurements = summarize_run(runs[setting.name], bead_centers)
        summary_rows.append(summary)
        roi_rows.extend(measurements)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "full_recording_summary.csv", summary_rows)
    write_csv(output_dir / "roi_measurements.csv", roi_rows)
    save_comparison_figure(summary_rows, roi_rows, output_dir)

    trace_replays: list[TraceReplay] = []
    for setting_index, setting in enumerate(SETTINGS):
        trace_replays.extend(
            replay_random_legacy_traces(
                runs[setting.name],
                random_seed=TRACE_RANDOM_SEED + setting_index,
            )
        )
    write_trace_source_data(trace_replays, output_dir)
    save_trace_figures(trace_replays, output_dir)
    write_manifest(runs, bead_centers, trace_replays, output_dir)
    write_report(summary_rows, output_dir)


def discover_runs(dataset_root: Path) -> dict[str, RunData]:
    runs: dict[str, RunData] = {}
    for setting in SETTINGS:
        recording_dir = dataset_root / setting.name / setting.recording_name
        run_directory = find_latest_matching_run(recording_dir)
        runs[setting.name] = load_run(setting, run_directory)
    return runs


def find_latest_matching_run(recording_dir: Path) -> Path:
    matches = []
    for candidate in recording_dir.iterdir():
        config_path = candidate / "share" / "metadata" / "effective_config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        if run_matches(config):
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(
            f"No completed full-minute calibrated legacy run below {recording_dir}"
        )
    return max(matches, key=lambda path: path.name)


def run_matches(config: dict[str, Any]) -> bool:
    return (
        config.get("slice_start") == RECORDING_START_US
        and config.get("slice_end") == RECORDING_STOP_US
        and config.get("slice_duration") == RECORDING_STOP_US
        and not bool(config.get("temporal_segmentation_enabled"))
        and float(config.get("peak_time_threshold", -1.0)) == PEAK_TIME_THRESHOLD_US
        and config.get("peak_neighbors") == PEAK_NEIGHBORS_PX
    )


def load_run(setting: Setting, run_directory: Path) -> RunData:
    array_dir = run_directory / "debug" / "arrays"
    report_dir = run_directory / "debug" / "reports"
    rois = np.load(single_path(array_dir, "rois*.npy"), allow_pickle=False)
    localizations = np.load(
        single_path(array_dir, "localizations*.npy"), allow_pickle=False
    )
    metrics = read_json(single_path(report_dir, "slice_metrics*.json"))
    if not isinstance(metrics, list) or len(metrics) != 1:
        raise ValueError(f"Expected one full-recording slice in {run_directory}")
    config = read_json(run_directory / "share" / "metadata" / "effective_config.json")
    metadata = read_json(run_directory / "share" / "metadata" / "run_metadata.json")
    raw_path = Path(str(metadata["input_file"])).expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"RAW recording is unavailable: {raw_path}")
    return RunData(
        setting=setting,
        run_directory=run_directory,
        raw_path=raw_path,
        config=config,
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


def infer_dominant_bead_centers(
    peak_coordinates: np.ndarray, expected_count: int
) -> np.ndarray:
    """Keep the most frequently detected spatial clusters as bead ground truth."""
    candidate_centers = infer_bead_centers(peak_coordinates)
    assignments, _ = assign_beads(peak_coordinates, candidate_centers)
    counts = np.asarray(
        [
            np.count_nonzero(assignments == index)
            for index in range(len(candidate_centers))
        ]
    )
    if len(candidate_centers) < expected_count:
        raise ValueError(
            f"Found only {len(candidate_centers)} bead clusters; expected {expected_count}"
        )
    selected = np.argsort(counts, kind="stable")[-expected_count:]
    centers = candidate_centers[selected]
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def summarize_run(run: RunData, bead_centers: np.ndarray) -> tuple[Row, list[Row]]:
    rois = run.rois
    peak_times = rois["t_peak"].astype(np.int64)
    phase_us = circular_phase(peak_times.astype(np.float64), PERIOD_US)
    cycle_indices = np.rint((peak_times - phase_us) / PERIOD_US).astype(np.int64)
    phase_residuals = wrapped_residual(
        peak_times.astype(np.float64), phase_us, PERIOD_US
    )
    bead_indices, spatial_distances = assign_beads(rois["peak"], bead_centers)
    valid_cycles = (cycle_indices >= 0) & (cycle_indices < EXPECTED_CYCLE_COUNT)
    valid = (
        (bead_indices >= 0)
        & valid_cycles
        & (np.abs(phase_residuals) <= PEAK_TIME_THRESHOLD_US)
    )
    matched_keys = {
        (int(bead), int(cycle))
        for bead, cycle, keep in zip(bead_indices, cycle_indices, valid, strict=True)
        if keep
    }
    expected_bead_cycles = len(bead_centers) * EXPECTED_CYCLE_COUNT
    valid_count = int(np.count_nonzero(valid))
    starts = rois["t_1st"].astype(np.int64)
    stops = rois["t_last"].astype(np.int64)
    durations = stops - starts
    positive_events = rois["total_events_roi"].astype(np.int64)
    negative_events = rois["total_neg_events_roi"].astype(np.int64)
    total_events = positive_events + negative_events

    start_phase = circular_phase(starts.astype(np.float64), PERIOD_US)
    stop_phase = circular_phase(stops.astype(np.float64), PERIOD_US)
    start_residuals = wrapped_residual(
        starts.astype(np.float64), start_phase, PERIOD_US
    )
    stop_residuals = wrapped_residual(stops.astype(np.float64), stop_phase, PERIOD_US)
    summary: Row = {
        "setting": run.setting.name,
        "bias_refr": run.setting.bias_refr,
        "method": "legacy",
        "run_directory": str(run.run_directory),
        "recording_start_us": RECORDING_START_US,
        "recording_stop_us": RECORDING_STOP_US,
        "event_count": int(run.slice_metrics["event_count"]),
        "candidate_count": int(run.slice_metrics["unique_peak_count"]),
        "roi_count": len(rois),
        "localization_count": run.localization_count,
        "expected_bead_cycle_count": expected_bead_cycles,
        "matched_bead_cycle_count": len(matched_keys),
        "bead_cycle_recall_percent": len(matched_keys) / expected_bead_cycles * 100.0,
        "duplicate_count": valid_count - len(matched_keys),
        "unassigned_count": int(np.count_nonzero(~valid)),
        "peak_absolute_phase_error_median_us": float(
            np.median(np.abs(phase_residuals))
        ),
        "peak_absolute_phase_error_p90_us": float(
            np.quantile(np.abs(phase_residuals), 0.9)
        ),
        "start_jitter_median_us": float(np.median(np.abs(start_residuals))),
        "stop_jitter_median_us": float(np.median(np.abs(stop_residuals))),
        "duration_median_ms": float(np.median(durations) * 1e-3),
        "duration_p10_ms": float(np.quantile(durations, 0.1) * 1e-3),
        "duration_p90_ms": float(np.quantile(durations, 0.9) * 1e-3),
        "duration_absolute_error_median_us": float(
            np.median(np.abs(durations - BRIGHT_DURATION_US))
        ),
        "duration_absolute_error_p90_us": float(
            np.quantile(np.abs(durations - BRIGHT_DURATION_US), 0.9)
        ),
        "fit_yield_percent": run.localization_count / max(len(rois), 1) * 100.0,
        "processing_seconds": float(run.slice_metrics["elapsed_seconds"]),
        "peak_rss_gib": float(run.slice_metrics["peak_rss_bytes"]) / 2**30,
    }
    for prefix, values in (
        ("events_per_roi_total", total_events),
        ("events_per_roi_positive", positive_events),
        ("events_per_roi_negative", negative_events),
    ):
        summary.update(distribution_statistics(prefix, values))

    measurements: list[Row] = []
    for index in range(len(rois)):
        measurements.append(
            {
                "setting": run.setting.name,
                "bias_refr": run.setting.bias_refr,
                "method": "legacy",
                "roi_index": index,
                "bead_index": int(bead_indices[index]),
                "cycle_index": int(cycle_indices[index]),
                "matched_reference": bool(valid[index]),
                "spatial_distance_px": float(spatial_distances[index]),
                "peak_time_us": int(peak_times[index]),
                "peak_phase_residual_us": float(phase_residuals[index]),
                "start_time_us": int(starts[index]),
                "stop_time_us": int(stops[index]),
                "duration_us": int(durations[index]),
                "start_phase_residual_us": float(start_residuals[index]),
                "stop_phase_residual_us": float(stop_residuals[index]),
                "positive_events": int(positive_events[index]),
                "negative_events": int(negative_events[index]),
                "total_events": int(total_events[index]),
            }
        )
    return summary, measurements


def distribution_statistics(prefix: str, values: np.ndarray) -> dict[str, int | float]:
    quantiles = np.quantile(values, QUANTILES)
    return {
        f"{prefix}_nonzero_roi_count": int(np.count_nonzero(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(quantiles[0]),
        f"{prefix}_p25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_p75": float(quantiles[3]),
        f"{prefix}_p90": float(quantiles[4]),
    }


def replay_random_legacy_traces(run: RunData, *, random_seed: int) -> list[TraceReplay]:
    rois = run.rois
    eligible = np.flatnonzero(
        (rois["t_peak"] >= TRACE_EDGE_MARGIN_US)
        & (rois["t_peak"] < RECORDING_STOP_US - TRACE_EDGE_MARGIN_US)
        & (rois["t_1st"] > 0)
        & (rois["t_last"] > rois["t_1st"])
        & (rois["total_events_roi"] > 0)
        & (rois["total_neg_events_roi"] > 0)
    )
    if eligible.size < TRACE_SAMPLE_SIZE:
        raise ValueError(
            f"Only {eligible.size} legacy ROIs are eligible for trace replay"
        )
    rng = np.random.default_rng(random_seed)
    candidate_indices = rng.permutation(eligible)[: TRACE_SAMPLE_SIZE * 4]
    samples = [
        blink_sample_from_roi(rois, int(roi_index), sample_id)
        for sample_id, roi_index in enumerate(candidate_indices, start=1)
    ]
    target_mask, support_mask = _load_spatial_masks(run.run_directory, run.config)
    temporal_settings = TemporalSegmentationSettings(
        context_pre_us=250_000,
        context_post_us=250_000,
    )
    exclusions = _load_exclusions(run.run_directory)
    regional_events = {}
    for sample in samples:
        regional_events.update(
            _read_sample_regions(
                run.raw_path,
                [sample],
                run.config,
                exclusions,
                support_mask,
                temporal_settings,
            )
        )
    replays = []
    for sample in samples:
        try:
            trace = _reconstruct_detection(
                regional_events[sample.sample_id],
                sample,
                run.config,
                target_mask,
                support_mask,
            )
        except ValueError:
            continue
        replay_sample = replace(sample, sample_id=len(replays) + 1)
        polarity_gate_us = float(run.config["polarity_time_gate_us"])
        replays.append(
            TraceReplay(
                setting=run.setting,
                sample=replay_sample,
                trace=trace,
                count_window_start_us=(
                    sample.t_peak_us + polarity_gate_us - sample.dt_positive_s * 1e6
                ),
                count_window_stop_us=(
                    sample.t_peak_us - polarity_gate_us + sample.dt_negative_s * 1e6
                ),
            )
        )
        if len(replays) == TRACE_SAMPLE_SIZE:
            break
    if len(replays) != TRACE_SAMPLE_SIZE:
        raise ValueError(
            f"Reconstructed only {len(replays)} of {TRACE_SAMPLE_SIZE} requested "
            f"{run.setting.name} traces"
        )
    return replays


def blink_sample_from_roi(
    rois: np.ndarray, roi_index: int, sample_id: int
) -> BlinkSample:
    roi = rois[roi_index]
    roi_radius = int(roi["roi"].shape[0] // 2)
    return BlinkSample(
        sample_id=sample_id,
        localization_index=-1,
        roi_index=roi_index,
        t_peak_us=int(roi["t_peak"]),
        peak_y=int(roi["peak"][0]),
        peak_x=int(roi["peak"][1]),
        fit_y=float("nan"),
        fit_x=float("nan"),
        sub_y=float(roi_radius),
        sub_x=float(roi_radius),
        positive_event_count=int(roi["total_events_roi"]),
        negative_event_count=int(roi["total_neg_events_roi"]),
        t_first_stored_us=int(roi["t_1st"]),
        t_last_stored_us=int(roi["t_last"]),
        dt_positive_s=float(roi["dt_pos_s"]),
        dt_negative_s=float(roi["dt_neg_s"]),
        uncertainty_nm=float("nan"),
        nll_per_event=float("nan"),
        roi_positive=np.asarray(roi["roi"], dtype=np.uint32).copy(),
        roi_negative=np.asarray(roi["roi_n"], dtype=np.uint32).copy(),
    )


def cropped_trace(
    replay: TraceReplay,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trace = replay.trace
    start = replay.count_window_start_us
    stop = replay.count_window_stop_us
    padding = max((stop - start) * 0.1, 2_000.0)
    raw_mask = (trace.raw_time_us >= start - padding) & (
        trace.raw_time_us <= stop + padding
    )
    interpolated_mask = (trace.interpolated_time_us >= start - padding) & (
        trace.interpolated_time_us <= stop + padding
    )
    raw_time = trace.raw_time_us[raw_mask]
    raw_values = trace.raw_cumulative_polarity[raw_mask]
    interpolated_time = trace.interpolated_time_us[interpolated_mask]
    interpolated_values = trace.interpolated_cumulative_polarity[interpolated_mask]
    if raw_time.size == 0 or interpolated_time.size == 0:
        raise ValueError("Legacy replay has no trace samples in the ROI window")
    baseline = float(raw_values[0])
    return (
        raw_time,
        raw_values - baseline,
        interpolated_time,
        interpolated_values - baseline,
    )


def write_trace_source_data(replays: list[TraceReplay], output_dir: Path) -> None:
    selection_rows: list[Row] = []
    trace_rows: list[Row] = []
    for replay in replays:
        sample = replay.sample
        selection_rows.append(
            {
                "setting": replay.setting.name,
                "random_seed": TRACE_RANDOM_SEED
                + int(replay.setting.name == "Maximum"),
                "sample_id": sample.sample_id,
                "roi_index": sample.roi_index,
                "peak_time_us": sample.t_peak_us,
                "peak_y_px": sample.peak_y,
                "peak_x_px": sample.peak_x,
                "positive_events": sample.positive_event_count,
                "negative_events": sample.negative_event_count,
                "total_events": sample.total_event_count,
                "spline_start_relative_ms": (
                    replay.trace.window_start_us - sample.t_peak_us
                )
                * 1e-3,
                "spline_stop_relative_ms": (
                    replay.trace.window_stop_us - sample.t_peak_us
                )
                * 1e-3,
                "roi_first_event_relative_ms": (
                    sample.t_first_stored_us - sample.t_peak_us
                )
                * 1e-3,
                "roi_last_event_relative_ms": (
                    sample.t_last_stored_us - sample.t_peak_us
                )
                * 1e-3,
            }
        )
        raw_time, raw_values, interpolated_time, interpolated_values = cropped_trace(
            replay
        )
        for series, times, values in (
            ("raw", raw_time, raw_values),
            ("interpolated", interpolated_time, interpolated_values),
        ):
            for point_index, (timestamp, value) in enumerate(
                zip(times, values, strict=True)
            ):
                trace_rows.append(
                    {
                        "setting": replay.setting.name,
                        "sample_id": sample.sample_id,
                        "roi_index": sample.roi_index,
                        "series": series,
                        "point_index": point_index,
                        "time_us": float(timestamp),
                        "time_relative_to_peak_ms": (
                            float(timestamp) - sample.t_peak_us
                        )
                        * 1e-3,
                        "cumulative_polarity_relative_events": float(value),
                    }
                )
    write_csv(output_dir / "trace_sample_selection.csv", selection_rows)
    write_csv(output_dir / "legacy_cumulative_trace_data.csv", trace_rows)


def save_trace_figures(replays: list[TraceReplay], output_dir: Path) -> None:
    for setting in SETTINGS:
        setting_replays = [
            replay for replay in replays if replay.setting.name == setting.name
        ]
        figure, axes = plt.subplots(
            TRACE_SAMPLE_SIZE,
            1,
            figsize=(DOUBLE_COLUMN_WIDTH_IN, 7.1),
            constrained_layout=True,
            sharex=False,
        )
        for axis, replay in zip(axes, setting_replays, strict=True):
            raw_time, raw_values, interpolated_time, interpolated_values = (
                cropped_trace(replay)
            )
            peak = replay.sample.t_peak_us
            axis.step(
                (raw_time - peak) * 1e-3,
                raw_values,
                where="post",
                color=PLOT_COLORS["gray"],
                linewidth=0.7,
                label="Raw cumulative polarity",
            )
            axis.plot(
                (interpolated_time - peak) * 1e-3,
                interpolated_values,
                color=PLOT_COLORS["blue"],
                linewidth=1.0,
                label="Interpolated cumulative polarity",
            )
            axis.axvspan(
                (replay.trace.window_start_us - peak) * 1e-3,
                (replay.trace.window_stop_us - peak) * 1e-3,
                color=PLOT_COLORS["sky_blue"],
                alpha=0.2,
                linewidth=0,
                label="Spline start–stop window",
            )
            axis.axvline(
                0,
                color=PLOT_COLORS["vermillion"],
                linewidth=1.0,
                label="Retained peak",
            )
            axis.axvline(
                (replay.sample.t_first_stored_us - peak) * 1e-3,
                color=PLOT_COLORS["green"],
                linewidth=0.9,
                linestyle="--",
                label="First/last ROI event",
            )
            axis.axvline(
                (replay.sample.t_last_stored_us - peak) * 1e-3,
                color=PLOT_COLORS["green"],
                linewidth=0.9,
                linestyle="--",
            )
            axis.set_ylabel("Cumulative\npolarity (events)")
            axis.set_title(
                f"Random blink {replay.sample.sample_id}: ROI {replay.sample.roi_index}, "
                f"{replay.sample.positive_event_count} positive + "
                f"{replay.sample.negative_event_count} negative events",
                loc="left",
            )
            style_publication_axis(axis)
        axes[-1].set_xlabel("Time from retained peak (ms)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=4,
            frameon=False,
            fontsize=6.5,
        )
        figure.suptitle(
            f"{setting.name} dead time: legacy cumulative-sum blink extraction",
            fontsize=9,
            y=1.025,
        )
        save_publication_figure(
            figure,
            output_dir / f"legacy_cumulative_traces_{setting.name.lower()}.png",
            dpi=PUBLICATION_DPI,
            save_vector=True,
        )
        plt.close(figure)


def save_comparison_figure(
    summary_rows: list[Row],
    roi_rows: list[Row],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.3),
        constrained_layout=True,
    )
    names = [setting.name for setting in SETTINGS]
    colors = [PLOT_COLORS["gray"], PLOT_COLORS["blue"]]
    x_positions = np.arange(len(names), dtype=np.float64)

    event_counts = np.asarray([float(row["event_count"]) for row in summary_rows])
    axes[0, 0].bar(x_positions, event_counts / 1e6, color=colors, width=0.62, zorder=2)
    axes[0, 0].set_ylabel("RAW events in 0–60 s (millions)")
    axes[0, 0].set_title("Acquisition load")
    for position, count in zip(x_positions, event_counts, strict=True):
        axes[0, 0].text(
            position,
            count / 1e6 + 0.25,
            f"{count / 1e6:.2f} M",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    recall = np.asarray(
        [float(row["bead_cycle_recall_percent"]) for row in summary_rows]
    )
    axes[0, 1].bar(x_positions, recall, color=colors, width=0.62, zorder=2)
    axes[0, 1].axhline(100, color=PLOT_COLORS["black"], linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylim(0, 100.15)
    axes[0, 1].set_ylabel("Unique matched bead-cycles (%)")
    axes[0, 1].set_title("50 Hz recovery")
    for position, value in zip(x_positions, recall, strict=True):
        axes[0, 1].text(
            position,
            value - 0.05,
            f"{value:.3f}%",
            ha="center",
            va="top",
            fontsize=6.5,
        )

    box_data = []
    box_positions = []
    box_colors = []
    polarity_labels = ("Positive", "Negative", "Total")
    polarity_fields = ("positive_events", "negative_events", "total_events")
    position = 0.0
    group_centers = []
    for polarity, field in zip(polarity_labels, polarity_fields, strict=True):
        group_centers.append(position + 0.18)
        for setting_index, setting_name in enumerate(names):
            values = [
                float(row[field]) for row in roi_rows if row["setting"] == setting_name
            ]
            box_data.append(np.asarray(values))
            box_positions.append(position + setting_index * 0.36)
            box_colors.append(colors[setting_index])
        position += 1.0
    boxplot = axes[1, 0].boxplot(
        box_data,
        positions=box_positions,
        widths=0.3,
        showfliers=False,
        whis=(10, 90),
        patch_artist=True,
        medianprops={"color": PLOT_COLORS["black"], "linewidth": 0.9},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(boxplot["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_linewidth(0.8)
    axes[1, 0].set_xticks(group_centers, polarity_labels)
    axes[1, 0].set_ylabel("Events per ROI")
    axes[1, 0].set_title("Polarity-resolved ROI event load")
    axes[1, 0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in colors],
        labels=names,
        frameon=False,
        fontsize=6.5,
        loc="upper left",
    )

    medians = np.asarray([float(row["duration_median_ms"]) for row in summary_rows])
    low = np.asarray([float(row["duration_p10_ms"]) for row in summary_rows])
    high = np.asarray([float(row["duration_p90_ms"]) for row in summary_rows])
    axes[1, 1].errorbar(
        x_positions,
        medians,
        yerr=np.vstack((medians - low, high - medians)),
        fmt="o",
        color=PLOT_COLORS["black"],
        markerfacecolor=PLOT_COLORS["sky_blue"],
        markeredgewidth=0.8,
        capsize=3,
        linewidth=1.0,
        zorder=3,
        label="Median; 10th–90th percentile",
    )
    axes[1, 1].axhline(
        BRIGHT_DURATION_US * 1e-3,
        color=PLOT_COLORS["vermillion"],
        linestyle="--",
        linewidth=0.9,
        label="10 ms laser-bright reference",
    )
    axes[1, 1].set_ylabel("Legacy first-to-last ROI event (ms)")
    axes[1, 1].set_title("Extracted blink window")
    axes[1, 1].legend(frameon=False, fontsize=6.5, loc="upper right")

    for panel_label, axis in zip("ABCD", axes.flat, strict=True):
        if axis is not axes[1, 0]:
            axis.set_xticks(x_positions, names)
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
        output_dir / "full_recording_comparison.png",
        dpi=PUBLICATION_DPI,
        save_vector=True,
    )
    plt.close(figure)


def write_manifest(
    runs: dict[str, RunData],
    bead_centers: np.ndarray,
    trace_replays: list[TraceReplay],
    output_dir: Path,
) -> None:
    payload = {
        "ground_truth": {
            "frequency_hz": 50,
            "duty_cycle_percent": 50,
            "period_us": PERIOD_US,
            "bright_duration_us": BRIGHT_DURATION_US,
            "recording_start_us": RECORDING_START_US,
            "recording_stop_us": RECORDING_STOP_US,
            "cycles_per_recording": EXPECTED_CYCLE_COUNT,
            "inferred_bead_count": len(bead_centers),
            "expected_bead_cycles": len(bead_centers) * EXPECTED_CYCLE_COUNT,
        },
        "analysis": {
            "method": "legacy",
            "peak_time_threshold_us": PEAK_TIME_THRESHOLD_US,
            "peak_neighbors_px": PEAK_NEIGHBORS_PX,
            "bead_assignment_radius_px": BEAD_ASSIGNMENT_RADIUS_PX,
            "trace_sample_size_per_setting": TRACE_SAMPLE_SIZE,
            "trace_random_seed_default": TRACE_RANDOM_SEED,
            "trace_random_seed_maximum": TRACE_RANDOM_SEED + 1,
            "trace_sampling_population": (
                "all non-edge legacy ROIs with nonzero positive and negative counts"
            ),
        },
        "bead_centers_yx_px": bead_centers.tolist(),
        "runs": {name: str(run.run_directory) for name, run in runs.items()},
        "trace_roi_indices": {
            setting.name: [
                replay.sample.roi_index
                for replay in trace_replays
                if replay.setting.name == setting.name
            ]
            for setting in SETTINGS
        },
    }
    path = output_dir / "full_recording_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(summary_rows: list[Row], output_dir: Path) -> None:
    rows = {str(row["setting"]): row for row in summary_rows}
    default = rows["Default"]
    maximum = rows["Maximum"]
    event_reduction = (
        1.0 - float(maximum["event_count"]) / float(default["event_count"])
    ) * 100.0
    speedup = float(default["processing_seconds"]) / float(
        maximum["processing_seconds"]
    )
    report = f"""# Full-minute Maximum vs Default dead-time comparison

## Result

Using the calibrated **legacy-only** workflow on exactly 0–60 s, both settings recovered essentially
all of the expected 50 Hz bead cycles. Maximum dead time reduced RAW event traffic by
**{event_reduction:.1f}%** and completed the slice **{speedup:.2f}× faster**, while preserving
{float(maximum["bead_cycle_recall_percent"]):.3f}% unique bead-cycle recovery and a
{float(maximum["fit_yield_percent"]):.2f}% localization fit yield.

## Direct comparison

| Metric | Default | Maximum |
|---|---:|---:|
| RAW events, 0–60 s | {int(default["event_count"]):,} | {int(maximum["event_count"]):,} |
| Retained legacy ROIs | {int(default["roi_count"]):,} | {int(maximum["roi_count"]):,} |
| Accepted localizations | {int(default["localization_count"]):,} | {int(maximum["localization_count"]):,} |
| Unique matched bead-cycles / 24,000 | {int(default["matched_bead_cycle_count"]):,} | {int(maximum["matched_bead_cycle_count"]):,} |
| Bead-cycle recall | {float(default["bead_cycle_recall_percent"]):.3f}% | {float(maximum["bead_cycle_recall_percent"]):.3f}% |
| Duplicate detections | {int(default["duplicate_count"]):,} | {int(maximum["duplicate_count"]):,} |
| Unassigned detections | {int(default["unassigned_count"]):,} | {int(maximum["unassigned_count"]):,} |
| Fit yield | {float(default["fit_yield_percent"]):.2f}% | {float(maximum["fit_yield_percent"]):.2f}% |
| Median absolute 50 Hz phase error | {float(default["peak_absolute_phase_error_median_us"]):.0f} µs | {float(maximum["peak_absolute_phase_error_median_us"]):.0f} µs |
| Median legacy first-to-last window | {float(default["duration_median_ms"]):.2f} ms | {float(maximum["duration_median_ms"]):.2f} ms |
| Processing time | {float(default["processing_seconds"]):.1f} s | {float(maximum["processing_seconds"]):.1f} s |
| Peak resident memory | {float(default["peak_rss_gib"]):.2f} GiB | {float(maximum["peak_rss_gib"]):.2f} GiB |

## Events per ROI

Whisker ranges below are the 10th–90th percentiles; means and all additional quartiles are in
`full_recording_summary.csv`.

| Polarity count | Default median (P10–P90) | Maximum median (P10–P90) |
|---|---:|---:|
| Positive events | {format_event_range(default, "positive")} | {format_event_range(maximum, "positive")} |
| Negative events | {format_event_range(default, "negative")} | {format_event_range(maximum, "negative")} |
| Total events | {format_event_range(default, "total")} | {format_event_range(maximum, "total")} |

Maximum therefore lowers the typical positive-lobe event count more strongly than Default, but the
remaining counts are sufficient for the configured joint positive/negative fit. The separate
nonzero-ROI counts are included in the summary table so zero-count tails are not hidden.

## How legacy start/end extraction is visualized

For each setting, five ROIs were sampled reproducibly at random from all non-edge legacy detections
with both polarities present. Each trace plot shows:

1. the raw cumulative polarity step trace (+1 for a positive event, −1 for a negative event);
2. the linearly interpolated cumulative trace used to locate a prominent maximum;
3. the spline-curvature-derived start/stop interval (blue shading);
4. the retained peak (red line); and
5. the first and last events actually stored in the final gated ROI (green dashed lines).

The spline interval proposes the temporal window. Legacy ROI generation then expands it when needed
to include the ±5 ms polarity gates, and `t_1st`/`t_last` are the first/last events found inside that
counting window. Consequently, the stored first-to-last duration is an event-window proxy, not a
direct optical measurement of the 10 ms laser-bright interval. The replay uses a local ±250 ms RAW
context, so its spline shading is a diagnostic reconstruction; the stored retained peak and stored
`t_1st`/`t_last` values remain the authoritative full-slice outputs.

## Recommendation

Use **Maximum dead time (`bias_refr = -20`)** for blinking-fluorophore acquisition under comparable
event rates. Across the full minute it keeps essentially complete 50 Hz recovery, has the higher fit
yield, halves event traffic, reduces peak memory, and shortens processing. Keep the calibrated
legacy settings (`peak_time_threshold = 10 ms`, `peak_neighbors = 9`) for this acquisition regime.
No new blink-extraction method is justified by this comparison.

The exact 50 Hz recovery result is specific to these bright, periodically driven 100 nm beads.
Before making Maximum the permanent fluorophore default, confirm it on a short sparse-fluorophore
recording because real blinks are dimmer, non-periodic, and lack this ground truth.

## Reproduction

```bash
pixi run python -m scripts.dead_time_full_recording_comparison
```

## Artifacts

- `full_recording_comparison.png` / `.pdf`: acquisition, recovery, polarity-resolved ROI counts, and
  blink-window comparison.
- `legacy_cumulative_traces_default.png` / `.pdf`: five random Default legacy replays.
- `legacy_cumulative_traces_maximum.png` / `.pdf`: five random Maximum legacy replays.
- `full_recording_summary.csv`: setting-level statistics.
- `roi_measurements.csv`: one row per ROI, including positive, negative, and total events.
- `trace_sample_selection.csv`: sampled ROI identities and extracted boundaries.
- `legacy_cumulative_trace_data.csv`: plotted cumulative-trace source values.
- `full_recording_manifest.json`: run paths, settings, ground truth, and random seeds.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def format_event_range(row: Row, polarity: str) -> str:
    prefix = f"events_per_roi_{polarity}"
    return (
        f"{float(row[f'{prefix}_median']):.0f} "
        f"({float(row[f'{prefix}_p10']):.0f}–"
        f"{float(row[f'{prefix}_p90']):.0f})"
    )


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        return payload
    return payload


if __name__ == "__main__":
    main()
