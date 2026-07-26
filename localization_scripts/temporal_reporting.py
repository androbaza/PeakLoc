"""Collaborator-facing temporal blink figures and statistics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import (
    DOUBLE_COLUMN_WIDTH_IN,
    PLOT_COLORS,
    save_publication_figure,
    style_publication_axis,
)
from localization_scripts.python_compat import strict_zip


MAX_SPATIAL_BINS = 24
MIN_LOCALIZATIONS_PER_SPATIAL_BIN = 5
TIMING_DISPLAY_MAX_MS = 1_000.0


def save_temporal_dynamics_artifacts(
    localizations: np.ndarray,
    figure_dir: Path,
    statistics_dir: Path,
    config: PeakLocConfig,
) -> list[Path]:
    """Save temporal dynamics figures plus their compact machine-readable summaries."""
    data = _temporal_data(localizations)
    if data is None:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    spatial_bins = _spatial_bins(data, config)
    dynamics = _time_binned_dynamics(data, bin_count=30)
    summary = _summary(data, spatial_bins)

    summary_path = statistics_dir / "temporal_blink_statistics.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spatial_path = statistics_dir / "temporal_blink_spatial_bin_statistics.csv"
    _write_rows(spatial_path, spatial_bins)
    dynamics_path = statistics_dir / "temporal_blink_dynamics_over_recording.csv"
    _write_rows(dynamics_path, dynamics)

    timing_path = figure_dir / "temporal_blink_timing_estimates.png"
    dynamics_figure_path = figure_dir / "temporal_blink_dynamics_over_recording.png"
    map_path = figure_dir / "temporal_blink_spatial_maps.png"
    artifacts = [summary_path, spatial_path, dynamics_path]
    artifacts.extend(_save_timing_figure(data, timing_path, config))
    artifacts.extend(_save_dynamics_figure(dynamics, dynamics_figure_path, config))
    artifacts.extend(_save_spatial_map_figure(spatial_bins, map_path, config))
    return artifacts


def _temporal_data(localizations: np.ndarray) -> dict[str, np.ndarray] | None:
    required = {"x", "y", "t_1st", "t_peak", "t_last"}
    names = localizations.dtype.names or ()
    if localizations.size == 0 or not required.issubset(names):
        return None

    x = np.asarray(localizations["x"], dtype=np.float64)
    y = np.asarray(localizations["y"], dtype=np.float64)
    first = np.asarray(localizations["t_1st"], dtype=np.float64)
    peak = np.asarray(localizations["t_peak"], dtype=np.float64)
    last = np.asarray(localizations["t_last"], dtype=np.float64)
    segmented_fields = {
        "temporal_segmented",
        "t_on_first",
        "t_on_last",
        "t_off_first",
        "t_off_last",
        "quiet_dwell_us",
    }
    segmented = segmented_fields.issubset(names) and np.any(
        localizations["temporal_segmented"]
    )
    if segmented:
        valid_segmented = np.asarray(localizations["temporal_segmented"], dtype=bool)
        first = np.asarray(localizations["t_on_first"], dtype=np.float64)
        on_end = np.asarray(localizations["t_on_last"], dtype=np.float64)
        off_start = np.asarray(localizations["t_off_first"], dtype=np.float64)
        last = np.asarray(localizations["t_off_last"], dtype=np.float64)
        duration_us = np.asarray(localizations["quiet_dwell_us"], dtype=np.float64)
        structural_valid = (
            valid_segmented
            & (first <= on_end)
            & (on_end <= off_start)
            & (off_start <= last)
        )
        source = "segmented_transition_trains"
    else:
        duration_us = last - first
        structural_valid = np.ones(localizations.size, dtype=bool)
        source = "roi_window_event_span"

    valid = (
        structural_valid
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(first)
        & np.isfinite(peak)
        & np.isfinite(last)
        & np.isfinite(duration_us)
        & (first >= 0)
        & (first <= peak)
        & (peak <= last)
        & (duration_us >= 0)
    )
    if not np.any(valid):
        return None
    reference_us = float(np.min(first[valid]))
    return {
        "x": x[valid],
        "y": y[valid],
        "turn_on_s": (first[valid] - reference_us) * 1e-6,
        "peak_s": (peak[valid] - reference_us) * 1e-6,
        "turn_off_s": (last[valid] - reference_us) * 1e-6,
        "rise_to_peak_ms": (peak[valid] - first[valid]) * 1e-3,
        "on_duration_ms": duration_us[valid] * 1e-3,
        "peak_to_last_ms": (last[valid] - peak[valid]) * 1e-3,
        "source": np.asarray([source]),
        "reference_us": np.asarray([reference_us]),
    }


def _spatial_bins(data: dict[str, np.ndarray], config: PeakLocConfig) -> np.ndarray:
    bins_x = min(MAX_SPATIAL_BINS, config.sensor_width)
    bins_y = min(MAX_SPATIAL_BINS, config.sensor_height)
    x_index = np.clip(
        (data["x"] * bins_x / config.sensor_width).astype(np.intp), 0, bins_x - 1
    )
    y_index = np.clip(
        (data["y"] * bins_y / config.sensor_height).astype(np.intp), 0, bins_y - 1
    )
    dtype = np.dtype(
        [
            ("bin_x", np.int32),
            ("bin_y", np.int32),
            ("x_start_px", np.float64),
            ("x_end_px", np.float64),
            ("y_start_px", np.float64),
            ("y_end_px", np.float64),
            ("localization_count", np.int32),
            ("median_turn_on_s", np.float64),
            ("median_on_duration_ms", np.float64),
            ("median_turn_off_s", np.float64),
            ("eligible_for_summary", np.bool_),
        ]
    )
    rows: list[tuple[Any, ...]] = []
    bin_ids = y_index * bins_x + x_index
    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id
        bin_y, bin_x = divmod(int(bin_id), bins_x)
        count = int(np.count_nonzero(mask))
        rows.append(
            (
                bin_x,
                bin_y,
                bin_x * config.sensor_width / bins_x,
                (bin_x + 1) * config.sensor_width / bins_x,
                bin_y * config.sensor_height / bins_y,
                (bin_y + 1) * config.sensor_height / bins_y,
                count,
                float(np.median(data["turn_on_s"][mask])),
                float(np.median(data["on_duration_ms"][mask])),
                float(np.median(data["turn_off_s"][mask])),
                count >= MIN_LOCALIZATIONS_PER_SPATIAL_BIN,
            )
        )
    return np.asarray(rows, dtype=dtype)


def _time_binned_dynamics(data: dict[str, np.ndarray], bin_count: int) -> np.ndarray:
    dtype = np.dtype(
        [
            ("time_start_s", np.float64),
            ("time_stop_s", np.float64),
            ("time_center_s", np.float64),
            ("localization_count", np.int32),
            ("median_rise_to_peak_ms", np.float64),
            ("median_on_duration_ms", np.float64),
            ("median_peak_to_last_ms", np.float64),
        ]
    )
    start = float(np.min(data["peak_s"]))
    stop = float(np.max(data["peak_s"]))
    if start == stop:
        stop = start + 1.0
    edges = np.linspace(start, stop, num=bin_count + 1)
    bin_index = np.clip(np.digitize(data["peak_s"], edges) - 1, 0, bin_count - 1)
    rows: list[tuple[Any, ...]] = []
    for index in range(bin_count):
        mask = bin_index == index
        if not np.any(mask):
            continue
        rows.append(
            (
                float(edges[index]),
                float(edges[index + 1]),
                float(np.median(data["peak_s"][mask])),
                int(np.count_nonzero(mask)),
                float(np.median(data["rise_to_peak_ms"][mask])),
                float(np.median(data["on_duration_ms"][mask])),
                float(np.median(data["peak_to_last_ms"][mask])),
            )
        )
    return np.asarray(rows, dtype=dtype)


def _summary(data: dict[str, np.ndarray], spatial_bins: np.ndarray) -> dict[str, Any]:
    duration_fields = (
        "rise_to_peak_ms",
        "on_duration_ms",
        "peak_to_last_ms",
    )
    return {
        "valid_temporal_localizations": int(data["x"].size),
        "timing_source": str(data["source"][0]),
        "reference_time_us": float(data["reference_us"][0]),
        "timing_estimates_ms": {
            field: _distribution(data[field]) for field in duration_fields
        },
        "outside_display_range_count": {
            field: int(np.count_nonzero(data[field] > TIMING_DISPLAY_MAX_MS))
            for field in duration_fields
        },
        "spatial_bins": {
            "occupied": int(spatial_bins.size),
            "qualified": int(np.count_nonzero(spatial_bins["eligible_for_summary"])),
            "minimum_localizations_per_bin": MIN_LOCALIZATIONS_PER_SPATIAL_BIN,
        },
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _save_timing_figure(
    data: dict[str, np.ndarray], path: Path, config: PeakLocConfig
) -> list[Path]:
    figure, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 3.0), constrained_layout=True
    )
    for field, label, color in (
        ("turn_on_s", "Turn-on boundary", PLOT_COLORS["blue"]),
        ("peak_s", "Localization peak", PLOT_COLORS["green"]),
        ("turn_off_s", "Turn-off boundary", PLOT_COLORS["vermillion"]),
    ):
        axes[0].hist(
            data[field],
            bins=60,
            histtype="step",
            linewidth=1.25,
            label=label,
            color=color,
        )
    axes[0].set(
        title="Temporal boundaries through the recording",
        xlabel="Time from first valid boundary (s)",
        ylabel="Localizations",
    )
    axes[0].legend(frameon=False, fontsize=6)
    style_publication_axis(axes[0])

    note_lines: list[str] = []
    for field, label, color in (
        ("rise_to_peak_ms", "Turn-on to peak", PLOT_COLORS["blue"]),
        ("on_duration_ms", "ON-state interval", PLOT_COLORS["green"]),
        ("peak_to_last_ms", "Peak to turn-off", PLOT_COLORS["vermillion"]),
    ):
        in_range = data[field][data[field] <= TIMING_DISPLAY_MAX_MS]
        axes[1].hist(
            in_range,
            bins=np.linspace(0, TIMING_DISPLAY_MAX_MS, 61),
            histtype="step",
            linewidth=1.25,
            label=label,
            color=color,
        )
        outside = int(np.count_nonzero(data[field] > TIMING_DISPLAY_MAX_MS))
        if outside:
            note_lines.append(f"{label}: n={outside:,} > 1000 ms")
    axes[1].set(
        title="Blink timing estimates",
        xlabel="Timing estimate (ms)",
        ylabel="Localizations",
        xlim=(0, TIMING_DISPLAY_MAX_MS),
    )
    axes[1].legend(frameon=False, fontsize=6)
    if note_lines:
        axes[1].text(
            0.98,
            0.97,
            "\n".join(note_lines),
            ha="right",
            va="top",
            fontsize=6,
            transform=axes[1].transAxes,
            bbox={"facecolor": "white", "edgecolor": "#B3B3B3", "pad": 2.0},
        )
    style_publication_axis(axes[1])
    paths = save_publication_figure(
        figure,
        path,
        dpi=max(config.qc_static_dpi, 450),
        save_vector=config.qc_save_vector,
    )
    plt.close(figure)
    return paths


def _save_dynamics_figure(
    dynamics: np.ndarray, path: Path, config: PeakLocConfig
) -> list[Path]:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 4.2),
        constrained_layout=True,
        sharex=True,
    )
    if dynamics.size:
        for field, label, color in (
            ("median_rise_to_peak_ms", "Turn-on to peak", PLOT_COLORS["blue"]),
            ("median_on_duration_ms", "ON-state interval", PLOT_COLORS["green"]),
            ("median_peak_to_last_ms", "Peak to turn-off", PLOT_COLORS["vermillion"]),
        ):
            axes[0].plot(
                dynamics["time_center_s"],
                dynamics[field],
                marker="o",
                markersize=2.2,
                linewidth=1.1,
                color=color,
                label=label,
            )
        axes[1].bar(
            dynamics["time_center_s"],
            dynamics["localization_count"],
            width=np.maximum(dynamics["time_stop_s"] - dynamics["time_start_s"], 0.1)
            * 0.85,
            color=PLOT_COLORS["gray"],
            edgecolor="none",
        )
    axes[0].set(
        title="Median blink timing through the recording", ylabel="Median timing (ms)"
    )
    axes[0].legend(frameon=False, ncol=3, fontsize=6, loc="upper left")
    axes[1].set(
        xlabel="Localization peak time from reference (s)", ylabel="Localizations/bin"
    )
    for axis in axes:
        style_publication_axis(axis, show_grid=True)
    paths = save_publication_figure(
        figure,
        path,
        dpi=max(config.qc_static_dpi, 450),
        save_vector=config.qc_save_vector,
    )
    plt.close(figure)
    return paths


def _save_spatial_map_figure(
    spatial_bins: np.ndarray, path: Path, config: PeakLocConfig
) -> list[Path]:
    bins_x = min(MAX_SPATIAL_BINS, config.sensor_width)
    bins_y = min(MAX_SPATIAL_BINS, config.sensor_height)
    figure, axes = plt.subplots(
        1, 3, figsize=(DOUBLE_COLUMN_WIDTH_IN * 1.5, 3.1), constrained_layout=True
    )
    specifications = (
        ("median_turn_on_s", "Median turn-on time", "s", "viridis"),
        ("median_on_duration_ms", "Median ON-state interval", "ms", "magma"),
        ("median_turn_off_s", "Median turn-off time", "s", "cividis"),
    )
    eligible = spatial_bins[spatial_bins["eligible_for_summary"]]
    for axis, (field, title, unit, cmap_name) in strict_zip(axes, specifications):
        image = np.full((bins_y, bins_x), np.nan, dtype=np.float64)
        if eligible.size:
            image[eligible["bin_y"], eligible["bin_x"]] = eligible[field]
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#E6E6E6")
        values = image[np.isfinite(image)]
        if values.size:
            minimum, maximum = float(np.min(values)), float(np.max(values))
            if minimum == maximum:
                maximum = minimum + max(abs(minimum) * 0.01, 1.0)
            image_plot = axis.imshow(
                np.ma.masked_invalid(image),
                cmap=cmap,
                interpolation="nearest",
                origin="upper",
                extent=(0, config.sensor_width, config.sensor_height, 0),
                vmin=minimum,
                vmax=maximum,
                aspect="equal",
            )
            colorbar = figure.colorbar(image_plot, ax=axis, fraction=0.045, pad=0.03)
            colorbar.set_label(f"{title} ({unit})", fontsize=7)
            colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
        else:
            axis.set_facecolor("#E6E6E6")
            axis.text(
                0.5,
                0.5,
                "No qualified spatial bins",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=7,
            )
        axis.set(title=title, xlabel="x (sensor pixel)", ylabel="y (sensor pixel)")
        style_publication_axis(axis)
    paths = save_publication_figure(
        figure,
        path,
        dpi=max(config.qc_static_dpi, 450),
        save_vector=config.qc_save_vector,
    )
    plt.close(figure)
    return paths


def _write_rows(path: Path, rows: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(rows.dtype.names or ())
        writer.writerows(row.tolist() for row in rows)
