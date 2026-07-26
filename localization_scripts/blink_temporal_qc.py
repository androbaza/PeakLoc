from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.python_compat import strict_zip
from localization_scripts.smlm_visualization import RENDER_OVERSAMPLING


TEMPORAL_FIELDS = frozenset({"x", "y", "t_1st", "t_peak", "t_last"})
SEGMENTED_TEMPORAL_FIELDS = frozenset(
    {
        "temporal_segmented",
        "t_on_first",
        "t_on_last",
        "t_off_first",
        "t_off_last",
        "quiet_dwell_us",
    }
)
MAX_SPATIAL_BINS = 24
MIN_LOCALIZATIONS_PER_SPATIAL_BIN = 5
INTERACTIVE_TIME_BIN_COUNTS = (30, 60, 120)
INTERACTIVE_SLIDER_STEPS = 1_000


@dataclass(frozen=True)
class TemporalMetric:
    field: str
    label: str
    axis_label: str
    matplotlib_cmap: str
    plotly_colorscale: str
    unit: str


@dataclass(frozen=True)
class TemporalPixelStatistics:
    """Median timing values grouped by their native sensor pixel."""

    x: np.ndarray
    y: np.ndarray
    count: np.ndarray
    medians: dict[str, np.ndarray]
    maps: dict[str, np.ndarray]


@dataclass(frozen=True)
class TemporalDynamicsBins:
    recording_time_s: np.ndarray
    rise_to_peak_ms: np.ndarray
    on_duration_ms: np.ndarray
    peak_to_last_ms: np.ndarray
    localization_count: np.ndarray


TEMPORAL_METRICS = (
    TemporalMetric(
        field="turn_on_s",
        label="Turn-on boundary",
        axis_label="Turn-on boundary relative to reference (s)",
        matplotlib_cmap="viridis",
        plotly_colorscale="Viridis",
        unit="s",
    ),
    TemporalMetric(
        field="on_duration_ms",
        label="ON-state interval",
        axis_label="ON-state interval (ms)",
        matplotlib_cmap="magma",
        plotly_colorscale="Magma",
        unit="ms",
    ),
    TemporalMetric(
        field="turn_off_s",
        label="Turn-off boundary",
        axis_label="Turn-off boundary relative to reference (s)",
        matplotlib_cmap="cividis",
        plotly_colorscale="Cividis",
        unit="s",
    ),
)


def save_blink_temporal_qc(
    localizations: np.ndarray,
    output_dir: Path,
    config: PeakLocConfig,
) -> list[Path]:
    """Save accepted-localization timing artifacts for a completed recording."""
    data = _temporal_data(localizations)
    if data is None:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_bins = _spatial_bins(data, config)
    pixel_statistics = _pixel_statistics(data, config)
    summary = _summary(data, spatial_bins, pixel_statistics, config)
    summary_json_path = output_dir / "temporal_blink_statistics.json"
    summary_md_path = output_dir / "temporal_blink_statistics.md"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    legacy_spatial_path = output_dir / "18_temporal_blink_spatial_maps.png"
    legacy_spatial_path.unlink(missing_ok=True)
    spatial_paths = _save_sensor_resolution_spatial_maps(pixel_statistics, output_dir)
    bin_csv_path = output_dir / "19_temporal_blink_spatial_bin_statistics.csv"
    distribution_path = output_dir / "20_temporal_blink_timing_distributions.png"
    _write_spatial_bins_csv(spatial_bins, bin_csv_path)
    _save_timing_distributions(data, config, distribution_path)
    paths = [
        summary_json_path,
        summary_md_path,
        *spatial_paths,
        bin_csv_path,
        distribution_path,
    ]

    if config.qc_generate_temporal_3d:
        paths.extend(
            _save_interactive_temporal_visualizations(
                data,
                pixel_statistics,
                config,
                output_dir,
            )
        )
    return paths


def _temporal_data(localizations: np.ndarray) -> dict[str, np.ndarray] | None:
    if localizations.size == 0 or not TEMPORAL_FIELDS.issubset(
        localizations.dtype.names or ()
    ):
        return None
    names = frozenset(localizations.dtype.names or ())
    x = np.asarray(localizations["x"], dtype=np.float64)
    y = np.asarray(localizations["y"], dtype=np.float64)
    peak = np.asarray(localizations["t_peak"], dtype=np.float64)
    has_segmented_timings = SEGMENTED_TEMPORAL_FIELDS.issubset(names) and np.any(
        localizations["temporal_segmented"]
    )
    if has_segmented_timings:
        segmented = np.asarray(localizations["temporal_segmented"], dtype=np.bool_)
        first = np.asarray(localizations["t_on_first"], dtype=np.float64)
        on_end = np.asarray(localizations["t_on_last"], dtype=np.float64)
        off_start = np.asarray(localizations["t_off_first"], dtype=np.float64)
        last = np.asarray(localizations["t_off_last"], dtype=np.float64)
        on_duration = np.asarray(localizations["quiet_dwell_us"], dtype=np.float64)
        timing_source = "segmented_transition_trains"
        structural_valid = (
            segmented & (first <= on_end) & (off_start <= last) & (on_duration >= 0)
        )
    else:
        first = np.asarray(localizations["t_1st"], dtype=np.float64)
        last = np.asarray(localizations["t_last"], dtype=np.float64)
        on_duration = last - first
        timing_source = "legacy_roi_extrema"
        structural_valid = np.ones(localizations.size, dtype=np.bool_)
    valid = (
        structural_valid
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(first)
        & np.isfinite(peak)
        & np.isfinite(last)
        & np.isfinite(on_duration)
        & (first >= 0)
        & (first <= peak)
        & (peak <= last)
    )
    if not np.any(valid):
        return None

    first = first[valid]
    peak = peak[valid]
    last = last[valid]
    on_duration = on_duration[valid]
    reference = float(np.min(first))
    return {
        "x": x[valid],
        "y": y[valid],
        "turn_on_s": (first - reference) * 1e-6,
        "peak_s": (peak - reference) * 1e-6,
        "turn_off_s": (last - reference) * 1e-6,
        "rise_to_peak_ms": (peak - first) * 1e-3,
        "on_duration_ms": on_duration * 1e-3,
        "peak_to_last_ms": (last - peak) * 1e-3,
        "reference_time_us": np.asarray([reference]),
        "timing_source": np.asarray([timing_source]),
    }


def _summary(
    data: dict[str, np.ndarray],
    spatial_bins: np.ndarray,
    pixel_statistics: TemporalPixelStatistics,
    config: PeakLocConfig,
) -> dict[str, Any]:
    timing_source = str(data["timing_source"][0])
    segmented = timing_source == "segmented_transition_trains"
    return {
        "valid_temporal_localizations": int(data["x"].size),
        "timing_source": timing_source,
        "time_reference": (
            "first valid ON-train event in this run"
            if segmented
            else "first valid ROI event in this run"
        ),
        "reference_time_us": float(data["reference_time_us"][0]),
        "turn_on_relative_s": _distribution_summary(data["turn_on_s"]),
        "peak_relative_s": _distribution_summary(data["peak_s"]),
        "turn_off_relative_s": _distribution_summary(data["turn_off_s"]),
        "rise_to_peak_ms": _distribution_summary(data["rise_to_peak_ms"]),
        "on_duration_ms": _distribution_summary(data["on_duration_ms"]),
        "peak_to_last_event_ms": _distribution_summary(data["peak_to_last_ms"]),
        "spatial_binning": _spatial_summary(spatial_bins),
        "sensor_pixel_medians": {
            "image_shape_yx": [config.sensor_height, config.sensor_width],
            "occupied_pixel_count": int(pixel_statistics.x.size),
            "positioning": (
                "Each median belongs to floor(x), floor(y) and is displayed in "
                f"the {RENDER_OVERSAMPLING}x SMLM render coordinate system."
            ),
        },
        "plotly_3d": {
            "enabled": config.qc_generate_temporal_3d,
            "spatial_point_count": int(
                min(data["x"].size, config.qc_max_events_for_interactive)
            ),
            "spatial_sampling_limit": config.qc_max_events_for_interactive,
            "pixel_median_point_count": int(pixel_statistics.x.size),
            "xy_time_point_count": int(data["x"].size),
            "axis_ranges_use_all_valid_localizations": True,
        },
        "interpretation": (
            "Turn-on and turn-off are the first ON-train and last OFF-train raw events. "
            "The ON-state interval is the non-negative gap from the last ON-train event "
            "to the first OFF-train event. These event-camera transition estimates are "
            "not direct molecular-state timestamps."
            if segmented
            else "t_1st and t_last are the first and last event timestamps inside the "
            "fitted ROI window. They are legacy proxies, not direct molecular transition "
            "timestamps."
        ),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _spatial_bins(data: dict[str, np.ndarray], config: PeakLocConfig) -> np.ndarray:
    bins_y = min(MAX_SPATIAL_BINS, config.sensor_height)
    bins_x = min(MAX_SPATIAL_BINS, config.sensor_width)
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
    x_bin = np.clip(
        (data["x"] * bins_x / config.sensor_width).astype(np.intp), 0, bins_x - 1
    )
    y_bin = np.clip(
        (data["y"] * bins_y / config.sensor_height).astype(np.intp), 0, bins_y - 1
    )
    bin_ids = y_bin * bins_x + x_bin
    rows = []
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


def _pixel_statistics(
    data: dict[str, np.ndarray], config: PeakLocConfig
) -> TemporalPixelStatistics:
    x = data["x"]
    y = data["y"]
    valid = (x >= 0) & (x < config.sensor_width) & (y >= 0) & (y < config.sensor_height)
    if not np.any(valid):
        return _empty_pixel_statistics(config)

    x_index = np.floor(x[valid]).astype(np.intp)
    y_index = np.floor(y[valid]).astype(np.intp)
    pixel_ids = y_index.astype(np.int64) * config.sensor_width + x_index
    order = np.argsort(pixel_ids, kind="stable")
    sorted_pixel_ids = pixel_ids[order]
    unique_ids, starts, counts = np.unique(
        sorted_pixel_ids,
        return_index=True,
        return_counts=True,
    )
    pixel_x = (unique_ids % config.sensor_width).astype(np.float64)
    pixel_y = (unique_ids // config.sensor_width).astype(np.float64)
    medians: dict[str, np.ndarray] = {}
    maps: dict[str, np.ndarray] = {}
    for metric in TEMPORAL_METRICS:
        values = data[metric.field][valid][order]
        grouped_medians = _group_medians(values, starts, counts)
        image = np.full(config.sensor_shape, np.nan, dtype=np.float32)
        image[pixel_y.astype(np.intp), pixel_x.astype(np.intp)] = grouped_medians
        medians[metric.field] = grouped_medians
        maps[metric.field] = image
    return TemporalPixelStatistics(
        x=pixel_x,
        y=pixel_y,
        count=np.asarray(counts, dtype=np.int64),
        medians=medians,
        maps=maps,
    )


def _empty_pixel_statistics(config: PeakLocConfig) -> TemporalPixelStatistics:
    maps = {
        metric.field: np.full(config.sensor_shape, np.nan, dtype=np.float32)
        for metric in TEMPORAL_METRICS
    }
    medians = {
        metric.field: np.empty(0, dtype=np.float64) for metric in TEMPORAL_METRICS
    }
    return TemporalPixelStatistics(
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        count=np.empty(0, dtype=np.int64),
        medians=medians,
        maps=maps,
    )


def _group_medians(
    values: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    return np.fromiter(
        (
            float(np.median(values[int(start) : int(start + count)]))
            for start, count in strict_zip(starts, counts)
        ),
        dtype=np.float64,
        count=int(starts.size),
    )


def _spatial_summary(spatial_bins: np.ndarray) -> dict[str, Any]:
    eligible = spatial_bins[spatial_bins["eligible_for_summary"]]
    metrics = {
        "turn_on_s": "median_turn_on_s",
        "on_duration_ms": "median_on_duration_ms",
        "turn_off_s": "median_turn_off_s",
    }
    return {
        "occupied_bin_count": int(spatial_bins.size),
        "qualified_bin_count": int(eligible.size),
        "minimum_localizations_per_bin": MIN_LOCALIZATIONS_PER_SPATIAL_BIN,
        "median_by_qualified_bin": {
            label: _optional_distribution_summary(eligible[field])
            for label, field in metrics.items()
        },
    }


def _optional_distribution_summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return _distribution_summary(values)


def _save_sensor_resolution_spatial_maps(
    pixel_statistics: TemporalPixelStatistics,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for suffix, name, metric in zip(
        ("a", "b", "c"),
        ("turn_on", "on_duration", "turn_off"),
        TEMPORAL_METRICS,
        strict=True,
    ):
        path = output_dir / f"18{suffix}_temporal_blink_{name}_spatial_map.png"
        _save_sensor_resolution_spatial_map(
            pixel_statistics.maps[metric.field], metric, path
        )
        paths.append(path)
    return paths


def _save_sensor_resolution_spatial_map(
    image: np.ndarray,
    metric: TemporalMetric,
    path: Path,
) -> None:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        rgba = np.zeros((*image.shape, 4), dtype=np.uint8)
        plt.imsave(path, rgba, origin="upper")
        return

    cmap = plt.get_cmap(metric.matplotlib_cmap).copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    vmin, vmax = _value_range(finite, padding=0.0)
    plt.imsave(
        path,
        np.ma.masked_invalid(image),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )


def _write_spatial_bins_csv(spatial_bins: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(spatial_bins.dtype.names)
        for row in spatial_bins:
            writer.writerow(row.tolist())


def _save_timing_distributions(
    data: dict[str, np.ndarray], config: PeakLocConfig, path: Path
) -> None:
    segmented = str(data["timing_source"][0]) == "segmented_transition_trains"
    boundary_prefix = "Segmented" if segmented else "ROI-window"
    duration_label = (
        "Inter-train ON-state interval" if segmented else "ROI-window event span"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for field, label, color in (
        ("turn_on_s", f"{boundary_prefix} turn-on", "#0072B2"),
        ("peak_s", "Peak", "#009E73"),
        ("turn_off_s", f"{boundary_prefix} turn-off", "#D55E00"),
    ):
        axes[0].hist(
            data[field],
            bins=60,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=color,
        )
    axes[0].set(
        xlabel="Time relative to first valid turn-on boundary (s)",
        ylabel="Localizations",
    )
    axes[0].legend()
    for field, label, color in (
        ("rise_to_peak_ms", f"{boundary_prefix} turn-on to peak", "#0072B2"),
        ("on_duration_ms", duration_label, "#009E73"),
        ("peak_to_last_ms", f"Peak to {boundary_prefix.lower()} turn-off", "#D55E00"),
    ):
        axes[1].hist(
            data[field],
            bins=60,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=color,
        )
    axes[1].set(xlabel="Event-camera timing estimate (ms)", ylabel="Localizations")
    axes[1].legend()
    figure.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(figure)


def _save_interactive_temporal_visualizations(
    data: dict[str, np.ndarray],
    pixel_statistics: TemporalPixelStatistics,
    config: PeakLocConfig,
    output_dir: Path,
) -> list[Path]:
    paths = [
        output_dir / "temporal_blink_spatial_map_2d.html",
        output_dir / "temporal_blink_spatial_3d.html",
        output_dir / "temporal_blink_pixel_median_3d.html",
        output_dir / "temporal_blink_dynamics_over_time_3d.html",
        output_dir / "temporal_blink_localizations_xy_time_3d.html",
    ]
    try:
        import plotly.graph_objects as go
    except ImportError:
        for path in paths:
            _write_plotly_unavailable_page(path)
        return paths

    _save_plotly_spatial_map_2d(data, config, paths[0], go)
    _save_plotly_spatial_3d(data, config, paths[1], go)
    _save_plotly_pixel_median_3d(data, pixel_statistics, paths[2], go)
    _save_plotly_dynamics_over_time_3d(data, paths[3], go)
    _save_plotly_localizations_xy_time_3d(data, config, paths[4], go)
    return paths


def _save_plotly_spatial_map_2d(
    data: dict[str, np.ndarray],
    config: PeakLocConfig,
    path: Path,
    go: Any,
) -> None:
    traces = []
    for index, metric in enumerate(TEMPORAL_METRICS):
        cmin, cmax = _value_range(data[metric.field], padding=0.0)
        traces.append(
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=np.zeros_like(data["x"]),
                mode="markers",
                name=metric.label,
                visible=index == 0,
                marker={
                    "size": 2.5,
                    "color": data[metric.field],
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": metric.plotly_colorscale,
                    "opacity": 0.72,
                    "colorbar": {"title": metric.unit},
                },
                text=data[metric.field],
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{text:.5g}"
                    f" {metric.unit}<extra></extra>"
                ),
            )
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title="Full-resolution temporal blink spatial map (flat z=0 plane)",
        height=850,
        margin={"l": 0, "r": 0, "t": 110, "b": 0},
        scene=_flat_spatial_scene(config),
        updatemenus=[_metric_2d_menu(), _camera_menu()],
    )
    plot_id = "temporal-blink-spatial-map-2d"
    ranges = {
        index: {"color": _value_range(data[metric.field], padding=0.0)}
        for index, metric in enumerate(TEMPORAL_METRICS)
    }
    _write_plotly_document(
        path,
        figure,
        plot_id,
        _two_d_controls_html(plot_id),
        _two_d_controls_script(plot_id, ranges),
    )


def _save_plotly_spatial_3d(
    data: dict[str, np.ndarray],
    config: PeakLocConfig,
    path: Path,
    go: Any,
) -> None:
    sampled = _sample_for_interactive(data, config.qc_max_events_for_interactive)
    traces = _spatial_metric_traces(sampled, data, go, marker_size=2.0, opacity=0.74)
    initial_metric = TEMPORAL_METRICS[0]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=(
            "Temporal blink timing in spatial coordinates "
            f"({sampled['x'].size:,} visualized; full data range retained)"
        ),
        height=820,
        margin={"l": 0, "r": 0, "t": 105, "b": 0},
        scene=_spatial_scene(
            config,
            initial_metric.axis_label,
            _value_range(data[initial_metric.field]),
        ),
        updatemenus=[_metric_3d_menu(data), _camera_menu()],
    )
    plot_id = "temporal-blink-spatial-3d"
    ranges = {
        index: {
            "z": _value_range(data[metric.field]),
            "color": _value_range(data[metric.field], padding=0.0),
        }
        for index, metric in enumerate(TEMPORAL_METRICS)
    }
    _write_plotly_document(
        path,
        figure,
        plot_id,
        _three_d_controls_html(plot_id),
        _three_d_controls_script(plot_id, ranges, marker_size=2.0, opacity=0.74),
    )


def _save_plotly_pixel_median_3d(
    data: dict[str, np.ndarray],
    pixel_statistics: TemporalPixelStatistics,
    path: Path,
    go: Any,
) -> None:
    if pixel_statistics.x.size == 0:
        _write_plotly_unavailable_page(
            path, "No temporal localizations fell within the sensor bounds."
        )
        return

    render_x = (pixel_statistics.x + 0.5) * RENDER_OVERSAMPLING
    render_y = (pixel_statistics.y + 0.5) * RENDER_OVERSAMPLING
    render_width, render_height = _render_shape(data)
    traces = []
    for index, metric in enumerate(TEMPORAL_METRICS):
        values = pixel_statistics.medians[metric.field]
        cmin, cmax = _value_range(values, padding=0.0)
        traces.append(
            go.Scatter3d(
                x=render_x,
                y=render_y,
                z=values,
                mode="markers",
                name=metric.label,
                visible=index == 0,
                marker={
                    "size": 2.8,
                    "color": values,
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": metric.plotly_colorscale,
                    "opacity": 0.8,
                    "colorbar": {"title": metric.unit},
                },
                customdata=np.column_stack(
                    (
                        pixel_statistics.count,
                        pixel_statistics.medians["turn_on_s"],
                        pixel_statistics.medians["on_duration_ms"],
                        pixel_statistics.medians["turn_off_s"],
                    )
                ),
                hovertemplate=(
                    "render x=%{x:.1f}<br>render y=%{y:.1f}<br>median value=%{z:.5g}"
                    f" {metric.unit}<br>localizations=%{{customdata[0]}}"
                    "<br>median turn-on=%{customdata[1]:.4g} s"
                    "<br>median duration=%{customdata[2]:.4g} ms"
                    "<br>median turn-off=%{customdata[3]:.4g} s<extra></extra>"
                ),
            )
        )
    initial_metric = TEMPORAL_METRICS[0]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=(
            "Median temporal dynamics per sensor pixel in final SMLM render coordinates "
            f"({pixel_statistics.x.size:,} occupied pixels)"
        ),
        height=820,
        margin={"l": 0, "r": 0, "t": 105, "b": 0},
        scene=_render_scene(
            render_width,
            render_height,
            initial_metric.axis_label,
            _value_range(pixel_statistics.medians[initial_metric.field]),
        ),
        updatemenus=[_pixel_metric_3d_menu(pixel_statistics), _camera_menu()],
    )
    plot_id = "temporal-blink-pixel-median-3d"
    ranges = {
        index: {
            "z": _value_range(pixel_statistics.medians[metric.field]),
            "color": _value_range(pixel_statistics.medians[metric.field], padding=0.0),
        }
        for index, metric in enumerate(TEMPORAL_METRICS)
    }
    _write_plotly_document(
        path,
        figure,
        plot_id,
        _three_d_controls_html(plot_id),
        _three_d_controls_script(plot_id, ranges, marker_size=2.8, opacity=0.8),
    )


def _save_plotly_dynamics_over_time_3d(
    data: dict[str, np.ndarray], path: Path, go: Any
) -> None:
    binned = [
        _time_binned_dynamics(data, count) for count in INTERACTIVE_TIME_BIN_COUNTS
    ]
    traces = []
    for index, (count, values) in enumerate(
        strict_zip(INTERACTIVE_TIME_BIN_COUNTS, binned)
    ):
        cmin, cmax = _value_range(values.peak_to_last_ms, padding=0.0)
        traces.append(
            go.Scatter3d(
                x=values.recording_time_s,
                y=values.rise_to_peak_ms,
                z=values.on_duration_ms,
                mode="markers+lines",
                name=f"{count} time bins",
                visible=index == 1,
                marker={
                    "size": 5,
                    "color": values.peak_to_last_ms,
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": "Cividis",
                    "opacity": 0.9,
                    "colorbar": {"title": "median peak-to-last (ms)"},
                },
                line={"color": "rgba(70,70,70,0.45)", "width": 2},
                customdata=np.column_stack(
                    (values.localization_count, values.peak_to_last_ms)
                ),
                hovertemplate=(
                    "recording time=%{x:.4g} s"
                    "<br>median turn-on to peak=%{y:.4g} ms"
                    "<br>median on-duration=%{z:.4g} ms"
                    "<br>median peak-to-last=%{customdata[1]:.4g} ms"
                    "<br>localizations=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    initial_index = 1
    initial_values = binned[initial_index]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title="Time-binned median temporal dynamics through the recording",
        height=820,
        margin={"l": 0, "r": 0, "t": 105, "b": 0},
        scene={
            "xaxis": {
                "title": "Peak time relative to reference (s)",
                "range": list(_value_range(data["peak_s"])),
            },
            "yaxis": {
                "title": "Median turn-on to peak (ms)",
                "range": list(_value_range(data["rise_to_peak_ms"])),
            },
            "zaxis": {
                "title": "Median on-duration (ms)",
                "range": list(_value_range(initial_values.on_duration_ms)),
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.7, "y": 1.0, "z": 0.8},
        },
        updatemenus=[_time_bin_menu(binned), _camera_menu()],
    )
    plot_id = "temporal-blink-dynamics-over-time-3d"
    ranges = {
        index: {
            "z": _value_range(values.on_duration_ms),
            "color": _value_range(values.peak_to_last_ms, padding=0.0),
        }
        for index, values in enumerate(binned)
    }
    _write_plotly_document(
        path,
        figure,
        plot_id,
        _three_d_controls_html(plot_id),
        _three_d_controls_script(plot_id, ranges, marker_size=5.0, opacity=0.9),
    )


def _save_plotly_localizations_xy_time_3d(
    data: dict[str, np.ndarray],
    config: PeakLocConfig,
    path: Path,
    go: Any,
) -> None:
    cmin, cmax = _value_range(data["on_duration_ms"], padding=0.0)
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["peak_s"],
                mode="markers",
                name="accepted localizations",
                marker={
                    "size": 1.6,
                    "color": data["on_duration_ms"],
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": "Magma",
                    "opacity": 0.32,
                    "colorbar": {"title": "on-duration (ms)"},
                },
                customdata=np.column_stack(
                    (
                        data["turn_on_s"],
                        data["turn_off_s"],
                        data["on_duration_ms"],
                    )
                ),
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>peak=%{z:.5g} s"
                    "<br>turn-on=%{customdata[0]:.5g} s"
                    "<br>turn-off=%{customdata[1]:.5g} s"
                    "<br>on-duration=%{customdata[2]:.5g} ms<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title=f"All accepted localizations on x/y/time axes ({data['x'].size:,} points)",
        height=820,
        margin={"l": 0, "r": 0, "t": 105, "b": 0},
        scene=_spatial_scene(
            config, "Peak time relative to reference (s)", _value_range(data["peak_s"])
        ),
        updatemenus=[_camera_menu()],
    )
    plot_id = "temporal-blink-localizations-xy-time-3d"
    _write_plotly_document(
        path,
        figure,
        plot_id,
        _three_d_controls_html(plot_id),
        _three_d_controls_script(
            plot_id,
            {
                0: {
                    "z": _value_range(data["peak_s"]),
                    "color": _value_range(data["on_duration_ms"], padding=0.0),
                }
            },
            marker_size=1.6,
            opacity=0.32,
        ),
    )


def _spatial_metric_traces(
    sampled: dict[str, np.ndarray],
    full_data: dict[str, np.ndarray],
    go: Any,
    *,
    marker_size: float,
    opacity: float,
) -> list[Any]:
    traces = []
    for index, metric in enumerate(TEMPORAL_METRICS):
        cmin, cmax = _value_range(full_data[metric.field], padding=0.0)
        traces.append(
            go.Scatter3d(
                x=sampled["x"],
                y=sampled["y"],
                z=sampled[metric.field],
                mode="markers",
                name=metric.label,
                visible=index == 0,
                marker={
                    "size": marker_size,
                    "color": sampled[metric.field],
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": metric.plotly_colorscale,
                    "opacity": opacity,
                    "colorbar": {"title": metric.unit},
                },
                customdata=np.column_stack(
                    (
                        sampled["turn_on_s"],
                        sampled["on_duration_ms"],
                        sampled["turn_off_s"],
                        sampled["rise_to_peak_ms"],
                        sampled["peak_to_last_ms"],
                    )
                ),
                hovertemplate=(
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>selected value=%{z:.5g}"
                    f" {metric.unit}<br>turn-on=%{{customdata[0]:.5g}} s"
                    "<br>duration=%{customdata[1]:.5g} ms"
                    "<br>turn-off=%{customdata[2]:.5g} s"
                    "<br>turn-on to peak=%{customdata[3]:.5g} ms"
                    "<br>peak to turn-off=%{customdata[4]:.5g} ms<extra></extra>"
                ),
            )
        )
    return traces


def _time_binned_dynamics(
    data: dict[str, np.ndarray], bin_count: int
) -> TemporalDynamicsBins:
    start, stop = _value_range(data["peak_s"], padding=0.0)
    edges = np.linspace(start, stop, num=bin_count + 1)
    bin_ids = np.clip(np.digitize(data["peak_s"], edges) - 1, 0, bin_count - 1)
    recording_time: list[float] = []
    rise_to_peak: list[float] = []
    on_duration: list[float] = []
    peak_to_last: list[float] = []
    counts: list[int] = []
    for index in range(bin_count):
        mask = bin_ids == index
        if not np.any(mask):
            continue
        recording_time.append(float(np.median(data["peak_s"][mask])))
        rise_to_peak.append(float(np.median(data["rise_to_peak_ms"][mask])))
        on_duration.append(float(np.median(data["on_duration_ms"][mask])))
        peak_to_last.append(float(np.median(data["peak_to_last_ms"][mask])))
        counts.append(int(np.count_nonzero(mask)))
    return TemporalDynamicsBins(
        recording_time_s=np.asarray(recording_time, dtype=np.float64),
        rise_to_peak_ms=np.asarray(rise_to_peak, dtype=np.float64),
        on_duration_ms=np.asarray(on_duration, dtype=np.float64),
        peak_to_last_ms=np.asarray(peak_to_last, dtype=np.float64),
        localization_count=np.asarray(counts, dtype=np.int64),
    )


def _metric_2d_menu() -> dict[str, Any]:
    return {
        "buttons": [
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {
                        "visible": [
                            item == index for item in range(len(TEMPORAL_METRICS))
                        ]
                    },
                    {"title.text": f"Full-resolution spatial map: {metric.label}"},
                ],
            }
            for index, metric in enumerate(TEMPORAL_METRICS)
        ],
        "direction": "down",
        "x": 0.0,
        "y": 1.14,
    }


def _metric_3d_menu(data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "buttons": [
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {
                        "visible": [
                            item == index for item in range(len(TEMPORAL_METRICS))
                        ]
                    },
                    {
                        "scene.zaxis.title.text": metric.axis_label,
                        "scene.zaxis.range": list(_value_range(data[metric.field])),
                    },
                ],
            }
            for index, metric in enumerate(TEMPORAL_METRICS)
        ],
        "direction": "down",
        "x": 0.0,
        "y": 1.14,
    }


def _pixel_metric_3d_menu(
    pixel_statistics: TemporalPixelStatistics,
) -> dict[str, Any]:
    return {
        "buttons": [
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {
                        "visible": [
                            item == index for item in range(len(TEMPORAL_METRICS))
                        ]
                    },
                    {
                        "scene.zaxis.title.text": metric.axis_label,
                        "scene.zaxis.range": list(
                            _value_range(pixel_statistics.medians[metric.field])
                        ),
                    },
                ],
            }
            for index, metric in enumerate(TEMPORAL_METRICS)
        ],
        "direction": "down",
        "x": 0.0,
        "y": 1.14,
    }


def _time_bin_menu(binned: list[TemporalDynamicsBins]) -> dict[str, Any]:
    return {
        "buttons": [
            {
                "label": f"{bin_count} time bins",
                "method": "update",
                "args": [
                    {"visible": [item == index for item in range(len(binned))]},
                    {"scene.zaxis.range": list(_value_range(values.on_duration_ms))},
                ],
            }
            for index, (bin_count, values) in enumerate(
                strict_zip(INTERACTIVE_TIME_BIN_COUNTS, binned)
            )
        ],
        "direction": "down",
        "x": 0.0,
        "y": 1.14,
    }


def _camera_menu() -> dict[str, Any]:
    return {
        "buttons": [
            {
                "label": "Perspective",
                "method": "relayout",
                "args": [
                    {
                        "scene.camera": {
                            "eye": {"x": 1.55, "y": 1.55, "z": 0.9},
                            "projection": {"type": "perspective"},
                        }
                    }
                ],
            },
            {
                "label": "Top",
                "method": "relayout",
                "args": [
                    {
                        "scene.camera": {
                            "eye": {"x": 0.0, "y": 0.0, "z": 2.4},
                            "projection": {"type": "orthographic"},
                        }
                    }
                ],
            },
            {
                "label": "Side",
                "method": "relayout",
                "args": [
                    {
                        "scene.camera": {
                            "eye": {"x": 2.4, "y": 0.0, "z": 0.0},
                            "projection": {"type": "orthographic"},
                        }
                    }
                ],
            },
        ],
        "direction": "down",
        "x": 0.42,
        "y": 1.14,
    }


def _spatial_scene(
    config: PeakLocConfig, z_title: str, z_range: tuple[float, float]
) -> dict[str, Any]:
    return {
        "xaxis": {"title": "x (sensor pixel)", "range": [0, config.sensor_width]},
        "yaxis": {"title": "y (sensor pixel)", "range": [config.sensor_height, 0]},
        "zaxis": {"title": z_title, "range": list(z_range)},
        "aspectmode": "manual",
        "aspectratio": {
            "x": config.sensor_width / config.sensor_height,
            "y": 1,
            "z": 0.7,
        },
    }


def _flat_spatial_scene(config: PeakLocConfig) -> dict[str, Any]:
    """Return a rotatable spatial scene whose data remain on z=0."""
    return {
        "xaxis": {"title": "x (sensor pixel)", "range": [0, config.sensor_width]},
        "yaxis": {"title": "y (sensor pixel)", "range": [config.sensor_height, 0]},
        "zaxis": {"visible": False, "range": [-1, 1]},
        "aspectmode": "manual",
        "aspectratio": {
            "x": config.sensor_width / config.sensor_height,
            "y": 1,
            "z": 0.15,
        },
        "camera": {
            "eye": {"x": 0.0, "y": 0.0, "z": 2.4},
            "projection": {"type": "orthographic"},
        },
    }


def _render_scene(
    width: int,
    height: int,
    z_title: str,
    z_range: tuple[float, float],
) -> dict[str, Any]:
    return {
        "xaxis": {"title": "x (SMLM render pixel)", "range": [0, width]},
        "yaxis": {"title": "y (SMLM render pixel)", "range": [height, 0]},
        "zaxis": {"title": z_title, "range": list(z_range)},
        "aspectmode": "manual",
        "aspectratio": {"x": width / height, "y": 1, "z": 0.7},
    }


def _render_shape(data: dict[str, np.ndarray]) -> tuple[int, int]:
    width = max(1, int(np.ceil((np.max(data["x"]) + 1) * RENDER_OVERSAMPLING)))
    height = max(1, int(np.ceil((np.max(data["y"]) + 1) * RENDER_OVERSAMPLING)))
    return width, height


def _value_range(values: np.ndarray, padding: float = 0.02) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        delta = max(abs(lower) * 0.01, 1.0)
        return lower - delta, upper + delta
    return lower, upper + (upper - lower) * padding


def _sample_for_interactive(
    data: dict[str, np.ndarray], maximum: int
) -> dict[str, np.ndarray]:
    count = data["x"].size
    if count <= maximum:
        return data
    generator = np.random.default_rng(0)
    indices = np.sort(generator.choice(count, size=maximum, replace=False))
    return {
        name: values[indices] if values.size == count else values
        for name, values in data.items()
    }


def _two_d_controls_html(plot_id: str) -> str:
    return f"""
<section class="controls">
  <strong>Active map controls</strong>
  <label>Color lower <input id="{plot_id}-color-low" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="0"></label>
  <output id="{plot_id}-color-low-value"></output>
  <label>Color upper <input id="{plot_id}-color-high" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="{INTERACTIVE_SLIDER_STEPS}"></label>
  <output id="{plot_id}-color-high-value"></output>
  <label>Marker size <input id="{plot_id}-size" type="range" min="1" max="8"
      value="2.5" step="0.25"></label>
  <label>Opacity <input id="{plot_id}-opacity" type="range" min="0.05" max="1"
      value="0.72" step="0.05"></label>
  <p>Choose a timing proxy from the menu above; drag to rotate the flat z=0 plane.</p>
</section>
"""


def _three_d_controls_html(plot_id: str) -> str:
    return f"""
<section class="controls">
  <strong>Active 3D trace controls</strong>
  <label>Z lower <input id="{plot_id}-z-low" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="0"></label>
  <output id="{plot_id}-z-low-value"></output>
  <label>Z upper <input id="{plot_id}-z-high" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="{INTERACTIVE_SLIDER_STEPS}"></label>
  <output id="{plot_id}-z-high-value"></output>
  <label>Color lower <input id="{plot_id}-color-low" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="0"></label>
  <output id="{plot_id}-color-low-value"></output>
  <label>Color upper <input id="{plot_id}-color-high" type="range" min="0"
      max="{INTERACTIVE_SLIDER_STEPS}" value="{INTERACTIVE_SLIDER_STEPS}"></label>
  <output id="{plot_id}-color-high-value"></output>
  <label>Marker size <input id="{plot_id}-size" type="range" min="1" max="10"
      value="3" step="0.2"></label>
  <label>Opacity <input id="{plot_id}-opacity" type="range" min="0.05" max="1"
      value="0.8" step="0.05"></label>
  <p>Rotate with drag. The camera menu provides top and side views; sliders affect the active trace.</p>
</section>
"""


def _two_d_controls_script(
    plot_id: str, ranges: dict[int, dict[str, tuple[float, float]]]
) -> str:
    template = """
(() => {
  const gd = document.getElementById(__PLOT_ID__);
  const ranges = __RANGES__;
  const low = document.getElementById(__PLOT_ID__ + '-color-low');
  const high = document.getElementById(__PLOT_ID__ + '-color-high');
  const lowValue = document.getElementById(__PLOT_ID__ + '-color-low-value');
  const highValue = document.getElementById(__PLOT_ID__ + '-color-high-value');
  const size = document.getElementById(__PLOT_ID__ + '-size');
  const opacity = document.getElementById(__PLOT_ID__ + '-opacity');
  const value = (slider, range) => range[0] + (range[1] - range[0]) *
    Number(slider.value) / __SLIDER_STEPS__;
  const format = number => Number(number).toPrecision(5);
  const activeTrace = () => gd.data.findIndex(trace =>
    trace.visible !== false && trace.visible !== 'legendonly');
  const applyColor = () => {
    const trace = activeTrace();
    const range = ranges[trace].color;
    let lower = value(low, range);
    let upper = value(high, range);
    if (lower >= upper) {
      upper = Math.min(range[1], lower + (range[1] - range[0]) / __SLIDER_STEPS__);
    }
    lowValue.textContent = format(lower);
    highValue.textContent = format(upper);
    Plotly.restyle(gd, {'marker.cmin': lower, 'marker.cmax': upper}, [trace]);
  };
  const applyStyle = () => {
    const trace = activeTrace();
    Plotly.restyle(gd, {
      'marker.size': Number(size.value),
      'marker.opacity': Number(opacity.value),
    }, [trace]);
  };
  const reset = () => {
    low.value = 0;
    high.value = __SLIDER_STEPS__;
    applyColor();
  };
  low.addEventListener('input', applyColor);
  high.addEventListener('input', applyColor);
  size.addEventListener('input', applyStyle);
  opacity.addEventListener('input', applyStyle);
  gd.on('plotly_buttonclicked', () => window.setTimeout(reset, 0));
  window.setTimeout(reset, 0);
})();
"""
    return _controls_script(template, plot_id, ranges)


def _three_d_controls_script(
    plot_id: str,
    ranges: dict[int, dict[str, tuple[float, float]]],
    *,
    marker_size: float,
    opacity: float,
) -> str:
    template = """
(() => {
  const gd = document.getElementById(__PLOT_ID__);
  const ranges = __RANGES__;
  const zLow = document.getElementById(__PLOT_ID__ + '-z-low');
  const zHigh = document.getElementById(__PLOT_ID__ + '-z-high');
  const zLowValue = document.getElementById(__PLOT_ID__ + '-z-low-value');
  const zHighValue = document.getElementById(__PLOT_ID__ + '-z-high-value');
  const colorLow = document.getElementById(__PLOT_ID__ + '-color-low');
  const colorHigh = document.getElementById(__PLOT_ID__ + '-color-high');
  const colorLowValue = document.getElementById(__PLOT_ID__ + '-color-low-value');
  const colorHighValue = document.getElementById(__PLOT_ID__ + '-color-high-value');
  const size = document.getElementById(__PLOT_ID__ + '-size');
  const opacity = document.getElementById(__PLOT_ID__ + '-opacity');
  const value = (slider, range) => range[0] + (range[1] - range[0]) *
    Number(slider.value) / __SLIDER_STEPS__;
  const format = number => Number(number).toPrecision(5);
  const activeTrace = () => gd.data.findIndex(trace =>
    trace.visible !== false && trace.visible !== 'legendonly');
  const bounded = (lower, upper, range) => {
    if (lower < upper) return [lower, upper];
    return [lower, Math.min(range[1], lower + (range[1] - range[0]) / __SLIDER_STEPS__)];
  };
  const applyZ = () => {
    const range = ranges[activeTrace()].z;
    const [lower, upper] = bounded(value(zLow, range), value(zHigh, range), range);
    zLowValue.textContent = format(lower);
    zHighValue.textContent = format(upper);
    Plotly.relayout(gd, {'scene.zaxis.range': [lower, upper]});
  };
  const applyColor = () => {
    const trace = activeTrace();
    const range = ranges[trace].color;
    const [lower, upper] = bounded(value(colorLow, range), value(colorHigh, range), range);
    colorLowValue.textContent = format(lower);
    colorHighValue.textContent = format(upper);
    Plotly.restyle(gd, {'marker.cmin': lower, 'marker.cmax': upper}, [trace]);
  };
  const applyStyle = () => {
    const trace = activeTrace();
    Plotly.restyle(gd, {
      'marker.size': Number(size.value),
      'marker.opacity': Number(opacity.value),
    }, [trace]);
  };
  const reset = () => {
    zLow.value = 0;
    zHigh.value = __SLIDER_STEPS__;
    colorLow.value = 0;
    colorHigh.value = __SLIDER_STEPS__;
    applyZ();
    applyColor();
  };
  size.value = __MARKER_SIZE__;
  opacity.value = __OPACITY__;
  zLow.addEventListener('input', applyZ);
  zHigh.addEventListener('input', applyZ);
  colorLow.addEventListener('input', applyColor);
  colorHigh.addEventListener('input', applyColor);
  size.addEventListener('input', applyStyle);
  opacity.addEventListener('input', applyStyle);
  gd.on('plotly_buttonclicked', () => window.setTimeout(reset, 0));
  window.setTimeout(reset, 0);
})();
"""
    return (
        _controls_script(template, plot_id, ranges)
        .replace("__MARKER_SIZE__", str(marker_size))
        .replace("__OPACITY__", str(opacity))
    )


def _controls_script(
    template: str,
    plot_id: str,
    ranges: dict[int, dict[str, tuple[float, float]]],
) -> str:
    serializable_ranges = {
        str(index): {name: list(value) for name, value in trace_ranges.items()}
        for index, trace_ranges in ranges.items()
    }
    return (
        template.replace("__PLOT_ID__", json.dumps(plot_id))
        .replace("__RANGES__", json.dumps(serializable_ranges))
        .replace("__SLIDER_STEPS__", str(INTERACTIVE_SLIDER_STEPS))
    )


def _write_plotly_document(
    path: Path,
    figure: Any,
    plot_id: str,
    controls_html: str,
    controls_script: str,
) -> None:
    figure_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
        div_id=plot_id,
    )
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>PeakLoc temporal visualization</title>\n'
        "<style>\n"
        "body { margin: 0; font-family: sans-serif; }\n"
        ".controls { display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: center; "
        "padding: 0.75rem 1rem; background: #f3f5f7; }\n"
        ".controls strong, .controls p { flex-basis: 100%; margin: 0; }\n"
        ".controls label { display: flex; gap: 0.35rem; align-items: center; }\n"
        ".controls input { width: 9rem; }\n"
        ".controls output { min-width: 4.5rem; }\n"
        "</style></head>\n<body>\n"
        f"{controls_html}\n{figure_html}\n<script>{controls_script}</script>\n"
        "</body></html>\n",
        encoding="utf-8",
    )


def _write_plotly_unavailable_page(path: Path, message: str | None = None) -> None:
    text = message or "Plotly is unavailable in this environment."
    path.write_text(f"<html><body>{text}</body></html>\n", encoding="utf-8")


def _summary_markdown(summary: dict[str, Any]) -> str:
    duration = summary["on_duration_ms"]
    rise = summary["rise_to_peak_ms"]
    decay = summary["peak_to_last_event_ms"]
    spatial = summary["spatial_binning"]
    pixels = summary["sensor_pixel_medians"]
    segmented = summary["timing_source"] == "segmented_transition_trains"
    duration_name = (
        "ON-train to OFF-train state interval"
        if segmented
        else "first-to-last ROI event duration"
    )
    first_name = "ON-train start" if segmented else "first event"
    last_name = "OFF-train end" if segmented else "last event"
    return "\n".join(
        [
            "# Temporal Blink Statistics",
            "",
            f"- Valid temporal localizations: `{summary['valid_temporal_localizations']}`",
            f"- Timing source: `{summary['timing_source']}`",
            f"- Time reference: `{summary['time_reference']}`",
            f"- Median {duration_name}: `{duration['median']:.3g} ms`",
            f"- 90th percentile duration: `{duration['p90']:.3g} ms`",
            f"- Median {first_name}-to-peak delay: `{rise['median']:.3g} ms`",
            f"- Median peak-to-{last_name} delay: `{decay['median']:.3g} ms`",
            f"- Occupied native sensor pixels: `{pixels['occupied_pixel_count']}`",
            f"- Spatial bins with at least {spatial['minimum_localizations_per_bin']} "
            f"localizations: `{spatial['qualified_bin_count']}`",
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
        ]
    )
