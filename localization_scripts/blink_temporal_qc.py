from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from localization_scripts.pipeline_config import PeakLocConfig


TEMPORAL_FIELDS = frozenset({"x", "y", "t_1st", "t_peak", "t_last"})
MAX_SPATIAL_BINS = 24
MIN_LOCALIZATIONS_PER_SPATIAL_BIN = 5


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
    summary = _summary(data, spatial_bins, config)
    summary_json_path = output_dir / "temporal_blink_statistics.json"
    summary_md_path = output_dir / "temporal_blink_statistics.md"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    spatial_path = output_dir / "18_temporal_blink_spatial_maps.png"
    bin_csv_path = output_dir / "19_temporal_blink_spatial_bin_statistics.csv"
    distribution_path = output_dir / "20_temporal_blink_timing_distributions.png"
    _save_spatial_maps(data, spatial_bins, config, spatial_path)
    _write_spatial_bins_csv(spatial_bins, bin_csv_path)
    _save_timing_distributions(data, config, distribution_path)
    paths = [
        summary_json_path,
        summary_md_path,
        spatial_path,
        bin_csv_path,
        distribution_path,
    ]

    if config.qc_generate_temporal_3d:
        plotly_path = output_dir / "temporal_blink_spatial_3d.html"
        _save_plotly_3d(data, config, plotly_path)
        paths.append(plotly_path)
    return paths


def _temporal_data(localizations: np.ndarray) -> dict[str, np.ndarray] | None:
    if localizations.size == 0 or not TEMPORAL_FIELDS.issubset(
        localizations.dtype.names or ()
    ):
        return None
    x = np.asarray(localizations["x"], dtype=np.float64)
    y = np.asarray(localizations["y"], dtype=np.float64)
    first = np.asarray(localizations["t_1st"], dtype=np.float64)
    peak = np.asarray(localizations["t_peak"], dtype=np.float64)
    last = np.asarray(localizations["t_last"], dtype=np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(first)
        & np.isfinite(peak)
        & np.isfinite(last)
        & (first >= 0)
        & (first <= peak)
        & (peak <= last)
    )
    if not np.any(valid):
        return None

    first = first[valid]
    peak = peak[valid]
    last = last[valid]
    reference = float(np.min(first))
    return {
        "x": x[valid],
        "y": y[valid],
        "turn_on_s": (first - reference) * 1e-6,
        "peak_s": (peak - reference) * 1e-6,
        "turn_off_s": (last - reference) * 1e-6,
        "rise_to_peak_ms": (peak - first) * 1e-3,
        "on_duration_ms": (last - first) * 1e-3,
        "peak_to_last_ms": (last - peak) * 1e-3,
        "reference_time_us": np.asarray([reference]),
    }


def _summary(
    data: dict[str, np.ndarray], spatial_bins: np.ndarray, config: PeakLocConfig
) -> dict[str, Any]:
    return {
        "valid_temporal_localizations": int(data["x"].size),
        "time_reference": "first valid ROI event in this run",
        "reference_time_us": float(data["reference_time_us"][0]),
        "turn_on_relative_s": _distribution_summary(data["turn_on_s"]),
        "peak_relative_s": _distribution_summary(data["peak_s"]),
        "turn_off_relative_s": _distribution_summary(data["turn_off_s"]),
        "rise_to_peak_ms": _distribution_summary(data["rise_to_peak_ms"]),
        "on_duration_ms": _distribution_summary(data["on_duration_ms"]),
        "peak_to_last_event_ms": _distribution_summary(data["peak_to_last_ms"]),
        "spatial_binning": _spatial_summary(spatial_bins),
        "plotly_3d": {
            "enabled": config.qc_generate_temporal_3d,
            "point_count": int(
                min(data["x"].size, config.qc_max_events_for_interactive)
            ),
            "sampling_limit": config.qc_max_events_for_interactive,
        },
        "interpretation": (
            "t_1st and t_last are the first and last event timestamps inside the "
            "fitted ROI window. They are turn-on, on-duration, and turn-off "
            "proxies, not direct molecular transition timestamps."
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


def _save_spatial_maps(
    data: dict[str, np.ndarray],
    spatial_bins: np.ndarray,
    config: PeakLocConfig,
    path: Path,
) -> None:
    density = np.histogram2d(
        data["y"],
        data["x"],
        bins=(config.sensor_height, config.sensor_width),
        range=((0, config.sensor_height), (0, config.sensor_width)),
    )[0]
    fields = (
        (
            "turn_on_s",
            "ROI-window turn-on proxy relative to reference (s)",
            "viridis",
        ),
        ("on_duration_ms", "ROI-window on-duration proxy (ms)", "magma"),
        (
            "turn_off_s",
            "ROI-window turn-off proxy relative to reference (s)",
            "cividis",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (field, label, cmap) in zip(axes.flat[:3], fields, strict=True):
        _plot_density_background(axis, density, config)
        points = axis.scatter(
            data["x"],
            data["y"],
            c=data[field],
            cmap=cmap,
            s=3,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )
        axis.set(title=label, xlabel="x (pixel)", ylabel="y (pixel)")
        figure.colorbar(points, ax=axis, fraction=0.046, label=label)

    _plot_binned_duration_map(axes.flat[3], spatial_bins, config)
    figure.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_density_background(
    axis: Axes, density: np.ndarray, config: PeakLocConfig
) -> None:
    axis.imshow(
        np.log1p(density),
        cmap="gray",
        origin="upper",
        extent=(0, config.sensor_width, config.sensor_height, 0),
        aspect="auto",
    )


def _plot_binned_duration_map(
    axis: Axes, spatial_bins: np.ndarray, config: PeakLocConfig
) -> None:
    qualified = spatial_bins[spatial_bins["eligible_for_summary"]]
    if qualified.size == 0:
        axis.set(title="No spatial bins with sufficient localizations")
        axis.set(xlabel="x (pixel)", ylabel="y (pixel)")
        return
    points = axis.scatter(
        (qualified["x_start_px"] + qualified["x_end_px"]) / 2,
        (qualified["y_start_px"] + qualified["y_end_px"]) / 2,
        c=qualified["median_on_duration_ms"],
        cmap="magma",
        s=np.clip(qualified["localization_count"] * 4, 24, 180),
        marker="s",
        linewidths=0,
    )
    axis.set(
        title="Median ROI-window on-duration by spatial bin",
        xlabel="x (pixel)",
        ylabel="y (pixel)",
        xlim=(0, config.sensor_width),
        ylim=(config.sensor_height, 0),
    )
    plt.colorbar(points, ax=axis, fraction=0.046, label="median duration (ms)")


def _write_spatial_bins_csv(spatial_bins: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(spatial_bins.dtype.names)
        for row in spatial_bins:
            writer.writerow(row.tolist())


def _save_timing_distributions(
    data: dict[str, np.ndarray], config: PeakLocConfig, path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for field, label, color in (
        ("turn_on_s", "ROI-window turn-on", "#0072B2"),
        ("peak_s", "Peak", "#009E73"),
        ("turn_off_s", "ROI-window turn-off", "#D55E00"),
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
        xlabel="Time relative to first valid ROI event (s)", ylabel="Localizations"
    )
    axes[0].legend()
    for field, label, color in (
        ("rise_to_peak_ms", "ROI-window turn-on to peak", "#0072B2"),
        ("on_duration_ms", "ROI-window on-duration", "#009E73"),
        ("peak_to_last_ms", "Peak to ROI-window turn-off", "#D55E00"),
    ):
        axes[1].hist(
            data[field],
            bins=60,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=color,
        )
    axes[1].set(xlabel="ROI event timing proxy (ms)", ylabel="Localizations")
    axes[1].legend()
    figure.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(figure)


def _save_plotly_3d(
    data: dict[str, np.ndarray], config: PeakLocConfig, path: Path
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        path.write_text(
            "<html><body>Plotly is unavailable in this environment.</body></html>\n",
            encoding="utf-8",
        )
        return

    sampled = _sample_for_interactive(data, config.qc_max_events_for_interactive)
    fields = (
        (
            "turn_on_s",
            "ROI-window turn-on proxy",
            "ROI-window turn-on proxy relative to reference (s)",
            "Viridis",
            "s",
        ),
        (
            "on_duration_ms",
            "ROI-window on-duration proxy",
            "ROI-window on-duration proxy (ms)",
            "Magma",
            "ms",
        ),
        (
            "turn_off_s",
            "ROI-window turn-off proxy",
            "ROI-window turn-off proxy relative to reference (s)",
            "Cividis",
            "s",
        ),
    )
    traces = []
    for index, (field, label, z_label, colorscale, unit) in enumerate(fields):
        traces.append(
            go.Scatter3d(
                x=sampled["x"],
                y=sampled["y"],
                z=sampled[field],
                mode="markers",
                name=label,
                visible=index == 0,
                marker={
                    "size": 2,
                    "color": sampled[field],
                    "colorscale": colorscale,
                    "opacity": 0.75,
                    "colorbar": {"title": unit},
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
                    "x=%{x:.2f}<br>y=%{y:.2f}<br>selected value=%{z:.4f}"
                    f" {unit}<br>turn-on=%{{customdata[0]:.4f}} s"
                    "<br>duration=%{customdata[1]:.2f} ms"
                    "<br>turn-off=%{customdata[2]:.4f} s"
                    "<br>turn-on to peak=%{customdata[3]:.2f} ms"
                    "<br>peak to turn-off=%{customdata[4]:.2f} ms<extra></extra>"
                ),
            )
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=(
            "ROI-window timing in spatial coordinates "
            f"({sampled['x'].size:,} of {data['x'].size:,} localizations)"
        ),
        scene={
            "xaxis_title": "x (pixel)",
            "yaxis_title": "y (pixel)",
            "zaxis_title": fields[0][2],
            "yaxis": {"autorange": "reversed"},
        },
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": label,
                        "method": "update",
                        "args": [
                            {"visible": [item == index for item in range(len(traces))]},
                            {"scene.zaxis.title.text": z_label},
                        ],
                    }
                    for index, (_, label, z_label, _, _) in enumerate(fields)
                ],
                "direction": "down",
                "x": 0.0,
                "y": 1.1,
            }
        ],
    )
    figure.write_html(path, full_html=True)


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


def _summary_markdown(summary: dict[str, Any]) -> str:
    duration = summary["on_duration_ms"]
    rise = summary["rise_to_peak_ms"]
    decay = summary["peak_to_last_event_ms"]
    spatial = summary["spatial_binning"]
    return "\n".join(
        [
            "# Temporal Blink Statistics",
            "",
            f"- Valid temporal localizations: `{summary['valid_temporal_localizations']}`",
            f"- Time reference: `{summary['time_reference']}`",
            f"- Median first-to-last ROI event duration: `{duration['median']:.3g} ms`",
            f"- 90th percentile duration: `{duration['p90']:.3g} ms`",
            f"- Median first-event-to-peak delay: `{rise['median']:.3g} ms`",
            f"- Median peak-to-last-event delay: `{decay['median']:.3g} ms`",
            f"- Spatial bins with at least {spatial['minimum_localizations_per_bin']} "
            f"localizations: `{spatial['qualified_bin_count']}`",
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
        ]
    )
