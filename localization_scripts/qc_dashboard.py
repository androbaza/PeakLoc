from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.fit_review import save_uncertainty_montages
from localization_scripts.localization_fitting import localization_uncertainty_px
from localization_scripts.pipeline_config import PeakLocConfig, write_effective_config
from localization_scripts.plot_style import (
    EVENT_DENSITY_CMAP,
    PLOT_COLORS,
    SEQUENTIAL_CMAP,
)
from localization_scripts.postprocessing import save_postprocessing_qc
from localization_scripts.preflight import run_preflight, write_preflight_report


@dataclass(frozen=True)
class QCDashboardSummary:
    input_file: str
    output_dir: str
    event_count: int
    attempted_fit_count: int
    accepted_localization_count: int
    accepted_from_qc_count: int
    detection_funnel: dict[str, int]
    rejection_reasons: dict[str, int]
    median_uncertainty_px: float | None
    median_uncertainty_nm: float | None
    p90_uncertainty_px: float | None
    p90_uncertainty_nm: float | None
    warnings: list[str]


def save_run_qc_dashboard(
    *,
    recording: Any,
    config: PeakLocConfig,
    localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    rois: np.ndarray | None,
    events: np.ndarray | None,
    timestamp: str,
) -> list[Path]:
    output_dir = Path(recording.output_folder) / config.qc_output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    summary = build_qc_summary(
        recording=recording,
        config=config,
        localizations=localizations,
        attempted_localizations=attempted_localizations,
        localization_qc=localization_qc,
        rois=rois,
    )
    summary_json_path = output_dir / "run_qc_summary.json"
    summary_md_path = output_dir / "run_qc_summary.md"
    summary_json_path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    artifacts.extend([summary_json_path, summary_md_path])

    effective_config_path = output_dir / "effective_config.json"
    write_effective_config(config, effective_config_path)
    artifacts.append(effective_config_path)

    preflight_path = output_dir / "preflight_report.md"
    write_preflight_report(run_preflight(config), preflight_path)
    artifacts.append(preflight_path)

    static_paths = save_static_qc_figures(
        output_dir=output_dir,
        config=config,
        recording=recording,
        summary=summary,
        localizations=localizations,
        attempted_localizations=attempted_localizations,
        localization_qc=localization_qc,
        rois=rois,
        events=events,
    )
    artifacts.extend(static_paths)

    interactive_paths: list[Path] = []
    if config.qc_generate_interactive:
        interactive_paths = save_interactive_qc_figures(
            output_dir=output_dir,
            config=config,
            localizations=localizations,
            attempted_localizations=attempted_localizations,
            localization_qc=localization_qc,
            events=events,
        )
        artifacts.extend(interactive_paths)

    if config.qc_generate_html:
        index_path = output_dir / "index.html"
        index_path.write_text(
            _index_html(summary, static_paths, interactive_paths, timestamp),
            encoding="utf-8",
        )
        artifacts.append(index_path)

    return artifacts


def build_qc_summary(
    *,
    recording: Any,
    config: PeakLocConfig,
    localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    rois: np.ndarray | None,
) -> QCDashboardSummary:
    accepted_from_qc = _bool_sum(localization_qc, "accepted")
    accepted_count = int(localizations.size)
    attempted_count = int(attempted_localizations.size)
    successful_count = _bool_sum(localization_qc, "fit_success")
    unique_peaks = int(
        sum(result.unique_peak_count for result in recording.slice_results)
    )
    roi_count = int(
        rois.size
        if rois is not None
        else sum(result.roi_count for result in recording.slice_results)
    )
    warnings: list[str] = []
    if attempted_count != int(localization_qc.size):
        warnings.append("Attempted localization count differs from QC table row count.")
    if accepted_from_qc != accepted_count:
        warnings.append("Accepted localization count differs from QC accepted rows.")
    if accepted_count == 0:
        warnings.append("No accepted localizations were produced.")

    uncertainty_px = _uncertainty_from_qc_or_locs(localization_qc, localizations)
    finite_uncertainty = uncertainty_px[np.isfinite(uncertainty_px)]
    median_px = _percentile(finite_uncertainty, 50)
    p90_px = _percentile(finite_uncertainty, 90)
    median_nm = None if median_px is None else median_px * config.optical_pixel_size_nm
    p90_nm = None if p90_px is None else p90_px * config.optical_pixel_size_nm

    detection_funnel = {
        "events_loaded": int(recording.event_count),
        "peak_candidates": unique_peaks,
        "local_maxima": unique_peaks,
        "rois_generated": roi_count,
        "attempted_fits": attempted_count,
        "successful_fits": successful_count,
        "accepted_fits": accepted_from_qc,
    }
    return QCDashboardSummary(
        input_file=str(recording.input_file),
        output_dir=str(Path(recording.output_folder) / config.qc_output_dirname),
        event_count=int(recording.event_count),
        attempted_fit_count=attempted_count,
        accepted_localization_count=accepted_count,
        accepted_from_qc_count=accepted_from_qc,
        detection_funnel=detection_funnel,
        rejection_reasons=_rejection_counts(localization_qc),
        median_uncertainty_px=median_px,
        median_uncertainty_nm=median_nm,
        p90_uncertainty_px=p90_px,
        p90_uncertainty_nm=p90_nm,
        warnings=warnings,
    )


def save_static_qc_figures(
    *,
    output_dir: Path,
    config: PeakLocConfig,
    recording: Any,
    summary: QCDashboardSummary,
    localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    rois: np.ndarray | None,
    events: np.ndarray | None,
) -> list[Path]:
    paths: list[Path] = []
    density_total, density_pos, density_neg = _event_density_images(events, config)
    figure_specs = [
        (
            "01_event_density_total.png",
            lambda ax: _plot_image(ax, density_total, "Event density"),
        ),
        (
            "02_event_density_polarity.png",
            lambda ax: _plot_polarity_density(ax, density_pos, density_neg),
        ),
        (
            "03_peak_candidate_density.png",
            lambda ax: _plot_peak_candidate_density(
                ax, attempted_localizations, config
            ),
        ),
        (
            "04_detection_funnel.png",
            lambda ax: _plot_bar(ax, summary.detection_funnel, "Detection funnel"),
        ),
        (
            "05_fit_rejection_reasons.png",
            lambda ax: _plot_bar(
                ax, summary.rejection_reasons, "Fit rejection reasons"
            ),
        ),
        (
            "06_uncertainty_histogram.png",
            lambda ax: _plot_uncertainty_histogram(ax, localization_qc, config),
        ),
        (
            "07_uncertainty_vs_events.png",
            lambda ax: _plot_uncertainty_vs_field(
                ax, localization_qc, "E_total", "Events"
            ),
        ),
        (
            "08_uncertainty_vs_nll.png",
            lambda ax: _plot_uncertainty_vs_field(
                ax, localization_qc, "nll_per_event", "NLL/event"
            ),
        ),
        (
            "09_fit_condition_vs_uncertainty.png",
            lambda ax: _plot_uncertainty_vs_field(
                ax, localization_qc, "fit_cond", "Fit condition"
            ),
        ),
        (
            "10_background_vs_signal.png",
            lambda ax: _plot_background_vs_signal(ax, attempted_localizations),
        ),
        (
            "11_localization_density_render.png",
            lambda ax: _plot_localization_density(ax, localizations, config),
        ),
        (
            "12_time_binned_localization_count.png",
            lambda ax: _plot_time_binned_count(ax, localizations),
        ),
        (
            "13_time_binned_median_uncertainty.png",
            lambda ax: _plot_time_binned_uncertainty(ax, localizations, config),
        ),
        (
            "14_spatial_uncertainty_heatmap.png",
            lambda ax: _plot_spatial_uncertainty(ax, localizations, config),
        ),
        (
            "15_hot_pixel_overlay.png",
            lambda ax: _plot_hot_pixel_overlay(
                ax, density_total, attempted_localizations
            ),
        ),
    ]
    for filename, plotter in figure_specs:
        paths.extend(_save_single_axis_figure(output_dir / filename, config, plotter))

    montage_paths = save_uncertainty_montages(
        attempted_localizations,
        localizations,
        localization_qc,
        output_dir,
        config=config,
        n=config.qc_uncertainty_montage_n,
        dpi=config.qc_static_dpi,
    )
    paths.extend(montage_paths)
    lowest = (
        output_dir
        / f"uncertainty_lowest_{config.qc_uncertainty_montage_n}_combined.png"
    )
    highest = (
        output_dir
        / f"uncertainty_highest_{config.qc_uncertainty_montage_n}_combined.png"
    )
    paths.extend(
        _copy_if_exists(
            [
                (lowest, output_dir / "16_lowest_uncertainty_36.png"),
                (highest, output_dir / "17_highest_uncertainty_36.png"),
            ]
        )
    )
    paths.extend(save_postprocessing_qc(localizations, config, output_dir))
    return paths


def save_interactive_qc_figures(
    *,
    output_dir: Path,
    config: PeakLocConfig,
    localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    events: np.ndarray | None,
) -> list[Path]:
    try:
        import plotly.graph_objects as go
    except Exception:
        return []

    paths: list[Path] = []
    if events is not None and events.size and _has_fields(events, {"x", "y", "t", "p"}):
        sampled = _sample_events(events, config.qc_max_events_for_interactive)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=sampled["x"],
                    y=sampled["y"],
                    z=sampled["t"],
                    mode="markers",
                    marker={"size": 2, "color": sampled["p"], "colorscale": "Viridis"},
                    text=[f"p={p}, t={t}" for p, t in zip(sampled["p"], sampled["t"])],
                    name="events",
                )
            ]
        )
        fig.update_layout(
            title="3D event point cloud",
            scene={"xaxis_title": "x px", "yaxis_title": "y px", "zaxis_title": "t us"},
        )
        path = output_dir / "interactive_event_point_cloud.html"
        fig.write_html(path)
        paths.append(path)

    loc_source = (
        attempted_localizations if attempted_localizations.size else localizations
    )
    if loc_source.size and _has_fields(loc_source, {"x", "y", "t_peak"}):
        uncertainty = _uncertainty_for_locs(loc_source)
        reasons = _reason_by_id(localization_qc)
        ids = _ids_or_range(loc_source)
        hover_text = [
            f"id={row_id}<br>uncertainty={unc:.3g}px<br>reason={reasons.get(int(row_id), 'accepted')}"
            for row_id, unc in zip(ids, uncertainty)
        ]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=loc_source["x"],
                    y=loc_source["y"],
                    z=loc_source["t_peak"],
                    mode="markers",
                    marker={"size": 3, "color": uncertainty, "colorscale": "Cividis"},
                    text=hover_text,
                    name="localizations",
                )
            ]
        )
        fig.update_layout(
            title="3D localization point cloud",
            scene={
                "xaxis_title": "x px",
                "yaxis_title": "y px",
                "zaxis_title": "t_peak us",
            },
        )
        path = output_dir / "interactive_localization_point_cloud.html"
        fig.write_html(path)
        paths.append(path)

        map_path = output_dir / "interactive_time_slider_localization_map.html"
        _write_static_interactive_map(map_path, loc_source, uncertainty, hover_text)
        paths.append(map_path)
    return paths


def _save_single_axis_figure(path: Path, config: PeakLocConfig, plotter) -> list[Path]:
    fig, axis = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    plotter(axis)
    saved = [path]
    fig.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    if config.qc_save_vector:
        for suffix in (".svg", ".pdf"):
            vector_path = path.with_suffix(suffix)
            fig.savefig(vector_path, bbox_inches="tight")
            saved.append(vector_path)
    plt.close(fig)
    return saved


def _event_density_images(
    events: np.ndarray | None, config: PeakLocConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = config.sensor_shape
    total = np.zeros(shape, dtype=np.float32)
    positive = np.zeros(shape, dtype=np.float32)
    negative = np.zeros(shape, dtype=np.float32)
    if events is None or events.size == 0 or not _has_fields(events, {"x", "y", "p"}):
        return total, positive, negative
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    np.add.at(total, (y[valid], x[valid]), 1)
    np.add.at(positive, (y[valid & (p == 1)], x[valid & (p == 1)]), 1)
    np.add.at(negative, (y[valid & (p == 0)], x[valid & (p == 0)]), 1)
    return total, positive, negative


def _plot_image(axis, image: np.ndarray, title: str) -> None:
    im = axis.imshow(image, cmap=EVENT_DENSITY_CMAP, origin="upper")
    axis.set_title(title)
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")
    plt.colorbar(im, ax=axis, fraction=0.046)


def _plot_polarity_density(axis, positive: np.ndarray, negative: np.ndarray) -> None:
    axis.imshow(positive, cmap="Greens", origin="upper", alpha=0.65)
    axis.imshow(negative, cmap="Blues", origin="upper", alpha=0.55)
    axis.set_title("Polarity event density")
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")


def _plot_peak_candidate_density(
    axis, attempted_localizations: np.ndarray, config: PeakLocConfig
) -> None:
    if attempted_localizations.size == 0 or not _has_fields(
        attempted_localizations, {"x", "y"}
    ):
        _empty_axis(axis, "No peak candidate coordinates")
        return
    axis.hist2d(
        attempted_localizations["x"],
        attempted_localizations["y"],
        bins=80,
        range=[[0, config.sensor_width], [0, config.sensor_height]],
        cmap="viridis",
    )
    axis.invert_yaxis()
    axis.set_title("Peak/attempt density")
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")


def _plot_bar(axis, values: dict[str, int], title: str) -> None:
    if not values:
        _empty_axis(axis, "No data")
        return
    labels = list(values)
    counts = [values[label] for label in labels]
    axis.bar(labels, counts, color=PLOT_COLORS["blue"])
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylabel("count")


def _plot_uncertainty_histogram(
    axis, localization_qc: np.ndarray, config: PeakLocConfig
) -> None:
    uncertainty = _field(localization_qc, "uncertainty_nm")
    finite = uncertainty[np.isfinite(uncertainty)]
    if finite.size == 0:
        _empty_axis(axis, "No finite uncertainty")
        return
    axis.hist(finite, bins=30, color=PLOT_COLORS["green"], alpha=0.85)
    if config.max_localization_uncertainty_nm is not None:
        axis.axvline(
            config.max_localization_uncertainty_nm,
            color=PLOT_COLORS["vermillion"],
            linestyle="--",
        )
    axis.set_title("Localization uncertainty")
    axis.set_xlabel("uncertainty nm")
    axis.set_ylabel("count")


def _plot_uncertainty_vs_field(
    axis, localization_qc: np.ndarray, field_name: str, x_label: str
) -> None:
    x = _field(localization_qc, field_name)
    y = _field(localization_qc, "uncertainty_nm")
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        _empty_axis(axis, f"No {x_label} data")
        return
    axis.scatter(x[valid], y[valid], s=8, color=PLOT_COLORS["blue"], alpha=0.7)
    axis.set_title(f"Uncertainty vs {x_label}")
    axis.set_xlabel(x_label)
    axis.set_ylabel("uncertainty nm")


def _plot_background_vs_signal(axis, attempted_localizations: np.ndarray) -> None:
    if not _has_fields(
        attempted_localizations,
        {"A_pos", "A_neg", "bg_pos_local", "bg_neg_local"},
    ):
        _empty_axis(axis, "No signal/background fields")
        return
    signal = attempted_localizations["A_pos"] + attempted_localizations["A_neg"]
    background = _field(attempted_localizations, "bg_pos_local") + _field(
        attempted_localizations, "bg_neg_local"
    )
    valid = np.isfinite(signal) & np.isfinite(background)
    if not np.any(valid):
        _empty_axis(axis, "No finite signal/background")
        return
    axis.scatter(
        background[valid],
        signal[valid],
        s=8,
        color=PLOT_COLORS["reddish_purple"],
        alpha=0.7,
    )
    axis.set_title("Background vs signal")
    axis.set_xlabel("local background events")
    axis.set_ylabel("signal events")


def _plot_localization_density(
    axis, localizations: np.ndarray, config: PeakLocConfig
) -> None:
    if localizations.size == 0 or not _has_fields(localizations, {"x", "y"}):
        _empty_axis(axis, "No accepted localizations")
        return
    axis.hist2d(
        localizations["x"],
        localizations["y"],
        bins=100,
        range=[[0, config.sensor_width], [0, config.sensor_height]],
        cmap=SEQUENTIAL_CMAP,
    )
    axis.invert_yaxis()
    axis.set_title("Accepted localization density")
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")


def _plot_time_binned_count(axis, localizations: np.ndarray) -> None:
    if localizations.size == 0 or "t_peak" not in (localizations.dtype.names or ()):
        _empty_axis(axis, "No localization timing")
        return
    axis.hist(
        localizations["t_peak"],
        bins=min(30, max(1, localizations.size)),
        color=PLOT_COLORS["orange"],
    )
    axis.set_title("Time-binned localization count")
    axis.set_xlabel("t_peak us")
    axis.set_ylabel("count")


def _plot_time_binned_uncertainty(
    axis, localizations: np.ndarray, config: PeakLocConfig
) -> None:
    if localizations.size == 0 or "t_peak" not in (localizations.dtype.names or ()):
        _empty_axis(axis, "No localization timing")
        return
    uncertainty = _uncertainty_for_locs(localizations) * config.optical_pixel_size_nm
    valid = np.isfinite(uncertainty)
    if not np.any(valid):
        _empty_axis(axis, "No finite uncertainty")
        return
    axis.scatter(
        localizations["t_peak"][valid],
        uncertainty[valid],
        s=8,
        color=PLOT_COLORS["green"],
    )
    axis.set_title("Time vs uncertainty")
    axis.set_xlabel("t_peak us")
    axis.set_ylabel("uncertainty nm")


def _plot_spatial_uncertainty(
    axis, localizations: np.ndarray, config: PeakLocConfig
) -> None:
    if localizations.size == 0 or not _has_fields(localizations, {"x", "y"}):
        _empty_axis(axis, "No spatial uncertainty")
        return
    uncertainty = _uncertainty_for_locs(localizations) * config.optical_pixel_size_nm
    valid = np.isfinite(uncertainty)
    if not np.any(valid):
        _empty_axis(axis, "No finite uncertainty")
        return
    plot = axis.scatter(
        localizations["x"][valid],
        localizations["y"][valid],
        c=uncertainty[valid],
        s=8,
        cmap=SEQUENTIAL_CMAP,
    )
    axis.invert_yaxis()
    axis.set_title("Spatial uncertainty heatmap")
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")
    plt.colorbar(plot, ax=axis, fraction=0.046, label="uncertainty nm")


def _plot_hot_pixel_overlay(
    axis, density_total: np.ndarray, attempted_localizations: np.ndarray
) -> None:
    axis.imshow(density_total, cmap="gray", origin="upper")
    if _has_fields(attempted_localizations, {"x", "y", "hot_pixel_count"}):
        hot = attempted_localizations["hot_pixel_count"] > 0
        axis.scatter(
            attempted_localizations["x"][hot],
            attempted_localizations["y"][hot],
            s=8,
            color=PLOT_COLORS["vermillion"],
        )
    axis.set_title("Hot-pixel overlay")
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")


def _copy_if_exists(copies: list[tuple[Path, Path]]) -> list[Path]:
    paths: list[Path] = []
    for source, destination in copies:
        if source.is_file():
            shutil.copyfile(source, destination)
            paths.append(destination)
    return paths


def _write_static_interactive_map(
    path: Path, loc_source: np.ndarray, uncertainty: np.ndarray, hover_text: list[str]
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        path.write_text(
            "<html><body>Plotly unavailable</body></html>\n", encoding="utf-8"
        )
        return
    fig = go.Figure(
        data=[
            go.Scatter(
                x=loc_source["x"],
                y=loc_source["y"],
                mode="markers",
                marker={"size": 5, "color": uncertainty, "colorscale": "Cividis"},
                text=hover_text,
            )
        ]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title="Time-slider localization map", xaxis_title="x px", yaxis_title="y px"
    )
    fig.write_html(path)


def _summary_markdown(summary: QCDashboardSummary) -> str:
    lines = [
        "# PeakLoc Run QC Summary",
        "",
        f"- Input file: `{summary.input_file}`",
        f"- Events loaded: `{summary.event_count}`",
        f"- Attempted fits: `{summary.attempted_fit_count}`",
        f"- Accepted localizations: `{summary.accepted_localization_count}`",
        f"- Accepted from QC table: `{summary.accepted_from_qc_count}`",
        f"- Median uncertainty: `{_fmt(summary.median_uncertainty_nm)} nm`",
        f"- 90th percentile uncertainty: `{_fmt(summary.p90_uncertainty_nm)} nm`",
        "",
        "## Detection Funnel",
        "",
    ]
    lines.extend(
        f"- {key}: `{value}`" for key, value in summary.detection_funnel.items()
    )
    lines.extend(["", "## Rejection Reasons", ""])
    if summary.rejection_reasons:
        lines.extend(
            f"- {key}: `{value}`" for key, value in summary.rejection_reasons.items()
        )
    else:
        lines.append("No rejection reasons recorded.")
    lines.extend(["", "## Warnings", ""])
    if summary.warnings:
        lines.extend(f"- {warning}" for warning in summary.warnings)
    else:
        lines.append("No dashboard warnings.")
    return "\n".join(lines) + "\n"


def _index_html(
    summary: QCDashboardSummary,
    static_paths: list[Path],
    interactive_paths: list[Path],
    timestamp: str,
) -> str:
    links = "\n".join(
        f'<li><a href="{path.name}">{path.name}</a></li>'
        for path in static_paths + interactive_paths
        if path.suffix in {".png", ".html"}
    )
    warnings = "".join(f"<li>{warning}</li>" for warning in summary.warnings)
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>PeakLoc QC</title></head>
<body>
<h1>PeakLoc QC</h1>
<p>Run timestamp: <code>{timestamp}</code></p>
<p>Attempted fits: <code>{summary.attempted_fit_count}</code>; accepted:
<code>{summary.accepted_from_qc_count}</code></p>
<h2>Warnings</h2>
<ul>{warnings or "<li>No dashboard warnings.</li>"}</ul>
<h2>Artifacts</h2>
<ul>{links}</ul>
</body>
</html>
"""


def _json_ready(summary: QCDashboardSummary) -> dict[str, Any]:
    return summary.__dict__


def _bool_sum(array: np.ndarray, field_name: str) -> int:
    if array.size == 0 or field_name not in (array.dtype.names or ()):
        return 0
    return int(np.count_nonzero(array[field_name]))


def _rejection_counts(localization_qc: np.ndarray) -> dict[str, int]:
    if localization_qc.size == 0 or "primary_rejection_reason" not in (
        localization_qc.dtype.names or ()
    ):
        return {}
    return dict(
        Counter(str(reason) for reason in localization_qc["primary_rejection_reason"])
    )


def _uncertainty_from_qc_or_locs(
    localization_qc: np.ndarray, localizations: np.ndarray
) -> np.ndarray:
    uncertainty = _field(localization_qc, "uncertainty_px")
    if uncertainty.size:
        return uncertainty
    return _uncertainty_for_locs(localizations)


def _uncertainty_for_locs(localizations: np.ndarray) -> np.ndarray:
    if localizations.size == 0 or not _has_fields(
        localizations, {"sigma_x", "sigma_y", "cov_xy"}
    ):
        return np.asarray([], dtype=np.float64)
    return localization_uncertainty_px(localizations)


def _field(array: np.ndarray, field_name: str) -> np.ndarray:
    if array.size == 0 or field_name not in (array.dtype.names or ()):
        return np.asarray([], dtype=np.float64)
    return np.asarray(array[field_name], dtype=np.float64)


def _has_fields(array: np.ndarray, fields: set[str]) -> bool:
    return fields.issubset(array.dtype.names or ())


def _ids_or_range(array: np.ndarray) -> np.ndarray:
    if "id" in (array.dtype.names or ()):
        return array["id"]
    return np.arange(array.size)


def _reason_by_id(localization_qc: np.ndarray) -> dict[int, str]:
    if not _has_fields(localization_qc, {"id", "primary_rejection_reason"}):
        return {}
    return {
        int(row["id"]): str(row["primary_rejection_reason"]) for row in localization_qc
    }


def _sample_events(events: np.ndarray, max_events: int) -> np.ndarray:
    if events.size <= max_events:
        return events
    indices = np.linspace(0, events.size - 1, max_events, dtype=np.int64)
    return events[indices]


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def _empty_axis(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
