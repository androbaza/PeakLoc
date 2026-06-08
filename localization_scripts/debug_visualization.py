from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar


DEBUG_MARKER = ".peakloc_debug_artifacts"
DEBUG_COLORS = {
    "positive_events": "#E69F00",
    "negative_events": "#56B4E9",
    "truth": "#0072B2",
    "matched_localization": "#009E73",
    "unmatched_localization": "#D55E00",
    "residual": "#CC79A7",
    "roi": "#F0E442",
    "text_box": "#000000",
}


@dataclass(frozen=True)
class TruthPoint:
    x_px: float
    y_px: float
    peak_us: int
    label: str
    n_pos: int | None = None
    n_neg: int | None = None


@dataclass(frozen=True)
class LocalizationMatch:
    truth_index: int
    localization_index: int | None
    spatial_error_px: float | None
    time_error_us: float | None
    passed_spatial: bool
    passed_time: bool
    passed_event_counts: bool | None


@dataclass(frozen=True)
class DebugVisualizationConfig:
    output_dir: Path
    scenario_name: str
    sensor_shape: tuple[int, int]
    optical_pixel_size_nm: float
    max_spatial_error_px: float
    max_abs_time_error_us: int
    min_events_per_polarity: int | None = None
    overwrite: bool = True
    save_png: bool = True
    save_svg: bool = False  # keep false by default
    save_pdf: bool = False  # keep false by default
    save_tiff: bool = True
    save_interactive_html: bool = True
    show_residual_vectors: bool = False
    max_events_for_interactive: int = 50_000
    static_dpi: int = 450


@dataclass(frozen=True)
class DebugArtifactResult:
    output_dir: Path
    summary_markdown: Path
    static_paths: tuple[Path, ...]
    tiff_paths: tuple[Path, ...]
    html_paths: tuple[Path, ...]
    match_table_path: Path


def save_synthetic_localization_debug_artifacts(
    *,
    events: np.ndarray,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    config: DebugVisualizationConfig,
    rois: np.ndarray | None = None,
    attempted_localizations: np.ndarray | None = None,
    test_status: str = "unknown",
) -> DebugArtifactResult:
    """Write fixed-name debug artifacts for one synthetic scenario."""
    return save_localization_qc_artifacts(
        events=events,
        localizations=localizations,
        rois=rois,
        output_dir=config.output_dir,
        sensor_shape=config.sensor_shape,
        optical_pixel_size_nm=config.optical_pixel_size_nm,
        scenario_name=config.scenario_name,
        truth=truth,
        attempted_localizations=attempted_localizations,
        test_status=test_status,
        config=config,
    )


def save_localization_qc_artifacts(
    *,
    events: np.ndarray | None,
    localizations: np.ndarray,
    rois: np.ndarray | None,
    output_dir: Path,
    sensor_shape: tuple[int, int],
    optical_pixel_size_nm: float,
    scenario_name: str,
    truth: Sequence[TruthPoint] | None = None,
    attempted_localizations: np.ndarray | None = None,
    test_status: str = "unknown",
    config: DebugVisualizationConfig | None = None,
) -> DebugArtifactResult:
    if config is None:
        config = DebugVisualizationConfig(
            output_dir=output_dir,
            scenario_name=scenario_name,
            sensor_shape=sensor_shape,
            optical_pixel_size_nm=optical_pixel_size_nm,
            max_spatial_error_px=np.inf,
            max_abs_time_error_us=np.iinfo(np.int64).max,
        )
    prepare_debug_output_dir(config.output_dir, overwrite=config.overwrite)

    truth_points = list(truth or [])
    matches = match_truth_to_localizations(
        truth_points,
        localizations,
        max_spatial_error_px=config.max_spatial_error_px,
        max_abs_time_error_us=config.max_abs_time_error_us,
        min_events_per_polarity=config.min_events_per_polarity,
    )
    metrics = _summary_metrics(
        events=events,
        localizations=localizations,
        rois=rois,
        attempted_localizations=attempted_localizations,
        truth=truth_points,
        matches=matches,
        config=config,
    )
    status = _computed_status(test_status, matches)
    csv_path, json_path = write_match_tables(
        config.output_dir,
        truth_points,
        localizations,
        matches,
    )

    static_paths: list[Path] = []
    tiff_paths: list[Path] = []
    html_paths: list[Path] = []

    if events is not None:
        density_images = events_to_polarity_density_images(events, sensor_shape)
    else:
        empty = np.zeros(sensor_shape, dtype=np.float32)
        density_images = (empty, empty, empty)

    static_paths.extend(
        save_xy_summary_figure(
            events=events,
            density_images=density_images,
            localizations=localizations,
            attempted_localizations=attempted_localizations,
            truth=truth_points,
            matches=matches,
            rois=rois,
            metrics=metrics,
            status=status,
            config=config,
        )
    )
    static_paths.extend(
        save_polarity_density_figure(
            density_images=density_images,
            localizations=localizations,
            attempted_localizations=attempted_localizations,
            truth=truth_points,
            matches=matches,
            config=config,
        )
    )
    static_paths.extend(
        save_time_projection_figure(
            events=events,
            localizations=localizations,
            truth=truth_points,
            matches=matches,
            config=config,
        )
    )
    static_paths.extend(
        save_roi_montage_figure(
            rois=rois,
            localizations=localizations,
            truth=truth_points,
            matches=matches,
            config=config,
        )
    )
    static_paths.extend(
        save_qc_metrics_figure(
            localizations=localizations,
            truth=truth_points,
            matches=matches,
            config=config,
        )
    )

    if config.save_tiff:
        tiff_paths.extend(
            save_density_tiffs(
                density_images,
                config.output_dir,
                optical_pixel_size_nm=config.optical_pixel_size_nm,
            )
        )
    if config.save_interactive_html and events is not None:
        html_paths.append(
            save_interactive_spacetime_point_cloud(
                events=events,
                localizations=localizations,
                truth=truth_points,
                matches=matches,
                config=config,
            )
        )

    report_path = write_debug_report(
        config.output_dir,
        metrics=metrics,
        status=status,
        csv_path=csv_path,
        json_path=json_path,
        static_paths=static_paths,
        tiff_paths=tiff_paths,
        html_paths=html_paths,
    )
    return DebugArtifactResult(
        output_dir=config.output_dir,
        summary_markdown=report_path,
        static_paths=tuple(static_paths),
        tiff_paths=tuple(tiff_paths),
        html_paths=tuple(html_paths),
        match_table_path=csv_path,
    )


def prepare_debug_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Debug output path is not a directory: {output_dir}")
    marker_path = output_dir / DEBUG_MARKER
    if output_dir.is_dir() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"Debug output directory is not empty: {output_dir}")
        if not marker_path.is_file():
            raise ValueError(
                f"Refusing to overwrite unmarked debug output directory: {output_dir}"
            )
        for child in output_dir.iterdir():
            if child.name == DEBUG_MARKER:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path.touch()


def truth_points_from_blinks(blinks: Sequence[object]) -> list[TruthPoint]:
    return [
        TruthPoint(
            x_px=float(getattr(blink, "x_px")),
            y_px=float(getattr(blink, "y_px")),
            peak_us=int(getattr(blink, "peak_us")),
            label=f"truth_{idx}",
            n_pos=_optional_int(getattr(blink, "n_pos", None)),
            n_neg=_optional_int(getattr(blink, "n_neg", None)),
        )
        for idx, blink in enumerate(blinks)
    ]


def match_truth_to_localizations(
    truth: Sequence[TruthPoint],
    localizations: np.ndarray,
    *,
    max_spatial_error_px: float,
    max_abs_time_error_us: int,
    min_events_per_polarity: int | None = None,
) -> list[LocalizationMatch]:
    if not truth:
        return []
    if localizations.size == 0 or localizations.dtype.names is None:
        return [
            LocalizationMatch(idx, None, None, None, False, False, None)
            for idx, _ in enumerate(truth)
        ]
    required = {"x", "y", "t_peak"}
    if not required.issubset(localizations.dtype.names):
        raise ValueError("Localization array must contain x, y, and t_peak fields")

    candidates: list[tuple[float, float, int, int, float, float]] = []
    for truth_idx, truth_point in enumerate(truth):
        for loc_idx, loc in enumerate(localizations):
            spatial = float(
                np.hypot(
                    float(loc["x"]) - truth_point.x_px,
                    float(loc["y"]) - truth_point.y_px,
                )
            )
            time_error = float(float(loc["t_peak"]) - truth_point.peak_us)
            candidates.append(
                (
                    _candidate_cost(spatial, time_error, max_abs_time_error_us),
                    spatial,
                    truth_idx,
                    loc_idx,
                    spatial,
                    time_error,
                )
            )
    candidates.sort()

    assigned_truth: set[int] = set()
    assigned_locs: set[int] = set()
    by_truth: dict[int, LocalizationMatch] = {}
    for _, _, truth_idx, loc_idx, spatial, time_error in candidates:
        if truth_idx in assigned_truth or loc_idx in assigned_locs:
            continue
        passed_spatial = spatial <= max_spatial_error_px
        passed_time = abs(time_error) <= max_abs_time_error_us
        passed_event_counts = _passes_event_counts(
            localizations[loc_idx],
            min_events_per_polarity,
        )
        by_truth[truth_idx] = LocalizationMatch(
            truth_index=truth_idx,
            localization_index=loc_idx,
            spatial_error_px=spatial,
            time_error_us=time_error,
            passed_spatial=passed_spatial,
            passed_time=passed_time,
            passed_event_counts=passed_event_counts,
        )
        assigned_truth.add(truth_idx)
        assigned_locs.add(loc_idx)

    return [
        by_truth.get(idx, LocalizationMatch(idx, None, None, None, False, False, None))
        for idx, _ in enumerate(truth)
    ]


def events_to_polarity_density_images(
    events: np.ndarray,
    sensor_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = sensor_shape
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros((height, width), dtype=np.float32)
    if events.size == 0:
        return positive, negative, positive + negative
    _require_event_fields(events)
    xs = np.asarray(events["x"], dtype=np.int64)
    ys = np.asarray(events["y"], dtype=np.int64)
    valid = (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
    pos_mask = valid & (events["p"] == 1)
    neg_mask = valid & ~pos_mask
    np.add.at(positive, (ys[pos_mask], xs[pos_mask]), 1.0)
    np.add.at(negative, (ys[neg_mask], xs[neg_mask]), 1.0)
    return positive, negative, positive + negative


def localizations_to_xy(localizations: np.ndarray) -> np.ndarray:
    if localizations.size == 0 or localizations.dtype.names is None:
        return np.empty((0, 2), dtype=np.float64)
    if not {"x", "y"}.issubset(localizations.dtype.names):
        raise ValueError("Localization array must contain x and y fields")
    xy = np.column_stack((localizations["x"], localizations["y"])).astype(np.float64)
    return xy[np.isfinite(xy).all(axis=1)]


def save_figure_bundle(
    fig,
    output_stem: Path,
    *,
    dpi: int,
    config: DebugVisualizationConfig | None = None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    extensions = []
    if config is None or config.save_png:
        extensions.append("png")
    if config is None or config.save_svg:
        extensions.append("svg")
    if config is None or config.save_pdf:
        extensions.append("pdf")
    for extension in extensions:
        output_path = output_stem.with_suffix(f".{extension}")
        save_kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)
        paths.append(output_path)
    plt.close(fig)
    return tuple(paths)


def save_density_tiffs(
    density_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_dir: Path,
    *,
    optical_pixel_size_nm: float,
) -> tuple[Path, ...]:
    positive, negative, total = density_images
    outputs = [
        ("positive_event_density.tiff", positive),
        ("negative_event_density.tiff", negative),
        ("total_event_density.tiff", total),
    ]
    paths = []
    for filename, image in outputs:
        output_path = output_dir / filename
        tifffile.imwrite(
            output_path,
            np.asarray(image, dtype=np.float32),
            photometric="minisblack",
            metadata={
                "axes": "YX",
                "PhysicalSizeX": optical_pixel_size_nm,
                "PhysicalSizeY": optical_pixel_size_nm,
                "PhysicalSizeXUnit": "nm",
                "PhysicalSizeYUnit": "nm",
            },
        )
        paths.append(output_path)
    return tuple(paths)


def save_xy_summary_figure(
    *,
    events: np.ndarray | None,
    density_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    rois: np.ndarray | None,
    metrics: dict[str, object],
    status: str,
    config: DebugVisualizationConfig,
    attempted_localizations: np.ndarray | None = None,
) -> tuple[Path, ...]:
    del events
    total_density = density_images[2]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(np.log1p(total_density), origin="upper", cmap="magma")
    _draw_rois(ax, rois)
    _draw_attempted_localizations(ax, attempted_localizations, localizations)
    _draw_truth_and_localizations(ax, truth, localizations, matches)
    _format_sensor_axes(ax, config.sensor_shape)
    ax.set_title(f"{config.scenario_name} - {status}", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        _metrics_text(metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=8.5,
        bbox={
            "facecolor": DEBUG_COLORS["text_box"],
            "alpha": 0.72,
            "edgecolor": "none",
            "boxstyle": "square,pad=0.35",
        },
    )
    if config.optical_pixel_size_nm > 0:
        ax.add_artist(
            ScaleBar(
                config.optical_pixel_size_nm,
                units="nm",
                location="lower right",
                frameon=False,
                color="white",
                box_alpha=0.0,
            )
        )
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    return save_figure_bundle(
        fig,
        config.output_dir / "01_xy_detection_summary",
        dpi=config.static_dpi,
        config=config,
    )


def save_polarity_density_figure(
    *,
    density_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
    attempted_localizations: np.ndarray | None = None,
) -> tuple[Path, ...]:
    positive, negative, total = density_images
    signed = positive - negative
    panels = [
        ("Positive events", positive, "magma", None),
        ("Negative events", negative, "viridis", None),
        ("Signed pos - neg", signed, "coolwarm", _signed_limits(signed)),
        ("Total events", total, "magma", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), constrained_layout=True)
    for ax, (title, image, cmap, limits) in zip(axes.flat, panels, strict=True):
        kwargs = {}
        if limits is not None:
            kwargs.update({"vmin": limits[0], "vmax": limits[1]})
        ax.imshow(image, origin="upper", cmap=cmap, **kwargs)
        _draw_attempted_localizations(ax, attempted_localizations, localizations)
        _draw_truth_and_localizations(ax, truth, localizations, matches, legend=False)
        _format_sensor_axes(ax, config.sensor_shape)
        ax.set_title(title)
    return save_figure_bundle(
        fig,
        config.output_dir / "02_polarity_density",
        dpi=config.static_dpi,
        config=config,
    )


def save_time_projection_figure(
    *,
    events: np.ndarray | None,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
) -> tuple[Path, ...]:
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), constrained_layout=True)
    if events is not None and events.size:
        sampled = _downsample_events(
            events, min(config.max_events_for_interactive, 20_000)
        )
        pos = sampled["p"] == 1
        time_ms = sampled["t"].astype(np.float64) / 1_000.0
        axes[0].scatter(
            time_ms[pos],
            sampled["x"][pos],
            s=1.5,
            color=DEBUG_COLORS["positive_events"],
            alpha=0.35,
            lw=0,
            label="positive events",
        )
        axes[0].scatter(
            time_ms[~pos],
            sampled["x"][~pos],
            s=1.5,
            color=DEBUG_COLORS["negative_events"],
            alpha=0.35,
            lw=0,
            label="negative events",
        )
        axes[1].scatter(
            time_ms[pos],
            sampled["y"][pos],
            s=1.5,
            color=DEBUG_COLORS["positive_events"],
            alpha=0.35,
            lw=0,
        )
        axes[1].scatter(
            time_ms[~pos],
            sampled["y"][~pos],
            s=1.5,
            color=DEBUG_COLORS["negative_events"],
            alpha=0.35,
            lw=0,
        )
        bins = min(120, max(20, int(np.sqrt(events.size))))
        axes[2].hist(
            events["t"][events["p"] == 1].astype(np.float64) / 1_000.0,
            bins=bins,
            histtype="step",
            color=DEBUG_COLORS["positive_events"],
            label="positive event rate",
        )
        axes[2].hist(
            events["t"][events["p"] == 0].astype(np.float64) / 1_000.0,
            bins=bins,
            histtype="step",
            color=DEBUG_COLORS["negative_events"],
            label="negative event rate",
        )

    for point in truth:
        for ax in axes:
            ax.axvline(
                point.peak_us / 1_000.0,
                color=DEBUG_COLORS["truth"],
                lw=1.0,
                alpha=0.8,
            )
    if (
        localizations.size
        and localizations.dtype.names is not None
        and "t_peak" in localizations.dtype.names
    ):
        for t_peak in localizations["t_peak"]:
            axes[2].axvline(
                float(t_peak) / 1_000.0,
                color=DEBUG_COLORS["matched_localization"],
                lw=0.8,
                alpha=0.7,
            )
    axes[0].set_ylabel("x [sensor px]")
    axes[1].set_ylabel("y [sensor px]")
    axes[0].set_xlabel("time [ms]")
    axes[1].set_xlabel("time [ms]")
    axes[2].set_xlabel("time [ms]")
    axes[2].set_ylabel("event count / bin")
    _deduplicate_legend(axes[0])
    _deduplicate_legend(axes[2])
    return save_figure_bundle(
        fig,
        config.output_dir / "03_time_projection",
        dpi=config.static_dpi,
        config=config,
    )


def save_roi_montage_figure(
    *,
    rois: np.ndarray | None,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
) -> tuple[Path, ...]:
    matched_indices = [
        match.localization_index
        for match in matches
        if match.localization_index is not None
    ][:8]
    if rois is None or rois.size == 0:
        matched_indices = []
    n_rows = max(len(matched_indices), 1)
    fig, axes = plt.subplots(n_rows, 3, figsize=(8.5, 2.6 * n_rows), squeeze=False)
    if not matched_indices:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No matched ROI data", ha="center", va="center")
    for row, loc_idx in enumerate(matched_indices):
        loc = localizations[loc_idx]
        roi = _nearest_roi(loc, rois)
        images = [
            ("Positive ROI", roi["roi"] if roi is not None else np.zeros((1, 1))),
            ("Negative ROI", roi["roi_n"] if roi is not None else np.zeros((1, 1))),
            (
                "Combined ROI",
                (
                    roi["roi"].astype(np.float64) + roi["roi_n"].astype(np.float64)
                    if roi is not None
                    else np.zeros((1, 1))
                ),
            ),
        ]
        for col, (title, image) in enumerate(images):
            ax = axes[row, col]
            ax.imshow(image, origin="upper", cmap="magma")
            _draw_roi_local_markers(ax, loc, roi, truth, matches, loc_idx)
            ax.set_title(_roi_panel_title(title, loc), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    return save_figure_bundle(
        fig,
        config.output_dir / "04_roi_fit_montage",
        dpi=config.static_dpi,
        config=config,
    )


def save_qc_metrics_figure(
    *,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
) -> tuple[Path, ...]:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True)
    indices = np.arange(len(matches))
    spatial = np.asarray(
        [
            np.nan if match.spatial_error_px is None else match.spatial_error_px
            for match in matches
        ],
        dtype=np.float64,
    )
    time = np.asarray(
        [
            np.nan if match.time_error_us is None else match.time_error_us
            for match in matches
        ],
        dtype=np.float64,
    )
    axes[0, 0].bar(indices, spatial, color=DEBUG_COLORS["matched_localization"])
    axes[0, 0].axhline(config.max_spatial_error_px, color="black", lw=1, ls="--")
    axes[0, 0].set_title("Spatial error")
    axes[0, 0].set_ylabel("px")
    axes[0, 1].bar(indices, np.abs(time), color=DEBUG_COLORS["residual"])
    axes[0, 1].axhline(config.max_abs_time_error_us, color="black", lw=1, ls="--")
    axes[0, 1].set_title("Absolute time error")
    axes[0, 1].set_ylabel("us")

    event_counts = _matched_event_counts(localizations, matches)
    if event_counts.size:
        axes[1, 0].bar(indices, event_counts[:, 0], label="positive")
        axes[1, 0].bar(
            indices, event_counts[:, 1], bottom=event_counts[:, 0], label="negative"
        )
        if config.min_events_per_polarity is not None:
            axes[1, 0].axhline(
                config.min_events_per_polarity, color="black", lw=1, ls="--"
            )
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Matched event counts")
    axes[1, 0].set_ylabel("events")

    uncertainty = _matched_uncertainty(localizations, matches)
    if uncertainty.size:
        axes[1, 1].bar(indices[: uncertainty.size], uncertainty, color="#999999")
    axes[1, 1].set_title("Localization uncertainty")
    axes[1, 1].set_ylabel("px")
    for ax in axes.flat:
        ax.set_xlabel("truth index")
        ax.set_xticks(indices)
        ax.set_xticklabels([point.label for point in truth], rotation=30, ha="right")
    return save_figure_bundle(
        fig,
        config.output_dir / "05_qc_metrics",
        dpi=config.static_dpi,
        config=config,
    )


def save_interactive_spacetime_point_cloud(
    *,
    events: np.ndarray,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
) -> Path:
    fig = build_interactive_spacetime_point_cloud_figure(
        events=events,
        localizations=localizations,
        truth=truth,
        matches=matches,
        config=config,
    )
    output_path = config.output_dir / "06_spacetime_point_cloud.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def build_interactive_spacetime_point_cloud_figure(
    *,
    events: np.ndarray,
    localizations: np.ndarray,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
):
    import plotly.graph_objects as go

    sampled = _downsample_events(events, config.max_events_for_interactive)
    fig = go.Figure()
    for polarity, name, color in [
        (1, "positive events", DEBUG_COLORS["positive_events"]),
        (0, "negative events", DEBUG_COLORS["negative_events"]),
    ]:
        mask = sampled["p"] == polarity
        fig.add_trace(
            go.Scatter3d(
                x=sampled["x"][mask],
                y=sampled["y"][mask],
                z=sampled["t"][mask].astype(np.float64) / 1_000.0,
                mode="markers",
                name=name,
                marker={"size": 2, "color": color, "opacity": 0.28},
                text=[
                    f"p={polarity}<br>t_us={int(time)}" for time in sampled["t"][mask]
                ],
                hoverinfo="text+x+y+z",
            )
        )
    if truth:
        fig.add_trace(
            go.Scatter3d(
                x=[point.x_px for point in truth],
                y=[point.y_px for point in truth],
                z=[point.peak_us / 1_000.0 for point in truth],
                mode="markers+text",
                name="truth",
                marker={
                    "size": 6,
                    "color": DEBUG_COLORS["truth"],
                    "symbol": "circle-open",
                },
                text=[point.label for point in truth],
                hovertext=[
                    f"{point.label}<br>n_pos={point.n_pos}<br>n_neg={point.n_neg}"
                    for point in truth
                ],
            )
        )
    matched_indices = {
        match.localization_index
        for match in matches
        if match.localization_index is not None
    }
    if localizations.size and localizations.dtype.names is not None:
        matched_mask = np.asarray(
            [idx in matched_indices for idx in range(localizations.size)],
            dtype=bool,
        )
        for mask, name, color in [
            (
                matched_mask,
                "matched localizations",
                DEBUG_COLORS["matched_localization"],
            ),
            (
                ~matched_mask,
                "unmatched localizations",
                DEBUG_COLORS["unmatched_localization"],
            ),
        ]:
            locs = localizations[mask]
            if locs.size == 0:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=locs["x"],
                    y=locs["y"],
                    z=locs["t_peak"].astype(np.float64) / 1_000.0,
                    mode="markers",
                    name=name,
                    marker={"size": 5, "color": color},
                    hovertext=_localization_hover_text(locs),
                )
            )
    if config.show_residual_vectors:
        for match in matches:
            if match.localization_index is None:
                continue
            loc = localizations[match.localization_index]
            point = truth[match.truth_index]
            fig.add_trace(
                go.Scatter3d(
                    x=[point.x_px, float(loc["x"])],
                    y=[point.y_px, float(loc["y"])],
                    z=[point.peak_us / 1_000.0, float(loc["t_peak"]) / 1_000.0],
                    mode="lines",
                    name="truth-localization error vector",
                    line={"color": DEBUG_COLORS["residual"], "width": 2},
                    visible="legendonly",
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        title=config.scenario_name,
        scene={
            "xaxis_title": "x [sensor px]",
            "yaxis_title": "y [sensor px]",
            "zaxis_title": "time [ms]",
            "yaxis": {"autorange": "reversed"},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def write_debug_report(
    output_dir: Path,
    *,
    metrics: dict[str, object],
    status: str,
    csv_path: Path,
    json_path: Path,
    static_paths: Sequence[Path],
    tiff_paths: Sequence[Path],
    html_paths: Sequence[Path],
) -> Path:
    report_path = output_dir / "debug_report.md"
    lines = [
        f"# PeakLoc Debug Report: {metrics['scenario_name']}",
        "",
        f"- status: {status}",
        f"- matched: {metrics['matched_count']} / {metrics['expected_count']}",
        f"- max_spatial_error_px: {metrics['max_spatial_error_px']}",
        f"- max_time_error_us: {metrics['max_time_error_us']}",
        f"- total_events: {metrics['total_events']}",
        f"- roi_count: {metrics['roi_count']}",
        f"- localization_count: {metrics['localization_count']}",
        f"- attempted_localization_count: {metrics['attempted_localization_count']}",
        "",
        "## Tables",
        f"- {csv_path.name}",
        f"- {json_path.name}",
        "",
        "## Figures",
    ]
    lines.extend(f"- {path.name}" for path in static_paths)
    if tiff_paths:
        lines.append("")
        lines.append("## TIFFs")
        lines.extend(f"- {path.name}" for path in tiff_paths)
    if html_paths:
        lines.append("")
        lines.append("## Interactive")
        lines.extend(f"- {path.name}" for path in html_paths)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_match_tables(
    output_dir: Path,
    truth: Sequence[TruthPoint],
    localizations: np.ndarray,
    matches: Sequence[LocalizationMatch],
) -> tuple[Path, Path]:
    rows = [
        _match_row(match, truth[match.truth_index], localizations) for match in matches
    ]
    csv_path = output_dir / "matches.csv"
    json_path = output_dir / "matches.json"
    fieldnames = list(rows[0]) if rows else _empty_match_fields()
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _draw_truth_and_localizations(
    ax,
    truth: Sequence[TruthPoint],
    localizations: np.ndarray,
    matches: Sequence[LocalizationMatch],
    *,
    legend: bool = True,
) -> None:
    del legend
    if truth:
        ax.scatter(
            [point.x_px for point in truth],
            [point.y_px for point in truth],
            s=80,
            marker="o",
            facecolors="none",
            edgecolors=DEBUG_COLORS["truth"],
            linewidths=1.8,
            label="truth",
        )
    xy = localizations_to_xy(localizations)
    matched_indices = {
        match.localization_index
        for match in matches
        if match.localization_index is not None
    }
    if xy.size:
        matched_mask = np.asarray(
            [idx in matched_indices for idx in range(localizations.size)],
            dtype=bool,
        )
        if np.any(matched_mask):
            ax.scatter(
                xy[matched_mask, 0],
                xy[matched_mask, 1],
                s=38,
                marker="x",
                color=DEBUG_COLORS["matched_localization"],
                label="matched localization",
            )
        if np.any(~matched_mask):
            ax.scatter(
                xy[~matched_mask, 0],
                xy[~matched_mask, 1],
                s=30,
                marker="^",
                color=DEBUG_COLORS["unmatched_localization"],
                label="unmatched localization",
            )
    for match in matches:
        if match.localization_index is None:
            continue
        point = truth[match.truth_index]
        loc = localizations[match.localization_index]
        ax.plot(
            [point.x_px, float(loc["x"])],
            [point.y_px, float(loc["y"])],
            color=DEBUG_COLORS["residual"],
            lw=1.1,
            label="residual",
        )
    _deduplicate_legend(ax)


def _draw_attempted_localizations(
    ax,
    attempted: np.ndarray | None,
    accepted: np.ndarray,
) -> None:
    if attempted is None or attempted.size == 0:
        return
    if attempted.dtype.names is None or not {"x", "y"}.issubset(attempted.dtype.names):
        return
    if accepted.dtype.names is None or not {"x", "y"}.issubset(accepted.dtype.names):
        accepted_xy: set[tuple[float, float]] = set()
    else:
        accepted_xy = {
            (round(float(row["x"]), 6), round(float(row["y"]), 6)) for row in accepted
        }
    attempted_xy = np.column_stack((attempted["x"], attempted["y"])).astype(np.float64)
    finite = np.isfinite(attempted_xy).all(axis=1)

    failed = np.zeros(attempted.size, dtype=bool)
    if "fit_success" in attempted.dtype.names:
        failed = ~attempted["fit_success"].astype(bool)
    failed &= finite

    rejected = np.zeros(attempted.size, dtype=bool)
    for idx, row in enumerate(attempted):
        key = (round(float(row["x"]), 6), round(float(row["y"]), 6))
        rejected[idx] = finite[idx] and key not in accepted_xy and not failed[idx]

    if np.any(failed):
        ax.scatter(
            attempted_xy[failed, 0],
            attempted_xy[failed, 1],
            marker="^",
            s=58,
            facecolors="none",
            edgecolors=DEBUG_COLORS["unmatched_localization"],
            linewidths=1.1,
            label="failed fit attempt",
        )
    if np.any(rejected):
        ax.scatter(
            attempted_xy[rejected, 0],
            attempted_xy[rejected, 1],
            marker="s",
            s=48,
            facecolors="none",
            edgecolors=DEBUG_COLORS["residual"],
            linewidths=1.1,
            label="filtered/rejected attempt",
        )


def _draw_rois(ax, rois: np.ndarray | None) -> None:
    if rois is None or rois.size == 0 or rois.dtype.names is None:
        return
    if not {"roi_x0", "roi_y0", "roi"}.issubset(rois.dtype.names):
        return
    for roi in rois[:80]:
        height, width = roi["roi"].shape
        ax.add_patch(
            Rectangle(
                (float(roi["roi_x0"]) - 0.5, float(roi["roi_y0"]) - 0.5),
                width,
                height,
                fill=False,
                edgecolor=DEBUG_COLORS["roi"],
                linewidth=0.55,
                alpha=0.42,
            )
        )


def _format_sensor_axes(ax, sensor_shape: tuple[int, int]) -> None:
    height, width = sensor_shape
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [sensor px]")
    ax.set_ylabel("y [sensor px]")


def _summary_metrics(
    *,
    events: np.ndarray | None,
    localizations: np.ndarray,
    rois: np.ndarray | None,
    attempted_localizations: np.ndarray | None,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    config: DebugVisualizationConfig,
) -> dict[str, object]:
    spatial_errors = [
        match.spatial_error_px
        for match in matches
        if match.spatial_error_px is not None
    ]
    time_errors = [
        abs(match.time_error_us) for match in matches if match.time_error_us is not None
    ]
    uncertainty = _matched_uncertainty(localizations, matches)
    max_spatial_error_px = _optional_float_max(spatial_errors)
    return {
        "scenario_name": config.scenario_name,
        "matched_count": sum(_match_passed(match) for match in matches),
        "expected_count": len(truth),
        "max_spatial_error_px": max_spatial_error_px,
        "max_spatial_error_nm": (
            max_spatial_error_px * config.optical_pixel_size_nm
            if max_spatial_error_px is not None
            else None
        ),
        "max_time_error_us": _optional_float_max(time_errors),
        "median_uncertainty_px": (
            float(np.nanmedian(uncertainty)) if uncertainty.size else None
        ),
        "total_events": 0 if events is None else int(events.size),
        "roi_count": 0 if rois is None else int(rois.size),
        "localization_count": int(localizations.size),
        "attempted_localization_count": (
            0 if attempted_localizations is None else int(attempted_localizations.size)
        ),
    }


def _metrics_text(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            f"scenario: {metrics['scenario_name']}",
            f"matched/expected: {metrics['matched_count']} / {metrics['expected_count']}",
            f"max spatial error: {_format_optional(metrics['max_spatial_error_px'])} px",
            f"max spatial error: {_format_optional(metrics['max_spatial_error_nm'])} nm",
            f"max time error: {_format_optional(metrics['max_time_error_us'])} us",
            f"median uncertainty: {_format_optional(metrics['median_uncertainty_px'])} px",
            f"events: {metrics['total_events']}",
            f"ROIs: {metrics['roi_count']}",
            f"localizations: {metrics['localization_count']}",
            f"attempted: {metrics['attempted_localization_count']}",
        ]
    )


def _computed_status(test_status: str, matches: Sequence[LocalizationMatch]) -> str:
    if test_status.upper().startswith("XFAIL"):
        return test_status
    if test_status == "pre_assertion":
        return "PASS" if all(_match_passed(match) for match in matches) else "FAIL"
    return test_status


def _match_passed(match: LocalizationMatch) -> bool:
    count_ok = True if match.passed_event_counts is None else match.passed_event_counts
    return (
        match.localization_index is not None
        and match.passed_spatial
        and match.passed_time
        and count_ok
    )


def _candidate_cost(
    spatial_error_px: float,
    time_error_us: float,
    max_abs_time_error_us: int,
) -> float:
    time_scale = max(float(max_abs_time_error_us), 1.0)
    return spatial_error_px + abs(time_error_us) / time_scale


def _passes_event_counts(
    loc: np.void,
    min_events_per_polarity: int | None,
) -> bool | None:
    if min_events_per_polarity is None:
        return None
    names = loc.dtype.names or ()
    if not {"E_total", "E_total_n"}.issubset(names):
        return None
    return (
        int(loc["E_total"]) >= min_events_per_polarity
        and int(loc["E_total_n"]) >= min_events_per_polarity
    )


def _match_row(
    match: LocalizationMatch,
    truth: TruthPoint,
    localizations: np.ndarray,
) -> dict[str, object]:
    row = {
        **asdict(match),
        "truth_label": truth.label,
        "truth_x_px": truth.x_px,
        "truth_y_px": truth.y_px,
        "truth_peak_us": truth.peak_us,
        "truth_n_pos": truth.n_pos,
        "truth_n_neg": truth.n_neg,
        "localization_x_px": None,
        "localization_y_px": None,
        "localization_t_peak_us": None,
        "localization_e_pos": None,
        "localization_e_neg": None,
        "fit_success": None,
        "fit_cond": None,
    }
    if match.localization_index is None:
        return row
    loc = localizations[match.localization_index]
    names = loc.dtype.names or ()
    row.update(
        {
            "localization_x_px": float(loc["x"]) if "x" in names else None,
            "localization_y_px": float(loc["y"]) if "y" in names else None,
            "localization_t_peak_us": float(loc["t_peak"])
            if "t_peak" in names
            else None,
            "localization_e_pos": int(loc["E_total"]) if "E_total" in names else None,
            "localization_e_neg": int(loc["E_total_n"])
            if "E_total_n" in names
            else None,
            "fit_success": bool(loc["fit_success"]) if "fit_success" in names else None,
            "fit_cond": float(loc["fit_cond"]) if "fit_cond" in names else None,
        }
    )
    return row


def _empty_match_fields() -> list[str]:
    return [
        "truth_index",
        "localization_index",
        "spatial_error_px",
        "time_error_us",
        "passed_spatial",
        "passed_time",
        "passed_event_counts",
        "truth_label",
        "truth_x_px",
        "truth_y_px",
        "truth_peak_us",
        "truth_n_pos",
        "truth_n_neg",
        "localization_x_px",
        "localization_y_px",
        "localization_t_peak_us",
        "localization_e_pos",
        "localization_e_neg",
        "fit_success",
        "fit_cond",
    ]


def _downsample_events(events: np.ndarray, max_events: int) -> np.ndarray:
    if events.size <= max_events:
        return events
    rng = np.random.default_rng(0)
    indices = rng.choice(events.size, size=max_events, replace=False)
    return events[np.sort(indices)]


def _require_event_fields(events: np.ndarray) -> None:
    if events.dtype.names is None or not {"x", "y", "p", "t"}.issubset(
        events.dtype.names
    ):
        raise ValueError("Event array must contain x, y, p, and t fields")


def _optional_int(value: int | None) -> int | None:
    return None if value is None else int(value)


def _optional_float_max(values: Sequence[float]) -> float | None:
    return float(np.max(values)) if values else None


def _format_optional(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _signed_limits(image: np.ndarray) -> tuple[float, float]:
    limit = float(np.max(np.abs(image))) if image.size else 1.0
    if limit == 0:
        limit = 1.0
    return -limit, limit


def _deduplicate_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def _matched_event_counts(
    localizations: np.ndarray,
    matches: Sequence[LocalizationMatch],
) -> np.ndarray:
    if localizations.dtype.names is None or not {"E_total", "E_total_n"}.issubset(
        localizations.dtype.names
    ):
        return np.empty((0, 2), dtype=np.float64)
    rows = []
    for match in matches:
        if match.localization_index is None:
            rows.append((np.nan, np.nan))
            continue
        loc = localizations[match.localization_index]
        rows.append((float(loc["E_total"]), float(loc["E_total_n"])))
    return np.asarray(rows, dtype=np.float64)


def _matched_uncertainty(
    localizations: np.ndarray,
    matches: Sequence[LocalizationMatch],
) -> np.ndarray:
    if localizations.dtype.names is None or not {
        "sigma_x",
        "sigma_y",
        "cov_xy",
    }.issubset(localizations.dtype.names):
        return np.empty(0, dtype=np.float64)
    values = []
    for match in matches:
        if match.localization_index is None:
            values.append(np.nan)
            continue
        loc = localizations[match.localization_index]
        trace = float(loc["sigma_x"]) ** 2 + float(loc["sigma_y"]) ** 2
        determinant_term = np.sqrt(
            (float(loc["sigma_x"]) ** 2 - float(loc["sigma_y"]) ** 2) ** 2
            + 4.0 * float(loc["cov_xy"]) ** 2
        )
        values.append(float(np.sqrt(max(0.5 * (trace + determinant_term), 0.0))))
    return np.asarray(values, dtype=np.float64)


def _nearest_roi(loc: np.void, rois: np.ndarray | None) -> np.void | None:
    if rois is None or rois.size == 0 or rois.dtype.names is None:
        return None
    if "t_peak" not in rois.dtype.names:
        return rois[0]
    idx = int(np.argmin(np.abs(rois["t_peak"].astype(float) - float(loc["t_peak"]))))
    return rois[idx]


def _draw_roi_local_markers(
    ax,
    loc: np.void,
    roi: np.void | None,
    truth: Sequence[TruthPoint],
    matches: Sequence[LocalizationMatch],
    loc_idx: int,
) -> None:
    names = loc.dtype.names or ()
    if {"sub_x", "sub_y"}.issubset(names):
        ax.scatter(float(loc["sub_x"]), float(loc["sub_y"]), marker="x", color="white")
    if (
        roi is None
        or roi.dtype.names is None
        or not {"roi_x0", "roi_y0"}.issubset(roi.dtype.names)
    ):
        return
    for match in matches:
        if match.localization_index != loc_idx:
            continue
        truth_point = truth[match.truth_index]
        ax.scatter(
            truth_point.x_px - float(roi["roi_x0"]),
            truth_point.y_px - float(roi["roi_y0"]),
            marker="o",
            facecolors="none",
            edgecolors=DEBUG_COLORS["truth"],
        )


def _roi_panel_title(title: str, loc: np.void) -> str:
    names = loc.dtype.names or ()
    parts = [title]
    if {"E_total", "E_total_n"}.issubset(names):
        parts.append(f"E+={int(loc['E_total'])} E-={int(loc['E_total_n'])}")
    if "fit_cond" in names:
        parts.append(f"cond={float(loc['fit_cond']):.2g}")
    if "nll_per_event" in names:
        parts.append(f"NLL/event={float(loc['nll_per_event']):.2g}")
    return "\n".join(parts)


def _localization_hover_text(localizations: np.ndarray) -> list[str]:
    names = set(localizations.dtype.names or ())
    text = []
    for loc in localizations:
        parts = [
            f"id={int(loc['id'])}" if "id" in names else "id=n/a",
            f"t_peak_us={float(loc['t_peak']):.1f}" if "t_peak" in names else "",
            f"E+={int(loc['E_total'])}" if "E_total" in names else "",
            f"E-={int(loc['E_total_n'])}" if "E_total_n" in names else "",
            f"fit_success={bool(loc['fit_success'])}" if "fit_success" in names else "",
        ]
        text.append("<br>".join(part for part in parts if part))
    return text
