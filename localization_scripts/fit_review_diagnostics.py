"""Diagnostic fit-review figures for identifying implausible PSF candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import colors
from matplotlib.patches import Ellipse, Rectangle

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.localization_fitting import localization_uncertainty_px
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import (
    PUBLICATION_DPI,
    PLOT_COLORS,
    save_publication_figure,
    style_publication_axis,
)


QUANTILE_FILENAME = "fit_uncertainty_quantile_montage.png"
HOT_PIXEL_FILENAME = "fit_hot_pixel_dominated_rois.png"
REPLAY_FILENAME = "roi_detection_decision_replay.png"
FAILED_FILENAME = "fit_failed_or_nonfinite_montage.png"


def save_fit_review_diagnostics(
    attempted_localizations: np.ndarray,
    accepted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    output_dir: Path,
    *,
    config: PeakLocConfig,
    n: int,
    dpi: int,
) -> list[Path]:
    """Save stratified fit diagnostics without silently changing fit acceptance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_dpi = max(dpi, PUBLICATION_DPI)
    quantile_indices = _quantile_indices(attempted_localizations, n=n)
    hot_indices = _hot_pixel_indices(attempted_localizations, n=n)
    failed_indices = _failed_indices(attempted_localizations, localization_qc, n=n)
    artifacts: list[Path] = []
    artifacts.extend(
        _save_fit_montage(
            attempted_localizations,
            quantile_indices,
            accepted_localizations,
            localization_qc,
            output_dir / QUANTILE_FILENAME,
            config=config,
            title="Fit uncertainty quantiles (Q0–2%, Q2–20%, Q20–80%, Q80–98%, Q98–100%)",
            dpi=effective_dpi,
        )
    )
    artifacts.extend(
        _save_fit_montage(
            attempted_localizations,
            hot_indices,
            accepted_localizations,
            localization_qc,
            output_dir / HOT_PIXEL_FILENAME,
            config=config,
            title="Candidates dominated by one positive pixel (≥80% of positive ROI events)",
            dpi=effective_dpi,
        )
    )
    artifacts.extend(
        _save_decision_replay(
            attempted_localizations,
            _replay_indices(quantile_indices, hot_indices),
            output_dir / REPLAY_FILENAME,
            dpi=effective_dpi,
        )
    )
    if failed_indices.size:
        artifacts.extend(
            _save_fit_montage(
                attempted_localizations,
                failed_indices,
                accepted_localizations,
                localization_qc,
                output_dir / FAILED_FILENAME,
                config=config,
                title="Failed or rejected fit examples",
                dpi=effective_dpi,
            )
        )
    return artifacts


def _quantile_indices(localizations: np.ndarray, *, n: int) -> np.ndarray:
    uncertainty = _uncertainty(localizations)
    finite = np.flatnonzero(np.isfinite(uncertainty))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64)
    ordered = finite[np.argsort(uncertainty[finite], kind="stable")]
    boundaries = np.asarray([0.0, 0.02, 0.20, 0.80, 0.98, 1.0])
    counts = _allocation(max(n, 1), len(boundaries) - 1)
    selected: list[int] = []
    for index, count in enumerate(counts):
        start = int(np.floor(boundaries[index] * ordered.size))
        stop = int(np.ceil(boundaries[index + 1] * ordered.size))
        chunk = ordered[start : max(stop, start + 1)]
        if index == len(counts) - 1:
            chunk = chunk[::-1]
        selected.extend(_evenly_spaced(chunk, count).tolist())
    return _unique_indices(selected)


def _hot_pixel_indices(localizations: np.ndarray, *, n: int) -> np.ndarray:
    shares = _hot_pixel_shares(localizations)
    finite = np.flatnonzero(np.isfinite(shares))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64)
    high = finite[shares[finite] >= 0.80]
    candidates = high if high.size else finite
    ordered = candidates[np.argsort(shares[candidates])[::-1]]
    return _evenly_spaced(ordered, max(n, 1))


def _failed_indices(
    localizations: np.ndarray, localization_qc: np.ndarray, *, n: int
) -> np.ndarray:
    if localizations.size == 0:
        return np.empty(0, dtype=np.int64)
    failed = ~np.isfinite(_uncertainty(localizations))
    names = localizations.dtype.names or ()
    if "fit_success" in names:
        failed |= ~np.asarray(localizations["fit_success"], dtype=bool)
    qc_by_id = _qc_by_id(localization_qc)
    if "id" in names:
        failed |= np.asarray(
            [
                not bool(qc_by_id.get(int(row_id), {}).get("accepted", True))
                for row_id in localizations["id"]
            ],
            dtype=bool,
        )
    return _evenly_spaced(np.flatnonzero(failed), max(n, 1))


def _save_fit_montage(
    localizations: np.ndarray,
    indices: np.ndarray,
    accepted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    path: Path,
    *,
    config: PeakLocConfig,
    title: str,
    dpi: int,
) -> list[Path]:
    columns = min(6, max(int(indices.size), 1))
    rows = max(1, int(np.ceil(max(int(indices.size), 1) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.05, rows * 2.55),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(title, fontsize=10)
    accepted_ids = _id_set(accepted_localizations)
    qc_by_id = _qc_by_id(localization_qc)
    for axis in axes.ravel():
        axis.set_axis_off()
    if indices.size == 0:
        axes.ravel()[0].text(
            0.5,
            0.5,
            "No candidates with the required review fields",
            ha="center",
            va="center",
            transform=axes.ravel()[0].transAxes,
        )
    for axis, index in zip(axes.ravel(), indices, strict=False):
        row = localizations[int(index)]
        _draw_fit_tile(
            axis,
            row,
            qc_by_id.get(_row_id(row)),
            _row_id(row) in accepted_ids,
            config,
        )
    paths = save_publication_figure(
        figure, path, dpi=dpi, save_vector=config.qc_save_vector
    )
    plt.close(figure)
    return paths


def _draw_fit_tile(
    axis,
    row: np.void,
    qc_row: dict[str, object] | None,
    accepted_by_output: bool,
    config: PeakLocConfig,
) -> None:
    image = _positive_roi(row)
    vmax = _display_vmax(image)
    norm = colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
    axis.imshow(image, cmap="magma", norm=norm, interpolation="none", origin="upper")
    height, width = image.shape
    sub_x = _float(row, "sub_x", (width - 1) / 2)
    sub_y = _float(row, "sub_y", (height - 1) / 2)
    axis.scatter(
        sub_x, sub_y, marker="x", s=24, linewidths=0.9, color="white", zorder=5
    )
    _draw_expected_psf(axis, sub_x, sub_y, config)
    _draw_covariance(axis, row, sub_x, sub_y, width, height)
    share, max_y, max_x = _hot_pixel_share(row)
    if np.isfinite(share) and share >= 0.80:
        axis.add_patch(
            Rectangle(
                (max_x - 0.5, max_y - 0.5),
                1.0,
                1.0,
                fill=False,
                linewidth=1.0,
                edgecolor=PLOT_COLORS["vermillion"],
            )
        )
    accepted = (
        bool(qc_row.get("accepted", accepted_by_output))
        if qc_row
        else accepted_by_output
    )
    reason = str(qc_row.get("reason", "accepted")) if qc_row else "accepted"
    state = "ACCEPTED" if accepted else f"REJECTED: {reason}"
    uncertainty = _uncertainty_one(row)
    state_color = PLOT_COLORS["green"] if accepted else PLOT_COLORS["vermillion"]
    axis.set(
        title=(
            f"id {_row_id(row)} | {state}\n"
            f"σ={uncertainty:.3g} px ({uncertainty * config.optical_pixel_size_nm:.3g} nm)\n"
            f"E+={_integer(row, 'E_total', 0)} E−={_integer(row, 'E_total_n', 0)} | "
            f"max+/E+={share:.1%}\n"
            f"NLL/e={_float(row, 'nll_per_event', np.nan):.3g}; "
            f"cond={_float(row, 'fit_cond', np.nan):.3g}"
        ),
        xlim=(-0.5, width - 0.5),
        ylim=(height - 0.5, -0.5),
        xticks=[],
        yticks=[],
        aspect="equal",
    )
    axis.title.set_color(state_color)
    axis.title.set_fontsize(5.8)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor(state_color)


def _save_decision_replay(
    localizations: np.ndarray, indices: np.ndarray, path: Path, *, dpi: int
) -> list[Path]:
    indices = indices[:12]
    columns = min(4, max(int(indices.size), 1))
    tile_rows = max(1, int(np.ceil(max(int(indices.size), 1) / columns)))
    figure = plt.figure(
        figsize=(columns * 2.8, tile_rows * 3.25), constrained_layout=True
    )
    grid = figure.add_gridspec(
        tile_rows * 2, columns, height_ratios=[2.1, 1.0] * tile_rows
    )
    figure.suptitle(
        "ROI decision replay: detector seed (×), fitted center (+), and selected event windows",
        fontsize=10,
    )
    if indices.size == 0:
        axis = figure.add_subplot(grid[:, :])
        axis.text(
            0.5,
            0.5,
            "No ROI candidates available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    for tile, index in enumerate(indices):
        row_index, column_index = divmod(tile, columns)
        image_axis = figure.add_subplot(grid[row_index * 2, column_index])
        trace_axis = figure.add_subplot(grid[row_index * 2 + 1, column_index])
        _draw_replay_tile(image_axis, trace_axis, localizations[int(index)])
    paths = save_publication_figure(figure, path, dpi=dpi, save_vector=True)
    plt.close(figure)
    return paths


def _draw_replay_tile(image_axis, trace_axis, row: np.void) -> None:
    image = _composite_roi(row)
    image_axis.imshow(image, interpolation="none", origin="upper")
    height, width = image.shape[:2]
    sub_x = _float(row, "sub_x", (width - 1) / 2)
    sub_y = _float(row, "sub_y", (height - 1) / 2)
    seed = _detector_seed_in_roi(row, width, height)
    if seed is not None:
        image_axis.scatter(
            *seed,
            marker="x",
            s=24,
            linewidths=0.9,
            color=PLOT_COLORS["orange"],
            zorder=5,
        )
    image_axis.scatter(
        sub_x, sub_y, marker="+", s=30, linewidths=1.0, color="white", zorder=5
    )
    seed_label = "seed ×" if seed is not None else "seed unavailable"
    image_axis.set(
        title=f"id {_row_id(row)} | {seed_label}",
        xlim=(-0.5, width - 0.5),
        ylim=(height - 0.5, -0.5),
        xticks=[],
        yticks=[],
        aspect="equal",
    )
    _draw_event_trace(trace_axis, row)


def _draw_event_trace(axis, row: np.void) -> None:
    trace = _cumulative_trace(row)
    anchor = _float(row, "parent_seed_peak_us", _float(row, "t_peak", 0.0))
    if anchor <= 0:
        anchor = _float(row, "t_peak", 0.0)
    if trace is None:
        axis.text(
            0.5,
            0.55,
            "Cumulative selected-event trace\nunavailable in this legacy ROI",
            ha="center",
            va="center",
            fontsize=5.5,
            transform=axis.transAxes,
        )
        axis.set(
            xlabel="Time from detector peak (ms)",
            ylabel="Cumulative events",
            xlim=(-1000, 1000),
        )
        style_publication_axis(axis)
        return
    times, positive, negative = trace
    time_ms = (times - anchor) * 1e-3
    axis.step(
        time_ms,
        positive,
        where="post",
        color=PLOT_COLORS["orange"],
        linewidth=0.8,
        label="positive",
    )
    axis.step(
        time_ms,
        negative,
        where="post",
        color=PLOT_COLORS["sky_blue"],
        linewidth=0.8,
        label="negative",
    )
    for index, (start, stop, color) in enumerate(_selected_windows(row, anchor)):
        start_ms = (start - anchor) * 1e-3
        stop_ms = (stop - anchor) * 1e-3
        y = 0.08 + index * 0.10
        axis.hlines(
            y,
            start_ms,
            stop_ms,
            transform=axis.get_xaxis_transform(),
            color=color,
            linewidth=1.0,
        )
    axis.axvline(0.0, color=PLOT_COLORS["black"], linewidth=0.7)
    lower = max(float(np.min(time_ms)), -1000.0)
    upper = min(float(np.max(time_ms)), 1000.0)
    if lower >= upper:
        lower, upper = -1.0, 1.0
    axis.set(
        xlabel="Time from detector peak (ms)",
        ylabel="Cumulative events",
        xlim=(lower, upper),
    )
    axis.legend(frameon=False, fontsize=5, loc="upper left")
    style_publication_axis(axis)


def _positive_roi(row: np.void) -> np.ndarray:
    names = row.dtype.names or ()
    if "roi" not in names:
        return np.zeros((3, 3), dtype=np.float64)
    return np.asarray(row["roi"], dtype=np.float64)


def _composite_roi(row: np.void) -> np.ndarray:
    positive = _normalize(_positive_roi(row))
    names = row.dtype.names or ()
    negative = (
        _normalize(np.asarray(row["roi_n"], dtype=np.float64))
        if "roi_n" in names
        else np.zeros_like(positive)
    )
    return np.stack((positive, 0.25 * positive, negative), axis=-1)


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image) & (image > 0)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float64)
    upper = float(np.percentile(finite, 99.5))
    return np.clip(image / max(upper, 1.0), 0.0, 1.0)


def _display_vmax(image: np.ndarray) -> float:
    finite = image[np.isfinite(image) & (image > 0)]
    return max(float(np.percentile(finite, 99.5)) if finite.size else 1.0, 1.0)


def _draw_expected_psf(axis, x: float, y: float, config: PeakLocConfig) -> None:
    sigma = config.sigma_psf_px or config.dataset_fwhm / 2.354820045
    axis.add_patch(
        Ellipse(
            (x, y), 2 * sigma, 2 * sigma, fill=False, edgecolor="white", linewidth=0.65
        )
    )


def _draw_covariance(
    axis, row: np.void, x: float, y: float, width: int, height: int
) -> None:
    sigma_x = _float(row, "sigma_x", np.nan)
    sigma_y = _float(row, "sigma_y", np.nan)
    covariance_xy = _float(row, "cov_xy", np.nan)
    covariance = np.asarray([[sigma_x**2, covariance_xy], [covariance_xy, sigma_y**2]])
    if not np.all(np.isfinite(covariance)):
        return
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values < 0) or np.max(values) > max(width, height) ** 2:
        return
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    axis.add_patch(
        Ellipse(
            (x, y),
            2 * np.sqrt(values[0]),
            2 * np.sqrt(values[1]),
            angle=angle,
            fill=False,
            edgecolor=PLOT_COLORS["sky_blue"],
            linewidth=0.7,
        )
    )


def _hot_pixel_shares(localizations: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_hot_pixel_share(row)[0] for row in localizations], dtype=np.float64
    )


def _hot_pixel_share(row: np.void) -> tuple[float, int, int]:
    image = _positive_roi(row)
    if image.size == 0 or not np.any(np.isfinite(image)):
        return np.nan, 0, 0
    max_index = int(np.nanargmax(image))
    max_y, max_x = np.unravel_index(max_index, image.shape)
    denominator = _float(row, "E_total", 0.0)
    if denominator <= 0:
        denominator = float(np.nansum(image))
    if denominator <= 0:
        return np.nan, int(max_y), int(max_x)
    return float(image[max_y, max_x] / denominator), int(max_y), int(max_x)


def _cumulative_trace(row: np.void) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    required = {
        "roi_event_histogram",
        "roi_event_histogram_start_us",
        "roi_event_histogram_bin_us",
    }
    if not required.issubset(row.dtype.names or ()):
        return None
    histogram = np.asarray(row["roi_event_histogram"], dtype=np.float64)
    start = _float(row, "roi_event_histogram_start_us", 0.0)
    bin_size = _float(row, "roi_event_histogram_bin_us", 0.0)
    if (
        histogram.ndim != 2
        or histogram.shape[0] != 2
        or histogram.shape[1] == 0
        or not np.all(np.isfinite(histogram))
        or not np.any(histogram > 0)
        or not np.isfinite(start)
        or start < 0
        or not np.isfinite(bin_size)
        or bin_size <= 0
    ):
        return None
    times = start + np.arange(histogram.shape[1] + 1) * bin_size
    return (
        times,
        np.r_[0.0, np.cumsum(histogram[0])],
        np.r_[0.0, np.cumsum(histogram[1])],
    )


def _selected_windows(row: np.void, fallback: float) -> list[tuple[float, float, str]]:
    names = row.dtype.names or ()
    fields = (
        ("t_on_window_start", "t_on_window_stop", PLOT_COLORS["orange"]),
        ("t_off_window_start", "t_off_window_stop", PLOT_COLORS["sky_blue"]),
    )
    windows: list[tuple[float, float, str]] = []
    for start_name, stop_name, color in fields:
        if not {start_name, stop_name}.issubset(names):
            continue
        start = _float(row, start_name, np.nan)
        stop = _float(row, stop_name, np.nan)
        if np.isfinite(start) and np.isfinite(stop) and 0 <= start < stop:
            windows.append((start, stop, color))
    if windows:
        return windows

    first = _float(row, "t_1st", fallback)
    last = _float(row, "t_last", fallback)
    if np.isfinite(first) and np.isfinite(last) and 0 <= first < last:
        return [(first, last, PLOT_COLORS["gray"])]
    return []


def _detector_seed_in_roi(
    row: np.void, width: int, height: int
) -> tuple[float, float] | None:
    required = {"parent_seed_x", "parent_seed_y", "x", "y", "sub_x", "sub_y"}
    if not required.issubset(row.dtype.names or ()):
        return None
    origin_x = _float(row, "x", np.nan) - _float(row, "sub_x", np.nan)
    origin_y = _float(row, "y", np.nan) - _float(row, "sub_y", np.nan)
    seed_x = _float(row, "parent_seed_x", np.nan) - origin_x
    seed_y = _float(row, "parent_seed_y", np.nan) - origin_y
    if (
        not np.isfinite(seed_x)
        or not np.isfinite(seed_y)
        or not (-0.5 <= seed_x < width and -0.5 <= seed_y < height)
    ):
        return None
    return seed_x, seed_y


def _qc_by_id(qc: np.ndarray) -> dict[int, dict[str, object]]:
    names = qc.dtype.names or ()
    if "id" not in names:
        return {}
    return {
        int(row["id"]): {
            "accepted": bool(row["accepted"]) if "accepted" in names else True,
            "reason": str(row["primary_rejection_reason"])
            if "primary_rejection_reason" in names
            else "accepted",
        }
        for row in qc
    }


def _uncertainty(localizations: np.ndarray) -> np.ndarray:
    fields = {"sigma_x", "sigma_y", "cov_xy"}
    if localizations.size == 0 or not fields.issubset(localizations.dtype.names or ()):
        return np.full(localizations.size, np.nan, dtype=np.float64)
    return localization_uncertainty_px(localizations)


def _uncertainty_one(row: np.void) -> float:
    dtype = row.dtype
    array = np.empty(1, dtype=dtype)
    array[0] = row
    return float(_uncertainty(array)[0])


def _id_set(localizations: np.ndarray) -> set[int]:
    if "id" not in (localizations.dtype.names or ()):
        return set()
    return {int(value) for value in localizations["id"]}


def _row_id(row: np.void) -> int:
    return _integer(row, "id", -1)


def _float(row: np.void, field: str, default: float) -> float:
    if field not in (row.dtype.names or ()):
        return default
    return float(row[field])


def _integer(row: np.void, field: str, default: int) -> int:
    if field not in (row.dtype.names or ()):
        return default
    return int(row[field])


def _allocation(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + int(index < remainder) for index in range(parts)]


def _evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    if indices.size <= count:
        return np.asarray(indices, dtype=np.int64)
    positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
    return np.asarray(indices[positions], dtype=np.int64)


def _unique_indices(indices: list[int]) -> np.ndarray:
    return np.asarray(list(dict.fromkeys(indices)), dtype=np.int64)


def _replay_indices(quantile: np.ndarray, hot: np.ndarray) -> np.ndarray:
    combined = list(quantile[:6]) + list(hot[:6])
    return _unique_indices(combined)
