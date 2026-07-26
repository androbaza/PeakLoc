"""Compact figures intended for direct collaborator hand-off."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import (
    DOUBLE_COLUMN_WIDTH_IN,
    PLOT_COLORS,
    SEQUENTIAL_CMAP,
    save_publication_figure,
    style_publication_axis,
)
from localization_scripts.python_compat import strict_zip


def save_detection_and_fit_summary(
    *,
    detection_funnel: dict[str, int],
    accepted_localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    figure_dir: Path,
    config: PeakLocConfig,
) -> list[Path]:
    """Save a four-panel quantitative overview of detection and fitting quality."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COLUMN_WIDTH_IN, 6.1), constrained_layout=True
    )
    _plot_detection_funnel(axes[0, 0], detection_funnel)
    _plot_uncertainty_distribution(axes[0, 1], localization_qc, config)
    _plot_cropped_density(axes[1, 0], accepted_localizations, config)
    _plot_hot_pixel_distribution(axes[1, 1], attempted_localizations)
    for label, axis in strict_zip("ABCD", axes.ravel()):
        axis.text(
            -0.18,
            1.08,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
            va="top",
        )
    path = figure_dir / "detection_and_fit_summary.png"
    paths = save_publication_figure(
        figure,
        path,
        dpi=max(config.qc_static_dpi, 450),
        save_vector=config.qc_save_vector,
    )
    plt.close(figure)
    return paths


def _plot_detection_funnel(axis, values: dict[str, int]) -> None:
    labels = [label.replace("_", "\n") for label in values]
    counts = np.asarray(list(values.values()), dtype=np.float64)
    axis.bar(np.arange(counts.size), counts, color=PLOT_COLORS["blue"], zorder=2)
    axis.set(
        title="Detection and fitting funnel",
        xticks=np.arange(counts.size),
        xticklabels=labels,
        ylabel="Candidates / events (log scale)",
        yscale="log",
    )
    axis.tick_params(axis="x", labelrotation=35)
    style_publication_axis(axis, show_grid=True)


def _plot_uncertainty_distribution(
    axis, localization_qc: np.ndarray, config: PeakLocConfig
) -> None:
    values = _field(localization_qc, "uncertainty_nm")
    values = values[np.isfinite(values)]
    if values.size:
        axis.hist(values, bins="auto", color=PLOT_COLORS["green"], alpha=0.9)
        if config.max_localization_uncertainty_nm is not None:
            axis.axvline(
                config.max_localization_uncertainty_nm,
                color=PLOT_COLORS["vermillion"],
                linestyle="--",
                linewidth=1.0,
                label="acceptance threshold",
            )
            axis.legend(frameon=False, fontsize=6)
    else:
        axis.text(
            0.5,
            0.5,
            "No finite uncertainty values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(title="Localization uncertainty", xlabel="Uncertainty (nm)", ylabel="Fits")
    style_publication_axis(axis, show_grid=True)


def _plot_cropped_density(
    axis, localizations: np.ndarray, config: PeakLocConfig
) -> None:
    if localizations.size == 0 or not {"x", "y"}.issubset(
        localizations.dtype.names or ()
    ):
        axis.text(
            0.5,
            0.5,
            "No accepted localizations",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    x = np.asarray(localizations["x"], dtype=np.float64)
    y = np.asarray(localizations["y"], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        axis.text(
            0.5,
            0.5,
            "No finite localization coordinates",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    x, y = x[valid], y[valid]
    x_range, y_range = _data_coordinate_ranges(x, y, config)
    histogram = axis.hist2d(
        x, y, bins=160, range=[x_range, y_range], cmap=SEQUENTIAL_CMAP
    )
    axis.invert_yaxis()
    colorbar = plt.colorbar(histogram[3], ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Localizations / bin", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    axis.set(
        title="Accepted localization density (data crop)",
        xlabel="x (sensor pixel)",
        ylabel="y (sensor pixel)",
    )
    style_publication_axis(axis)


def _plot_hot_pixel_distribution(axis, localizations: np.ndarray) -> None:
    shares = _hot_pixel_shares(localizations)
    shares = shares[np.isfinite(shares)]
    if shares.size:
        axis.hist(
            shares * 100,
            bins=np.linspace(0, 100, 41),
            color=PLOT_COLORS["orange"],
            alpha=0.9,
        )
        axis.axvline(80, color=PLOT_COLORS["vermillion"], linewidth=1.0, linestyle="--")
        count = int(np.count_nonzero(shares >= 0.8))
        axis.text(
            0.97,
            0.95,
            f"{count:,} candidates ≥80%",
            ha="right",
            va="top",
            fontsize=6,
            transform=axis.transAxes,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No positive ROI images",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        title="Single-pixel dominance screen",
        xlabel="Largest positive pixel / positive ROI events (%)",
        ylabel="Candidates",
        xlim=(0, 100),
    )
    style_publication_axis(axis, show_grid=True)


def _data_coordinate_ranges(
    x: np.ndarray, y: np.ndarray, config: PeakLocConfig
) -> tuple[list[float], list[float]]:
    x_low, x_high = float(np.min(x)), float(np.max(x))
    y_low, y_high = float(np.min(y)), float(np.max(y))
    x_pad = max((x_high - x_low) * 0.08, 4.0)
    y_pad = max((y_high - y_low) * 0.08, 4.0)
    return (
        [
            max(0.0, float(x_low - x_pad)),
            min(float(config.sensor_width), float(x_high + x_pad)),
        ],
        [
            max(0.0, float(y_low - y_pad)),
            min(float(config.sensor_height), float(y_high + y_pad)),
        ],
    )


def _hot_pixel_shares(localizations: np.ndarray) -> np.ndarray:
    names = localizations.dtype.names or ()
    if localizations.size == 0 or "roi" not in names:
        return np.empty(0, dtype=np.float64)
    values: list[float] = []
    for row in localizations:
        image = np.asarray(row["roi"], dtype=np.float64)
        denominator = _scalar(row, "E_total", float(np.nansum(image)))
        if denominator > 0 and image.size:
            values.append(float(np.nanmax(image) / denominator))
    return np.asarray(values, dtype=np.float64)


def _field(array: np.ndarray, name: str) -> np.ndarray:
    if array.size == 0 or name not in (array.dtype.names or ()):
        return np.empty(0, dtype=np.float64)
    return np.asarray(array[name], dtype=np.float64)


def _scalar(row: np.void, name: str, default: float) -> float:
    if name not in (row.dtype.names or ()):
        return default
    return float(row[name])
