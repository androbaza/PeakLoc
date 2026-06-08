from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriftTrace:
    time_us: np.ndarray
    dx_px: np.ndarray
    dy_px: np.ndarray
    method: str


def estimate_drift_cross_correlation(
    localizations: np.ndarray,
    *,
    bins: int,
    render_pixel_size_px: float,
    upsample_factor: int = 20,
) -> DriftTrace:
    """Estimate drift from binned localization centroids.

    The first implementation keeps the public cross-correlation API but uses robust
    binned medians. It is deterministic, handles sparse data explicitly, and can be
    replaced by image cross-correlation without changing callers.
    """
    _ = (render_pixel_size_px, upsample_factor)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if localizations.size < 2 or not _has_fields(localizations, {"x", "y"}):
        return DriftTrace(
            time_us=np.asarray([], dtype=np.float64),
            dx_px=np.asarray([], dtype=np.float64),
            dy_px=np.asarray([], dtype=np.float64),
            method="insufficient_data",
        )

    time = _time_values(localizations)
    if time is None:
        return DriftTrace(
            time_us=np.asarray([], dtype=np.float64),
            dx_px=np.asarray([], dtype=np.float64),
            dy_px=np.asarray([], dtype=np.float64),
            method="missing_time",
        )

    if np.max(time) == np.min(time):
        return DriftTrace(
            time_us=np.asarray([float(np.min(time))]),
            dx_px=np.asarray([0.0]),
            dy_px=np.asarray([0.0]),
            method="single_time_bin",
        )

    edges = np.linspace(float(np.min(time)), float(np.max(time)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    median_time = np.full(bins, np.nan, dtype=np.float64)
    median_x = np.full(bins, np.nan, dtype=np.float64)
    median_y = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        if index == bins - 1:
            mask = (time >= edges[index]) & (time <= edges[index + 1])
        else:
            mask = (time >= edges[index]) & (time < edges[index + 1])
        if np.any(mask):
            median_time[index] = float(np.median(time[mask]))
            median_x[index] = float(np.median(localizations["x"][mask]))
            median_y[index] = float(np.median(localizations["y"][mask]))

    valid = np.isfinite(median_x) & np.isfinite(median_y)
    if np.count_nonzero(valid) < 2:
        return DriftTrace(
            time_us=centers[valid],
            dx_px=np.zeros(np.count_nonzero(valid), dtype=np.float64),
            dy_px=np.zeros(np.count_nonzero(valid), dtype=np.float64),
            method="insufficient_bins",
        )

    centers = median_time[valid]
    median_x = median_x[valid]
    median_y = median_y[valid]
    return DriftTrace(
        time_us=centers,
        dx_px=median_x - median_x[0],
        dy_px=median_y - median_y[0],
        method="binned_median",
    )


def apply_drift_correction(
    localizations: np.ndarray,
    drift: DriftTrace,
) -> np.ndarray:
    corrected = localizations.copy()
    if corrected.size == 0 or drift.time_us.size == 0:
        return corrected
    time = _time_values(corrected)
    if time is None:
        return corrected
    corrected["x"] = corrected["x"] - np.interp(time, drift.time_us, drift.dx_px)
    corrected["y"] = corrected["y"] - np.interp(time, drift.time_us, drift.dy_px)
    return corrected


def _time_values(localizations: np.ndarray) -> np.ndarray | None:
    names = localizations.dtype.names or ()
    if "t_peak" in names:
        return np.asarray(localizations["t_peak"], dtype=np.float64)
    if "t" in names:
        return np.asarray(localizations["t"], dtype=np.float64)
    return None


def _has_fields(array: np.ndarray, fields: set[str]) -> bool:
    return fields.issubset(array.dtype.names or ())
