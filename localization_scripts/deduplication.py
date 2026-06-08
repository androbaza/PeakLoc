from __future__ import annotations

import numpy as np

from localization_scripts.localization_fitting import localization_uncertainty_px


def merge_duplicate_localizations(
    localizations: np.ndarray,
    *,
    spatial_radius_px: float,
    time_radius_us: float,
    ranking_field: str = "nll_per_event",
) -> np.ndarray:
    if spatial_radius_px < 0:
        raise ValueError("spatial_radius_px must be non-negative")
    if time_radius_us < 0:
        raise ValueError("time_radius_us must be non-negative")
    if localizations.size == 0:
        return localizations.copy()
    if not {"x", "y"}.issubset(localizations.dtype.names or ()):
        raise ValueError("localizations must have x and y fields")

    remaining = set(range(localizations.size))
    selected: list[int] = []
    order = sorted(
        range(localizations.size),
        key=lambda index: _ranking_key(localizations, index, ranking_field),
    )
    for index in order:
        if index not in remaining:
            continue
        selected.append(index)
        remaining.remove(index)
        duplicate_indices = [
            other
            for other in list(remaining)
            if _is_duplicate(
                localizations[index],
                localizations[other],
                spatial_radius_px=spatial_radius_px,
                time_radius_us=time_radius_us,
            )
        ]
        for duplicate_index in duplicate_indices:
            remaining.remove(duplicate_index)
    selected.sort()
    return localizations[selected].copy()


def _ranking_key(
    localizations: np.ndarray, index: int, ranking_field: str
) -> tuple[float, float, float, float, int]:
    accepted_rank = 0.0 if _accepted(localizations, index) else 1.0
    uncertainty = _uncertainty(localizations, index)
    ranking_value = _float_field(localizations, ranking_field, index, np.inf)
    event_count = _float_field(localizations, "E_total", index, 0.0) + _float_field(
        localizations, "E_total_n", index, 0.0
    )
    return accepted_rank, uncertainty, ranking_value, -event_count, index


def _accepted(localizations: np.ndarray, index: int) -> bool:
    names = localizations.dtype.names or ()
    if "accepted" in names:
        return bool(localizations["accepted"][index])
    if "fit_success" in names:
        return bool(localizations["fit_success"][index])
    return True


def _uncertainty(localizations: np.ndarray, index: int) -> float:
    if {"sigma_x", "sigma_y", "cov_xy"}.issubset(localizations.dtype.names or ()):
        return float(localization_uncertainty_px(localizations[index : index + 1])[0])
    return np.inf


def _is_duplicate(
    first: np.void,
    second: np.void,
    *,
    spatial_radius_px: float,
    time_radius_us: float,
) -> bool:
    distance = float(
        np.hypot(
            float(first["x"]) - float(second["x"]),
            float(first["y"]) - float(second["y"]),
        )
    )
    if distance > spatial_radius_px:
        return False
    first_time = _time_value(first)
    second_time = _time_value(second)
    if first_time is None or second_time is None:
        return True
    return abs(first_time - second_time) <= time_radius_us


def _time_value(row: np.void) -> float | None:
    names = row.dtype.names or ()
    if "t_peak" in names:
        return float(row["t_peak"])
    if "t" in names:
        return float(row["t"])
    return None


def _float_field(
    localizations: np.ndarray, field_name: str, index: int, default: float
) -> float:
    if field_name not in (localizations.dtype.names or ()):
        return default
    value = float(localizations[field_name][index])
    return value if np.isfinite(value) else default
