from __future__ import annotations

import numpy as np


def localization_overlap_flags_dtype() -> list[tuple[str, object]]:
    return [
        ("id", np.uint64),
        ("possible_multi_emitter", np.bool_),
        ("asymmetric_residual", np.bool_),
        ("edge_truncated", np.bool_),
    ]


def evaluate_overlap_flags(
    localizations: np.ndarray,
    *,
    peak_fraction: float = 0.5,
    edge_distance_px: float = 1.0,
) -> np.ndarray:
    flags = np.zeros(localizations.size, dtype=localization_overlap_flags_dtype())
    if localizations.size == 0:
        return flags
    flags["id"] = (
        localizations["id"]
        if "id" in (localizations.dtype.names or ())
        else np.arange(localizations.size)
    )
    for index, localization in enumerate(localizations):
        roi = _combined_roi(localization)
        if roi is None or roi.size == 0:
            continue
        flags["possible_multi_emitter"][index] = _has_two_local_maxima(
            roi, peak_fraction=peak_fraction
        )
        flags["asymmetric_residual"][index] = _is_asymmetric(roi)
        flags["edge_truncated"][index] = _is_edge_truncated(
            localization, roi.shape, edge_distance_px=edge_distance_px
        )
    return flags


def _combined_roi(localization: np.void) -> np.ndarray | None:
    names = localization.dtype.names or ()
    positive = (
        np.asarray(localization["roi"], dtype=np.float64) if "roi" in names else None
    )
    negative = (
        np.asarray(localization["roi_n"], dtype=np.float64)
        if "roi_n" in names
        else None
    )
    if positive is not None and negative is not None:
        return positive + negative
    if positive is not None:
        return positive
    if negative is not None:
        return negative
    return None


def _has_two_local_maxima(roi: np.ndarray, *, peak_fraction: float) -> bool:
    if roi.size == 0 or np.max(roi) <= 0:
        return False
    threshold = float(np.max(roi) * peak_fraction)
    maxima: list[tuple[int, int]] = []
    for y in range(roi.shape[0]):
        for x in range(roi.shape[1]):
            value = roi[y, x]
            if value < threshold:
                continue
            y0 = max(0, y - 1)
            y1 = min(roi.shape[0], y + 2)
            x0 = max(0, x - 1)
            x1 = min(roi.shape[1], x + 2)
            if value >= np.max(roi[y0:y1, x0:x1]):
                maxima.append((y, x))
    for first_index, first in enumerate(maxima):
        for second in maxima[first_index + 1 :]:
            if np.hypot(first[1] - second[1], first[0] - second[0]) >= 2.0:
                return True
    return False


def _is_asymmetric(roi: np.ndarray) -> bool:
    total = float(np.sum(roi))
    if total <= 0:
        return False
    center_y = (roi.shape[0] - 1) / 2
    center_x = (roi.shape[1] - 1) / 2
    yy, xx = np.indices(roi.shape)
    x_moment = float(np.sum((xx - center_x) * roi) / total)
    y_moment = float(np.sum((yy - center_y) * roi) / total)
    return np.hypot(x_moment, y_moment) > 0.35


def _is_edge_truncated(
    localization: np.void, roi_shape: tuple[int, int], *, edge_distance_px: float
) -> bool:
    names = localization.dtype.names or ()
    if "sub_x" not in names or "sub_y" not in names:
        return False
    sub_x = float(localization["sub_x"])
    sub_y = float(localization["sub_y"])
    return (
        sub_x <= edge_distance_px
        or sub_y <= edge_distance_px
        or sub_x >= roi_shape[1] - 1 - edge_distance_px
        or sub_y >= roi_shape[0] - 1 - edge_distance_px
    )
