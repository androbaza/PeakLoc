from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label


@dataclass(frozen=True)
class SpatialMask:
    """Target centres and the event support required to evaluate them safely."""

    target_mask: np.ndarray | None
    support_mask: np.ndarray | None
    target_coords: np.ndarray | None
    support_coords: np.ndarray | None
    sample_start_us: int | None
    sample_stop_us: int | None
    calibration_event_count: int
    seed_pixel_count: int
    retained_seed_pixel_count: int
    fallback_reason: str | None = None

    @property
    def is_active(self) -> bool:
        return self.target_mask is not None and self.support_mask is not None

    @property
    def target_coverage(self) -> float | None:
        if self.target_mask is None:
            return None
        return float(np.mean(self.target_mask))

    @property
    def support_coverage(self) -> float | None:
        if self.support_mask is None:
            return None
        return float(np.mean(self.support_mask))

    def metadata(self) -> dict[str, object]:
        return {
            "active": self.is_active,
            "sample_start_us": self.sample_start_us,
            "sample_stop_us": self.sample_stop_us,
            "calibration_event_count": self.calibration_event_count,
            "seed_pixel_count": self.seed_pixel_count,
            "retained_seed_pixel_count": self.retained_seed_pixel_count,
            "target_pixel_count": (
                None
                if self.target_mask is None
                else int(np.count_nonzero(self.target_mask))
            ),
            "support_pixel_count": (
                None
                if self.support_mask is None
                else int(np.count_nonzero(self.support_mask))
            ),
            "target_coverage": self.target_coverage,
            "support_coverage": self.support_coverage,
            "fallback_reason": self.fallback_reason,
        }


def accumulate_event_density(
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Count in-bounds events in a native-resolution image with bounded temporaries."""
    density = np.zeros(sensor_shape, dtype=np.uint32)
    return density, _accumulate_density(density, events, sensor_shape, chunk_size)


def accumulate_event_density_in_time_window(
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    start_us: int,
    stop_us: int,
    timestamps_monotonic: bool,
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Count a time range without materializing a full boolean-selected recording."""
    density = np.zeros(sensor_shape, dtype=np.uint32)
    timestamps = events["t"]
    if timestamps_monotonic:
        start_index = int(np.searchsorted(timestamps, start_us, side="left"))
        stop_index = int(np.searchsorted(timestamps, stop_us, side="left"))
        event_count = _accumulate_density(
            density,
            events[start_index:stop_index],
            sensor_shape,
            chunk_size,
        )
        return density, event_count

    event_count = 0
    for start in range(0, events.size, chunk_size):
        chunk = events[start : start + chunk_size]
        in_window = (chunk["t"] >= start_us) & (chunk["t"] < stop_us)
        if np.any(in_window):
            event_count += _accumulate_density(
                density,
                chunk[in_window],
                sensor_shape,
                chunk_size,
            )
    return density, event_count


def build_spatial_mask(
    density: np.ndarray,
    *,
    sample_start_us: int,
    sample_stop_us: int,
    calibration_event_count: int,
    min_events: int,
    min_component_pixels: int,
    margin_px: int,
    support_margin_px: int,
    max_support_coverage: float,
) -> SpatialMask:
    """Build a conservative morphology-expanded target mask from event density."""
    seed_mask = density >= min_events
    seed_pixel_count = int(np.count_nonzero(seed_mask))
    if seed_pixel_count == 0:
        return _inactive_mask(
            sample_start_us,
            sample_stop_us,
            calibration_event_count,
            seed_pixel_count,
            fallback_reason="No pixels reached spatial_mask_min_events.",
        )

    labels, _ = label(seed_mask, structure=np.ones((3, 3), dtype=bool))
    component_sizes = np.bincount(labels.ravel())
    component_labels = component_sizes >= min_component_pixels
    component_labels[0] = False
    retained_seed_mask = component_labels[labels]
    retained_seed_pixel_count = int(np.count_nonzero(retained_seed_mask))
    if retained_seed_pixel_count == 0:
        return _inactive_mask(
            sample_start_us,
            sample_stop_us,
            calibration_event_count,
            seed_pixel_count,
            fallback_reason="No density components reached spatial_mask_min_component_pixels.",
        )

    target_mask = _dilate(retained_seed_mask, margin_px)
    support_mask = _dilate(target_mask, support_margin_px)
    support_coverage = float(np.mean(support_mask))
    if support_coverage >= max_support_coverage:
        return _inactive_mask(
            sample_start_us,
            sample_stop_us,
            calibration_event_count,
            seed_pixel_count,
            retained_seed_pixel_count,
            fallback_reason=(
                "Spatial support covers "
                f"{support_coverage:.1%}, above spatial_mask_max_support_coverage."
            ),
        )

    return SpatialMask(
        target_mask=target_mask,
        support_mask=support_mask,
        target_coords=np.argwhere(target_mask).astype(np.int32, copy=False),
        support_coords=np.argwhere(support_mask).astype(np.int32, copy=False),
        sample_start_us=sample_start_us,
        sample_stop_us=sample_stop_us,
        calibration_event_count=calibration_event_count,
        seed_pixel_count=seed_pixel_count,
        retained_seed_pixel_count=retained_seed_pixel_count,
    )


def disabled_spatial_mask(fallback_reason: str) -> SpatialMask:
    """Describe an intentionally disabled mask without allocating sensor-sized arrays."""
    return _inactive_mask(
        None,
        None,
        calibration_event_count=0,
        seed_pixel_count=0,
        fallback_reason=fallback_reason,
    )


def _dilate(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px == 0:
        return mask.copy()
    footprint = np.ones((margin_px * 2 + 1, margin_px * 2 + 1), dtype=bool)
    return binary_dilation(mask, structure=footprint)


def _accumulate_density(
    density: np.ndarray,
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    chunk_size: int,
) -> int:
    event_count = 0
    height, width = sensor_shape
    for start in range(0, events.size, chunk_size):
        chunk = events[start : start + chunk_size]
        x = chunk["x"]
        y = chunk["y"]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(density, (y[valid], x[valid]), 1)
        event_count += int(np.count_nonzero(valid))
    return event_count


def _inactive_mask(
    sample_start_us: int | None,
    sample_stop_us: int | None,
    calibration_event_count: int,
    seed_pixel_count: int,
    retained_seed_pixel_count: int = 0,
    *,
    fallback_reason: str,
) -> SpatialMask:
    return SpatialMask(
        target_mask=None,
        support_mask=None,
        target_coords=None,
        support_coords=None,
        sample_start_us=sample_start_us,
        sample_stop_us=sample_stop_us,
        calibration_event_count=calibration_event_count,
        seed_pixel_count=seed_pixel_count,
        retained_seed_pixel_count=retained_seed_pixel_count,
        fallback_reason=fallback_reason,
    )
