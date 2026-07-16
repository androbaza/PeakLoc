from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

from localization_scripts.diffuse_flash import TimeInterval, iter_retained_event_spans


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
    mean_events_per_pixel: float | None = None
    min_density_quotient: float | None = None
    seed_threshold_events: float | None = None

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
            "mean_events_per_pixel": self.mean_events_per_pixel,
            "min_density_quotient": self.min_density_quotient,
            "seed_threshold_events": self.seed_threshold_events,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class SpatialMaskPreview:
    """Intermediate masks and normalized threshold values for mask review."""

    seed_mask: np.ndarray
    retained_seed_mask: np.ndarray
    target_mask: np.ndarray | None
    support_mask: np.ndarray | None
    calibration_event_count: int
    mean_events_per_pixel: float
    min_density_quotient: float
    seed_threshold_events: float
    seed_pixel_count: int
    retained_seed_pixel_count: int
    fallback_reason: str | None = None

    @property
    def support_coverage(self) -> float | None:
        if self.support_mask is None:
            return None
        return float(np.mean(self.support_mask))


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
    excluded_intervals: tuple[TimeInterval, ...] = (),
    chunk_size: int = 1_000_000,
) -> tuple[np.ndarray, int]:
    """Count a time range without materializing a full boolean-selected recording."""
    density = np.zeros(sensor_shape, dtype=np.uint32)
    if timestamps_monotonic:
        event_count = 0
        for retained_events in iter_retained_event_spans(
            events,
            excluded_intervals,
            start_us=start_us,
            stop_us=stop_us,
        ):
            event_count += _accumulate_density(
                density,
                retained_events,
                sensor_shape,
                chunk_size,
            )
        return density, event_count

    event_count = 0
    for start in range(0, events.size, chunk_size):
        chunk = events[start : start + chunk_size]
        event_count += _accumulate_density_in_time_range(
            density,
            chunk,
            sensor_shape,
            start_us=start_us,
            stop_us=stop_us,
            chunk_size=chunk_size,
        )
    return density, event_count


def build_spatial_mask(
    density: np.ndarray,
    *,
    sample_start_us: int,
    sample_stop_us: int,
    calibration_event_count: int,
    min_density_quotient: float,
    min_component_pixels: int,
    margin_px: int,
    support_margin_px: int,
    max_support_coverage: float,
) -> SpatialMask:
    """Build a conservative morphology-expanded target mask from event density."""
    preview = preview_spatial_mask(
        density,
        calibration_event_count=calibration_event_count,
        min_density_quotient=min_density_quotient,
        min_component_pixels=min_component_pixels,
        margin_px=margin_px,
        support_margin_px=support_margin_px,
        max_support_coverage=max_support_coverage,
    )
    if preview.fallback_reason is not None:
        return _inactive_mask(
            sample_start_us,
            sample_stop_us,
            calibration_event_count,
            preview.seed_pixel_count,
            preview.retained_seed_pixel_count,
            fallback_reason=preview.fallback_reason,
            mean_events_per_pixel=preview.mean_events_per_pixel,
            min_density_quotient=preview.min_density_quotient,
            seed_threshold_events=preview.seed_threshold_events,
        )
    if preview.target_mask is None or preview.support_mask is None:
        raise RuntimeError("Active spatial-mask preview has no target/support masks")

    return SpatialMask(
        target_mask=preview.target_mask,
        support_mask=preview.support_mask,
        target_coords=np.argwhere(preview.target_mask).astype(np.int32, copy=False),
        support_coords=np.argwhere(preview.support_mask).astype(np.int32, copy=False),
        sample_start_us=sample_start_us,
        sample_stop_us=sample_stop_us,
        calibration_event_count=calibration_event_count,
        seed_pixel_count=preview.seed_pixel_count,
        retained_seed_pixel_count=preview.retained_seed_pixel_count,
        mean_events_per_pixel=preview.mean_events_per_pixel,
        min_density_quotient=preview.min_density_quotient,
        seed_threshold_events=preview.seed_threshold_events,
    )


def preview_spatial_mask(
    density: np.ndarray,
    *,
    calibration_event_count: int,
    min_density_quotient: float,
    min_component_pixels: int,
    margin_px: int,
    support_margin_px: int,
    max_support_coverage: float,
) -> SpatialMaskPreview:
    """Evaluate a mask configuration without creating localization coordinates.

    The normalized density threshold is the requested quotient times the mean number
    of calibration events per sensor pixel. It responds to both exposure duration and
    global bias/background changes without relying on a recording-specific count.
    """
    if density.ndim != 2:
        raise ValueError("Spatial-mask density must be a two-dimensional image")
    if density.size == 0:
        raise ValueError("Spatial-mask density image must not be empty")
    if calibration_event_count < 0:
        raise ValueError("Spatial-mask calibration_event_count must not be negative")
    if min_density_quotient <= 0:
        raise ValueError("Spatial-mask min_density_quotient must be positive")
    if min_component_pixels <= 0:
        raise ValueError("Spatial-mask min_component_pixels must be positive")
    if margin_px < 0 or support_margin_px < 0:
        raise ValueError("Spatial-mask dilation margins must not be negative")
    if not 0 < max_support_coverage <= 1:
        raise ValueError(
            "Spatial-mask max_support_coverage must be in the interval (0, 1]"
        )

    mean_events_per_pixel = float(calibration_event_count) / density.size
    seed_threshold_events = mean_events_per_pixel * min_density_quotient
    if calibration_event_count <= 0:
        empty_mask = np.zeros(density.shape, dtype=bool)
        return SpatialMaskPreview(
            seed_mask=empty_mask,
            retained_seed_mask=empty_mask.copy(),
            target_mask=None,
            support_mask=None,
            calibration_event_count=calibration_event_count,
            mean_events_per_pixel=mean_events_per_pixel,
            min_density_quotient=min_density_quotient,
            seed_threshold_events=seed_threshold_events,
            seed_pixel_count=0,
            retained_seed_pixel_count=0,
            fallback_reason="No calibration events were available for spatial masking.",
        )

    seed_mask = density >= seed_threshold_events
    seed_pixel_count = int(np.count_nonzero(seed_mask))
    if seed_pixel_count == 0:
        return SpatialMaskPreview(
            seed_mask=seed_mask,
            retained_seed_mask=np.zeros(density.shape, dtype=bool),
            target_mask=None,
            support_mask=None,
            calibration_event_count=calibration_event_count,
            mean_events_per_pixel=mean_events_per_pixel,
            min_density_quotient=min_density_quotient,
            seed_threshold_events=seed_threshold_events,
            seed_pixel_count=seed_pixel_count,
            retained_seed_pixel_count=0,
            fallback_reason="No pixels reached the normalized spatial-mask threshold.",
        )

    labels, _ = label(seed_mask, structure=np.ones((3, 3), dtype=bool))
    component_sizes = np.bincount(labels.ravel())
    component_labels = component_sizes >= min_component_pixels
    component_labels[0] = False
    retained_seed_mask = component_labels[labels]
    retained_seed_pixel_count = int(np.count_nonzero(retained_seed_mask))
    if retained_seed_pixel_count == 0:
        return SpatialMaskPreview(
            seed_mask=seed_mask,
            retained_seed_mask=retained_seed_mask,
            target_mask=None,
            support_mask=None,
            calibration_event_count=calibration_event_count,
            mean_events_per_pixel=mean_events_per_pixel,
            min_density_quotient=min_density_quotient,
            seed_threshold_events=seed_threshold_events,
            seed_pixel_count=seed_pixel_count,
            retained_seed_pixel_count=retained_seed_pixel_count,
            fallback_reason="No density components reached spatial_mask_min_component_pixels.",
        )

    target_mask = _dilate(retained_seed_mask, margin_px)
    support_mask = _dilate(target_mask, support_margin_px)
    support_coverage = float(np.mean(support_mask))
    fallback_reason = None
    if support_coverage >= max_support_coverage:
        fallback_reason = (
            "Spatial support covers "
            f"{support_coverage:.1%}, above spatial_mask_max_support_coverage."
        )
    return SpatialMaskPreview(
        seed_mask=seed_mask,
        retained_seed_mask=retained_seed_mask,
        target_mask=target_mask,
        support_mask=support_mask,
        calibration_event_count=calibration_event_count,
        mean_events_per_pixel=mean_events_per_pixel,
        min_density_quotient=min_density_quotient,
        seed_threshold_events=seed_threshold_events,
        seed_pixel_count=seed_pixel_count,
        retained_seed_pixel_count=retained_seed_pixel_count,
        fallback_reason=fallback_reason,
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


def _accumulate_density_in_time_range(
    density: np.ndarray,
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    start_us: int,
    stop_us: int,
    chunk_size: int,
) -> int:
    """Count a time-bounded event range without a selected-event array copy."""
    event_count = 0
    height, width = sensor_shape
    for start in range(0, events.size, chunk_size):
        chunk = events[start : start + chunk_size]
        x = chunk["x"]
        y = chunk["y"]
        timestamps = chunk["t"]
        valid = (
            (timestamps >= start_us)
            & (timestamps < stop_us)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
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
    mean_events_per_pixel: float | None = None,
    min_density_quotient: float | None = None,
    seed_threshold_events: float | None = None,
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
        mean_events_per_pixel=mean_events_per_pixel,
        min_density_quotient=min_density_quotient,
        seed_threshold_events=seed_threshold_events,
    )
