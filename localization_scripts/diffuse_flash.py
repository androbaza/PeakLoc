from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True)
class TimeInterval:
    start_us: int
    stop_us: int

    @property
    def duration_us(self) -> int:
        return self.stop_us - self.start_us


@dataclass(frozen=True)
class DiffuseFlashDetection:
    intervals: tuple[TimeInterval, ...]
    excluded_event_count: int
    transition_bin_count: int


def detect_diffuse_flash_intervals(
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    bin_duration_us: int,
    min_events_per_polarity: int,
    min_active_pixel_fraction: float,
    max_gap_us: int,
    padding_us: int,
    chunk_size: int = 1_000_000,
) -> DiffuseFlashDetection:
    """Find full-sensor illumination intervals in a monotonic event stream.

    Event counts cheaply select candidate time bins. Only those candidates pay the
    cost of counting unique active pixels, keeping the scan bounded for long RAW
    recordings. Nearby broad transitions are merged so an ON/OFF pair excludes the
    complete interval between them.
    """
    if events.size == 0:
        return DiffuseFlashDetection((), 0, 0)

    timestamps = events["t"]

    first_bin = int(timestamps[0]) // bin_duration_us
    last_bin = int(timestamps[-1]) // bin_duration_us
    bin_count = last_bin - first_bin + 1
    positive_counts = np.zeros(bin_count, dtype=np.int64)
    negative_counts = np.zeros(bin_count, dtype=np.int64)

    previous_timestamp: int | None = None
    for start in range(0, events.size, chunk_size):
        chunk = events[start : start + chunk_size]
        chunk_timestamps = chunk["t"]
        if (
            previous_timestamp is not None and chunk_timestamps[0] < previous_timestamp
        ) or np.any(chunk_timestamps[1:] < chunk_timestamps[:-1]):
            raise ValueError("Diffuse flash detection requires monotonic timestamps")
        previous_timestamp = int(chunk_timestamps[-1])
        relative_bins = (chunk["t"] // bin_duration_us - first_bin).astype(
            np.int64, copy=False
        )
        positive = chunk["p"] == 1
        positive_counts += np.bincount(relative_bins[positive], minlength=bin_count)
        negative_counts += np.bincount(relative_bins[~positive], minlength=bin_count)

    candidate_bins = np.flatnonzero(
        (positive_counts >= min_events_per_polarity)
        | (negative_counts >= min_events_per_polarity)
    )
    required_active_pixels = math.ceil(
        sensor_shape[0] * sensor_shape[1] * min_active_pixel_fraction
    )
    transition_bins = []
    for relative_bin in candidate_bins:
        bin_start = (first_bin + int(relative_bin)) * bin_duration_us
        start_index = int(np.searchsorted(timestamps, bin_start, side="left"))
        stop_index = int(
            np.searchsorted(timestamps, bin_start + bin_duration_us, side="left")
        )
        bin_events = events[start_index:stop_index]
        if _polarity_has_broad_coverage(
            bin_events,
            polarity=1,
            event_count=int(positive_counts[relative_bin]),
            min_events=min_events_per_polarity,
            required_active_pixels=required_active_pixels,
            sensor_shape=sensor_shape,
        ) or _polarity_has_broad_coverage(
            bin_events,
            polarity=0,
            event_count=int(negative_counts[relative_bin]),
            min_events=min_events_per_polarity,
            required_active_pixels=required_active_pixels,
            sensor_shape=sensor_shape,
        ):
            transition_bins.append(bin_start)

    intervals = _merge_transition_bins(
        transition_bins,
        bin_duration_us=bin_duration_us,
        max_gap_us=max_gap_us,
        padding_us=padding_us,
        recording_start_us=int(timestamps[0]),
        recording_stop_us=int(timestamps[-1]) + 1,
    )
    excluded_event_count = sum(
        _event_count_in_interval(timestamps, interval) for interval in intervals
    )
    return DiffuseFlashDetection(
        intervals=intervals,
        excluded_event_count=excluded_event_count,
        transition_bin_count=len(transition_bins),
    )


def exclude_time_intervals(
    events: np.ndarray, intervals: tuple[TimeInterval, ...]
) -> tuple[np.ndarray, int]:
    """Return events outside excluded intervals and the number removed."""
    if events.size == 0 or not intervals:
        return events, 0

    retained_parts = list(iter_retained_event_spans(events, intervals))
    retained_count = sum(part.size for part in retained_parts)
    excluded_count = int(events.size) - retained_count
    if excluded_count == 0:
        return events, 0
    if not retained_parts:
        return events[:0], excluded_count
    return np.concatenate(retained_parts), excluded_count


def iter_retained_event_spans(
    events: np.ndarray,
    intervals: tuple[TimeInterval, ...],
    *,
    start_us: int | None = None,
    stop_us: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield monotonic event-array views outside excluded time intervals."""
    if events.size == 0:
        return
    timestamps = events["t"]
    lower = int(timestamps[0]) if start_us is None else start_us
    upper = int(timestamps[-1]) + 1 if stop_us is None else stop_us
    cursor = int(np.searchsorted(timestamps, lower, side="left"))
    stop_index = int(np.searchsorted(timestamps, upper, side="left"))
    for interval in intervals:
        if interval.stop_us <= lower:
            continue
        if interval.start_us >= upper:
            break
        interval_start = int(
            np.searchsorted(timestamps, max(interval.start_us, lower), side="left")
        )
        if interval_start > cursor:
            yield events[cursor:interval_start]
        cursor = max(
            cursor,
            int(np.searchsorted(timestamps, min(interval.stop_us, upper), side="left")),
        )
    if cursor < stop_index:
        yield events[cursor:stop_index]


def _polarity_has_broad_coverage(
    events: np.ndarray,
    *,
    polarity: int,
    event_count: int,
    min_events: int,
    required_active_pixels: int,
    sensor_shape: tuple[int, int],
) -> bool:
    if event_count < min_events:
        return False
    height, width = sensor_shape
    selected = events[events["p"] == polarity]
    valid = (selected["x"] < width) & (selected["y"] < height)
    pixel_ids = selected["y"][valid].astype(np.int64) * width + selected["x"][valid]
    return np.unique(pixel_ids).size >= required_active_pixels


def _merge_transition_bins(
    transition_bins: list[int],
    *,
    bin_duration_us: int,
    max_gap_us: int,
    padding_us: int,
    recording_start_us: int,
    recording_stop_us: int,
) -> tuple[TimeInterval, ...]:
    if not transition_bins:
        return ()
    intervals = []
    current_start = transition_bins[0]
    current_stop = current_start + bin_duration_us
    for bin_start in transition_bins[1:]:
        if bin_start - current_stop <= max_gap_us:
            current_stop = bin_start + bin_duration_us
            continue
        intervals.append((current_start, current_stop))
        current_start = bin_start
        current_stop = bin_start + bin_duration_us
    intervals.append((current_start, current_stop))
    return tuple(
        TimeInterval(
            start_us=max(start - padding_us, recording_start_us),
            stop_us=min(stop + padding_us, recording_stop_us),
        )
        for start, stop in intervals
    )


def _event_count_in_interval(timestamps: np.ndarray, interval: TimeInterval) -> int:
    start = int(np.searchsorted(timestamps, interval.start_us, side="left"))
    stop = int(np.searchsorted(timestamps, interval.stop_us, side="left"))
    return stop - start
