"""Best-effort, atomic progress snapshots for the desktop live monitor."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

LIVE_PROGRESS_ENV = "PEAKLOC_LIVE_PROGRESS_DIR"
LIVE_PROGRESS_SCHEMA_VERSION = 1
PEAK_SAMPLE_LIMIT = 12
PEAK_TRACE_POINT_LIMIT = 512
ROI_SAMPLE_LIMIT = 12
ROI_EVENT_LIMIT = 20_000


def initialize_recording_progress(
    filename: Path,
    *,
    time_min: int | None,
    time_max: int,
    time_slices: list[Any],
    sensor_shape: tuple[int, int],
) -> None:
    """Publish immutable recording and slice bounds if live monitoring is enabled."""
    try:
        recording_directory = _recording_directory(filename)
        if recording_directory is None:
            return
        recording_directory.mkdir(parents=True, exist_ok=True)
        (recording_directory / "states").mkdir(exist_ok=True)
        (recording_directory / "snapshots").mkdir(exist_ok=True)
        slices = [
            {"start_us": int(time_slice.start), "stop_us": int(time_slice.stop)}
            for time_slice in time_slices
        ]
        selected_start = min(
            (item["start_us"] for item in slices), default=time_min or 0
        )
        selected_stop = max((item["stop_us"] for item in slices), default=time_max)
        manifest = {
            "schema_version": LIVE_PROGRESS_SCHEMA_VERSION,
            "recording_id": recording_directory.name,
            "recording_name": filename.name,
            "recording_path": str(filename),
            "recording_start_us": int(time_min or 0),
            "recording_stop_us": int(time_max),
            "selected_start_us": int(selected_start),
            "selected_stop_us": int(selected_stop),
            "sensor_height": int(sensor_shape[0]),
            "sensor_width": int(sensor_shape[1]),
            "slices": slices,
            "state": "running",
        }
        _atomic_write_json(recording_directory / "manifest.json", manifest)
        root = _progress_root()
        if root is not None:
            _atomic_write_json(
                root / "current.json",
                {
                    "schema_version": LIVE_PROGRESS_SCHEMA_VERSION,
                    "recording_id": recording_directory.name,
                },
            )
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn("initialize recording progress", error)


def publish_recording_state(
    filename: Path, state: str, *, error_message: str | None = None
) -> None:
    """Update recording completion state without affecting the pipeline."""
    try:
        manifest_path = _manifest_path(filename)
        if manifest_path is None or not manifest_path.is_file():
            return
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload["state"] = state
        if error_message:
            payload["error"] = error_message[:1000]
        _atomic_write_json(manifest_path, payload)
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn("publish recording state", error)


def publish_slice_started(filename: Path, slice_start: int, slice_stop: int) -> None:
    _publish_slice_state(filename, slice_start, slice_stop, "active")


def publish_slice_skipped(filename: Path, slice_start: int, slice_stop: int) -> None:
    _publish_slice_state(filename, slice_start, slice_stop, "skipped")


def publish_slice_failed(
    filename: Path, slice_start: int, slice_stop: int, error_message: str
) -> None:
    _publish_slice_state(
        filename,
        slice_start,
        slice_stop,
        "failed",
        error_message=error_message,
    )


def publish_slice_snapshot(
    filename: Path,
    *,
    slice_start: int,
    slice_stop: int,
    sensor_shape: tuple[int, int],
    localizations: np.ndarray,
    peak_samples: dict[str, np.ndarray],
    roi_samples: dict[str, np.ndarray],
) -> None:
    """Write one bounded snapshot, then expose its completed state atomically."""
    try:
        recording_directory = _recording_directory(filename)
        if recording_directory is None:
            return
        snapshot_path = recording_directory / "snapshots" / f"slice_{slice_stop}.npz"
        payload = {
            "schema_version": np.asarray(
                [LIVE_PROGRESS_SCHEMA_VERSION], dtype=np.uint16
            ),
            "slice_start_us": np.asarray([slice_start], dtype=np.int64),
            "slice_stop_us": np.asarray([slice_stop], dtype=np.int64),
            "localization_image": _localization_histogram(localizations, sensor_shape),
            **peak_samples,
            **roi_samples,
        }
        _atomic_save_npz(snapshot_path, payload)
        _publish_slice_state(filename, slice_start, slice_stop, "completed")
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn(f"publish live snapshot for slice {slice_stop}", error)
        _publish_slice_state(
            filename,
            slice_start,
            slice_stop,
            "completed",
            error_message="Live monitor snapshot unavailable",
        )


def sample_peak_traces(
    unique_peaks: dict,
    times: Any,
    cumulative_sums: Any,
    coordinates: Any,
) -> dict[str, np.ndarray]:
    """Copy a deterministic sample of retained peak-center cumulative traces."""
    try:
        if _progress_root() is None:
            return {}
        selected = heapq.nsmallest(
            PEAK_SAMPLE_LIMIT,
            _iter_peak_candidates(unique_peaks),
            key=lambda item: (-item[0], item[1], item[2], item[3]),
        )
        signal_lookup = _selected_signal_lookup(
            {(item[1], item[2]) for item in selected},
            times,
            cumulative_sums,
            coordinates,
        )
        retained = [item for item in selected if (item[1], item[2]) in signal_lookup]
        trace_times = np.full(
            (len(retained), PEAK_TRACE_POINT_LIMIT), np.nan, dtype=np.float64
        )
        trace_values = np.full_like(trace_times, np.nan)
        for index, item in enumerate(retained):
            signal_times, signal_values = signal_lookup[(item[1], item[2])]
            point_indices = _bounded_indices(len(signal_times), PEAK_TRACE_POINT_LIMIT)
            point_count = len(point_indices)
            trace_times[index, :point_count] = signal_times[point_indices]
            trace_values[index, :point_count] = signal_values[point_indices]
        return {
            "peak_y": np.asarray([item[1] for item in retained], dtype=np.int32),
            "peak_x": np.asarray([item[2] for item in retained], dtype=np.int32),
            "peak_time_us": np.asarray(
                [item[3] for item in retained], dtype=np.float64
            ),
            "peak_on_us": np.asarray([item[4] for item in retained], dtype=np.float64),
            "peak_off_us": np.asarray([item[5] for item in retained], dtype=np.float64),
            "peak_prominence": np.asarray(
                [item[0] for item in retained], dtype=np.float64
            ),
            "peak_trace_time_us": trace_times,
            "peak_trace_cumsum": trace_values,
        }
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn("sample cumulative peak traces", error)
        return _empty_peak_samples()


def sample_roi_events(
    rois: np.ndarray, events_by_coordinate: Any
) -> dict[str, np.ndarray]:
    """Copy exact bounded event samples for interactive ROI time-window inspection."""
    try:
        if _progress_root() is None:
            return {}
        if rois.size == 0 or rois.dtype.names is None:
            return _empty_roi_samples()
        scores = np.asarray(rois["total_events_roi"], dtype=np.uint64) + np.asarray(
            rois["total_neg_events_roi"], dtype=np.uint64
        )
        selected_indices = np.argsort(scores, kind="stable")[::-1][:ROI_SAMPLE_LIMIT]
        selected_indices = selected_indices[
            np.argsort(rois["t_peak"][selected_indices], kind="stable")
        ]
        selected_rois = rois[selected_indices]
        event_y_parts = []
        event_x_parts = []
        event_p_parts = []
        event_t_parts = []
        offsets = [0]
        for roi in selected_rois:
            event_y, event_x, event_p, event_t = _events_for_roi(
                roi, events_by_coordinate
            )
            if event_t.size > ROI_EVENT_LIMIT:
                indices = _bounded_indices(event_t.size, ROI_EVENT_LIMIT)
                event_y = event_y[indices]
                event_x = event_x[indices]
                event_p = event_p[indices]
                event_t = event_t[indices]
            event_y_parts.append(event_y)
            event_x_parts.append(event_x)
            event_p_parts.append(event_p)
            event_t_parts.append(event_t)
            offsets.append(offsets[-1] + int(event_t.size))
        return {
            "roi_center_y": np.asarray(selected_rois["peak"][:, 0], dtype=np.int32),
            "roi_center_x": np.asarray(selected_rois["peak"][:, 1], dtype=np.int32),
            "roi_t_first_us": np.asarray(selected_rois["t_1st"], dtype=np.uint64),
            "roi_t_peak_us": np.asarray(selected_rois["t_peak"], dtype=np.uint64),
            "roi_t_last_us": np.asarray(selected_rois["t_last"], dtype=np.uint64),
            "roi_positive": np.asarray(selected_rois["roi"], dtype=np.uint32),
            "roi_negative": np.asarray(selected_rois["roi_n"], dtype=np.uint32),
            "roi_event_offsets": np.asarray(offsets, dtype=np.uint32),
            "roi_event_y": _concatenate_or_empty(event_y_parts, np.int32),
            "roi_event_x": _concatenate_or_empty(event_x_parts, np.int32),
            "roi_event_p": _concatenate_or_empty(event_p_parts, np.int8),
            "roi_event_t_us": _concatenate_or_empty(event_t_parts, np.uint64),
        }
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn("sample blink ROI events", error)
        return _empty_roi_samples()


def _events_for_roi(roi: np.void, events_by_coordinate: Any) -> tuple[np.ndarray, ...]:
    roi_shape = roi["roi"].shape
    center_y, center_x = int(roi["peak"][0]), int(roi["peak"][1])
    radius_y, radius_x = roi_shape[0] // 2, roi_shape[1] // 2
    first_time, last_time = int(roi["t_1st"]), int(roi["t_last"])
    rows = []
    pixel_limit = max(ROI_EVENT_LIMIT // (roi_shape[0] * roi_shape[1]), 1)
    for y in range(center_y - radius_y, center_y + radius_y + 1):
        for x in range(center_x - radius_x, center_x + radius_x + 1):
            key = (np.int32(y), np.int32(x))
            if key not in events_by_coordinate:
                continue
            pixel_events = events_by_coordinate[key]
            pairs = (
                pixel_events.items() if hasattr(pixel_events, "items") else pixel_events
            )
            pixel_rows = []
            seen = 0
            reservoir_state = ((y & 0xFFFF) << 16) ^ (x & 0xFFFF)
            for timestamp, polarity in pairs:
                timestamp_int = int(timestamp)
                if first_time <= timestamp_int <= last_time:
                    seen += 1
                    row = (timestamp_int, y, x, int(polarity))
                    if len(pixel_rows) < pixel_limit:
                        pixel_rows.append(row)
                        continue
                    reservoir_state = (
                        reservoir_state * 1_664_525 + 1_013_904_223
                    ) & 0xFFFFFFFF
                    replacement = reservoir_state % seen
                    if replacement < pixel_limit:
                        pixel_rows[replacement] = row
            rows.extend(pixel_rows)
    rows.sort(key=lambda row: row[0])
    if not rows:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.uint64),
        )
    values = np.asarray(rows, dtype=np.int64)
    return (
        values[:, 1].astype(np.int32),
        values[:, 2].astype(np.int32),
        values[:, 3].astype(np.int8),
        values[:, 0].astype(np.uint64),
    )


def _selected_signal_lookup(
    selected_coordinates: set[tuple[int, int]],
    times: Any,
    cumulative_sums: Any,
    coordinates: Any,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    lookup = {}
    for time_chunk, sum_chunk, coordinate_chunk in zip(
        times, cumulative_sums, coordinates
    ):
        for signal_times, signal_values, coordinate in zip(
            time_chunk, sum_chunk, coordinate_chunk
        ):
            key = (int(coordinate[0]), int(coordinate[1]))
            if key not in selected_coordinates or key in lookup:
                continue
            lookup[key] = (
                np.asarray(signal_times, dtype=np.float64),
                np.asarray(signal_values, dtype=np.float64),
            )
        if len(lookup) == len(selected_coordinates):
            break
    return lookup


def _iter_peak_candidates(unique_peaks: dict) -> Any:
    for coordinate, peak_payloads in unique_peaks.items():
        y, x = int(coordinate[0]), int(coordinate[1])
        for payload in peak_payloads:
            interval = payload[2]
            yield (
                float(payload[1]),
                y,
                x,
                float(payload[0]),
                float(interval[0]),
                float(interval[1]),
            )


def _localization_histogram(
    localizations: np.ndarray, sensor_shape: tuple[int, int]
) -> np.ndarray:
    histogram = np.zeros(sensor_shape, dtype=np.uint32)
    if localizations.size == 0 or localizations.dtype.names is None:
        return histogram
    if not {"x", "y"}.issubset(localizations.dtype.names):
        return histogram
    x = np.floor(np.asarray(localizations["x"], dtype=np.float64)).astype(np.int64)
    y = np.floor(np.asarray(localizations["y"], dtype=np.float64)).astype(np.int64)
    valid = (
        np.isfinite(localizations["x"])
        & np.isfinite(localizations["y"])
        & (x >= 0)
        & (y >= 0)
        & (x < sensor_shape[1])
        & (y < sensor_shape[0])
    )
    np.add.at(histogram, (y[valid], x[valid]), 1)
    return histogram


def _publish_slice_state(
    filename: Path,
    slice_start: int,
    slice_stop: int,
    state: str,
    *,
    error_message: str | None = None,
) -> None:
    try:
        recording_directory = _recording_directory(filename)
        if recording_directory is None:
            return
        payload = {
            "schema_version": LIVE_PROGRESS_SCHEMA_VERSION,
            "slice_start_us": int(slice_start),
            "slice_stop_us": int(slice_stop),
            "state": state,
        }
        if error_message:
            payload["error"] = error_message[:1000]
        _atomic_write_json(
            recording_directory / "states" / f"slice_{slice_stop}.json", payload
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must never fail processing
        _warn(f"publish {state} slice state", error)


def _empty_peak_samples() -> dict[str, np.ndarray]:
    return {
        "peak_y": np.empty(0, dtype=np.int32),
        "peak_x": np.empty(0, dtype=np.int32),
        "peak_time_us": np.empty(0, dtype=np.float64),
        "peak_on_us": np.empty(0, dtype=np.float64),
        "peak_off_us": np.empty(0, dtype=np.float64),
        "peak_prominence": np.empty(0, dtype=np.float64),
        "peak_trace_time_us": np.empty((0, PEAK_TRACE_POINT_LIMIT)),
        "peak_trace_cumsum": np.empty((0, PEAK_TRACE_POINT_LIMIT)),
    }


def _empty_roi_samples() -> dict[str, np.ndarray]:
    return {
        "roi_center_y": np.empty(0, dtype=np.int32),
        "roi_center_x": np.empty(0, dtype=np.int32),
        "roi_t_first_us": np.empty(0, dtype=np.uint64),
        "roi_t_peak_us": np.empty(0, dtype=np.uint64),
        "roi_t_last_us": np.empty(0, dtype=np.uint64),
        "roi_positive": np.empty((0, 0, 0), dtype=np.uint32),
        "roi_negative": np.empty((0, 0, 0), dtype=np.uint32),
        "roi_event_offsets": np.asarray([0], dtype=np.uint32),
        "roi_event_y": np.empty(0, dtype=np.int32),
        "roi_event_x": np.empty(0, dtype=np.int32),
        "roi_event_p": np.empty(0, dtype=np.int8),
        "roi_event_t_us": np.empty(0, dtype=np.uint64),
    }


def _bounded_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, num=limit, dtype=np.int64)


def _concatenate_or_empty(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    non_empty = [part for part in parts if part.size]
    if not non_empty:
        return np.empty(0, dtype=dtype)
    return np.concatenate(non_empty).astype(dtype, copy=False)


def _progress_root() -> Path | None:
    raw_path = os.environ.get(LIVE_PROGRESS_ENV, "").strip()
    return Path(raw_path) if raw_path else None


def _recording_id(filename: Path) -> str:
    path_hash = hashlib.sha256(str(filename.resolve()).encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(
        character if character.isalnum() else "_" for character in filename.stem
    )
    return f"{safe_stem[:40]}_{path_hash}"


def _recording_directory(filename: Path) -> Path | None:
    root = _progress_root()
    if root is None:
        return None
    return root / "recordings" / _recording_id(filename)


def _manifest_path(filename: Path) -> Path | None:
    directory = _recording_directory(filename)
    return None if directory is None else directory / "manifest.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial_path.replace(path)


def _atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial_path.open("wb") as output_file:
        np.savez_compressed(output_file, **payload)
    partial_path.replace(path)


def _warn(action: str, error: Exception) -> None:
    logger.warning("Live monitor could not {}: {}", action, error)
