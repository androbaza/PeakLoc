from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from localization_scripts.event_array_processing import EVENT_DTYPE
from localization_scripts.roi_generation import get_times_polarities, roi_record_dtype
from localization_scripts.python_compat import strict_zip
from localization_scripts.temporal_segmentation import (
    BlinkInterval,
    SegmentationResult,
    TemporalSegmentationSettings,
    segment_candidate_events,
)

DETECTION_REPLAY_TIME_BIN_COUNT = 128


@dataclass(frozen=True)
class TemporalRoiGenerationResult:
    rois: np.ndarray
    qc: np.ndarray
    elapsed_seconds: float

    @property
    def accepted_count(self) -> int:
        return int(self.rois.size)

    @property
    def rejected_count(self) -> int:
        return int(np.count_nonzero(~self.qc["accepted"])) if self.qc.size else 0


def temporal_roi_record_dtype(roi_radius: int) -> list[tuple]:
    return [
        *roi_record_dtype(roi_radius),
        (
            "roi_event_histogram",
            np.uint32,
            (2, DETECTION_REPLAY_TIME_BIN_COUNT),
        ),
        ("roi_event_histogram_start_us", np.uint64),
        ("roi_event_histogram_bin_us", np.uint32),
        ("temporal_segmented", np.bool_),
        ("segmentation_id", np.uint64),
        ("parent_seed_peak_us", np.uint64),
        ("parent_seed_y", np.int32),
        ("parent_seed_x", np.int32),
        ("t_on_window_start", np.uint64),
        ("t_on_window_stop", np.uint64),
        ("t_off_window_start", np.uint64),
        ("t_off_window_stop", np.uint64),
        ("t_on_first", np.uint64),
        ("t_on_last", np.uint64),
        ("t_off_first", np.uint64),
        ("t_off_last", np.uint64),
        ("on_core_events", np.uint32),
        ("off_core_events", np.uint32),
        ("on_active_pixels", np.uint16),
        ("off_active_pixels", np.uint16),
        ("on_centroid_x", np.float64),
        ("on_centroid_y", np.float64),
        ("off_centroid_x", np.float64),
        ("off_centroid_y", np.float64),
        ("temporal_center_x", np.float64),
        ("temporal_center_y", np.float64),
        ("on_core_density_ratio", np.float64),
        ("off_core_density_ratio", np.float64),
        ("on_interval_deviance", np.float64),
        ("off_interval_deviance", np.float64),
        ("on_polarity_purity", np.float64),
        ("off_polarity_purity", np.float64),
        ("pair_centroid_distance_px", np.float64),
        ("pair_score", np.float64),
        ("quiet_dwell_us", np.int64),
        ("endpoint_overlap_us", np.uint64),
        ("cycle_span_us", np.uint64),
        ("segmentation_iteration_count", np.uint8),
    ]


def temporal_segmentation_qc_dtype() -> list[tuple]:
    return [
        ("id", np.uint64),
        ("candidate_id", np.uint64),
        ("accepted", np.bool_),
        ("rejection_reason", "U64"),
        ("parent_seed_peak_us", np.uint64),
        ("parent_seed_y", np.int32),
        ("parent_seed_x", np.int32),
        ("context_event_count", np.uint32),
        ("provisional_on_train_count", np.uint16),
        ("provisional_off_train_count", np.uint16),
        ("refined_on_train_count", np.uint16),
        ("refined_off_train_count", np.uint16),
        ("t_on_window_start", np.uint64),
        ("t_on_window_stop", np.uint64),
        ("t_off_window_start", np.uint64),
        ("t_off_window_stop", np.uint64),
        ("on_core_events", np.uint32),
        ("off_core_events", np.uint32),
        ("on_active_pixels", np.uint16),
        ("off_active_pixels", np.uint16),
        ("on_core_density_ratio", np.float64),
        ("off_core_density_ratio", np.float64),
        ("pair_centroid_distance_px", np.float64),
        ("pair_score", np.float64),
        ("cycle_span_us", np.uint64),
    ]


def generate_temporally_segmented_rois(
    unique_peaks: Mapping[tuple[int, int], Iterable[Sequence[Any]]],
    events_t_p_dict: dict,
    *,
    roi_radius: int,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    slice_start_us: int,
    slice_stop_us: int,
    settings: TemporalSegmentationSettings | None = None,
) -> TemporalRoiGenerationResult:
    """Generate full rectangular fit ROIs from compact temporal transition trains."""
    started = time.perf_counter()
    settings = settings or TemporalSegmentationSettings()
    settings.validate()
    index = _build_event_index(events_t_p_dict)
    roi_chunks = []
    qc_rows = []
    candidate_id = 0
    segmentation_id = 0
    for (seed_y, seed_x), peak_payloads in sorted(unique_peaks.items()):
        for peak_payload in peak_payloads:
            seed_peak_us = int(round(float(peak_payload[0])))
            qc_row = np.zeros(1, dtype=temporal_segmentation_qc_dtype())
            qc_row["id"] = candidate_id
            qc_row["candidate_id"] = candidate_id
            qc_row["parent_seed_peak_us"] = seed_peak_us
            qc_row["parent_seed_y"] = int(seed_y)
            qc_row["parent_seed_x"] = int(seed_x)
            candidate_id += 1
            edge_reason = _candidate_edge_reason(
                seed_y=int(seed_y),
                seed_x=int(seed_x),
                seed_peak_us=seed_peak_us,
                roi_radius=roi_radius,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                slice_start_us=slice_start_us,
                slice_stop_us=slice_stop_us,
                settings=settings,
            )
            if edge_reason is not None:
                qc_row["rejection_reason"] = edge_reason
                qc_rows.append(qc_row)
                continue
            arrays = _extract_candidate_event_arrays(
                index.dict_indices,
                index.times,
                index.polarities,
                int(seed_y),
                int(seed_x),
                roi_radius,
                seed_peak_us - settings.context_pre_us,
                seed_peak_us + settings.context_post_us,
            )
            events = _structured_events(*arrays)
            qc_row["context_event_count"] = events.size
            segmentation = segment_candidate_events(
                events,
                seed_peak_us=seed_peak_us,
                seed_x=float(seed_x),
                seed_y=float(seed_y),
                roi_radius_px=roi_radius,
                settings=settings,
            )
            _fill_segmentation_qc(qc_row, segmentation)
            if not segmentation.accepted:
                qc_rows.append(qc_row)
                continue
            interval = segmentation.interval
            if interval is None:
                raise RuntimeError("Accepted temporal segmentation has no interval")
            center_y = int(round(interval.refined_y))
            center_x = int(round(interval.refined_x))
            if (
                center_y - roi_radius < min_y
                or center_x - roi_radius < min_x
                or center_y + roi_radius > max_y
                or center_x + roi_radius > max_x
            ):
                qc_row["accepted"] = False
                qc_row["rejection_reason"] = "refined_roi_out_of_bounds"
                qc_rows.append(qc_row)
                continue
            histogram_start_us = seed_peak_us - settings.context_pre_us
            histogram_bin_us = int(
                np.ceil(
                    (settings.context_pre_us + settings.context_post_us)
                    / DETECTION_REPLAY_TIME_BIN_COUNT
                )
            )
            base_values = _materialize_segmented_roi(
                index.dict_indices,
                index.times,
                index.polarities,
                center_y,
                center_x,
                roi_radius,
                interval.on_train.support_start_us,
                interval.on_train.support_stop_us,
                interval.off_train.support_start_us,
                interval.off_train.support_stop_us,
                histogram_start_us,
                histogram_bin_us,
                DETECTION_REPLAY_TIME_BIN_COUNT,
            )
            roi = _build_temporal_roi_record(
                base_values,
                interval=interval,
                segmentation_id=segmentation_id,
                seed_y=int(seed_y),
                seed_x=int(seed_x),
                center_y=center_y,
                center_x=center_x,
                image_start=(min_y, min_x),
                roi_radius=roi_radius,
                histogram_start_us=histogram_start_us,
                histogram_bin_us=histogram_bin_us,
            )
            segmentation_id += 1
            roi_chunks.append(roi)
            qc_rows.append(qc_row)

    rois = (
        np.concatenate(roi_chunks)
        if roi_chunks
        else np.empty(0, dtype=temporal_roi_record_dtype(roi_radius))
    )
    if rois.size:
        rois = np.sort(rois, order="t_peak")
    qc = (
        np.concatenate(qc_rows)
        if qc_rows
        else np.empty(0, dtype=temporal_segmentation_qc_dtype())
    )
    return TemporalRoiGenerationResult(
        rois=rois,
        qc=qc,
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class _NumbaEventIndex:
    dict_indices: object
    times: object
    polarities: object


def _build_event_index(events_t_p_dict: dict) -> _NumbaEventIndex:
    event_coords = list(events_t_p_dict)
    times, polarities = get_times_polarities(event_coords, events_t_p_dict)
    dict_indices = Dict.empty(
        key_type=types.UniTuple(types.int64, 2),
        value_type=types.int64,
    )
    numba_times = List.empty_list(types.uint64[::1])
    numba_polarities = List.empty_list(types.int8[::1])
    for index, (coord, pixel_times, pixel_polarities) in enumerate(
        strict_zip(event_coords, times, polarities)
    ):
        dict_indices[(int(coord[0]), int(coord[1]))] = index
        numba_times.append(np.asarray(pixel_times, dtype=np.uint64))
        numba_polarities.append(np.asarray(pixel_polarities, dtype=np.int8))
    return _NumbaEventIndex(dict_indices, numba_times, numba_polarities)


@njit(cache=True, nogil=True)
def _extract_candidate_event_arrays(
    dict_indices,
    times,
    polarities,
    center_y,
    center_x,
    roi_radius,
    context_start_us,
    context_stop_us,
):
    count = 0
    for y in range(center_y - roi_radius, center_y + roi_radius + 1):
        for x in range(center_x - roi_radius, center_x + roi_radius + 1):
            if (y, x) not in dict_indices:
                continue
            row = dict_indices[(y, x)]
            start = np.searchsorted(times[row], context_start_us)
            stop = np.searchsorted(times[row], context_stop_us)
            count += stop - start
    event_x = np.empty(count, dtype=np.uint16)
    event_y = np.empty(count, dtype=np.uint16)
    event_p = np.empty(count, dtype=np.int8)
    event_t = np.empty(count, dtype=np.uint64)
    output_index = 0
    for y in range(center_y - roi_radius, center_y + roi_radius + 1):
        for x in range(center_x - roi_radius, center_x + roi_radius + 1):
            if (y, x) not in dict_indices:
                continue
            row = dict_indices[(y, x)]
            start = np.searchsorted(times[row], context_start_us)
            stop = np.searchsorted(times[row], context_stop_us)
            for event_index in range(start, stop):
                event_x[output_index] = x
                event_y[output_index] = y
                event_p[output_index] = polarities[row][event_index]
                event_t[output_index] = times[row][event_index]
                output_index += 1
    return event_x, event_y, event_p, event_t


def _structured_events(
    event_x: np.ndarray,
    event_y: np.ndarray,
    event_p: np.ndarray,
    event_t: np.ndarray,
) -> np.ndarray:
    events = np.empty(event_t.size, dtype=EVENT_DTYPE)
    events["x"] = event_x
    events["y"] = event_y
    events["p"] = event_p
    events["t"] = event_t
    return events


@njit(cache=True, nogil=True)
def _materialize_segmented_roi(
    dict_indices,
    times,
    polarities,
    center_y,
    center_x,
    roi_radius,
    on_start_us,
    on_stop_us,
    off_start_us,
    off_stop_us,
    histogram_start_us,
    histogram_bin_us,
    histogram_bin_count,
):
    roi_side = roi_radius * 2 + 1
    roi_positive = np.zeros((roi_side, roi_side), dtype=np.uint32)
    roi_negative = np.zeros((roi_side, roi_side), dtype=np.uint32)
    roi_event_times = np.zeros((2, roi_side, roi_side), dtype=np.uint64)
    roi_event_histogram = np.zeros((2, histogram_bin_count), dtype=np.uint32)
    total_positive = 0
    total_negative = 0
    first_event = 0
    last_event = 0
    for y in range(center_y - roi_radius, center_y + roi_radius + 1):
        for x in range(center_x - roi_radius, center_x + roi_radius + 1):
            if (y, x) not in dict_indices:
                continue
            row = dict_indices[(y, x)]
            first_pixel_event = 0
            last_pixel_event = 0
            on_start = np.searchsorted(times[row], on_start_us)
            on_stop = np.searchsorted(times[row], on_stop_us)
            for event_index in range(on_start, on_stop):
                if polarities[row][event_index] != 1:
                    continue
                timestamp = times[row][event_index]
                roi_positive[y - center_y + roi_radius, x - center_x + roi_radius] += 1
                histogram_index = (timestamp - histogram_start_us) // histogram_bin_us
                if 0 <= histogram_index < histogram_bin_count:
                    roi_event_histogram[0, histogram_index] += 1
                total_positive += 1
                if first_pixel_event == 0 or timestamp < first_pixel_event:
                    first_pixel_event = timestamp
                if timestamp > last_pixel_event:
                    last_pixel_event = timestamp
            off_start = np.searchsorted(times[row], off_start_us)
            off_stop = np.searchsorted(times[row], off_stop_us)
            for event_index in range(off_start, off_stop):
                if polarities[row][event_index] != 0:
                    continue
                timestamp = times[row][event_index]
                roi_negative[y - center_y + roi_radius, x - center_x + roi_radius] += 1
                histogram_index = (timestamp - histogram_start_us) // histogram_bin_us
                if 0 <= histogram_index < histogram_bin_count:
                    roi_event_histogram[1, histogram_index] += 1
                total_negative += 1
                if first_pixel_event == 0 or timestamp < first_pixel_event:
                    first_pixel_event = timestamp
                if timestamp > last_pixel_event:
                    last_pixel_event = timestamp
            roi_y = y - center_y + roi_radius
            roi_x = x - center_x + roi_radius
            roi_event_times[0, roi_y, roi_x] = first_pixel_event
            roi_event_times[1, roi_y, roi_x] = last_pixel_event
            if first_pixel_event > 0 and (
                first_event == 0 or first_pixel_event < first_event
            ):
                first_event = first_pixel_event
            if last_pixel_event > last_event:
                last_event = last_pixel_event
    return (
        roi_positive,
        roi_negative,
        roi_event_times,
        roi_event_histogram,
        total_positive,
        total_negative,
        first_event,
        last_event,
    )


def _build_temporal_roi_record(
    base_values: tuple,
    *,
    interval: BlinkInterval,
    segmentation_id: int,
    seed_y: int,
    seed_x: int,
    center_y: int,
    center_x: int,
    image_start: tuple[int, int],
    roi_radius: int,
    histogram_start_us: int,
    histogram_bin_us: int,
) -> np.ndarray:
    (
        roi_positive,
        roi_negative,
        roi_event_times,
        roi_event_histogram,
        total_positive,
        total_negative,
        first_event,
        last_event,
    ) = base_values
    record = np.zeros(1, dtype=temporal_roi_record_dtype(roi_radius))
    record["roi"] = roi_positive
    record["roi_n"] = roi_negative
    record["roi_event_times"] = roi_event_times
    record["roi_event_histogram"] = roi_event_histogram
    record["roi_event_histogram_start_us"] = histogram_start_us
    record["roi_event_histogram_bin_us"] = histogram_bin_us
    record["total_events_roi"] = total_positive
    record["total_neg_events_roi"] = total_negative
    record["t_1st"] = first_event
    record["t_peak"] = interval.cycle_peak_us
    record["t_last"] = last_event
    record["peak"] = (center_y, center_x)
    record["rel_peak"] = (center_y - image_start[0], center_x - image_start[1])
    record["roi_y0"] = center_y - roi_radius
    record["roi_x0"] = center_x - roi_radius
    record["dt_pos_s"] = (
        interval.on_train.support_stop_us - interval.on_train.support_start_us
    ) * 1e-6
    record["dt_neg_s"] = (
        interval.off_train.support_stop_us - interval.off_train.support_start_us
    ) * 1e-6
    record["temporal_segmented"] = True
    record["segmentation_id"] = segmentation_id
    record["parent_seed_peak_us"] = interval.parent_seed_peak_us
    record["parent_seed_y"] = seed_y
    record["parent_seed_x"] = seed_x
    record["t_on_window_start"] = interval.on_train.support_start_us
    record["t_on_window_stop"] = interval.on_train.support_stop_us
    record["t_off_window_start"] = interval.off_train.support_start_us
    record["t_off_window_stop"] = interval.off_train.support_stop_us
    record["t_on_first"] = interval.on_train.first_event_us
    record["t_on_last"] = interval.on_train.last_event_us
    record["t_off_first"] = interval.off_train.first_event_us
    record["t_off_last"] = interval.off_train.last_event_us
    record["on_core_events"] = interval.on_train.event_count
    record["off_core_events"] = interval.off_train.event_count
    record["on_active_pixels"] = interval.on_train.active_pixel_count
    record["off_active_pixels"] = interval.off_train.active_pixel_count
    record["on_centroid_x"] = interval.on_train.centroid_x
    record["on_centroid_y"] = interval.on_train.centroid_y
    record["off_centroid_x"] = interval.off_train.centroid_x
    record["off_centroid_y"] = interval.off_train.centroid_y
    record["temporal_center_x"] = interval.refined_x
    record["temporal_center_y"] = interval.refined_y
    record["on_core_density_ratio"] = interval.on_train.core_density_ratio
    record["off_core_density_ratio"] = interval.off_train.core_density_ratio
    record["on_interval_deviance"] = interval.on_train.interval_deviance
    record["off_interval_deviance"] = interval.off_train.interval_deviance
    record["on_polarity_purity"] = interval.on_train.polarity_purity
    record["off_polarity_purity"] = interval.off_train.polarity_purity
    record["pair_centroid_distance_px"] = interval.centroid_distance_px
    record["pair_score"] = interval.pair_score
    record["quiet_dwell_us"] = interval.quiet_dwell_us
    record["endpoint_overlap_us"] = interval.endpoint_overlap_us
    record["cycle_span_us"] = interval.cycle_span_us
    record["segmentation_iteration_count"] = interval.iteration_count
    return record


def _fill_segmentation_qc(
    row: np.ndarray,
    segmentation: SegmentationResult,
) -> None:
    row["accepted"] = segmentation.accepted
    row["rejection_reason"] = segmentation.rejection_reason
    row["provisional_on_train_count"] = len(segmentation.provisional_on_trains)
    row["provisional_off_train_count"] = len(segmentation.provisional_off_trains)
    row["refined_on_train_count"] = len(segmentation.refined_on_trains)
    row["refined_off_train_count"] = len(segmentation.refined_off_trains)
    interval = segmentation.interval
    if interval is None:
        return
    row["t_on_window_start"] = interval.on_train.support_start_us
    row["t_on_window_stop"] = interval.on_train.support_stop_us
    row["t_off_window_start"] = interval.off_train.support_start_us
    row["t_off_window_stop"] = interval.off_train.support_stop_us
    row["on_core_events"] = interval.on_train.event_count
    row["off_core_events"] = interval.off_train.event_count
    row["on_active_pixels"] = interval.on_train.active_pixel_count
    row["off_active_pixels"] = interval.off_train.active_pixel_count
    row["on_core_density_ratio"] = interval.on_train.core_density_ratio
    row["off_core_density_ratio"] = interval.off_train.core_density_ratio
    row["pair_centroid_distance_px"] = interval.centroid_distance_px
    row["pair_score"] = interval.pair_score
    row["cycle_span_us"] = interval.cycle_span_us


def _candidate_edge_reason(
    *,
    seed_y: int,
    seed_x: int,
    seed_peak_us: int,
    roi_radius: int,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    slice_start_us: int,
    slice_stop_us: int,
    settings: TemporalSegmentationSettings,
) -> str | None:
    if (
        seed_y - roi_radius < min_y
        or seed_x - roi_radius < min_x
        or seed_y + roi_radius > max_y
        or seed_x + roi_radius > max_x
    ):
        return "seed_roi_out_of_bounds"
    if seed_peak_us - settings.context_pre_us < slice_start_us:
        return "context_before_slice"
    if seed_peak_us + settings.context_post_us > slice_stop_us:
        return "context_after_slice"
    return None
