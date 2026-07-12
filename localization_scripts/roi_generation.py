from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from numba import jit, njit, types
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed import Dict, List


@dataclass(frozen=True)
class RoiGenerationResult:
    """ROIs retained for fitting and candidates removed before ROI output."""

    rois: np.ndarray
    diffuse_flash_rejected_count: int


def generate_rois(
    unique_peaks: Mapping[tuple[int, int], object],
    events_t_p_dict: dict,
    roi_rad: int,
    min_x: int,
    min_y: int,
    num_cores: int,
    max_x: int,
    max_y: int,
    polarity_time_gate_us: float = 5e3,
    diffuse_flash_min_positive_events: int = 1_000,
    diffuse_flash_min_active_pixel_fraction: float = 0.9,
    diffuse_flash_max_local_fraction: float = 0.1,
) -> np.ndarray:
    """Generate ROIs from peak data, omitting diffuse high-event flashes."""
    return generate_rois_with_selection_stats(
        unique_peaks,
        events_t_p_dict,
        roi_rad,
        min_x,
        min_y,
        num_cores,
        max_x,
        max_y,
        polarity_time_gate_us=polarity_time_gate_us,
        diffuse_flash_min_positive_events=diffuse_flash_min_positive_events,
        diffuse_flash_min_active_pixel_fraction=(
            diffuse_flash_min_active_pixel_fraction
        ),
        diffuse_flash_max_local_fraction=diffuse_flash_max_local_fraction,
    ).rois


def generate_rois_with_selection_stats(
    unique_peaks: Mapping[tuple[int, int], object],
    events_t_p_dict: dict,
    roi_rad: int,
    min_x: int,
    min_y: int,
    num_cores: int,
    max_x: int,
    max_y: int,
    polarity_time_gate_us: float = 5e3,
    diffuse_flash_min_positive_events: int = 1_000,
    diffuse_flash_min_active_pixel_fraction: float = 0.9,
    diffuse_flash_max_local_fraction: float = 0.1,
) -> RoiGenerationResult:
    """Generate ROIs and retain the count of rejected diffuse flash candidates."""
    sliced_dict = split_dict_to_multiple(unique_peaks, num_cores)
    event_coords = list(events_t_p_dict.keys())
    dict_indices = tuples_to_dict(event_coords)
    times_arr, polarities_arr = get_times_polarities(event_coords, events_t_p_dict)
    return generate_rois_parallel(
        sliced_dict,
        num_cores,
        dict_indices,
        times_arr,
        polarities_arr,
        roi_rad,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        polarity_time_gate_us=polarity_time_gate_us,
        diffuse_flash_min_positive_events=diffuse_flash_min_positive_events,
        diffuse_flash_min_active_pixel_fraction=(
            diffuse_flash_min_active_pixel_fraction
        ),
        diffuse_flash_max_local_fraction=diffuse_flash_max_local_fraction,
    )


def roi_record_dtype(roi_rad: int):
    roi_shape = (roi_rad * 2 + 1, roi_rad * 2 + 1)
    return [
        ("roi", np.uint32, roi_shape),
        ("roi_n", np.uint32, roi_shape),
        ("roi_event_times", np.uint64, (2, *roi_shape)),
        ("total_events_roi", np.uint64),
        ("total_neg_events_roi", np.uint64),
        ("t_1st", np.uint64),
        ("t_peak", np.uint64),
        ("t_last", np.uint64),
        ("peak", np.int32, (2,)),
        ("rel_peak", np.int32, (2,)),
        ("roi_y0", np.int32),
        ("roi_x0", np.int32),
        ("dt_pos_s", np.float64),
        ("dt_neg_s", np.float64),
    ]


def get_coords_dicts(sliced_dict, num_cores):
    coords_dicts = []
    for i in range(num_cores):
        coords_dicts.extend(list(sliced_dict[i].keys()))
    return coords_dicts


def tuples_to_dict(lst):
    """
    Convert a list of (x, y) tuples to a dict {(x,y):id}
    """
    result = {}
    for i, tpl in enumerate(lst):
        result[tpl] = i
    return result


def split_dict_to_multiple(input_dict, num_cores):
    """Splits dict into multiple dicts with given maximum size.
    Returns a list of dictionaries."""
    max_limit = len(list(input_dict.keys())) // num_cores + 1
    chunks = []
    curr_dict = {}
    for k, v in input_dict.items():
        if v == []:
            continue
        if len(curr_dict.keys()) < max_limit:
            curr_dict.update({k: v})
        else:
            # ROI chunks are read-only. Copying their peak payloads multiplies the
            # per-slice memory footprint before workers even start.
            chunks.append(curr_dict)
            curr_dict = {k: v}
    chunks.append(curr_dict)
    return chunks


def get_times_polarities(coords_dict, events_t_p_dict):
    times_arr = []
    polarities_arr = []
    for key in coords_dict:
        event_key = (np.int32(key[0]), np.int32(key[1]))
        if event_key not in events_t_p_dict:
            times_arr.append([])
            polarities_arr.append([])
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            pixel_events = events_t_p_dict[event_key]
            if hasattr(pixel_events, "items"):
                event_pairs = list(pixel_events.items())
            else:
                event_pairs = list(pixel_events)
            event_pairs.sort(key=lambda event: event[0])
            times_arr.append([event[0] for event in event_pairs])
            polarities_arr.append([event[1] for event in event_pairs])
    return times_arr, polarities_arr


@jit(nopython=True, cache=True, nogil=True)
def generate_coord_lists(start_y, fin_y, start_x, fin_x):
    return np.array(
        [(y, x) for x in range(start_x, fin_x + 1) for y in range(start_y, fin_y + 1)],
        dtype=np.int32,
    )


@njit(cache=True, fastmath=True, nogil=True)
def count_values_in_range(
    times_arr,
    polarities_arr,
    row_id,
    id_data,
    lower,
    upper,
    peak,
    polarity_time_gate_us,
):
    count_negative, count_positive, t_1st, t_last = 0, 0, 0, 0
    # lower = peak - 50e3
    # upper = peak + 100e3
    ind_lower = np.searchsorted(times_arr[row_id], lower)
    ind_upper = np.searchsorted(times_arr[row_id], upper)
    if ind_lower == ind_upper:
        return count_positive, count_negative, t_1st, 0
    for i in range(ind_lower, ind_upper):
        key = times_arr[row_id][i]
        t_last = key
        if t_1st == 0:
            t_1st = key
        if polarities_arr[row_id][i] == 0 and key > peak - polarity_time_gate_us:
            count_negative += 1
        elif polarities_arr[row_id][i] == 1 and key < peak + polarity_time_gate_us:
            count_positive += 1
    return count_positive, count_negative, t_1st, t_last


@njit(cache=True, fastmath=True, nogil=True)
def slice_t_p_dict(
    dict_indices,
    times_arr,
    polarities_arr,
    id_data,
    time_back,
    time_advance,
    t_peak,
    coord_lists,
    center_coord,
    roi_rad,
    image_start,
    polarity_time_gate_us,
):
    roi_shape = (roi_rad * 2 + 1, roi_rad * 2 + 1)
    new_roi = np.zeros(roi_shape, dtype=np.uint32)
    new_roi_neg = np.zeros(roi_shape, dtype=np.uint32)
    roi_event_times = np.zeros((2, roi_rad * 2 + 1, roi_rad * 2 + 1), dtype=np.uint64)
    total_events_roi, total_events_roi_n = 0, 0
    t_first_roi, t_last_roi = 0, 0

    # The spline-derived ON/OFF interval can be very narrow. Use it as timing
    # metadata, but expand the actual ROI event-counting window with the
    # configured polarity gate so the positive and negative lobes of the same
    # blink can both enter the joint Poisson fit.
    peak_minus_gate = t_peak - polarity_time_gate_us
    peak_plus_gate = t_peak + polarity_time_gate_us

    count_lower = time_back
    if peak_minus_gate < count_lower:
        count_lower = peak_minus_gate
    if count_lower < 0:
        count_lower = 0

    count_upper = time_advance
    if peak_plus_gate > count_upper:
        count_upper = peak_plus_gate

    for id in range(len(coord_lists)):
        y, x = coord_lists[id]
        if (y, x) not in dict_indices:
            continue
        row_id = dict_indices[(y, x)]
        positives, negatives, t_1st, t_last = count_values_in_range(
            times_arr,
            polarities_arr,
            row_id,
            id_data,
            count_lower,
            count_upper,
            t_peak,
            polarity_time_gate_us,
        )
        # print(positives, negatives, (y,x))
        roi_y = y - center_coord[0] + roi_rad
        roi_x = x - center_coord[1] + roi_rad
        new_roi[roi_y, roi_x] += positives
        roi_event_times[0, roi_y, roi_x] = t_1st
        roi_event_times[1, roi_y, roi_x] = t_last
        new_roi_neg[y - center_coord[0] + roi_rad, x - center_coord[1] + roi_rad] += (
            negatives
        )
        total_events_roi += positives
        total_events_roi_n += negatives
        if t_1st > 0 and (t_first_roi == 0 or t_1st < t_first_roi):
            t_first_roi = t_1st
        if t_last > t_last_roi:
            t_last_roi = t_last
    roi_y0 = center_coord[0] - roi_rad
    roi_x0 = center_coord[1] - roi_rad
    pos_end = count_upper
    if peak_plus_gate < pos_end:
        pos_end = peak_plus_gate

    neg_start = count_lower
    if peak_minus_gate > neg_start:
        neg_start = peak_minus_gate

    dt_pos_s = max(pos_end - count_lower, 0)
    dt_neg_s = max(count_upper - neg_start, 0)
    return (
        new_roi,
        new_roi_neg,
        roi_event_times,
        total_events_roi,
        total_events_roi_n,
        t_first_roi,
        t_peak,
        t_last_roi,
        center_coord,
        (center_coord[0] - image_start[0], center_coord[1] - image_start[1]),
        roi_y0,
        roi_x0,
        dt_pos_s * 1e-6,
        dt_neg_s * 1e-6,
    )


@njit(cache=True, fastmath=True, nogil=True)
def is_diffuse_flash_roi(
    positive_roi,
    total_positive_events,
    min_positive_events,
    min_active_pixel_fraction,
    max_local_fraction,
):
    """Identify broad illumination flashes before their ROI is retained.

    PeakLoc detects an ON peak, so only its positive-event image is relevant here.
    A single-molecule blink has a compact local maximum, while focus-light changes
    produce a high-count image with nearly every ROI pixel active and no local core.
    """
    if (
        min_positive_events <= 0
        or total_positive_events < min_positive_events
        or positive_roi.shape[0] < 3
    ):
        return False

    active_pixels = 0
    for y in range(positive_roi.shape[0]):
        for x in range(positive_roi.shape[1]):
            if positive_roi[y, x] > 0:
                active_pixels += 1
    active_fraction = active_pixels / positive_roi.size
    if active_fraction < min_active_pixel_fraction:
        return False

    max_local_events = 0
    for y in range(positive_roi.shape[0] - 2):
        for x in range(positive_roi.shape[1] - 2):
            local_events = 0
            for local_y in range(y, y + 3):
                for local_x in range(x, x + 3):
                    local_events += positive_roi[local_y, local_x]
            if local_events > max_local_events:
                max_local_events = local_events

    return max_local_events / total_positive_events <= max_local_fraction


def gen_rois_from_peaks_dict(
    coords_dict,
    dict_indices,
    times_arr,
    polarities_arr,
    max_x,
    max_y,
    roi_rad=5,
    image_start=(0, 0),
    i=1,
    polarity_time_gate_us=5e3,
    diffuse_flash_min_positive_events=1_000,
    diffuse_flash_min_active_pixel_fraction=0.9,
    diffuse_flash_max_local_fraction=0.1,
):
    id_data = 0
    id_loc = 0
    roi_chunks = []
    diffuse_flash_rejected_count = 0
    total_coordinates = len(list(coords_dict.keys()))
    numba_dict_indices = Dict.empty(
        key_type=types.UniTuple(types.int32, 2),
        value_type=types.int64,
    )
    for k, v in dict_indices.items():
        numba_dict_indices[(np.int32(k[0]), np.int32(k[1]))] = v
    numba_times = List()
    numba_polarities = List()
    for times, polarities in zip(times_arr, polarities_arr):
        numba_times.append(np.asarray(times, dtype=np.uint64))
        numba_polarities.append(np.asarray(polarities, dtype=np.int8))
    # events_t_p_dict = List(events_t_p_dict)
    for center_coord, data in coords_dict.items():
        if (id_data % 2e3 == 0 or id_data == total_coordinates - 1) and i == 1:
            logger.debug(
                "completed {} % --> ~{} localizations found",
                int(id_data / total_coordinates * 100),
                id_loc * 10,
            )
        y, x = center_coord
        if (
            y - roi_rad < image_start[0]
            or x - roi_rad < image_start[1]
            or y + roi_rad > max_y
            or x + roi_rad > max_x
        ):
            continue
        coord_list = generate_coord_lists(
            y - roi_rad, y + roi_rad, x - roi_rad, x + roi_rad
        )
        full_rois_list = np.empty(
            (len(data)),
            dtype=roi_record_dtype(roi_rad),
        )
        retained_count = 0

        for id in range(len(data)):
            roi_data = slice_t_p_dict(
                numba_dict_indices,
                numba_times,
                numba_polarities,
                id_data,
                time_back=data[id][2][0],
                time_advance=data[id][2][1],
                t_peak=data[id][0],
                coord_lists=coord_list,
                center_coord=center_coord,
                roi_rad=roi_rad,
                image_start=image_start,
                polarity_time_gate_us=polarity_time_gate_us,
            )
            if is_diffuse_flash_roi(
                roi_data[0],
                roi_data[3],
                diffuse_flash_min_positive_events,
                diffuse_flash_min_active_pixel_fraction,
                diffuse_flash_max_local_fraction,
            ):
                diffuse_flash_rejected_count += 1
                continue
            full_rois_list[retained_count] = roi_data
            retained_count += 1
            id_loc += 1
        if retained_count:
            roi_chunks.append(full_rois_list[:retained_count])

        id_data += 1
    if not roi_chunks:
        return RoiGenerationResult(
            rois=np.empty(0, dtype=roi_record_dtype(roi_rad)),
            diffuse_flash_rejected_count=diffuse_flash_rejected_count,
        )
    return RoiGenerationResult(
        rois=np.concatenate(roi_chunks),
        diffuse_flash_rejected_count=diffuse_flash_rejected_count,
    )


def generate_rois_parallel(
    sliced_dict,
    num_cores,
    dict_indices,
    times_arr,
    polarities_arr,
    roi_rad,
    min_x,
    min_y,
    max_x,
    max_y,
    polarity_time_gate_us=5e3,
    diffuse_flash_min_positive_events=1_000,
    diffuse_flash_min_active_pixel_fraction=0.9,
    diffuse_flash_max_local_fraction=0.1,
):
    RES = Parallel(
        n_jobs=num_cores,
        backend="threading",
        pre_dispatch="n_jobs",
        batch_size=1,
    )(
        delayed(gen_rois_from_peaks_dict)(
            coords_dict=sliced_dict[i],
            dict_indices=dict_indices,
            times_arr=times_arr,
            polarities_arr=polarities_arr,
            i=i,
            roi_rad=roi_rad,
            image_start=(min_y, min_x),
            max_x=max_x,
            max_y=max_y,
            polarity_time_gate_us=polarity_time_gate_us,
            diffuse_flash_min_positive_events=diffuse_flash_min_positive_events,
            diffuse_flash_min_active_pixel_fraction=(
                diffuse_flash_min_active_pixel_fraction
            ),
            diffuse_flash_max_local_fraction=diffuse_flash_max_local_fraction,
        )
        for i in range(len(sliced_dict))
    )
    diffuse_flash_rejected_count = sum(
        result.diffuse_flash_rejected_count for result in RES
    )
    non_empty_results = [result.rois for result in RES if result.rois.size]
    if not non_empty_results:
        return RoiGenerationResult(
            rois=np.empty(0, dtype=roi_record_dtype(roi_rad)),
            diffuse_flash_rejected_count=diffuse_flash_rejected_count,
        )

    rois = np.concatenate(non_empty_results)
    return RoiGenerationResult(
        rois=np.sort(rois, order="t_peak"),
        diffuse_flash_rejected_count=diffuse_flash_rejected_count,
    )
