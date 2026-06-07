import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from numba import jit, njit, types
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed import Dict, List

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange


def generate_rois(
    unique_peaks: dict[tuple[int, int], object],
    events_t_p_dict: dict,
    roi_rad: int,
    min_x: int,
    min_y: int,
    num_cores: int,
    max_x: int,
    max_y: int,
) -> np.ndarray:
    """
    Generate ROIs from peak data.
    """
    sliced_dict = split_dict_to_multiple(unique_peaks, num_cores)
    coords_dicts = list(unique_peaks.keys())
    dict_indices = tuples_to_dict(coords_dicts)
    times_arr, polarities_arr = get_times_polarities(coords_dicts, events_t_p_dict)
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
    )


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
    num_nonzero = 0
    for value in input_dict.values():
        if value != []:
            num_nonzero += 1
    max_limit = len(list(input_dict.keys())) // num_cores + 1
    chunks = []
    curr_dict = {}
    for k, v in input_dict.items():
        if v == []:
            continue
        if len(curr_dict.keys()) < max_limit:
            curr_dict.update({k: v})
        else:
            chunks.append(copy.deepcopy(curr_dict))
            curr_dict = {k: v}
    chunks.append(curr_dict)
    return chunks


def get_times_polarities(coords_dict, events_t_p_dict):
    times_arr = []
    polarities_arr = []
    for key in coords_dict:
        event_key = (np.uint16(key[0]), np.uint16(key[1]))
        if event_key not in events_t_p_dict:
            times_arr.append([])
            polarities_arr.append([])
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            times_arr.append(list(events_t_p_dict[event_key].keys()))
            polarities_arr.append(list(events_t_p_dict[event_key].values()))
    return times_arr, polarities_arr


@jit(nopython=True, cache=True)
def generate_coord_lists(start_y, fin_y, start_x, fin_x):
    return np.array(
        [(y, x) for x in range(start_x, fin_x + 1) for y in range(start_y, fin_y + 1)],
        dtype=np.uint16,
    )


@njit(cache=True, fastmath=True)
def count_values_in_range(
    times_arr, polarities_arr, row_id, id_data, lower, upper, peak
):
    count_negative, count_positive, t_1st = 0, 0, 0
    # lower = peak - 50e3
    # upper = peak + 100e3
    ind_lower = np.searchsorted(times_arr[row_id], lower)
    ind_upper = np.searchsorted(times_arr[row_id], upper)
    if ind_lower == ind_upper:
        return count_positive, count_negative, t_1st, 0
    for i in prange(ind_lower, ind_upper):
        key = times_arr[row_id][i]
        t_last = key
        if t_1st == 0:
            t_1st = key
        if polarities_arr[row_id][i] == 0 and key > peak - 5e3:
            count_negative += 1
        elif polarities_arr[row_id][i] == 1 and key < peak + 5e3:
            count_positive += 1
    return count_positive, count_negative, t_1st, t_last


@njit(cache=True, fastmath=True, parallel=True)
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
):
    new_roi, new_roi_neg = (
        np.zeros((3, roi_rad * 2 + 1, roi_rad * 2 + 1)),
        np.zeros((roi_rad * 2 + 1, roi_rad * 2 + 1)),
    )
    total_events_roi, total_events_roi_n = 0, 0
    for id in prange(len(coord_lists)):
        y, x = coord_lists[id]
        if (y, x) not in dict_indices:  # peaks, but need other pixels too
            continue
        row_id = dict_indices[(y, x)]
        positives, negatives, t_1st, t_last = count_values_in_range(
            times_arr, polarities_arr, row_id, id_data, time_back, time_advance, t_peak
        )
        # print(positives, negatives, (y,x))
        new_roi[0, y - center_coord[0] + roi_rad, x - center_coord[1] + roi_rad] += (
            positives
        )
        new_roi[1, y - center_coord[0] + roi_rad, x - center_coord[1] + roi_rad] = t_1st
        new_roi[2, y - center_coord[0] + roi_rad, x - center_coord[1] + roi_rad] = (
            t_last
        )
        new_roi_neg[y - center_coord[0] + roi_rad, x - center_coord[1] + roi_rad] += (
            negatives
        )
        total_events_roi += positives
        total_events_roi_n += negatives
    new_roi[1:3]
    return (
        new_roi[0],
        new_roi_neg,
        new_roi[1:3],
        total_events_roi,
        total_events_roi_n,
        np.mean(new_roi[1]),
        t_peak,
        np.mean(new_roi[2]),
        center_coord,
        (center_coord[0] - image_start[0], center_coord[1] - image_start[1]),
    )


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
):
    id_data = 0
    id_loc = 0
    rois_list = []
    all = len(list(coords_dict.keys()))
    numba_dict_indices = Dict.empty(
        key_type=types.UniTuple(types.uint16, 2),
        value_type=types.int64,
    )
    for k, v in dict_indices.items():
        numba_dict_indices[(np.uint16(k[0]), np.uint16(k[1]))] = v
    numba_times = List()
    numba_polarities = List()
    for times, polarities in zip(times_arr, polarities_arr):
        numba_times.append(np.asarray(times, dtype=np.uint64))
        numba_polarities.append(np.asarray(polarities, dtype=np.int8))
    # events_t_p_dict = List(events_t_p_dict)
    for center_coord, data in coords_dict.items():
        if (id_data % 2e3 == 0 or id_data == all - 1) and i == 1:
            logger.debug(
                "completed {} % --> ~{} localizations found",
                int(id_data / all * 100),
                id_loc * 10,
            )
        y, x = center_coord
        if (
            y - roi_rad / 2 < 0
            or x - roi_rad / 2 < 0
            or y + roi_rad / 2 > max_y
            or x + roi_rad / 2 > max_x
        ):
            continue
        coord_list = generate_coord_lists(
            y - roi_rad, y + roi_rad, x - roi_rad, x + roi_rad
        )
        full_rois_list = np.zeros(
            (len(data)),
            dtype=[
                ("roi", np.uint16, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("roi_n", np.uint16, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("roi_event_times", np.uint64, (2, roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("total_events_roi", np.uint16),
                ("total_neg_events_roi", np.uint16),
                ("t_1st", np.uint64),
                ("t_peak", np.uint64),
                ("t_last", np.uint64),
                ("peak", np.uint16, (2)),
                ("rel_peak", np.uint16, (2)),
            ],
        )

        for id in prange(len(data)):
            full_rois_list[id] = slice_t_p_dict(
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
            )
            id_loc += 1
        rois_list = (
            np.concatenate((rois_list, full_rois_list))
            if id_data != 0
            else full_rois_list
        )

        id_data += 1
    return rois_list


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
):
    RES = Parallel(n_jobs=num_cores, backend="loky")(
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
        )
        for i in range(len(sliced_dict))
    )
    rois = []
    for i in np.arange(len(RES)):
        rois.extend(RES[i])
    timesorted = np.sort(rois, order="t_peak")
    return timesorted
