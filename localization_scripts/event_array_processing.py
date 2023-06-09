from numba import njit, prange, types
from numba.typed import List, Dict
import numpy as np
from localization_scripts.roi_generation import generate_coord_lists
import gc, pickle
from joblib import Parallel, delayed


def raw_events_to_array(filename):
    buffer_size = 4e9
    from metavision_core.event_io.raw_reader import RawReader

    record_raw = RawReader(filename, max_events=int(buffer_size))
    sums = 0
    while not record_raw.is_done() and record_raw.current_event_index() < buffer_size:
        events = record_raw.load_delta_t(50000)
        sums += events.size
    record_raw.reset()
    events = record_raw.load_n_events(sums)
    return events


@njit(cache=True, nogil=True, fastmath=True)
def array_to_polarity_map(arr, coords):
    """
    Converts a structured NumPy ndarray with fields x, y, p, t into a dictionary with keys as (x, y) pairs and
    values as a nested dictionary with keys from p and corresponding values from t as a list for that coordinate pair.
    """
    dict_out = {}
    for id in prange(len(coords)):
        y, x = coords[id]
        key = (y, x)
        if key in dict_out:
            continue
        else:
            dict_out[key] = {
                0: List.empty_list(types.uint64),
                1: List.empty_list(types.uint64),
            }
    max_len = 0
    for id in prange(len(arr)):
        key = (arr[id]["y"], arr[id]["x"])
        dict_out[key][arr[id]["p"]].append(arr[id]["t"])
        if len(dict_out[key][1]) > max_len:
            max_len = len(dict_out[key][1])
        if len(dict_out[key][0]) > max_len:
            max_len = len(dict_out[key][0])
    return dict_out, max_len


@njit(cache=True, nogil=True, fastmath=True)
def array_to_time_map(arr):
    """
    Converts a structured NumPy ndarray with fields x, y, p, t into a dictionary with keys as (x, y) pairs and
    values as a nested dictionary with keys from t and corresponding values from p for that coordinate pair.
    """
    dict_out = {}
    for id in prange(len(arr)):
        key = (arr[id]["y"], arr[id]["x"])
        if key in dict_out:
            dict_out[key][arr[id]["t"]] = arr[id]["p"]
        else:
            dict_out[key] = {arr[id]["t"]: arr[id]["p"]}
    return dict_out


@njit(cache=True, nogil=True)
def polarity_map_to_array(d):
    """
    Convert a dictionary of p:t key-value pairs to a NumPy ndarray with fields 't' and 'p'.
    """
    arr = List()
    for p, times in d.items():
        for t in times:
            arr.append((t, p))
    return arr


@njit(cache=True, nogil=True, fastmath=True)
def append_conv_data(coord_pair, roi_rad, events_dict):
    coord_convolution_data = []
    for y, x in generate_coord_lists(
        coord_pair[0] - roi_rad,
        coord_pair[0] + roi_rad,
        coord_pair[1] - roi_rad,
        coord_pair[1] + roi_rad,
    ):
        if (y, x) not in events_dict:
            # if (y > 190 and x > 100) and (y < 719 and x < 1279):
            #     print(y,x)
            continue
        coord_convolution_data.extend(polarity_map_to_array(events_dict[(y, x)]))
    return coord_convolution_data


@njit(cache=True, nogil=True, fastmath=True)
def check_monotonicity(lst):
    inc_indices = []
    if len(lst) == 0:
        return inc_indices
    for i in prange(1, len(lst)):
        if lst[i] <= lst[i - 1]:
            inc_indices.append(i)
    return inc_indices

# requires a lot of memory. using an awkward array instead of a numpy array might help 
@njit(parallel=True, cache=True, nogil=True)
def process_conv_list_parallel(events_dict, coords_split, max_len, roi_rad=1):
    times = np.empty(shape=(len(coords_split), max_len), dtype=np.uint64)
    cumsum = np.empty(shape=(len(coords_split), max_len), dtype=np.int32)
    lengths = np.empty(shape=(len(coords_split)), dtype=np.uint32)
    coords = np.empty(shape=(len(coords_split), 2), dtype=np.uint16)
    for coord_pair in prange(len(coords_split)):
        coord_convolution_data = np.asarray(
            append_conv_data(coords_split[coord_pair], roi_rad, events_dict)
        )
        coord_convolution_data = coord_convolution_data[
            coord_convolution_data[:, 0].argsort()
        ]
        repeating_timestamps = check_monotonicity(coord_convolution_data[:, 0])
        if len(repeating_timestamps) != 0:
            coord_convolution_data = delete_workaround_jit(
                coord_convolution_data, repeating_timestamps
            )
        coord_convolution_data[:, 1][coord_convolution_data[:, 1] == 0] = -1
        # print(coord_convolution_data.shape, times.shape)
        times[coord_pair, :len(coord_convolution_data[:, 0])] = coord_convolution_data[
            :, 0
        ]
        cumsum[coord_pair, : len(coord_convolution_data[:, 0])] = np.cumsum(
            coord_convolution_data[:, 1]
        )
        lengths[coord_pair] = len(coord_convolution_data[:, 0])
        coords[coord_pair] = coords_split[coord_pair]
    return (
        [row[:num_relevant] for row, num_relevant in zip(times, lengths)],
        [row[:num_relevant] for row, num_relevant in zip(cumsum, lengths)],
        coords,
    )


# @njit(cache=True)
def create_signal(dict_events, coords, max_len):
    times, cumsum, coordinates = [], [], []
    num_coords = 24
    for i in range(num_coords, len(coords), num_coords):
        output1, output2, output3 = process_conv_list_parallel(
            dict_events, coords[i - num_coords : i], max_len
        )
        times.extend(output1)
        cumsum.extend(output2)
        coordinates.extend(output3)
        if i + num_coords > len(coords):
            output1, output2, output3 = process_conv_list_parallel(
                dict_events, coords[i:], max_len
            )
            times.extend(output1)
            cumsum.extend(output2)
            coordinates.extend(output3)
    gc.collect()
    return times, cumsum, coordinates


def create_convolved_signals(
    dict_events: Dict[int, List[int]], coords: np.ndarray, max_len: int, num_cores: int
) -> List[np.ndarray]:
    """
    Create the convolved signals for the given events and coordinates.
    Cals the create_signal function that gives the data to convolve in chucks.
    Then slices the data into the given number of cores.
    """
    times, cumsum, coordinates = create_signal(dict_events, coords, max_len)
    ind = []
    for i in range(len(times)):
        res = check_monotonicity(times[i])
        if len(res) != 0:
            ind.append(i)
    ind = np.asarray(ind)
    if len(ind) != 0:
        times = np.delete(times, ind, axis=0)
        cumsum = np.delete(cumsum, ind, axis=0)
        # times = delete_workaround_single_col(np.asarray(times), ind)
        # cumsum = delete_workaround_single_col(np.asarray(cumsum), ind)
        coordinates = delete_workaround(np.asarray(coordinates), ind)
    gc.collect()
    assert len(times) == len(cumsum) == len(coordinates), f"Length check not passed: {len(times)} != {len(cumsum)} != {len(coordinates)}"
    return (
        slice_data(times, num_cores),
        slice_data(cumsum, num_cores),
        slice_data(coordinates, num_cores),
    )


@njit(cache=True, nogil=True, fastmath=True)
def delete_workaround_jit(arr, num):
    mask = np.zeros(arr.shape[0]) == 0
    for i in range(len(num)):
        mask[num[i]] = False
    return arr[mask, :]


def delete_workaround(arr, num):
    mask = np.zeros(arr.shape[0]) == 0
    for i in range(len(num)):
        mask[num[i]] = False
    return arr[mask, :]


# @njit(cache=True, nogil=True, fastmath=True)
def delete_lrow(arr_list, num):
    idx_list = []
    for i in range(len(arr_list)):
        if (arr_list[i] != num).all():
            idx_list.append(i)
    res_list = [arr_list[i] for i in idx_list]
    return res_list


def delete_workaround_single_col(arr, num):
    mask = np.zeros(arr.shape[0], dtype=np.int8) == 0
    for i in range(len(num)):
        mask[num[i]] = False
    return arr[mask]


def slice_data(data, nb_slices):
    slice_size = 1.0 * len(data) / nb_slices
    slice_size = np.int64(np.ceil(slice_size))
    data_split = []
    for k in np.arange(nb_slices):
        ind = [np.compat.long(k * slice_size), np.compat.long((k + 1) * slice_size)]
        data_split.append(data[ind[0] : ind[1]])
    return data_split

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di