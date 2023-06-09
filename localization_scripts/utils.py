from numba import njit, prange, jit
from numba.typed import List
import numpy as np
from collections import defaultdict

def ndarray_to_dict_t_p_python(arr):
    """
    Converts a structured NumPy ndarray with fields x, y, p, t into a dictionary with keys as (x, y) pairs and
    values as a nested dictionary with keys from t and corresponding values from p for that coordinate pair.
    """
    dict_out = {}
    for id in prange(len(arr)):
        # if id % 1e8 == 0:
        #     print('loop #: ', id, ' of ', len(arr))
        key = (arr[id]["y"], arr[id]["x"])
        if key in dict_out:
            dict_out[key][arr[id]["t"]] = arr[id]["p"]
        else:
            dict_out[key] = {arr[id]["t"]: arr[id]["p"]}
    return dict_out

# def convert_to_hashmap__parallel(events, coords):
#     num_cores = 2
#     RES = Parallel(n_jobs=num_cores)(
#         (delayed(array_to_polarity_map)(events, coords), delayed(array_to_time_map)(events, coords)))
#     return RES[0][0], RES[0][1], RES[1]

@njit(cache=True, nogil=True)
def t_p_dict_to_ndarray(d):
    """
    Convert a dictionary of t:p key-value pairs to a NumPy ndarray with fields 't' and 'p'.
    """
    arr = List()
    for t, p in d.items():
        arr.append((t, p))
    return arr


def ndarray_to_python_dict_p_t(arr, coords):
    python_dict_events = {}
    for id in prange(len(coords)):
        y, x = coords[id]
        key = (x, y)
        python_dict_events[key] = defaultdict(list)
        python_dict_events[key][0] = [0]
        python_dict_events[key][1] = [1]
    for id in prange(len(arr)):
        python_dict_events[(arr[id]["x"], arr[id]["y"])][arr[id]["p"]].append(
            arr[id]["t"]
        )
    return python_dict_events


def get_indices_in_range(coords, range):
    x_mask = np.logical_and(coords[:, 0] >= range[0], coords[:, 0] <= range[1])
    y_mask = np.logical_and(coords[:, 1] >= range[2], coords[:, 1] <= range[3])
    indices = np.where(np.logical_and(x_mask, y_mask))[0]
    return indices


def slice_coords_convolved(coordinates, times, cumsum, range=[400, 700, 500, 700]):
    coordinates = np.asarray(coordinates)
    times = np.asarray(times)
    cumsum = np.asarray(cumsum)
    indices = get_indices_in_range(coordinates, [400, 700, 500, 700])
    coord_slice = coordinates[indices, :]
    times_slice = times[indices]
    cumsum_slice = cumsum[indices]
    return coord_slice, times_slice, cumsum_slice


@njit(cache=True, nogil=True)
def subarray_lengths_histogram(arr):
    subarray_lengths = []
    for i in prange(len(arr)):
        subarray_lengths.append(len(arr[i]))
    return subarray_lengths


from csaps import CubicSmoothingSpline
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from interpolation import interp


@njit(cache=True, fastmath=True, nogil=True)
def find_on_off_plot(p, der_2, tnew, ynew):
    on_off = []
    on_off_t = []
    for peak in p:
        negative, positive = peak - 20, peak + 30
        # negative, positive = peak, peak
        if negative < 0:
            negative = 2
        if positive > len(der_2) - 2:
            positive = len(der_2) - 4
        # print('next\n\n')
        while True:
            if (
                der_2[negative] < -1e-5
                or negative < 1
                or abs(der_2[negative]) < 1e-5
                or peak - negative > 200
            ):
                # print(abs(der_2[negative]) < 1e-5, ' der')
                # print(der_2[negative] < -1e-5, ' pos der')
                # print( peak - negative > 200, ' time')
                # print(peak - negative)
                break
            negative -= 1

        while True:
            if (
                der_2[positive] > 1e-5
                or positive > (len(der_2) - 2)
                or abs(der_2[positive]) < 1e-5
                or positive - peak > 200
            ):
                # print(abs(der_2[positive]) < 1e-5, ' der')
                # print(der_2[positive] > 1e-5, ' pos der')
                # print(positive - peak > 200, ' time')
                # print(peak - positive)
                break
            positive += 1
        # print(tnew[positive] - tnew[negative] > 500e3)
        # print(tnew[positive] - tnew[negative])
        if tnew[positive] - tnew[negative] > 600e3:
            negative, positive = peak - 40, peak + 50
            if negative < 0:
                negative = 2
            if positive > len(der_2) - 2:
                positive = len(der_2) - 2
        on_off.append((tnew[negative], tnew[positive]))
        on_off_t.append((negative, positive))
    return on_off, on_off_t


import copy


@njit(cache=True, fastmath=True, nogil=True)
def split_events_dict_to_multiple_listed(input_dict, coords_dicts):
    chunks = List()
    for sublist in coords_dicts:
        curr_list = List()
        curr_list2 = List()
        curr_list3 = List()
        for key in sublist:
            curr_list.append(key)
            curr_list2.append(List(input_dict[key].keys()))
            curr_list3.append(List(input_dict[key].values()))
        chunks.append((curr_list, curr_list2, curr_list3))
    return chunks


@jit(nopython=True, fastmath=True)
def belongs(val, vec):
    res = False
    if np.where(vec == val)[0].shape[0] > 0:
        res = True
    return res


@jit(nopython=True, cache=True)
def generate_coord_lists(start_y, fin_y, start_x, fin_x):
    return np.array(
        [(y, x) for x in range(start_x, fin_x + 1) for y in range(start_y, fin_y + 1)]
    )


@jit(nopython=True, cache=True)
def generate_split_coord_lists(start_y, fin_y, start_x, fin_x, N):
    coords = np.array(
        [(y, x) for x in range(start_x, fin_x + 1) for y in range(start_y, fin_y + 1)]
    )
    sub_length = coords.shape[0] // N
    # Create a list of indices at which to split the array
    split_indices = [i * sub_length for i in range(1, N)]
    # Use numpy.split() to split the array at the calculated indices
    sub_arrays = np.split(coords, split_indices)
    # Return the list of sub-arrays
    return sub_arrays
