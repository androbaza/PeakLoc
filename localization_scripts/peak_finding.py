from collections import defaultdict
import numpy as np
from bisect import bisect_left
from joblib import Parallel, delayed
from numba import njit
from interpolation import interp
from csaps import CubicSmoothingSpline
from scipy.signal import find_peaks


def find_peaks_parallel(
    times: np.ndarray,
    cumsum: np.ndarray,
    coordinates: np.ndarray,
    num_cores: int,
    prominence: float,
    interpolation_coefficient: int,
    spline_smooth: int,
):
    """
    Finds the peaks of the cumulative sum of the data and returns the ON times of the peaks, the coordinates of the peaks, and the prominences of the peaks.

    Parameters
    ----------
    times : list of arrays
        Array of time points of the data.
    cumsum : list of arrays
        Array of cumulative sums of the data.
    coordinates : list of arrays
        Array of coordinates of the data.
    num_cores : int
        Number of cores to use for the multiprocessing.
    prominence : float
        Minimum prominence of the peaks.
    interpolation_coefficient : float
        Interpolation coefficient for the interpolation function.
    spline_smooth : float
        Smoothing parameter for the spline function.

    Returns
    -------
    list of arrays
        Array of ON times of the peaks.
    list of arrays
        Array of coordinates of the peaks.
    list of arrays
        Array of prominences of the peaks.
    """
    RES = Parallel(n_jobs=num_cores, backend="loky")(
        delayed(interpolate_parallel)(
            times[i],
            cumsum[i],
            coordinates[i],
            i,
            prominence,
            interpolation_coefficient,
            spline_smooth,
        )
        for i in range(len(times))
    )

    return RES

def create_peak_lists(RES):
    peaks, prominences, on_times, coordinates_peaks = [], [], [], []
    for i in np.arange(len(RES)):
        peaks.extend(RES[i][0])
        prominences.extend(RES[i][1])
        on_times.extend(RES[i][2])
        coordinates_peaks.extend(RES[i][3])
    return peaks, prominences, on_times, coordinates_peaks

def interpolate_parallel(
    times,
    cumsum,
    coordinates,
    i,
    prominence=25,
    interpolation_coefficient=3,
    cutoff_event_count=1000,
    spline_smooth=0.8,
):
    peaks, prominences, on_times, id_to_delete = [], [], [], []
    id_peak = 0
    for id in range(len(times)):
        if len(times[id]) < cutoff_event_count:
            id_to_delete.append(id)
            continue
        if (id % 1e4 == 0 or id == len(times)-1) and i == 1:
            print("completed ", int(id / len(times) * 100), " % --> ", id_peak, " peaks found in 1 of 24 slices")

        """Interpolate linearly, find peaks"""
        tnew = np.linspace(
            0,
            times[id].max(),
            num=len(times[id]) * interpolation_coefficient,
            dtype=np.uint64,
        )
        ynew = jit_interpolate(times[id], cumsum[id], tnew)
        p, p_props = find_peaks(ynew, prominence=prominence)
        id_peak += len(p)
        if len(p) == 0:
            id_to_delete.append(id)
            continue

        """Interpolate with cubic spline to find second derivative"""
        s = CubicSmoothingSpline(
            times[id], cumsum[id], smooth=spline_smooth, normalizedsmooth=True
        ).spline
        der_2 = s.derivative()(tnew)
        on_off = find_on_off(p, der_2, tnew, ynew)
        peaks.append(tnew[p])
        prominences.append(p_props["prominences"])
        on_times.append(on_off)
    try:
        coordinates = np.delete(np.asarray(coordinates), np.asarray(id_to_delete, dtype=np.uint64), axis=0)
    except:
        pass
    assert len(peaks) == len(prominences) == len(on_times) == len(coordinates), f"Length check not passed: {len(peaks)}, {len(prominences)}, {len(on_times)}, {len(coordinates)}"
    return peaks, prominences, on_times, coordinates


@njit(cache=True, fastmath=True, nogil=True)
def jit_interpolate(times, cumsum, tnew):
    return interp(times, cumsum, tnew)


@njit(cache=True, fastmath=True, nogil=True)
def find_on_off(p, der_2, tnew, ynew):
    on_off = []
    for peak in p:
        negative, positive = peak - 20, peak + 30
        # negative, positive = peak, peak
        if negative < 0:
            negative = 2
        if positive > len(der_2) - 2:
            positive = len(der_2) - 4
        while True:
            if (
                der_2[negative] < -1e-5
                or negative < 1
                or abs(der_2[negative]) < 1e-5
                or peak - negative > 200
            ):
                break
            negative -= 1
        while True:
            if (
                der_2[positive] > 1e-5
                or positive > (len(der_2) - 2)
                or abs(der_2[positive]) < 1e-5
                or positive - peak > 200
            ):
                break
            positive += 1
        if tnew[positive] - tnew[negative] > 600e3:
            negative, positive = peak - 40, peak + 50
            if negative < 0:
                negative = 2
            if positive > len(der_2) - 2:
                positive = len(der_2) - 2
        on_off.append((tnew[negative], tnew[positive]))
    return on_off


def group_timestamps_by_coordinate(
    coordinates: np.ndarray,
    time_points: np.ndarray,
    prominences: np.ndarray,
    on_times: np.ndarray,
) -> dict:
    """
    Group the time points and prominences by the given coordinates.
    """
    peaks_dict = defaultdict(list)
    for i, coord in enumerate(coordinates):
        for j, peak in enumerate(time_points[i]):
            peaks_dict[(coord[0], coord[1])].append(
                (np.squeeze(peak).tolist(), prominences[i][j].tolist(), on_times[i][j])
            )
    return peaks_dict


def find_local_max_peak(
    coords_dict: dict, threshold: float = 50e3, neighbors: int = 5
) -> dict:
    """
    Find the maximum value of a point and its neighbors in a dictionary of coordinates.

    Parameters
    ----------
    coords_dict : dict
        A dictionary with coordinates as keys and a value as values.
    threshold : int or float
        The threshold value to use to find the maximum value of points and neighbors.
    neighbors : int, optional
        The number of neighboring points to use to find the maximum value. Default: 4.

    Returns
    -------
    max_coords : tuple
        A tuple containing the coordinates and the maximum value.

    """
    # Loop through the coordinates and their data
    for coord, data in coords_dict.items():
        # Check if the coordinate is iterable
        try:
            some_object_iterator = iter(coord)
        except TypeError as te:
            print(coord, " is not iterable")
            continue
        # Unpack the coordinate
        y, x = coord
        # Create a list of the x-values for the current coordinate
        peaks_list = [x[0] for x in data]

        # Loop through the 8-neighborhood of the current coordinate
        for i in range(x - neighbors, x + neighbors + 1):
            for j in range(y - neighbors, y + neighbors + 1):
                neighbor_coord = (j, i)

                # Check if the neighbor coordinate is in coords_dict and not the current coordinate
                if neighbor_coord != coord and neighbor_coord in coords_dict:
                    # Loop through the values at the neighbor coordinate
                    for neighbor_data in coords_dict[neighbor_coord]:
                        t, v, _ = neighbor_data
                        # Check if the current coordinate has any data
                        if len(data) != 0:
                            # Find the index in the peaks_list of the closest value to t
                            index = take_closest_index(peaks_list, t)
                            # Check if the value at the neighbor coordinate is within the threshold
                            if abs(t - data[index][0]) <= threshold:
                                original_v = data[index][1]
                                # If the value at the neighbor coordinate is greater than the current max_v, update max_v and max_coord
                                if v >= original_v:
                                    # Remove the data at the index
                                    del coords_dict[coord][index]
                                    # Remove the value at the index
                                    del peaks_list[index]
    return coords_dict


def take_closest_index(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos
    if pos == len(myList):
        return pos - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos - 1
