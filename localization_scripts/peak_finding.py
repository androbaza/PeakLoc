from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import warnings

import numpy as np
from csaps import CubicSmoothingSpline
from interpolation import interp
from joblib import Parallel, delayed
from loguru import logger
from numba import njit
from scipy.signal import find_peaks
from scipy.sparse import SparseEfficiencyWarning


@dataclass(frozen=True)
class PeakCandidate:
    coord: tuple[int, int]
    peak_index: int
    peak_time: float
    prominence: float
    event_count: float
    payload: tuple


def find_peaks_parallel(
    times: np.ndarray,
    cumsum: np.ndarray,
    coordinates: np.ndarray,
    num_cores: int,
    prominence: float,
    interpolation_coefficient: int,
    cutoff_event_count: int,
    spline_smooth: float,
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
    cutoff_event_count : int
        Minimum number of events required before attempting peak interpolation.
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
            cutoff_event_count,
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
    cutoff_event_count=2000,
    spline_smooth=0.8,
):
    peaks, prominences, on_times, id_to_delete = [], [], [], []
    id_peak = 0
    for id in range(len(times)):
        if len(times[id]) < cutoff_event_count:
            id_to_delete.append(id)
            continue
        if (id % 1e4 == 0 or id == len(times) - 1) and i == 1:
            logger.debug(
                "completed {} % --> {} peaks found in 1 of 24 slices",
                int(id / len(times) * 100),
                id_peak,
            )

        """Interpolate linearly, find peaks"""
        t, y = prepare_interpolation_axis(times[id], cumsum[id])
        if t is None:
            id_to_delete.append(id)
            continue

        num = max(int(len(t) * interpolation_coefficient), 2)
        tnew = np.linspace(
            t[0],
            t[-1],
            num=num,
            dtype=np.float64,
        )
        try:
            ynew = jit_interpolate(t, y, tnew)
        except ZeroDivisionError:
            id_to_delete.append(id)
            continue
        p, p_props = find_peaks(ynew, prominence=prominence)
        id_peak += len(p)
        if len(p) == 0:
            id_to_delete.append(id)
            continue

        """Interpolate with cubic spline to find second derivative"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            s = CubicSmoothingSpline(
                t, y, smooth=spline_smooth, normalizedsmooth=True
            ).spline
        der_2 = s.derivative()(tnew)
        on_off = find_on_off(p, der_2, tnew, ynew)
        peaks.append(tnew[p])
        prominences.append(p_props["prominences"])
        on_times.append(on_off)
    try:
        coordinates = np.delete(
            np.asarray(coordinates), np.asarray(id_to_delete, dtype=np.uint64), axis=0
        )
    except Exception:
        pass
    assert len(peaks) == len(prominences) == len(on_times) == len(coordinates), (
        f"Length check not passed: {len(peaks)}, {len(prominences)}, {len(on_times)}, {len(coordinates)}"
    )
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
    Select deterministic local maxima from candidate peaks.

    Parameters
    ----------
    coords_dict : dict
        A dictionary with coordinates as keys and candidate peak tuples as values.
    threshold : int or float
        Maximum temporal separation for candidates to be considered one peak.
    neighbors : int, optional
        Spatial radius for candidates to be considered one peak. Default: 4.

    Returns
    -------
    dict
        A new dictionary containing one winner per spatial/time peak group.
    """
    candidates = _collect_peak_candidates(coords_dict)
    if not candidates:
        return {}

    parent = list(range(len(candidates)))
    coord_index = _index_candidates_by_coordinate(candidates)

    for candidate_id, candidate in enumerate(candidates):
        y, x = candidate.coord
        min_time = candidate.peak_time - threshold
        max_time = candidate.peak_time + threshold
        for neighbor_y in range(y - neighbors, y + neighbors + 1):
            for neighbor_x in range(x - neighbors, x + neighbors + 1):
                neighbor_items = coord_index.get((neighbor_y, neighbor_x))
                if neighbor_items is None:
                    continue
                neighbor_times, neighbor_ids = neighbor_items
                start = bisect_left(neighbor_times, min_time)
                stop = bisect_right(neighbor_times, max_time)
                for neighbor_id in neighbor_ids[start:stop]:
                    if neighbor_id > candidate_id:
                        _union(parent, candidate_id, neighbor_id)

    groups = defaultdict(list)
    for candidate_id, candidate in enumerate(candidates):
        groups[_find(parent, candidate_id)].append(candidate)

    winners = [max(group, key=_candidate_rank) for group in groups.values()]
    winners.sort(key=lambda candidate: (candidate.coord, candidate.peak_time))

    output = defaultdict(list)
    for winner in winners:
        output[winner.coord].append(winner.payload)
    return dict(output)


def _collect_peak_candidates(coords_dict: dict) -> list[PeakCandidate]:
    candidates = []
    for coord in sorted(coords_dict, key=_coordinate_sort_key):
        try:
            y, x = coord
        except (TypeError, ValueError):
            logger.warning("{} is not a valid 2D coordinate", coord)
            continue
        for peak_index, peak_data in enumerate(coords_dict[coord]):
            peak_time = float(peak_data[0])
            prominence = float(peak_data[1])
            event_count = _peak_event_count(peak_data)
            candidates.append(
                PeakCandidate(
                    coord=(int(y), int(x)),
                    peak_index=peak_index,
                    peak_time=peak_time,
                    prominence=prominence,
                    event_count=event_count,
                    payload=peak_data,
                )
            )
    return candidates


def _coordinate_sort_key(coord) -> tuple[int, int, int, str]:
    try:
        y = int(coord[0])
        x = int(coord[1])
    except (TypeError, ValueError, IndexError):
        return (1, 0, 0, str(coord))
    return (0, y, x, "")


def _index_candidates_by_coordinate(
    candidates: list[PeakCandidate],
) -> dict[tuple[int, int], tuple[list[float], list[int]]]:
    coord_index = defaultdict(list)
    for candidate_id, candidate in enumerate(candidates):
        coord_index[candidate.coord].append((candidate.peak_time, candidate_id))
    for coord, items in list(coord_index.items()):
        items.sort()
        coord_index[coord] = (
            [item[0] for item in items],
            [item[1] for item in items],
        )
    return coord_index


def _peak_event_count(peak_data: tuple) -> float:
    if len(peak_data) < 4:
        return 0.0
    return float(peak_data[3])


def _candidate_rank(
    candidate: PeakCandidate,
) -> tuple[float, float, int, int, float, int]:
    y, x = candidate.coord
    return (
        candidate.prominence,
        candidate.event_count,
        -y,
        -x,
        -candidate.peak_time,
        -candidate.peak_index,
    )


def _find(parent: list[int], item: int) -> int:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: list[int], first: int, second: int) -> None:
    first_root = _find(parent, first)
    second_root = _find(parent, second)
    if first_root != second_root:
        parent[second_root] = first_root


def prepare_interpolation_axis(times, cumsum):
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(cumsum, dtype=np.float64)

    if t.size < 2:
        return None, None

    order = np.argsort(t, kind="stable")
    t = t[order]
    y = y[order]

    unique_t, first_idx, counts = np.unique(
        t,
        return_index=True,
        return_counts=True,
    )

    # For repeated timestamps, keep the final cumulative value at that timestamp.
    last_idx = first_idx + counts - 1
    unique_y = y[last_idx]

    if unique_t.size < 2:
        return None, None

    if not np.isfinite(unique_t[0]) or not np.isfinite(unique_t[-1]):
        return None, None

    if unique_t[-1] <= unique_t[0]:
        return None, None

    return unique_t, unique_y
