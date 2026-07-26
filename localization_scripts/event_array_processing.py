from __future__ import annotations

import gc
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from localization_scripts.roi_generation import generate_coord_lists

if TYPE_CHECKING:
    prange = range
else:
    from numba import prange


OPENEB_SYSTEM_SITE_PACKAGES = Path("/usr/lib/python3/dist-packages")
OPENEB_WINDOWS_SITE_PACKAGES = Path(
    r"C:\Program Files\Prophesee\lib\python3\site-packages"
)
OPENEB_SITE_PACKAGES_ENV_VAR = "PEAKLOC_OPENEB_SITE_PACKAGES"
EVENT_DTYPE = np.dtype(
    [("x", np.uint16), ("y", np.uint16), ("p", np.int8), ("t", np.uint64)]
)
RAW_READ_DURATION_US = 50_000


def openeb_site_packages() -> list[Path]:
    """Return existing locations containing the installed OpenEB Python bindings."""
    configured_path = os.environ.get(OPENEB_SITE_PACKAGES_ENV_VAR)
    candidates = (
        [Path(configured_path)]
        if configured_path
        else [
            OPENEB_WINDOWS_SITE_PACKAGES
            if sys.platform == "win32"
            else OPENEB_SYSTEM_SITE_PACKAGES
        ]
    )
    return [path for path in candidates if path.is_dir()]


def add_openeb_system_site_packages() -> list[str]:
    """Temporarily expose external OpenEB bindings to the active Pixi interpreter."""
    added_paths = []
    for site_packages_path in openeb_site_packages():
        openeb_path = str(site_packages_path)
        if openeb_path not in sys.path:
            sys.path.append(openeb_path)
            added_paths.append(openeb_path)
    return added_paths


@contextmanager
def temporary_openeb_system_site_packages() -> Iterator[None]:
    """Expose OpenEB only while decoding RAW events.

    Loky copies the parent's import path into workers. Leaving the system OpenEB
    path in place lets system-only packages override Pixi dependencies there.
    """
    added_paths = add_openeb_system_site_packages()
    try:
        yield
    finally:
        for openeb_path in added_paths:
            if openeb_path in sys.path:
                sys.path.remove(openeb_path)


def raw_events_to_array(filename: str, max_events: int = 1_000_000) -> np.ndarray:
    with temporary_openeb_system_site_packages():
        RawReader = import_module("metavision_core.event_io.raw_reader").RawReader
        EventCD = import_module("metavision_sdk_base").EventCD

        record_raw = RawReader(filename, max_events=max_events)
        event_chunks = []
        while not record_raw.is_done():
            events = record_raw.load_delta_t(RAW_READ_DURATION_US)
            if events.size:
                event_chunks.append(events.copy())
        if not event_chunks:
            return np.empty(0, dtype=EventCD)
        return np.concatenate(event_chunks)


def materialize_raw_events(
    filename: str | Path,
    output_path: str | Path,
    max_events: int = 1_000_000,
    start_time_us: int = 0,
    stop_time_us: int | None = None,
) -> np.ndarray:
    """Decode a bounded RAW interval into a disk-backed normalized event array.

    RAW readers produce many chunks, but concatenating them keeps both the chunks and
    the final array alive at once. Writing each normalized chunk directly to disk keeps
    the peak allocation bounded by one reader chunk; the returned memory map is then
    consumed one processing slice at a time.
    """
    if start_time_us < 0:
        raise ValueError("start_time_us must be non-negative")
    if stop_time_us is not None and stop_time_us <= start_time_us:
        raise ValueError("stop_time_us must be greater than start_time_us")
    source_path = str(filename)
    cache_path = Path(output_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.unlink(missing_ok=True)
    event_count = 0

    with temporary_openeb_system_site_packages():
        RawReader = import_module("metavision_core.event_io.raw_reader").RawReader
        record_raw = RawReader(source_path, max_events=max_events)
        if start_time_us:
            record_raw.seek_time(start_time_us)
        with cache_path.open("wb") as output_file:
            while not record_raw.is_done() and (
                stop_time_us is None or record_raw.current_time < stop_time_us
            ):
                try:
                    events = record_raw.load_delta_t(RAW_READ_DURATION_US)
                except ValueError as error:
                    if "buffer size too small" not in str(error):
                        raise
                    raise ValueError(
                        "RAW reader buffer is too small for this recording. Increase "
                        "max_raw_events; it must exceed the peak decoder batch size."
                    ) from error
                if events.size == 0:
                    continue
                normalized_events = np.asarray(events, dtype=EVENT_DTYPE)
                within_interval = normalized_events["t"] >= start_time_us
                if stop_time_us is not None:
                    within_interval &= normalized_events["t"] < stop_time_us
                if not np.all(within_interval):
                    normalized_events = normalized_events[within_interval]
                if normalized_events.size == 0:
                    continue
                normalized_events.tofile(output_file)
                event_count += int(normalized_events.size)

    if event_count == 0:
        return np.empty(0, dtype=EVENT_DTYPE)
    return np.memmap(
        cache_path,
        dtype=EVENT_DTYPE,
        mode="r",
        shape=(event_count,),
    )


@njit(cache=True, nogil=True, fastmath=True)
def array_to_polarity_map(arr, coords):
    """
    Converts a structured NumPy ndarray with fields x, y, p, t into a dictionary with keys as (x, y) pairs and
    values as a nested dictionary with keys from p and corresponding values from t as a list for that coordinate pair.
    """
    dict_out = {}
    for id in prange(len(coords)):
        y, x = coords[id]
        key = (np.int32(y), np.int32(x))
        if key in dict_out:
            continue
        else:
            dict_out[key] = {
                0: List.empty_list(types.uint64),
                1: List.empty_list(types.uint64),
            }
    max_len = 0
    for id in prange(len(arr)):
        key = (np.int32(arr[id]["y"]), np.int32(arr[id]["x"]))
        if key not in dict_out:
            continue
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
    values as lists of (t, p) events for that coordinate pair.
    """
    dict_out = {}
    for id in prange(len(arr)):
        key = (np.int32(arr[id]["y"]), np.int32(arr[id]["x"]))
        if key not in dict_out:
            dict_out[key] = List.empty_list((np.uint64(0), np.int8(0)))
        dict_out[key].append((arr[id]["t"], np.int8(arr[id]["p"])))
    return dict_out


@njit(cache=True, nogil=True, fastmath=True)
def array_to_time_map_for_coords(arr, coords):
    """Convert only events whose coordinates belong to a sparse support mask."""
    allowed_coords = {}
    for id in prange(len(coords)):
        y, x = coords[id]
        allowed_coords[(np.int32(y), np.int32(x))] = np.int8(1)

    dict_out = {}
    for id in prange(len(arr)):
        key = (np.int32(arr[id]["y"]), np.int32(arr[id]["x"]))
        if key not in allowed_coords:
            continue
        if key not in dict_out:
            dict_out[key] = List.empty_list((np.uint64(0), np.int8(0)))
        dict_out[key].append((arr[id]["t"], np.int8(arr[id]["p"])))
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
    coords = np.empty(shape=(len(coords_split), 2), dtype=np.int32)
    for coord_pair in prange(len(coords_split)):
        coord_convolution_events = append_conv_data(
            coords_split[coord_pair], roi_rad, events_dict
        )
        if len(coord_convolution_events) == 0:
            lengths[coord_pair] = 0
            coords[coord_pair] = coords_split[coord_pair]
            continue
        coord_convolution_data = np.asarray(coord_convolution_events)
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
        times[coord_pair, : len(coord_convolution_data[:, 0])] = coord_convolution_data[
            :, 0
        ]
        cumsum[coord_pair, : len(coord_convolution_data[:, 0])] = np.cumsum(
            coord_convolution_data[:, 1]
        )
        lengths[coord_pair] = len(coord_convolution_data[:, 0])
        coords[coord_pair] = coords_split[coord_pair]
    return (
        # Slices would retain `times`/`cumsum` as their ndarray base. Those arrays
        # are padded to the busiest pixel in this chunk, so retaining views for every
        # coordinate makes a long recording accumulate mostly unused padding.
        [row[:num_relevant].copy() for row, num_relevant in zip(times, lengths)],
        [row[:num_relevant].copy() for row, num_relevant in zip(cumsum, lengths)],
        coords,
    )


# @njit(cache=True)
def create_signal(dict_events, coords, max_len, convolution_roi_radius=1):
    times, cumsum, coordinates = [], [], []
    num_coords = 24
    for start in range(0, len(coords), num_coords):
        chunk = coords[start : start + num_coords]
        output_times, output_cumsum, output_coords = process_conv_list_parallel(
            dict_events, chunk, max_len, roi_rad=convolution_roi_radius
        )
        times.extend(output_times)
        cumsum.extend(output_cumsum)
        coordinates.extend(output_coords)
    gc.collect()
    return times, cumsum, coordinates


def create_convolved_signals(
    dict_events: Dict[int, List[int]],
    coords: np.ndarray,
    max_len: int,
    num_cores: int,
    convolution_roi_radius: int = 1,
) -> List[np.ndarray]:
    """
    Create the convolved signals for the given events and coordinates.
    Cals the create_signal function that gives the data to convolve in chucks.
    Then slices the data into the given number of cores.
    """
    times, cumsum, coordinates = create_signal(
        dict_events,
        coords,
        max_len,
        convolution_roi_radius=convolution_roi_radius,
    )
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
    assert len(times) == len(cumsum) == len(coordinates), (
        f"Length check not passed: {len(times)} != {len(cumsum)} != {len(coordinates)}"
    )
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
        ind = [int(k * slice_size), int((k + 1) * slice_size)]
        data_split.append(data[ind[0] : ind[1]])
    return data_split


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di
