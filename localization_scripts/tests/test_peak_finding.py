import numpy as np

from localization_scripts.peak_finding import (
    find_local_max_peak,
    interpolate_parallel,
    prepare_interpolation_axis,
)


def test_prepare_interpolation_axis_sorts_and_compresses_duplicate_timestamps():
    times = np.asarray([3, 1, 1, 2, 2, 4], dtype=np.uint64)
    cumsum = np.asarray([30, 10, 11, 20, 21, 40], dtype=np.int32)

    t, y = prepare_interpolation_axis(times, cumsum)

    assert np.array_equal(t, np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(y, np.asarray([11.0, 21.0, 30.0, 40.0]))
    assert np.all(np.diff(t) > 0)


def test_interpolate_parallel_uses_prepared_axis_for_duplicate_timestamps():
    times = [np.asarray([0, 1, 1, 2, 3, 4, 5], dtype=np.uint64)]
    cumsum = [np.asarray([0, 1, 2, 6, 2, 1, 0], dtype=np.int32)]
    coordinates = np.asarray([[10, 20]], dtype=np.int32)

    peaks, prominences, on_times, kept_coordinates = interpolate_parallel(
        times,
        cumsum,
        coordinates,
        i=0,
        prominence=1,
        interpolation_coefficient=8,
        cutoff_event_count=2,
        spline_smooth=0.8,
    )

    assert len(peaks) == 1
    assert len(prominences) == 1
    assert len(on_times) == 1
    assert np.array_equal(kept_coordinates, coordinates)


def test_interpolate_parallel_starts_at_first_trace_timestamp():
    times = [np.asarray([100, 101, 102, 103, 104, 105], dtype=np.uint64)]
    cumsum = [np.asarray([0, 1, 6, 2, 1, 0], dtype=np.int32)]
    coordinates = np.asarray([[10, 20]], dtype=np.int32)

    peaks, _, _, kept_coordinates = interpolate_parallel(
        times,
        cumsum,
        coordinates,
        i=0,
        prominence=1,
        interpolation_coefficient=8,
        cutoff_event_count=2,
        spline_smooth=0.8,
    )

    assert len(peaks) == 1
    assert np.all(peaks[0] >= 100)
    assert np.all(peaks[0] <= 105)
    assert np.array_equal(kept_coordinates, coordinates)


def test_find_local_max_peak_does_not_mutate_input_and_chooses_highest_prominence():
    coords_dict = {
        (5, 5): [(100.0, 8.0, (80.0, 120.0))],
        (5, 6): [(105.0, 12.0, (85.0, 125.0))],
    }
    original = {coord: list(peaks) for coord, peaks in coords_dict.items()}

    result = find_local_max_peak(coords_dict, threshold=20.0, neighbors=1)

    assert coords_dict == original
    assert result == {(5, 6): [(105.0, 12.0, (85.0, 125.0))]}


def test_find_local_max_peak_breaks_prominence_ties_by_event_count_then_coordinate():
    coords_dict = {
        (4, 5): [(100.0, 10.0, (80.0, 120.0), 20)],
        (4, 4): [(100.0, 10.0, (80.0, 120.0), 20)],
        (5, 4): [(100.0, 10.0, (80.0, 120.0), 25)],
    }

    result = find_local_max_peak(coords_dict, threshold=20.0, neighbors=1)

    assert result == {(5, 4): [(100.0, 10.0, (80.0, 120.0), 25)]}


def test_find_local_max_peak_breaks_exact_ties_by_deterministic_coordinate_order():
    coords_dict = {
        (4, 5): [(100.0, 10.0, (80.0, 120.0))],
        (4, 4): [(100.0, 10.0, (80.0, 120.0))],
        (5, 4): [(100.0, 10.0, (80.0, 120.0))],
    }

    result = find_local_max_peak(coords_dict, threshold=20.0, neighbors=1)

    assert result == {(4, 4): [(100.0, 10.0, (80.0, 120.0))]}


def test_find_local_max_peak_keeps_temporally_distinct_nearby_peaks():
    coords_dict = {
        (5, 5): [(100.0, 8.0, (80.0, 120.0))],
        (5, 6): [(300.0, 12.0, (280.0, 320.0))],
    }

    result = find_local_max_peak(coords_dict, threshold=20.0, neighbors=1)

    assert result == {
        (5, 5): [(100.0, 8.0, (80.0, 120.0))],
        (5, 6): [(300.0, 12.0, (280.0, 320.0))],
    }
