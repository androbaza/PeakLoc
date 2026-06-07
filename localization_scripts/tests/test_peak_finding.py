import numpy as np

from localization_scripts.peak_finding import (
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
