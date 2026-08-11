from __future__ import annotations

import numpy as np

from localization_scripts.blink_temporal_qc import _temporal_data

BASE_DTYPE = [
    ("x", np.float64),
    ("y", np.float64),
    ("t_1st", np.float64),
    ("t_peak", np.float64),
    ("t_last", np.float64),
]


def test_temporal_data_prefers_segmented_transition_boundaries() -> None:
    dtype = [
        *BASE_DTYPE,
        ("temporal_segmented", np.bool_),
        ("t_on_first", np.uint64),
        ("t_on_last", np.uint64),
        ("t_off_first", np.uint64),
        ("t_off_last", np.uint64),
        ("quiet_dwell_us", np.int64),
    ]
    localizations = np.zeros(2, dtype=dtype)
    localizations["x"] = [4.0, 8.0]
    localizations["y"] = [5.0, 9.0]
    localizations["t_1st"] = [100.0, 200.0]
    localizations["t_peak"] = [1_000.0, 2_000.0]
    localizations["t_last"] = [1_900.0, 2_900.0]
    localizations["temporal_segmented"] = [True, False]
    localizations["t_on_first"] = [900, 0]
    localizations["t_on_last"] = [980, 0]
    localizations["t_off_first"] = [1_050, 0]
    localizations["t_off_last"] = [1_100, 0]
    localizations["quiet_dwell_us"] = [70, 0]

    data = _temporal_data(localizations)

    assert data is not None
    assert data["timing_source"].tolist() == ["segmented_transition_trains"]
    assert data["x"].tolist() == [4.0]
    assert data["rise_to_peak_ms"].tolist() == [0.1]
    assert data["on_duration_ms"].tolist() == [0.07]
    assert data["peak_to_last_ms"].tolist() == [0.1]


def test_temporal_data_retains_legacy_roi_extrema_fallback() -> None:
    localizations = np.zeros(1, dtype=BASE_DTYPE)
    localizations["x"] = 4.0
    localizations["y"] = 5.0
    localizations["t_1st"] = 900.0
    localizations["t_peak"] = 1_000.0
    localizations["t_last"] = 1_300.0

    data = _temporal_data(localizations)

    assert data is not None
    assert data["timing_source"].tolist() == ["legacy_roi_extrema"]
    assert data["rise_to_peak_ms"].tolist() == [0.1]
    assert data["on_duration_ms"].tolist() == [0.4]
    assert data["peak_to_last_ms"].tolist() == [0.3]
