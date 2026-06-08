import numpy as np

from localization_scripts.deduplication import merge_duplicate_localizations
from localization_scripts.multi_emitter import evaluate_overlap_flags


def test_merge_duplicate_localizations_prefers_accepted_lower_uncertainty_fit():
    localizations = _localizations(3)
    localizations["id"] = [0, 1, 2]
    localizations["x"] = [10.0, 10.4, 20.0]
    localizations["y"] = [10.0, 10.3, 20.0]
    localizations["t_peak"] = [100.0, 110.0, 100.0]
    localizations["accepted"] = [False, True, True]
    localizations["sigma_x"] = [0.1, 0.3, 0.2]
    localizations["sigma_y"] = [0.1, 0.3, 0.2]

    merged = merge_duplicate_localizations(
        localizations,
        spatial_radius_px=1.0,
        time_radius_us=50.0,
    )

    assert list(merged["id"]) == [1, 2]


def test_merge_duplicate_localizations_uses_event_count_as_tie_breaker():
    localizations = _localizations(2)
    localizations["id"] = [0, 1]
    localizations["x"] = [10.0, 10.2]
    localizations["y"] = [10.0, 10.2]
    localizations["t_peak"] = [100.0, 101.0]
    localizations["E_total"] = [10, 20]
    localizations["E_total_n"] = [10, 30]

    merged = merge_duplicate_localizations(
        localizations,
        spatial_radius_px=1.0,
        time_radius_us=10.0,
    )

    assert list(merged["id"]) == [1]


def test_overlap_flags_detect_two_peaks_and_roi_edge_truncation():
    localizations = _localizations(1)
    localizations["id"] = [5]
    localizations["sub_x"] = [0.5]
    localizations["sub_y"] = [2.0]
    localizations["roi"][0, 1, 1] = 10
    localizations["roi"][0, 3, 3] = 8

    flags = evaluate_overlap_flags(localizations, peak_fraction=0.5)

    assert flags["id"][0] == 5
    assert flags["possible_multi_emitter"][0]
    assert flags["edge_truncated"][0]


def _localizations(count: int) -> np.ndarray:
    localizations = np.zeros(
        count,
        dtype=[
            ("id", np.uint64),
            ("x", np.float64),
            ("y", np.float64),
            ("t_peak", np.float64),
            ("accepted", np.bool_),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("nll_per_event", np.float64),
            ("E_total", np.uint64),
            ("E_total_n", np.uint64),
            ("sub_x", np.float64),
            ("sub_y", np.float64),
            ("roi", np.uint32, (5, 5)),
            ("roi_n", np.uint32, (5, 5)),
        ],
    )
    localizations["accepted"] = True
    localizations["sigma_x"] = 0.2
    localizations["sigma_y"] = 0.2
    localizations["cov_xy"] = 0.0
    localizations["nll_per_event"] = 1.0
    localizations["E_total"] = 10
    localizations["E_total_n"] = 10
    localizations["sub_x"] = 2.0
    localizations["sub_y"] = 2.0
    return localizations
