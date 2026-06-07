import numpy as np
import pytest

from localization_scripts.roi_generation import generate_coord_lists, generate_rois


def test_generate_coord_lists_uses_signed_coordinates():
    coords = generate_coord_lists(-1, 1, -2, 0)

    assert coords.dtype == np.int32
    assert (-1, -2) in [tuple(coord) for coord in coords]


def test_generate_rois_rejects_true_boundary_overflow_and_keeps_metadata():
    unique_peaks = {
        (1, 1): [(100, 20, (80, 130))],
        (2, 2): [(100, 20, (80, 130))],
    }
    events_t_p_dict = {
        (np.int32(2), np.int32(2)): [
            (np.uint64(90), np.int8(1)),
            (np.uint64(105), np.int8(0)),
            (np.uint64(120), np.int8(1)),
        ]
    }

    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=2,
        min_x=0,
        min_y=0,
        num_cores=1,
        max_x=5,
        max_y=5,
    )

    assert len(rois) == 1
    assert rois["roi"].dtype == np.uint32
    assert rois["roi_n"].dtype == np.uint32
    assert rois["total_events_roi"][0] == 2
    assert rois["total_neg_events_roi"][0] == 1
    assert rois["t_1st"][0] == 90
    assert rois["t_last"][0] == 120
    assert rois["roi_y0"][0] == 0
    assert rois["roi_x0"][0] == 0
    assert rois["dt_pos_s"][0] == pytest.approx(50e-6)
    assert rois["dt_neg_s"][0] == pytest.approx(50e-6)


def test_generate_rois_counts_non_peak_pixels_inside_roi():
    unique_peaks = {
        (2, 2): [(100, 20, (80, 130))],
    }
    events_t_p_dict = {
        (np.int32(2), np.int32(2)): [
            (np.uint64(90), np.int8(1)),
        ],
        (np.int32(2), np.int32(3)): [
            (np.uint64(92), np.int8(1)),
            (np.uint64(108), np.int8(0)),
        ],
    }

    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=2,
        min_x=0,
        min_y=0,
        num_cores=1,
        max_x=5,
        max_y=5,
    )

    assert len(rois) == 1
    assert rois["roi"][0, 2, 2] == 1
    assert rois["roi"][0, 2, 3] == 1
    assert rois["roi_n"][0, 2, 3] == 1
    assert rois["total_events_roi"][0] == 2
    assert rois["total_neg_events_roi"][0] == 1


def test_generate_rois_preserves_simultaneous_events_inside_roi():
    unique_peaks = {
        (2, 2): [(100, 20, (80, 130))],
    }
    events_t_p_dict = {
        (np.int32(2), np.int32(2)): [
            (np.uint64(100), np.int8(1)),
            (np.uint64(100), np.int8(0)),
            (np.uint64(100), np.int8(1)),
        ],
    }

    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=2,
        min_x=0,
        min_y=0,
        num_cores=1,
        max_x=5,
        max_y=5,
    )

    assert len(rois) == 1
    assert rois["roi"][0, 2, 2] == 2
    assert rois["roi_n"][0, 2, 2] == 1
    assert rois["total_events_roi"][0] == 2
    assert rois["total_neg_events_roi"][0] == 1
