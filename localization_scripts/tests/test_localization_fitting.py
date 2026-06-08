import numpy as np

from localization_scripts.calibration import NullCalibration
from localization_scripts.localization_fitting import (
    evaluate_poisson_localization_filters,
    filter_poisson_localizations,
    localize_rois,
    localize_rois_with_attempts,
)
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.psf_model import pixel_integrated_gaussian


def test_localize_rois_runs_joint_poisson_path_and_emits_qc_fields():
    psf = pixel_integrated_gaussian((9, 9), 4.2, 3.8, 1.7)
    roi_pos = np.rint(2.0 + 300.0 * psf).astype(np.uint32)
    roi_neg = np.rint(1.0 + 220.0 * psf).astype(np.uint32)
    rois = _roi_records(roi_pos, roi_neg)
    config = PeakLocConfig(
        fit_model="poisson_joint",
        num_cores=1,
        sigma_psf_px=1.7,
        max_fit_cond=1e18,
    )

    localizations = localize_rois(rois, config, NullCalibration((9, 9)))

    assert localizations.size == 1
    assert localizations["x"][0] == np.float64(localizations["sub_x"][0])
    assert localizations["y"][0] == np.float64(localizations["sub_y"][0])
    assert localizations["A_pos"][0] > localizations["A_neg"][0]
    assert localizations["FWHM"][0] == np.float32(2.354820045 * 1.7)
    assert localizations["sigma_psf_px"][0] == np.float64(1.7)
    assert localizations["fit_success"][0]
    assert localizations["calibrated_background"][0] is np.False_
    assert localizations["uncertainty_mode"][0] == "model_based_uncalibrated"

    tables = localize_rois_with_attempts(rois, config, NullCalibration((9, 9)))
    assert tables.qc_table.size == 1
    assert tables.qc_table["accepted"][0]


def test_localize_rois_filters_poisson_results_before_downstream_outputs():
    psf = pixel_integrated_gaussian((9, 9), 4.2, 3.8, 1.7)
    roi_pos = np.rint(2.0 + 300.0 * psf).astype(np.uint32)
    roi_neg = np.rint(1.0 + 220.0 * psf).astype(np.uint32)
    rois = _roi_records(roi_pos, roi_neg)
    config = PeakLocConfig(
        fit_model="poisson_joint",
        num_cores=1,
        sigma_psf_px=1.7,
        min_valid_pixels=999,
    )

    localizations = localize_rois(rois, config, NullCalibration((9, 9)))

    assert localizations.size == 0


def test_filter_poisson_localizations_drops_failed_or_invalid_fits():
    localizations = np.zeros(
        (5,),
        dtype=[
            ("fit_success", np.bool_),
            ("x", np.float64),
            ("y", np.float64),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("fit_cond", np.float64),
            ("valid_pixel_count", np.uint32),
        ],
    )
    localizations["fit_success"] = [True, False, True, True, True]
    localizations["x"] = [1.0, 1.0, np.nan, 1.0, 1.0]
    localizations["y"] = [2.0, 2.0, 2.0, np.inf, 2.0]
    localizations["sigma_x"] = 0.2
    localizations["sigma_y"] = 0.2
    localizations["cov_xy"] = 0.0
    localizations["fit_cond"] = [10.0, 10.0, 10.0, 10.0, 1e6]
    localizations["valid_pixel_count"] = [16, 16, 16, 16, 3]
    config = PeakLocConfig(max_fit_cond=100.0, min_valid_pixels=8)

    filtered = filter_poisson_localizations(localizations, config)

    assert filtered.size == 1
    assert filtered["x"][0] == 1.0
    assert filtered["y"][0] == 2.0


def test_filter_poisson_localizations_drops_large_uncertainty():
    localizations = np.zeros(
        (3,),
        dtype=[
            ("fit_success", np.bool_),
            ("x", np.float64),
            ("y", np.float64),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("fit_cond", np.float64),
            ("valid_pixel_count", np.uint32),
        ],
    )

    localizations["fit_success"] = True
    localizations["x"] = [1.0, 2.0, 3.0]
    localizations["y"] = [1.0, 2.0, 3.0]
    localizations["sigma_x"] = [0.2, 0.8, 0.2]
    localizations["sigma_y"] = [0.2, 0.8, 0.2]
    localizations["cov_xy"] = [0.0, 0.0, 0.3]
    localizations["fit_cond"] = 10.0
    localizations["valid_pixel_count"] = 100

    config = PeakLocConfig(
        max_localization_uncertainty_px=0.5,
        min_valid_pixels=8,
        max_fit_cond=100.0,
    )

    filtered = filter_poisson_localizations(localizations, config)

    assert filtered.size == 1
    assert filtered["x"][0] == 1.0


def test_evaluate_poisson_filters_records_required_rejection_reasons():
    localizations = _filter_records(5)
    localizations["fit_success"] = [False, True, True, True, True]
    localizations["fit_cond"] = [10.0, 1e6, 10.0, 10.0, 10.0]
    localizations["valid_pixel_count"] = [100, 100, 3, 100, 100]
    localizations["sigma_x"] = [0.2, 0.2, 0.2, 1.0, 0.2]
    localizations["sigma_y"] = [0.2, 0.2, 0.2, 1.0, 0.2]
    config = PeakLocConfig(
        max_fit_cond=100.0,
        min_valid_pixels=8,
        max_localization_uncertainty_px=0.5,
    )

    result = evaluate_poisson_localization_filters(localizations, config)

    assert result.accepted.size == 1
    assert list(result.qc_table["primary_rejection_reason"]) == [
        "fit_failed",
        "fit_condition",
        "valid_pixels",
        "uncertainty",
        "accepted",
    ]
    assert int(result.qc_table["id"][4]) == 4
    assert result.qc_table["uncertainty_nm"][4] == 0.2 * config.optical_pixel_size_nm


def _roi_records(roi_pos: np.ndarray, roi_neg: np.ndarray) -> np.ndarray:
    records = np.zeros(
        (1,),
        dtype=[
            ("roi", np.uint32, roi_pos.shape),
            ("roi_n", np.uint32, roi_neg.shape),
            ("roi_event_times", np.uint64, (2, *roi_pos.shape)),
            ("total_events_roi", np.uint64),
            ("total_neg_events_roi", np.uint64),
            ("t_1st", np.uint64),
            ("t_peak", np.uint64),
            ("t_last", np.uint64),
            ("peak", np.int32, (2,)),
            ("rel_peak", np.int32, (2,)),
            ("roi_y0", np.int32),
            ("roi_x0", np.int32),
            ("dt_pos_s", np.float64),
            ("dt_neg_s", np.float64),
        ],
    )
    records["roi"][0] = roi_pos
    records["roi_n"][0] = roi_neg
    records["total_events_roi"][0] = int(np.sum(roi_pos))
    records["total_neg_events_roi"][0] = int(np.sum(roi_neg))
    records["t_peak"][0] = 100
    records["t_1st"][0] = 90
    records["t_last"][0] = 120
    records["peak"][0] = (4, 4)
    records["rel_peak"][0] = (4, 4)
    return records


def _filter_records(count: int) -> np.ndarray:
    localizations = np.zeros(
        (count,),
        dtype=[
            ("id", np.uint64),
            ("fit_success", np.bool_),
            ("x", np.float64),
            ("y", np.float64),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("fit_cond", np.float64),
            ("valid_pixel_count", np.uint32),
            ("nll_per_event", np.float64),
            ("E_total", np.uint64),
            ("E_total_n", np.uint64),
        ],
    )
    localizations["id"] = np.arange(count)
    localizations["fit_success"] = True
    localizations["x"] = np.arange(count, dtype=np.float64)
    localizations["y"] = np.arange(count, dtype=np.float64)
    localizations["sigma_x"] = 0.2
    localizations["sigma_y"] = 0.2
    localizations["cov_xy"] = 0.0
    localizations["fit_cond"] = 10.0
    localizations["valid_pixel_count"] = 100
    localizations["nll_per_event"] = 1.5
    localizations["E_total"] = 10
    localizations["E_total_n"] = 8
    return localizations
