import numpy as np

from localization_scripts.calibration import NullCalibration
from localization_scripts.localization_fitting import fit_single_gaussian, localize_rois
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.psf_model import pixel_integrated_gaussian


def test_localize_rois_runs_legacy_lsq_path():
    roi = _legacy_gaussian_roi()
    rois = _roi_records(roi, roi)
    config = PeakLocConfig(fit_model="legacy_lsq", num_cores=1, dataset_fwhm=5.0)

    localizations = localize_rois(rois, config, NullCalibration((9, 9)))

    assert localizations.size == 1
    assert {"x", "y", "roi"}.issubset(localizations.dtype.names)


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
    assert localizations["fit_success"][0]
    assert localizations["calibrated_background"][0] is np.False_
    assert localizations["uncertainty_mode"][0] == "model_based_uncalibrated"


def test_fit_single_gaussian_uses_explicit_xy_grid():
    center_x = 6.2
    center_y = 3.4
    yy, xx = np.indices((11, 13))
    image = 80.0 * np.exp(
        -(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 1.5**2))
    )

    fit_result = fit_single_gaussian(image)

    np.testing.assert_allclose(fit_result.x[1], center_x, atol=0.05)
    np.testing.assert_allclose(fit_result.x[2], center_y, atol=0.05)


def _legacy_gaussian_roi() -> np.ndarray:
    yy, xx = np.indices((9, 9))
    image = 100.0 * np.exp(-(((xx - 4.0) ** 2 + (yy - 4.0) ** 2) / (2.0 * 1.7**2)))
    return np.rint(image).astype(np.uint32)


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
