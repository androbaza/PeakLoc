import numpy as np
import pytest

from localization_scripts.calibration import EventCalibration, NullCalibration
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.poisson_fitting import _mean_maps, fit_joint_poisson_roi
from localization_scripts.psf_model import pixel_integrated_gaussian


def test_joint_poisson_fit_recovers_synthetic_single_emitter():
    rng = np.random.default_rng(1)
    expected_x = 8.3
    expected_y = 7.7
    sigma = 2.1
    psf = pixel_integrated_gaussian((17, 17), expected_x, expected_y, sigma)
    roi_pos = rng.poisson(2.0 + 900.0 * psf).astype(np.uint32)
    roi_neg = rng.poisson(3.0 + 700.0 * psf).astype(np.uint32)
    roi_record = _roi_record(roi_pos, roi_neg)
    config = PeakLocConfig(sigma_psf_px=sigma, max_fit_cond=1e18)

    result = fit_joint_poisson_roi(roi_record, NullCalibration((17, 17)), config)

    assert result.fit_success is True
    assert result.x == pytest.approx(expected_x, abs=0.2)
    assert result.y == pytest.approx(expected_y, abs=0.2)
    assert result.A_pos > result.A_neg
    assert result.calibrated_background is False
    assert result.uncertainty_mode == "model_based_uncalibrated"
    assert result.sigma_x > 0
    assert result.sigma_y > 0


def test_joint_poisson_fit_uses_calibrated_background_and_masks_hot_pixel():
    sigma = 2.0
    expected_x = 8.0
    expected_y = 8.0
    psf = pixel_integrated_gaussian((17, 17), expected_x, expected_y, sigma)
    roi_pos = np.rint(3.0 + 500.0 * psf).astype(np.uint32)
    roi_neg = np.rint(2.0 + 400.0 * psf).astype(np.uint32)
    roi_pos[0, 0] = 5000
    hot_pixel_mask = np.zeros((17, 17), dtype=bool)
    hot_pixel_mask[0, 0] = True
    calibration = EventCalibration(
        dark_rate_pos=np.zeros((17, 17)),
        dark_rate_neg=np.zeros((17, 17)),
        blank_rate_pos=np.ones((17, 17)),
        blank_rate_neg=np.ones((17, 17)),
        hot_pixel_mask=hot_pixel_mask,
        valid_pixel_mask=np.ones((17, 17), dtype=bool),
        calibration_id="pytest-cal",
        calibrated=True,
    )
    roi_record = _roi_record(roi_pos, roi_neg, dt_pos_s=1.0, dt_neg_s=1.0)
    config = PeakLocConfig(sigma_psf_px=sigma, max_fit_cond=1e18)

    result = fit_joint_poisson_roi(roi_record, calibration, config)

    assert result.fit_success is True
    assert result.x == pytest.approx(expected_x, abs=0.2)
    assert result.y == pytest.approx(expected_y, abs=0.2)
    assert result.bg_pos_cal_sum == pytest.approx(288.0)
    assert result.bg_neg_cal_sum == pytest.approx(288.0)
    assert result.calibration_id == "pytest-cal"
    assert result.calibrated_background is True
    assert result.hot_pixel_count == 1
    assert result.valid_pixel_count == 288


def test_joint_poisson_fit_fails_before_optimizer_with_too_few_valid_pixels():
    roi_pos = np.ones((5, 5), dtype=np.uint32)
    roi_neg = np.ones((5, 5), dtype=np.uint32)
    calibration = NullCalibration((5, 5))
    calibration = EventCalibration(
        dark_rate_pos=calibration.dark_rate_pos,
        dark_rate_neg=calibration.dark_rate_neg,
        blank_rate_pos=calibration.blank_rate_pos,
        blank_rate_neg=calibration.blank_rate_neg,
        hot_pixel_mask=np.zeros((5, 5), dtype=bool),
        valid_pixel_mask=np.zeros((5, 5), dtype=bool),
        calibration_id="pytest-cal",
        calibrated=False,
    )
    config = PeakLocConfig(sigma_psf_px=1.5, min_valid_pixels=1)

    result = fit_joint_poisson_roi(_roi_record(roi_pos, roi_neg), calibration, config)

    assert result.fit_success is False
    assert result.fit_status == "too_few_valid_pixels"
    assert np.isnan(result.x)
    assert np.isnan(result.y)
    assert result.fit_cond == np.inf
    assert result.valid_pixel_count == 0


def test_joint_poisson_fit_marks_ill_conditioned_fit_unsuccessful():
    sigma = 2.0
    psf = pixel_integrated_gaussian((17, 17), 8.0, 8.0, sigma)
    roi_pos = np.rint(2.0 + 500.0 * psf).astype(np.uint32)
    roi_neg = np.rint(2.0 + 400.0 * psf).astype(np.uint32)
    config = PeakLocConfig(sigma_psf_px=sigma, max_fit_cond=1.0)

    result = fit_joint_poisson_roi(
        _roi_record(roi_pos, roi_neg), NullCalibration((17, 17)), config
    )

    assert result.fit_success is False
    assert "fisher condition exceeded" in result.fit_status
    assert result.fit_cond > config.max_fit_cond


def test_mean_maps_honor_background_mode():
    params = np.asarray(
        [0.0, 0.0, np.log(10.0), np.log(20.0), np.log(2.0), np.log(3.0)]
    )
    psf = np.full((2, 2), 0.25)
    bg_pos_cal = np.full((2, 2), 5.0)
    bg_neg_cal = np.full((2, 2), 7.0)

    pos_cal, neg_cal = _mean_maps(
        params, psf, bg_pos_cal, bg_neg_cal, "calibrated_only"
    )
    pos_local, neg_local = _mean_maps(params, psf, bg_pos_cal, bg_neg_cal, "local_only")
    pos_both, neg_both = _mean_maps(
        params, psf, bg_pos_cal, bg_neg_cal, "calibrated_plus_local"
    )

    assert np.all(pos_cal == 7.5)
    assert np.all(neg_cal == 12.0)
    assert np.all(pos_local == 4.5)
    assert np.all(neg_local == 8.0)
    assert np.all(pos_both == 9.5)
    assert np.all(neg_both == 15.0)


def test_local_only_background_mode_does_not_mark_calibrated_background_used():
    sigma = 2.0
    psf = pixel_integrated_gaussian((9, 9), 4.0, 4.0, sigma)
    roi_pos = np.rint(3.0 + 250.0 * psf).astype(np.uint32)
    roi_neg = np.rint(2.0 + 200.0 * psf).astype(np.uint32)
    calibration = EventCalibration(
        dark_rate_pos=np.zeros((9, 9)),
        dark_rate_neg=np.zeros((9, 9)),
        blank_rate_pos=np.ones((9, 9)) * 100.0,
        blank_rate_neg=np.ones((9, 9)) * 100.0,
        hot_pixel_mask=np.zeros((9, 9), dtype=bool),
        valid_pixel_mask=np.ones((9, 9), dtype=bool),
        calibration_id="pytest-cal",
        calibrated=True,
    )
    config = PeakLocConfig(
        background_mode="local_only",
        sigma_psf_px=sigma,
        max_fit_cond=1e18,
    )

    result = fit_joint_poisson_roi(_roi_record(roi_pos, roi_neg), calibration, config)

    assert result.fit_success is True
    assert result.calibrated_background is False
    assert result.uncertainty_mode == "model_based_uncalibrated"


def _roi_record(
    roi_pos: np.ndarray,
    roi_neg: np.ndarray,
    *,
    dt_pos_s: float = 0.0,
    dt_neg_s: float = 0.0,
) -> np.ndarray:
    record = np.zeros(
        (),
        dtype=[
            ("roi", np.uint32, roi_pos.shape),
            ("roi_n", np.uint32, roi_neg.shape),
            ("roi_y0", np.int32),
            ("roi_x0", np.int32),
            ("dt_pos_s", np.float64),
            ("dt_neg_s", np.float64),
        ],
    )
    record["roi"] = roi_pos
    record["roi_n"] = roi_neg
    record["dt_pos_s"] = dt_pos_s
    record["dt_neg_s"] = dt_neg_s
    return record
