import numpy as np
import pytest

from localization_scripts.calibration import NullCalibration, load_calibration


def test_null_calibration_returns_zero_background_and_valid_mask():
    calibration = NullCalibration((6, 7))

    roi = calibration.get_roi_background(
        global_y0=1,
        global_x0=2,
        roi_radius=1,
        dt_pos_s=0.1,
        dt_neg_s=0.2,
    )

    assert calibration.calibration_id == "none"
    assert calibration.calibrated is False
    assert np.all(roi.bg_pos == 0)
    assert np.all(roi.bg_neg == 0)
    assert np.all(roi.valid_mask)
    assert roi.hot_pixel_count == 0
    assert roi.valid_pixel_count == 9


def test_load_npz_calibration_scales_background_and_masks_hot_pixels(tmp_path):
    path = tmp_path / "calibration_event_model.npz"
    dark_rate_pos = np.ones((4, 5))
    blank_rate_pos = dark_rate_pos + 3
    dark_rate_neg = np.ones((4, 5)) * 2
    blank_rate_neg = dark_rate_neg + 5
    hot_pixel_mask = np.zeros((4, 5), dtype=bool)
    hot_pixel_mask[1, 2] = True
    valid_pixel_mask = np.ones((4, 5), dtype=bool)
    np.savez(
        path,
        dark_rate_pos=dark_rate_pos,
        dark_rate_neg=dark_rate_neg,
        blank_rate_pos=blank_rate_pos,
        blank_rate_neg=blank_rate_neg,
        hot_pixel_mask=hot_pixel_mask,
        valid_pixel_mask=valid_pixel_mask,
        pixel_size_nm=np.asarray(67.0),
        sensor_model=np.asarray("pytest-sensor"),
        calibration_id=np.asarray("pytest-cal"),
    )

    calibration = load_calibration(path, (4, 5), allow_uncalibrated=False)
    roi = calibration.get_roi_background(
        global_y0=0,
        global_x0=1,
        roi_radius=1,
        dt_pos_s=0.5,
        dt_neg_s=0.25,
    )

    assert calibration.calibrated is True
    assert calibration.calibration_id == "pytest-cal"
    assert calibration.sensor_model == "pytest-sensor"
    assert calibration.pixel_size_nm == 67.0
    assert np.all(roi.bg_pos == 2.0)
    assert np.all(roi.bg_neg == 1.75)
    assert roi.hot_pixel_count == 1
    assert roi.valid_pixel_count == 8


def test_load_calibration_rejects_uncalibrated_when_required():
    with pytest.raises(ValueError, match="Calibration path is required"):
        load_calibration(None, (4, 5), allow_uncalibrated=False)
