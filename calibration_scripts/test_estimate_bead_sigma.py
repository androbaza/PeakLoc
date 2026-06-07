import numpy as np
import pytest

from calibration_scripts.estimate_bead_sigma import estimate_sigma_from_rois


def test_estimate_sigma_from_isolated_synthetic_bead_rois():
    sigma_px = 2.2
    yy, xx = np.indices((17, 17))
    rois = []
    for center_x, center_y in [(8.0, 8.0), (8.2, 7.9), (7.8, 8.1)]:
        image = 3.0 + 200.0 * np.exp(
            -(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma_px**2))
        )
        rois.append(image)

    estimate = estimate_sigma_from_rois(
        np.asarray(rois),
        pixel_size_nm=67.0,
        dataset_name="synthetic-beads",
        sensitivity="standard",
    )

    assert estimate.bead_count == 3
    assert estimate.roi_radius == 8
    assert estimate.median_sigma_px == pytest.approx(sigma_px, abs=0.02)
    assert estimate.mad_sigma_px < 0.01
