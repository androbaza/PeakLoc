import numpy as np

from localization_scripts.drift import (
    apply_drift_correction,
    estimate_drift_cross_correlation,
)


def test_simulated_linear_drift_is_corrected_to_near_zero_residual_shift():
    localizations = _drifted_localizations()

    drift = estimate_drift_cross_correlation(
        localizations,
        bins=5,
        render_pixel_size_px=1.0,
    )
    corrected = apply_drift_correction(localizations, drift)

    residual_x = _binned_medians(corrected, "x")
    residual_y = _binned_medians(corrected, "y")
    assert np.ptp(residual_x) < 0.25
    assert np.ptp(residual_y) < 0.25


def test_drift_estimator_returns_explicit_method_for_small_inputs():
    localizations = np.zeros(
        1,
        dtype=[("x", np.float64), ("y", np.float64), ("t_peak", np.float64)],
    )

    drift = estimate_drift_cross_correlation(
        localizations,
        bins=5,
        render_pixel_size_px=1.0,
    )

    assert drift.method == "insufficient_data"
    assert drift.time_us.size == 0


def _drifted_localizations() -> np.ndarray:
    base = np.asarray([(0.0, 0.0), (2.0, 1.0), (4.0, -1.0), (6.0, 2.0)])
    records = []
    for bin_index in range(5):
        time = bin_index * 100.0
        dx = bin_index * 0.8
        dy = bin_index * -0.4
        for x, y in base:
            records.append((x + dx, y + dy, time))
    return np.asarray(
        records,
        dtype=[("x", np.float64), ("y", np.float64), ("t_peak", np.float64)],
    )


def _binned_medians(localizations: np.ndarray, field_name: str) -> np.ndarray:
    medians = []
    for time in sorted(set(localizations["t_peak"])):
        mask = localizations["t_peak"] == time
        medians.append(np.median(localizations[field_name][mask]))
    return np.asarray(medians)
