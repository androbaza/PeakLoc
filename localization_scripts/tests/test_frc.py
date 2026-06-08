import numpy as np

from localization_scripts.frc import (
    compute_frc_resolution_nm,
    split_localizations_for_frc,
)


def test_identical_localization_halves_have_better_frc_than_noisy_halves():
    base = _localizations(80, seed=1)
    noisy = base.copy()
    rng = np.random.default_rng(2)
    noisy["x"] += rng.normal(0, 5.0, size=noisy.size)
    noisy["y"] += rng.normal(0, 5.0, size=noisy.size)

    identical = compute_frc_resolution_nm(
        base,
        base.copy(),
        optical_pixel_size_nm=67.0,
        render_pixel_size_nm=20.0,
    )
    degraded = compute_frc_resolution_nm(
        base,
        noisy,
        optical_pixel_size_nm=67.0,
        render_pixel_size_nm=20.0,
    )

    assert identical.resolution_nm is not None
    assert degraded.resolution_nm is not None
    assert identical.resolution_nm < degraded.resolution_nm


def test_split_localizations_for_frc_uses_odd_even_time_order():
    localizations = _localizations(6, seed=3)
    localizations["t_peak"] = [50, 10, 60, 20, 70, 30]

    locs_a, locs_b = split_localizations_for_frc(localizations)

    assert list(locs_a["t_peak"]) == [10, 30, 60]
    assert list(locs_b["t_peak"]) == [20, 50, 70]


def test_too_small_frc_input_returns_warning_not_fake_resolution():
    localizations = _localizations(1, seed=4)

    result = compute_frc_resolution_nm(
        localizations,
        localizations,
        optical_pixel_size_nm=67.0,
        render_pixel_size_nm=20.0,
    )

    assert result.resolution_nm is None
    assert result.warning == "too_few_localizations"


def _localizations(count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    localizations = np.zeros(
        count,
        dtype=[("x", np.float64), ("y", np.float64), ("t_peak", np.float64)],
    )
    localizations["x"] = rng.uniform(10, 80, size=count)
    localizations["y"] = rng.uniform(10, 80, size=count)
    localizations["t_peak"] = np.arange(count)
    return localizations
