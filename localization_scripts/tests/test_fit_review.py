import matplotlib
import numpy as np

matplotlib.use("Agg")

from localization_scripts.fit_review import (
    MONTAGE_FILENAMES,
    save_uncertainty_montages,
    select_uncertainty_extremes,
)
from localization_scripts.localization_fitting import localization_qc_dtype
from localization_scripts.pipeline_config import PeakLocConfig


def test_select_uncertainty_extremes_returns_36_or_fewer():
    localizations = _localizations(40)

    lowest, highest = select_uncertainty_extremes(localizations, n=36)

    assert lowest.size == 36
    assert highest.size == 36
    assert lowest[0] == 0
    assert highest[0] == 39

    lowest, highest = select_uncertainty_extremes(localizations[:8], n=36)
    assert lowest.size == 8
    assert highest.size == 8


def test_select_uncertainty_extremes_excludes_nan_uncertainties():
    localizations = _localizations(4)
    localizations["sigma_x"][1] = np.nan
    localizations["sigma_y"][2] = np.nan

    lowest, highest = select_uncertainty_extremes(localizations, n=36)

    assert set(lowest.tolist()) == {0, 3}
    assert set(highest.tolist()) == {0, 3}


def test_save_uncertainty_montages_writes_stable_filenames(tmp_path):
    localizations = _localizations(5)
    qc_table = _qc_table(localizations)

    paths = save_uncertainty_montages(
        localizations,
        localizations[qc_table["accepted"]],
        qc_table,
        tmp_path,
        config=PeakLocConfig(sigma_psf_px=1.2),
        n=3,
        dpi=80,
    )

    expected = {
        "uncertainty_lowest_3_combined.png",
        "uncertainty_highest_3_combined.png",
        "uncertainty_lowest_3_positive.png",
        "uncertainty_highest_3_positive.png",
        "uncertainty_lowest_3_negative.png",
        "uncertainty_highest_3_negative.png",
        "uncertainty_quantile_samples.png",
    }
    assert expected.issubset({path.name for path in paths})
    assert all((tmp_path / filename).is_file() for filename in expected)
    assert "uncertainty_lowest_36_combined.png" in MONTAGE_FILENAMES


def test_save_uncertainty_montages_uses_sub_x_sub_y_scatter(monkeypatch, tmp_path):
    from matplotlib.axes import Axes

    localizations = _localizations(1)
    localizations["sub_x"][0] = 1.25
    localizations["sub_y"][0] = 2.75
    qc_table = _qc_table(localizations)
    scatter_calls = []
    original_scatter = Axes.scatter

    def scatter_spy(self, x, y, *args, **kwargs):
        scatter_calls.append((float(x), float(y)))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", scatter_spy)

    save_uncertainty_montages(
        localizations,
        localizations,
        qc_table,
        tmp_path,
        config=PeakLocConfig(sigma_psf_px=1.2),
        n=1,
        dpi=80,
    )

    assert (1.25, 2.75) in scatter_calls


def _localizations(count: int) -> np.ndarray:
    roi_shape = (5, 5)
    localizations = np.zeros(
        count,
        dtype=[
            ("id", np.uint64),
            ("sub_x", np.float64),
            ("sub_y", np.float64),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("E_total", np.uint64),
            ("E_total_n", np.uint64),
            ("nll_per_event", np.float64),
            ("fit_cond", np.float64),
            ("fit_success", np.bool_),
            ("roi", np.uint32, roi_shape),
            ("roi_n", np.uint32, roi_shape),
        ],
    )
    localizations["id"] = np.arange(count)
    localizations["sub_x"] = 2.0
    localizations["sub_y"] = 2.0
    localizations["sigma_x"] = np.linspace(0.1, 1.0, count)
    localizations["sigma_y"] = np.linspace(0.1, 1.0, count)
    localizations["cov_xy"] = 0.0
    localizations["E_total"] = 20
    localizations["E_total_n"] = 10
    localizations["nll_per_event"] = 1.2
    localizations["fit_cond"] = 10.0
    localizations["fit_success"] = True
    localizations["roi"] = 1
    localizations["roi_n"] = 2
    return localizations


def _qc_table(localizations: np.ndarray) -> np.ndarray:
    qc_table = np.zeros(localizations.size, dtype=localization_qc_dtype())
    qc_table["id"] = localizations["id"]
    qc_table["accepted"] = localizations["fit_success"]
    qc_table["fit_success"] = localizations["fit_success"]
    qc_table["finite_position"] = True
    qc_table["finite_uncertainty"] = True
    qc_table["positive_uncertainty"] = True
    qc_table["fit_cond_ok"] = True
    qc_table["valid_pixels_ok"] = True
    qc_table["uncertainty_px"] = localizations["sigma_x"]
    qc_table["uncertainty_nm"] = localizations["sigma_x"] * 67.0
    qc_table["uncertainty_ok"] = True
    qc_table["fit_cond"] = localizations["fit_cond"]
    qc_table["valid_pixel_count"] = 25
    qc_table["nll_per_event"] = localizations["nll_per_event"]
    qc_table["E_total"] = localizations["E_total"]
    qc_table["E_total_n"] = localizations["E_total_n"]
    qc_table["primary_rejection_reason"] = "accepted"
    return qc_table
