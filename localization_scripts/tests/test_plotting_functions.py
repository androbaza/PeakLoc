import matplotlib.pyplot as plt
import numpy as np

from localization_scripts import plotting_functions


def test_plot_single_fit_plots_image_coordinates_as_xy(monkeypatch):
    scatter_calls = []

    def record_scatter(x, y, *args, **kwargs):
        scatter_calls.append((x, y))
        return None

    def old_fit_callable(*args):
        raise AssertionError("plot_single_fit should build an explicit xx, yy grid")

    monkeypatch.setattr(plotting_functions.plt, "scatter", record_scatter)
    fit_params = np.asarray([10.0, 4.5, 2.5, 1.2])

    plotting_functions.plot_single_fit(
        fit_params,
        rms=0.1,
        fit=old_fit_callable,
        data=np.ones((7, 9)),
    )

    assert scatter_calls == [(4.5, 2.5)]
    plt.close("all")


def test_plot_rois_from_locs_uses_sigma_psf_px_for_poisson_contour(monkeypatch):
    recorded_sigmas = []

    def record_gaussian(height, center_x, center_y, sigma):
        recorded_sigmas.append(float(sigma))
        return lambda x, y: np.zeros_like(x, dtype=np.float64)

    monkeypatch.setattr(plotting_functions, "_gaussian2d", record_gaussian)

    localizations = np.zeros(
        1,
        dtype=[
            ("double", np.uint8),
            ("roi", np.uint32, (3, 3)),
            ("I", np.float32),
            ("sub_x", np.float64),
            ("sub_y", np.float64),
            ("FWHM", np.float32),
            ("sigma_psf_px", np.float64),
        ],
    )
    localizations["roi"][0] = np.ones((3, 3), dtype=np.uint32)
    localizations["I"][0] = 10.0
    localizations["sub_x"][0] = 1.0
    localizations["sub_y"][0] = 1.0
    localizations["FWHM"][0] = 10.0
    localizations["sigma_psf_px"][0] = 1.5

    plotting_functions.plot_rois_from_locs(localizations, subplotsize=1)

    assert recorded_sigmas == [1.5]
    plt.close("all")


def test_plot_rois_from_locs_falls_back_to_fwhm_when_sigma_absent(monkeypatch):
    recorded_sigmas = []

    def record_gaussian(height, center_x, center_y, sigma):
        recorded_sigmas.append(float(sigma))
        return lambda x, y: np.zeros_like(x, dtype=np.float64)

    monkeypatch.setattr(plotting_functions, "_gaussian2d", record_gaussian)

    localizations = np.zeros(
        1,
        dtype=[
            ("double", np.uint8),
            ("roi", np.uint32, (3, 3)),
            ("I", np.float32),
            ("sub_x", np.float64),
            ("sub_y", np.float64),
            ("FWHM", np.float32),
        ],
    )
    localizations["roi"][0] = np.ones((3, 3), dtype=np.uint32)
    localizations["I"][0] = 10.0
    localizations["sub_x"][0] = 1.0
    localizations["sub_y"][0] = 1.0
    localizations["FWHM"][0] = np.float32(2.354820045 * 2.0)

    plotting_functions.plot_rois_from_locs(localizations, subplotsize=1)

    assert recorded_sigmas == [np.float32(2.354820045 * 2.0) / 2.354820045]
    plt.close("all")
