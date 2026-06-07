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
