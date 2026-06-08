from matplotlib_scalebar.scalebar import ScaleBar
import warnings

import matplotlib.pyplot as plt
import numpy as np
from csaps import CubicSmoothingSpline
from numba import njit
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks
from scipy.sparse import SparseEfficiencyWarning

from localization_scripts.peak_finding import jit_interpolate

FWHM_FROM_SIGMA = 2.354820045


def _gaussian2d(height, center_x, center_y, sigma):
    sigma = float(sigma)
    return lambda x, y: (
        height
        * np.exp(-(((center_x - x) / sigma) ** 2 + ((center_y - y) / sigma) ** 2) / 2)
    )


def _double_gaussian2d(
    height, center_x, center_y, sigma, height2, center_x_2, center_y_2, sigma2
):
    sigma, sigma2 = float(sigma), float(sigma2)
    return lambda x, y: (
        height
        * np.exp(-(((center_x - x) / sigma) ** 2 + ((center_y - y) / sigma) ** 2) / 2)
        + height2
        * np.exp(
            -(((center_x_2 - x) / sigma) ** 2 + ((center_y_2 - y) / sigma2) ** 2) / 2
        )
    )


def plot_event_signals_2d(signal_and_peaks):
    fig = plt.figure(figsize=(15, 5), dpi=200)
    ax = fig.add_subplot()
    plots = []
    for id, peak in enumerate(signal_and_peaks):
        plots.append(
            ax.plot(
                peak[0], peak[1], label=f"y: {peak['coord'][0]}, x: {peak['coord'][1]}"
            )
        )
    # ax.set_title('Cumulative sum of events from 4 event sensor pixels', fontsize = 12)
    plt.xlabel("time, ms", fontsize=14)
    plt.ylabel(
        "cumulative sum of positive \nand negative brightness changes", fontsize=14
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.legend(
        [plots[0][0], plots[1][0], plots[2][0], plots[3][0]],
        ["y: 88, x: 22", "y: 88, x: 23", "y: 88, x: 24", "y: 88, x: 25"],
        loc="upper left",
        fontsize=14,
    )
    # plt.legend([maxes], ['Detected fluorophore excitation'], loc = 'upper left')
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/roi_cumsum_calibration.png',dpi=300, bbox_inches = 'tight')
    plt.show()
    return fig


def plot_event_signals_3d(signal_and_peaks):
    fig = plt.figure(figsize=(12, 15), dpi=300)
    ax = fig.add_subplot(projection="3d")
    xs, yaspect, zs = 2.5, 1.5, 0.5
    ax.set_box_aspect((xs, yaspect, zs))
    for id, peak in enumerate(signal_and_peaks):
        x, y1 = peak[0], peak[1]
        # id *= 2
        # create a 3D projection
        # plot the signal as scatter plot with z values set to their amplitudes
        ax.scatter(x, id * np.ones_like(x), y1, alpha=0.6, s=4, marker=".", zorder=1)
        # plot the signal as lines with a transparency of 0.5
        ax.plot(x, id * np.ones_like(x), y1, alpha=0.2, zorder=2)
        # mark the peaks with 'x' markers
        ys = [np.interp(p, peak[0], peak[1]) for p in peak["peaks"]]
        maxes = ax.scatter(
            peak["peaks"],
            id * np.ones_like(peak["peaks"]),
            ys,
            marker="+",
            color="magenta",
            s=95,
            alpha=1,
            zorder=5000,
            label="Detected\n fluorophore\n excitation",
        )
        # set the labels for the x, y, and z axes
        ax.set_xlabel("Time in [ms]", labelpad=19, fontsize=12)
        ax.set_ylabel(
            "Convolved event flux from 3x3 pixels area", fontsize=12, labelpad=-4
        )
        ax.set_yticks([])
        # ax.set_xticks(np.arange(0, 10001, 2000),fontsize=14)
        plt.xticks(fontsize=12)
        # ax.set_zticks([0, 60, 120], fontsize=12)
        ax.set_zlabel("Cumulative sum\nof events", fontsize=12)
        ax.view_init(elev=40.0, azim=-35)
        maxes.set_zorder(20)

    legend = plt.legend(
        [maxes], ["Detected\nfluorophore\nexcitation"], loc="center left", fontsize=12
    )
    legend.set_zorder(200)
    maxes.set_zorder(2000)
    return fig


def plot_single_fit(fit_params, rms, fit, data):
    fig = plt.figure(figsize=(7, 7))
    imm = plt.imshow(data, cmap="gray")
    plt.axis("off")
    (height, center_x, center_y, sigma) = fit_params
    yy, xx = np.indices(data.shape)
    model = _gaussian2d(height, center_x, center_y, sigma)
    fit_img = model(xx, yy)
    plt.contour(fit_img)
    ax = plt.gca()
    # (height, center_x, center_y, sigma) = fit_params

    plt.text(
        0.97,
        0.88,
        """
        FWHM: %.2f
        RMS Error: %.2f"""
        % (sigma * 2.35, rms),
        fontsize=17,
        horizontalalignment="right",
        verticalalignment="baseline",
        transform=ax.transAxes,
        c="w",
    )
    plt.scatter(center_x, center_y, c="magenta", s=90, marker="x")
    # plt.gca().add_artist(scalebar)
    cbar = fig.colorbar(imm, fraction=0.01, pad=-0.0001, aspect=99)
    cbar.ax.tick_params(size=4, labelsize=17, pad=1)
    # cbar.ax.fontsize = 7
    cbar.ax.set_ylabel("# of events", rotation=270, fontsize=17, labelpad=17)
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/signle_to_double_gauss_fit.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)


def plot_double_fit(fit_params, rms, fit, data):
    fig = plt.figure(figsize=(7, 7))
    imm = plt.imshow(data, cmap="gray")
    plt.axis("off")
    (height, center_x, center_y, sigma, height2, center_x2, center_y2, sigma2) = (
        fit_params
    )
    # plt.matshow(data)
    yy, xx = np.indices(data.shape)
    model = _double_gaussian2d(
        height,
        center_x,
        center_y,
        sigma,
        height2,
        center_x2,
        center_y2,
        sigma2,
    )
    fit_img = model(xx, yy)
    plt.contour(fit_img)
    ax = plt.gca()
    plt.text(
        0.97,
        0.84,
        """
        FWHM #1: %.2f
        FWHM #2: %.2f
        RMS Error: %.2f"""
        % (sigma * 2.35, sigma2 * 2.35, rms),
        fontsize=17,
        horizontalalignment="right",
        verticalalignment="baseline",
        transform=ax.transAxes,
        c="w",
    )
    plt.scatter(center_x, center_y, c="magenta", s=90, marker="x", zorder=100)
    plt.scatter(center_x2, center_y2, c="magenta", s=90, marker="x", zorder=100)
    scalebar = ScaleBar(
        65,
        units="nm",
        length_fraction=0.15,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.7,
        font_properties={"size": 17},
    )
    plt.gca().add_artist(scalebar)
    cbar = fig.colorbar(imm, fraction=0.01, pad=-0.0001, aspect=104)
    cbar.ax.tick_params(size=4, labelsize=17, pad=1)
    cbar.ax.set_ylabel("# of events", rotation=270, fontsize=17, labelpad=17)
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/double_gauss_fit.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)


def plot_rois(rois_list, subplotsize=6, sign=1, dataset_FWHM=7):
    subplot_index = 0
    # data = data[data['E_total'] > 120]
    #   fig = plt.figure(figsize=(17,17), dpi=300)
    fig, axs = plt.subplots(subplotsize, subplotsize, figsize=(20, 20))
    fig.tight_layout()
    plt.axis("off")
    padded_all = []
    for roi in rois_list:
        # padding_size = 1
        if sign == 1:
            padded = roi[0]
        else:
            padded = roi[1]

        if not padded.any():
            continue
        ax = plt.subplot(
            subplotsize,
            subplotsize,
            subplot_index + 1,
        )
        plt.axis("off")
        center_y, center_x = center_of_mass(padded)
        if np.isfinite(center_x) and np.isfinite(center_y):
            plt.scatter(center_x, center_y, c="magenta", s=90, marker="x", zorder=100)
        ax.set_title(f"t:{roi['t_peak']}, tot_events: {roi['total_events_roi']}")

        # ax.title.set_text(str(r['t_peak']) + str(r['rel_peak']))
        # ax.set_ylabel('#px__sum_px\n'+str(np.count_nonzero(padded)) +'__'+ str(np.sum(padded)), fontsize=17)
        # plt.scatter(cmx, cmy, facecolors='red', marker='x', s=55)
        imm = plt.imshow(padded, cmap="gray", interpolation="none")
        cbar = fig.colorbar(
            imm,
            fraction=0.0322,
            pad=-0.0001,
            aspect=30,
            ticks=np.arange(np.min(padded), np.max(padded) + 1, 2),
        )
        cbar.ax.tick_params(size=4, labelsize=17, pad=1)
        cbar.ax.set_ylabel("# of events", rotation=270, fontsize=17, labelpad=17)
        subplot_index += 1
        # if l == subplotsize**2:
        if subplot_index == subplotsize * subplotsize:
            break

    scalebar = ScaleBar(
        65,
        units="nm",
        length_fraction=0.15,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.7,
        font_properties={"size": 17},
    )
    plt.gca().add_artist(scalebar)
    fig.tight_layout()
    #   plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/12_rois_fit_example_negatives.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)
    return padded_all


def plot_rois_from_locs(
    rois_list, filename=None, subplotsize=6, sign=1, dataset_FWHM=7
):
    if rois_list.size == 0:
        return None

    subplot_index = 0
    fig, axs = plt.subplots(subplotsize, subplotsize, figsize=(20, 20), squeeze=False)
    fig.tight_layout()
    for ax in axs.ravel():
        ax.axis("off")
    plot_count = min(rois_list.size, subplotsize * subplotsize)
    for id in range(plot_count):
        ax = axs.ravel()[subplot_index]
        plt.sca(ax)
        roi = rois_list["roi"][id]
        plt.axis("off")
        if rois_list["double"][id] == 1:
            pass
            # yy, xx = np.indices(roi.shape)
            # plt.contour(fit(xx, yy))
            # plt.scatter(rois_list["x"][id], rois_list["y"][id], c="magenta", s=90, marker="x", zorder=100)
            # plt.scatter(rois_list["x2"][id], rois_list["y2"][id], c="magenta", s=90, marker="x", zorder=100)
            # ax.set_title('%.2f, %.2f, %.2f, %.2f' %(x, y, x2, y2))
            # ax.set_title(f"t:{roi['t_peak']}, tot_events: {roi['total_events_roi']}")
        else:
            yy, xx = np.indices(roi.shape)
            sigma = rois_list["FWHM"][id] / FWHM_FROM_SIGMA
            model = _gaussian2d(
                rois_list["I"][id],
                rois_list["sub_x"][id],
                rois_list["sub_y"][id],
                sigma,
            )
            fit_img = model(xx, yy)
            plt.contour(fit_img)
            plt.scatter(
                rois_list["sub_x"][id],
                rois_list["sub_y"][id],
                c="magenta",
                s=90,
                marker="x",
                zorder=100,
            )
            # print(rois_list["y"][id], rois_list["x"][id], rois_list["y_p"][id], rois_list["x_p"][id])
            # print(rois_list["y"][id] - rois_list["y_p"][id], rois_list["x"][id] - rois_list["x_p"][id])
        # ax.title.set_text(str(r['t_peak']) + str(r['rel_peak']))
        # ax.set_ylabel('#px__sum_px\n'+str(np.count_nonzero(padded)) +'__'+ str(np.sum(padded)), fontsize=17)
        # plt.scatter(cmx, cmy, facecolors='red', marker='x', s=55)
        plt.imshow(roi, cmap="gray", interpolation="none")

        subplot_index += 1
        # if l == subplotsize**2:
        if subplot_index == subplotsize * subplotsize:
            break

    scalebar = ScaleBar(
        65,
        units="nm",
        length_fraction=0.15,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.7,
        font_properties={"size": 17},
    )
    plt.gca().add_artist(scalebar)
    fig.tight_layout()
    # plt.savefig(filename[:-4] + '_rois_examples.png', dpi=300, transparent=True, bbox_inches = 'tight')
    #   plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/12_rois_fit_example_negatives.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)
    return fig


def plot_num_events_histogram(times):
    @njit(cache=True, nogil=True)
    def subarray_lengths_histogram(arr):
        subarray_lengths = []
        for i in range(len(arr)):
            subarray_lengths.append(len(arr[i]))
        return subarray_lengths

    subarray_lengths = subarray_lengths_histogram(times)

    plt.figure(figsize=(20, 10))
    plt.hist(x=subarray_lengths, bins=1000, color="#0504aa", alpha=0.7, rwidth=0.85)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(
        "My Very Own Histogram, mean="
        + str(np.mean(subarray_lengths))
        + ", std="
        + str(np.std(subarray_lengths))
    )
    plt.vlines(
        np.mean(subarray_lengths) + np.std(subarray_lengths),
        0,
        500,
        colors="k",
        linestyles="dashed",
        label="Std",
    )
    plt.text(23, 45, r"$\mu=15, b=3$")
    plt.xlim(xmin=0, xmax=50000)


def plot_peak_ON_OFF_detection(
    times,
    cumsum,
    prominence=15,
    spline_smoothness=0.8,
    interpolation_coefficient=5,
    cutoff_event_count=2000,
    plot_spline=True,
    plot_linear=False,
    num_of_plots=5,
):
    #  fig = plt.figure(figsize=(24,8), dpi=200)
    from localization_scripts.utils import find_on_off_plot

    fig = plt.figure(figsize=(15, 5), dpi=130)
    ax = fig.add_subplot()
    up = 0
    length = 10000
    counter2 = 0
    max_x = 0
    for i in range(0, len(times)):
        if len(times[i]) < cutoff_event_count:
            continue
        if counter2 == num_of_plots:
            break
        tnew = np.linspace(
            0,
            times[i].max(),
            num=len(times[i]) * interpolation_coefficient,
            dtype=np.uint64,
        )
        ynew = jit_interpolate(times[i], cumsum[i], tnew)
        if plot_linear:
            plt.plot(tnew[:length], ynew[:length] + up, "-", alpha=0.2, color="black")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            s = CubicSmoothingSpline(
                times[i], cumsum[i], smooth=spline_smoothness, normalizedsmooth=True
            ).spline
        if plot_spline:
            plt.plot(tnew[:length], s(tnew)[:length] + up, alpha=0.8)
        p, p_props = find_peaks(ynew[:length], prominence=prominence)
        ys = [np.interp(p, (tnew)[:length], ynew[:length] + up) for p in tnew[p]]
        maxes = plt.scatter(tnew[p], ys, color="magenta", s=85, marker="x", label="max")
        der_2 = s.derivative()(tnew)
        on_off, on_off_t = find_on_off_plot(p, der_2, tnew, ynew)
        on_off = np.asarray(on_off)
        on_off_t = np.asarray(on_off_t)
        #  print(on_off )
        if len(on_off) == 0:
            continue
        ons = plt.scatter(
            on_off[:, 0],
            ynew[on_off_t[:, 0]] + up,
            color="blue",
            s=185,
            marker="|",
            label="max",
        )
        offs = plt.scatter(
            on_off[:, 1],
            ynew[on_off_t[:, 1]] + up,
            color="red",
            s=185,
            marker="|",
            label="max",
        )
        up += 100
        counter2 += 1
        if (tnew)[:length].max() > max_x:
            max_x = (tnew)[:length].max()
        plt.legend(
            [maxes, ons, offs],
            ["Detected fluorophore excitation", "First ON event", "Last OFF event"],
            loc="upper left",
        )
    ax.set_xlim(0, max_x + 1)
    plt.xlabel("time, μs", fontsize=14)
    plt.ylabel(
        "cumulative sum of positive \nand negative brightness changes", fontsize=14
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #  plt.grid()
    #  plt.legend([plots[0][0], plots[1][0], plots[2][0], plots[3][0]], ['y: 88, x: 22', 'y: 88, x: 23', 'y: 88, x: 24', 'y: 88, x: 25'], loc = 'upper left', fontsize = 14)
    #  plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/roi_cumsum_on_off.png',dpi=300, bbox_inches = 'tight')
    plt.show()


def plot_3d_time(start_times, end_times):
    if not np.any(start_times) and not np.any(end_times):
        return None

    # Determine the number of rows and columns in the arrays
    num_rows, num_cols = start_times.shape

    # Create arrays of x, y, and z values
    x_values = np.arange(num_cols)
    y_values = np.arange(num_rows)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_values = np.concatenate((start_times.flatten(), end_times.flatten()))

    # Create a 3D plot
    fig = plt.figure(figsize=(11, 11), dpi=300)
    ax = fig.add_subplot(projection="3d")

    # ax.set_yticks([])
    # ax.set_xticks([])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_proj_type("persp", focal_length=0.5)
    ax.view_init(elev=30.0, azim=-16)
    # Set the style of the plot
    plt.style.use("seaborn-v0_8-darkgrid")

    # Set the color map
    cmap = plt.get_cmap("viridis")

    line_lengths = np.abs(end_times - start_times)
    max_line_length = np.max(line_lengths)
    # Plot the lines connecting the start and end times
    for i in range(num_rows):
        for j in range(num_cols):
            start_time = start_times[i][j]
            end_time = end_times[i][j]
            length = line_lengths[i][j]
            color = cmap(0 if max_line_length == 0 else length / max_line_length)
            ax.plot(
                [j, j],
                [i, i],
                [start_time, end_time],
                color=color,
                linewidth=3,
                alpha=1,
            )
            # Plot crosses at the start and end times
            # if start_time!=end_time:
            #     ax.scatter(j, i, start_time, marker='x', s=10, color='cyan')
            #     ax.scatter(j, i, end_time, marker='x', s=10, color='magenta')

    # Set the labels for the axes
    ax.set_xlabel("X position", labelpad=12, fontsize=12)
    ax.set_ylabel("Y position", labelpad=12, fontsize=12)
    ax.set_zlabel("Time in [ms]", labelpad=12, fontsize=12)

    # Set the limits for the axes
    ax.set_xlim(2, num_cols - 4)
    ax.set_ylim(3, num_rows - 2)
    nonzero_times = z_values[np.nonzero(z_values)]
    if nonzero_times.size:
        ax.set_zlim(np.min(nonzero_times), np.max(z_values))
    # plt.grid(True)
    # Show the plot
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/time_on_of_plot_single_roi.png',dpi=300, bbox_inches = 'tight', pad_inches = 0, transparent=True)
    return fig
