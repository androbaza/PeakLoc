from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import numpy as np
from localization_scripts.localization_fitting import *
from collections import Counter
from numba import njit, prange
from localization_scripts.utils import *
from csaps import CubicSmoothingSpline
from scipy.ndimage import center_of_mass

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
        if len(peak["peaks"]) > 0 and len(peak[0]) > 0:
            ys = [np.interp(p, peak[0], peak[1]) for p in peak["peaks"]]
            # maxes = ax.scatter(peak['peaks'], ys, color = 'magenta', s = 85, marker = 'x', label = 'max')

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

    l = plt.legend(
        [maxes], ["Detected\nfluorophore\nexcitation"], loc="center left", fontsize=12
    )
    l.set_zorder(200)
    maxes.set_zorder(2000)
    ax.dist = 12
    return fig


def plot_single_fit(fit_params, rms, fit, data):
    # fit_params, rms = fit_gaussian(data)
    # fit = gaussian2D(*fit_params)
    fig = plt.figure(figsize=(7, 7))
    imm = plt.imshow(data, cmap="gray")
    plt.axis("off")
    (height, x, y, sigma) = fit_params
    plt.contour(fit(*np.indices(data.shape)))
    ax = plt.gca()
    # (height, x, y, sigma) = fit_params

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
    plt.scatter(y, x, c="magenta", s=90, marker="x")
    scalebar = ScaleBar(
        65,
        units="nm",
        length_fraction=0.15,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.7,
        font_properties={"size": 6},
    )
    # plt.gca().add_artist(scalebar)
    cbar = fig.colorbar(imm, fraction=0.01, pad=-0.0001, aspect=99)
    cbar.ax.tick_params(size=4, labelsize=17, pad=1)
    # cbar.ax.fontsize = 7
    cbar.ax.set_ylabel("# of events", rotation=270, fontsize=17, labelpad=17)
    cbar.outline.set_visible(False)
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/signle_to_double_gauss_fit.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)


def plot_double_fit(fit_params, rms, fit, data):
    # fit = double_gaussian2D(*fit_params)
    fig = plt.figure(figsize=(7, 7))
    imm = plt.imshow(data, cmap="gray")
    plt.axis("off")
    (height, x, y, sigma, height2, x2, y2, sigma2) = fit_params
    # plt.matshow(data)
    plt.contour(fit(*np.indices(data.shape)))
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
    plt.scatter(y, x, c="magenta", s=90, marker="x", zorder=100)
    plt.scatter(y2, x2, c="magenta", s=90, marker="x", zorder=100)
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
    cbar.outline.set_visible(False)
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/double_gauss_fit.png',dpi=300, bbox_inches = 'tight', pad_inches = 0.01)


def plot_rois(rois_list, subplotsize=6, sign=1, dataset_FWHM=7):
    l = 0
    roi_rad = rois_list[0]["roi"].shape[0] // 2
    # data = data[data['E_total'] > 120]
    #   fig = plt.figure(figsize=(17,17), dpi=300)
    fig, axs = plt.subplots(subplotsize, subplotsize, figsize=(20, 20))
    fig.tight_layout()
    plt.axis("off")
    padded_all = []
    fits = []
    for roi in rois_list:
        # padding_size = 1
        if sign == 1:
            padded = roi[0]
        else:
            padded = roi[1]

        if not padded.any():
            continue
        fit_params, rms = fit_gaussian(padded, dataset_FWHM=dataset_FWHM)
        ax = plt.subplot(
            subplotsize,
            subplotsize,
            l + 1,
        )
        plt.axis("off")
        if fit_params.shape[0] == 8:
            # 2 gaussians were fitted
            fit = double_gaussian2D(*fit_params)
            plt.contour(fit(*np.indices(padded.shape)))
            (height, x, y, sigma, height2, x2, y2, sigma2) = fit_params
            plt.text(
                0.97,
                0.78,
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
            plt.scatter(y, x, c="magenta", s=90, marker="x", zorder=100)
            plt.scatter(y2, x2, c="magenta", s=90, marker="x", zorder=100)
            # ax.set_title('%.2f, %.2f, %.2f, %.2f' %(y, x, y2, x2))
            ax.set_title(f"t:{roi['t_peak']}, tot_events: {roi['total_events_roi']}")
        else:
            (height, x, y, sigma) = fit_params
            fit = gaussian2D(*fit_params)
            roi_ft = np.fft.fft2(padded)
            x_posp, y_posp = est_coord(roi_ft, (1, 0), roi_rad), est_coord(
                roi_ft, (0, 1), roi_rad
            )
            cmy, cmx = center_of_mass(padded)
            plt.scatter(y_posp, x_posp,c="cyan", s=140, marker="x")
            # plt.scatter(cmy, cmx, c="r", s=140, marker="x")
            plt.contour(fit(*np.indices(padded.shape)))
            plt.text(
                0.97,
                0.8,
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
            plt.scatter(y, x, c="magenta", s=90, marker="x")
            # ax.set_title('%.2f, %.2f' %(y, x))
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
        cbar.outline.set_visible(False)
        l += 1
        # if l == subplotsize**2:
        if l == subplotsize*subplotsize:
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

def plot_rois_from_locs(rois_list, filename=None, subplotsize=5, sign=1, dataset_FWHM=8):
    l = 0
    roi_rad = rois_list["roi"][0].shape[0] // 2
    fig, axs = plt.subplots(subplotsize, subplotsize, figsize=(17, 17), dpi=300)
    # fig, axs = plt.figure(figsize=(14, 14))
    fig.tight_layout()
    plt.axis("off")
    for id in range(subplotsize*subplotsize):

        ax = plt.subplot(
            subplotsize,
            subplotsize,
            l + 1,
        )
        roi = rois_list["roi"][id]
        plt.axis("off")

        if rois_list["double"][id] == 1:
            # 2 gaussians were fitted
            fit_params, rms = fit_gaussian(roi, dataset_FWHM=dataset_FWHM)
            if fit_params.size == 4:
                ax.set_title('was double')
                fit = gaussian2D(*fit_params)
                plt.contour(fit(*np.indices(roi.shape)))
                plt.scatter(rois_list["sub_y"][id], rois_list["sub_x"][id], c="magenta", s=90, marker="x", zorder=100)
                plt.text(
                    0.97,
                    0.8,
                    """
                FWHM: %.2f
                RMS Error: %.2f"""
                    % (fit_params[3] * 2.35, rms),
                    fontsize=17,
                    horizontalalignment="right",
                    verticalalignment="baseline",
                    transform=ax.transAxes,
                    c="w",
                )
            else: 
                fit = double_gaussian2D(*fit_params)
                plt.contour(fit(*np.indices(roi.shape)))
                (height, x, y, sigma, height2, x2, y2, sigma2) = fit_params
                plt.text(
                    0.94,
                    0.78,
                    """
                FWHM #1: %.2f
                FWHM #2: %.2f
                RMS Error: %.2f"""
                    % (sigma * 2.35, sigma2 * 2.35, rms),
                    fontsize=15,
                    horizontalalignment="right",
                    verticalalignment="baseline",
                    transform=ax.transAxes,
                    c="w",
                )
                plt.scatter(y, x, c="magenta", s=90, marker="x", zorder=100)
                plt.scatter(y2, x2, c="magenta", s=90, marker="x", zorder=100)
            # ax.set_title('%.2f, %.2f, %.2f, %.2f' %(y, x, y2, x2))
            # imm = plt.imshow(roi, cmap="gray", interpolation="none")
            
            # fit = double_gaussian2D(rois_list["I"][id], rois_list["x"][id], rois_list["y"][id], rois_list["FWHM"][id], rois_list["I"][id], rois_list["x2"][id], rois_list["y2"][id], rois_list["FWHM"][id])
            # plt.contour(fit(*np.indices(roi.shape)))
            # plt.scatter(rois_list["y"][id], rois_list["x"][id], c="magenta", s=90, marker="x", zorder=100)
            # plt.scatter(rois_list["y2"][id], rois_list["x2"][id], c="magenta", s=90, marker="x", zorder=100)
            # ax.set_title('%.2f, %.2f, %.2f, %.2f' %(y, x, y2, x2))
            # ax.set_title(f"t:{roi['t_peak']}, tot_events: {roi['total_events_roi']}")
        else:
            # fit_params = fit_single_gaussian(roi)
            # fit = gaussian2D(rois_list["I"][id], rois_list["sub_x"][id], rois_list["sub_y"][id], rois_list["FWHM"][id])
            # imm = plt.imshow(roi - fit(*np.indices(roi.shape)), cmap="gray", interpolation="none")
            fit_params, rms = fit_gaussian(roi, dataset_FWHM=dataset_FWHM)
            fit = gaussian2D(*fit_params)
            plt.contour(fit(*np.indices(roi.shape)))
            plt.scatter(rois_list["sub_y"][id], rois_list["sub_x"][id], c="magenta", s=90, marker="x", zorder=100)
            plt.text(
                0.97,
                0.8,
                """
            FWHM: %.2f
            RMS Error: %.2f"""
                % (fit_params[3] * 2.35, rms),
                fontsize=17,
                horizontalalignment="right",
                verticalalignment="baseline",
                transform=ax.transAxes,
                c="w",
            )
            # print(rois_list["y"][id], rois_list["x"][id], rois_list["y_p"][id], rois_list["x_p"][id])
            # print(rois_list["y"][id] - rois_list["y_p"][id], rois_list["x"][id] - rois_list["x_p"][id])
        # ax.title.set_text(str(r['t_peak']) + str(r['rel_peak']))
        
        # plt.scatter(cmx, cmy, facecolors='red', marker='x', s=55)
        imm = plt.imshow(roi, cmap="gray", interpolation="none")
        ax.set_title(f"tot_events: {rois_list['E_total'][id]}, sum: {np.sum(roi)}")
        # ax.set_title('#px__sum_px\n'+str(np.count_nonzero(roi)) +'__'+ str(np.sum(roi)), fontsize=17)

        cbar = fig.colorbar(
            imm,
            fraction=0.0322,
            pad=-0.0001,
            aspect=30,
            ticks=np.arange(np.min(roi), np.max(roi) + 1, 2),
        )
        cbar.ax.tick_params(size=4, labelsize=17, pad=1)
        # cbar.ax.set_ylabel("# of events", rotation=270, fontsize=17, labelpad=17)
        cbar.outline.set_visible(False)
        l += 1
        # if l == subplotsize**2:
        if l == subplotsize*subplotsize:
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

def plot_num_events_histogram(times):
    @njit(cache=True, nogil=True)
    def subarray_lengths_histogram(arr):
        subarray_lengths = []
        for i in prange(len(arr)):
            subarray_lengths.append(len(arr[i]))
        return subarray_lengths

    subarray_lengths = subarray_lengths_histogram(times)

    recounted = Counter(subarray_lengths)
    plt.figure(figsize=(20, 10))
    n, bins, patches = plt.hist(
        x=recounted, bins=1000, color="#0504aa", alpha=0.7, rwidth=0.85
    )
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
    maxfreq = n.max()
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
    # Determine the number of rows and columns in the arrays
    num_rows, num_cols = start_times.shape

    # Create arrays of x, y, and z values
    x_values = np.arange(num_cols)
    y_values = np.arange(num_rows)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_values = np.concatenate((start_times.flatten(), end_times.flatten()))
    

    # Create a 3D plot
    fig = plt.figure(figsize=(11, 11), dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    # ax.set_yticks([])
    # ax.set_xticks([])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_proj_type('persp', focal_length=0.5)
    ax.view_init(elev=30.0, azim=-16)
    ax.dist = 12
    # Set the style of the plot
    plt.style.use('seaborn-v0_8-darkgrid')

    # Set the color map
    cmap = plt.get_cmap('viridis')

    line_lengths = np.abs(end_times - start_times)
    # Plot the lines connecting the start and end times
    for i in range(num_rows):
        for j in range(num_cols):
            start_time = start_times[i][j]
            end_time = end_times[i][j]
            length = line_lengths[i][j]
            color = cmap(length/np.max(line_lengths))
            ax.plot([j, j], [i, i], [start_time, end_time], color=color, linewidth=3, alpha=1)
            # Plot crosses at the start and end times
            # if start_time!=end_time:
            #     ax.scatter(j, i, start_time, marker='x', s=10, color='cyan')
            #     ax.scatter(j, i, end_time, marker='x', s=10, color='magenta')

    # Set the labels for the axes
    ax.set_xlabel('X position', labelpad=12, fontsize=12)
    ax.set_ylabel('Y position', labelpad=12, fontsize=12)
    ax.set_zlabel("Time in [ms]", labelpad=12, fontsize=12)

    # Set the limits for the axes
    ax.set_xlim(2, num_cols-4)
    ax.set_ylim(3, num_rows-2)
    ax.set_zlim(np.min(z_values[np.nonzero(z_values)]), np.max(z_values))
    # plt.grid(True)
    # Show the plot
    # plt.savefig('/home/smlm-workstation/event-smlm/event-smlm-thesis/figures/time_on_of_plot_single_roi.png',dpi=300, bbox_inches = 'tight', pad_inches = 0, transparent=True)
    plt.show()