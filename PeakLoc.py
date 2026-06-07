import gc
import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger
from natsort import natsorted

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.event_array_processing import (
    array_to_polarity_map,
    array_to_time_map,
    create_convolved_signals,
    raw_events_to_array,
    save_dict,
)
from localization_scripts.localization_fitting import perfrom_localization_parallel
from localization_scripts.peak_finding import (
    create_peak_lists,
    find_local_max_peak,
    find_peaks_parallel,
    group_timestamps_by_coordinate,
)
from localization_scripts.plotting_functions import plot_3d_time, plot_rois_from_locs
from localization_scripts.roi_generation import generate_coord_lists, generate_rois

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""

NUM_CORES = multiprocessing.cpu_count()

"""PROMINENCE is the prominence of the peaks in the convolved signals.
Smaller value detects more peaks, increasing the evaluation time."""
PROMINENCE = 12

"""DATASET_FWHM is the FWHM of the PSF in the dataset in pixels."""
DATASET_FWHM = 6

"""PEAK_TIME_THRESHOLD is the maximum time difference between two peaks in order to be considered as the same peak."""
PEAK_TIME_THRESHOLD = 40e3

"""PEAK_NEIGHBORS is the number of neighboring pixels to be considered when filtering same peaks."""
PEAK_NEIGHBORS = 9

"""ROI_RADIUS is the radius of the generated ROI in pixels."""
ROI_RADIUS = 8

"""CONVOLUTION_ROI_RADIUS is the pixel radius used for peak-finding signals."""
CONVOLUTION_ROI_RADIUS = 1

DEFAULT_INPUT_FOLDER = "/home/smlm-workstation/event-smlm/Paris/process/"
SLICE_START = int(float(os.environ.get("PEAKLOC_SLICE_START", 0)))
SLICE_DURATION = int(float(os.environ.get("PEAKLOC_SLICE_DURATION", 100e6)))

"""RAW recording or converted events file location."""
# INPUT_FILE = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/MT_5May_S2_reduced_bias_580sec/MT_5May_S2_reduced_bias_580sec.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/TubulinAF647/recording_2023-05-22T11-51-48.153Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-04-01.505Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-25-34.554Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-44-30.494Z.raw" #crashes

# MT
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-54-31.417Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-33-27.882Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-12-08.025Z.raw"

# MT+CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T10-04-58.785Z.raw" #error in process conv list
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T10-18-39.156Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T10-39-44.577Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T10-04-58.785Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T09-45-55.674Z.raw"

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T10-22-18.002Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T10-44-21.770Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T11-10-27.991Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T11-29-21.251Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T11-47-31.679Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/mt_cl/recording_2023-05-24T12-03-37.655Z.raw"

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/MT_CL/recording_2023-05-25T10-01-15.299Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/MT_CL/recording_2023-05-25T10-33-08.518Z.raw"
INPUT_FILE = "data/AF647_coverslip.raw"

# CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/cl/recording_2023-05-23T11-48-47.787Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/CL/recording_2023-05-25T08-32-14.720Z.raw"

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Evb-SMLM/raw_data/tubulin300x400_200sec_cuts/tubulin300x400_both_[200, 400.0]reduced.npy"


def save_processed_plots(localizations: np.ndarray, out_folder: str) -> None:
    if localizations.size == 0:
        logger.info("Skipping plots because no localizations were produced")
        return

    figure_folder = Path(out_folder) / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    roi_fit_figure = plot_rois_from_locs(
        localizations,
        subplotsize=6,
        dataset_FWHM=DATASET_FWHM,
    )
    if roi_fit_figure is not None:
        roi_fit_path = figure_folder / f"roi_fits_{timestamp}.png"
        roi_fit_figure.savefig(roi_fit_path, dpi=300, bbox_inches="tight")
        plt.close(roi_fit_figure)
        logger.info("Saved ROI fit plot to {}", roi_fit_path)

    localization = next(
        (
            loc
            for loc in localizations
            if np.any(loc["roi_event_times"]) and np.any(loc["roi_event_times_n"])
        ),
        None,
    )
    if localization is None:
        logger.info("Skipping ROI event-time plot because no timed ROI data was found")
        return

    roi_time_figure = plot_3d_time(
        localization["roi_event_times"],
        localization["roi_event_times_n"],
    )
    if roi_time_figure is None:
        return
    roi_time_path = figure_folder / f"roi_event_times_{timestamp}.png"
    roi_time_figure.savefig(roi_time_path, dpi=300, bbox_inches="tight")
    plt.close(roi_time_figure)
    logger.info("Saved ROI event-time plot to {}", roi_time_path)


def main(event_slice, time_slice, filename):
    events = event_slice
    if events.size == 0:
        logger.info(
            "No events found in time slice ending at {} for {}", time_slice, filename
        )
        return

    start_time = time.time()

    # Get the minimum and maximum x and y coordinates
    min_x = events["x"].min()
    min_y = events["y"].min()
    max_x = events["x"].max()
    max_y = events["y"].max()

    # Create coordinate lists
    y_coords, x_coords = [min_y, max_y], [min_x, max_x]
    coords = generate_coord_lists(y_coords[0], y_coords[1], x_coords[0], x_coords[1])

    # Generate dictionaries and calculate max length
    logger.info("Analyzing the data using {} cores", NUM_CORES)
    logger.info(
        "Converting events to dictionaries; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    dict_events, max_len = array_to_polarity_map(events, coords)
    events_t_p_dict = array_to_time_map(events)
    del events
    gc.collect()

    # Create signals, cleanup and slice data
    logger.info(
        "Creating convolved signals; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    max_len = int(max_len * 2 * (CONVOLUTION_ROI_RADIUS * 2 + 1) ** 2)
    times, cumsum, coordinates = create_convolved_signals(
        dict_events, coords, max_len, NUM_CORES
    )

    del dict_events, max_len

    logger.info("Finding peaks; elapsed time: {:.2f} seconds", time.time() - start_time)
    peak_list = find_peaks_parallel(
        times,
        cumsum,
        coordinates,
        NUM_CORES,
        prominence=PROMINENCE,
        interpolation_coefficient=5,
        spline_smooth=0.7,
    )
    peaks, prominences, on_times, coordinates_peaks = create_peak_lists(peak_list)
    peaks_dict = group_timestamps_by_coordinate(
        coordinates_peaks, peaks, prominences, on_times
    )

    # possible to speed up with numba
    logger.info(
        "Filtering peaks; elapsed time: {:.2f} seconds", time.time() - start_time
    )
    unique_peaks = find_local_max_peak(
        peaks_dict, threshold=PEAK_TIME_THRESHOLD, neighbors=PEAK_NEIGHBORS
    )

    out_folder_localizations = filename[:-4] + "/"
    temp_files_localization = out_folder_localizations + "temp_files/"
    if not os.path.exists(out_folder_localizations):
        os.makedirs(out_folder_localizations)
    if not os.path.exists(temp_files_localization):
        os.makedirs(temp_files_localization)

    save_dict(
        unique_peaks,
        temp_files_localization
        + "unique_peaks_fwhm_"
        + str(DATASET_FWHM)
        + "_prominence_"
        + str(PROMINENCE)
        + "_time_slice_"
        + str(time_slice)
        + ".pkl",
    )

    logger.info(
        "Generating ROIs; elapsed time: {:.2f} seconds", time.time() - start_time
    )
    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=ROI_RADIUS,
        min_x=min_x,
        min_y=min_y,
        num_cores=NUM_CORES,
        max_x=max_x,
        max_y=max_y,
    )

    logger.info(
        "Performing localization; elapsed time: {:.2f} seconds",
        time.time() - start_time,
    )
    localizations = perfrom_localization_parallel(rois, dataset_FWHM=DATASET_FWHM)

    logger.info(
        "Finished; total elapsed time: {:.2f} seconds", time.time() - start_time
    )
    np.save(
        temp_files_localization
        + "localizations_prominence_fwhm_"
        + str(DATASET_FWHM)
        + "_prominence_"
        + str(PROMINENCE)
        + "_time_slice_"
        + str(time_slice)
        + ".npy",
        localizations,
    )
    np.save(
        temp_files_localization
        + "rois_prominence_fwhm_"
        + str(DATASET_FWHM)
        + "_prominence_"
        + str(PROMINENCE)
        + "_time_slice_"
        + str(time_slice)
        + ".npy",
        rois,
    )


if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     filename = sys.argv[1]
    # else:
    #     filename = INPUT_FILE
    folder = os.environ.get("PEAKLOC_INPUT_FOLDER", DEFAULT_INPUT_FOLDER)
    # folder = '/home/smlm-workstation/event-smlm/Paris/25.05/CL/'
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}. Set PEAKLOC_INPUT_FOLDER."
        )
    if not folder.endswith(os.sep):
        folder += os.sep
    for filename in natsorted(os.listdir(folder)):
        filename = folder + filename
        basename = os.path.basename(filename)
        if basename[-4:] == ".raw":
            events = raw_events_to_array(filename).astype(
                [("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")]
            )
        elif basename[-4:] == ".npy":
            events = np.load(filename)
        elif basename[-5:] == ".bias" or os.path.isdir(filename):
            continue
        else:
            continue
        time_slices = range(
            SLICE_START + SLICE_DURATION,
            int(events["t"].max()) + SLICE_DURATION + 1,
            SLICE_DURATION,
        )
        if len(time_slices) == 0:
            logger.info("No time slices to process for {}", filename)
            continue
        for time_slice in time_slices:
            event_slice = events[
                (events["t"] >= time_slice - SLICE_DURATION)
                * (events["t"] < time_slice)
            ]
            main(event_slice, time_slice, filename)

        out_folder_localizations = filename[:-4] + "/"
        temp_files_localization = out_folder_localizations + "temp_files/"

        if not os.path.isdir(temp_files_localization):
            logger.info("No temporary localization folder found for {}", filename)
            continue
        sorted_names = natsorted(os.listdir(temp_files_localization))
        loc_names = [name for name in sorted_names if name.startswith("localizations")]
        roi_names = [name for name in sorted_names if name.startswith("rois")]
        if not loc_names or not roi_names:
            logger.info("No localization outputs found for {}", filename)
            continue

        localizations_full_list = None
        rois_full_list = None
        for loc_file in sorted_names:
            if loc_file.startswith("localizations"):
                locs_slice = np.load(temp_files_localization + loc_file)
                if localizations_full_list is not None:
                    locs_slice["id"] += np.max(localizations_full_list["id"])
                localizations_full_list = (
                    np.concatenate((localizations_full_list, locs_slice))
                    if localizations_full_list is not None
                    else locs_slice
                )
                # np.delete(temp_files_localization + loc_file)
            elif loc_file.startswith("rois"):
                rois_slice = np.load(temp_files_localization + loc_file)
                rois_full_list = (
                    np.concatenate((rois_full_list, rois_slice))
                    if rois_full_list is not None
                    else rois_slice
                )
                # np.delete(temp_files_localization + loc_file)

        if localizations_full_list is None or rois_full_list is None:
            logger.info("No localization outputs found for {}", filename)
            continue

        np.save(
            out_folder_localizations
            + "localizations_prominence_fwhm_"
            + str(DATASET_FWHM)
            + "_prominence_"
            + str(PROMINENCE)
            + ".npy",
            localizations_full_list,
        )

        np.save(
            out_folder_localizations
            + "rois_prominence_fwhm_"
            + str(DATASET_FWHM)
            + "_prominence_"
            + str(PROMINENCE)
            + ".npy",
            rois_full_list,
        )

        save_processed_plots(localizations_full_list, out_folder_localizations)

        for loc_file in sorted_names:
            os.remove(temp_files_localization + loc_file) if loc_file.startswith(
                "localizations"
            ) or loc_file.startswith("rois") else None
