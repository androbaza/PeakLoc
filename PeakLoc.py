from localization_scripts.imports import *

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""

NUM_CORES = multiprocessing.cpu_count()

"""PROMINENCE is the prominence of the peaks in the convolved signals.
Smaller value detects more peaks, increasing the evaluation time."""
PROMINENCE = 12

"""DATASEET_FWHM is the FWHM of the PSF in the dataset in pixels."""
DATASEET_FWHM = 8

"""PEAK_TIME_THRESHOLD is the maximum time difference between two peaks in order to be considered as the same peak."""
PEAK_TIME_THRESHOLD = 30e3

"""PEAK_NEIGHBORS is the number of neighboring pixels to be considered when filtering same peaks."""
PEAK_NEIGHBORS = 9

"""ROI_RADIUS is the radius of the generated ROI in pixels."""
ROI_RADIUS = 8

"""RAW recording or converted events file location."""
# INPUT_FILE = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/MT_5May_S2_reduced_bias_580sec/MT_5May_S2_reduced_bias_580sec.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/TubulinAF647/recording_2023-05-22T11-51-48.153Z.raw" 
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-04-01.505Z.raw" 
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-25-34.554Z.raw" 
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/Tubulin+Clqthrin/recording_2023-05-22T13-44-30.494Z.raw" #crashes

#MT
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-54-31.417Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-33-27.882Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-12-08.025Z.raw"

#MT+CL
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
INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/MT_CL/recording_2023-05-25T09-42-18.758Z.raw" 

#CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/cl/recording_2023-05-23T11-48-47.787Z.raw" 
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/CL/recording_2023-05-25T08-32-14.720Z.raw" 

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Evb-SMLM/raw_data/tubulin300x400_200sec_cuts/tubulin300x400_both_[200, 400.0]reduced.npy"


def main(filename):
    start_time = time.time()

    if os.path.basename(filename)[-4:] == ".raw":
        events = raw_events_to_array(filename).astype(
            [("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")]
        )
    elif os.path.basename(filename)[-4:] == ".npy":
        events = np.load(filename)
    else:
        raise ValueError("File format not recognized!")

    # Get the minimum and maximum x and y coordinates
    min_x = events["x"].min()
    min_y = events["y"].min()
    max_x = events["x"].max()
    max_y = events["y"].max()
    events = events[events['t'] < 600e6]

    # Create coordinate lists
    y_coords, x_coords = [min_y, max_y], [min_x, max_x]
    coords = generate_coord_lists(y_coords[0], y_coords[1], x_coords[0], x_coords[1])

    # Generate dictionaries and calculate max length
    print(f"Analyzing the data using {NUM_CORES} cores... Events go brrrrrrrrrrrr!")
    print(
        f"Converting events to dictionaries... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    dict_events, max_len = array_to_polarity_map(events, coords)
    events_t_p_dict = array_to_time_map(events)
    del events
    gc.collect()

    # Create signals, cleanup and slice data
    print(
        f"Creating convolved signals... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    max_len = int(max_len * 9)
    times, cumsum, coordinates = create_convolved_signals(
        dict_events, coords, max_len, NUM_CORES
    )

    del dict_events, max_len

    print(f"Finding peaks... Elapsed time: {time.time() - start_time:.2f} seconds")
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
    peaks_dict = group_timestamps_by_coordinate(coordinates_peaks, peaks, prominences, on_times)

    # possible to speed up with numba
    print(f"Filtering peaks... Elapsed time: {time.time() - start_time:.2f} seconds")
    unique_peaks = find_local_max_peak(
        peaks_dict, threshold=PEAK_TIME_THRESHOLD, neighbors=PEAK_NEIGHBORS
    )

    out_folder_localizations = filename[:-4] + "/"
    if not os.path.exists(out_folder_localizations):
        os.makedirs(out_folder_localizations)

    save_dict(unique_peaks,
        out_folder_localizations
        + "unique_peaks_fwhm_" 
        + str(DATASEET_FWHM) 
        + "_prominence_"
        + str(PROMINENCE)
        + ".pkl"
    )
    
    print(f"Generating ROIs... Elapsed time: {time.time() - start_time:.2f} seconds")
    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=ROI_RADIUS,
        min_x=min_x,
        min_y=min_y,
        num_cores=NUM_CORES,
        max_x=max_x,
        max_y=max_y
    )

    print(
        f"Performing localization... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    localizations = perfrom_localization_parallel(rois, dataset_FWHM=DATASEET_FWHM)

    print(f"Finished! Total elapsed time: {time.time() - start_time:.2f} seconds")


    np.save(
        out_folder_localizations
        + "localizations_prominence_fwhm_" 
        + str(DATASEET_FWHM) 
        + "_prominence_"
        + str(PROMINENCE)
        + ".npy",
        localizations,
    )
    np.save(
        out_folder_localizations
        + "rois_prominence_fwhm_" 
        + str(DATASEET_FWHM) 
        + "_prominence_"
        + str(PROMINENCE)
        + ".npy",
        rois,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = INPUT_FILE
    main(filename)
