from localization_scripts.imports import *

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""

NUM_CORES = multiprocessing.cpu_count()

"""PROMINENCE is the prominence of the peaks in the convolved signals.
Smaller value detects more peaks, increasing the evaluation time."""
PROMINENCE = 12

"""DATASEET_FWHM is the FWHM of the PSF in the dataset in pixels."""
DATASEET_FWHM = 7

"""PEAK_TIME_THRESHOLD is the maximum time difference between two peaks in order to be considered as the same peak."""
PEAK_TIME_THRESHOLD = 40e3

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
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/MT_CL/recording_2023-05-25T09-42-18.758Z.raw"

# CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/cl/recording_2023-05-23T11-48-47.787Z.raw"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/25.05/CL/recording_2023-05-25T08-32-14.720Z.raw"

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Evb-SMLM/raw_data/tubulin300x400_200sec_cuts/tubulin300x400_both_[200, 400.0]reduced.npy"


def main(slice, filename):
    events = slice
    time_slice = slice["t"].max()//1e3
    start_time = time.time()

    # Get the minimum and maximum x and y coordinates
    min_x = events["x"].min()
    min_y = events["y"].min()
    max_x = events["x"].max()
    max_y = events["y"].max()

    out_folder_localizations = filename[:-4] + "/"
    temp_files_localization = out_folder_localizations + "temp_files/"
    if not os.path.exists(out_folder_localizations):
        os.makedirs(out_folder_localizations)
    if not os.path.exists(temp_files_localization):
        os.makedirs(temp_files_localization)
    # Create coordinate lists
    # y_coords, x_coords = [min_y, max_y], [min_x, max_x]
    # coords = generate_coord_lists(y_coords[0], y_coords[1], x_coords[0], x_coords[1])

    # Generate dictionaries and calculate max length
    print(f"Analyzing the data using {NUM_CORES} cores... Events go brrrrrrrrrrrr!")
    print(
        f"Converting events to dictionaries... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    
    dict_events, events_t_p_dict, coords, max_length = convert_to_hashmaps(events, out_folder_localizations, max_x, max_y)
    del events
    gc.collect()

    # Create signals, cleanup and slice data
    print(
        f"Creating convolved signals... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    
    times, cumsum, coordinates = create_convolved_signals(
        dict_events, coords, max_len=max_length*3, num_cores=NUM_CORES
    )

    del dict_events

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
    peaks_dict = group_timestamps_by_coordinate(
        coordinates_peaks, peaks, prominences, on_times
    )

    # possible to speed up with numba
    print(f"Filtering peaks... Elapsed time: {time.time() - start_time:.2f} seconds")
    unique_peaks = find_local_max_peak(
        peaks_dict, threshold=PEAK_TIME_THRESHOLD, neighbors=PEAK_NEIGHBORS
    )

    save_dict(
        unique_peaks,
        temp_files_localization
        + "unique_peaks_fwhm_"
        + str(DATASEET_FWHM)
        + "_prominence_"
        + str(PROMINENCE)
        + "_time_slice_"
        + str(time_slice)
        + ".pkl",
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
        max_y=max_y,
    )

    print(
        f"Performing localization... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    localizations = perfrom_localization_parallel(rois, dataset_FWHM=DATASEET_FWHM)

    print(f"Finished! Total elapsed time: {time.time() - start_time:.2f} seconds")
    np.save(
        temp_files_localization
        + "localizations_prominence_fwhm_"
        + str(DATASEET_FWHM)
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
        + str(DATASEET_FWHM)
        + "_prominence_"
        + str(PROMINENCE)
        + "_time_slice_"
        + str(time_slice)
        + ".npy",
        rois,
    )
    return 

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     filename = sys.argv[1]
    # else:
    #     filename = INPUT_FILE
    folder = '/home/smlm-workstation/event-smlm/Paris/process/'
    # folder = '/home/smlm-workstation/event-smlm/Paris/25.05/CL/'
    for filename in natsorted(os.listdir(folder)):
        filename = folder + filename
        if os.path.basename(filename)[-4:] == ".raw":
            events = raw_events_to_array(filename).astype(
                [("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")]
            )
        elif os.path.basename(filename)[-5:] == '.bias' or os.path.isdir(filename):
            continue

        event_slices = []
        num_cores = multiprocessing.cpu_count() // 2
        slice_size = int(len(events) / num_cores)
        for i in range(slice_size, len(events), slice_size):
            event_slices.append(events[i - slice_size : i])
            if i + slice_size > len(events):
                event_slices.append(events[i:])
        del events
        RES = Parallel(n_jobs=num_cores)(
            delayed(main)(event_slices[i], filename)
            for i in range(len(event_slices))
        )

        # for time_slice in range(int(200e6), events["t"].max(), int(200e6)):
        #     slice = events[(events["t"] > time_slice - 200e6) * (events["t"] < time_slice)]
        #     main(slice, time_slice, filename)
        if os.path.isdir(filename):
            continue
        out_folder_localizations = filename[:-4] + "/"
        temp_files_localization = out_folder_localizations + "temp_files/"

        sorted_names = natsorted(os.listdir(temp_files_localization))

        id, id2 = 0, 0
        for loc_file in sorted_names:
            if loc_file.startswith("localizations"):
                locs_slice = np.load(temp_files_localization + loc_file)
                if id != 0:
                    locs_slice["id"] += np.max(localizations_full_list["id"])
                localizations_full_list = (
                    np.concatenate((localizations_full_list, locs_slice))
                    if id != 0
                    else locs_slice
                )
                id += 1
                # np.delete(temp_files_localization + loc_file)
            elif loc_file.startswith("rois"):
                rois_slice = np.load(temp_files_localization + loc_file)
                rois_full_list = (
                    np.concatenate((rois_full_list, rois_slice)) if id2 != 0 else rois_slice
                )
                id2 += 1
                # np.delete(temp_files_localization + loc_file)
            

        np.save(
            out_folder_localizations
            + "localizations_prominence_fwhm_"
            + str(DATASEET_FWHM)
            + "_prominence_"
            + str(PROMINENCE)
            + ".npy",
            localizations_full_list,
        )

        np.save(
            out_folder_localizations
            + "rois_prominence_fwhm_"
            + str(DATASEET_FWHM)
            + "_prominence_"
            + str(PROMINENCE)
            + ".npy",
            rois_full_list,
        )

        for loc_file in sorted_names:
            os.remove(temp_files_localization + loc_file) if loc_file.startswith("localizations") or loc_file.startswith("rois") else None
