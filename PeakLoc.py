from localization_scripts.imports import *

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""

NUM_CORES = multiprocessing.cpu_count()
PROMINENCE = 25
DATASEET_FWHM = 5.5
PEAK_TIME_THRESHOLD = 50e3
PEAK_NEIGHBORS = 6
ROI_RADIUS = 6
# WORKDIR = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/"


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
    # events = events[events['t'] < 10e6]

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

    # Create signals, cleanup and slice data
    print(
        f"Creating convolved signals... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    times, cumsum, coordinates = create_convolved_signals(
        dict_events, coords, max_len, NUM_CORES
    )

    del dict_events, max_len

    print(f"Finding peaks... Elapsed time: {time.time() - start_time:.2f} seconds")
    peaks_dict = find_peaks_parallel(
        times,
        cumsum,
        coordinates,
        NUM_CORES,
        prominence=PROMINENCE,
        interpolation_coefficient=5,
        spline_smooth=0.7,
    )

    # possible to speed up with numba
    print(f"Filtering peaks... Elapsed time: {time.time() - start_time:.2f} seconds")
    unique_peaks = find_local_max_peak(
        peaks_dict, threshold=PEAK_TIME_THRESHOLD, neighbors=PEAK_NEIGHBORS
    )

    print(f"Generating ROIs... Elapsed time: {time.time() - start_time:.2f} seconds")
    rois = generate_rois(
        unique_peaks,
        events_t_p_dict,
        roi_rad=ROI_RADIUS,
        min_x=min_x,
        min_y=min_y,
        num_cores=NUM_CORES,
    )

    print(
        f"Performing localization... Elapsed time: {time.time() - start_time:.2f} seconds"
    )
    localizations = perfrom_localization_parallel(rois, dataset_FWHM=DATASEET_FWHM)

    print(f"Finished! Total elapsed time: {time.time() - start_time:.2f} seconds")

    out_folder_localizations = filename[:-4] + "/"
    if not os.path.exists(out_folder_localizations):
        os.makedirs(out_folder_localizations)

    np.save(
        out_folder_localizations
        + "localizations_prominence_"
        + str(PROMINENCE)
        + ".npy",
        localizations,
    )


if __name__ == "__main__":
    if sys.argv[1]:
        filename = sys.argv[1]
    else:
        filename = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/MT_5May_S2_reduced_bias_580sec/MT_5May_S2_reduced_bias_580sec.raw"
        # filename = "/home/smlm-workstation/event-smlm/Evb-SMLM/raw_data/tubulin300x400_200sec_cuts/tubulin300x400_both_[200, 400.0]reduced.npy"
    main(filename)
