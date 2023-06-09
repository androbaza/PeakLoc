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
DATASEET_FWHM = 7

"""PEAK_TIME_THRESHOLD is the maximum time difference between two peaks in order to be considered as the same peak."""
PEAK_TIME_THRESHOLD = 50e3

"""PEAK_NEIGHBORS is the number of neighboring pixels to be considered when filtering same peaks."""
PEAK_NEIGHBORS = 8

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
# INPUT_FILE_PEAKS = '/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-12-08.025Z/unique_peaks_fwhm_7_prominence_12.pkl'

INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-33-27.882Z.raw"
INPUT_FILE_PEAKS = '/home/smlm-workstation/event-smlm/Paris/24.05/MT/recording_2023-05-24T09-12-08.025Z/unique_peaks_fwhm_7_prominence_15.pkl'

#MT+CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/mt+cl/recording_2023-05-23T10-04-58.785Z.raw" #error in process conv list


#CL
# INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/23.05/cl/recording_2023-05-23T11-48-47.787Z.raw" 

# INPUT_FILE = "/home/smlm-workstation/event-smlm/Evb-SMLM/raw_data/tubulin300x400_200sec_cuts/tubulin300x400_both_[200, 400.0]reduced.npy"


# def main(filename, peaks_file):
filename = INPUT_FILE
peaks_file = INPUT_FILE_PEAKS
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
events = events[events['t'] < 550e6]

# Create coordinate lists
y_coords, x_coords = [min_y, max_y], [min_x, max_x]
coords = generate_coord_lists(y_coords[0], y_coords[1], x_coords[0], x_coords[1])

# Generate dictionaries and calculate max length
print(f"Analyzing the data using {NUM_CORES} cores... Events go brrrrrrrrrrrr!")
print(
    f"Converting events to dictionaries... Elapsed time: {time.time() - start_time:.2f} seconds"
)
events_t_p_dict = array_to_time_map(events)
del events
gc.collect()

unique_peaks = load_dict(peaks_file)

out_folder_localizations = filename[:-4] + "/"
if not os.path.exists(out_folder_localizations):
    os.makedirs(out_folder_localizations)

print(f"Generating ROIs... Elapsed time: {time.time() - start_time:.2f} seconds")
rois = generate_rois(
    unique_peaks,
    events_t_p_dict,
    roi_rad=ROI_RADIUS,
    min_x=min_x,
    min_y=min_y,
    num_cores=NUM_CORES,
    max_x = max_x,
    max_y = max_y,
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


# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         filename = sys.argv[1]
#     else:
#         filename = INPUT_FILE
#         peaks_file = INPUT_FILE_PEAKS
#     main(filename, peaks_file)
