from localization_scripts.imports import *

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""

NUM_CORES = multiprocessing.cpu_count()


"""DATASEET_FWHM is the FWHM of the PSF in the dataset in pixels."""
DATASEET_FWHM = 8.5

"""ROI_RADIUS is the radius of the generated ROI in pixels."""
ROI_RADIUS = 8

# INPUT_FILE = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/recording_2023-05-31_16-22-10/rois_prominence_fwhm_7_prominence_14.npy"
# INPUT_FILE = "/home/smlm-workstation/event-smlm/our_ev_smlm_recordings/recording_2023-05-31_15-57-47/rois_prominence_fwhm_6_prominence_13.npy"
INPUT_FILE = "/home/smlm-workstation/event-smlm/Paris/MT_CL/recording_2023-05-22T13-44-30.494Z/rois_prominence_fwhm_8_prominence_12.npy"

rois = np.load(INPUT_FILE, allow_pickle=True)

localizations = perfrom_localization_parallel(rois, dataset_FWHM=DATASEET_FWHM)

np.save(
    INPUT_FILE[:-4]
    + "_localizations"
    + ".npy",
    localizations
)