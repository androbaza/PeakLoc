import gc, multiprocessing, os, sys
import numpy as np
from localization_scripts.roi_generation import generate_rois
from localization_scripts.peak_finding import find_peaks_parallel, find_local_max_peak
from localization_scripts.event_array_processing import (
    array_to_polarity_map,
    array_to_time_map,
    create_convolved_signals,
    raw_events_to_array,
)
from localization_scripts.roi_generation import generate_rois, generate_coord_lists
from localization_scripts.localization_fitting import perfrom_localization_parallel
import warnings
from cryptography.utils import CryptographyDeprecationWarning
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaTypeSafetyWarning,
)
import time

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)
