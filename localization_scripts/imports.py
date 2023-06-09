import gc, multiprocessing, os, sys
import numpy as np
from joblib import Parallel, delayed
from localization_scripts.roi_generation import generate_rois
from localization_scripts.peak_finding import (
    find_peaks_parallel,
    find_local_max_peak,
    create_peak_lists,
    group_timestamps_by_coordinate,
)
from localization_scripts.event_array_processing import (
    convert_to_hashmaps,
    create_convolved_signals,
    raw_events_to_array,
    save_dict,
    load_dict,
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
from natsort import natsorted

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)
