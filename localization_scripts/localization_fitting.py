from scipy.optimize import least_squares
import numpy as np
from numba import jit, prange
import multiprocessing
from joblib import Parallel, delayed
from localization_scripts.event_array_processing import slice_data
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

def gaussian2D(height, center_x, center_y, width):
    width = float(width)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2
    )


def double_gaussian2D(
    height, center_x, center_y, width, height2, center_x_2, center_y_2, width2
):
    width, width2 = float(width), float(width2)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width) ** 2 + ((center_y - y) / width) ** 2) / 2
    ) + height2 * np.exp(
        -(((center_x_2 - x) / width) ** 2 + ((center_y_2 - y) / width2) ** 2) / 2
    )

def fit_single_gaussian(data):
    params = [np.max(data), data.shape[1]//2, data.shape[1]//2, 2.5]
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) - data)
    bounds = ([0, 0, 0, 1], [3*params[0], data.shape[1], data.shape[0], 11])
    return least_squares(
        errorfunction, params, method="trf", bounds=bounds, ftol=1e-4, xtol=1e-4
    )


def fit_two_gaussians(data, lm=False):
    params = [np.max(data), data.shape[1]//2, data.shape[1]//2, 3.5]
    params2 = [np.max(data), data.shape[1]//2+1, data.shape[1]//2-1, 4]
    params = np.append(params, params2)
    errorfunction = lambda p: np.ravel(
        double_gaussian2D(*p)(*np.indices(data.shape)) - data
    )
    bounds = (
        (0, 0, 0, 3, 0, 0, 0, 3),
        (
            3*params[0] ,
            data.shape[1],
            data.shape[0],
            7,
            2*params[0],
            data.shape[1],
            data.shape[0],
            7,
        ),
    )
    return least_squares(errorfunction, params, method="trf", bounds=bounds, ftol=1e-4, xtol=1e-4)


@jit(nopython=True, fastmath=True, cache=True)
def res_rmse(residue):
    return np.sqrt(np.mean(residue**2))


def fit_gaussian(roi, dataset_FWHM=5.5):
    fit_params = fit_single_gaussian(roi)
    # rms = res_rmse(gaussian2D(*fit_params.x)(*np.indices(roi.shape)) - roi)
    rms = res_rmse(fit_params.fun)
    # rms = calculate_fit_error(roi, gaussian2D(*fit_params.x)(*np.indices(roi.shape)))
    # FWHM=2.35*sigma
    sigma_2_locs = dataset_FWHM / 2.35
    if fit_params.x[3] > sigma_2_locs*1.5:
        return fit_params.x, 5
    if fit_params.x[3] > sigma_2_locs:
        try:
            fit_params2 = fit_two_gaussians(roi)
        except:
            return np.asarray([0,0,0,0]), 5
        rms2 = res_rmse(fit_params2.fun)
        if rms2 > rms:
            return fit_params.x, rms
        else:
            return fit_params2.x, rms2
    return fit_params.x, rms


def concatenate_locs(localized_data):
    id = localized_data[0]["id"][-1] + 1
    concatenated_data = localized_data[0]
    for i in range(1, len(localized_data)):
        if localized_data[i] is None:
            continue
        localized_data[i]["id"] += int(id)
        id = localized_data[i]["id"][-1] + 1
        concatenated_data = np.concatenate(
            (concatenated_data, localized_data[i]), axis=0
        )
    return concatenated_data


def perfrom_localization_parallel(rois: np.ndarray, dataset_FWHM: float = 5.5):
    """
    Performs localization of ROIs on a given image.

    Parameters
    ----------
    rois: list of ROIs (each ROI is a list of coordinates)
    dataset_FWHM: FWHM of the dataset

    Returns
    -------
    list of ROIs (each ROI is a list of coordinates)
    """
    num_cores = multiprocessing.cpu_count()
    rois = slice_data(rois, num_cores)
    RES = Parallel(n_jobs=num_cores)(
        delayed(localize_MLE)(rois[i], dataset_FWHM=dataset_FWHM)
        for i in range(len(rois))
    )
    localized_data = []
    for i in np.arange(np.shape(RES)[0]):
        localized_data.append(RES[i])
    return concatenate_locs(localized_data)


def est_coord(roi_ft, coord_type, roi_rad):
    phase_angle = np.arctan(roi_ft[coord_type].imag / roi_ft[coord_type].real) - np.pi
    return np.abs(phase_angle) / (2 * np.pi / (roi_rad * 2 + 1))

def calculate_mean_and_subtract(data, filter=True):
    on = data['roi_event_times']
    off = data['roi_event_times_n']
    t_peak = data['t_peak']
    if filter:
        on = on*(data['roi']>1)
        off = off*(data['roi_n']>1)
    mean_value_on = np.mean(on[on != 0])
    mean_value_off = np.mean(off[off != 0])
    result = np.zeros(3)
    result[0] = (t_peak - mean_value_on)/1e3
    result[1] = (mean_value_off - t_peak)/1e3
    result[2] = (mean_value_off - mean_value_on)/1e3
    return result

def localize_MLE(rois_list, dataset_FWHM):
    if rois_list.size == 0:
        return None
    else:
        roi_rad = rois_list[0]["roi"].shape[0] // 2
        localizations = np.zeros(
            (len(rois_list)),
            dtype=[
                ("id", np.uint64),
                ("t_peak", np.float64),
                ("t_on", np.float64),
                ("t_off", np.float64),
                ("ON_t", np.float64),
                ("double", np.uint8),
                # positives
                ("x", np.float64),
                ("x2", np.float64),
                ("y", np.float64),
                ("y2", np.float64),
                ("x_p", np.float64),
                ("y_p", np.float64),
                ("I", np.float32),
                ("FWHM", np.float32),
                ("rms", np.float32),
                ("E_total", np.uint16),
                ("sub_x", np.float64),
                ("sub_y", np.float64),
                ("t_1st", np.float64),
                # negatives
                ("x_n", np.float64),
                ("x_n2", np.float64),
                ("y_n", np.float64),
                ("y_n2", np.float64),
                ("x_np", np.float64),
                ("y_np", np.float64),
                ("I_n", np.float32),
                ("FWHM_n", np.float32),
                ("rms_n", np.float32),
                ("E_total_n", np.uint16),
                ("sub_x_n", np.float64),
                ("sub_y_n", np.float64),
                ("t_last", np.float64),
                # roi
                ("roi_event_times", np.uint64, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("roi_event_times_n", np.uint64, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("roi", np.uint16, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
                ("roi_n", np.uint16, (roi_rad * 2 + 1, roi_rad * 2 + 1)),
            ],
        )

        # print('Found '+str(len(rois_list))+' blinks, fitting...\n')
        id_to_remove = []
        for id in prange(len(rois_list)):
            if not rois_list[id]["roi"].any() or not rois_list[id]["roi_n"].any():
                continue
            if np.sum(rois_list[id]["roi"]) < 30:
                id_to_remove.append(id)
                continue
            fit_result, rms = fit_gaussian(rois_list[id]["roi"], dataset_FWHM=dataset_FWHM)
            fit_result_n, rms_n = fit_gaussian(
                rois_list[id]["roi_n"], dataset_FWHM=dataset_FWHM
            )

            roi_ft = np.fft.fft2(rois_list[id]["roi"])
            roi_ft_n = np.fft.fft2(rois_list[id]["roi_n"])
            if rms == 5:
                id_to_remove.append(id)
                continue

            time_stats = calculate_mean_and_subtract(rois_list[id], filter=True)
                
            if fit_result.shape[0] == 8 and fit_result_n.shape[0] == 8:
                fit_results_1 = fit_result[:4]
                fit_results_2 = fit_result[4:]
                fit_results_1_n = fit_result_n[:4]
                fit_results_2_n = fit_result_n[4:]
                y_pos, x_pos = (
                    rois_list[id]["rel_peak"][0] + fit_results_1[1] - roi_rad,
                    rois_list[id]["rel_peak"][0] + fit_results_2[1] - roi_rad,
                ), (
                    rois_list[id]["rel_peak"][1] + fit_results_1[2] - roi_rad,
                    rois_list[id]["rel_peak"][1] + fit_results_2[2] - roi_rad,
                )
                y_pos_n, x_pos_n = (
                    rois_list[id]["rel_peak"][0] + fit_results_1_n[1] - roi_rad,
                    rois_list[id]["rel_peak"][0] + fit_results_2_n[1] - roi_rad,
                ), (
                    rois_list[id]["rel_peak"][1] + fit_results_1_n[2] - roi_rad,
                    rois_list[id]["rel_peak"][1] + fit_results_2_n[2] - roi_rad,
                )
                y_posp, x_posp = (
                    rois_list[id]["rel_peak"][0]
                    + est_coord(roi_ft, (1, 0), roi_rad)
                    - roi_rad,
                    rois_list[id]["rel_peak"][1]
                    + est_coord(roi_ft, (0, 1), roi_rad)
                    - roi_rad,
                )
                y_pos_np, x_pos_np = (
                    rois_list[id]["rel_peak"][0]
                    + est_coord(roi_ft_n, (1, 0), roi_rad)
                    - roi_rad,
                    rois_list[id]["rel_peak"][1]
                    + est_coord(roi_ft_n, (0, 1), roi_rad)
                    - roi_rad,
                )

                """write the localizations to ndarray"""
                localizations[id] = (
                    id,
                    rois_list[id]["t_peak"],
                    time_stats[0],
                    time_stats[1],
                    time_stats[2],
                    1,
                    x_pos[0],
                    x_pos[1],
                    y_pos[0],
                    y_pos[1],
                    0,
                    0,
                    fit_result[0],
                    fit_result[3],
                    rms,
                    rois_list[id]["total_events_roi"],
                    fit_result[1],
                    fit_result[2],
                    rois_list[id]["t_1st"],
                    x_pos_n[0],
                    x_pos_n[1],
                    y_pos_n[0],
                    y_pos_n[1],
                    0,
                    0,
                    fit_result_n[0],
                    fit_result_n[3],
                    rms_n,
                    rois_list[id]["total_neg_events_roi"],
                    fit_result_n[1],
                    fit_result_n[2],
                    rois_list[id]["t_last"],
                    rois_list[id]["roi_event_times"][0],
                    rois_list[id]["roi_event_times"][1],
                    rois_list[id]["roi"],
                    rois_list[id]["roi_n"],
                )

            elif fit_result.shape[0] == 8 and fit_result_n.shape[0] != 8:
                id_to_remove.append(id)
                continue
            else:
                y_pos, x_pos = (
                    rois_list[id]["rel_peak"][0] + fit_result[1] - roi_rad,
                    rois_list[id]["rel_peak"][1] + fit_result[2] - roi_rad,
                )
                y_pos_n, x_pos_n = (
                    rois_list[id]["rel_peak"][0] + fit_result_n[1] - roi_rad,
                    rois_list[id]["rel_peak"][1] + fit_result_n[2] - roi_rad,
                )

                y_posp, x_posp = (
                    rois_list[id]["rel_peak"][0]
                    + est_coord(roi_ft, (1, 0), roi_rad)
                    - roi_rad,
                    rois_list[id]["rel_peak"][1]
                    + est_coord(roi_ft, (0, 1), roi_rad)
                    - roi_rad,
                )
                y_pos_np, x_pos_np = (
                    rois_list[id]["rel_peak"][0]
                    + est_coord(roi_ft_n, (1, 0), roi_rad)
                    - roi_rad,
                    rois_list[id]["rel_peak"][1]
                    + est_coord(roi_ft_n, (0, 1), roi_rad)
                    - roi_rad,
                )

                """write the localizations to ndarray"""
                localizations[id] = (
                    id,
                    rois_list[id]["t_peak"],
                    time_stats[0],
                    time_stats[1],
                    time_stats[2],
                    0,
                    x_pos,
                    0,
                    y_pos,
                    0,
                    x_posp,
                    y_posp,
                    fit_result[0],
                    fit_result[3],
                    rms,
                    rois_list[id]["total_events_roi"],
                    fit_result[1],
                    fit_result[2],
                    rois_list[id]["t_1st"],
                    x_pos_n,
                    0,
                    y_pos_n,
                    0,
                    x_pos_np,
                    y_pos_np,
                    fit_result_n[0],
                    fit_result_n[3],
                    rms_n,
                    rois_list[id]["total_neg_events_roi"],
                    fit_result_n[1],
                    fit_result_n[2],
                    rois_list[id]["t_last"],
                    rois_list[id]["roi_event_times"][0],
                    rois_list[id]["roi_event_times"][1],
                    rois_list[id]["roi"],
                    rois_list[id]["roi_n"],
                )
        localizations = np.delete(localizations, np.asarray(id_to_remove, dtype=np.uint64), axis=0)

    return localizations
