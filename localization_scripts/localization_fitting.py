from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from localization_scripts.calibration import EventCalibration
from localization_scripts.event_array_processing import slice_data
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.poisson_fitting import fit_joint_poisson_roi

FWHM_FROM_SIGMA = 2.354820045


@dataclass(frozen=True)
class LocalizationTables:
    attempted: np.ndarray
    filtered: np.ndarray


def concatenate_locs(localized_data: list[np.ndarray]) -> np.ndarray:
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


def localize_rois(
    rois: np.ndarray,
    config: PeakLocConfig,
    calibration: EventCalibration | None = None,
) -> np.ndarray:
    return localize_rois_with_attempts(rois, config, calibration).filtered


def localize_rois_with_attempts(
    rois: np.ndarray,
    config: PeakLocConfig,
    calibration: EventCalibration | None = None,
) -> LocalizationTables:
    if config.fit_model != "poisson_joint":
        raise ValueError(f"Unsupported fit_model: {config.fit_model}")
    if calibration is None:
        raise ValueError("calibration is required for poisson_joint localization")
    attempted = perform_joint_poisson_localization_parallel(rois, config, calibration)
    filtered = filter_poisson_localizations(attempted, config)
    return LocalizationTables(attempted=attempted, filtered=filtered)


def perform_joint_poisson_localization_parallel(
    rois: np.ndarray,
    config: PeakLocConfig,
    calibration: EventCalibration,
) -> np.ndarray:
    if rois.size == 0:
        roi_rad = _roi_radius_from_dtype(rois)
        if roi_rad is None:
            return np.array([])
        return np.empty(0, dtype=_joint_poisson_localization_dtype(roi_rad))
    rois_split = slice_data(rois, config.num_cores)
    results = Parallel(n_jobs=config.num_cores)(
        delayed(localize_joint_poisson)(chunk, config, calibration)
        for chunk in rois_split
    )
    localized_data = [
        localized_chunk
        for localized_chunk in results
        if localized_chunk is not None and localized_chunk.size > 0
    ]
    if not localized_data:
        return np.array([])
    return concatenate_locs(localized_data)


def localize_joint_poisson(
    rois_list: np.ndarray,
    config: PeakLocConfig,
    calibration: EventCalibration,
) -> np.ndarray | None:
    if rois_list.size == 0:
        return None

    roi_rad = rois_list[0]["roi"].shape[0] // 2
    localizations = np.zeros(
        (len(rois_list)),
        dtype=_joint_poisson_localization_dtype(roi_rad),
    )
    id_to_remove = []
    for row_id in range(len(rois_list)):
        roi_record = rois_list[row_id]
        if (
            roi_record["total_events_roi"] < config.min_events_pos
            or roi_record["total_neg_events_roi"] < config.min_events_neg
        ):
            id_to_remove.append(row_id)
            continue
        fit_result = fit_joint_poisson_roi(roi_record, calibration, config)
        roi_y0 = int(roi_record["roi_y0"])
        roi_x0 = int(roi_record["roi_x0"])
        fwhm_px = FWHM_FROM_SIGMA * fit_result.sigma_psf_px
        localizations[row_id] = (
            row_id,
            roi_record["t_peak"],
            0,
            fit_result.x,
            0.0,
            fit_result.y,
            0.0,
            fit_result.A_pos,
            fwhm_px,
            roi_record["total_events_roi"],
            roi_record["total_neg_events_roi"],
            fit_result.x - roi_x0,
            fit_result.y - roi_y0,
            roi_record["t_1st"],
            roi_record["t_last"],
            fit_result.sigma_x,
            fit_result.sigma_y,
            fit_result.cov_xy,
            fit_result.A_pos,
            fit_result.A_neg,
            fit_result.bg_pos_local,
            fit_result.bg_neg_local,
            fit_result.bg_pos_cal_sum,
            fit_result.bg_neg_cal_sum,
            fit_result.sigma_psf_px,
            fit_result.nll,
            fit_result.nll_per_event,
            fit_result.fit_success,
            fit_result.fit_status,
            fit_result.fit_cond,
            fit_result.calibration_id,
            fit_result.calibrated_background,
            fit_result.uncertainty_mode,
            fit_result.hot_pixel_count,
            fit_result.valid_pixel_count,
            roi_record["roi_event_times"][0],
            roi_record["roi_event_times"][1],
            roi_record["roi"],
            roi_record["roi_n"],
        )
    localizations = np.delete(
        localizations, np.asarray(id_to_remove, dtype=np.uint64), axis=0
    )
    return localizations


def localization_uncertainty_px(localizations: np.ndarray) -> np.ndarray:
    """Return worst-axis 1-sigma localization uncertainty in pixels."""
    var_x = np.asarray(localizations["sigma_x"], dtype=np.float64) ** 2
    var_y = np.asarray(localizations["sigma_y"], dtype=np.float64) ** 2
    cov_xy = np.asarray(localizations["cov_xy"], dtype=np.float64)

    trace = var_x + var_y
    determinant_term = np.sqrt((var_x - var_y) ** 2 + 4.0 * cov_xy**2)

    largest_eigenvalue = 0.5 * (trace + determinant_term)
    return np.sqrt(np.maximum(largest_eigenvalue, 0.0))


def filter_poisson_localizations(
    localizations: np.ndarray,
    config: PeakLocConfig,
) -> np.ndarray:
    if localizations.size == 0:
        return localizations
    keep = (
        localizations["fit_success"]
        & np.isfinite(localizations["x"])
        & np.isfinite(localizations["y"])
        & np.isfinite(localizations["sigma_x"])
        & np.isfinite(localizations["sigma_y"])
        & np.isfinite(localizations["cov_xy"])
        & (localizations["sigma_x"] > 0)
        & (localizations["sigma_y"] > 0)
        & (localizations["fit_cond"] < config.max_fit_cond)
        & (localizations["valid_pixel_count"] >= config.min_valid_pixels)
    )
    uncertainty_px = localization_uncertainty_px(localizations)
    keep &= np.isfinite(uncertainty_px)

    if config.max_localization_uncertainty_px is not None:
        keep &= uncertainty_px <= config.max_localization_uncertainty_px

    if config.max_localization_uncertainty_nm is not None:
        max_uncertainty_px = (
            config.max_localization_uncertainty_nm / config.optical_pixel_size_nm
        )
        keep &= uncertainty_px <= max_uncertainty_px

    return localizations[keep]


def _roi_radius_from_dtype(rois: np.ndarray) -> int | None:
    if rois.dtype.names is None or "roi" not in rois.dtype.names:
        return None
    roi_shape = rois.dtype["roi"].shape
    if len(roi_shape) != 2:
        return None
    return int(roi_shape[0] // 2)


def _joint_poisson_localization_dtype(roi_rad: int) -> list[tuple]:
    roi_shape = (roi_rad * 2 + 1, roi_rad * 2 + 1)
    return [
        ("id", np.uint64),
        ("t_peak", np.float64),
        ("double", np.uint8),
        ("x", np.float64),
        ("x2", np.float64),
        ("y", np.float64),
        ("y2", np.float64),
        ("I", np.float32),
        ("FWHM", np.float32),
        ("E_total", np.uint64),
        ("E_total_n", np.uint64),
        ("sub_x", np.float64),
        ("sub_y", np.float64),
        ("t_1st", np.float64),
        ("t_last", np.float64),
        ("sigma_x", np.float64),
        ("sigma_y", np.float64),
        ("cov_xy", np.float64),
        ("A_pos", np.float64),
        ("A_neg", np.float64),
        ("bg_pos_local", np.float64),
        ("bg_neg_local", np.float64),
        ("bg_pos_cal_sum", np.float64),
        ("bg_neg_cal_sum", np.float64),
        ("sigma_psf_px", np.float64),
        ("nll", np.float64),
        ("nll_per_event", np.float64),
        ("fit_success", np.bool_),
        ("fit_status", "U256"),
        ("fit_cond", np.float64),
        ("calibration_id", "U128"),
        ("calibrated_background", np.bool_),
        ("uncertainty_mode", "U64"),
        ("hot_pixel_count", np.uint32),
        ("valid_pixel_count", np.uint32),
        ("roi_event_times", np.uint64, roi_shape),
        ("roi_event_times_n", np.uint64, roi_shape),
        ("roi", np.uint32, roi_shape),
        ("roi_n", np.uint32, roi_shape),
    ]
