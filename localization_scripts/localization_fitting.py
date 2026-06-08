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
    qc_table: np.ndarray


@dataclass(frozen=True)
class LocalizationFilterResult:
    accepted: np.ndarray
    qc_table: np.ndarray


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
    filter_result = evaluate_poisson_localization_filters(attempted, config)
    return LocalizationTables(
        attempted=attempted,
        filtered=filter_result.accepted,
        qc_table=filter_result.qc_table,
    )


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
        roi_rad = _roi_radius_from_dtype(rois)
        if roi_rad is None:
            roi_rad = config.roi_radius
        return np.empty(0, dtype=_joint_poisson_localization_dtype(roi_rad))
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
    return evaluate_poisson_localization_filters(localizations, config).accepted


def evaluate_poisson_localization_filters(
    localizations: np.ndarray,
    config: PeakLocConfig,
) -> LocalizationFilterResult:
    if localizations.size == 0:
        return LocalizationFilterResult(
            accepted=localizations,
            qc_table=np.empty(0, dtype=localization_qc_dtype()),
        )

    uncertainty_px = localization_uncertainty_px(localizations)
    uncertainty_nm = uncertainty_px * config.optical_pixel_size_nm
    finite_position = np.isfinite(localizations["x"]) & np.isfinite(localizations["y"])
    finite_uncertainty = (
        np.isfinite(localizations["sigma_x"])
        & np.isfinite(localizations["sigma_y"])
        & np.isfinite(localizations["cov_xy"])
        & np.isfinite(uncertainty_px)
    )
    positive_uncertainty = (localizations["sigma_x"] > 0) & (
        localizations["sigma_y"] > 0
    )
    fit_cond_ok = localizations["fit_cond"] < config.max_fit_cond
    valid_pixels_ok = localizations["valid_pixel_count"] >= config.min_valid_pixels

    uncertainty_ok = finite_uncertainty.copy()

    if config.max_localization_uncertainty_px is not None:
        uncertainty_ok &= uncertainty_px <= config.max_localization_uncertainty_px

    if config.max_localization_uncertainty_nm is not None:
        max_uncertainty_px = (
            config.max_localization_uncertainty_nm / config.optical_pixel_size_nm
        )
        uncertainty_ok &= uncertainty_px <= max_uncertainty_px

    keep = (
        localizations["fit_success"]
        & finite_position
        & finite_uncertainty
        & positive_uncertainty
        & fit_cond_ok
        & valid_pixels_ok
        & uncertainty_ok
    )

    qc_table = np.zeros(localizations.size, dtype=localization_qc_dtype())
    qc_table["id"] = _field_or_default(
        localizations, "id", np.arange(localizations.size)
    )
    qc_table["accepted"] = keep
    qc_table["fit_success"] = localizations["fit_success"]
    qc_table["finite_position"] = finite_position
    qc_table["finite_uncertainty"] = finite_uncertainty
    qc_table["positive_uncertainty"] = positive_uncertainty
    qc_table["fit_cond_ok"] = fit_cond_ok
    qc_table["valid_pixels_ok"] = valid_pixels_ok
    qc_table["uncertainty_px"] = uncertainty_px
    qc_table["uncertainty_nm"] = uncertainty_nm
    qc_table["uncertainty_ok"] = uncertainty_ok
    qc_table["fit_cond"] = localizations["fit_cond"]
    qc_table["valid_pixel_count"] = localizations["valid_pixel_count"]
    qc_table["nll_per_event"] = _field_or_default(
        localizations, "nll_per_event", np.nan
    )
    qc_table["E_total"] = _field_or_default(localizations, "E_total", 0)
    qc_table["E_total_n"] = _field_or_default(localizations, "E_total_n", 0)
    qc_table["primary_rejection_reason"] = _primary_rejection_reasons(
        keep,
        localizations["fit_success"],
        finite_position,
        finite_uncertainty,
        positive_uncertainty,
        fit_cond_ok,
        valid_pixels_ok,
        uncertainty_ok,
    )
    return LocalizationFilterResult(
        accepted=localizations[keep],
        qc_table=qc_table,
    )


def localization_qc_dtype() -> list[tuple[str, object]]:
    return [
        ("id", np.uint64),
        ("accepted", np.bool_),
        ("fit_success", np.bool_),
        ("finite_position", np.bool_),
        ("finite_uncertainty", np.bool_),
        ("positive_uncertainty", np.bool_),
        ("fit_cond_ok", np.bool_),
        ("valid_pixels_ok", np.bool_),
        ("uncertainty_px", np.float64),
        ("uncertainty_nm", np.float64),
        ("uncertainty_ok", np.bool_),
        ("fit_cond", np.float64),
        ("valid_pixel_count", np.uint32),
        ("nll_per_event", np.float64),
        ("E_total", np.uint64),
        ("E_total_n", np.uint64),
        ("primary_rejection_reason", "U64"),
    ]


def _primary_rejection_reasons(
    accepted: np.ndarray,
    fit_success: np.ndarray,
    finite_position: np.ndarray,
    finite_uncertainty: np.ndarray,
    positive_uncertainty: np.ndarray,
    fit_cond_ok: np.ndarray,
    valid_pixels_ok: np.ndarray,
    uncertainty_ok: np.ndarray,
) -> np.ndarray:
    reasons = np.full(accepted.size, "accepted", dtype="U64")
    rejected = ~accepted
    reasons[rejected & ~fit_success] = "fit_failed"
    reasons[rejected & fit_success & ~finite_position] = "invalid_position"
    reasons[rejected & fit_success & finite_position & ~finite_uncertainty] = (
        "invalid_uncertainty"
    )
    reasons[
        rejected
        & fit_success
        & finite_position
        & finite_uncertainty
        & ~positive_uncertainty
    ] = "invalid_uncertainty"
    reasons[
        rejected
        & fit_success
        & finite_position
        & finite_uncertainty
        & positive_uncertainty
        & ~fit_cond_ok
    ] = "fit_condition"
    reasons[
        rejected
        & fit_success
        & finite_position
        & finite_uncertainty
        & positive_uncertainty
        & fit_cond_ok
        & ~valid_pixels_ok
    ] = "valid_pixels"
    reasons[
        rejected
        & fit_success
        & finite_position
        & finite_uncertainty
        & positive_uncertainty
        & fit_cond_ok
        & valid_pixels_ok
        & ~uncertainty_ok
    ] = "uncertainty"
    return reasons


def _field_or_default(
    localizations: np.ndarray, field_name: str, default: object
) -> np.ndarray:
    if (
        localizations.dtype.names is not None
        and field_name in localizations.dtype.names
    ):
        return localizations[field_name]
    if np.isscalar(default):
        return np.full(localizations.size, default)
    return np.asarray(default)


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
