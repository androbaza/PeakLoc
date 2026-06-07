from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from localization_scripts.calibration import EventCalibration, RoiCalibration
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.psf_model import (
    finite_difference_psf_derivatives,
    pixel_integrated_gaussian,
)

MIN_MU = 1e-9
MIN_PARAM = 1e-6


@dataclass(frozen=True)
class JointPoissonFitResult:
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    cov_xy: float
    A_pos: float
    A_neg: float
    bg_pos_local: float
    bg_neg_local: float
    bg_pos_cal_sum: float
    bg_neg_cal_sum: float
    sigma_psf_px: float
    nll: float
    nll_per_event: float
    fit_success: bool
    fit_status: str
    fit_cond: float
    calibration_id: str
    calibrated_background: bool
    uncertainty_mode: str
    hot_pixel_count: int
    valid_pixel_count: int


def fit_joint_poisson_roi(
    roi_record: Any,
    calibration: EventCalibration,
    config: PeakLocConfig,
) -> JointPoissonFitResult:
    if config.fit_sigma:
        raise NotImplementedError(
            "fit_sigma=True is reserved for the later bead-validated model"
        )

    roi_pos = np.asarray(_field(roi_record, "roi"), dtype=np.float64)
    roi_neg = np.asarray(_field(roi_record, "roi_n"), dtype=np.float64)
    sigma_psf_px = _sigma_psf_px(config)
    roi_radius = roi_pos.shape[0] // 2
    roi_y0 = int(_field(roi_record, "roi_y0", default=0))
    roi_x0 = int(_field(roi_record, "roi_x0", default=0))
    dt_pos_s = float(_field(roi_record, "dt_pos_s", default=0.0))
    dt_neg_s = float(_field(roi_record, "dt_neg_s", default=0.0))
    roi_calibration = calibration.get_roi_background(
        roi_y0,
        roi_x0,
        roi_radius,
        dt_pos_s,
        dt_neg_s,
    )
    valid_mask = roi_calibration.valid_mask
    if roi_calibration.valid_pixel_count < config.min_valid_pixels:
        return _failed_fit_result(
            status="too_few_valid_pixels",
            sigma_psf_px=sigma_psf_px,
            roi_calibration=roi_calibration,
            calibration=calibration,
            config=config,
        )
    initial = _initial_parameters(
        roi_pos,
        roi_neg,
        roi_calibration.bg_pos,
        roi_calibration.bg_neg,
        sigma_psf_px,
        valid_mask,
        config.background_mode,
    )
    bounds = [
        (0.0, float(roi_pos.shape[1] - 1)),
        (0.0, float(roi_pos.shape[0] - 1)),
        (np.log(MIN_PARAM), np.log(max(float(np.sum(roi_pos)) * 2.0, 1.0))),
        (np.log(MIN_PARAM), np.log(max(float(np.sum(roi_neg)) * 2.0, 1.0))),
        (np.log(MIN_PARAM), np.log(max(float(np.max(roi_pos)) * 2.0, 1.0))),
        (np.log(MIN_PARAM), np.log(max(float(np.max(roi_neg)) * 2.0, 1.0))),
    ]

    def objective(params: np.ndarray) -> float:
        psf = pixel_integrated_gaussian(
            _shape_2d(roi_pos), params[0], params[1], sigma_psf_px
        )
        mu_pos, mu_neg = _mean_maps(
            params,
            psf,
            roi_calibration.bg_pos,
            roi_calibration.bg_neg,
            config.background_mode,
        )
        return _poisson_nll(roi_pos, mu_pos, valid_mask) + _poisson_nll(
            roi_neg,
            mu_neg,
            valid_mask,
        )

    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    params = np.asarray(result.x, dtype=np.float64)
    psf = pixel_integrated_gaussian(
        _shape_2d(roi_pos), params[0], params[1], sigma_psf_px
    )
    mu_pos, mu_neg = _mean_maps(
        params,
        psf,
        roi_calibration.bg_pos,
        roi_calibration.bg_neg,
        config.background_mode,
    )
    nll = objective(params)
    event_count = max(
        float(np.sum(roi_pos[valid_mask]) + np.sum(roi_neg[valid_mask])), 1.0
    )
    covariance, condition = _estimate_covariance(
        params,
        psf,
        mu_pos,
        mu_neg,
        valid_mask,
        sigma_psf_px,
        config.max_fit_cond,
        config.background_mode,
    )
    status = str(result.message)
    if condition > config.max_fit_cond:
        status = f"{status}; fisher condition exceeded {config.max_fit_cond:g}"
    fit_success = bool(result.success and condition <= config.max_fit_cond)
    return JointPoissonFitResult(
        x=roi_x0 + float(params[0]),
        y=roi_y0 + float(params[1]),
        sigma_x=float(np.sqrt(max(covariance[0, 0], 0.0))),
        sigma_y=float(np.sqrt(max(covariance[1, 1], 0.0))),
        cov_xy=float(covariance[0, 1]),
        A_pos=float(np.exp(params[2])),
        A_neg=float(np.exp(params[3])),
        bg_pos_local=float(np.exp(params[4])),
        bg_neg_local=float(np.exp(params[5])),
        bg_pos_cal_sum=float(np.sum(roi_calibration.bg_pos[valid_mask])),
        bg_neg_cal_sum=float(np.sum(roi_calibration.bg_neg[valid_mask])),
        sigma_psf_px=sigma_psf_px,
        nll=float(nll),
        nll_per_event=float(nll / event_count),
        fit_success=fit_success,
        fit_status=status,
        fit_cond=float(condition),
        calibration_id=calibration.calibration_id,
        calibrated_background=bool(
            calibration.calibrated and config.background_mode != "local_only"
        ),
        uncertainty_mode=(
            "model_based_calibrated"
            if calibration.calibrated and config.background_mode != "local_only"
            else "model_based_uncalibrated"
        ),
        hot_pixel_count=roi_calibration.hot_pixel_count,
        valid_pixel_count=roi_calibration.valid_pixel_count,
    )


def _initial_parameters(
    roi_pos: np.ndarray,
    roi_neg: np.ndarray,
    bg_pos_cal: np.ndarray,
    bg_neg_cal: np.ndarray,
    sigma_psf_px: float,
    valid_mask: np.ndarray,
    background_mode: str,
) -> np.ndarray:
    if background_mode == "local_only":
        corrected_pos = roi_pos
        corrected_neg = roi_neg
    else:
        corrected_pos = np.clip(roi_pos - bg_pos_cal, 0, None)
        corrected_neg = np.clip(roi_neg - bg_neg_cal, 0, None)
    combined = np.where(valid_mask, corrected_pos + corrected_neg, 0.0)
    local_x, local_y = _center_of_mass(combined)
    bg_pos = max(_border_median(corrected_pos), MIN_PARAM)
    bg_neg = max(_border_median(corrected_neg), MIN_PARAM)
    psf = pixel_integrated_gaussian(_shape_2d(roi_pos), local_x, local_y, sigma_psf_px)
    amp_pos = max(float(np.sum(np.clip(corrected_pos - bg_pos, 0, None))), MIN_PARAM)
    amp_neg = max(float(np.sum(np.clip(corrected_neg - bg_neg, 0, None))), MIN_PARAM)
    if np.max(psf) <= 0:
        amp_pos = max(float(np.sum(roi_pos)), MIN_PARAM)
        amp_neg = max(float(np.sum(roi_neg)), MIN_PARAM)
    return np.asarray(
        [
            local_x,
            local_y,
            np.log(amp_pos),
            np.log(amp_neg),
            np.log(bg_pos),
            np.log(bg_neg),
        ],
        dtype=np.float64,
    )


def _mean_maps(
    params: np.ndarray,
    psf: np.ndarray,
    bg_pos_cal: np.ndarray,
    bg_neg_cal: np.ndarray,
    background_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    amp_pos = np.exp(params[2])
    amp_neg = np.exp(params[3])
    bg_pos_local = np.exp(params[4])
    bg_neg_local = np.exp(params[5])
    bg_pos = _background_map(bg_pos_cal, bg_pos_local, background_mode)
    bg_neg = _background_map(bg_neg_cal, bg_neg_local, background_mode)
    mu_pos = bg_pos + amp_pos * psf
    mu_neg = bg_neg + amp_neg * psf
    return np.maximum(mu_pos, MIN_MU), np.maximum(mu_neg, MIN_MU)


def _background_map(
    bg_cal: np.ndarray,
    bg_local: float,
    background_mode: str,
) -> np.ndarray:
    if background_mode == "calibrated_only":
        return bg_cal
    if background_mode == "local_only":
        return np.full_like(bg_cal, bg_local)
    if background_mode == "calibrated_plus_local":
        return bg_cal + bg_local
    raise ValueError(f"Unsupported background_mode: {background_mode}")


def _estimate_covariance(
    params: np.ndarray,
    psf: np.ndarray,
    mu_pos: np.ndarray,
    mu_neg: np.ndarray,
    valid_mask: np.ndarray,
    sigma_psf_px: float,
    max_fit_cond: float,
    background_mode: str,
) -> tuple[np.ndarray, float]:
    dpsf_dx, dpsf_dy = finite_difference_psf_derivatives(
        _shape_2d(psf),
        params[0],
        params[1],
        sigma_psf_px,
    )
    amp_pos = np.exp(params[2])
    amp_neg = np.exp(params[3])
    bg_pos = np.exp(params[4])
    bg_neg = np.exp(params[5])
    pos_bg_derivative = (
        np.zeros_like(psf)
        if background_mode == "calibrated_only"
        else np.full_like(psf, bg_pos)
    )
    neg_bg_derivative = (
        np.zeros_like(psf)
        if background_mode == "calibrated_only"
        else np.full_like(psf, bg_neg)
    )
    derivs_pos = [
        amp_pos * dpsf_dx,
        amp_pos * dpsf_dy,
        amp_pos * psf,
        np.zeros_like(psf),
        pos_bg_derivative,
        np.zeros_like(psf),
    ]
    derivs_neg = [
        amp_neg * dpsf_dx,
        amp_neg * dpsf_dy,
        np.zeros_like(psf),
        amp_neg * psf,
        np.zeros_like(psf),
        neg_bg_derivative,
    ]
    fisher = np.zeros((6, 6), dtype=np.float64)
    for row in range(6):
        for col in range(6):
            fisher[row, col] += np.sum(
                derivs_pos[row][valid_mask]
                * derivs_pos[col][valid_mask]
                / mu_pos[valid_mask]
            )
            fisher[row, col] += np.sum(
                derivs_neg[row][valid_mask]
                * derivs_neg[col][valid_mask]
                / mu_neg[valid_mask]
            )
    active_indices = _active_fisher_indices(background_mode)
    active_fisher = fisher[np.ix_(active_indices, active_indices)]
    condition = float(np.linalg.cond(active_fisher))
    covariance = np.linalg.pinv(fisher, rcond=1.0 / max_fit_cond)
    return covariance, condition


def _active_fisher_indices(background_mode: str) -> list[int]:
    if background_mode == "calibrated_only":
        return [0, 1, 2, 3]
    if background_mode in {"calibrated_plus_local", "local_only"}:
        return [0, 1, 2, 3, 4, 5]
    raise ValueError(f"Unsupported background_mode: {background_mode}")


def _failed_fit_result(
    *,
    status: str,
    sigma_psf_px: float,
    roi_calibration: RoiCalibration,
    calibration: EventCalibration,
    config: PeakLocConfig,
) -> JointPoissonFitResult:
    valid_mask = roi_calibration.valid_mask
    return JointPoissonFitResult(
        x=np.nan,
        y=np.nan,
        sigma_x=np.nan,
        sigma_y=np.nan,
        cov_xy=np.nan,
        A_pos=np.nan,
        A_neg=np.nan,
        bg_pos_local=np.nan,
        bg_neg_local=np.nan,
        bg_pos_cal_sum=float(np.sum(roi_calibration.bg_pos[valid_mask])),
        bg_neg_cal_sum=float(np.sum(roi_calibration.bg_neg[valid_mask])),
        sigma_psf_px=sigma_psf_px,
        nll=np.nan,
        nll_per_event=np.nan,
        fit_success=False,
        fit_status=status,
        fit_cond=np.inf,
        calibration_id=calibration.calibration_id,
        calibrated_background=bool(
            calibration.calibrated and config.background_mode != "local_only"
        ),
        uncertainty_mode=(
            "model_based_calibrated"
            if calibration.calibrated and config.background_mode != "local_only"
            else "model_based_uncalibrated"
        ),
        hot_pixel_count=roi_calibration.hot_pixel_count,
        valid_pixel_count=roi_calibration.valid_pixel_count,
    )


def _poisson_nll(
    counts: np.ndarray, means: np.ndarray, valid_mask: np.ndarray
) -> float:
    valid_counts = counts[valid_mask]
    valid_means = means[valid_mask]
    return float(
        np.sum(
            valid_means - valid_counts * np.log(valid_means) + gammaln(valid_counts + 1)
        )
    )


def _center_of_mass(image: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(image))
    if total <= 0:
        return (image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0
    yy, xx = np.indices(image.shape)
    return float(np.sum(xx * image) / total), float(np.sum(yy * image) / total)


def _border_median(image: np.ndarray) -> float:
    border = np.concatenate(
        [image[0, :], image[-1, :], image[1:-1, 0], image[1:-1, -1]]
    )
    return float(np.median(border))


def _sigma_psf_px(config: PeakLocConfig) -> float:
    if config.sigma_psf_px is not None:
        return config.sigma_psf_px
    return config.dataset_fwhm / 2.35


def _shape_2d(array: np.ndarray) -> tuple[int, int]:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    return int(array.shape[0]), int(array.shape[1])


def _field(record: Any, name: str, *, default: Any | None = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    if getattr(record, "dtype", None) is not None and record.dtype.names:
        if name in record.dtype.names:
            return record[name]
        return default
    return getattr(record, name, default)
