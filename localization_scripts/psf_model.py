from __future__ import annotations

import numpy as np
from scipy.special import erf


def pixel_integrated_gaussian(
    shape: tuple[int, int],
    x: float,
    y: float,
    sigma_px: float,
) -> np.ndarray:
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    height, width = shape
    x_edges = np.arange(width + 1, dtype=np.float64) - 0.5
    y_edges = np.arange(height + 1, dtype=np.float64) - 0.5
    inv = 1.0 / (np.sqrt(2.0) * sigma_px)
    x_integral = 0.5 * (erf((x_edges[1:] - x) * inv) - erf((x_edges[:-1] - x) * inv))
    y_integral = 0.5 * (erf((y_edges[1:] - y) * inv) - erf((y_edges[:-1] - y) * inv))
    psf = np.outer(y_integral, x_integral)
    total = float(np.sum(psf))
    if total <= 0:
        raise ValueError("PSF integral over ROI is zero")
    return psf / total


def finite_difference_psf_derivatives(
    shape: tuple[int, int],
    x: float,
    y: float,
    sigma_px: float,
    *,
    step: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    x_plus = pixel_integrated_gaussian(shape, x + step, y, sigma_px)
    x_minus = pixel_integrated_gaussian(shape, x - step, y, sigma_px)
    y_plus = pixel_integrated_gaussian(shape, x, y + step, sigma_px)
    y_minus = pixel_integrated_gaussian(shape, x, y - step, sigma_px)
    return (x_plus - x_minus) / (2.0 * step), (y_plus - y_minus) / (2.0 * step)
