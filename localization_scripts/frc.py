from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FRCResult:
    spatial_frequency_per_nm: np.ndarray
    frc: np.ndarray
    resolution_nm: float | None
    threshold: float
    warning: str | None = None


def split_localizations_for_frc(
    localizations: np.ndarray,
    *,
    method: Literal["random", "odd_even_time"] = "odd_even_time",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if localizations.size == 0:
        return localizations.copy(), localizations.copy()
    if method == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(localizations.size)
    elif method == "odd_even_time":
        if "t_peak" in (localizations.dtype.names or ()):
            order = np.argsort(localizations["t_peak"], kind="stable")
        elif "t" in (localizations.dtype.names or ()):
            order = np.argsort(localizations["t"], kind="stable")
        else:
            order = np.arange(localizations.size)
    else:
        raise ValueError("method must be 'random' or 'odd_even_time'")
    return localizations[order[::2]].copy(), localizations[order[1::2]].copy()


def compute_frc_resolution_nm(
    locs_a: np.ndarray,
    locs_b: np.ndarray,
    *,
    optical_pixel_size_nm: float,
    render_pixel_size_nm: float,
    threshold: float = 1 / 7,
) -> FRCResult:
    if optical_pixel_size_nm <= 0:
        raise ValueError("optical_pixel_size_nm must be positive")
    if render_pixel_size_nm <= 0:
        raise ValueError("render_pixel_size_nm must be positive")
    if locs_a.size < 2 or locs_b.size < 2:
        return FRCResult(
            spatial_frequency_per_nm=np.asarray([], dtype=np.float64),
            frc=np.asarray([], dtype=np.float64),
            resolution_nm=None,
            threshold=threshold,
            warning="too_few_localizations",
        )
    if not _has_fields(locs_a, {"x", "y"}) or not _has_fields(locs_b, {"x", "y"}):
        return FRCResult(
            spatial_frequency_per_nm=np.asarray([], dtype=np.float64),
            frc=np.asarray([], dtype=np.float64),
            resolution_nm=None,
            threshold=threshold,
            warning="missing_xy_fields",
        )

    image_a, image_b = _render_pair(
        locs_a,
        locs_b,
        optical_pixel_size_nm=optical_pixel_size_nm,
        render_pixel_size_nm=render_pixel_size_nm,
    )
    frequencies, frc_values = _frc_curve(
        image_a, image_b, render_pixel_size_nm=render_pixel_size_nm
    )
    if frequencies.size == 0:
        return FRCResult(
            spatial_frequency_per_nm=frequencies,
            frc=frc_values,
            resolution_nm=None,
            threshold=threshold,
            warning="too_small_render",
        )

    below = np.flatnonzero(
        (frequencies > 0) & np.isfinite(frc_values) & (frc_values < threshold)
    )
    if below.size:
        resolution_nm = float(1.0 / frequencies[below[0]])
        warning = None
    else:
        finite_frequencies = frequencies[frequencies > 0]
        resolution_nm = (
            None
            if finite_frequencies.size == 0
            else float(1.0 / np.max(finite_frequencies))
        )
        warning = "threshold_not_crossed"
    return FRCResult(
        spatial_frequency_per_nm=frequencies,
        frc=frc_values,
        resolution_nm=resolution_nm,
        threshold=threshold,
        warning=warning,
    )


def _render_pair(
    locs_a: np.ndarray,
    locs_b: np.ndarray,
    *,
    optical_pixel_size_nm: float,
    render_pixel_size_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    render_pixel_size_px = render_pixel_size_nm / optical_pixel_size_nm
    all_x = np.concatenate([locs_a["x"], locs_b["x"]])
    all_y = np.concatenate([locs_a["y"], locs_b["y"]])
    min_x = float(np.floor(np.min(all_x))) - 2
    min_y = float(np.floor(np.min(all_y))) - 2
    max_x = float(np.ceil(np.max(all_x))) + 2
    max_y = float(np.ceil(np.max(all_y))) + 2
    width = max(8, int(np.ceil((max_x - min_x) / render_pixel_size_px)) + 1)
    height = max(8, int(np.ceil((max_y - min_y) / render_pixel_size_px)) + 1)
    width = _next_power_of_two(width)
    height = _next_power_of_two(height)
    return (
        _render_localizations(
            locs_a, min_x, min_y, width, height, render_pixel_size_px
        ),
        _render_localizations(
            locs_b, min_x, min_y, width, height, render_pixel_size_px
        ),
    )


def _render_localizations(
    localizations: np.ndarray,
    min_x: float,
    min_y: float,
    width: int,
    height: int,
    render_pixel_size_px: float,
) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float64)
    x = np.floor((localizations["x"] - min_x) / render_pixel_size_px).astype(np.int64)
    y = np.floor((localizations["y"] - min_y) / render_pixel_size_px).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.add.at(image, (y[valid], x[valid]), 1.0)
    return image


def _frc_curve(
    image_a: np.ndarray, image_b: np.ndarray, *, render_pixel_size_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    fft_a = np.fft.fftshift(np.fft.fft2(image_a - np.mean(image_a)))
    fft_b = np.fft.fftshift(np.fft.fft2(image_b - np.mean(image_b)))
    height, width = image_a.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height, d=render_pixel_size_nm))
    fx = np.fft.fftshift(np.fft.fftfreq(width, d=render_pixel_size_nm))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    max_radius = np.max(radius)
    if max_radius <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    bins = np.linspace(0, max_radius, min(height, width) // 2 + 1)
    frequencies = []
    frc_values = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (radius >= low) & (radius < high)
        if np.count_nonzero(mask) == 0:
            continue
        numerator = np.sum(fft_a[mask] * np.conj(fft_b[mask]))
        denominator = np.sqrt(
            np.sum(np.abs(fft_a[mask]) ** 2) * np.sum(np.abs(fft_b[mask]) ** 2)
        )
        if denominator == 0:
            frc = np.nan
        else:
            frc = float(np.real(numerator) / denominator)
        frequencies.append(float(0.5 * (low + high)))
        frc_values.append(frc)
    return np.asarray(frequencies), np.asarray(frc_values)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _has_fields(array: np.ndarray, fields: set[str]) -> bool:
    return fields.issubset(array.dtype.names or ())
