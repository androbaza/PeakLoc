from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class BeadSigmaEstimate:
    dataset_name: str
    sensitivity: str
    pixel_size_nm: float
    roi_radius: int
    fit_model: str
    bead_count: int
    median_sigma_px: float
    mad_sigma_px: float
    generated_at: str


def estimate_sigma_from_rois(
    rois: np.ndarray,
    *,
    pixel_size_nm: float,
    dataset_name: str,
    sensitivity: str,
    roi_radius: int | None = None,
    min_total_events: float = 10.0,
) -> BeadSigmaEstimate:
    count_images = _as_count_images(rois)
    sigmas = []
    for image in count_images:
        if float(np.sum(image)) < min_total_events:
            continue
        result = _fit_single_bead(image)
        if result is None:
            continue
        sigmas.append(result)

    if not sigmas:
        raise ValueError(
            "No bead ROIs passed the minimum-event and fit-quality filters"
        )

    sigma_values = np.asarray(sigmas, dtype=np.float64)
    median_sigma = float(np.median(sigma_values))
    mad_sigma = float(np.median(np.abs(sigma_values - median_sigma)))
    radius = count_images.shape[1] // 2 if roi_radius is None else roi_radius
    return BeadSigmaEstimate(
        dataset_name=dataset_name,
        sensitivity=sensitivity,
        pixel_size_nm=pixel_size_nm,
        roi_radius=radius,
        fit_model="least_squares_integrated_gaussian_approx",
        bead_count=int(sigma_values.size),
        median_sigma_px=median_sigma,
        mad_sigma_px=mad_sigma,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate empirical PeakLoc sigma_psf_px from isolated bead ROIs."
    )
    parser.add_argument("rois_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration_scripts/bead_sigma_estimate.json"),
    )
    parser.add_argument(
        "--dataset-name", default="Beads40nm_coverslip_standard_sensitivity"
    )
    parser.add_argument("--sensitivity", default="standard")
    parser.add_argument("--pixel-size-nm", type=float, default=67.0)
    parser.add_argument("--min-total-events", type=float, default=10.0)
    args = parser.parse_args()

    rois = np.load(args.rois_path)
    estimate = estimate_sigma_from_rois(
        rois,
        pixel_size_nm=args.pixel_size_nm,
        dataset_name=args.dataset_name,
        sensitivity=args.sensitivity,
        min_total_events=args.min_total_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(estimate), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Median sigma_psf_px: {estimate.median_sigma_px:.3f}")
    print(f"Wrote estimate to {args.output}")


def _as_count_images(rois: np.ndarray) -> np.ndarray:
    if rois.dtype.names and "roi" in rois.dtype.names:
        return np.asarray(rois["roi"], dtype=np.float64)
    if rois.ndim == 2:
        return np.asarray(rois[np.newaxis, :, :], dtype=np.float64)
    if rois.ndim == 3:
        return np.asarray(rois, dtype=np.float64)
    raise ValueError("Expected a 2D ROI, a stack of 2D ROIs, or structured ROIs")


def _fit_single_bead(image: np.ndarray) -> float | None:
    height, center_x, center_y, sigma, background = _initial_guess(image)
    if height <= 0:
        return None

    yy, xx = np.indices(image.shape)

    def residual(params: np.ndarray) -> np.ndarray:
        fit_height, fit_x, fit_y, fit_sigma, fit_bg = params
        model = fit_bg + fit_height * np.exp(
            -(
                ((xx - fit_x) ** 2 + (yy - fit_y) ** 2)
                / (2.0 * max(fit_sigma, 1e-6) ** 2)
            )
        )
        return np.ravel(model - image)

    bounds = (
        [0.0, 0.0, 0.0, 0.4, 0.0],
        [
            max(float(image.max()) * 2.0, 1.0),
            float(image.shape[1] - 1),
            float(image.shape[0] - 1),
            float(max(image.shape) / 2),
            max(float(np.median(image)) * 3.0, 1.0),
        ],
    )
    result = least_squares(
        residual,
        np.asarray([height, center_x, center_y, sigma, background]),
        bounds=bounds,
        method="trf",
    )
    if not result.success:
        return None
    return float(result.x[3])


def _initial_guess(image: np.ndarray) -> tuple[float, float, float, float, float]:
    background = _border_median(image)
    signal = np.clip(image - background, 0, None)
    total_signal = float(np.sum(signal))
    if total_signal <= 0:
        center_y = (image.shape[0] - 1) / 2
        center_x = (image.shape[1] - 1) / 2
        return 0.0, center_x, center_y, 2.0, background
    yy, xx = np.indices(image.shape)
    center_x = float(np.sum(xx * signal) / total_signal)
    center_y = float(np.sum(yy * signal) / total_signal)
    radius2 = ((xx - center_x) ** 2 + (yy - center_y) ** 2) * signal
    sigma = float(np.sqrt(np.sum(radius2) / (2.0 * total_signal)))
    return float(signal.max()), center_x, center_y, max(sigma, 0.4), background


def _border_median(image: np.ndarray) -> float:
    border = np.concatenate(
        [image[0, :], image[-1, :], image[1:-1, 0], image[1:-1, -1]]
    )
    return float(np.median(border))


if __name__ == "__main__":
    main()
