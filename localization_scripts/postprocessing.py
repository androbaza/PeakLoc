from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.drift import (
    DriftTrace,
    apply_drift_correction,
    estimate_drift_cross_correlation,
)
from localization_scripts.frc import (
    FRCResult,
    compute_frc_resolution_nm,
    split_localizations_for_frc,
)
from localization_scripts.pipeline_config import PeakLocConfig


def save_postprocessing_qc(
    localizations: np.ndarray,
    config: PeakLocConfig,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    drift = estimate_drift_cross_correlation(
        localizations,
        bins=10,
        render_pixel_size_px=1.0,
    )
    corrected = apply_drift_correction(localizations, drift)
    artifacts.append(_save_drift_trace(drift, output_dir / "drift_trace.png", config))
    artifacts.append(
        _save_render(
            localizations, output_dir / "render_before_drift_correction.png", config
        )
    )
    artifacts.append(
        _save_render(
            corrected, output_dir / "render_after_drift_correction.png", config
        )
    )

    locs_a, locs_b = split_localizations_for_frc(corrected)
    frc_result = compute_frc_resolution_nm(
        locs_a,
        locs_b,
        optical_pixel_size_nm=config.optical_pixel_size_nm,
        render_pixel_size_nm=max(config.optical_pixel_size_nm / 4, 1.0),
    )
    artifacts.append(_save_frc_curve(frc_result, output_dir / "frc_curve.png", config))
    summary_path = output_dir / "frc_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "resolution_nm": frc_result.resolution_nm,
                "threshold": frc_result.threshold,
                "warning": frc_result.warning,
                "drift_method": drift.method,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(summary_path)
    return artifacts


def _save_drift_trace(drift: DriftTrace, path: Path, config: PeakLocConfig) -> Path:
    fig, axis = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    if drift.time_us.size:
        axis.plot(drift.time_us, drift.dx_px, label="dx px")
        axis.plot(drift.time_us, drift.dy_px, label="dy px")
        axis.legend()
    else:
        axis.text(
            0.5, 0.5, drift.method, ha="center", va="center", transform=axis.transAxes
        )
    axis.set_title("Drift trace")
    axis.set_xlabel("time us")
    axis.set_ylabel("drift px")
    fig.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_render(localizations: np.ndarray, path: Path, config: PeakLocConfig) -> Path:
    fig, axis = plt.subplots(figsize=(4, 4), constrained_layout=True)
    if localizations.size and {"x", "y"}.issubset(localizations.dtype.names or ()):
        axis.hist2d(
            localizations["x"],
            localizations["y"],
            bins=100,
            range=[[0, config.sensor_width], [0, config.sensor_height]],
            cmap="cividis",
        )
        axis.invert_yaxis()
    else:
        axis.text(
            0.5,
            0.5,
            "No localizations",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    axis.set_title(path.stem.replace("_", " "))
    axis.set_xlabel("x px")
    axis.set_ylabel("y px")
    fig.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_frc_curve(result: FRCResult, path: Path, config: PeakLocConfig) -> Path:
    fig, axis = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    if result.spatial_frequency_per_nm.size:
        axis.plot(result.spatial_frequency_per_nm, result.frc, color="#0072B2")
        axis.axhline(result.threshold, color="#D55E00", linestyle="--")
    else:
        axis.text(
            0.5,
            0.5,
            result.warning or "No FRC data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title("FRC curve")
    axis.set_xlabel("spatial frequency 1/nm")
    axis.set_ylabel("FRC")
    fig.savefig(path, dpi=config.qc_static_dpi, bbox_inches="tight")
    plt.close(fig)
    return path
