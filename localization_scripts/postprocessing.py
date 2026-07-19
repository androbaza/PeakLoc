from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.frc import (
    FRCResult,
    compute_frc_resolution_nm,
    split_localizations_for_frc,
)
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import (
    PLOT_COLORS,
    save_publication_figure,
    style_publication_axis,
)


def save_postprocessing_qc(
    localizations: np.ndarray,
    config: PeakLocConfig,
    figure_dir: Path,
    statistics_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    locs_a, locs_b = split_localizations_for_frc(localizations)
    frc_result = compute_frc_resolution_nm(
        locs_a,
        locs_b,
        optical_pixel_size_nm=config.optical_pixel_size_nm,
        render_pixel_size_nm=max(config.optical_pixel_size_nm / 4, 1.0),
    )
    figure_path = figure_dir / "frc_resolution.png"
    artifacts = _save_frc_curve(frc_result, figure_path, config)
    summary_path = statistics_dir / "frc_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "resolution_nm": frc_result.resolution_nm,
                "threshold": frc_result.threshold,
                "warning": frc_result.warning,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(summary_path)
    return artifacts


def _save_frc_curve(result: FRCResult, path: Path, config: PeakLocConfig) -> list[Path]:
    fig, axis = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    if result.spatial_frequency_per_nm.size:
        axis.plot(
            result.spatial_frequency_per_nm,
            result.frc,
            color=PLOT_COLORS["blue"],
        )
        axis.axhline(
            result.threshold,
            color=PLOT_COLORS["vermillion"],
            linestyle="--",
            label="1/7 criterion",
        )
    else:
        axis.text(
            0.5,
            0.5,
            result.warning or "No FRC data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        title="Fourier ring correlation",
        xlabel="Spatial frequency (nm⁻¹)",
        ylabel="FRC",
    )
    if result.spatial_frequency_per_nm.size:
        axis.legend(frameon=False, fontsize=6)
    style_publication_axis(axis)
    artifacts = save_publication_figure(
        fig, path, dpi=max(config.qc_static_dpi, 450), save_vector=config.qc_save_vector
    )
    plt.close(fig)
    return artifacts
