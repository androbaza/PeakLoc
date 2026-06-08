from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from localization_scripts.debug_visualization import DEBUG_COLORS
from localization_scripts.localization_fitting import localization_uncertainty_px
from localization_scripts.pipeline_config import PeakLocConfig

MONTAGE_FILENAMES = (
    "uncertainty_lowest_36_combined.png",
    "uncertainty_highest_36_combined.png",
    "uncertainty_lowest_36_positive.png",
    "uncertainty_highest_36_positive.png",
    "uncertainty_lowest_36_negative.png",
    "uncertainty_highest_36_negative.png",
    "uncertainty_quantile_samples.png",
)
FAILED_MONTAGE_FILENAME = "uncertainty_failed_fits.png"


def select_uncertainty_extremes(
    localizations: np.ndarray,
    *,
    n: int = 36,
    include_rejected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of n lowest and n highest finite uncertainty fits."""
    if localizations.size == 0 or not _has_uncertainty_fields(localizations):
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    uncertainty_px = localization_uncertainty_px(localizations)
    finite = np.isfinite(uncertainty_px)
    if not include_rejected and "fit_success" in (localizations.dtype.names or ()):
        finite &= localizations["fit_success"]

    finite_indices = np.flatnonzero(finite)
    if finite_indices.size == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty

    order = finite_indices[np.argsort(uncertainty_px[finite_indices], kind="stable")]
    count = min(n, order.size)
    return order[:count], order[-count:][::-1]


def save_uncertainty_montages(
    attempted_localizations: np.ndarray,
    accepted_localizations: np.ndarray,
    qc_table: np.ndarray,
    output_dir: Path,
    *,
    config: PeakLocConfig,
    n: int = 36,
    dpi: int = 450,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    save_vector = bool(getattr(config, "qc_save_vector", False))
    lowest, highest = select_uncertainty_extremes(
        attempted_localizations, n=n, include_rejected=True
    )

    for label, indices in (("lowest", lowest), ("highest", highest)):
        for polarity in ("combined", "positive", "negative"):
            path = output_dir / f"uncertainty_{label}_{n}_{polarity}.png"
            paths.extend(
                _save_index_montage(
                    attempted_localizations,
                    indices,
                    accepted_localizations,
                    qc_table,
                    path,
                    config=config,
                    polarity=polarity,
                    title=f"{label.title()} uncertainty fits ({polarity})",
                    dpi=dpi,
                    show_covariance=label == "highest",
                    save_vector=save_vector,
                )
            )

    quantile_path = output_dir / "uncertainty_quantile_samples.png"
    paths.extend(
        _save_index_montage(
            attempted_localizations,
            _select_quantile_sample_indices(
                attempted_localizations, qc_table, per_bin=6
            ),
            accepted_localizations,
            qc_table,
            quantile_path,
            config=config,
            polarity="combined",
            title="Uncertainty quantile samples",
            dpi=dpi,
            show_covariance=True,
            save_vector=save_vector,
        )
    )

    failed_indices = _failed_fit_indices(attempted_localizations, qc_table, limit=n)
    if failed_indices.size:
        failed_path = output_dir / FAILED_MONTAGE_FILENAME
        paths.extend(
            _save_index_montage(
                attempted_localizations,
                failed_indices,
                accepted_localizations,
                qc_table,
                failed_path,
                config=config,
                polarity="combined",
                title="Failed or non-finite uncertainty fits",
                dpi=dpi,
                show_covariance=True,
                save_vector=save_vector,
            )
        )

    return paths


def _save_index_montage(
    attempted_localizations: np.ndarray,
    indices: np.ndarray,
    accepted_localizations: np.ndarray,
    qc_table: np.ndarray,
    output_path: Path,
    *,
    config: PeakLocConfig,
    polarity: str,
    title: str,
    dpi: int,
    show_covariance: bool,
    save_vector: bool,
) -> list[Path]:
    max_tiles = max(int(indices.size), 1)
    columns = min(6, max_tiles)
    rows = int(np.ceil(max_tiles / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.3, rows * 2.6),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=11)
    qc_by_id = _qc_rows_by_id(qc_table)
    accepted_ids = _id_set(accepted_localizations)

    for axis in axes.ravel():
        axis.axis("off")

    for axis, localization_index in zip(axes.ravel(), indices):
        row = attempted_localizations[int(localization_index)]
        _draw_fit_tile(
            axis,
            row,
            qc_by_id.get(int(row["id"])) if "id" in row.dtype.names else None,
            accepted_ids,
            config=config,
            polarity=polarity,
            show_covariance=show_covariance,
        )

    if indices.size == 0:
        axes.ravel()[0].text(
            0.5,
            0.5,
            "No finite fits",
            ha="center",
            va="center",
            transform=axes.ravel()[0].transAxes,
        )
    paths = [output_path]
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if save_vector:
        for suffix in (".svg", ".pdf"):
            vector_path = output_path.with_suffix(suffix)
            fig.savefig(vector_path, bbox_inches="tight")
            paths.append(vector_path)
    plt.close(fig)
    return paths


def _draw_fit_tile(
    axis,
    row: np.void,
    qc_row: np.void | None,
    accepted_ids: set[int],
    *,
    config: PeakLocConfig,
    polarity: str,
    show_covariance: bool,
) -> None:
    image = _roi_image(row, polarity)
    axis.imshow(image, cmap="gray", interpolation="none", origin="upper")
    sub_x = _float_field(row, "sub_x", (image.shape[1] - 1) / 2)
    sub_y = _float_field(row, "sub_y", (image.shape[0] - 1) / 2)
    axis.scatter(sub_x, sub_y, c=DEBUG_COLORS["residual"], s=24, marker="x", zorder=5)
    _draw_psf_contour(axis, image.shape, sub_x, sub_y, config)

    bad_covariance = False
    if show_covariance:
        bad_covariance = not _draw_uncertainty_ellipse(axis, row, sub_x, sub_y)

    reason = _qc_reason(qc_row)
    accepted = _qc_accepted(qc_row, row, accepted_ids)
    uncertainty_px = _localization_uncertainty_scalar(row)
    uncertainty_nm = uncertainty_px * config.optical_pixel_size_nm
    title_color = (
        DEBUG_COLORS["matched_localization"]
        if accepted
        else DEBUG_COLORS["unmatched_localization"]
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#D55E00" if bad_covariance else title_color)
    if bad_covariance:
        axis.text(
            0.02,
            0.05,
            "bad covariance",
            color="#D55E00",
            fontsize=6,
            transform=axis.transAxes,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    axis.set_title(
        "\n".join(
            [
                f"id {_int_field(row, 'id', -1)} | {reason}",
                f"unc {uncertainty_px:.2g}px / {uncertainty_nm:.2g}nm",
                f"E {_int_field(row, 'E_total', 0)}/{_int_field(row, 'E_total_n', 0)} "
                f"NLL {_float_field(row, 'nll_per_event', np.nan):.2g}",
                f"cond {_float_field(row, 'fit_cond', np.nan):.2g} "
                f"({sub_x:.2f}, {sub_y:.2f})",
            ]
        ),
        fontsize=6,
        color=title_color,
    )


def _roi_image(row: np.void, polarity: str) -> np.ndarray:
    names = row.dtype.names or ()
    positive = np.asarray(row["roi"], dtype=np.float64) if "roi" in names else None
    negative = np.asarray(row["roi_n"], dtype=np.float64) if "roi_n" in names else None
    if polarity == "positive" and positive is not None:
        return positive
    if polarity == "negative" and negative is not None:
        return negative
    if positive is not None and negative is not None:
        return positive + negative
    if positive is not None:
        return positive
    if negative is not None:
        return negative
    return np.zeros((3, 3), dtype=np.float64)


def _draw_psf_contour(
    axis,
    image_shape: tuple[int, int],
    sub_x: float,
    sub_y: float,
    config: PeakLocConfig,
) -> None:
    sigma = config.sigma_psf_px or config.dataset_fwhm / 2.354820045
    yy, xx = np.indices(image_shape)
    model = np.exp(-0.5 * (((xx - sub_x) / sigma) ** 2 + ((yy - sub_y) / sigma) ** 2))
    axis.contour(
        xx,
        yy,
        model,
        levels=[np.exp(-0.5)],
        colors=[DEBUG_COLORS["roi"]],
        linewidths=0.8,
    )


def _draw_uncertainty_ellipse(axis, row: np.void, sub_x: float, sub_y: float) -> bool:
    sigma_x = _float_field(row, "sigma_x", np.nan)
    sigma_y = _float_field(row, "sigma_y", np.nan)
    cov_xy = _float_field(row, "cov_xy", np.nan)
    covariance = np.asarray(
        [[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]], dtype=np.float64
    )
    if not np.all(np.isfinite(covariance)):
        return False
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues < -1e-12):
        return False
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        (sub_x, sub_y),
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        edgecolor=DEBUG_COLORS["negative_events"],
        linewidth=0.8,
    )
    axis.add_patch(ellipse)
    return True


def _select_quantile_sample_indices(
    localizations: np.ndarray, qc_table: np.ndarray, *, per_bin: int
) -> np.ndarray:
    if localizations.size == 0 or not _has_uncertainty_fields(localizations):
        return np.asarray([], dtype=np.int64)
    uncertainty_px = localization_uncertainty_px(localizations)
    finite_indices = np.flatnonzero(np.isfinite(uncertainty_px))
    selected: list[int] = []
    if finite_indices.size:
        quantiles = np.quantile(
            uncertainty_px[finite_indices],
            [0, 0.1, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0],
        )
        ranges = [
            (quantiles[0], quantiles[1]),
            (quantiles[2], quantiles[3]),
            (quantiles[4], quantiles[5]),
            (quantiles[6], quantiles[7]),
            (quantiles[8], quantiles[9]),
        ]
        for low, high in ranges:
            in_bin = finite_indices[
                (uncertainty_px[finite_indices] >= low)
                & (uncertainty_px[finite_indices] <= high)
            ]
            ordered = in_bin[np.argsort(uncertainty_px[in_bin], kind="stable")]
            selected.extend(ordered[:per_bin].tolist())

    selected.extend(
        _failed_fit_indices(localizations, qc_table, limit=per_bin).tolist()
    )
    return np.asarray(selected, dtype=np.int64)


def _failed_fit_indices(
    localizations: np.ndarray, qc_table: np.ndarray, *, limit: int
) -> np.ndarray:
    if localizations.size == 0 or not _has_uncertainty_fields(localizations):
        return np.asarray([], dtype=np.int64)
    uncertainty_px = localization_uncertainty_px(localizations)
    failed = ~np.isfinite(uncertainty_px)
    if qc_table.size and "primary_rejection_reason" in (qc_table.dtype.names or ()):
        reasons_by_id = {
            int(row["id"]): str(row["primary_rejection_reason"]) for row in qc_table
        }
        ids = _row_ids(localizations)
        failed |= np.asarray(
            [reasons_by_id.get(int(row_id)) == "fit_failed" for row_id in ids],
            dtype=bool,
        )
    return np.flatnonzero(failed)[:limit]


def _qc_rows_by_id(qc_table: np.ndarray) -> dict[int, np.void]:
    if qc_table.size == 0 or "id" not in (qc_table.dtype.names or ()):
        return {}
    return {int(row["id"]): row for row in qc_table}


def _id_set(localizations: np.ndarray) -> set[int]:
    if localizations.size == 0 or "id" not in (localizations.dtype.names or ()):
        return set()
    return {int(row_id) for row_id in localizations["id"]}


def _row_ids(localizations: np.ndarray) -> np.ndarray:
    if "id" in (localizations.dtype.names or ()):
        return localizations["id"]
    return np.arange(localizations.size)


def _has_uncertainty_fields(localizations: np.ndarray) -> bool:
    return {"sigma_x", "sigma_y", "cov_xy"}.issubset(localizations.dtype.names or ())


def _qc_reason(qc_row: np.void | None) -> str:
    if qc_row is None or "primary_rejection_reason" not in (qc_row.dtype.names or ()):
        return "accepted"
    return str(qc_row["primary_rejection_reason"])


def _qc_accepted(qc_row: np.void | None, row: np.void, accepted_ids: set[int]) -> bool:
    if qc_row is not None and "accepted" in (qc_row.dtype.names or ()):
        return bool(qc_row["accepted"])
    if "id" in (row.dtype.names or ()):
        return int(row["id"]) in accepted_ids
    return True


def _localization_uncertainty_scalar(row: np.void) -> float:
    names = row.dtype.names or ()
    if not {"sigma_x", "sigma_y", "cov_xy"}.issubset(names):
        return float("nan")
    single = np.empty(1, dtype=row.dtype)
    single[0] = row
    return float(localization_uncertainty_px(single)[0])


def _float_field(row: np.void, field_name: str, default: float) -> float:
    if field_name not in (row.dtype.names or ()):
        return default
    return float(row[field_name])


def _int_field(row: np.void, field_name: str, default: int) -> int:
    if field_name not in (row.dtype.names or ()):
        return default
    return int(row[field_name])
