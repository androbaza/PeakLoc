from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter
import tifffile

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RENDER_OVERSAMPLING = 5
GAUSSIAN_SIGMA_RENDER_PIXELS = 1.0
NORMALIZATION_PERCENTILE = 99.8
TIFF_BIT_DEPTH = 12


@dataclass(frozen=True)
class SmlmRenderResult:
    png_path: Path
    tiff_path: Path
    localization_count: int
    render_pixel_size_nm: float
    image_shape: tuple[int, int]


def save_smlm_visualization(
    localizations: np.ndarray,
    localizations_path: str | Path,
    output_dir: str | Path,
    optical_pixel_size_nm: float,
    timestamp: str,
) -> SmlmRenderResult | None:
    coordinates = extract_localization_coordinates(localizations)
    if coordinates.size == 0:
        return None

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    source_path = Path(localizations_path)
    render_pixel_size_nm = optical_pixel_size_nm / RENDER_OVERSAMPLING
    density = render_density_image(coordinates, RENDER_OVERSAMPLING)

    image_8bit = normalize_to_uint(density, bit_depth=8)
    image_12bit = normalize_to_uint(density, bit_depth=TIFF_BIT_DEPTH)

    base_name = f"{source_path.stem}_smlm_{timestamp}"
    png_path = output_folder / f"{base_name}.png"
    tiff_path = output_folder / f"{base_name}_12bit.tiff"

    save_png_preview(image_8bit, png_path, render_pixel_size_nm)
    tifffile.imwrite(
        tiff_path,
        image_12bit,
        photometric="minisblack",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": render_pixel_size_nm,
            "PhysicalSizeY": render_pixel_size_nm,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeYUnit": "nm",
            "SignificantBits": TIFF_BIT_DEPTH,
        },
    )

    return SmlmRenderResult(
        png_path=png_path,
        tiff_path=tiff_path,
        localization_count=coordinates.shape[0],
        render_pixel_size_nm=render_pixel_size_nm,
        image_shape=(int(image_12bit.shape[0]), int(image_12bit.shape[1])),
    )


def extract_localization_coordinates(localizations: np.ndarray) -> np.ndarray:
    if localizations.size == 0 or localizations.dtype.names is None:
        return np.empty((0, 2), dtype=np.float64)

    required_fields = {"x", "y"}
    if not required_fields.issubset(localizations.dtype.names):
        raise ValueError("Localization array must contain x and y fields")

    coordinates = [np.column_stack((localizations["x"], localizations["y"]))]
    double_fields = {"double", "x2", "y2"}
    if double_fields.issubset(localizations.dtype.names):
        double_mask = localizations["double"] == 1
        if np.any(double_mask):
            coordinates.append(
                np.column_stack(
                    (localizations["x2"][double_mask], localizations["y2"][double_mask])
                )
            )

    all_coordinates = np.concatenate(coordinates, axis=0)
    finite_mask = np.isfinite(all_coordinates).all(axis=1)
    positive_mask = (all_coordinates[:, 0] > 0) & (all_coordinates[:, 1] > 0)
    return all_coordinates[finite_mask & positive_mask]


def coordinates_for_napari(
    localizations: np.ndarray, optical_pixel_size_nm: float
) -> np.ndarray:
    coordinates_xy = extract_localization_coordinates(localizations)
    if coordinates_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    coordinates_yx = coordinates_xy[:, [1, 0]]
    return coordinates_yx * optical_pixel_size_nm


def render_density_image(coordinates_xy: np.ndarray, oversampling: int) -> np.ndarray:
    if coordinates_xy.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    scaled = coordinates_xy * oversampling
    width = max(1, int(np.ceil((coordinates_xy[:, 0].max() + 1) * oversampling)))
    height = max(1, int(np.ceil((coordinates_xy[:, 1].max() + 1) * oversampling)))
    image = np.zeros((height, width), dtype=np.float32)

    x_indices = np.clip(np.floor(scaled[:, 0]).astype(np.int64), 0, width - 1)
    y_indices = np.clip(np.floor(scaled[:, 1]).astype(np.int64), 0, height - 1)
    np.add.at(image, (y_indices, x_indices), 1.0)
    return gaussian_filter(image, sigma=GAUSSIAN_SIGMA_RENDER_PIXELS)


def normalize_to_uint(image: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth not in {8, 12}:
        raise ValueError("Only 8-bit and 12-bit normalization are supported")

    max_value = (1 << bit_depth) - 1
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    nonzero_values = image[image > 0]
    if nonzero_values.size == 0:
        return np.zeros_like(image, dtype=dtype)

    upper = np.percentile(nonzero_values, NORMALIZATION_PERCENTILE)
    if upper <= 0:
        return np.zeros_like(image, dtype=dtype)

    normalized = np.clip(image / upper, 0, 1)
    return np.rint(normalized * max_value).astype(dtype)


def save_png_preview(
    image_8bit: np.ndarray, output_path: Path, render_pixel_size_nm: float
) -> None:
    height, width = image_8bit.shape
    figure_width = 8
    figure_height = max(1, figure_width * height / width)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=300)
    ax.imshow(image_8bit, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    scalebar = ScaleBar(
        render_pixel_size_nm,
        units="nm",
        length_fraction=0.2,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.0,
    )
    ax.add_artist(scalebar)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
