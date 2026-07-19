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
PNG_DISPLAY_GAMMA = 2.0
PNG_DPI = 450


@dataclass(frozen=True)
class SmlmRenderResult:
    png_path: Path
    tiff_path: Path
    localization_count: int
    render_pixel_size_nm: float
    image_shape: tuple[int, int]
    crop_bounds_px: tuple[float, float, float, float] | None


def save_smlm_visualization(
    localizations: np.ndarray,
    localizations_path: str | Path,
    output_dir: str | Path,
    optical_pixel_size_nm: float,
    timestamp: str,
    sensor_shape: tuple[int, int] | None = None,
    output_stem: str | None = None,
    crop_to_data: bool = False,
) -> SmlmRenderResult | None:
    coordinates = extract_localization_coordinates(localizations)
    if coordinates.size == 0:
        return None

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    source_path = Path(localizations_path)
    render_pixel_size_nm = optical_pixel_size_nm / RENDER_OVERSAMPLING
    crop_bounds_px: tuple[float, float, float, float] | None = None
    coordinates_to_render = coordinates
    render_shape = sensor_shape
    if crop_to_data:
        crop_bounds_px = _data_crop_bounds(coordinates, sensor_shape)
        x_start, y_start, x_stop, y_stop = crop_bounds_px
        coordinates_to_render = coordinates - np.asarray([x_start, y_start])
        render_shape = (
            max(1, int(np.ceil(y_stop - y_start))),
            max(1, int(np.ceil(x_stop - x_start))),
        )
    density = render_density_image(
        coordinates_to_render,
        RENDER_OVERSAMPLING,
        sensor_shape=render_shape,
    )

    image_8bit = normalize_to_uint(density, bit_depth=8)
    image_12bit = normalize_to_uint(density, bit_depth=TIFF_BIT_DEPTH)

    base_name = output_stem or f"{source_path.stem}_smlm_{timestamp}"
    png_path = output_folder / f"{base_name}.png"
    tiff_path = output_folder / f"{base_name}_12bit.tiff"

    tiff_metadata: dict[str, object] = {
        "axes": "YX",
        "PhysicalSizeX": render_pixel_size_nm,
        "PhysicalSizeY": render_pixel_size_nm,
        "PhysicalSizeXUnit": "nm",
        "PhysicalSizeYUnit": "nm",
        "SignificantBits": TIFF_BIT_DEPTH,
    }
    if crop_bounds_px is not None:
        tiff_metadata["PeakLocCropBoundsSensorPixels"] = list(crop_bounds_px)

    save_png_preview(image_8bit, png_path, render_pixel_size_nm)
    tifffile.imwrite(
        tiff_path,
        image_12bit,
        photometric="minisblack",
        metadata=tiff_metadata,
    )

    return SmlmRenderResult(
        png_path=png_path,
        tiff_path=tiff_path,
        localization_count=coordinates.shape[0],
        render_pixel_size_nm=render_pixel_size_nm,
        image_shape=(int(image_12bit.shape[0]), int(image_12bit.shape[1])),
        crop_bounds_px=crop_bounds_px,
    )


def _data_crop_bounds(
    coordinates: np.ndarray, sensor_shape: tuple[int, int] | None
) -> tuple[float, float, float, float]:
    """Return an inclusive data crop with a small physical-coordinate margin."""
    x_min, y_min = np.min(coordinates, axis=0)
    x_max, y_max = np.max(coordinates, axis=0)
    x_margin = max((x_max - x_min) * 0.08, 4.0)
    y_margin = max((y_max - y_min) * 0.08, 4.0)
    x_start = float(np.floor(x_min - x_margin))
    y_start = float(np.floor(y_min - y_margin))
    x_stop = float(np.ceil(x_max + x_margin + 1.0))
    y_stop = float(np.ceil(y_max + y_margin + 1.0))
    if sensor_shape is not None:
        height, width = sensor_shape
        x_start = max(0.0, x_start)
        y_start = max(0.0, y_start)
        x_stop = min(float(width), x_stop)
        y_stop = min(float(height), y_stop)
    return x_start, y_start, max(x_stop, x_start + 1.0), max(y_stop, y_start + 1.0)


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


def render_density_image(
    coordinates_xy: np.ndarray,
    oversampling: int,
    *,
    sensor_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if coordinates_xy.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    scaled = coordinates_xy * oversampling
    if sensor_shape is None:
        height = max(1, int(np.ceil((coordinates_xy[:, 1].max() + 1) * oversampling)))
        width = max(1, int(np.ceil((coordinates_xy[:, 0].max() + 1) * oversampling)))
    else:
        sensor_height, sensor_width = sensor_shape
        if sensor_height <= 0 or sensor_width <= 0:
            raise ValueError("sensor_shape dimensions must be positive")
        height = sensor_height * oversampling
        width = sensor_width * oversampling
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
    fig = plt.figure(figsize=(width / PNG_DPI, height / PNG_DPI), dpi=PNG_DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    display_image = np.power(image_8bit / 255.0, 1 / PNG_DISPLAY_GAMMA)
    ax.imshow(display_image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    scalebar = ScaleBar(
        render_pixel_size_nm,
        units="nm",
        length_fraction=0.1,
        location="lower right",
        frameon=False,
        color="white",
        box_alpha=0.0,
    )
    ax.add_artist(scalebar)
    fig.savefig(output_path, dpi=PNG_DPI, pad_inches=0)
    plt.close(fig)
