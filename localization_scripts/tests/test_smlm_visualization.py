import numpy as np
import tifffile

from localization_scripts.smlm_visualization import (
    coordinates_for_napari,
    extract_localization_coordinates,
    save_smlm_visualization,
)


LOCALIZATION_DTYPE = [
    ("x", np.float64),
    ("y", np.float64),
    ("double", np.uint8),
    ("x2", np.float64),
    ("y2", np.float64),
]


def test_extract_localization_coordinates_includes_double_fit_component():
    localizations = np.array(
        [
            (2.0, 3.0, 0, 0.0, 0.0),
            (4.0, 5.0, 1, 6.0, 7.0),
        ],
        dtype=LOCALIZATION_DTYPE,
    )

    coordinates = extract_localization_coordinates(localizations)

    np.testing.assert_allclose(coordinates, [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])


def test_coordinates_for_napari_returns_yx_nanometers():
    localizations = np.array([(2.0, 3.0, 0, 0.0, 0.0)], dtype=LOCALIZATION_DTYPE)

    points = coordinates_for_napari(localizations, optical_pixel_size_nm=67.0)

    np.testing.assert_allclose(points, [[201.0, 134.0]])


def test_save_smlm_visualization_writes_8bit_png_and_12bit_tiff(tmp_path):
    localizations = np.array(
        [
            (2.0, 3.0, 0, 0.0, 0.0),
            (2.2, 3.1, 0, 0.0, 0.0),
            (4.0, 5.0, 1, 6.0, 7.0),
        ],
        dtype=LOCALIZATION_DTYPE,
    )
    localizations_path = tmp_path / "localizations.npy"
    np.save(localizations_path, localizations)

    result = save_smlm_visualization(
        localizations,
        localizations_path,
        tmp_path,
        optical_pixel_size_nm=67.0,
        timestamp="20260607_120000",
    )

    assert result is not None
    assert result.png_path.is_file()
    assert result.tiff_path.is_file()
    image = tifffile.imread(result.tiff_path)
    assert image.dtype == np.uint16
    assert image.max() <= 4095
    assert result.localization_count == 4
    assert result.render_pixel_size_nm == 13.4
