import numpy as np
import matplotlib.pyplot as plt
import os, tifffile
from scipy.interpolate import interp1d

input_data = "/home/smlm-workstation/event-smlm/generated_data/localizations/npy/tubulin300x400[7, 1800]_localizations_full.npy"
out_folder = (
    "/home/smlm-workstation/event-smlm/event-smlm-localization/generated_images/"
)


def neighbor_interpolation(
    localizations,
    pixel_dim=0.3,
    loc_source="gaussian",
    take_negatives=0,
    take_positives=1,
    interpolate=1,
    fixed_size=None,
    i_field="I",
):
    image_max_x, image_max_y = (
        int(np.ceil(np.max(localizations["x"]))) + 5,
        int(np.ceil(np.max(localizations["y"]))) + 5,
    )

    if fixed_size:
        image_max_x, image_max_y = fixed_size[0], fixed_size[1]

    histogram_interpolated = np.zeros(
        (int(image_max_x / pixel_dim) + 4, int(image_max_y / pixel_dim) + 4)
    )

    histogram_num_locs = np.zeros(
        (int(image_max_x / pixel_dim) + 4, int(image_max_y / pixel_dim) + 4)
    )
    if take_positives:
        for localization in localizations:
            # determine the main pixel coordinate
            if loc_source == "gaussian":
                coord_x, coord_y = round(localization["x"] / pixel_dim), round(
                    localization["y"] / pixel_dim
                )

                # determine the subpixel position
                sub_x, sub_y = (
                    localization["x"] % pixel_dim - pixel_dim / 2,
                    localization["y"] % pixel_dim - pixel_dim / 2,
                )

            elif loc_source == "phasor":
                coord_x, coord_y = round(localization["x_p"] / pixel_dim), round(
                    localization["y_p"] / pixel_dim
                )

                # determine the subpixel position
                sub_x, sub_y = (
                    localization["x_p"] % pixel_dim - pixel_dim / 2,
                    localization["y_p"] % pixel_dim - pixel_dim / 2,
                )

            if interpolate:
                # intensity of the pixel value based on subpixel position
                I_x, I_y = (-np.abs(sub_x) / pixel_dim + 1), (
                    -np.abs(sub_y) / pixel_dim + 1
                )

                # fill the histogram at calculated coords
                histogram_interpolated[coord_x, coord_y] += (
                    I_x * I_y * localization[i_field]
                )

                # vertical neighbors
                if sub_y > 0:
                    histogram_interpolated[coord_x, coord_y + 1] += (
                        I_x * (1 - I_y) * localization[i_field]
                    )
                else:
                    histogram_interpolated[coord_x, coord_y - 1] += (
                        I_x * (1 - I_y) * localization[i_field]
                    )

                # horizontal and diagonal neighbors
                if sub_x > 0:
                    histogram_interpolated[coord_x + 1, coord_y] += (
                        (1 - I_x) * I_y * localization[i_field]
                    )
                    if sub_y > 0:
                        histogram_interpolated[coord_x + 1, coord_y + 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                    else:
                        histogram_interpolated[coord_x + 1, coord_y - 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                else:
                    histogram_interpolated[coord_x - 1, coord_y] += (
                        (1 - I_x) * I_y * localization[i_field]
                    )
                    if sub_y > 0:
                        histogram_interpolated[coord_x - 1, coord_y + 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                    else:
                        histogram_interpolated[coord_x - 1, coord_y - 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
            else:
                histogram_interpolated[coord_x, coord_y] += localization[i_field]
                histogram_num_locs[coord_x, coord_y] += 1

    if take_negatives:
        for localization in localizations:
            # determine the main pixel coordinate
            if loc_source == "gaussian" and localization["E_total_n"] > 40:
                coord_x, coord_y = round(localization["x_n"] / pixel_dim), round(
                    localization["y_n"] / pixel_dim
                )

                # determine the subpixel position
                sub_x, sub_y = (
                    localization["x_n"] % pixel_dim - pixel_dim / 2,
                    localization["y_n"] % pixel_dim - pixel_dim / 2,
                )

            elif loc_source == "phasor":
                coord_x, coord_y = round(localization["x_p"] / pixel_dim), round(
                    localization["y_p"] / pixel_dim
                )

                # determine the subpixel position
                sub_x, sub_y = (
                    localization["x_p"] % pixel_dim - pixel_dim / 2,
                    localization["y_p"] % pixel_dim - pixel_dim / 2,
                )

            if interpolate:
                # intensity of the pixel value based on subpixel position
                I_x, I_y = (-np.abs(sub_x) / pixel_dim + 1), (
                    -np.abs(sub_y) / pixel_dim + 1
                )

                # fill the histogram at calculated coords
                histogram_interpolated[coord_x, coord_y] += (
                    I_x * I_y * localization[i_field]
                )

                # vertical neighbors
                if sub_y > 0:
                    histogram_interpolated[coord_x, coord_y + 1] += (
                        I_x * (1 - I_y) * localization[i_field]
                    )
                else:
                    histogram_interpolated[coord_x, coord_y - 1] += (
                        I_x * (1 - I_y) * localization[i_field]
                    )

                # horizontal and diagonal neighbors
                if sub_x > 0:
                    histogram_interpolated[coord_x + 1, coord_y] += (
                        (1 - I_x) * I_y * localization[i_field]
                    )
                    if sub_y > 0:
                        histogram_interpolated[coord_x + 1, coord_y + 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                    else:
                        histogram_interpolated[coord_x + 1, coord_y - 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                else:
                    histogram_interpolated[coord_x - 1, coord_y] += (
                        (1 - I_x) * I_y * localization[i_field]
                    )
                    if sub_y > 0:
                        histogram_interpolated[coord_x - 1, coord_y + 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
                    else:
                        histogram_interpolated[coord_x - 1, coord_y - 1] += (
                            (1 - I_x) * (1 - I_y) * localization[i_field]
                        )
            else:
                histogram_interpolated[coord_x, coord_y] += localization[i_field]
                histogram_num_locs[coord_x, coord_y] += 1
    return histogram_interpolated, histogram_num_locs


def histogram_binning(
    localizations,
    image_max_x,
    image_max_y,
    pixel_dim=0.3,
    take_negatives=0,
    take_positives=1,
):
    """useful for drift correction and FRC resolution calculation"""
    # image_max_x, image_max_y = int(np.ceil(np.max(localizations['x']))) + 1, int(np.ceil(np.max(localizations['y']))) + 1
    image, _, _ = np.histogram2d(
        localizations["x"],
        localizations["y"],
        bins=[int(image_max_x / pixel_dim), int(image_max_y / pixel_dim)],
    )
    if take_negatives:
        image2, _, _ = np.histogram2d(
            localizations["x_n"],
            localizations["y_n"],
            bins=[int(image_max_x / pixel_dim), int(image_max_y / pixel_dim)],
        )
        return image + image2
    return image


"""implemented drift correction does not function properly with given data. probably the present drift is non-linear (vibrations?)."""


def drift_correction(
    input_loc, num_images, out_loc=None, mode="same", pixel_dim=0.3, pixel_dim_out=0.3
):
    from skimage.registration import phase_cross_correlation

    slice_start = 0
    shifts = []
    localizations = np.copy(input_loc)
    image_max_x, image_max_y = (
        int(np.ceil(np.max(localizations["x"]))) + 1,
        int(np.ceil(np.max(localizations["y"]))) + 1,
    )
    for slice_end in np.arange(
        len(localizations) // num_images,
        len(localizations),
        len(localizations) // num_images,
    ):
        image = histogram_binning(
            localizations[slice_start:slice_end], image_max_x, image_max_y, pixel_dim
        )
        # image = neighbor_interpolation(localizations[slice_start : slice_end], pixel_dim)
        if slice_start == 0:
            start_image = image
        if slice_start != 0:
            detected_shift = phase_cross_correlation(
                image, start_image, upsample_factor=150
            )
            shifts.append((*detected_shift[0], slice_end))
        slice_start = slice_end
    shifts = np.array(shifts)
    interpx = interp1d(
        shifts[:, -1], shifts[:, 0], kind="quadratic", fill_value="extrapolate"
    )
    interpy = interp1d(
        shifts[:, -1], shifts[:, 1], kind="quadratic", fill_value="extrapolate"
    )
    if mode == "crop":
        localizations = np.copy(out_loc)
        pixel_dim = pixel_dim_out
    data_interp = np.arange(0, len(localizations), 1)
    interpx = interpx(data_interp)
    interpy = interpy(data_interp)
    for i in np.arange(len(data_interp)):
        localizations[i]["x"] = np.subtract(
            localizations[i]["x"], interpx[i] * pixel_dim
        )
        localizations[i]["y"] = np.subtract(
            localizations[i]["y"], interpy[i] * pixel_dim
        )
        localizations[i]["x_n"] = np.subtract(
            localizations[i]["x_n"], interpx[i] * pixel_dim
        )
        localizations[i]["y_n"] = np.subtract(
            localizations[i]["y_n"], interpy[i] * pixel_dim
        )
    return localizations


"""randomly shuffle localizations before binning for FRC resolution mesurement"""


def FRC_split(localizations, pixel_dim=0.3):
    data_shuffle = np.random.shuffle(localizations)
    s1, s2 = (
        data_shuffle[len(localizations) // 2 :],
        data_shuffle[: len(localizations) // 2],
    )
    frc1, frc2 = histogram_binning(s1, pixel_dim), histogram_binning(s2, pixel_dim)
    tifffile.imwrite(
        "/home/smlm-workstation/event-smlm/event-smlm-localization/figures/FRC1.tif",
        frc1.T.astype("float32"),
    )
    tifffile.imwrite(
        "/home/smlm-workstation/event-smlm/event-smlm-localization/figures/FRC2.tif",
        frc2.T.astype("float32"),
    )


localizations = np.load(input_data)
pixel_dim = 0.3

"""basic filtering"""
localizations = localizations[~np.isnan(localizations["y"])]
localizations = localizations[~np.isnan(localizations["x"])]

"""filter out PSFs that were not complete"""
localizations = localizations[localizations["E_total"] > 70]
# localizations = localizations[:500000]
# localizations = localizations[-100000:]

# localizations = drift_correction(localizations, 10, 0.3)

"""if all the localizations without drift correction are taken, the image is significantly blurred in y direction"""
# histogram_interpolated = neighbor_interpolation(localizations, pixel_dim=0.3)

"""a slice of 5e5 localizations does not experience drift"""
localizations = localizations[:500000]
histogram_interpolated = neighbor_interpolation(localizations, pixel_dim=0.3)

tifffile.imwrite(
    out_folder + "superres_image.tif", histogram_interpolated.T.astype("float32")
)
