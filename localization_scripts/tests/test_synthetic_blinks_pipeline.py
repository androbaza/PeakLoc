from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import process_recording


EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),      # PeakLoc convention: 1 = positive, 0 = negative
        ("t", np.uint64),    # microseconds
    ]
)


@dataclass(frozen=True)
class BlinkTruth:
    x_px: float
    y_px: float
    peak_us: int
    n_pos: int = 8_000
    n_neg: int = 8_000


SENSOR_HEIGHT = 96
SENSOR_WIDTH = 96
SENSOR_SHAPE = (SENSOR_HEIGHT, SENSOR_WIDTH)

SIGMA_PSF_PX = 1.25
DATASET_FWHM_PX = 2.355 * SIGMA_PSF_PX
ROI_RADIUS = 5

TRUTH = (
    BlinkTruth(x_px=30.35, y_px=31.70, peak_us=200_000),
    BlinkTruth(x_px=61.25, y_px=60.30, peak_us=600_000),
)


def test_synthetic_blinks_are_detected_and_localized_end_to_end(
    tmp_path: Path,
) -> None:
    """
    End-to-end synthetic event-camera SMLM test.

    The synthetic recording contains two temporally separated blinks with known
    subpixel centers. The test exercises the real process_recording() path:
    event loading, polarity-map conversion, convolved peak detection, local
    maximum filtering, ROI generation, and joint Poisson fitting.
    """
    events = synthetic_event_recording(
        blinks=TRUTH,
        sensor_shape=SENSOR_SHAPE,
        sigma_px=SIGMA_PSF_PX,
        support_radius_px=7,
    )

    input_path = tmp_path / "synthetic_blinks.npy"
    np.save(input_path, events)

    config = _make_config(tmp_path)

    result = process_recording(
        input_path,
        config,
        run_timestamp="pytest_synthetic",
    )

    assert result.event_count == events.size
    assert len(result.slice_results) == 2

    total_unique_peaks = sum(s.unique_peak_count for s in result.slice_results)
    total_rois = sum(s.roi_count for s in result.slice_results)
    total_localizations = sum(s.localization_count for s in result.slice_results)

    assert total_unique_peaks >= len(TRUTH)
    assert total_rois >= len(TRUTH)
    assert total_localizations >= len(TRUTH)

    loc_path = (
        input_path.with_suffix("")
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    assert loc_path.is_file(), f"Missing final localization output: {loc_path}"

    locs = np.load(loc_path)
    assert locs.size >= len(TRUTH)
    assert locs.dtype.names is not None

    assert len(np.unique(locs["id"])) == locs.size, (
        "Localization IDs are not unique after concatenating time slices. "
        "The per-slice offset should use max_id + 1."
    )

    if "fit_success" in locs.dtype.names:
        locs = locs[locs["fit_success"]]

    assert locs.size >= len(TRUTH), "Too few successful fitted localizations."

    for truth in TRUTH:
        matched = _localizations_near_time(
            locs,
            truth.peak_us,
            max_abs_time_error_us=80_000,
        )
        assert matched.size > 0, (
            f"No localization found near synthetic blink peak "
            f"t={truth.peak_us} us."
        )

        best, best_dist = _best_spatial_match(matched, truth)

        assert best_dist <= 1.0, (
            "Recovered localization is not close to the known global emitter "
            f"coordinate for blink at {truth.peak_us} us. "
            f"Expected approximately x={truth.x_px:.2f}, y={truth.y_px:.2f}; "
            f"closest was x={float(best['x']):.2f}, y={float(best['y']):.2f}; "
            f"distance={best_dist:.2f} px."
        )

        assert int(best["E_total"]) > 1_000
        assert int(best["E_total_n"]) > 1_000

        assert np.count_nonzero(best["roi"]) >= 12, (
            "Positive ROI contains too few populated pixels. This usually "
            "means ROI generation is not including all event pixels inside "
            "the spatial ROI."
        )
        assert np.count_nonzero(best["roi_n"]) >= 12, (
            "Negative ROI contains too few populated pixels. This usually "
            "means ROI generation is not including all event pixels inside "
            "the spatial ROI."
        )


def synthetic_event_recording(
    *,
    blinks: tuple[BlinkTruth, ...],
    sensor_shape: tuple[int, int],
    sigma_px: float,
    support_radius_px: int,
) -> np.ndarray:
    records: list[tuple[int, int, int, int]] = []

    for blink in blinks:
        records.extend(
            _events_for_one_blink(
                blink=blink,
                sensor_shape=sensor_shape,
                sigma_px=sigma_px,
                support_radius_px=support_radius_px,
            )
        )

    events = np.asarray(records, dtype=EVENT_DTYPE)
    events.sort(order="t")

    _assert_no_per_pixel_timestamp_collisions(events)
    return events


def _events_for_one_blink(
    *,
    blink: BlinkTruth,
    sensor_shape: tuple[int, int],
    sigma_px: float,
    support_radius_px: int,
) -> list[tuple[int, int, int, int]]:
    height, width = sensor_shape

    x0 = blink.x_px
    y0 = blink.y_px

    x_c = int(round(x0))
    y_c = int(round(y0))

    ys = np.arange(
        max(0, y_c - support_radius_px),
        min(height, y_c + support_radius_px + 1),
        dtype=np.int32,
    )
    xs = np.arange(
        max(0, x_c - support_radius_px),
        min(width, x_c + support_radius_px + 1),
        dtype=np.int32,
    )

    weights = _pixel_integrated_gaussian_weights(xs, ys, x0, y0, sigma_px)
    probabilities = weights.ravel() / np.sum(weights)

    rng = np.random.default_rng(blink.peak_us)
    pos_counts = rng.multinomial(blink.n_pos, probabilities)
    neg_counts = rng.multinomial(blink.n_neg, probabilities)

    pixel_coords = [(int(y), int(x)) for y in ys for x in xs]

    pos_times = _unique_burst_times(
        start_us=blink.peak_us - 80_000,
        stop_us=blink.peak_us - 5_000,
        count=blink.n_pos,
    )
    neg_times = _unique_burst_times(
        start_us=blink.peak_us + 5_000,
        stop_us=blink.peak_us + 80_000,
        count=blink.n_neg,
    )

    records: list[tuple[int, int, int, int]] = []
    pos_cursor = 0
    neg_cursor = 0

    for (y, x), n_pos, n_neg in zip(
        pixel_coords,
        pos_counts,
        neg_counts,
        strict=True,
    ):
        n_pos_int = int(n_pos)
        n_neg_int = int(n_neg)

        for t in pos_times[pos_cursor : pos_cursor + n_pos_int]:
            records.append((x, y, 1, int(t)))
        pos_cursor += n_pos_int

        for t in neg_times[neg_cursor : neg_cursor + n_neg_int]:
            records.append((x, y, 0, int(t)))
        neg_cursor += n_neg_int

    assert pos_cursor == blink.n_pos
    assert neg_cursor == blink.n_neg

    return records


def _pixel_integrated_gaussian_weights(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    sigma_px: float,
) -> np.ndarray:
    sqrt2_sigma = np.sqrt(2.0) * sigma_px

    wx = 0.5 * (
        erf((xs + 0.5 - x0) / sqrt2_sigma)
        - erf((xs - 0.5 - x0) / sqrt2_sigma)
    )
    wy = 0.5 * (
        erf((ys + 0.5 - y0) / sqrt2_sigma)
        - erf((ys - 0.5 - y0) / sqrt2_sigma)
    )

    weights = np.outer(wy, wx)
    return np.clip(weights, 0.0, None)


def _unique_burst_times(
    *,
    start_us: int,
    stop_us: int,
    count: int,
) -> np.ndarray:
    assert count > 0
    assert stop_us > start_us

    duration = stop_us - start_us
    assert count < duration, (
        "Synthetic blink has more events than unique integer timestamps in "
        "the burst window. Increase the window or reduce event count."
    )

    step = max(duration // (count + 1), 1)
    return np.asarray(
        start_us + step * np.arange(1, count + 1, dtype=np.uint64),
        dtype=np.uint64,
    )


def _assert_no_per_pixel_timestamp_collisions(events: np.ndarray) -> None:
    seen: set[tuple[int, int, int]] = set()

    for event in events:
        key = (int(event["y"]), int(event["x"]), int(event["t"]))
        assert key not in seen, (
            "Synthetic event generator created duplicate timestamps for the "
            "same pixel. That would make per-pixel event counting ambiguous."
        )
        seen.add(key)


def _make_config(tmp_path: Path) -> PeakLocConfig:
    return PeakLocConfig(
        input_folder=str(tmp_path),
        slice_start=0,
        slice_duration=400_000,
        num_cores=1,
        prominence=80.0,
        dataset_fwhm=DATASET_FWHM_PX,
        peak_time_threshold=80_000.0,
        polarity_time_gate_us=120_000.0,
        peak_neighbors=4,
        roi_radius=ROI_RADIUS,
        convolution_roi_radius=1,
        peak_min_event_count=80,
        interpolation_coefficient=3,
        spline_smooth=0.7,
        plot_subplotsize=4,
        plot_result=False,
        optical_pixel_size=67.0,
        sensor_height=SENSOR_HEIGHT,
        sensor_width=SENSOR_WIDTH,
        max_raw_events=500_000,
        cleanup_temp_outputs=False,
        fit_model="poisson_joint",
        allow_uncalibrated=True,
        calibration_path=None,
        sigma_psf_px=SIGMA_PSF_PX,
        fit_sigma=False,
        psf_model="pixel_integrated_gaussian",
        background_mode="local_only",
        hot_pixel_policy="mask",
        min_events_pos=20,
        min_events_neg=20,
        min_valid_pixels=1,
        max_fit_cond=1e12,
    )


def _localizations_near_time(
    locs: np.ndarray,
    peak_us: int,
    *,
    max_abs_time_error_us: int,
) -> np.ndarray:
    delta = np.abs(locs["t_peak"].astype(float) - float(peak_us))
    return locs[delta <= max_abs_time_error_us]


def _best_spatial_match(
    locs: np.ndarray,
    truth: BlinkTruth,
):
    distances = np.sqrt(
        (locs["x"].astype(float) - truth.x_px) ** 2
        + (locs["y"].astype(float) - truth.y_px) ** 2
    )
    idx = int(np.argmin(distances))
    return locs[idx], float(distances[idx])