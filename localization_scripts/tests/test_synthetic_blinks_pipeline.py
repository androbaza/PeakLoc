from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import numpy as np
from scipy.special import erf
import pytest
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import process_recording
from dataclasses import replace

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
    pos_start_offset_us: int = -80_000
    pos_stop_offset_us: int = -5_000
    neg_start_offset_us: int = 5_000
    neg_stop_offset_us: int = 80_000


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
    locs = _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="synthetic_blinks",
        blinks=TRUTH,
        max_spatial_error_px=1.0,
        min_events_per_polarity=1_000,
    )

    assert len(np.unique(locs["id"])) == locs.size, (
        "Localization IDs are not unique after concatenating time slices."
    )

def test_synthetic_blinks_with_different_event_counts_are_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=24.40, y_px=24.10, peak_us=200_000, n_pos=6_000, n_neg=6_000),
        BlinkTruth(x_px=48.75, y_px=45.25, peak_us=600_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="different_blink_sizes",
        blinks=blinks,
        max_spatial_error_px=1.0,
        min_events_per_polarity=1_000,
        config_overrides={
            "slice_duration": 400_000,
            "prominence": 60.0,
            "peak_min_event_count": 40,
        },
    )

def test_late_high_count_synthetic_blink_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=72.20, y_px=70.60, peak_us=1_000_000, n_pos=12_000, n_neg=12_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="late_high_count_blink",
        blinks=blinks,
        max_spatial_error_px=1.0,
        min_events_per_polarity=1_000,
        config_overrides={
            "slice_duration": 400_000,
            "prominence": 60.0,
            "peak_min_event_count": 40,
        },
    )

def test_temporally_separated_spatially_overlapping_blinks_are_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=45.20, y_px=48.10, peak_us=220_000, n_pos=8_000, n_neg=8_000),
        BlinkTruth(x_px=47.00, y_px=48.60, peak_us=520_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="temporally_separated_spatial_overlap",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=1_000,
    )

@pytest.mark.xfail(
    reason=(
        "Current poisson_joint model fits one emitter per ROI. "
        "Simultaneous spatially overlapping emitters require a multi-emitter model "
        "or explicit split/deblend logic."
    ),
    strict=False,
)
def test_simultaneous_spatially_overlapping_blinks_are_not_yet_resolved(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=45.20, y_px=48.10, peak_us=300_000, n_pos=8_000, n_neg=8_000),
        BlinkTruth(x_px=47.00, y_px=48.60, peak_us=300_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="simultaneous_spatial_overlap",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_successful_localizations=2,
        min_events_per_polarity=1_000,
    )

def test_synthetic_blink_near_upper_sensor_edge_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=81.20, y_px=80.70, peak_us=600_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="upper_sensor_edge",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=800,
        config_overrides={
            "prominence": 60.0,
            "peak_min_event_count": 40,
        },
    )

def test_synthetic_blink_near_lower_sensor_edge_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=14.40, y_px=14.60, peak_us=200_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="lower_sensor_edge",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=800,
        config_overrides={
            "prominence": 60.0,
            "peak_min_event_count": 40,
        },
    )

def test_synthetic_blink_close_to_lower_sensor_edge_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(x_px=8.40, y_px=8.60, peak_us=200_000, n_pos=8_000, n_neg=8_000),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="lower_sensor_edge",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=800,
    )

def test_synthetic_long_blink_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(
            x_px=66.50,
            y_px=63.80,
            peak_us=600_000,
            n_pos=10_000,
            n_neg=10_000,
            pos_start_offset_us=-120_000,
            pos_stop_offset_us=-8_000,
            neg_start_offset_us=8_000,
            neg_stop_offset_us=120_000,
        ),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="long_temporal_width",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=800,
        max_abs_time_error_us=140_000,
    )

def test_synthetic_short_blink_is_localized(
    tmp_path: Path,
) -> None:
    blinks = (
        BlinkTruth(
            x_px=28.30,
            y_px=30.20,
            peak_us=220_000,
            n_pos=6_000,
            n_neg=6_000,
            pos_start_offset_us=-40_000,
            pos_stop_offset_us=-3_000,
            neg_start_offset_us=3_000,
            neg_stop_offset_us=40_000,
        ),
    )

    _run_synthetic_scenario(
        tmp_path=tmp_path,
        name="short_temporal_width",
        blinks=blinks,
        max_spatial_error_px=1.25,
        min_events_per_polarity=800,
        max_abs_time_error_us=80_000,
        config_overrides={
            "prominence": 40.0,
            "peak_min_event_count": 30,
        },
    )

def _run_synthetic_scenario(
    *,
    tmp_path: Path,
    name: str,
    blinks: tuple[BlinkTruth, ...],
    max_spatial_error_px: float = 1.0,
    min_successful_localizations: int | None = None,
    max_abs_time_error_us: int = 100_000,
    min_events_per_polarity: int = 300,
    config_overrides: dict[str, object] | None = None,
) -> np.ndarray:
    events = synthetic_event_recording(
        blinks=blinks,
        sensor_shape=SENSOR_SHAPE,
        sigma_px=SIGMA_PSF_PX,
        support_radius_px=7,
    )

    input_path = tmp_path / f"{name}.npy"
    np.save(input_path, events)

    base_config = _make_config(tmp_path)
    config = (
        replace(base_config, **config_overrides)
        if config_overrides is not None
        else base_config
    )

    result = process_recording(
        input_path,
        config,
        run_timestamp=f"pytest_{name}",
    )

    expected_count = len(blinks) if min_successful_localizations is None else min_successful_localizations

    assert result.event_count == events.size
    assert len(result.slice_results) >= 1

    total_rois = sum(s.roi_count for s in result.slice_results)
    total_localizations = sum(s.localization_count for s in result.slice_results)

    assert total_rois >= expected_count
    assert total_localizations >= expected_count

    loc_path = (
        input_path.with_suffix("")
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    assert loc_path.is_file(), f"Missing final localization output: {loc_path}"

    locs = np.load(loc_path)
    assert locs.dtype.names is not None

    if "fit_success" in locs.dtype.names:
        locs = locs[locs["fit_success"]]

    assert locs.size >= expected_count

    for truth in blinks[:expected_count]:
        matched = _localizations_near_time(
            locs,
            truth.peak_us,
            max_abs_time_error_us=max_abs_time_error_us,
        )
        assert matched.size > 0, f"No localization found near t={truth.peak_us} us."

        best, best_dist = _best_spatial_match(matched, truth)

        assert best_dist <= max_spatial_error_px, (
            f"Localization mismatch for {name} at t={truth.peak_us} us. "
            f"Expected x={truth.x_px:.2f}, y={truth.y_px:.2f}; "
            f"got x={float(best['x']):.2f}, y={float(best['y']):.2f}; "
            f"distance={best_dist:.2f} px."
        )

        assert int(best["E_total"]) >= min_events_per_polarity
        assert int(best["E_total_n"]) >= min_events_per_polarity

    return locs

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
        start_us=blink.peak_us + blink.pos_start_offset_us,
        stop_us=blink.peak_us + blink.pos_stop_offset_us,
        count=blink.n_pos,
    )
    neg_times = _unique_burst_times(
        start_us=blink.peak_us + blink.neg_start_offset_us,
        stop_us=blink.peak_us + blink.neg_stop_offset_us,
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

        # Important for this synthetic generator:
        # it creates signal events, but no explicit dark/blank/local background.
        background_mode="calibrated_only",

        hot_pixel_policy="mask",

        # Keep these non-trivial. Do not lower them just to pass.
        min_events_pos=100,
        min_events_neg=100,

        min_valid_pixels=1,

        # Start with this; only raise if diagnostics show condition-only rejection.
        max_fit_cond=1e15,
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

def _log_synthetic_pipeline_artifacts(input_path: Path, config: PeakLocConfig) -> None:
    temp_dir = input_path.with_suffix("") / "temp_files"

    logger.warning("=== Synthetic pipeline diagnostic ===")
    logger.warning("input_path={}", input_path)
    logger.warning("temp_dir={} exists={}", temp_dir, temp_dir.is_dir())
    logger.warning(
        "config: fit_model={} background_mode={} sigma_psf_px={} "
        "min_events_pos={} min_events_neg={} max_fit_cond={} "
        "polarity_time_gate_us={} prominence={} peak_min_event_count={} "
        "slice_duration={}",
        config.fit_model,
        config.background_mode,
        config.sigma_psf_px,
        config.min_events_pos,
        config.min_events_neg,
        config.max_fit_cond,
        config.polarity_time_gate_us,
        config.prominence,
        config.peak_min_event_count,
        config.slice_duration,
    )

    if not temp_dir.is_dir():
        logger.warning("No temp_files directory was created.")
        return

    roi_paths = sorted(temp_dir.glob("rois_*.npy"))
    loc_paths = sorted(temp_dir.glob("localizations_*.npy"))

    logger.warning("roi_files={}", [path.name for path in roi_paths])
    logger.warning("localization_files={}", [path.name for path in loc_paths])

    for roi_path in roi_paths:
        rois = np.load(roi_path)
        _log_roi_file_summary(roi_path, rois, config)

    for loc_path in loc_paths:
        locs = np.load(loc_path)
        _log_localization_file_summary(loc_path, locs)


def _log_roi_file_summary(
    roi_path: Path,
    rois: np.ndarray,
    config: PeakLocConfig,
) -> None:
    logger.warning("--- ROI file: {} ---", roi_path.name)
    logger.warning("roi_count={}", rois.size)

    if rois.size == 0:
        return

    eligible = (
        (rois["total_events_roi"] >= config.min_events_pos)
        & (rois["total_neg_events_roi"] >= config.min_events_neg)
    )

    logger.warning(
        "eligible_rois={} / {} using min_events_pos={} min_events_neg={}",
        int(np.count_nonzero(eligible)),
        rois.size,
        config.min_events_pos,
        config.min_events_neg,
    )
    logger.warning(
        "pos totals: min={} median={} max={} first10={}",
        int(np.min(rois["total_events_roi"])),
        float(np.median(rois["total_events_roi"])),
        int(np.max(rois["total_events_roi"])),
        rois["total_events_roi"][:10].tolist(),
    )
    logger.warning(
        "neg totals: min={} median={} max={} first10={}",
        int(np.min(rois["total_neg_events_roi"])),
        float(np.median(rois["total_neg_events_roi"])),
        int(np.max(rois["total_neg_events_roi"])),
        rois["total_neg_events_roi"][:10].tolist(),
    )
    logger.warning("t_peak first10={}", rois["t_peak"][:10].tolist())
    logger.warning("t_1st first10={}", rois["t_1st"][:10].tolist())
    logger.warning("t_last first10={}", rois["t_last"][:10].tolist())
    logger.warning("peak first10={}", rois["peak"][:10].tolist())
    logger.warning("rel_peak first10={}", rois["rel_peak"][:10].tolist())
    logger.warning("roi_y0 first10={}", rois["roi_y0"][:10].tolist())
    logger.warning("roi_x0 first10={}", rois["roi_x0"][:10].tolist())
    logger.warning("dt_pos_s first10={}", rois["dt_pos_s"][:10].tolist())
    logger.warning("dt_neg_s first10={}", rois["dt_neg_s"][:10].tolist())

    best_idx = int(
        np.argmax(
            rois["total_events_roi"].astype(np.int64)
            + rois["total_neg_events_roi"].astype(np.int64)
        )
    )
    logger.warning(
        "best ROI idx={} t_peak={} peak={} rel_peak={} roi_y0={} roi_x0={} "
        "pos={} neg={} dt_pos_s={} dt_neg_s={}",
        best_idx,
        int(rois["t_peak"][best_idx]),
        rois["peak"][best_idx].tolist(),
        rois["rel_peak"][best_idx].tolist(),
        int(rois["roi_y0"][best_idx]),
        int(rois["roi_x0"][best_idx]),
        int(rois["total_events_roi"][best_idx]),
        int(rois["total_neg_events_roi"][best_idx]),
        float(rois["dt_pos_s"][best_idx]),
        float(rois["dt_neg_s"][best_idx]),
    )
    logger.warning("best ROI positive map:\n{}", rois["roi"][best_idx])
    logger.warning("best ROI negative map:\n{}", rois["roi_n"][best_idx])


def _log_localization_file_summary(loc_path: Path, locs: np.ndarray) -> None:
    logger.warning("--- Localization file: {} ---", loc_path.name)
    logger.warning("localization_count={}", locs.size)

    if locs.size == 0:
        return

    logger.warning("dtype names={}", locs.dtype.names)

    if locs.dtype.names is None:
        return

    fields_to_log = [
        "id",
        "t_peak",
        "x",
        "y",
        "sub_x",
        "sub_y",
        "E_total",
        "E_total_n",
        "fit_success",
        "fit_status",
        "fit_cond",
        "valid_pixel_count",
        "sigma_x",
        "sigma_y",
        "nll_per_event",
    ]
    present = [field for field in fields_to_log if field in locs.dtype.names]

    for idx in range(min(locs.size, 10)):
        logger.warning(
            "loc[{}]: {}",
            idx,
            {field: locs[field][idx].item() if np.ndim(locs[field][idx]) == 0 else locs[field][idx].tolist()
             for field in present},
        )