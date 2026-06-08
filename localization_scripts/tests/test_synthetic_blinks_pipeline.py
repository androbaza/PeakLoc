from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import os
from pathlib import Path

from loguru import logger
import numpy as np
import pytest
from scipy.special import erf

from localization_scripts.debug_visualization import (
    DebugVisualizationConfig,
    TruthPoint,
    save_synthetic_localization_debug_artifacts,
)
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import process_recording

EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),  # PeakLoc convention: 1 = positive, 0 = negative
        ("t", np.uint64),  # microseconds
    ]
)

MINIMAL_LOCALIZATION_DTYPE = np.dtype(
    [
        ("id", np.uint64),
        ("t_peak", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("E_total", np.uint64),
        ("E_total_n", np.uint64),
        ("fit_success", np.bool_),
    ]
)


@dataclass(frozen=True)
class BlinkTruth:
    x_px: float
    y_px: float
    peak_us: int
    n_pos: int = 80
    n_neg: int = 60
    signal_peak: float = 200
    background: float = 1.0
    contrast_threshold_log: float = 0.08
    negative_event_bias: float = 0.8
    refractory_period_us: int = 5
    turn_on_duration_us: int = 80_000
    plateau_duration_us: int = 10_000
    turn_off_duration_us: int = 50_000
    sample_step_us: int = 250
    signal_event_jitter_us: int = 120
    signal_event_dropout_probability: float = 0.08
    signal_event_extra_probability: float = 0.15
    background_noise_events_per_pixel: float = 4.0
    label: str | None = None


SENSOR_HEIGHT = 96
SENSOR_WIDTH = 96
SENSOR_SHAPE = (SENSOR_HEIGHT, SENSOR_WIDTH)

SIGMA_PSF_PX = 1.703
DATASET_FWHM_PX = 2.355 * SIGMA_PSF_PX
ROI_RADIUS = 7

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
        BlinkTruth(
            x_px=72.20, y_px=70.60, peak_us=1_000_000, n_pos=12_000, n_neg=12_000
        ),
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
            turn_on_duration_us=120_000,
            turn_off_duration_us=120_000,
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
            turn_on_duration_us=40_000,
            turn_off_duration_us=40_000,
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


def test_synthetic_blink_plateau_contains_only_sparse_background_noise() -> None:
    blink = BlinkTruth(
        x_px=30.35,
        y_px=31.70,
        peak_us=200_000,
        plateau_duration_us=20_000,
    )
    events = synthetic_event_recording(
        blinks=(blink,),
        sensor_shape=SENSOR_SHAPE,
        sigma_px=SIGMA_PSF_PX,
        support_radius_px=7,
    )

    plateau_start = blink.peak_us
    plateau_stop = blink.peak_us + blink.plateau_duration_us
    plateau_events = events[
        (events["t"] >= plateau_start) & (events["t"] <= plateau_stop)
    ]

    assert 0 < plateau_events.size < 0.15 * events.size


def test_synthetic_event_times_are_not_ordered_by_pixel_scan_order() -> None:
    blink = BlinkTruth(x_px=40.0, y_px=40.0, peak_us=200_000)
    events = synthetic_event_recording(
        blinks=(blink,),
        sensor_shape=SENSOR_SHAPE,
        sigma_px=SIGMA_PSF_PX,
        support_radius_px=7,
    )

    pos = events[events["p"] == 1]
    pixel_ids = pos["y"].astype(np.int64) * SENSOR_WIDTH + pos["x"].astype(np.int64)

    unique_ids = np.unique(pixel_ids)
    median_t = []
    scan_id = []
    for pixel_id in unique_ids:
        mask = pixel_ids == pixel_id
        if np.count_nonzero(mask) >= 2:
            scan_id.append(pixel_id)
            median_t.append(float(np.median(pos["t"][mask])))

    corr = np.corrcoef(scan_id, median_t)[0, 1]
    assert abs(corr) < 0.3


def test_synthetic_negative_signal_events_are_biased_lower_than_positive() -> None:
    blink = BlinkTruth(
        x_px=40.0,
        y_px=40.0,
        peak_us=200_000,
        signal_event_jitter_us=0,
        signal_event_dropout_probability=0.0,
        signal_event_extra_probability=0.0,
        background_noise_events_per_pixel=0.0,
    )
    events = synthetic_event_recording(
        blinks=(blink,),
        sensor_shape=SENSOR_SHAPE,
        sigma_px=SIGMA_PSF_PX,
        support_radius_px=7,
    )

    positive_count = int(np.count_nonzero(events["p"] == 1))
    negative_count = int(np.count_nonzero(events["p"] == 0))
    negative_ratio = negative_count / positive_count

    assert negative_ratio == pytest.approx(blink.negative_event_bias, abs=0.08)


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

    expected_count = (
        len(blinks)
        if min_successful_localizations is None
        else min_successful_localizations
    )

    loc_path = (
        input_path.with_suffix("")
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    locs = _load_final_localizations_if_available(loc_path)

    save_synthetic_localization_debug_artifacts(
        events=events,
        localizations=locs,
        attempted_localizations=_load_temp_arrays(
            input_path, "attempted_localizations"
        ),
        rois=_load_temp_arrays(input_path, "rois"),
        truth=_truth_points(blinks),
        config=DebugVisualizationConfig(
            output_dir=_debug_output_root() / name,
            scenario_name=name,
            sensor_shape=SENSOR_SHAPE,
            optical_pixel_size_nm=config.optical_pixel_size_nm,
            max_spatial_error_px=max_spatial_error_px,
            max_abs_time_error_us=max_abs_time_error_us,
            min_events_per_polarity=min_events_per_polarity,
            overwrite=True,
        ),
        test_status="pre_assertion",
    )

    assert result.event_count == events.size
    assert len(result.slice_results) >= 1

    total_rois = sum(s.roi_count for s in result.slice_results)
    total_localizations = sum(s.localization_count for s in result.slice_results)

    assert total_rois >= expected_count
    assert total_localizations >= expected_count
    assert loc_path.is_file(), f"Missing final localization output: {loc_path}"
    assert locs.dtype.names is not None
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


def _truth_points(blinks: tuple[BlinkTruth, ...]) -> list[TruthPoint]:
    return [
        TruthPoint(
            x_px=blink.x_px,
            y_px=blink.y_px,
            peak_us=blink.peak_us,
            label=f"truth_{idx}",
            n_pos=blink.n_pos,
            n_neg=blink.n_neg,
        )
        for idx, blink in enumerate(blinks)
    ]


def _debug_output_root() -> Path:
    persistent = os.environ.get("PEAKLOC_DEBUG_ARTIFACT_DIR")
    if persistent:
        return Path(persistent) / "synthetic_blinks"
    return Path("debug_artifacts") / "synthetic_blinks"


def _load_temp_arrays(input_path: Path, prefix: str) -> np.ndarray | None:
    temp_dir = input_path.with_suffix("") / "temp_files"
    paths = sorted(temp_dir.glob(f"{prefix}_*.npy"))
    arrays = [np.load(path) for path in paths if path.is_file()]
    arrays = [array for array in arrays if array.size > 0]
    if not arrays:
        return None
    return np.concatenate(arrays)


def _load_final_localizations_if_available(loc_path: Path) -> np.ndarray:
    if not loc_path.is_file():
        return np.empty(0, dtype=MINIMAL_LOCALIZATION_DTYPE)
    locs = np.load(loc_path)
    if locs.dtype.names is None:
        return np.empty(0, dtype=MINIMAL_LOCALIZATION_DTYPE)
    if "fit_success" in locs.dtype.names:
        return locs[locs["fit_success"]]
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
    events = _deduplicate_per_pixel_timestamp_collisions(events)

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
    t0 = blink.peak_us - blink.turn_on_duration_us - blink.sample_step_us
    t1 = (
        blink.peak_us
        + blink.plateau_duration_us
        + blink.turn_off_duration_us
        + blink.sample_step_us
    )
    times = np.arange(t0, t1 + 1, blink.sample_step_us, dtype=np.int64)
    envelope = _blink_envelope(times, blink)

    rng = np.random.default_rng(_blink_seed(blink))
    occupied: set[tuple[int, int, int]] = set()
    records: list[tuple[int, int, int, int]] = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            signal = blink.signal_peak * float(weights[iy, ix]) * envelope
            intensity = blink.background + signal
            log_intensity = np.log(np.maximum(intensity, 1e-12))
            signal_events = _threshold_crossing_events_for_pixel(
                x=int(x),
                y=int(y),
                times_us=times,
                log_intensity=log_intensity,
                positive_contrast_threshold_log=blink.contrast_threshold_log,
                negative_contrast_threshold_log=(
                    blink.contrast_threshold_log / blink.negative_event_bias
                ),
                refractory_period_us=blink.refractory_period_us,
            )
            records.extend(
                _noisy_signal_events(
                    signal_events,
                    blink=blink,
                    rng=rng,
                    min_time_us=t0,
                    max_time_us=t1,
                    occupied=occupied,
                )
            )
    records.extend(
        _background_noise_events_for_blink(
            xs=xs,
            ys=ys,
            blink=blink,
            rng=rng,
            min_time_us=t0,
            max_time_us=t1,
            occupied=occupied,
        )
    )
    return records


def _smoothstep(u: np.ndarray) -> np.ndarray:
    clipped = np.clip(u, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _blink_envelope(times_us: np.ndarray, blink: BlinkTruth) -> np.ndarray:
    on_start = blink.peak_us - blink.turn_on_duration_us
    plateau_start = blink.peak_us
    plateau_stop = blink.peak_us + blink.plateau_duration_us
    off_stop = plateau_stop + blink.turn_off_duration_us

    envelope = np.zeros_like(times_us, dtype=np.float64)

    on = (times_us >= on_start) & (times_us < plateau_start)
    envelope[on] = _smoothstep((times_us[on] - on_start) / blink.turn_on_duration_us)

    plateau = (times_us >= plateau_start) & (times_us < plateau_stop)
    envelope[plateau] = 1.0

    off = (times_us >= plateau_stop) & (times_us <= off_stop)
    envelope[off] = 1.0 - _smoothstep(
        (times_us[off] - plateau_stop) / blink.turn_off_duration_us
    )

    return envelope


def _threshold_crossing_events_for_pixel(
    *,
    x: int,
    y: int,
    times_us: np.ndarray,
    log_intensity: np.ndarray,
    positive_contrast_threshold_log: float,
    negative_contrast_threshold_log: float,
    refractory_period_us: int,
) -> list[tuple[int, int, int, int]]:
    records: list[tuple[int, int, int, int]] = []
    last_crossing = float(log_intensity[0])
    last_event_time = -(10**18)

    for idx in range(1, len(times_us)):
        t_prev = int(times_us[idx - 1])
        t_curr = int(times_us[idx])
        l_prev = float(log_intensity[idx - 1])
        l_curr = float(log_intensity[idx])

        if l_curr == l_prev:
            continue

        while True:
            delta = l_curr - last_crossing
            if delta >= positive_contrast_threshold_log:
                polarity = 1
                target = last_crossing + positive_contrast_threshold_log
            elif delta <= -negative_contrast_threshold_log:
                polarity = 0
                target = last_crossing - negative_contrast_threshold_log
            else:
                break

            alpha = (target - l_prev) / (l_curr - l_prev)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            event_time = int(round(t_prev + alpha * (t_curr - t_prev)))

            if event_time - last_event_time >= refractory_period_us:
                records.append((x, y, polarity, event_time))
                last_event_time = event_time

            last_crossing = target

    return records


def _noisy_signal_events(
    events: list[tuple[int, int, int, int]],
    *,
    blink: BlinkTruth,
    rng: np.random.Generator,
    min_time_us: int,
    max_time_us: int,
    occupied: set[tuple[int, int, int]],
) -> list[tuple[int, int, int, int]]:
    noisy: list[tuple[int, int, int, int]] = []
    for x, y, polarity, time_us in events:
        if rng.random() < blink.signal_event_dropout_probability:
            continue
        event_time = _jittered_event_time(
            time_us,
            jitter_us=blink.signal_event_jitter_us,
            min_time_us=min_time_us,
            max_time_us=max_time_us,
            rng=rng,
        )
        _append_unique_event(noisy, occupied, x, y, polarity, event_time)
        if rng.random() < blink.signal_event_extra_probability:
            extra_time = _jittered_event_time(
                event_time,
                jitter_us=max(blink.signal_event_jitter_us, blink.refractory_period_us),
                min_time_us=min_time_us,
                max_time_us=max_time_us,
                rng=rng,
            )
            _append_unique_event(noisy, occupied, x, y, polarity, extra_time)
    return noisy


def _background_noise_events_for_blink(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    blink: BlinkTruth,
    rng: np.random.Generator,
    min_time_us: int,
    max_time_us: int,
    occupied: set[tuple[int, int, int]],
) -> list[tuple[int, int, int, int]]:
    records: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            positive_count = int(rng.poisson(blink.background_noise_events_per_pixel))
            negative_count = int(
                rng.poisson(
                    blink.background_noise_events_per_pixel * blink.negative_event_bias
                )
            )
            for polarity, count in [(1, positive_count), (0, negative_count)]:
                if count == 0:
                    continue
                times = rng.integers(
                    min_time_us,
                    max_time_us + 1,
                    size=count,
                    endpoint=False,
                )
                for time_us in times:
                    _append_unique_event(
                        records,
                        occupied,
                        int(x),
                        int(y),
                        polarity,
                        int(time_us),
                    )
    return records


def _jittered_event_time(
    time_us: int,
    *,
    jitter_us: int,
    min_time_us: int,
    max_time_us: int,
    rng: np.random.Generator,
) -> int:
    if jitter_us <= 0:
        return int(time_us)
    jitter = int(rng.integers(-jitter_us, jitter_us + 1))
    return int(np.clip(time_us + jitter, min_time_us, max_time_us))


def _append_unique_event(
    records: list[tuple[int, int, int, int]],
    occupied: set[tuple[int, int, int]],
    x: int,
    y: int,
    polarity: int,
    time_us: int,
) -> None:
    key = (y, x, time_us)
    if key in occupied:
        return
    records.append((x, y, polarity, time_us))
    occupied.add(key)


def _blink_seed(blink: BlinkTruth) -> int:
    return int(
        (
            blink.peak_us * 31
            + round(blink.x_px * 1_000) * 131
            + round(blink.y_px * 1_000) * 257
        )
        % (2**32)
    )


def _pixel_integrated_gaussian_weights(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    sigma_px: float,
) -> np.ndarray:
    sqrt2_sigma = np.sqrt(2.0) * sigma_px

    wx = 0.5 * (erf((xs + 0.5 - x0) / sqrt2_sigma) - erf((xs - 0.5 - x0) / sqrt2_sigma))
    wy = 0.5 * (erf((ys + 0.5 - y0) / sqrt2_sigma) - erf((ys - 0.5 - y0) / sqrt2_sigma))

    weights = np.outer(wy, wx)
    return np.clip(weights, 0.0, None)


def _assert_no_per_pixel_timestamp_collisions(events: np.ndarray) -> None:
    seen: set[tuple[int, int, int]] = set()

    for event in events:
        key = (int(event["y"]), int(event["x"]), int(event["t"]))
        assert key not in seen, (
            "Synthetic event generator created duplicate timestamps for the "
            "same pixel. That would make per-pixel event counting ambiguous."
        )
        seen.add(key)


def _deduplicate_per_pixel_timestamp_collisions(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return events
    keys = np.empty(
        events.size,
        dtype=[("y", np.uint16), ("x", np.uint16), ("t", np.uint64)],
    )
    keys["y"] = events["y"]
    keys["x"] = events["x"]
    keys["t"] = events["t"]
    _, unique_indices = np.unique(keys, return_index=True)
    if unique_indices.size == events.size:
        return events
    return np.sort(events[np.sort(unique_indices)], order="t")


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

    eligible = (rois["total_events_roi"] >= config.min_events_pos) & (
        rois["total_neg_events_roi"] >= config.min_events_neg
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
            {
                field: locs[field][idx].item()
                if np.ndim(locs[field][idx]) == 0
                else locs[field][idx].tolist()
                for field in present
            },
        )
