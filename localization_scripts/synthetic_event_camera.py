from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),
        ("t", np.uint64),
    ]
)


@dataclass(frozen=True)
class SyntheticBlink:
    x_px: float
    y_px: float
    start_us: int
    rise_us: int
    plateau_us: int
    fall_us: int
    amplitude: float
    sigma_px: float

    @property
    def peak_us(self) -> int:
        return self.start_us + self.rise_us

    @property
    def stop_us(self) -> int:
        return self.start_us + self.rise_us + self.plateau_us + self.fall_us


@dataclass(frozen=True)
class EventCameraModel:
    contrast_threshold_pos: float
    contrast_threshold_neg: float
    threshold_jitter: float
    refractory_us: float
    dark_rate_pos_hz: float
    dark_rate_neg_hz: float
    hot_pixel_rate_hz: float
    seed: int


@dataclass(frozen=True)
class SyntheticEventRecording:
    events: np.ndarray
    blinks: tuple[SyntheticBlink, ...]
    expected_pos_events: tuple[int, ...]
    expected_neg_events: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticBenchmarkScenario:
    name: str
    blinks: tuple[SyntheticBlink, ...]
    expected_xfail: bool = False


def generate_synthetic_event_recording(
    blinks: tuple[SyntheticBlink, ...],
    *,
    sensor_shape: tuple[int, int],
    model: EventCameraModel,
    background: float = 1.0,
    sample_step_us: int = 1_000,
) -> SyntheticEventRecording:
    if sample_step_us <= 0:
        raise ValueError("sample_step_us must be positive")
    rng = np.random.default_rng(model.seed)
    records: list[tuple[int, int, int, int]] = []
    expected_pos = []
    expected_neg = []
    for blink in blinks:
        blink_records = _events_for_blink(
            blink,
            sensor_shape=sensor_shape,
            model=model,
            background=background,
            sample_step_us=sample_step_us,
            rng=rng,
        )
        expected_pos.append(
            sum(1 for _, _, polarity, _ in blink_records if polarity == 1)
        )
        expected_neg.append(
            sum(1 for _, _, polarity, _ in blink_records if polarity == 0)
        )
        records.extend(blink_records)

    duration_us = max((blink.stop_us for blink in blinks), default=sample_step_us)
    records.extend(
        _noise_events(
            sensor_shape=sensor_shape,
            duration_us=duration_us,
            model=model,
            rng=rng,
        )
    )
    events = np.asarray(records, dtype=EVENT_DTYPE)
    if events.size:
        events = _deduplicate_per_pixel_timestamp_collisions(np.sort(events, order="t"))
    return SyntheticEventRecording(
        events=events,
        blinks=blinks,
        expected_pos_events=tuple(expected_pos),
        expected_neg_events=tuple(expected_neg),
    )


def benchmark_scenarios() -> tuple[SyntheticBenchmarkScenario, ...]:
    bright = SyntheticBlink(32.0, 32.0, 10_000, 20_000, 20_000, 20_000, 80.0, 1.7)
    dim = SyntheticBlink(32.0, 32.0, 10_000, 20_000, 20_000, 20_000, 12.0, 1.7)
    return (
        SyntheticBenchmarkScenario("single_bright_isolated", (bright,)),
        SyntheticBenchmarkScenario("single_dim_near_threshold", (dim,)),
        SyntheticBenchmarkScenario(
            "two_spatially_separated_same_time",
            (
                bright,
                SyntheticBlink(52.0, 52.0, 10_000, 20_000, 20_000, 20_000, 80.0, 1.7),
            ),
        ),
        SyntheticBenchmarkScenario(
            "two_spatially_overlapping_different_times",
            (
                bright,
                SyntheticBlink(33.5, 32.5, 90_000, 20_000, 20_000, 20_000, 80.0, 1.7),
            ),
        ),
        SyntheticBenchmarkScenario(
            "two_temporally_overlapping_nearby",
            (
                bright,
                SyntheticBlink(33.0, 32.0, 15_000, 20_000, 20_000, 20_000, 80.0, 1.7),
            ),
            expected_xfail=True,
        ),
        SyntheticBenchmarkScenario(
            "boundary_blink_near_roi_edge",
            (SyntheticBlink(3.2, 3.5, 10_000, 20_000, 20_000, 20_000, 80.0, 1.7),),
        ),
        SyntheticBenchmarkScenario("high_background_field", (bright,)),
        SyntheticBenchmarkScenario("hot_pixel_contamination", (bright,)),
        SyntheticBenchmarkScenario(
            "slow_rise_fall",
            (SyntheticBlink(32.0, 32.0, 10_000, 80_000, 20_000, 80_000, 80.0, 1.7),),
        ),
        SyntheticBenchmarkScenario("known_drift_localizations", (bright,)),
    )


def _events_for_blink(
    blink: SyntheticBlink,
    *,
    sensor_shape: tuple[int, int],
    model: EventCameraModel,
    background: float,
    sample_step_us: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    records: list[tuple[int, int, int, int]] = []
    radius = max(1, int(np.ceil(4 * blink.sigma_px)))
    y_min = max(0, int(np.floor(blink.y_px)) - radius)
    y_max = min(sensor_shape[0] - 1, int(np.ceil(blink.y_px)) + radius)
    x_min = max(0, int(np.floor(blink.x_px)) - radius)
    x_max = min(sensor_shape[1] - 1, int(np.ceil(blink.x_px)) + radius)
    times = np.arange(blink.start_us, blink.stop_us + sample_step_us, sample_step_us)
    if times[-1] != blink.stop_us:
        times = np.append(times, blink.stop_us)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            weight = _psf_weight(x, y, blink)
            if weight < 1e-4:
                continue
            intensity = background + blink.amplitude * weight * np.asarray(
                [_temporal_profile(time_us, blink) for time_us in times]
            )
            log_intensity = np.log(np.maximum(intensity, np.finfo(float).tiny))
            records.extend(
                _threshold_crossings_for_trace(
                    x,
                    y,
                    times,
                    log_intensity,
                    model,
                    rng,
                )
            )
    return records


def _threshold_crossings_for_trace(
    x: int,
    y: int,
    times: np.ndarray,
    log_intensity: np.ndarray,
    model: EventCameraModel,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    records: list[tuple[int, int, int, int]] = []
    last_event_time = {0: -np.inf, 1: -np.inf}
    for index in range(1, times.size):
        previous = float(log_intensity[index - 1])
        current = float(log_intensity[index])
        delta = current - previous
        if delta == 0:
            continue
        polarity = 1 if delta > 0 else 0
        base_threshold = (
            model.contrast_threshold_pos
            if polarity == 1
            else model.contrast_threshold_neg
        )
        threshold = _jittered_threshold(base_threshold, model.threshold_jitter, rng)
        crossing_count = int(np.floor(abs(delta) / threshold))
        if crossing_count == 0:
            continue
        for crossing_index in range(1, crossing_count + 1):
            alpha = crossing_index / (crossing_count + 1)
            event_time = int(
                round(times[index - 1] + alpha * (times[index] - times[index - 1]))
            )
            if event_time - last_event_time[polarity] < model.refractory_us:
                continue
            records.append((x, y, polarity, event_time))
            last_event_time[polarity] = event_time
    return records


def _noise_events(
    *,
    sensor_shape: tuple[int, int],
    duration_us: int,
    model: EventCameraModel,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    records: list[tuple[int, int, int, int]] = []
    duration_s = duration_us / 1_000_000
    pixel_count = sensor_shape[0] * sensor_shape[1]
    for polarity, rate_hz in ((1, model.dark_rate_pos_hz), (0, model.dark_rate_neg_hz)):
        event_count = int(rng.poisson(max(rate_hz, 0.0) * duration_s * pixel_count))
        records.extend(
            _random_events(sensor_shape, event_count, polarity, duration_us, rng)
        )

    hot_pixel_count = (
        0 if model.hot_pixel_rate_hz <= 0 else max(1, pixel_count // 1_000)
    )
    for _ in range(hot_pixel_count):
        y = int(rng.integers(0, sensor_shape[0]))
        x = int(rng.integers(0, sensor_shape[1]))
        event_count = int(rng.poisson(model.hot_pixel_rate_hz * duration_s))
        for time_us in rng.integers(0, max(duration_us, 1), size=event_count):
            records.append((x, y, int(rng.integers(0, 2)), int(time_us)))
    return records


def _random_events(
    sensor_shape: tuple[int, int],
    event_count: int,
    polarity: int,
    duration_us: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    if event_count <= 0:
        return []
    xs = rng.integers(0, sensor_shape[1], size=event_count)
    ys = rng.integers(0, sensor_shape[0], size=event_count)
    times = rng.integers(0, max(duration_us, 1), size=event_count)
    return [
        (int(x), int(y), int(polarity), int(time_us))
        for x, y, time_us in zip(xs, ys, times)
    ]


def _temporal_profile(time_us: int, blink: SyntheticBlink) -> float:
    if time_us < blink.start_us or time_us > blink.stop_us:
        return 0.0
    rise_end = blink.start_us + blink.rise_us
    plateau_end = rise_end + blink.plateau_us
    if time_us <= rise_end:
        return (time_us - blink.start_us) / max(blink.rise_us, 1)
    if time_us <= plateau_end:
        return 1.0
    return max(0.0, 1.0 - (time_us - plateau_end) / max(blink.fall_us, 1))


def _psf_weight(x: int, y: int, blink: SyntheticBlink) -> float:
    return float(
        np.exp(
            -0.5
            * (
                ((x - blink.x_px) / blink.sigma_px) ** 2
                + ((y - blink.y_px) / blink.sigma_px) ** 2
            )
        )
    )


def _jittered_threshold(
    base_threshold: float, threshold_jitter: float, rng: np.random.Generator
) -> float:
    if base_threshold <= 0:
        raise ValueError("contrast thresholds must be positive")
    if threshold_jitter <= 0:
        return base_threshold
    return max(base_threshold * 0.05, base_threshold + rng.normal(0, threshold_jitter))


def _deduplicate_per_pixel_timestamp_collisions(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return events
    keys = np.zeros(
        events.size, dtype=[("y", np.uint16), ("x", np.uint16), ("t", np.uint64)]
    )
    keys["y"] = events["y"]
    keys["x"] = events["x"]
    keys["t"] = events["t"]
    _, unique_indices = np.unique(keys, return_index=True)
    return np.sort(events[np.sort(unique_indices)], order="t")
