import numpy as np

from localization_scripts.synthetic_event_camera import (
    EVENT_DTYPE,
    EventCameraModel,
    SyntheticBlink,
    benchmark_scenarios,
    generate_synthetic_event_recording,
)


def test_event_arrays_are_dot_events_with_peakloc_dtype():
    recording = generate_synthetic_event_recording(
        (_blink(),),
        sensor_shape=(32, 32),
        model=_model(),
        sample_step_us=1_000,
    )

    assert recording.events.dtype == EVENT_DTYPE
    assert set(recording.events.dtype.names) == {"x", "y", "p", "t"}
    assert "t_start" not in recording.events.dtype.names
    assert "t_stop" not in recording.events.dtype.names
    assert recording.events.size > 0


def test_positive_events_concentrate_during_rise_and_negative_during_fall():
    blink = _blink()
    recording = generate_synthetic_event_recording(
        (blink,),
        sensor_shape=(32, 32),
        model=_model(),
        sample_step_us=1_000,
    )
    events = recording.events
    rise = events[
        (events["t"] >= blink.start_us)
        & (events["t"] <= blink.start_us + blink.rise_us)
    ]
    fall = events[
        (events["t"] >= blink.start_us + blink.rise_us + blink.plateau_us)
        & (events["t"] <= blink.stop_us)
    ]

    assert np.count_nonzero(rise["p"] == 1) > 0.8 * np.count_nonzero(events["p"] == 1)
    assert np.count_nonzero(fall["p"] == 0) > 0.8 * np.count_nonzero(events["p"] == 0)


def test_expected_event_counts_are_stored_per_blink():
    recording = generate_synthetic_event_recording(
        (_blink(), _blink(x=20.0, y=20.0, start_us=120_000)),
        sensor_shape=(32, 32),
        model=_model(),
        sample_step_us=1_000,
    )

    assert len(recording.expected_pos_events) == 2
    assert len(recording.expected_neg_events) == 2
    assert all(count > 0 for count in recording.expected_pos_events)
    assert all(count > 0 for count in recording.expected_neg_events)


def test_benchmark_scenarios_include_overlap_xfail_cases():
    scenarios = benchmark_scenarios()

    names = {scenario.name for scenario in scenarios}
    assert "single_bright_isolated" in names
    assert "hot_pixel_contamination" in names
    assert any(scenario.expected_xfail for scenario in scenarios)


def _blink(
    *,
    x: float = 16.0,
    y: float = 16.0,
    start_us: int = 10_000,
) -> SyntheticBlink:
    return SyntheticBlink(
        x_px=x,
        y_px=y,
        start_us=start_us,
        rise_us=20_000,
        plateau_us=10_000,
        fall_us=20_000,
        amplitude=60.0,
        sigma_px=1.5,
    )


def _model() -> EventCameraModel:
    return EventCameraModel(
        contrast_threshold_pos=0.08,
        contrast_threshold_neg=0.08,
        threshold_jitter=0.0,
        refractory_us=50.0,
        dark_rate_pos_hz=0.0,
        dark_rate_neg_hz=0.0,
        hot_pixel_rate_hz=0.0,
        seed=1,
    )
