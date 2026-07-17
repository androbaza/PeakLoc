from __future__ import annotations

import numpy as np

from localization_scripts.event_array_processing import EVENT_DTYPE
from localization_scripts.temporal_roi_generation import (
    generate_temporally_segmented_rois,
)
from localization_scripts.temporal_segmentation import (
    TemporalSegmentationSettings,
    _signed_root_poisson_deviance,
    segment_candidate_events,
)


def _event_array(records: list[tuple[int, int, int, int]]) -> np.ndarray:
    events = np.asarray(records, dtype=EVENT_DTYPE)
    return np.sort(events, order="t", kind="stable")


def _train(
    *,
    center_x: int,
    center_y: int,
    polarity: int,
    start_us: int,
    count: int,
    pixel_count: int,
    step_us: int = 1_000,
) -> list[tuple[int, int, int, int]]:
    offsets = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
    )
    return [
        (
            center_x + offsets[index % pixel_count][0],
            center_y + offsets[index % pixel_count][1],
            polarity,
            start_us + index * step_us,
        )
        for index in range(count)
    ]


def test_segment_candidate_selects_compact_on_and_off_trains() -> None:
    records = [
        *_train(
            center_x=20,
            center_y=20,
            polarity=1,
            start_us=70_250,
            count=18,
            pixel_count=8,
        ),
        *_train(
            center_x=20,
            center_y=20,
            polarity=0,
            start_us=112_500,
            count=10,
            pixel_count=6,
        ),
        (25, 25, 1, 20_000),
        (25, 25, 0, 180_000),
    ]

    result = segment_candidate_events(
        _event_array(records),
        seed_peak_us=100_000,
        seed_x=20,
        seed_y=20,
        roi_radius_px=6,
    )

    assert result.accepted
    assert result.rejection_reason == "accepted"
    assert result.interval is not None
    assert result.interval.on_train.first_event_us == 70_250
    assert result.interval.on_train.last_event_us == 87_250
    assert result.interval.off_train.first_event_us == 112_500
    assert result.interval.off_train.last_event_us == 121_500
    assert result.interval.on_train.support_start_us == 70_000
    assert result.interval.on_train.support_stop_us == 88_000
    assert result.interval.off_train.support_start_us == 112_000
    assert result.interval.off_train.support_stop_us == 122_000
    assert result.interval.quiet_dwell_us == 25_250
    assert result.selected_on_event_indices.size == result.interval.on_train.event_count
    assert (
        result.selected_off_event_indices.size == result.interval.off_train.event_count
    )


def test_segment_candidate_allows_small_cross_pixel_transition_overlap() -> None:
    records = [
        *_train(
            center_x=20,
            center_y=20,
            polarity=1,
            start_us=82_000,
            count=18,
            pixel_count=8,
        ),
        *_train(
            center_x=20,
            center_y=20,
            polarity=0,
            start_us=96_000,
            count=18,
            pixel_count=6,
        ),
    ]

    result = segment_candidate_events(
        _event_array(records),
        seed_peak_us=100_000,
        seed_x=20,
        seed_y=20,
        roi_radius_px=6,
    )

    assert result.accepted
    assert result.interval is not None
    assert result.interval.endpoint_overlap_us == 3_000
    assert result.interval.quiet_dwell_us == 0


def test_segment_candidate_uses_stricter_on_than_off_event_gaps() -> None:
    records = [
        *_train(
            center_x=20,
            center_y=20,
            polarity=1,
            start_us=60_000,
            count=12,
            pixel_count=8,
        ),
        *_train(
            center_x=20,
            center_y=20,
            polarity=1,
            start_us=76_000,
            count=12,
            pixel_count=8,
        ),
        *_train(
            center_x=20,
            center_y=20,
            polarity=0,
            start_us=112_000,
            count=8,
            pixel_count=6,
            step_us=7_000,
        ),
    ]

    result = segment_candidate_events(
        _event_array(records),
        seed_peak_us=100_000,
        seed_x=20,
        seed_y=20,
        roi_radius_px=6,
    )

    assert result.accepted
    assert result.interval is not None
    assert result.interval.on_train.first_event_us == 76_000
    assert result.interval.on_train.last_event_us == 87_000
    assert result.interval.off_train.first_event_us == 112_000
    assert result.interval.off_train.last_event_us == 161_000


def test_signed_root_poisson_deviance_preserves_direction() -> None:
    assert _signed_root_poisson_deviance(2, 5.0) < 0
    assert _signed_root_poisson_deviance(8, 5.0) > 0


def test_segment_candidate_rejects_missing_off_train() -> None:
    events = _event_array(
        _train(
            center_x=20,
            center_y=20,
            polarity=1,
            start_us=70_000,
            count=18,
            pixel_count=8,
        )
    )

    result = segment_candidate_events(
        events,
        seed_peak_us=100_000,
        seed_x=20,
        seed_y=20,
        roi_radius_px=6,
    )

    assert not result.accepted
    assert result.rejection_reason == "missing_off_train"


def test_segment_candidate_rejects_single_hot_pixel() -> None:
    records = [(20, 20, 1, 70_000 + index * 1_000) for index in range(18)] + [
        (20, 20, 0, 112_000 + index * 1_000) for index in range(10)
    ]

    result = segment_candidate_events(
        _event_array(records),
        seed_peak_us=100_000,
        seed_x=20,
        seed_y=20,
        roi_radius_px=6,
    )

    assert not result.accepted
    assert result.rejection_reason == "missing_on_train"


def test_temporal_roi_counts_full_spatial_roi_only_in_train_windows() -> None:
    seed_peak_us = 500_000
    positive = _train(
        center_x=20,
        center_y=20,
        polarity=1,
        start_us=470_250,
        count=18,
        pixel_count=8,
    )
    negative = _train(
        center_x=20,
        center_y=20,
        polarity=0,
        start_us=512_500,
        count=10,
        pixel_count=6,
    )
    records = [
        *positive,
        *negative,
        # These PSF-tail events are outside the timing core but belong to the fit maps.
        (25, 20, 1, 474_000),
        (25, 20, 0, 516_000),
        # This intervening positive background event must not enter either fit map.
        (25, 20, 1, 500_000),
    ]
    event_map: dict[tuple[np.int32, np.int32], list[tuple[np.uint64, np.int8]]] = {}
    for x, y, polarity, timestamp in records:
        event_map.setdefault((np.int32(y), np.int32(x)), []).append(
            (np.uint64(timestamp), np.int8(polarity))
        )

    generated = generate_temporally_segmented_rois(
        {(20, 20): [(seed_peak_us, 20.0, (450_000, 550_000))]},
        event_map,
        roi_radius=6,
        min_x=0,
        min_y=0,
        max_x=39,
        max_y=39,
        slice_start_us=0,
        slice_stop_us=1_000_000,
    )

    assert generated.rois.size == 1
    roi = generated.rois[0]
    assert bool(roi["temporal_segmented"])
    assert roi["total_events_roi"] == 19
    assert roi["total_neg_events_roi"] == 11
    assert roi["roi"][6, 11] == 1
    assert roi["roi_n"][6, 11] == 1
    assert roi["dt_pos_s"] == (488_000 - 470_000) * 1e-6
    assert roi["dt_neg_s"] == (522_000 - 512_000) * 1e-6
    assert generated.qc["accepted"].tolist() == [True]


def test_temporal_roi_rejects_context_at_slice_edge() -> None:
    generated = generate_temporally_segmented_rois(
        {(20, 20): [(100_000, 20.0, (80_000, 130_000))]},
        {},
        roi_radius=6,
        min_x=0,
        min_y=0,
        max_x=39,
        max_y=39,
        slice_start_us=0,
        slice_stop_us=1_000_000,
        settings=TemporalSegmentationSettings(context_pre_us=250_000),
    )

    assert generated.rois.size == 0
    assert generated.qc["rejection_reason"].tolist() == ["context_before_slice"]


def test_temporal_roi_handles_empty_event_map_inside_slice() -> None:
    generated = generate_temporally_segmented_rois(
        {(20, 20): [(500_000, 20.0, (450_000, 550_000))]},
        {},
        roi_radius=6,
        min_x=0,
        min_y=0,
        max_x=39,
        max_y=39,
        slice_start_us=0,
        slice_stop_us=1_000_000,
    )

    assert generated.rois.size == 0
    assert generated.qc["rejection_reason"].tolist() == ["empty_context"]
