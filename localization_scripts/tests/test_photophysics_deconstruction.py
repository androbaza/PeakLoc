from __future__ import annotations

import numpy as np

from localization_scripts.event_array_processing import EVENT_DTYPE
from localization_scripts.photophysics_deconstruction import (
    _candidate_context_indices,
)
from localization_scripts.temporal_segmentation import (
    TemporalSegmentationSettings,
)


def test_candidate_context_indices_match_production_seed_rectangle() -> None:
    events = np.asarray(
        [
            (10, 10, 1, 50),
            (8, 12, 1, 75),
            (12, 8, 0, 149),
            (13, 10, 1, 100),
            (10, 7, 0, 100),
            (10, 10, 1, 150),
            (10, 10, 1, 49),
        ],
        dtype=EVENT_DTYPE,
    )
    events = np.sort(events, order="t", kind="stable")
    settings = TemporalSegmentationSettings(
        context_pre_us=50,
        context_post_us=50,
    )

    indices = _candidate_context_indices(
        events,
        seed_peak_us=100,
        seed_x=10,
        seed_y=10,
        roi_radius=2,
        settings=settings,
    )

    selected = events[indices]
    assert selected["t"].tolist() == [50, 75, 149]
    assert selected["x"].tolist() == [10, 8, 12]
    assert selected["y"].tolist() == [10, 12, 8]
