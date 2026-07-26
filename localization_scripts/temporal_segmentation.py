from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from localization_scripts.python_compat import strict_zip


POSITIVE_POLARITY = 1
NEGATIVE_POLARITY = 0


@dataclass(frozen=True)
class TemporalSegmentationSettings:
    context_pre_us: int = 250_000
    context_post_us: int = 250_000
    discovery_core_radius_px: float = 4.25
    core_radius_px: float = 3.5
    bin_us: int = 1_000
    max_on_interevent_gap_us: int = 3_000
    max_off_interevent_gap_us: int = 8_000
    min_on_events: int = 12
    min_off_events: int = 8
    min_on_active_pixels: int = 6
    min_off_active_pixels: int = 5
    min_polarity_purity: float = 0.80
    max_train_duration_us: int = 150_000
    min_core_density_ratio: float = 1.5
    min_interval_deviance: float = 2.0
    max_endpoint_overlap_us: int = 20_000
    max_cycle_span_us: int = 300_000
    max_centroid_distance_px: float = 1.75
    max_on_end_after_seed_us: int = 30_000
    max_off_start_before_seed_us: int = 30_000
    ambiguity_margin_us: int = 5_000
    background_pseudocount: float = 0.5

    def validate(self) -> None:
        positive_ints = {
            "context_pre_us": self.context_pre_us,
            "context_post_us": self.context_post_us,
            "bin_us": self.bin_us,
            "max_on_interevent_gap_us": self.max_on_interevent_gap_us,
            "max_off_interevent_gap_us": self.max_off_interevent_gap_us,
            "min_on_events": self.min_on_events,
            "min_off_events": self.min_off_events,
            "min_on_active_pixels": self.min_on_active_pixels,
            "min_off_active_pixels": self.min_off_active_pixels,
            "max_train_duration_us": self.max_train_duration_us,
            "max_cycle_span_us": self.max_cycle_span_us,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        positive_floats = {
            "discovery_core_radius_px": self.discovery_core_radius_px,
            "core_radius_px": self.core_radius_px,
            "min_core_density_ratio": self.min_core_density_ratio,
            "max_centroid_distance_px": self.max_centroid_distance_px,
            "background_pseudocount": self.background_pseudocount,
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        non_negative_ints = {
            "max_endpoint_overlap_us": self.max_endpoint_overlap_us,
            "max_on_end_after_seed_us": self.max_on_end_after_seed_us,
            "max_off_start_before_seed_us": self.max_off_start_before_seed_us,
            "ambiguity_margin_us": self.ambiguity_margin_us,
        }
        for name, value in non_negative_ints.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_interval_deviance <= 0:
            raise ValueError("min_interval_deviance must be positive")
        if not 0 < self.min_polarity_purity <= 1:
            raise ValueError("min_polarity_purity must be in (0, 1]")
        if self.core_radius_px > self.discovery_core_radius_px:
            raise ValueError("core_radius_px must not exceed discovery_core_radius_px")


def temporal_settings_from_config(config: object) -> TemporalSegmentationSettings:
    """Build internal temporal settings from a flat PeakLoc configuration."""
    return TemporalSegmentationSettings(
        context_pre_us=int(getattr(config, "temporal_context_pre_us")),
        context_post_us=int(getattr(config, "temporal_context_post_us")),
        discovery_core_radius_px=float(
            getattr(config, "temporal_discovery_core_radius_px")
        ),
        core_radius_px=float(getattr(config, "temporal_core_radius_px")),
        bin_us=int(getattr(config, "temporal_bin_us")),
        max_on_interevent_gap_us=int(
            getattr(config, "temporal_max_on_interevent_gap_us")
        ),
        max_off_interevent_gap_us=int(
            getattr(config, "temporal_max_off_interevent_gap_us")
        ),
        min_on_events=int(getattr(config, "temporal_min_on_events")),
        min_off_events=int(getattr(config, "temporal_min_off_events")),
        min_on_active_pixels=int(getattr(config, "temporal_min_on_active_pixels")),
        min_off_active_pixels=int(getattr(config, "temporal_min_off_active_pixels")),
        min_polarity_purity=float(getattr(config, "temporal_min_polarity_purity")),
        max_train_duration_us=int(getattr(config, "temporal_max_train_duration_us")),
        min_core_density_ratio=float(
            getattr(config, "temporal_min_core_density_ratio")
        ),
        min_interval_deviance=float(getattr(config, "temporal_min_interval_deviance")),
        max_endpoint_overlap_us=int(
            getattr(config, "temporal_max_endpoint_overlap_us")
        ),
        max_cycle_span_us=int(getattr(config, "temporal_max_cycle_span_us")),
        max_centroid_distance_px=float(
            getattr(config, "temporal_max_centroid_distance_px")
        ),
        max_on_end_after_seed_us=int(
            getattr(config, "temporal_max_on_end_after_seed_us")
        ),
        max_off_start_before_seed_us=int(
            getattr(config, "temporal_max_off_start_before_seed_us")
        ),
        ambiguity_margin_us=int(getattr(config, "temporal_ambiguity_margin_us")),
        background_pseudocount=float(
            getattr(config, "temporal_background_pseudocount")
        ),
    )


@dataclass(frozen=True)
class TransitionTrain:
    polarity: int
    support_start_us: int
    support_stop_us: int
    first_event_us: int
    last_event_us: int
    event_count: int
    active_pixel_count: int
    centroid_x: float
    centroid_y: float
    radial_rms_px: float
    polarity_purity: float
    core_density_ratio: float
    interval_deviance: float
    event_indices: np.ndarray = field(repr=False, compare=False)

    @property
    def duration_us(self) -> int:
        return self.last_event_us - self.first_event_us


@dataclass(frozen=True)
class BlinkInterval:
    parent_seed_peak_us: int
    cycle_peak_us: int
    seed_x: float
    seed_y: float
    refined_x: float
    refined_y: float
    on_train: TransitionTrain
    off_train: TransitionTrain
    pair_score: float
    pair_cost_us: float
    centroid_distance_px: float
    endpoint_overlap_us: int
    quiet_dwell_us: int
    cycle_span_us: int
    iteration_count: int

    @property
    def onset_to_peak_us(self) -> int:
        return self.cycle_peak_us - self.on_train.first_event_us


@dataclass(frozen=True)
class SegmentationResult:
    interval: BlinkInterval | None
    rejection_reason: str
    provisional_on_trains: tuple[TransitionTrain, ...] = ()
    provisional_off_trains: tuple[TransitionTrain, ...] = ()
    refined_on_trains: tuple[TransitionTrain, ...] = ()
    refined_off_trains: tuple[TransitionTrain, ...] = ()
    selected_on_event_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp),
        repr=False,
        compare=False,
    )
    selected_off_event_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.intp),
        repr=False,
        compare=False,
    )

    @property
    def accepted(self) -> bool:
        return self.interval is not None


@dataclass(frozen=True)
class _PairCandidate:
    on_train: TransitionTrain
    off_train: TransitionTrain
    centroid_distance_px: float
    endpoint_overlap_us: int
    quiet_dwell_us: int
    cycle_span_us: int
    anchor_cost_us: int
    total_event_count: int


def segment_candidate_events(
    events: np.ndarray,
    *,
    seed_peak_us: int,
    seed_x: float,
    seed_y: float,
    roi_radius_px: int,
    settings: TemporalSegmentationSettings | None = None,
) -> SegmentationResult:
    """Select one compact ON/OFF transition pair anchored to a retained peak."""
    settings = settings or TemporalSegmentationSettings()
    settings.validate()
    if roi_radius_px <= settings.discovery_core_radius_px:
        raise ValueError(
            "roi_radius_px must exceed discovery_core_radius_px so that the "
            "background annulus is non-empty"
        )
    _validate_event_array(events)
    context_start_us = max(seed_peak_us - settings.context_pre_us, 0)
    context_stop_us = seed_peak_us + settings.context_post_us
    context_mask = (events["t"] >= context_start_us) & (events["t"] < context_stop_us)
    context_indices = np.flatnonzero(context_mask)
    context_events = events[context_indices]
    if context_events.size == 0:
        return SegmentationResult(None, "empty_context")

    available_bounds = (
        int(math.floor(seed_x)) - roi_radius_px,
        int(math.floor(seed_x)) + roi_radius_px,
        int(math.floor(seed_y)) - roi_radius_px,
        int(math.floor(seed_y)) + roi_radius_px,
    )
    provisional_on = detect_transition_trains(
        context_events,
        polarity=POSITIVE_POLARITY,
        center_x=seed_x,
        center_y=seed_y,
        core_radius_px=settings.discovery_core_radius_px,
        outer_radius_px=float(roi_radius_px),
        available_bounds=available_bounds,
        settings=settings,
    )
    provisional_off = detect_transition_trains(
        context_events,
        polarity=NEGATIVE_POLARITY,
        center_x=seed_x,
        center_y=seed_y,
        core_radius_px=settings.discovery_core_radius_px,
        outer_radius_px=float(roi_radius_px),
        available_bounds=available_bounds,
        settings=settings,
    )
    provisional_pair, provisional_reason = _select_anchored_pair(
        provisional_on,
        provisional_off,
        seed_peak_us=seed_peak_us,
        settings=settings,
    )
    if provisional_pair is None:
        return SegmentationResult(
            None,
            provisional_reason,
            provisional_on_trains=provisional_on,
            provisional_off_trains=provisional_off,
        )

    refined_x, refined_y = _pair_centroid(provisional_pair)
    refined_on = detect_transition_trains(
        context_events,
        polarity=POSITIVE_POLARITY,
        center_x=refined_x,
        center_y=refined_y,
        core_radius_px=settings.core_radius_px,
        outer_radius_px=float(roi_radius_px),
        available_bounds=available_bounds,
        settings=settings,
    )
    refined_off = detect_transition_trains(
        context_events,
        polarity=NEGATIVE_POLARITY,
        center_x=refined_x,
        center_y=refined_y,
        core_radius_px=settings.core_radius_px,
        outer_radius_px=float(roi_radius_px),
        available_bounds=available_bounds,
        settings=settings,
    )
    refined_pair, refined_reason = _select_anchored_pair(
        refined_on,
        refined_off,
        seed_peak_us=seed_peak_us,
        settings=settings,
    )
    if refined_pair is None:
        return SegmentationResult(
            None,
            f"refined_{refined_reason}",
            provisional_on_trains=provisional_on,
            provisional_off_trains=provisional_off,
            refined_on_trains=refined_on,
            refined_off_trains=refined_off,
        )

    refined_x, refined_y = _pair_centroid(refined_pair)
    pair_cost_us = _pair_cost(refined_pair, seed_peak_us)
    pair_score = 1.0 / (1.0 + pair_cost_us * 1e-3)
    interval = BlinkInterval(
        parent_seed_peak_us=seed_peak_us,
        cycle_peak_us=seed_peak_us,
        seed_x=seed_x,
        seed_y=seed_y,
        refined_x=refined_x,
        refined_y=refined_y,
        on_train=refined_pair.on_train,
        off_train=refined_pair.off_train,
        pair_score=pair_score,
        pair_cost_us=pair_cost_us,
        centroid_distance_px=refined_pair.centroid_distance_px,
        endpoint_overlap_us=refined_pair.endpoint_overlap_us,
        quiet_dwell_us=refined_pair.quiet_dwell_us,
        cycle_span_us=refined_pair.cycle_span_us,
        iteration_count=2,
    )
    return SegmentationResult(
        interval,
        "accepted",
        provisional_on_trains=provisional_on,
        provisional_off_trains=provisional_off,
        refined_on_trains=refined_on,
        refined_off_trains=refined_off,
        selected_on_event_indices=context_indices[
            refined_pair.on_train.event_indices
        ].copy(),
        selected_off_event_indices=context_indices[
            refined_pair.off_train.event_indices
        ].copy(),
    )


def detect_transition_trains(
    events: np.ndarray,
    *,
    polarity: int,
    center_x: float,
    center_y: float,
    core_radius_px: float,
    outer_radius_px: float,
    available_bounds: tuple[int, int, int, int],
    settings: TemporalSegmentationSettings,
) -> tuple[TransitionTrain, ...]:
    """Return dense polarity-specific event groups passing spatial/background gates."""
    x = events["x"].astype(np.float64)
    y = events["y"].astype(np.float64)
    radius = np.hypot(x - center_x, y - center_y)
    core_indices = np.flatnonzero(
        (events["p"] == polarity) & (radius <= core_radius_px)
    )
    if core_indices.size == 0:
        return ()
    order = np.argsort(events["t"][core_indices], kind="stable")
    core_indices = core_indices[order]
    timestamps = events["t"][core_indices].astype(np.int64)
    max_gap_us = (
        settings.max_on_interevent_gap_us
        if polarity == POSITIVE_POLARITY
        else settings.max_off_interevent_gap_us
    )
    split_points = np.flatnonzero(np.diff(timestamps) > max_gap_us)
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.intp),
            split_points.astype(np.intp) + 1,
            np.asarray([core_indices.size], dtype=np.intp),
        )
    )
    min_events = (
        settings.min_on_events
        if polarity == POSITIVE_POLARITY
        else settings.min_off_events
    )
    min_active_pixels = (
        settings.min_on_active_pixels
        if polarity == POSITIVE_POLARITY
        else settings.min_off_active_pixels
    )
    core_pixel_count = _lattice_pixel_count(
        center_x,
        center_y,
        0.0,
        core_radius_px,
        available_bounds,
    )
    annulus_pixel_count = _lattice_pixel_count(
        center_x,
        center_y,
        core_radius_px,
        outer_radius_px,
        available_bounds,
    )
    trains = []
    for start, stop in strict_zip(boundaries[:-1], boundaries[1:]):
        indices = core_indices[start:stop]
        if indices.size < min_events:
            continue
        group_timestamps = events["t"][indices].astype(np.int64)
        first_event_us = int(group_timestamps[0])
        last_event_us = int(group_timestamps[-1])
        if last_event_us - first_event_us > settings.max_train_duration_us:
            continue
        active_pixel_count = _active_pixel_count(events, indices)
        if active_pixel_count < min_active_pixels:
            continue
        support_start_us = (first_event_us // settings.bin_us) * settings.bin_us
        support_stop_us = (
            (last_event_us + 1 + settings.bin_us - 1) // settings.bin_us
        ) * settings.bin_us
        within_support = (events["t"] >= support_start_us) & (
            events["t"] < support_stop_us
        )
        within_core_support = within_support & (radius <= core_radius_px)
        all_core_count = int(np.count_nonzero(within_core_support))
        purity = indices.size / max(all_core_count, 1)
        if purity < settings.min_polarity_purity:
            continue
        annulus_count = int(
            np.count_nonzero(
                within_support
                & (events["p"] == polarity)
                & (radius > core_radius_px)
                & (radius <= outer_radius_px)
            )
        )
        density_ratio, expected_core = _background_metrics(
            int(indices.size),
            annulus_count,
            core_pixel_count,
            annulus_pixel_count,
            settings.background_pseudocount,
        )
        deviance = _signed_root_poisson_deviance(int(indices.size), expected_core)
        if density_ratio < settings.min_core_density_ratio:
            continue
        if deviance < settings.min_interval_deviance:
            continue
        group_x = x[indices]
        group_y = y[indices]
        centroid_x = float(np.mean(group_x))
        centroid_y = float(np.mean(group_y))
        radial_rms_px = float(
            np.sqrt(np.mean((group_x - centroid_x) ** 2 + (group_y - centroid_y) ** 2))
        )
        trains.append(
            TransitionTrain(
                polarity=polarity,
                support_start_us=support_start_us,
                support_stop_us=support_stop_us,
                first_event_us=first_event_us,
                last_event_us=last_event_us,
                event_count=int(indices.size),
                active_pixel_count=active_pixel_count,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                radial_rms_px=radial_rms_px,
                polarity_purity=float(purity),
                core_density_ratio=density_ratio,
                interval_deviance=deviance,
                event_indices=indices.copy(),
            )
        )
    return tuple(trains)


def _select_anchored_pair(
    on_trains: tuple[TransitionTrain, ...],
    off_trains: tuple[TransitionTrain, ...],
    *,
    seed_peak_us: int,
    settings: TemporalSegmentationSettings,
) -> tuple[_PairCandidate | None, str]:
    if not on_trains:
        return None, "missing_on_train"
    if not off_trains:
        return None, "missing_off_train"
    candidates = []
    for on_train in on_trains:
        for off_train in off_trains:
            if on_train.first_event_us > seed_peak_us:
                continue
            if off_train.last_event_us < seed_peak_us:
                continue
            if (
                on_train.last_event_us
                > seed_peak_us + settings.max_on_end_after_seed_us
            ):
                continue
            if (
                off_train.first_event_us
                < seed_peak_us - settings.max_off_start_before_seed_us
            ):
                continue
            endpoint_overlap_us = max(
                on_train.last_event_us - off_train.first_event_us,
                0,
            )
            if endpoint_overlap_us > settings.max_endpoint_overlap_us:
                continue
            cycle_span_us = off_train.last_event_us - on_train.first_event_us
            if cycle_span_us <= 0 or cycle_span_us > settings.max_cycle_span_us:
                continue
            centroid_distance_px = math.hypot(
                on_train.centroid_x - off_train.centroid_x,
                on_train.centroid_y - off_train.centroid_y,
            )
            if centroid_distance_px > settings.max_centroid_distance_px:
                continue
            quiet_dwell_us = max(
                off_train.first_event_us - on_train.last_event_us,
                0,
            )
            candidates.append(
                _PairCandidate(
                    on_train=on_train,
                    off_train=off_train,
                    centroid_distance_px=centroid_distance_px,
                    endpoint_overlap_us=endpoint_overlap_us,
                    quiet_dwell_us=quiet_dwell_us,
                    cycle_span_us=cycle_span_us,
                    anchor_cost_us=int(
                        abs(on_train.last_event_us - seed_peak_us)
                        + abs(off_train.first_event_us - seed_peak_us)
                    ),
                    total_event_count=on_train.event_count + off_train.event_count,
                )
            )
    if not candidates:
        return None, "no_anchored_pair"
    candidates.sort(
        key=lambda candidate: (
            candidate.anchor_cost_us,
            candidate.centroid_distance_px,
            -candidate.total_event_count,
            candidate.on_train.first_event_us,
            candidate.off_train.first_event_us,
        )
    )
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        distinct_pair = (
            best.on_train.first_event_us != second.on_train.first_event_us
            or best.off_train.first_event_us != second.off_train.first_event_us
        )
        cost_margin_us = second.anchor_cost_us - best.anchor_cost_us
        if distinct_pair and cost_margin_us <= settings.ambiguity_margin_us:
            return None, "ambiguous_pairing"
    return best, "accepted"


def _pair_centroid(pair: _PairCandidate) -> tuple[float, float]:
    on_weight = pair.on_train.event_count
    off_weight = pair.off_train.event_count
    total_weight = on_weight + off_weight
    return (
        (pair.on_train.centroid_x * on_weight + pair.off_train.centroid_x * off_weight)
        / total_weight,
        (pair.on_train.centroid_y * on_weight + pair.off_train.centroid_y * off_weight)
        / total_weight,
    )


def _pair_cost(pair: _PairCandidate, seed_peak_us: int) -> float:
    del seed_peak_us
    return float(pair.anchor_cost_us + pair.centroid_distance_px * 10_000.0)


def _active_pixel_count(events: np.ndarray, indices: np.ndarray) -> int:
    packed = (events["y"][indices].astype(np.uint64) << np.uint64(32)) | events["x"][
        indices
    ].astype(np.uint64)
    return int(np.unique(packed).size)


def _background_metrics(
    core_count: int,
    annulus_count: int,
    core_pixel_count: int,
    annulus_pixel_count: int,
    pseudocount: float,
) -> tuple[float, float]:
    core_density = (core_count + pseudocount) / max(core_pixel_count, 1)
    annulus_density = (annulus_count + pseudocount) / max(annulus_pixel_count, 1)
    density_ratio = core_density / annulus_density
    expected_core = (
        (annulus_count + pseudocount)
        * core_pixel_count
        / max(
            annulus_pixel_count,
            1,
        )
    )
    return float(density_ratio), float(max(expected_core, np.finfo(float).tiny))


def _signed_root_poisson_deviance(observed: int, expected: float) -> float:
    if observed <= 0:
        return -math.sqrt(2.0 * expected)
    term = observed * math.log(observed / expected) - (observed - expected)
    root_deviance = math.sqrt(2.0 * max(term, 0.0))
    return math.copysign(root_deviance, observed - expected)


def _lattice_pixel_count(
    center_x: float,
    center_y: float,
    inner_radius_px: float,
    outer_radius_px: float,
    bounds: tuple[int, int, int, int],
) -> int:
    min_x, max_x, min_y, max_y = bounds
    count = 0
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            radius = math.hypot(x - center_x, y - center_y)
            if inner_radius_px < radius <= outer_radius_px:
                count += 1
    return count


def _validate_event_array(events: np.ndarray) -> None:
    required = {"x", "y", "p", "t"}
    names = set(events.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(
            "events must contain x, y, p, and t fields; missing "
            + ", ".join(sorted(required - names))
        )
