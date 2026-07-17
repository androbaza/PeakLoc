from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any
import warnings

from csaps import CubicSmoothingSpline
import matplotlib
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import SparseEfficiencyWarning

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.event_array_processing import (
    EVENT_DTYPE,
    array_to_polarity_map,
    create_signal,
)
from localization_scripts.peak_finding import (
    find_on_off,
    jit_interpolate,
    prepare_interpolation_axis,
)
from localization_scripts.plot_style import PLOT_COLORS, PUBLICATION_DPI
from localization_scripts.temporal_segmentation import (
    SegmentationResult,
    TemporalSegmentationSettings,
    TransitionTrain,
    segment_candidate_events,
)
from scripts.raw_to_video import open_raw_reader


DEFAULT_SAMPLE_SIZE = 5
DEFAULT_RANDOM_SEED = 647
RAW_READ_WINDOW_US = 500_000
EVENT_COUNT_QUANTILES = (0.60, 0.90)
MAX_3D_EVENTS_PER_SAMPLE = 6_000

LITERATURE = (
    {
        "citation": "Lin et al., PLOS ONE 10, e0128135 (2015)",
        "url": "https://doi.org/10.1371/journal.pone.0128135",
        "relevance": (
            "AF647 ON-state lifetimes depend on excitation intensity and fall to a few "
            "milliseconds at the high intensities tested."
        ),
    },
    {
        "citation": "Diekmann et al., Nature Methods 17, 909-912 (2020)",
        "url": "https://doi.org/10.1038/s41592-020-0918-5",
        "relevance": (
            "AF647 blinking, photon yield, and localization performance change with excitation "
            "intensity and acquisition conditions."
        ),
    },
    {
        "citation": "Gallego et al., IEEE TPAMI 44, 154-180 (2022)",
        "url": "https://doi.org/10.1109/TPAMI.2020.3008413",
        "relevance": (
            "Event cameras report asynchronous threshold crossings in log intensity, not direct "
            "measurements of fluorescence-state occupancy."
        ),
    },
)


@dataclass(frozen=True)
class BlinkSample:
    sample_id: int
    localization_index: int
    roi_index: int
    t_peak_us: int
    peak_y: int
    peak_x: int
    fit_y: float
    fit_x: float
    sub_y: float
    sub_x: float
    positive_event_count: int
    negative_event_count: int
    t_first_stored_us: int
    t_last_stored_us: int
    dt_positive_s: float
    dt_negative_s: float
    uncertainty_nm: float
    nll_per_event: float
    roi_positive: np.ndarray
    roi_negative: np.ndarray

    @property
    def total_event_count(self) -> int:
        return self.positive_event_count + self.negative_event_count


@dataclass(frozen=True)
class DetectionCandidate:
    y: int
    x: int
    t_peak_us: float
    prominence_events: float


@dataclass(frozen=True)
class DetectionTrace:
    raw_time_us: np.ndarray
    raw_cumulative_polarity: np.ndarray
    interpolated_time_us: np.ndarray
    interpolated_cumulative_polarity: np.ndarray
    second_derivative: np.ndarray
    candidate: DetectionCandidate
    window_start_us: float
    window_stop_us: float
    nearby_candidates: tuple[DetectionCandidate, ...]


@dataclass(frozen=True)
class DetectionPlotData:
    raw_time_us: np.ndarray
    raw_cumulative_polarity: np.ndarray
    interpolated_time_us: np.ndarray
    interpolated_cumulative_polarity: np.ndarray
    second_derivative: np.ndarray


@dataclass(frozen=True)
class TemporalEventMasks:
    inside_final_pair_core: np.ndarray
    selected_on_train: np.ndarray
    selected_off_train: np.ndarray
    selected_positive_fit: np.ndarray
    selected_negative_fit: np.ndarray


@dataclass(frozen=True)
class AnalyzedBlink:
    sample: BlinkSample
    detection: DetectionTrace
    roi_events: np.ndarray
    roi_event_indices: np.ndarray
    count_window_start_us: float
    count_window_stop_us: float
    positive_gate_stop_us: float
    negative_gate_start_us: float
    reconstructed_positive_count: int
    reconstructed_negative_count: int
    reconstructed_t_first_us: int
    reconstructed_t_last_us: int
    regional_events: np.ndarray
    segmentation_event_indices: np.ndarray
    segmentation: SegmentationResult
    temporal_settings: TemporalSegmentationSettings

    @property
    def rise_to_peak_ms(self) -> float:
        return (self.sample.t_peak_us - self.reconstructed_t_first_us) * 1e-3

    @property
    def peak_to_last_ms(self) -> float:
        return (self.reconstructed_t_last_us - self.sample.t_peak_us) * 1e-3

    @property
    def roi_duration_ms(self) -> float:
        return (self.reconstructed_t_last_us - self.reconstructed_t_first_us) * 1e-3

    @property
    def segmented_cycle_duration_ms(self) -> float:
        interval = self.segmentation.interval
        return float("nan") if interval is None else interval.cycle_span_us * 1e-3

    @property
    def segmented_onset_to_peak_ms(self) -> float:
        interval = self.segmentation.interval
        return float("nan") if interval is None else interval.onset_to_peak_us * 1e-3

    @property
    def segmented_peak_to_offset_ms(self) -> float:
        interval = self.segmentation.interval
        if interval is None:
            return float("nan")
        return (interval.off_train.last_event_us - self.sample.t_peak_us) * 1e-3

    @property
    def duration_reduction_percent(self) -> float:
        if not self.segmentation.accepted or self.roi_duration_ms <= 0:
            return float("nan")
        return 100.0 * (1.0 - self.segmented_cycle_duration_ms / self.roi_duration_ms)


@dataclass(frozen=True)
class RunArtifacts:
    run_directory: Path
    output_directory: Path
    sample_count: int
    median_roi_duration_ms: float
    all_validations_passed: bool
    segmented_sample_count: int
    median_segmented_cycle_ms: float
    paths: tuple[Path, ...]


def discover_completed_runs(paths: list[Path]) -> list[Path]:
    """Resolve run directories from explicit runs or dataset roots."""
    runs: list[Path] = []
    for supplied_path in paths:
        path = supplied_path.expanduser()
        if (path / "run_metadata.json").is_file():
            candidates = [path]
        else:
            candidates = sorted(
                metadata.parent for metadata in path.rglob("run_metadata.json")
            )
        if not candidates:
            raise FileNotFoundError(f"No completed PeakLoc run found under {path}")
        runs.extend(candidates)
    return list(dict.fromkeys(run.resolve() for run in runs))


def _temporal_settings_from_mapping(
    config: dict[str, Any],
) -> TemporalSegmentationSettings:
    defaults = TemporalSegmentationSettings()
    values = {
        field_name: config.get(f"temporal_{field_name}", default_value)
        for field_name, default_value in asdict(defaults).items()
    }
    return TemporalSegmentationSettings(**values)


def analyze_run(
    run_directory: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    temporal_settings: TemporalSegmentationSettings | None = None,
) -> RunArtifacts:
    """Reconstruct and export a five-blink photophysics deconstruction."""
    run_directory = run_directory.resolve()
    metadata = _read_json(run_directory / "run_metadata.json")
    config = _read_json(run_directory / "config_effective.json")
    raw_path = _resolve_recording_path(str(metadata["input_file"]), run_directory)
    temporal_settings = temporal_settings or _temporal_settings_from_mapping(config)
    temporal_settings.validate()
    localizations_path = _single_artifact(
        run_directory, "localizations_prominence_fwhm_*_prominence_*.npy"
    )
    rois_path = _single_artifact(
        run_directory, "rois_prominence_fwhm_*_prominence_*.npy"
    )
    localizations = np.load(localizations_path, mmap_mode="r", allow_pickle=False)
    rois = np.load(rois_path, mmap_mode="r", allow_pickle=False)

    samples = _select_samples(
        localizations,
        rois,
        optical_pixel_size_nm=float(config["optical_pixel_size"]),
        sample_size=sample_size,
        random_seed=random_seed,
    )
    exclusions = _load_exclusions(run_directory)
    target_mask, support_mask = _load_spatial_masks(run_directory, config)
    regional_events = _read_sample_regions(
        raw_path,
        samples,
        config,
        exclusions,
        support_mask,
        temporal_settings,
    )

    analyzed = []
    for sample in samples:
        detection = _reconstruct_detection(
            regional_events[sample.sample_id],
            sample,
            config,
            target_mask,
            support_mask,
        )
        analyzed.append(
            _analyze_roi_events(
                regional_events[sample.sample_id],
                sample,
                detection,
                polarity_time_gate_us=float(config["polarity_time_gate_us"]),
                temporal_settings=temporal_settings,
            )
        )

    output_directory = run_directory / str(config.get("qc_output_dirname", "qc"))
    output_directory = output_directory / "photophysics_deconstruction"
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = _write_outputs(
        analyzed,
        rois,
        output_directory,
        run_directory,
        raw_path,
        config,
        random_seed,
    )
    durations = np.asarray([blink.roi_duration_ms for blink in analyzed])
    validations = [_validation_row(blink) for blink in analyzed]
    segmented_durations = np.asarray(
        [blink.segmented_cycle_duration_ms for blink in analyzed], dtype=np.float64
    )
    finite_segmented = segmented_durations[np.isfinite(segmented_durations)]
    return RunArtifacts(
        run_directory=run_directory,
        output_directory=output_directory,
        sample_count=len(analyzed),
        median_roi_duration_ms=float(np.median(durations)),
        all_validations_passed=all(row["all_checks_pass"] for row in validations),
        segmented_sample_count=int(finite_segmented.size),
        median_segmented_cycle_ms=(
            float(np.median(finite_segmented))
            if finite_segmented.size
            else float("nan")
        ),
        paths=tuple(paths),
    )


def _select_samples(
    localizations: np.ndarray,
    rois: np.ndarray,
    *,
    optical_pixel_size_nm: float,
    sample_size: int,
    random_seed: int,
) -> list[BlinkSample]:
    names = set(localizations.dtype.names or ())
    required = {
        "E_total",
        "E_total_n",
        "fit_success",
        "t_peak",
        "x",
        "y",
        "sub_x",
        "sub_y",
        "sigma_x",
        "sigma_y",
        "cov_xy",
        "nll_per_event",
    }
    if not required.issubset(names):
        missing = sorted(required - names)
        raise ValueError(f"Localization artifact lacks required fields: {missing}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    total_events = np.asarray(localizations["E_total"], dtype=np.float64) + np.asarray(
        localizations["E_total_n"], dtype=np.float64
    )
    lower, upper = np.quantile(total_events, EVENT_COUNT_QUANTILES)
    finite = (
        np.isfinite(localizations["x"])
        & np.isfinite(localizations["y"])
        & np.isfinite(localizations["nll_per_event"])
    )
    eligible = np.flatnonzero(
        localizations["fit_success"]
        & finite
        & (total_events >= lower)
        & (total_events <= upper)
    )
    if eligible.size < sample_size:
        raise ValueError(
            f"Only {eligible.size} event-rich accepted fits are eligible for {sample_size} samples"
        )

    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(eligible, size=sample_size, replace=False)
    selected_indices = selected_indices[
        np.argsort(localizations["t_peak"][selected_indices], kind="stable")
    ]
    samples = []
    roi_radius = int(rois["roi"].shape[-1] // 2)
    for sample_id, localization_index in enumerate(selected_indices, start=1):
        localization = localizations[int(localization_index)]
        expected_x0 = int(round(float(localization["x"] - localization["sub_x"])))
        expected_y0 = int(round(float(localization["y"] - localization["sub_y"])))
        expected_peak_x = expected_x0 + roi_radius
        expected_peak_y = expected_y0 + roi_radius
        roi_index = _match_roi_index(
            rois,
            int(round(float(localization["t_peak"]))),
            expected_peak_y,
            expected_peak_x,
        )
        roi = rois[roi_index]
        uncertainty_px = _localization_uncertainty_px(localization)
        samples.append(
            BlinkSample(
                sample_id=sample_id,
                localization_index=int(localization_index),
                roi_index=roi_index,
                t_peak_us=int(roi["t_peak"]),
                peak_y=int(roi["peak"][0]),
                peak_x=int(roi["peak"][1]),
                fit_y=float(localization["y"]),
                fit_x=float(localization["x"]),
                sub_y=float(localization["sub_y"]),
                sub_x=float(localization["sub_x"]),
                positive_event_count=int(roi["total_events_roi"]),
                negative_event_count=int(roi["total_neg_events_roi"]),
                t_first_stored_us=int(roi["t_1st"]),
                t_last_stored_us=int(roi["t_last"]),
                dt_positive_s=float(roi["dt_pos_s"]),
                dt_negative_s=float(roi["dt_neg_s"]),
                uncertainty_nm=uncertainty_px * optical_pixel_size_nm,
                nll_per_event=float(localization["nll_per_event"]),
                roi_positive=np.asarray(roi["roi"], dtype=np.uint32).copy(),
                roi_negative=np.asarray(roi["roi_n"], dtype=np.uint32).copy(),
            )
        )
    return samples


def _match_roi_index(
    rois: np.ndarray,
    t_peak_us: int,
    expected_peak_y: int,
    expected_peak_x: int,
) -> int:
    time_matches = np.flatnonzero(rois["t_peak"] == t_peak_us)
    if time_matches.size == 0:
        raise ValueError(
            f"No ROI matches accepted localization peak time {t_peak_us} us"
        )
    peak_coordinates = rois["peak"][time_matches].astype(np.int64)
    squared_distance = (peak_coordinates[:, 0] - expected_peak_y) ** 2 + (
        peak_coordinates[:, 1] - expected_peak_x
    ) ** 2
    best = int(np.argmin(squared_distance))
    if int(squared_distance[best]) != 0:
        raise ValueError(
            "ROI timestamp matched, but its detection center did not match the localization "
            f"at {t_peak_us} us"
        )
    return int(time_matches[best])


def _localization_uncertainty_px(localization: np.void) -> float:
    variance_x = float(localization["sigma_x"]) ** 2
    variance_y = float(localization["sigma_y"]) ** 2
    covariance = float(localization["cov_xy"])
    largest_eigenvalue = 0.5 * (
        variance_x
        + variance_y
        + math.sqrt((variance_x - variance_y) ** 2 + 4.0 * covariance**2)
    )
    return math.sqrt(max(largest_eigenvalue, 0.0))


def _read_sample_regions(
    raw_path: Path,
    samples: list[BlinkSample],
    config: dict[str, Any],
    exclusions: list[tuple[int, int]],
    support_mask: np.ndarray,
    temporal_settings: TemporalSegmentationSettings,
) -> dict[int, np.ndarray]:
    roi_radius = int(config["roi_radius"])
    neighborhood_radius = max(
        int(config["peak_neighbors"]) + int(config["convolution_roi_radius"]),
        roi_radius * 2,
    )
    polarity_gate_us = float(config["polarity_time_gate_us"])
    sample_intervals = []
    for sample in samples:
        legacy_start_us = (
            sample.t_peak_us + polarity_gate_us - sample.dt_positive_s * 1e6
        )
        legacy_stop_us = (
            sample.t_peak_us - polarity_gate_us + sample.dt_negative_s * 1e6
        )
        start_us = int(
            math.floor(
                min(
                    legacy_start_us,
                    sample.t_peak_us - temporal_settings.context_pre_us,
                )
            )
        )
        stop_us = int(
            math.ceil(
                max(
                    legacy_stop_us,
                    sample.t_peak_us + temporal_settings.context_post_us,
                )
            )
        )
        sample_intervals.append((start_us, stop_us, sample))

    pieces: dict[int, list[np.ndarray]] = {sample.sample_id: [] for sample in samples}
    reader: Any = open_raw_reader(raw_path, int(config["max_raw_events"]))
    for start_us, stop_us, sample in sorted(
        sample_intervals, key=lambda item: (item[0], item[1], item[2].sample_id)
    ):
        reader.seek_time(max(start_us, 0))
        while reader.current_time < stop_us and not reader.is_done():
            read_duration = min(RAW_READ_WINDOW_US, stop_us - int(reader.current_time))
            if read_duration <= 0:
                break
            raw_chunk = reader.load_delta_t(read_duration)
            if raw_chunk.size == 0:
                continue
            chunk = np.asarray(raw_chunk, dtype=EVENT_DTYPE)
            chunk = chunk[chunk["t"] < stop_us]
            chunk = _exclude_intervals(chunk, exclusions)
            if chunk.size == 0:
                continue
            supported = support_mask[chunk["y"], chunk["x"]]
            chunk = chunk[supported]
            within = (
                (chunk["y"] >= sample.peak_y - neighborhood_radius)
                & (chunk["y"] <= sample.peak_y + neighborhood_radius)
                & (chunk["x"] >= sample.peak_x - neighborhood_radius)
                & (chunk["x"] <= sample.peak_x + neighborhood_radius)
            )
            if np.any(within):
                pieces[sample.sample_id].append(chunk[within].copy())
    return {
        sample_id: (
            np.concatenate(sample_pieces)
            if sample_pieces
            else np.empty(0, dtype=EVENT_DTYPE)
        )
        for sample_id, sample_pieces in pieces.items()
    }


def _exclude_intervals(
    events: np.ndarray, exclusions: list[tuple[int, int]]
) -> np.ndarray:
    if events.size == 0 or not exclusions:
        return events
    keep = np.ones(events.size, dtype=np.bool_)
    for start_us, stop_us in exclusions:
        keep &= (events["t"] < start_us) | (events["t"] >= stop_us)
    return events[keep]


def _reconstruct_detection(
    regional_events: np.ndarray,
    sample: BlinkSample,
    config: dict[str, Any],
    target_mask: np.ndarray,
    support_mask: np.ndarray,
) -> DetectionTrace:
    convolution_radius = int(config["convolution_roi_radius"])
    neighbor_radius = int(config["peak_neighbors"])
    y_min = max(sample.peak_y - neighbor_radius - convolution_radius, 0)
    y_max = min(
        sample.peak_y + neighbor_radius + convolution_radius,
        support_mask.shape[0] - 1,
    )
    x_min = max(sample.peak_x - neighbor_radius - convolution_radius, 0)
    x_max = min(
        sample.peak_x + neighbor_radius + convolution_radius,
        support_mask.shape[1] - 1,
    )
    support_y, support_x = np.nonzero(
        support_mask[y_min : y_max + 1, x_min : x_max + 1]
    )
    support_coordinates = np.column_stack(
        (support_y + y_min, support_x + x_min)
    ).astype(np.int32)
    target_y_min = max(sample.peak_y - neighbor_radius, 0)
    target_y_max = min(sample.peak_y + neighbor_radius, target_mask.shape[0] - 1)
    target_x_min = max(sample.peak_x - neighbor_radius, 0)
    target_x_max = min(sample.peak_x + neighbor_radius, target_mask.shape[1] - 1)
    target_y, target_x = np.nonzero(
        target_mask[target_y_min : target_y_max + 1, target_x_min : target_x_max + 1]
    )
    target_coordinates = np.column_stack(
        (target_y + target_y_min, target_x + target_x_min)
    ).astype(np.int32)
    if not np.any(
        (target_coordinates[:, 0] == sample.peak_y)
        & (target_coordinates[:, 1] == sample.peak_x)
    ):
        raise ValueError(
            "The recorded detection center is absent from the saved target mask"
        )

    polarity_map, max_events_per_polarity = array_to_polarity_map(
        regional_events, support_coordinates
    )
    maximum_signal_length = int(
        max_events_per_polarity * 2 * (convolution_radius * 2 + 1) ** 2
    )
    signal_times, signal_cumulative, signal_coordinates = create_signal(
        polarity_map,
        target_coordinates,
        maximum_signal_length,
        convolution_roi_radius=convolution_radius,
    )
    interpolation_coefficient = int(config["interpolation_coefficient"])
    prominence_threshold = float(config["prominence"])
    candidates: list[DetectionCandidate] = []
    winner_data: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None
    winner_peak_indices: np.ndarray | None = None
    winner_prominences: np.ndarray | None = None

    for raw_times, cumulative, coordinate in zip(
        signal_times, signal_cumulative, signal_coordinates, strict=True
    ):
        prepared_times, prepared_cumulative = prepare_interpolation_axis(
            raw_times, cumulative
        )
        if prepared_times is None or len(prepared_times) < int(
            config["peak_min_event_count"]
        ):
            continue
        interpolation_count = max(len(prepared_times) * interpolation_coefficient, 2)
        interpolated_times = np.linspace(
            prepared_times[0],
            prepared_times[-1],
            num=interpolation_count,
            dtype=np.float64,
        )
        interpolated_cumulative = jit_interpolate(
            prepared_times, prepared_cumulative, interpolated_times
        )
        peak_indices, properties = find_peaks(
            interpolated_cumulative, prominence=prominence_threshold
        )
        for peak_index, prominence in zip(
            peak_indices, properties["prominences"], strict=True
        ):
            candidates.append(
                DetectionCandidate(
                    y=int(coordinate[0]),
                    x=int(coordinate[1]),
                    t_peak_us=float(interpolated_times[peak_index]),
                    prominence_events=float(prominence),
                )
            )
        if int(coordinate[0]) == sample.peak_y and int(coordinate[1]) == sample.peak_x:
            winner_data = (
                np.asarray(prepared_times, dtype=np.float64),
                np.asarray(prepared_cumulative, dtype=np.float64),
                interpolated_times,
                np.asarray(interpolated_cumulative, dtype=np.float64),
                np.asarray(coordinate),
            )
            winner_peak_indices = np.asarray(peak_indices, dtype=np.intp)
            winner_prominences = np.asarray(properties["prominences"], dtype=np.float64)

    if (
        winner_data is None
        or winner_peak_indices is None
        or winner_peak_indices.size == 0
    ):
        raise ValueError(f"Could not reconstruct peak at {sample.t_peak_us} us")
    (
        prepared_times,
        prepared_cumulative,
        interpolated_times,
        interpolated_cumulative,
        _,
    ) = winner_data
    distances = np.abs(interpolated_times[winner_peak_indices] - sample.t_peak_us)
    nearest_index = int(np.argmin(distances))
    winner_peak_index = int(winner_peak_indices[nearest_index])
    if winner_prominences is None:
        raise RuntimeError("Peak prominences were not reconstructed")
    winner_candidate = DetectionCandidate(
        y=sample.peak_y,
        x=sample.peak_x,
        t_peak_us=float(interpolated_times[winner_peak_index]),
        prominence_events=float(winner_prominences[nearest_index]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        spline = CubicSmoothingSpline(
            prepared_times,
            prepared_cumulative,
            smooth=float(config["spline_smooth"]),
            normalizedsmooth=True,
        ).spline
    second_derivative = np.asarray(spline.derivative()(interpolated_times))
    on_off = find_on_off(
        np.asarray([winner_peak_index], dtype=np.int64),
        second_derivative,
        interpolated_times,
        interpolated_cumulative,
    )[0]
    threshold_us = float(config["peak_time_threshold"])
    nearby = tuple(
        candidate
        for candidate in candidates
        if abs(candidate.t_peak_us - winner_candidate.t_peak_us) <= threshold_us
    )
    return DetectionTrace(
        raw_time_us=prepared_times,
        raw_cumulative_polarity=prepared_cumulative,
        interpolated_time_us=interpolated_times,
        interpolated_cumulative_polarity=interpolated_cumulative,
        second_derivative=second_derivative,
        candidate=winner_candidate,
        window_start_us=float(on_off[0]),
        window_stop_us=float(on_off[1]),
        nearby_candidates=nearby,
    )


def _candidate_context_indices(
    events: np.ndarray,
    *,
    seed_peak_us: int,
    seed_x: int,
    seed_y: int,
    roi_radius: int,
    settings: TemporalSegmentationSettings,
) -> np.ndarray:
    context_start_us = max(seed_peak_us - settings.context_pre_us, 0)
    context_stop_us = seed_peak_us + settings.context_post_us
    within_context = (events["t"] >= context_start_us) & (events["t"] < context_stop_us)
    within_seed_roi = (
        (events["x"] >= seed_x - roi_radius)
        & (events["x"] <= seed_x + roi_radius)
        & (events["y"] >= seed_y - roi_radius)
        & (events["y"] <= seed_y + roi_radius)
    )
    return np.flatnonzero(within_context & within_seed_roi)


def _analyze_roi_events(
    regional_events: np.ndarray,
    sample: BlinkSample,
    detection: DetectionTrace,
    *,
    polarity_time_gate_us: float,
    temporal_settings: TemporalSegmentationSettings,
) -> AnalyzedBlink:
    roi_radius = sample.roi_positive.shape[0] // 2
    regional_events = np.sort(regional_events, order="t", kind="stable")
    positive_gate_stop_us = sample.t_peak_us + polarity_time_gate_us
    negative_gate_start_us = sample.t_peak_us - polarity_time_gate_us
    # ROI records persist the two exposure durations exactly. Because ROI generation always
    # enforces lower <= peak - gate and upper >= peak + gate, these identities recover the
    # original count window even when replayed spline boundaries differ by sub-grid timing.
    count_window_start_us = positive_gate_stop_us - sample.dt_positive_s * 1e6
    count_window_stop_us = negative_gate_start_us + sample.dt_negative_s * 1e6
    within_roi = (
        (regional_events["y"] >= sample.peak_y - roi_radius)
        & (regional_events["y"] <= sample.peak_y + roi_radius)
        & (regional_events["x"] >= sample.peak_x - roi_radius)
        & (regional_events["x"] <= sample.peak_x + roi_radius)
        & (regional_events["t"] >= count_window_start_us)
        & (regional_events["t"] < count_window_stop_us)
    )
    roi_event_indices = np.flatnonzero(within_roi)
    roi_events = regional_events[roi_event_indices]
    positive = (roi_events["p"] == 1) & (roi_events["t"] < positive_gate_stop_us)
    negative = (roi_events["p"] == 0) & (roi_events["t"] > negative_gate_start_us)
    segmentation_event_indices = _candidate_context_indices(
        regional_events,
        seed_peak_us=sample.t_peak_us,
        seed_x=sample.peak_x,
        seed_y=sample.peak_y,
        roi_radius=roi_radius,
        settings=temporal_settings,
    )
    segmentation_events = regional_events[segmentation_event_indices]
    segmentation = segment_candidate_events(
        segmentation_events,
        seed_peak_us=sample.t_peak_us,
        seed_x=float(sample.peak_x),
        seed_y=float(sample.peak_y),
        roi_radius_px=roi_radius,
        settings=temporal_settings,
    )
    if roi_events.size == 0:
        reconstructed_first = 0
        reconstructed_last = 0
    else:
        reconstructed_first = int(roi_events["t"][0])
        reconstructed_last = int(roi_events["t"][-1])
    return AnalyzedBlink(
        sample=sample,
        detection=detection,
        roi_events=roi_events,
        roi_event_indices=roi_event_indices,
        count_window_start_us=count_window_start_us,
        count_window_stop_us=count_window_stop_us,
        positive_gate_stop_us=positive_gate_stop_us,
        negative_gate_start_us=negative_gate_start_us,
        reconstructed_positive_count=int(np.count_nonzero(positive)),
        reconstructed_negative_count=int(np.count_nonzero(negative)),
        reconstructed_t_first_us=reconstructed_first,
        reconstructed_t_last_us=reconstructed_last,
        regional_events=regional_events,
        segmentation_event_indices=segmentation_event_indices,
        segmentation=segmentation,
        temporal_settings=temporal_settings,
    )


def _write_outputs(
    analyzed: list[AnalyzedBlink],
    rois: np.ndarray,
    output_directory: Path,
    run_directory: Path,
    raw_path: Path,
    config: dict[str, Any],
    random_seed: int,
) -> list[Path]:
    nearby_seed_rows = _nearby_seed_interval_rows(analyzed, rois, config)
    paths = [
        _write_sample_selection(analyzed, output_directory),
        _write_raw_events(analyzed, output_directory),
        _write_segmentation_context_events(analyzed, output_directory),
        _write_detection_candidates(analyzed, output_directory),
        _write_detection_trace(analyzed, output_directory),
        _write_timing_summary(analyzed, output_directory),
        _write_transition_trains(analyzed, output_directory),
        _write_blink_intervals(analyzed, output_directory),
        _write_temporal_activity_bins(analyzed, output_directory),
        _write_segmented_fit_events(analyzed, output_directory),
        _write_nearby_seed_intervals(nearby_seed_rows, output_directory),
    ]
    for blink in analyzed:
        paths.extend(_plot_sample_deconstruction(blink, output_directory))
    paths.extend(_plot_overview(analyzed, output_directory))
    paths.extend(_plot_event_clouds_3d(analyzed, output_directory))
    paths.extend(
        _plot_nearby_seed_intervals(analyzed, nearby_seed_rows, output_directory)
    )
    summary_path = _write_summary(
        analyzed,
        nearby_seed_rows,
        output_directory,
        run_directory,
        raw_path,
        config,
        random_seed,
    )
    report_path = _write_report(analyzed, output_directory, run_directory, summary_path)
    paths.extend([summary_path, report_path])
    return paths


def _write_sample_selection(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "sample_selection.csv"
    fieldnames = [
        "sample_id",
        "localization_index",
        "roi_index",
        "t_peak_us",
        "peak_y_px",
        "peak_x_px",
        "fit_y_px",
        "fit_x_px",
        "positive_fit_events",
        "negative_fit_events",
        "total_fit_events",
        "localization_uncertainty_nm",
        "nll_per_event",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            sample = blink.sample
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "localization_index": sample.localization_index,
                    "roi_index": sample.roi_index,
                    "t_peak_us": sample.t_peak_us,
                    "peak_y_px": sample.peak_y,
                    "peak_x_px": sample.peak_x,
                    "fit_y_px": sample.fit_y,
                    "fit_x_px": sample.fit_x,
                    "positive_fit_events": sample.positive_event_count,
                    "negative_fit_events": sample.negative_event_count,
                    "total_fit_events": sample.total_event_count,
                    "localization_uncertainty_nm": sample.uncertainty_nm,
                    "nll_per_event": sample.nll_per_event,
                }
            )
    return path


def _write_raw_events(analyzed: list[AnalyzedBlink], output_directory: Path) -> Path:
    path = output_directory / "raw_roi_events.csv"
    fieldnames = [
        "sample_id",
        "event_index",
        "t_us",
        "t_relative_to_peak_us",
        "x_px",
        "y_px",
        "polarity",
        "used_in_positive_fit_count",
        "used_in_negative_fit_count",
        "inside_final_pair_core",
        "selected_on_core_train",
        "selected_off_core_train",
        "also_used_in_segmented_positive_fit",
        "also_used_in_segmented_negative_fit",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            masks = _temporal_event_masks(
                blink, blink.roi_events, blink.roi_event_indices
            )
            for event_index, event in enumerate(blink.roi_events):
                timestamp = int(event["t"])
                polarity = int(event["p"])
                writer.writerow(
                    {
                        "sample_id": blink.sample.sample_id,
                        "event_index": event_index,
                        "t_us": timestamp,
                        "t_relative_to_peak_us": timestamp - blink.sample.t_peak_us,
                        "x_px": int(event["x"]),
                        "y_px": int(event["y"]),
                        "polarity": polarity,
                        "used_in_positive_fit_count": int(
                            polarity == 1 and timestamp < blink.positive_gate_stop_us
                        ),
                        "used_in_negative_fit_count": int(
                            polarity == 0 and timestamp > blink.negative_gate_start_us
                        ),
                        "inside_final_pair_core": int(
                            masks.inside_final_pair_core[event_index]
                        ),
                        "selected_on_core_train": int(
                            masks.selected_on_train[event_index]
                        ),
                        "selected_off_core_train": int(
                            masks.selected_off_train[event_index]
                        ),
                        "also_used_in_segmented_positive_fit": int(
                            masks.selected_positive_fit[event_index]
                        ),
                        "also_used_in_segmented_negative_fit": int(
                            masks.selected_negative_fit[event_index]
                        ),
                    }
                )
    return path


def _legacy_fit_mask(blink: AnalyzedBlink, events: np.ndarray) -> np.ndarray:
    roi_radius = blink.sample.roi_positive.shape[0] // 2
    inside_legacy_roi = (
        (events["x"] >= blink.sample.peak_x - roi_radius)
        & (events["x"] <= blink.sample.peak_x + roi_radius)
        & (events["y"] >= blink.sample.peak_y - roi_radius)
        & (events["y"] <= blink.sample.peak_y + roi_radius)
        & (events["t"] >= blink.count_window_start_us)
        & (events["t"] < blink.count_window_stop_us)
    )
    positive_lobe = (events["p"] == 1) & (events["t"] < blink.positive_gate_stop_us)
    negative_lobe = (events["p"] == 0) & (events["t"] > blink.negative_gate_start_us)
    return inside_legacy_roi & (positive_lobe | negative_lobe)


def _write_segmentation_context_events(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "segmentation_context_events.csv"
    fieldnames = [
        "sample_id",
        "context_event_index",
        "regional_event_index",
        "t_us",
        "t_relative_to_peak_us",
        "x_px",
        "y_px",
        "roi_pixel_index_row_major",
        "polarity",
        "used_in_legacy_fit",
        "inside_final_pair_core",
        "selected_on_core_train",
        "selected_off_core_train",
        "used_in_segmented_positive_fit",
        "used_in_segmented_negative_fit",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            regional_indices = blink.segmentation_event_indices
            events = blink.regional_events[regional_indices]
            masks = _temporal_event_masks(blink, events, regional_indices)
            legacy_fit = _legacy_fit_mask(blink, events)
            roi_radius = blink.sample.roi_positive.shape[0] // 2
            roi_width = roi_radius * 2 + 1
            for context_index, (regional_index, event) in enumerate(
                zip(regional_indices, events, strict=True)
            ):
                pixel_index = (
                    (int(event["y"]) - (blink.sample.peak_y - roi_radius)) * roi_width
                    + int(event["x"])
                    - (blink.sample.peak_x - roi_radius)
                )
                writer.writerow(
                    {
                        "sample_id": blink.sample.sample_id,
                        "context_event_index": context_index,
                        "regional_event_index": int(regional_index),
                        "t_us": int(event["t"]),
                        "t_relative_to_peak_us": int(event["t"])
                        - blink.sample.t_peak_us,
                        "x_px": int(event["x"]),
                        "y_px": int(event["y"]),
                        "roi_pixel_index_row_major": pixel_index,
                        "polarity": int(event["p"]),
                        "used_in_legacy_fit": int(legacy_fit[context_index]),
                        "inside_final_pair_core": int(
                            masks.inside_final_pair_core[context_index]
                        ),
                        "selected_on_core_train": int(
                            masks.selected_on_train[context_index]
                        ),
                        "selected_off_core_train": int(
                            masks.selected_off_train[context_index]
                        ),
                        "used_in_segmented_positive_fit": int(
                            masks.selected_positive_fit[context_index]
                        ),
                        "used_in_segmented_negative_fit": int(
                            masks.selected_negative_fit[context_index]
                        ),
                    }
                )
    return path


def _temporal_event_masks(
    blink: AnalyzedBlink,
    events: np.ndarray,
    regional_indices: np.ndarray,
) -> TemporalEventMasks:
    if events.size != regional_indices.size:
        raise ValueError("events and regional_indices must have matching lengths")
    empty = np.zeros(events.size, dtype=np.bool_)
    interval = blink.segmentation.interval
    if interval is None:
        return TemporalEventMasks(
            empty,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )

    radius = np.hypot(
        events["x"].astype(np.float64) - interval.refined_x,
        events["y"].astype(np.float64) - interval.refined_y,
    )
    inside_final_pair_core = radius <= blink.temporal_settings.core_radius_px
    selected_on_regional_indices = blink.segmentation_event_indices[
        blink.segmentation.selected_on_event_indices
    ]
    selected_off_regional_indices = blink.segmentation_event_indices[
        blink.segmentation.selected_off_event_indices
    ]
    selected_on_train = np.isin(regional_indices, selected_on_regional_indices)
    selected_off_train = np.isin(regional_indices, selected_off_regional_indices)
    center_x = int(round(interval.refined_x))
    center_y = int(round(interval.refined_y))
    roi_radius = blink.sample.roi_positive.shape[0] // 2
    inside_fit_roi = (
        (events["x"] >= center_x - roi_radius)
        & (events["x"] <= center_x + roi_radius)
        & (events["y"] >= center_y - roi_radius)
        & (events["y"] <= center_y + roi_radius)
    )
    selected_positive_fit = (
        inside_fit_roi
        & (events["p"] == 1)
        & (events["t"] >= interval.on_train.support_start_us)
        & (events["t"] < interval.on_train.support_stop_us)
    )
    selected_negative_fit = (
        inside_fit_roi
        & (events["p"] == 0)
        & (events["t"] >= interval.off_train.support_start_us)
        & (events["t"] < interval.off_train.support_stop_us)
    )
    return TemporalEventMasks(
        inside_final_pair_core,
        selected_on_train,
        selected_off_train,
        selected_positive_fit,
        selected_negative_fit,
    )


def _write_detection_candidates(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "detection_candidates.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "sample_id",
                "candidate_y_px",
                "candidate_x_px",
                "candidate_t_peak_us",
                "candidate_prominence_events",
                "is_retained_peak",
            ]
        )
        for blink in analyzed:
            retained = blink.detection.candidate
            for candidate in blink.detection.nearby_candidates:
                is_retained = (
                    candidate.y == retained.y
                    and candidate.x == retained.x
                    and candidate.t_peak_us == retained.t_peak_us
                )
                writer.writerow(
                    [
                        blink.sample.sample_id,
                        candidate.y,
                        candidate.x,
                        candidate.t_peak_us,
                        candidate.prominence_events,
                        int(is_retained),
                    ]
                )
    return path


def _detection_plot_data(blink: AnalyzedBlink) -> DetectionPlotData:
    trace = blink.detection
    start = blink.count_window_start_us
    stop = blink.count_window_stop_us
    padding = max((stop - start) * 0.08, 2_000.0)
    within_raw = (trace.raw_time_us >= start - padding) & (
        trace.raw_time_us <= stop + padding
    )
    within_interpolated = (trace.interpolated_time_us >= start - padding) & (
        trace.interpolated_time_us <= stop + padding
    )
    raw_time_us = trace.raw_time_us[within_raw]
    if raw_time_us.size == 0:
        raise ValueError("Detection trace has no raw samples in the plotted interval")
    reference = float(trace.raw_cumulative_polarity[within_raw][0])
    return DetectionPlotData(
        raw_time_us=raw_time_us,
        raw_cumulative_polarity=trace.raw_cumulative_polarity[within_raw] - reference,
        interpolated_time_us=trace.interpolated_time_us[within_interpolated],
        interpolated_cumulative_polarity=(
            trace.interpolated_cumulative_polarity[within_interpolated] - reference
        ),
        second_derivative=trace.second_derivative[within_interpolated],
    )


def _write_detection_trace(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "detection_trace.csv"
    fieldnames = [
        "sample_id",
        "series",
        "point_index",
        "t_us",
        "t_relative_to_peak_ms",
        "cumulative_polarity_relative",
        "spline_second_derivative",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            data = _detection_plot_data(blink)
            for point_index, (timestamp, cumulative) in enumerate(
                zip(
                    data.raw_time_us,
                    data.raw_cumulative_polarity,
                    strict=True,
                )
            ):
                writer.writerow(
                    {
                        "sample_id": blink.sample.sample_id,
                        "series": "raw",
                        "point_index": point_index,
                        "t_us": float(timestamp),
                        "t_relative_to_peak_ms": (
                            float(timestamp) - blink.sample.t_peak_us
                        )
                        * 1e-3,
                        "cumulative_polarity_relative": float(cumulative),
                        "spline_second_derivative": None,
                    }
                )
            for point_index, (timestamp, cumulative, second_derivative) in enumerate(
                zip(
                    data.interpolated_time_us,
                    data.interpolated_cumulative_polarity,
                    data.second_derivative,
                    strict=True,
                )
            ):
                writer.writerow(
                    {
                        "sample_id": blink.sample.sample_id,
                        "series": "interpolated",
                        "point_index": point_index,
                        "t_us": float(timestamp),
                        "t_relative_to_peak_ms": (
                            float(timestamp) - blink.sample.t_peak_us
                        )
                        * 1e-3,
                        "cumulative_polarity_relative": float(cumulative),
                        "spline_second_derivative": float(second_derivative),
                    }
                )
    return path


def _nearby_seed_interval_rows(
    analyzed: list[AnalyzedBlink],
    rois: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not analyzed:
        return []
    settings = analyzed[0].temporal_settings
    roi_radius = int(config["roi_radius"])
    nearby_seed_radius_px = settings.discovery_core_radius_px
    peak_times = rois["t_peak"]
    rows = []
    for blink in analyzed:
        sample = blink.sample
        read_start_us = int(
            math.floor(
                min(
                    blink.count_window_start_us,
                    sample.t_peak_us - settings.context_pre_us,
                )
            )
        )
        read_stop_us = int(
            math.ceil(
                max(
                    blink.count_window_stop_us,
                    sample.t_peak_us + settings.context_post_us,
                )
            )
        )
        first_seed_us = read_start_us + settings.context_pre_us
        last_seed_us = read_stop_us - settings.context_post_us
        start_index = int(np.searchsorted(peak_times, first_seed_us, side="left"))
        stop_index = int(np.searchsorted(peak_times, last_seed_us, side="right"))
        for roi_index in range(start_index, stop_index):
            seed = rois[roi_index]
            seed_y = int(seed["peak"][0])
            seed_x = int(seed["peak"][1])
            delta_y = seed_y - sample.peak_y
            delta_x = seed_x - sample.peak_x
            if math.hypot(delta_x, delta_y) > nearby_seed_radius_px:
                continue
            seed_peak_us = int(seed["t_peak"])
            candidate_event_indices = _candidate_context_indices(
                blink.regional_events,
                seed_peak_us=seed_peak_us,
                seed_x=seed_x,
                seed_y=seed_y,
                roi_radius=roi_radius,
                settings=settings,
            )
            candidate_events = blink.regional_events[candidate_event_indices]
            segmentation = segment_candidate_events(
                candidate_events,
                seed_peak_us=seed_peak_us,
                seed_x=float(seed_x),
                seed_y=float(seed_y),
                roi_radius_px=roi_radius,
                settings=settings,
            )
            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "seed_roi_index": roi_index,
                "is_displayed_seed": int(roi_index == sample.roi_index),
                "seed_peak_us": seed_peak_us,
                "seed_relative_to_sample_peak_ms": (seed_peak_us - sample.t_peak_us)
                * 1e-3,
                "seed_y_px": seed_y,
                "seed_x_px": seed_x,
                "seed_distance_from_sample_px": math.hypot(delta_x, delta_y),
                "context_event_count": int(candidate_events.size),
                "accepted": int(segmentation.accepted),
                "rejection_reason": segmentation.rejection_reason,
                "on_first_event_us": None,
                "on_last_event_us": None,
                "off_first_event_us": None,
                "off_last_event_us": None,
                "on_first_relative_to_sample_peak_ms": None,
                "off_last_relative_to_sample_peak_ms": None,
                "cycle_span_ms": None,
                "quiet_dwell_ms": None,
                "endpoint_overlap_ms": None,
                "on_core_event_count": None,
                "off_core_event_count": None,
            }
            interval = segmentation.interval
            if interval is not None:
                row.update(
                    {
                        "on_first_event_us": interval.on_train.first_event_us,
                        "on_last_event_us": interval.on_train.last_event_us,
                        "off_first_event_us": interval.off_train.first_event_us,
                        "off_last_event_us": interval.off_train.last_event_us,
                        "on_first_relative_to_sample_peak_ms": (
                            interval.on_train.first_event_us - sample.t_peak_us
                        )
                        * 1e-3,
                        "off_last_relative_to_sample_peak_ms": (
                            interval.off_train.last_event_us - sample.t_peak_us
                        )
                        * 1e-3,
                        "cycle_span_ms": interval.cycle_span_us * 1e-3,
                        "quiet_dwell_ms": interval.quiet_dwell_us * 1e-3,
                        "endpoint_overlap_ms": interval.endpoint_overlap_us * 1e-3,
                        "on_core_event_count": interval.on_train.event_count,
                        "off_core_event_count": interval.off_train.event_count,
                    }
                )
            rows.append(row)
    return rows


def _write_nearby_seed_intervals(
    rows: list[dict[str, Any]], output_directory: Path
) -> Path:
    path = output_directory / "nearby_seed_intervals.csv"
    if not rows:
        path.write_text("sample_id\n", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_transition_trains(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "transition_trains.csv"
    fieldnames = [
        "sample_id",
        "stage",
        "polarity",
        "train_index",
        "is_selected",
        "support_start_us",
        "support_stop_us",
        "first_event_us",
        "last_event_us",
        "first_relative_to_peak_ms",
        "last_relative_to_peak_ms",
        "duration_ms",
        "event_count",
        "active_pixel_count",
        "centroid_x_px",
        "centroid_y_px",
        "radial_rms_px",
        "polarity_purity",
        "core_density_ratio",
        "signed_root_poisson_deviance",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            stages = (
                (
                    "provisional",
                    blink.segmentation.provisional_on_trains,
                    blink.segmentation.provisional_off_trains,
                ),
                (
                    "refined",
                    blink.segmentation.refined_on_trains,
                    blink.segmentation.refined_off_trains,
                ),
            )
            for stage, on_trains, off_trains in stages:
                for polarity, trains in ((1, on_trains), (0, off_trains)):
                    for train_index, train in enumerate(trains, start=1):
                        writer.writerow(
                            _transition_train_row(
                                blink,
                                train,
                                stage=stage,
                                train_index=train_index,
                            )
                        )
    return path


def _transition_train_row(
    blink: AnalyzedBlink,
    train: TransitionTrain,
    *,
    stage: str,
    train_index: int,
) -> dict[str, Any]:
    interval = blink.segmentation.interval
    selected = interval is not None and (
        train is interval.on_train or train is interval.off_train
    )
    peak = blink.sample.t_peak_us
    return {
        "sample_id": blink.sample.sample_id,
        "stage": stage,
        "polarity": train.polarity,
        "train_index": train_index,
        "is_selected": int(selected),
        "support_start_us": train.support_start_us,
        "support_stop_us": train.support_stop_us,
        "first_event_us": train.first_event_us,
        "last_event_us": train.last_event_us,
        "first_relative_to_peak_ms": (train.first_event_us - peak) * 1e-3,
        "last_relative_to_peak_ms": (train.last_event_us - peak) * 1e-3,
        "duration_ms": train.duration_us * 1e-3,
        "event_count": train.event_count,
        "active_pixel_count": train.active_pixel_count,
        "centroid_x_px": train.centroid_x,
        "centroid_y_px": train.centroid_y,
        "radial_rms_px": train.radial_rms_px,
        "polarity_purity": train.polarity_purity,
        "core_density_ratio": train.core_density_ratio,
        "signed_root_poisson_deviance": train.interval_deviance,
    }


def _write_blink_intervals(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "blink_intervals.csv"
    rows = [_blink_interval_row(blink) for blink in analyzed]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _blink_interval_row(blink: AnalyzedBlink) -> dict[str, Any]:
    interval = blink.segmentation.interval
    regional_indices = np.arange(blink.regional_events.size, dtype=np.intp)
    masks = _temporal_event_masks(blink, blink.regional_events, regional_indices)
    selected_fit = masks.selected_positive_fit | masks.selected_negative_fit
    legacy_fit = _legacy_fit_mask(blink, blink.regional_events)
    selected_positive = int(np.count_nonzero(masks.selected_positive_fit))
    selected_negative = int(np.count_nonzero(masks.selected_negative_fit))
    selected_total = selected_positive + selected_negative
    legacy_total = blink.sample.total_event_count
    shared_total = int(np.count_nonzero(selected_fit & legacy_fit))
    newly_included_total = int(np.count_nonzero(selected_fit & ~legacy_fit))
    excluded_legacy_total = int(np.count_nonzero(legacy_fit & ~selected_fit))
    row = {
        "sample_id": blink.sample.sample_id,
        "accepted": int(blink.segmentation.accepted),
        "rejection_reason": blink.segmentation.rejection_reason,
        "legacy_first_event_us": blink.reconstructed_t_first_us,
        "legacy_last_event_us": blink.reconstructed_t_last_us,
        "legacy_cycle_duration_ms": blink.roi_duration_ms,
        "on_support_start_us": None,
        "on_support_stop_us": None,
        "on_first_event_us": None,
        "on_last_event_us": None,
        "off_support_start_us": None,
        "off_support_stop_us": None,
        "off_first_event_us": None,
        "off_last_event_us": None,
        "segmented_onset_to_peak_ms": None,
        "segmented_peak_to_offset_ms": None,
        "segmented_cycle_duration_ms": None,
        "duration_reduction_percent": None,
        "quiet_dwell_ms": None,
        "endpoint_overlap_ms": None,
        "pair_centroid_distance_px": None,
        "pair_score": None,
        "refined_x_px": None,
        "refined_y_px": None,
        "legacy_fit_event_count": legacy_total,
        "segmented_positive_fit_event_count": selected_positive,
        "segmented_negative_fit_event_count": selected_negative,
        "segmented_fit_event_count": selected_total,
        "segmented_to_legacy_fit_event_count_percent": (
            100.0 * selected_total / legacy_total if legacy_total else None
        ),
        "legacy_fit_events_retained_count": shared_total,
        "legacy_fit_events_retained_percent": (
            100.0 * shared_total / legacy_total if legacy_total else None
        ),
        "newly_included_segmented_fit_event_count": newly_included_total,
        "excluded_legacy_fit_event_count": excluded_legacy_total,
        "post_segmentation_refit_performed": 0,
    }
    if interval is None:
        return row
    row.update(
        {
            "on_support_start_us": interval.on_train.support_start_us,
            "on_support_stop_us": interval.on_train.support_stop_us,
            "on_first_event_us": interval.on_train.first_event_us,
            "on_last_event_us": interval.on_train.last_event_us,
            "off_support_start_us": interval.off_train.support_start_us,
            "off_support_stop_us": interval.off_train.support_stop_us,
            "off_first_event_us": interval.off_train.first_event_us,
            "off_last_event_us": interval.off_train.last_event_us,
            "segmented_onset_to_peak_ms": blink.segmented_onset_to_peak_ms,
            "segmented_peak_to_offset_ms": blink.segmented_peak_to_offset_ms,
            "segmented_cycle_duration_ms": blink.segmented_cycle_duration_ms,
            "duration_reduction_percent": blink.duration_reduction_percent,
            "quiet_dwell_ms": interval.quiet_dwell_us * 1e-3,
            "endpoint_overlap_ms": interval.endpoint_overlap_us * 1e-3,
            "pair_centroid_distance_px": interval.centroid_distance_px,
            "pair_score": interval.pair_score,
            "refined_x_px": interval.refined_x,
            "refined_y_px": interval.refined_y,
        }
    )
    return row


def _write_segmented_fit_events(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "segmented_fit_events.csv"
    fieldnames = [
        "sample_id",
        "event_index",
        "t_us",
        "t_relative_to_peak_us",
        "x_px",
        "y_px",
        "polarity",
        "fit_lobe",
        "used_in_legacy_fit",
        "inside_final_pair_core",
        "selected_on_core_train",
        "selected_off_core_train",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for blink in analyzed:
            regional_indices = np.arange(blink.regional_events.size, dtype=np.intp)
            masks = _temporal_event_masks(
                blink, blink.regional_events, regional_indices
            )
            legacy_fit = _legacy_fit_mask(blink, blink.regional_events)
            selected = masks.selected_positive_fit | masks.selected_negative_fit
            for event_index in np.flatnonzero(selected):
                event = blink.regional_events[event_index]
                writer.writerow(
                    {
                        "sample_id": blink.sample.sample_id,
                        "event_index": int(event_index),
                        "t_us": int(event["t"]),
                        "t_relative_to_peak_us": int(event["t"])
                        - blink.sample.t_peak_us,
                        "x_px": int(event["x"]),
                        "y_px": int(event["y"]),
                        "polarity": int(event["p"]),
                        "fit_lobe": (
                            "positive"
                            if masks.selected_positive_fit[event_index]
                            else "negative"
                        ),
                        "used_in_legacy_fit": int(legacy_fit[event_index]),
                        "inside_final_pair_core": int(
                            masks.inside_final_pair_core[event_index]
                        ),
                        "selected_on_core_train": int(
                            masks.selected_on_train[event_index]
                        ),
                        "selected_off_core_train": int(
                            masks.selected_off_train[event_index]
                        ),
                    }
                )
    return path


def _write_temporal_activity_bins(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "temporal_activity_bins.csv"
    rows = [row for blink in analyzed for row in _temporal_activity_rows(blink)]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _temporal_activity_rows(blink: AnalyzedBlink) -> list[dict[str, Any]]:
    settings = blink.temporal_settings
    peak = blink.sample.t_peak_us
    interval = blink.segmentation.interval
    center_x = (
        interval.refined_x if interval is not None else float(blink.sample.peak_x)
    )
    center_y = (
        interval.refined_y if interval is not None else float(blink.sample.peak_y)
    )
    events = blink.regional_events
    radius = np.hypot(
        events["x"].astype(np.float64) - center_x,
        events["y"].astype(np.float64) - center_y,
    )
    start_us = peak - settings.context_pre_us
    stop_us = peak + settings.context_post_us
    edges = np.arange(start_us, stop_us + settings.bin_us, settings.bin_us)
    rows = []
    for bin_index, (bin_start, bin_stop) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        within = (events["t"] >= bin_start) & (events["t"] < bin_stop)
        core = within & (radius <= settings.core_radius_px)
        annulus = (
            within
            & (radius > settings.core_radius_px)
            & (radius <= blink.sample.roi_positive.shape[0] // 2)
        )
        rows.append(
            {
                "sample_id": blink.sample.sample_id,
                "bin_index": bin_index,
                "bin_start_us": int(bin_start),
                "bin_stop_us": int(bin_stop),
                "bin_center_relative_to_peak_ms": ((bin_start + bin_stop) * 0.5 - peak)
                * 1e-3,
                "positive_core_events": int(
                    np.count_nonzero(core & (events["p"] == 1))
                ),
                "negative_core_events": int(
                    np.count_nonzero(core & (events["p"] == 0))
                ),
                "positive_annulus_events": int(
                    np.count_nonzero(annulus & (events["p"] == 1))
                ),
                "negative_annulus_events": int(
                    np.count_nonzero(annulus & (events["p"] == 0))
                ),
                "inside_selected_on_support": int(
                    interval is not None
                    and bin_start < interval.on_train.support_stop_us
                    and bin_stop > interval.on_train.support_start_us
                ),
                "inside_selected_off_support": int(
                    interval is not None
                    and bin_start < interval.off_train.support_stop_us
                    and bin_stop > interval.off_train.support_start_us
                ),
            }
        )
    return rows


def _write_timing_summary(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> Path:
    path = output_directory / "timing_summary.csv"
    rows = [_timing_row(blink) | _validation_row(blink) for blink in analyzed]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _timing_row(blink: AnalyzedBlink) -> dict[str, Any]:
    row = {
        "sample_id": blink.sample.sample_id,
        "detector_window_start_us": blink.detection.window_start_us,
        "count_window_start_us": blink.count_window_start_us,
        "roi_first_event_us": blink.reconstructed_t_first_us,
        "detected_peak_us": blink.sample.t_peak_us,
        "roi_last_event_us": blink.reconstructed_t_last_us,
        "count_window_stop_us": blink.count_window_stop_us,
        "detector_window_stop_us": blink.detection.window_stop_us,
        "rise_first_to_peak_ms": blink.rise_to_peak_ms,
        "roi_first_to_last_duration_ms": blink.roi_duration_ms,
        "decay_peak_to_last_ms": blink.peak_to_last_ms,
        "positive_gate_stop_us": blink.positive_gate_stop_us,
        "negative_gate_start_us": blink.negative_gate_start_us,
        "raw_roi_event_count": int(blink.roi_events.size),
        "reconstructed_positive_fit_count": blink.reconstructed_positive_count,
        "reconstructed_negative_fit_count": blink.reconstructed_negative_count,
        "segmentation_accepted": int(blink.segmentation.accepted),
        "segmentation_reason": blink.segmentation.rejection_reason,
        "segmented_onset_to_peak_ms": None,
        "segmented_peak_to_offset_ms": None,
        "segmented_cycle_duration_ms": None,
        "duration_reduction_percent": None,
    }
    if blink.segmentation.accepted:
        row.update(
            {
                "segmented_onset_to_peak_ms": blink.segmented_onset_to_peak_ms,
                "segmented_peak_to_offset_ms": blink.segmented_peak_to_offset_ms,
                "segmented_cycle_duration_ms": blink.segmented_cycle_duration_ms,
                "duration_reduction_percent": blink.duration_reduction_percent,
            }
        )
    return row


def _validation_row(blink: AnalyzedBlink) -> dict[str, bool | int]:
    first_matches = blink.reconstructed_t_first_us == blink.sample.t_first_stored_us
    last_matches = blink.reconstructed_t_last_us == blink.sample.t_last_stored_us
    positive_matches = (
        blink.reconstructed_positive_count == blink.sample.positive_event_count
    )
    negative_matches = (
        blink.reconstructed_negative_count == blink.sample.negative_event_count
    )
    return {
        "stored_first_event_us": blink.sample.t_first_stored_us,
        "stored_last_event_us": blink.sample.t_last_stored_us,
        "stored_positive_fit_count": blink.sample.positive_event_count,
        "stored_negative_fit_count": blink.sample.negative_event_count,
        "first_timestamp_matches": first_matches,
        "last_timestamp_matches": last_matches,
        "positive_count_matches": positive_matches,
        "negative_count_matches": negative_matches,
        "all_checks_pass": (
            first_matches and last_matches and positive_matches and negative_matches
        ),
    }


def _plot_sample_deconstruction(
    blink: AnalyzedBlink, output_directory: Path
) -> list[Path]:
    sample = blink.sample
    with plt.rc_context(_publication_style()):
        figure = plt.figure(figsize=(7.2, 6.3), constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
        positive_axis = figure.add_subplot(grid[0, 0])
        negative_axis = figure.add_subplot(grid[0, 1])
        detection_axis = figure.add_subplot(grid[0, 2])
        raster_axis = figure.add_subplot(grid[1, :2])
        timing_axis = figure.add_subplot(grid[1, 2])

        _plot_roi_map(
            positive_axis,
            sample.roi_positive,
            sample,
            cmap="Blues",
            label="Positive events",
        )
        _plot_roi_map(
            negative_axis,
            sample.roi_negative,
            sample,
            cmap="Oranges",
            label="Negative events",
        )
        _plot_detection_trace(detection_axis, blink)
        _plot_event_raster(raster_axis, blink)
        _plot_timing_calculation(timing_axis, blink)
        _panel_labels(
            [positive_axis, negative_axis, detection_axis, raster_axis, timing_axis]
        )
        figure.suptitle(
            f"Blink {sample.sample_id}: retained peak at {sample.t_peak_us / 1e6:.6f} s",
            fontsize=9,
        )
        stem = output_directory / f"blink_{sample.sample_id:02d}_deconstruction"
        paths = _save_figure(figure, stem)
        plt.close(figure)
    return paths


def _plot_roi_map(
    axis: Any,
    image: np.ndarray,
    sample: BlinkSample,
    *,
    cmap: str,
    label: str,
) -> None:
    artist = axis.imshow(image, origin="upper", cmap=cmap, interpolation="nearest")
    axis.scatter(
        sample.sub_x,
        sample.sub_y,
        marker="+",
        s=35,
        linewidths=1.2,
        color=PLOT_COLORS["black"],
        label="Fitted center",
    )
    radius = image.shape[0] // 2
    axis.scatter(
        radius,
        radius,
        marker="x",
        s=25,
        linewidths=1.0,
        color=PLOT_COLORS["vermillion"],
        label="Detection center",
    )
    axis.set(xlabel="ROI x (px)", ylabel="ROI y (px)", title=label)
    axis.set_xticks([0, radius, image.shape[1] - 1])
    axis.set_yticks([0, radius, image.shape[0] - 1])
    plt.colorbar(artist, ax=axis, label="Events per pixel", fraction=0.046, pad=0.03)


def _plot_detection_trace(axis: Any, blink: AnalyzedBlink) -> None:
    trace = blink.detection
    data = _detection_plot_data(blink)
    axis.step(
        (data.raw_time_us - blink.sample.t_peak_us) * 1e-3,
        data.raw_cumulative_polarity,
        where="post",
        color=PLOT_COLORS["gray"],
        linewidth=0.7,
        label="Raw cumulative polarity",
    )
    axis.plot(
        (data.interpolated_time_us - blink.sample.t_peak_us) * 1e-3,
        data.interpolated_cumulative_polarity,
        color=PLOT_COLORS["blue"],
        linewidth=1.0,
        label="Interpolated signal",
    )
    axis.axvspan(
        (trace.window_start_us - blink.sample.t_peak_us) * 1e-3,
        (trace.window_stop_us - blink.sample.t_peak_us) * 1e-3,
        color=PLOT_COLORS["sky_blue"],
        alpha=0.18,
        linewidth=0,
        label="Spline-derived window",
    )
    replay_peak_ms = (trace.candidate.t_peak_us - blink.sample.t_peak_us) * 1e-3
    axis.axvline(
        0.0,
        color=PLOT_COLORS["vermillion"],
        linewidth=1.0,
        label="Stored retained peak",
    )
    if not np.isclose(replay_peak_ms, 0.0):
        axis.axvline(
            replay_peak_ms,
            color=PLOT_COLORS["blue"],
            linewidth=0.8,
            linestyle=":",
            label="RAW replay candidate",
        )
    axis.set(
        xlabel="Time from retained peak (ms)",
        ylabel="Cumulative polarity (events)",
        title="Detection replay",
    )
    axis.legend(
        fontsize=5.5,
        loc="best",
        title=f"Prominence = {trace.candidate.prominence_events:.1f} events",
        title_fontsize=5.5,
    )


def _plot_event_raster(axis: Any, blink: AnalyzedBlink) -> None:
    radius = blink.sample.roi_positive.shape[0] // 2
    peak = blink.sample.t_peak_us
    regional_indices = blink.segmentation_event_indices
    events = blink.regional_events[regional_indices]
    masks = _temporal_event_masks(blink, events, regional_indices)
    roi_width = radius * 2 + 1
    pixel_index = (
        (events["y"].astype(np.int64) - (blink.sample.peak_y - radius)) * roi_width
        + events["x"].astype(np.int64)
        - (blink.sample.peak_x - radius)
    )
    time_ms = (events["t"].astype(np.float64) - blink.sample.t_peak_us) * 1e-3
    for polarity, label, color in (
        (1, "Positive", PLOT_COLORS["blue"]),
        (0, "Negative", PLOT_COLORS["vermillion"]),
    ):
        selected = events["p"] == polarity
        axis.scatter(
            time_ms[selected],
            pixel_index[selected],
            s=3,
            alpha=0.60,
            linewidths=0,
            color=color,
            rasterized=True,
            label=label,
        )
    interval = blink.segmentation.interval
    if interval is not None:
        axis.axvspan(
            (interval.on_train.support_start_us - peak) * 1e-3,
            (interval.on_train.support_stop_us - peak) * 1e-3,
            color=PLOT_COLORS["sky_blue"],
            alpha=0.18,
            linewidth=0,
            label="Selected ON window",
        )
        axis.axvspan(
            (interval.off_train.support_start_us - peak) * 1e-3,
            (interval.off_train.support_stop_us - peak) * 1e-3,
            color=PLOT_COLORS["orange"],
            alpha=0.18,
            linewidth=0,
            label="Selected OFF window",
        )
    selected_train = masks.selected_on_train | masks.selected_off_train
    axis.scatter(
        time_ms[selected_train],
        pixel_index[selected_train],
        s=11,
        facecolors="none",
        edgecolors=PLOT_COLORS["black"],
        linewidths=0.5,
        label="Selected core-train events",
    )
    axis.axvline(0.0, color=PLOT_COLORS["black"], linewidth=0.8)
    axis.axvline(
        (blink.negative_gate_start_us - blink.sample.t_peak_us) * 1e-3,
        color=PLOT_COLORS["gray"],
        linewidth=0.7,
        linestyle="--",
    )
    axis.axvline(
        (blink.positive_gate_stop_us - blink.sample.t_peak_us) * 1e-3,
        color=PLOT_COLORS["gray"],
        linewidth=0.7,
        linestyle="--",
    )
    axis.set(
        xlabel="Raw event time from retained peak (ms)",
        ylabel="ROI pixel index (row-major)",
        title=f"All {events.size:,} raw events in the +/-250 ms ROI context",
        ylim=(-2, roi_width**2 + 1),
    )
    axis.legend(ncol=2, loc="upper right")


def _plot_timing_calculation(axis: Any, blink: AnalyzedBlink) -> None:
    peak = blink.sample.t_peak_us
    points = {
        "first": (blink.reconstructed_t_first_us - peak) * 1e-3,
        "peak": 0.0,
        "last": (blink.reconstructed_t_last_us - peak) * 1e-3,
        "window_start": (blink.count_window_start_us - peak) * 1e-3,
        "window_stop": (blink.count_window_stop_us - peak) * 1e-3,
    }
    axis.hlines(2.0, points["window_start"], points["window_stop"], color="#BBBBBB")
    axis.hlines(1.0, points["first"], points["peak"], color=PLOT_COLORS["blue"], lw=2)
    axis.hlines(
        1.0,
        points["peak"],
        points["last"],
        color=PLOT_COLORS["vermillion"],
        lw=2,
    )
    axis.scatter(
        [points["first"], points["peak"], points["last"]],
        [1.0, 1.0, 1.0],
        s=14,
        color=PLOT_COLORS["black"],
        zorder=3,
    )
    axis.axvspan(-5.0, 5.0, color=PLOT_COLORS["yellow"], alpha=0.22, linewidth=0)
    axis.annotate(
        "first",
        (points["first"], 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=5.5,
    )
    axis.annotate(
        "peak",
        (points["peak"], 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=5.5,
    )
    axis.annotate(
        "last",
        (points["last"], 1.0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=5.5,
    )
    interval = blink.segmentation.interval
    if interval is not None:
        on_start = (interval.on_train.first_event_us - peak) * 1e-3
        on_last = (interval.on_train.last_event_us - peak) * 1e-3
        off_start = (interval.off_train.first_event_us - peak) * 1e-3
        off_last = (interval.off_train.last_event_us - peak) * 1e-3
        axis.hlines(0.55, on_start, on_last, color=PLOT_COLORS["blue"], lw=2.5)
        axis.hlines(0.15, off_start, off_last, color=PLOT_COLORS["vermillion"], lw=2.5)
        axis.scatter(
            [on_start, on_last, off_start, off_last],
            [0.55, 0.55, 0.15, 0.15],
            s=10,
            color=PLOT_COLORS["black"],
            zorder=3,
        )
        timing_text = (
            f"legacy span = {blink.roi_duration_ms:.1f} ms\n"
            f"segmented cycle = {blink.segmented_cycle_duration_ms:.1f} ms\n"
            f"reduction = {blink.duration_reduction_percent:.1f}%"
        )
    else:
        timing_text = f"rejected: {blink.segmentation.rejection_reason}"
    axis.text(
        0.03,
        0.04,
        timing_text,
        transform=axis.transAxes,
        va="bottom",
        fontsize=5.8,
    )
    axis.set(
        xlabel="Time from retained peak (ms)",
        yticks=[0.15, 0.55, 1.0, 2.0],
        yticklabels=["OFF train", "ON train", "legacy events", "legacy window"],
        title="Timing comparison",
        ylim=(-0.35, 2.4),
    )


def _plot_overview(analyzed: list[AnalyzedBlink], output_directory: Path) -> list[Path]:
    sample_labels = [f"Blink {blink.sample.sample_id}" for blink in analyzed]
    y = np.arange(len(analyzed))
    rise = np.asarray([blink.rise_to_peak_ms for blink in analyzed])
    decay = np.asarray([blink.peak_to_last_ms for blink in analyzed])
    positive_counts = np.asarray(
        [blink.reconstructed_positive_count for blink in analyzed]
    )
    negative_counts = np.asarray(
        [blink.reconstructed_negative_count for blink in analyzed]
    )
    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        axes[0].barh(
            y,
            rise,
            left=-rise,
            color=PLOT_COLORS["blue"],
            label="First event to peak",
        )
        axes[0].barh(
            y,
            decay,
            color=PLOT_COLORS["vermillion"],
            label="Peak to last event",
        )
        axes[0].axvline(0, color=PLOT_COLORS["black"], linewidth=0.8)
        axes[0].set(
            xlabel="Time relative to retained peak (ms)",
            yticks=y,
            yticklabels=sample_labels,
            title="ROI first-to-last timing proxy",
        )
        axes[0].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, borderaxespad=0
        )
        axes[0].invert_yaxis()
        axes[1].barh(
            y,
            positive_counts,
            color=PLOT_COLORS["blue"],
            label="Positive fit events",
        )
        axes[1].barh(
            y,
            negative_counts,
            left=positive_counts,
            color=PLOT_COLORS["vermillion"],
            label="Negative fit events",
        )
        axes[1].set(
            xlabel="Events contributing to the joint fit",
            yticks=y,
            yticklabels=sample_labels,
            title="Event-rich accepted sample",
        )
        axes[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, borderaxespad=0
        )
        axes[1].invert_yaxis()
        _panel_labels(list(axes))
        paths = _save_figure(figure, output_directory / "sample_overview")
        plt.close(figure)
    return paths


def _plot_event_clouds_3d(
    analyzed: list[AnalyzedBlink], output_directory: Path
) -> list[Path]:
    with plt.rc_context(_publication_style()):
        figure = plt.figure(figsize=(7.2, 5.0))
        figure.subplots_adjust(
            left=0.01,
            right=0.98,
            bottom=0.03,
            top=0.94,
            wspace=0.05,
            hspace=0.20,
        )
        for index, blink in enumerate(analyzed, start=1):
            axis = figure.add_subplot(2, 3, index, projection="3d")
            events = blink.roi_events
            if events.size > MAX_3D_EVENTS_PER_SAMPLE:
                indices = np.linspace(
                    0, events.size - 1, MAX_3D_EVENTS_PER_SAMPLE, dtype=np.intp
                )
                events = events[indices]
            relative_time_ms = (
                events["t"].astype(np.float64) - blink.sample.t_peak_us
            ) * 1e-3
            for polarity, color in (
                (1, PLOT_COLORS["blue"]),
                (0, PLOT_COLORS["vermillion"]),
            ):
                selected = events["p"] == polarity
                axis.scatter(
                    events["x"][selected],
                    events["y"][selected],
                    relative_time_ms[selected],
                    s=1.5,
                    alpha=0.35,
                    color=color,
                    linewidths=0,
                    rasterized=True,
                )
            axis.set_title(f"Blink {blink.sample.sample_id}", pad=0)
            axis.set_xlabel("x (px)", labelpad=-7)
            axis.set_ylabel("y (px)", labelpad=-8)
            axis.set_zlabel("t - peak (ms)", labelpad=-7)
            axis.tick_params(labelsize=4.5, pad=-2)
            axis.set_box_aspect((1.0, 1.0, 0.85))
            axis.view_init(elev=22, azim=-58)
        legend_axis = figure.add_subplot(2, 3, 6)
        legend_axis.axis("off")
        legend_axis.scatter([], [], color=PLOT_COLORS["blue"], label="Positive event")
        legend_axis.scatter(
            [], [], color=PLOT_COLORS["vermillion"], label="Negative event"
        )
        legend_axis.legend(loc="center", frameon=False)
        paths = _save_figure(figure, output_directory / "roi_event_clouds_3d")
        plt.close(figure)
    return paths


def _plot_nearby_seed_intervals(
    analyzed: list[AnalyzedBlink],
    rows: list[dict[str, Any]],
    output_directory: Path,
) -> list[Path]:
    with plt.rc_context(_publication_style()):
        figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
        accepted_label_available = True
        seed_label_available = True
        rejected_label_available = True
        for sample_index, blink in enumerate(analyzed):
            peak = blink.sample.t_peak_us
            legacy_start_ms = (blink.reconstructed_t_first_us - peak) * 1e-3
            legacy_stop_ms = (blink.reconstructed_t_last_us - peak) * 1e-3
            axis.hlines(
                sample_index,
                legacy_start_ms,
                legacy_stop_ms,
                color=PLOT_COLORS["gray"],
                linewidth=7,
                alpha=0.28,
                label=(
                    "Legacy first-to-last span" if sample_index == 0 else "_nolegend_"
                ),
            )
            sample_rows = [
                row for row in rows if row["sample_id"] == blink.sample.sample_id
            ]
            offsets = (
                np.linspace(-0.22, 0.22, len(sample_rows))
                if len(sample_rows) > 1
                else np.zeros(1)
            )
            for row, offset in zip(sample_rows, offsets, strict=True):
                y = sample_index + float(offset)
                seed_ms = float(row["seed_relative_to_sample_peak_ms"])
                if row["accepted"]:
                    axis.hlines(
                        y,
                        float(row["on_first_relative_to_sample_peak_ms"]),
                        float(row["off_last_relative_to_sample_peak_ms"]),
                        color=PLOT_COLORS["green"],
                        linewidth=2.2,
                        label="Accepted seed interval"
                        if accepted_label_available
                        else "_nolegend_",
                    )
                    accepted_label_available = False
                    axis.scatter(
                        seed_ms,
                        y,
                        s=12,
                        color=PLOT_COLORS["black"],
                        zorder=3,
                        label="Retained seed" if seed_label_available else None,
                    )
                    seed_label_available = False
                else:
                    axis.scatter(
                        seed_ms,
                        y,
                        marker="x",
                        s=22,
                        linewidths=0.9,
                        color=PLOT_COLORS["vermillion"],
                        label="Rejected seed" if rejected_label_available else None,
                    )
                    rejected_label_available = False
        axis.axvline(0, color=PLOT_COLORS["black"], linewidth=0.7, linestyle=":")
        axis.set(
            xlabel="Time from displayed retained peak (ms)",
            yticks=np.arange(len(analyzed)),
            yticklabels=[f"Blink {blink.sample.sample_id}" for blink in analyzed],
            title="Nearby retained seeds: per-seed transition intervals",
        )
        axis.invert_yaxis()
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)
        paths = _save_figure(figure, output_directory / "nearby_seed_intervals")
        plt.close(figure)
    return paths


def _write_summary(
    analyzed: list[AnalyzedBlink],
    nearby_seed_rows: list[dict[str, Any]],
    output_directory: Path,
    run_directory: Path,
    raw_path: Path,
    config: dict[str, Any],
    random_seed: int,
) -> Path:
    timing_rows = [_timing_row(blink) for blink in analyzed]
    validation_rows = [_validation_row(blink) for blink in analyzed]
    durations = np.asarray([blink.roi_duration_ms for blink in analyzed])
    interval_rows = [_blink_interval_row(blink) for blink in analyzed]
    accepted_rows = [row for row in interval_rows if row["accepted"]]
    segmented_durations = [row["segmented_cycle_duration_ms"] for row in accepted_rows]
    segmented_event_ratios = [
        row["segmented_to_legacy_fit_event_count_percent"] for row in accepted_rows
    ]
    legacy_overlap_percentages = [
        row["legacy_fit_events_retained_percent"] for row in accepted_rows
    ]
    payload = {
        "run_directory": str(run_directory),
        "raw_recording": str(raw_path),
        "sample_size": len(analyzed),
        "random_seed": random_seed,
        "selection": {
            "population": "accepted localizations",
            "event_count_quantile_band": list(EVENT_COUNT_QUANTILES),
            "sampling": "uniform without replacement, then sorted by peak time",
        },
        "pipeline_parameters": {
            key: config[key]
            for key in (
                "convolution_roi_radius",
                "interpolation_coefficient",
                "peak_neighbors",
                "peak_time_threshold",
                "polarity_time_gate_us",
                "prominence",
                "roi_radius",
                "slice_duration",
                "spline_smooth",
            )
        },
        "roi_duration_ms": {
            "minimum": float(np.min(durations)),
            "median": float(np.median(durations)),
            "maximum": float(np.max(durations)),
        },
        "temporal_segmentation": {
            "settings": asdict(analyzed[0].temporal_settings),
            "accepted_sample_count": len(accepted_rows),
            "intervals": interval_rows,
            "nearby_seed_intervals": nearby_seed_rows,
            "median_segmented_cycle_ms": (
                float(np.median(segmented_durations)) if segmented_durations else None
            ),
            "median_segmented_to_legacy_fit_event_count_percent": (
                float(np.median(segmented_event_ratios))
                if segmented_event_ratios
                else None
            ),
            "median_legacy_fit_events_retained_percent": (
                float(np.median(legacy_overlap_percentages))
                if legacy_overlap_percentages
                else None
            ),
            "post_segmentation_refit_performed": False,
            "detection_replay_scope": (
                "Bounded per-sample union of the legacy count window and +/-250 ms "
                "segmentation context, not the original complete slice; legacy event-count "
                "and first/last validation remains exact."
            ),
        },
        "timing": timing_rows,
        "validation": validation_rows,
        "all_validations_passed": all(
            row["all_checks_pass"] for row in validation_rows
        ),
        "interpretation": (
            "The ROI first-to-last event duration is a detection-window statistic. It spans all "
            "retained events in the fitted spatial ROI between the spline-derived bounds (expanded "
            "when needed to include the polarity gates). It is not a direct AF647 ON-state lifetime."
        ),
        "literature": LITERATURE,
    }
    path = output_directory / "analysis_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_report(
    analyzed: list[AnalyzedBlink],
    output_directory: Path,
    run_directory: Path,
    summary_path: Path,
) -> Path:
    durations = np.asarray([blink.roi_duration_ms for blink in analyzed])
    rise = np.asarray([blink.rise_to_peak_ms for blink in analyzed])
    decay = np.asarray([blink.peak_to_last_ms for blink in analyzed])
    validation_rows = [_validation_row(blink) for blink in analyzed]
    interval_rows = [_blink_interval_row(blink) for blink in analyzed]
    accepted_rows = [row for row in interval_rows if row["accepted"]]
    if accepted_rows:
        segmented_durations = [
            row["segmented_cycle_duration_ms"] for row in accepted_rows
        ]
        segmented_event_ratios = [
            row["segmented_to_legacy_fit_event_count_percent"] for row in accepted_rows
        ]
        legacy_overlap_percentages = [
            row["legacy_fit_events_retained_percent"] for row in accepted_rows
        ]
        comparison = (
            f"The train segmenter accepted **{len(accepted_rows)}/{len(analyzed)}** samples; "
            f"their median cycle span is **{np.median(segmented_durations):.1f} ms**. "
            f"The segmented-to-legacy fit-event count ratio has median "
            f"{np.median(segmented_event_ratios):.1f}%, while the median fraction of legacy "
            f"fit events also selected is {np.median(legacy_overlap_percentages):.1f}%."
        )
    else:
        comparison = (
            "The train segmenter rejected all five samples under the configured gates."
        )
    references = "\n".join(
        f"- [{item['citation']}]({item['url']}): {item['relevance']}"
        for item in LITERATURE
    )
    report = f"""# Blink photophysics deconstruction

Run: `{run_directory}`

## Result

The five reproducibly sampled, event-rich accepted fits have a median ROI first-to-last event span
of **{np.median(durations):.1f} ms** (range {np.min(durations):.1f}-{np.max(durations):.1f} ms).
The median first-event-to-peak interval is {np.median(rise):.1f} ms and the median peak-to-last-event
interval is {np.median(decay):.1f} ms.

{comparison}

This does **not** measure an Alexa Fluor 647 ON-state lifetime. PeakLoc first constructs a cumulative
polarity trace over the configured 3 x 3 detection neighborhood, detects a prominence peak, and
derives a broad interval from the smoothing spline's second derivative. ROI generation then uses the
earliest and latest retained event anywhere in the
{int(analyzed[0].sample.roi_positive.shape[0])} x
{int(analyzed[0].sample.roi_positive.shape[1])} fit ROI inside that interval. The
positive gate is one-sided (`t < peak + gate`) and the negative gate is one-sided
(`t > peak - gate`); these conditions classify polarity lobes but do not constrain the full ROI to a
10 ms interval. Background, neighboring emitters, and sparse tail events can therefore extend the
first-to-last span far beyond a molecular transition.

The RAW reconstruction matched stored first/last timestamps and positive/negative fit counts for
**{sum(bool(row["all_checks_pass"]) for row in validation_rows)}/{len(validation_rows)}** samples.
See `{summary_path.name}` for machine-readable values.

The detection trace shown here is replayed over a bounded per-sample interval: the union of the legacy
count window and the +/-250 ms segmentation context, not the original complete processing slice. Its
spline boundary is therefore diagnostic rather than a bitwise full-slice reconstruction. The legacy
count window is read in full, so stored count/timestamp checks remain exact. Segmented polarity maps
are **not refitted** in this notebook; event-count comparisons describe temporal selection, not
post-refit localization accuracy or fit quality.

## Exact timing definitions

- `turn-on proxy`: absolute `t_1st`, the earliest retained event anywhere in the fitted ROI count
  window. The per-blink rise proxy is `t_peak - t_1st`.
- `on-duration proxy`: `t_last - t_1st`.
- `turn-off proxy`: absolute `t_last`, the latest retained event anywhere in the fitted ROI count
  window. The per-blink decay proxy is `t_last - t_peak`.
- These are ROI-window event statistics, not state-transition estimators.
- Segmented turn-on is the first event in the selected dense positive-polarity core train.
- Segmented turn-off is the last event in the matched dense negative-polarity core train.
- Segmented cycle span is `OFF last - ON first`; ON- and OFF-train durations are reported separately.
- Segmented fit-event counts use the refined rectangular ROI inside the selected ON/OFF supports.
- The report separates the segmented-to-legacy event-count ratio from the fraction of legacy fit
  events also selected, and reports newly included and excluded counts.

## Files

- `sample_overview.png` and `.pdf`: timing spans and fit-event counts for all five samples.
- `blink_XX_deconstruction.png` and `.pdf`: legacy ROI maps, detection trace, every exact
  segmentation-context event, and timing arithmetic.
- `roi_event_clouds_3d.png` and `.pdf`: x-y-time event clouds for the five samples.
- `raw_roi_events.csv`: every event in the legacy ROI/count window, with explicit legacy and
  segmented-intersection flags.
- `segmentation_context_events.csv`: every exact production-domain context event and authoritative
  selected-train membership used by the raster.
- `detection_trace.csv`: every raw/interpolated point plotted in the detection replay.
- `sample_selection.csv`, `timing_summary.csv`, and `detection_candidates.csv`: remaining plotted
  source data.
- `transition_trains.csv`: every provisional/refined train and its density, purity, and deviance.
- `blink_intervals.csv`: accepted/rejected interval, timing comparison, and explicit event overlap,
  newly included, and excluded counts.
- `temporal_activity_bins.csv`: 1 ms core/annulus polarity counts underlying train inspection.
- `segmented_fit_events.csv`: every event that would populate segmented fit maps (no refit performed).
- `nearby_seed_intervals.csv`: every retained seed with complete context near each displayed ROI,
  including accepted/rejected train intervals.
- `nearby_seed_intervals.png` and `.pdf`: legacy spans versus independently evaluated per-seed
  intervals; cycle ownership is not deduplicated.

## Photophysics context

AF647 kinetics depend strongly on excitation, buffer, and analysis conditions. Lin et al. reported
few-millisecond ON-state lifetimes at high tested excitation intensities, but that observation cannot
be transferred quantitatively without matching conditions. Event-camera events are log-intensity
threshold crossings, so a fluorescence-state kinetic model must be fitted separately if molecular
dwell times are the goal.

{references}
"""
    path = output_directory / "README.md"
    path.write_text(report, encoding="utf-8")
    return path


def _publication_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _panel_labels(axes: list[Any]) -> None:
    for label, axis in zip(
        "abcdefghijklmnopqrstuvwxyz"[: len(axes)], axes, strict=True
    ):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    return [png_path, pdf_path]


def _load_exclusions(run_directory: Path) -> list[tuple[int, int]]:
    candidates = sorted(
        (run_directory / "reports").glob("diffuse_flash_intervals_*.json")
    )
    if not candidates:
        return []
    payload = _read_json(candidates[-1])
    return [
        (int(interval["start_us"]), int(interval["stop_us"]))
        for interval in payload.get("intervals", [])
    ]


def _load_spatial_masks(
    run_directory: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    shape = (int(config["sensor_height"]), int(config["sensor_width"]))
    metadata_paths = sorted((run_directory / "reports").glob("spatial_mask_*.json"))
    metadata_paths = [
        path for path in metadata_paths if path.name.startswith("spatial_mask_20")
    ]
    active = (
        bool(_read_json(metadata_paths[-1]).get("active")) if metadata_paths else False
    )
    if not active:
        full = np.ones(shape, dtype=np.bool_)
        return full, full
    target_path = _single_artifact(
        run_directory / "reports", "spatial_mask_target_*.npy"
    )
    support_path = _single_artifact(
        run_directory / "reports", "spatial_mask_support_*.npy"
    )
    target = np.asarray(np.load(target_path, allow_pickle=False), dtype=np.bool_)
    support = np.asarray(np.load(support_path, allow_pickle=False), dtype=np.bool_)
    if target.shape != shape or support.shape != shape:
        raise ValueError(
            "Saved spatial-mask shape does not match the effective configuration"
        )
    return target, support


def _resolve_recording_path(input_file: str, run_directory: Path) -> Path:
    path = Path(input_file).expanduser()
    if path.is_file():
        return path.resolve()
    for parent in run_directory.parents:
        candidate = parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Input recording from run metadata is unavailable: {input_file}"
    )


def _single_artifact(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one artifact matching {pattern} in {directory}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
