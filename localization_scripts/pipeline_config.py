from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Self


DEFAULT_INPUT_FOLDER = "data"
DEFAULT_SLICE_DURATION = int(100e6)
DEFAULT_CONFIG_PATH = Path("config.json")

ENVIRONMENT_OVERRIDES = {
    "PEAKLOC_INPUT_FOLDER": "input_folder",
    "PEAKLOC_SLICE_START": "slice_start",
    "PEAKLOC_SLICE_DURATION": "slice_duration",
    "PEAKLOC_SLICE_COUNT": "slice_count",
    "PEAKLOC_MAX_PARALLEL_WORKERS": "max_parallel_workers",
    "PEAKLOC_MAX_CONCURRENT_SLICES": "max_concurrent_slices",
    "PEAKLOC_CPU_WORKER_BUDGET": "cpu_worker_budget",
    "PEAKLOC_SPATIAL_MASK_ENABLED": "spatial_mask_enabled",
}


@dataclass(frozen=True, kw_only=True)
class PeakLocConfig:
    input_folder: str = DEFAULT_INPUT_FOLDER
    recursive_input: bool = False
    slice_start: int = 0
    slice_duration: int = DEFAULT_SLICE_DURATION
    slice_count: int | None = None
    num_cores: int = multiprocessing.cpu_count()
    max_parallel_workers: int = 4
    max_concurrent_slices: int = 1
    cpu_worker_budget: int | None = None
    max_workers_per_slice: int | None = None
    memory_reserve_gib: float = 16.0
    disk_reserve_gib: float = 10.0
    spatial_mask_enabled: bool = False
    spatial_mask_sample_duration_us: int = 60_000_000
    spatial_mask_min_density_quotient: float = 2.7
    spatial_mask_min_component_pixels: int = 20
    spatial_mask_margin_px: int = 12
    spatial_mask_max_support_coverage: float = 0.95
    prominence: float = 12.0
    dataset_fwhm: float = 6.0
    peak_time_threshold: float = 40e3
    polarity_time_gate_us: float = 5e3
    peak_neighbors: int = 9
    roi_radius: int = 8
    convolution_roi_radius: int = 1
    temporal_segmentation_enabled: bool = False
    temporal_context_pre_us: int = 250_000
    temporal_context_post_us: int = 250_000
    temporal_discovery_core_radius_px: float = 4.25
    temporal_core_radius_px: float = 3.5
    temporal_bin_us: int = 1_000
    temporal_max_on_interevent_gap_us: int = 3_000
    temporal_max_off_interevent_gap_us: int = 8_000
    temporal_min_on_events: int = 12
    temporal_min_off_events: int = 8
    temporal_min_on_active_pixels: int = 6
    temporal_min_off_active_pixels: int = 5
    temporal_min_polarity_purity: float = 0.80
    temporal_max_train_duration_us: int = 150_000
    temporal_min_core_density_ratio: float = 1.5
    temporal_min_interval_deviance: float = 2.0
    temporal_max_endpoint_overlap_us: int = 20_000
    temporal_max_cycle_span_us: int = 300_000
    temporal_max_centroid_distance_px: float = 1.75
    temporal_max_on_end_after_seed_us: int = 30_000
    temporal_max_off_start_before_seed_us: int = 30_000
    temporal_ambiguity_margin_us: int = 5_000
    temporal_background_pseudocount: float = 0.5
    peak_min_event_count: int = 2
    diffuse_flash_rejection_enabled: bool = True
    diffuse_flash_bin_duration_us: int = 5_000
    diffuse_flash_min_events_per_polarity: int = 100_000
    diffuse_flash_min_active_pixel_fraction: float = 0.1
    diffuse_flash_max_gap_us: int = 5_000_000
    diffuse_flash_padding_us: int = 50_000
    interpolation_coefficient: int = 5
    spline_smooth: float = 0.7
    plot_subplotsize: int = 6
    plot_result: bool = True
    optical_pixel_size: float = 67.0
    sensor_height: int = 720
    sensor_width: int = 1280
    max_raw_events: int = 1_000_000
    cleanup_temp_outputs: bool = True
    fit_model: str = "poisson_joint"
    allow_uncalibrated: bool = True
    calibration_path: str | None = None
    sigma_psf_px: float | None = None
    fit_sigma: bool = False
    psf_model: str = "pixel_integrated_gaussian"
    background_mode: str = "calibrated_plus_local"
    hot_pixel_policy: str = "mask"
    min_events_pos: int = 3
    min_events_neg: int = 3
    min_valid_pixels: int = 1
    max_fit_cond: float = 100.0
    max_fit_center_offset_px: float | None = 3.0
    max_localization_uncertainty_px: float | None = None
    max_localization_uncertainty_nm: float | None = None
    qc_enabled: bool = True
    qc_output_dirname: str = "qc"
    qc_static_dpi: int = 200
    qc_save_vector: bool = False
    qc_max_events_for_interactive: int = 50_000
    qc_uncertainty_montage_n: int = 36
    qc_generate_html: bool = True
    qc_generate_interactive: bool = False
    qc_generate_temporal_3d: bool = True
    qc_keep_intermediates: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"PeakLoc config must be a JSON object: {config_path}")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Self:
        if "spatial_mask_min_events" in payload:
            raise ValueError(
                "spatial_mask_min_events was replaced by "
                "spatial_mask_min_density_quotient. Choose a quotient relative "
                "to the sample mean event density instead of copying a raw count."
            )
        removed_diffuse_settings = {
            "diffuse_flash_min_positive_events",
            "diffuse_flash_max_local_fraction",
        }.intersection(payload)
        if removed_diffuse_settings:
            raise ValueError(
                "ROI-local diffuse flash settings were replaced by full-sensor "
                "time-interval filtering. Remove: "
                + ", ".join(sorted(removed_diffuse_settings))
            )
        allowed_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "Unknown PeakLoc config setting(s): " + ", ".join(unknown_fields)
            )
        config = cls(**payload)
        config.validate()
        return config

    def with_environment_overrides(
        self, environ: Mapping[str, str] | None = None
    ) -> Self:
        source = os.environ if environ is None else environ
        overrides: dict[str, Any] = {}
        for env_name, field_name in ENVIRONMENT_OVERRIDES.items():
            value = source.get(env_name)
            if value is None:
                continue
            current_value = getattr(self, field_name)
            overrides[field_name] = _coerce_environment_value(value, current_value)
        if not overrides:
            return self
        config = type(self)(**{**self.to_dict(), **overrides})
        config.validate()
        return config

    def validate(self) -> None:
        _require_non_negative("slice_start", self.slice_start)
        _require_positive("slice_duration", self.slice_duration)
        if self.slice_count is not None:
            _require_positive("slice_count", self.slice_count)
        _require_positive("num_cores", self.num_cores)
        _require_positive("max_parallel_workers", self.max_parallel_workers)
        _require_positive("max_concurrent_slices", self.max_concurrent_slices)
        if self.cpu_worker_budget is not None:
            _require_positive("cpu_worker_budget", self.cpu_worker_budget)
        if self.max_workers_per_slice is not None:
            _require_positive("max_workers_per_slice", self.max_workers_per_slice)
        _require_non_negative("memory_reserve_gib", self.memory_reserve_gib)
        _require_non_negative("disk_reserve_gib", self.disk_reserve_gib)
        _require_positive(
            "spatial_mask_sample_duration_us", self.spatial_mask_sample_duration_us
        )
        _require_positive(
            "spatial_mask_min_density_quotient",
            self.spatial_mask_min_density_quotient,
        )
        _require_positive(
            "spatial_mask_min_component_pixels", self.spatial_mask_min_component_pixels
        )
        _require_non_negative("spatial_mask_margin_px", self.spatial_mask_margin_px)
        _require_positive("prominence", self.prominence)
        _require_positive("dataset_fwhm", self.dataset_fwhm)
        _require_positive("peak_time_threshold", self.peak_time_threshold)
        _require_non_negative("polarity_time_gate_us", self.polarity_time_gate_us)
        _require_positive("peak_neighbors", self.peak_neighbors)
        _require_positive("roi_radius", self.roi_radius)
        _require_positive("convolution_roi_radius", self.convolution_roi_radius)
        _require_positive("temporal_context_pre_us", self.temporal_context_pre_us)
        _require_positive("temporal_context_post_us", self.temporal_context_post_us)
        _require_positive(
            "temporal_discovery_core_radius_px",
            self.temporal_discovery_core_radius_px,
        )
        _require_positive("temporal_core_radius_px", self.temporal_core_radius_px)
        if self.temporal_core_radius_px > self.temporal_discovery_core_radius_px:
            raise ValueError(
                "temporal_core_radius_px must not exceed "
                "temporal_discovery_core_radius_px"
            )
        if (
            self.temporal_segmentation_enabled
            and self.temporal_discovery_core_radius_px >= self.roi_radius
        ):
            raise ValueError(
                "temporal_discovery_core_radius_px must be smaller than roi_radius "
                "so that the background annulus is non-empty"
            )
        _require_positive("temporal_bin_us", self.temporal_bin_us)
        _require_positive(
            "temporal_max_on_interevent_gap_us",
            self.temporal_max_on_interevent_gap_us,
        )
        _require_positive(
            "temporal_max_off_interevent_gap_us",
            self.temporal_max_off_interevent_gap_us,
        )
        _require_positive("temporal_min_on_events", self.temporal_min_on_events)
        _require_positive("temporal_min_off_events", self.temporal_min_off_events)
        _require_positive(
            "temporal_min_on_active_pixels", self.temporal_min_on_active_pixels
        )
        _require_positive(
            "temporal_min_off_active_pixels", self.temporal_min_off_active_pixels
        )
        _require_positive(
            "temporal_max_train_duration_us", self.temporal_max_train_duration_us
        )
        _require_positive(
            "temporal_min_core_density_ratio",
            self.temporal_min_core_density_ratio,
        )
        _require_positive(
            "temporal_min_interval_deviance", self.temporal_min_interval_deviance
        )
        _require_non_negative(
            "temporal_max_endpoint_overlap_us",
            self.temporal_max_endpoint_overlap_us,
        )
        _require_positive("temporal_max_cycle_span_us", self.temporal_max_cycle_span_us)
        _require_positive(
            "temporal_max_centroid_distance_px",
            self.temporal_max_centroid_distance_px,
        )
        _require_non_negative(
            "temporal_max_on_end_after_seed_us",
            self.temporal_max_on_end_after_seed_us,
        )
        _require_non_negative(
            "temporal_max_off_start_before_seed_us",
            self.temporal_max_off_start_before_seed_us,
        )
        _require_non_negative(
            "temporal_ambiguity_margin_us", self.temporal_ambiguity_margin_us
        )
        _require_positive(
            "temporal_background_pseudocount",
            self.temporal_background_pseudocount,
        )
        if not 0 < self.temporal_min_polarity_purity <= 1:
            raise ValueError("temporal_min_polarity_purity must be in (0, 1]")
        if self.temporal_segmentation_enabled and self.slice_duration <= (
            self.temporal_context_pre_us + self.temporal_context_post_us
        ):
            raise ValueError(
                "slice_duration must exceed temporal_context_pre_us + "
                "temporal_context_post_us when temporal segmentation is enabled"
            )
        _require_positive("peak_min_event_count", self.peak_min_event_count)
        _require_positive(
            "diffuse_flash_bin_duration_us", self.diffuse_flash_bin_duration_us
        )
        _require_positive(
            "diffuse_flash_min_events_per_polarity",
            self.diffuse_flash_min_events_per_polarity,
        )
        _require_non_negative("diffuse_flash_max_gap_us", self.diffuse_flash_max_gap_us)
        _require_non_negative("diffuse_flash_padding_us", self.diffuse_flash_padding_us)
        _require_positive("interpolation_coefficient", self.interpolation_coefficient)
        _require_positive("plot_subplotsize", self.plot_subplotsize)
        _require_positive("optical_pixel_size", self.optical_pixel_size)
        _require_positive("sensor_height", self.sensor_height)
        _require_positive("sensor_width", self.sensor_width)
        _require_positive("max_raw_events", self.max_raw_events)
        _require_positive("min_events_pos", self.min_events_pos)
        _require_positive("min_events_neg", self.min_events_neg)
        _require_positive("min_valid_pixels", self.min_valid_pixels)
        _require_positive("max_fit_cond", self.max_fit_cond)
        if self.max_fit_center_offset_px is not None:
            _require_positive("max_fit_center_offset_px", self.max_fit_center_offset_px)
            if self.max_fit_center_offset_px > self.roi_radius:
                raise ValueError("max_fit_center_offset_px must not exceed roi_radius")
        _require_positive("qc_static_dpi", self.qc_static_dpi)
        _require_positive(
            "qc_max_events_for_interactive", self.qc_max_events_for_interactive
        )
        _require_positive("qc_uncertainty_montage_n", self.qc_uncertainty_montage_n)
        if self.max_localization_uncertainty_px is not None:
            _require_positive(
                "max_localization_uncertainty_px",
                self.max_localization_uncertainty_px,
            )
        if self.max_localization_uncertainty_nm is not None:
            _require_positive(
                "max_localization_uncertainty_nm",
                self.max_localization_uncertainty_nm,
            )
        _require_bool("plot_result", self.plot_result)
        _require_bool("recursive_input", self.recursive_input)
        _require_bool("cleanup_temp_outputs", self.cleanup_temp_outputs)
        _require_bool("spatial_mask_enabled", self.spatial_mask_enabled)
        _require_bool(
            "temporal_segmentation_enabled", self.temporal_segmentation_enabled
        )
        _require_bool(
            "diffuse_flash_rejection_enabled", self.diffuse_flash_rejection_enabled
        )
        _require_bool("allow_uncalibrated", self.allow_uncalibrated)
        _require_bool("fit_sigma", self.fit_sigma)
        _require_bool("qc_enabled", self.qc_enabled)
        _require_bool("qc_save_vector", self.qc_save_vector)
        _require_bool("qc_generate_html", self.qc_generate_html)
        _require_bool("qc_generate_interactive", self.qc_generate_interactive)
        _require_bool("qc_generate_temporal_3d", self.qc_generate_temporal_3d)
        _require_bool("qc_keep_intermediates", self.qc_keep_intermediates)
        if not self.qc_output_dirname:
            raise ValueError("qc_output_dirname must not be empty")
        if not 0 <= self.spline_smooth <= 1:
            raise ValueError("spline_smooth must be between 0 and 1")
        if not 0 < self.spatial_mask_max_support_coverage <= 1:
            raise ValueError(
                "spatial_mask_max_support_coverage must be in the interval (0, 1]"
            )
        if not 0 < self.diffuse_flash_min_active_pixel_fraction <= 1:
            raise ValueError(
                "diffuse_flash_min_active_pixel_fraction must be in the interval (0, 1]"
            )
        if self.fit_model != "poisson_joint":
            raise ValueError("fit_model must be 'poisson_joint'")
        if self.psf_model != "pixel_integrated_gaussian":
            raise ValueError("psf_model must be 'pixel_integrated_gaussian'")
        if self.background_mode not in {
            "calibrated_only",
            "calibrated_plus_local",
            "local_only",
        }:
            raise ValueError(
                "background_mode must be 'calibrated_only', "
                "'calibrated_plus_local', or 'local_only'"
            )
        if self.hot_pixel_policy != "mask":
            raise ValueError("hot_pixel_policy must be 'mask'")
        if self.sigma_psf_px is not None:
            _require_positive("sigma_psf_px", self.sigma_psf_px)
        if self.calibration_path is None and not self.allow_uncalibrated:
            raise ValueError(
                "calibration_path is required when allow_uncalibrated is false"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def optical_pixel_size_nm(self) -> float:
        return self.optical_pixel_size

    @property
    def available_cpu_count(self) -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    @property
    def resolved_cpu_worker_budget(self) -> int:
        requested = (
            self.num_cores if self.cpu_worker_budget is None else self.cpu_worker_budget
        )
        return max(1, min(requested, self.num_cores, self.available_cpu_count))

    @property
    def effective_concurrent_slices(self) -> int:
        return max(1, min(self.max_concurrent_slices, self.resolved_cpu_worker_budget))

    @property
    def parallel_workers(self) -> int:
        """Return the per-slice worker quota within the global CPU budget."""
        per_slice_cap = (
            self.max_parallel_workers
            if self.max_workers_per_slice is None
            else min(self.max_parallel_workers, self.max_workers_per_slice)
        )
        parallel_stage_budget = max(
            1,
            self.resolved_cpu_worker_budget - self.effective_concurrent_slices + 1,
        )
        return max(1, min(self.num_cores, per_slice_cap, parallel_stage_budget))

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return self.sensor_height, self.sensor_width


def load_peakloc_config(
    config_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> PeakLocConfig:
    source = os.environ if environ is None else environ
    path = Path(config_path) if config_path is not None else None
    if path is None and source.get("PEAKLOC_CONFIG"):
        path = Path(source["PEAKLOC_CONFIG"])
    if path is None and DEFAULT_CONFIG_PATH.is_file():
        path = DEFAULT_CONFIG_PATH
    config = PeakLocConfig.from_json(path) if path is not None else PeakLocConfig()
    return config.with_environment_overrides(source)


def write_effective_config(config: PeakLocConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def _coerce_environment_value(value: str, current_value: Any) -> Any:
    if isinstance(current_value, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int):
        return int(float(value))
    if isinstance(current_value, float):
        return float(value)
    return value


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
