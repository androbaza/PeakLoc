from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Self


DEFAULT_INPUT_FOLDER = "/home/smlm-workstation/event-smlm/Paris/process/"
DEFAULT_SLICE_DURATION = int(100e6)
DEFAULT_CONFIG_PATH = Path("config.json")

ENVIRONMENT_OVERRIDES = {
    "PEAKLOC_INPUT_FOLDER": "input_folder",
    "PEAKLOC_SLICE_START": "slice_start",
    "PEAKLOC_SLICE_DURATION": "slice_duration",
}


@dataclass(frozen=True, kw_only=True)
class PeakLocConfig:
    input_folder: str = DEFAULT_INPUT_FOLDER
    slice_start: int = 0
    slice_duration: int = DEFAULT_SLICE_DURATION
    num_cores: int = multiprocessing.cpu_count()
    prominence: float = 12.0
    dataset_fwhm: float = 6.0
    peak_time_threshold: float = 40e3
    polarity_time_gate_us: float = 5e3
    peak_neighbors: int = 9
    roi_radius: int = 8
    convolution_roi_radius: int = 1
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
    max_fit_cond: float = 1e10

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
        _require_positive("num_cores", self.num_cores)
        _require_positive("prominence", self.prominence)
        _require_positive("dataset_fwhm", self.dataset_fwhm)
        _require_positive("peak_time_threshold", self.peak_time_threshold)
        _require_non_negative("polarity_time_gate_us", self.polarity_time_gate_us)
        _require_positive("peak_neighbors", self.peak_neighbors)
        _require_positive("roi_radius", self.roi_radius)
        _require_positive("convolution_roi_radius", self.convolution_roi_radius)
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
        _require_bool("plot_result", self.plot_result)
        _require_bool("cleanup_temp_outputs", self.cleanup_temp_outputs)
        _require_bool("allow_uncalibrated", self.allow_uncalibrated)
        _require_bool("fit_sigma", self.fit_sigma)
        if not 0 <= self.spline_smooth <= 1:
            raise ValueError("spline_smooth must be between 0 and 1")
        if self.fit_model not in {"legacy_lsq", "poisson_joint"}:
            raise ValueError("fit_model must be 'legacy_lsq' or 'poisson_joint'")
        if self.psf_model != "pixel_integrated_gaussian":
            raise ValueError("psf_model must be 'pixel_integrated_gaussian'")
        if self.background_mode not in {"calibrated_plus_local", "local_only"}:
            raise ValueError(
                "background_mode must be 'calibrated_plus_local' or 'local_only'"
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
