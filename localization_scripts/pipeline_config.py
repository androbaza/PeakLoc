from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Self


DEFAULT_INPUT_FOLDER = "/home/smlm-workstation/event-smlm/Paris/process/"
DEFAULT_SLICE_DURATION = int(100e6)

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
    peak_neighbors: int = 9
    roi_radius: int = 8
    convolution_roi_radius: int = 1
    interpolation_coefficient: int = 5
    spline_smooth: float = 0.7
    plot_subplotsize: int = 6
    max_raw_events: int = 1_000_000
    cleanup_temp_outputs: bool = True

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
        _require_positive("peak_neighbors", self.peak_neighbors)
        _require_positive("roi_radius", self.roi_radius)
        _require_positive("convolution_roi_radius", self.convolution_roi_radius)
        _require_positive("interpolation_coefficient", self.interpolation_coefficient)
        _require_positive("plot_subplotsize", self.plot_subplotsize)
        _require_positive("max_raw_events", self.max_raw_events)
        if not 0 <= self.spline_smooth <= 1:
            raise ValueError("spline_smooth must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_peakloc_config(
    config_path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> PeakLocConfig:
    source = os.environ if environ is None else environ
    path = Path(config_path) if config_path is not None else None
    if path is None and source.get("PEAKLOC_CONFIG"):
        path = Path(source["PEAKLOC_CONFIG"])
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
