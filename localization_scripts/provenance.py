from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np

from localization_scripts.pipeline_config import PeakLocConfig, write_effective_config
from localization_scripts.preflight import effective_config_hash


@dataclass(frozen=True)
class ProvenanceArtifacts:
    accepted_csv: Path
    attempted_csv: Path
    qc_csv: Path
    run_metadata_json: Path
    software_versions_json: Path
    config_effective_json: Path
    config_hash_txt: Path


def save_portable_outputs(
    *,
    recording: Any,
    config: PeakLocConfig,
    accepted_localizations: np.ndarray,
    attempted_localizations: np.ndarray,
    localization_qc: np.ndarray,
    timestamp: str,
) -> list[Path]:
    output_dir = Path(recording.output_folder)
    artifacts = ProvenanceArtifacts(
        accepted_csv=output_dir / "localizations_accepted.csv",
        attempted_csv=output_dir / "localizations_attempted.csv",
        qc_csv=output_dir / "localization_qc.csv",
        run_metadata_json=output_dir / "run_metadata.json",
        software_versions_json=output_dir / "software_versions.json",
        config_effective_json=output_dir / "config_effective.json",
        config_hash_txt=output_dir / "config_hash.txt",
    )
    _write_structured_csv(accepted_localizations, artifacts.accepted_csv)
    _write_structured_csv(attempted_localizations, artifacts.attempted_csv)
    _write_structured_csv(localization_qc, artifacts.qc_csv)
    write_effective_config(config, artifacts.config_effective_json)
    artifacts.config_hash_txt.write_text(
        effective_config_hash(config) + "\n", encoding="utf-8"
    )
    artifacts.run_metadata_json.write_text(
        json.dumps(
            _run_metadata(recording, config, timestamp=timestamp),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.software_versions_json.write_text(
        json.dumps(software_versions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return list(asdict(artifacts).values())


def software_versions() -> dict[str, Any]:
    packages = {}
    for package_name in ("numpy", "scipy", "matplotlib", "plotly"):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "packages": packages,
    }


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_metadata(
    recording: Any, config: PeakLocConfig, *, timestamp: str
) -> dict[str, Any]:
    calibration = getattr(recording, "calibration_metadata", {}) or {}
    return {
        "timestamp": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "config_hash": effective_config_hash(config),
        "input_file": str(getattr(recording, "input_file", "")),
        "input_event_count": int(getattr(recording, "event_count", 0)),
        "time_min": getattr(recording, "time_min", None),
        "time_max": getattr(recording, "time_max", None),
        "calibration_id": calibration.get("calibration_id"),
        "calibrated": calibration.get("calibrated"),
        "pixel_size_nm": config.optical_pixel_size_nm,
        "sensor_shape": list(config.sensor_shape),
        "fit_model": config.fit_model,
        "psf_model": config.psf_model,
        "background_mode": config.background_mode,
        "filtering_thresholds": {
            "max_fit_cond": config.max_fit_cond,
            "min_valid_pixels": config.min_valid_pixels,
            "max_localization_uncertainty_px": config.max_localization_uncertainty_px,
            "max_localization_uncertainty_nm": config.max_localization_uncertainty_nm,
        },
        "parquet_status": "not_written_pyarrow_not_required",
    }


def _write_structured_csv(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = _scalar_field_names(array)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(field_names)
        for row in array:
            writer.writerow([_csv_value(row[field_name]) for field_name in field_names])


def _scalar_field_names(array: np.ndarray) -> list[str]:
    if array.dtype.names is None:
        return []
    return [
        field_name
        for field_name in array.dtype.names
        if array.dtype[field_name].shape == ()
    ]


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
