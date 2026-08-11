from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationRequest:
    dark_path: str
    blank_path: str
    output_path: str
    pixel_size_nm: float
    sensor_model: str
    calibration_id: str
    height: int | None
    width: int | None
    max_events: int

    @classmethod
    def from_json(cls, path: Path) -> CalibrationRequest:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise TypeError("Calibration request must be a JSON object")
        return cls(**payload)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def startup_config_path() -> Path:
    return application_directory() / "config.json"


def worker_command(*arguments: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    entrypoint = application_directory() / "PeakLocGUI.py"
    return [sys.executable, "-u", str(entrypoint), *arguments]


def run_pipeline_worker(config_path: Path, *, preflight_only: bool) -> None:
    from localization_scripts.pipeline_config import PeakLocConfig
    from localization_scripts.pipeline_runner import run_batch
    from localization_scripts.preflight import run_preflight

    config = PeakLocConfig.from_json(config_path)
    report = run_preflight(config, config_path=config_path)
    print(
        f"Preflight checked {len(report.event_files)} recording(s): "
        f"{'FAILED' if report.has_errors else 'PASSED'}",
        flush=True,
    )
    for issue in report.issues:
        field = f" [{issue.field}]" if issue.field else ""
        print(
            f"{issue.severity.upper():7}{field} {issue.message}",
            flush=True,
        )
        if issue.suggestion:
            print(f"        Suggestion: {issue.suggestion}", flush=True)
    if report.has_errors:
        raise RuntimeError("Preflight found errors. Correct them before processing.")
    if preflight_only:
        print("Setup check complete. PeakLoc is ready to run.", flush=True)
        return

    print("Starting PeakLoc processing", flush=True)
    results = run_batch(config)
    print(f"PeakLoc completed {len(results)} recording(s).", flush=True)
    for result in results:
        print(f"Output: {result.output_folder}", flush=True)


def run_calibration_worker(request_path: Path) -> None:
    from calibration_scripts.build_event_calibration import build_event_calibration

    request = CalibrationRequest.from_json(request_path)
    output_path = build_event_calibration(
        Path(request.dark_path),
        Path(request.blank_path),
        Path(request.output_path),
        pixel_size_nm=request.pixel_size_nm,
        sensor_model=request.sensor_model,
        calibration_id=request.calibration_id,
        height=request.height,
        width=request.width,
        max_events=request.max_events,
    )
    print(f"Calibration ready: {output_path}", flush=True)
