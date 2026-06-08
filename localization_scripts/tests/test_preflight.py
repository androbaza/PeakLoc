from pathlib import Path

import numpy as np

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.preflight import run_preflight, write_preflight_report


def test_missing_input_folder_gives_one_error(tmp_path):
    config = PeakLocConfig(input_folder=str(tmp_path / "missing"))

    report = run_preflight(config)

    errors = [issue for issue in report.issues if issue.severity == "error"]
    assert [issue.code for issue in errors] == ["missing_input_folder"]
    assert report.has_errors


def test_missing_calibration_warns_by_default_and_errors_in_strict_mode(tmp_path):
    _write_events(tmp_path / "recording.npy")
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        background_mode="calibrated_plus_local",
        allow_uncalibrated=True,
    )

    default_report = run_preflight(config)
    strict_report = run_preflight(config, strict_mode=True)

    default_missing = _issue(default_report.issues, "missing_calibration")
    strict_missing = _issue(strict_report.issues, "missing_calibration")
    assert default_missing.severity == "warning"
    assert strict_missing.severity == "error"


def test_inconsistent_fwhm_and_sigma_gives_warning(tmp_path):
    _write_events(tmp_path / "recording.npy")
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        background_mode="local_only",
        dataset_fwhm=10.0,
        sigma_psf_px=1.0,
    )

    report = run_preflight(config)

    assert _issue(report.issues, "inconsistent_psf_fwhm").severity == "warning"


def test_too_small_roi_radius_gives_warning(tmp_path):
    _write_events(tmp_path / "recording.npy")
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        background_mode="local_only",
        roi_radius=2,
        sigma_psf_px=1.0,
    )

    report = run_preflight(config)

    assert _issue(report.issues, "roi_truncates_psf").severity == "warning"


def test_bad_npy_event_dtype_gives_error(tmp_path):
    np.save(tmp_path / "bad.npy", np.zeros(3, dtype=[("x", np.uint16)]))
    config = PeakLocConfig(input_folder=str(tmp_path), background_mode="local_only")

    report = run_preflight(config)

    assert _issue(report.issues, "bad_event_dtype").severity == "error"
    assert report.has_errors


def test_valid_synthetic_npy_plus_null_calibration_passes_non_strict_mode(tmp_path):
    _write_events(tmp_path / "recording.npy")
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        background_mode="local_only",
        allow_uncalibrated=True,
        sensor_height=32,
        sensor_width=32,
        slice_duration=20,
    )

    report = run_preflight(config)

    assert not report.has_errors
    assert _issue(report.issues, "calibration_status").severity == "info"


def test_write_preflight_report_creates_markdown(tmp_path):
    _write_events(tmp_path / "recording.npy")
    config = PeakLocConfig(input_folder=str(tmp_path), background_mode="local_only")
    report = run_preflight(config, config_path=Path("config.json"))
    output_path = tmp_path / "reports" / "preflight.md"

    write_preflight_report(report, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "# PeakLoc Preflight Report" in text
    assert "recording.npy" in text
    assert report.effective_config_hash in text


def _write_events(path: Path) -> None:
    events = np.zeros(
        4,
        dtype=[("x", np.uint16), ("y", np.uint16), ("p", np.int8), ("t", np.uint64)],
    )
    events["x"] = [1, 2, 3, 4]
    events["y"] = [1, 2, 3, 4]
    events["p"] = [0, 1, 0, 1]
    events["t"] = [0, 10, 20, 30]
    np.save(path, events)


def _issue(issues, code: str):
    matches = [issue for issue in issues if issue.code == code]
    assert len(matches) == 1
    return matches[0]
