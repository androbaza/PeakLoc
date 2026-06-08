from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np

from localization_scripts.calibration import load_calibration
from localization_scripts.pipeline_config import PeakLocConfig

FWHM_FROM_SIGMA = 2.354820045
EVENT_FIELDS = frozenset({"x", "y", "p", "t"})


@dataclass(frozen=True)
class PreflightIssue:
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    field: str | None = None
    suggestion: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    config_path: Path | None
    input_folder: Path
    event_files: tuple[Path, ...]
    issues: tuple[PreflightIssue, ...]
    effective_config_hash: str
    strict_mode: bool

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def run_preflight(
    config: PeakLocConfig,
    *,
    config_path: Path | None = None,
    strict_mode: bool = False,
    sample_events_per_file: int = 100_000,
) -> PreflightReport:
    input_folder = Path(config.input_folder)
    issues: list[PreflightIssue] = []
    event_files: tuple[Path, ...] = ()

    if not input_folder.is_dir():
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_input_folder",
                field="input_folder",
                message=f"Input folder does not exist: {input_folder}",
                suggestion="Set input_folder or PEAKLOC_INPUT_FOLDER to a folder with .raw or .npy recordings.",
            )
        )
        return _report(
            config, config_path, input_folder, event_files, issues, strict_mode
        )

    event_files = _recording_files(input_folder)
    if not event_files:
        issues.append(
            PreflightIssue(
                severity="error",
                code="missing_recordings",
                field="input_folder",
                message=f"No .raw or .npy recordings were found in {input_folder}",
                suggestion="Place recordings in the input folder; .bias files and subdirectories are ignored.",
            )
        )

    _check_event_files(config, event_files, sample_events_per_file, issues)
    _check_config_consistency(config, strict_mode, issues)
    _check_calibration(config, strict_mode, issues)
    _add_output_info(event_files, issues)
    return _report(config, config_path, input_folder, event_files, issues, strict_mode)


def write_preflight_report(report: PreflightReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PeakLoc Preflight Report",
        "",
        f"- Config path: `{report.config_path}`",
        f"- Input folder: `{report.input_folder.resolve()}`",
        f"- Strict mode: `{report.strict_mode}`",
        f"- Effective config hash: `{report.effective_config_hash}`",
        f"- Recording files: `{len(report.event_files)}`",
        f"- Status: `{'failed' if report.has_errors else 'passed'}`",
        "",
        "## Recording Files",
        "",
    ]
    if report.event_files:
        lines.extend(f"- `{path.resolve()}`" for path in report.event_files)
    else:
        lines.append("No recording files found.")

    lines.extend(["", "## Issues", ""])
    if report.issues:
        lines.extend(
            [
                "| Severity | Code | Field | Message | Suggestion |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for issue in report.issues:
            lines.append(
                "| "
                + " | ".join(
                    [
                        issue.severity,
                        issue.code,
                        issue.field or "",
                        _escape_markdown_table(issue.message),
                        _escape_markdown_table(issue.suggestion or ""),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No issues found.")

    lines.extend(["", "## Output Folders", ""])
    if report.event_files:
        for path in report.event_files:
            lines.append(f"- `{path.with_suffix('').resolve()}`")
    else:
        lines.append("No output folders resolved because no recordings were found.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(
    config: PeakLocConfig,
    config_path: Path | None,
    input_folder: Path,
    event_files: tuple[Path, ...],
    issues: list[PreflightIssue],
    strict_mode: bool,
) -> PreflightReport:
    return PreflightReport(
        config_path=config_path,
        input_folder=input_folder,
        event_files=event_files,
        issues=tuple(issues),
        effective_config_hash=effective_config_hash(config),
        strict_mode=strict_mode,
    )


def effective_config_hash(config: PeakLocConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recording_files(input_folder: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix in {".raw", ".npy"}
        )
    )


def _check_event_files(
    config: PeakLocConfig,
    event_files: tuple[Path, ...],
    sample_events_per_file: int,
    issues: list[PreflightIssue],
) -> None:
    for path in event_files:
        if path.suffix == ".raw":
            issues.append(
                PreflightIssue(
                    severity="info",
                    code="raw_recording_detected",
                    field="input_folder",
                    message=f"Raw recording will be validated during OpenEB loading: {path.name}",
                )
            )
            continue
        _check_npy_event_file(config, path, sample_events_per_file, issues)


def _check_npy_event_file(
    config: PeakLocConfig,
    path: Path,
    sample_events_per_file: int,
    issues: list[PreflightIssue],
) -> None:
    try:
        events = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        issues.append(
            PreflightIssue(
                severity="error",
                code="unreadable_npy",
                field="input_folder",
                message=f"Could not read {path.name}: {exc}",
            )
        )
        return

    if events.dtype.names is None or not EVENT_FIELDS.issubset(events.dtype.names):
        issues.append(
            PreflightIssue(
                severity="error",
                code="bad_event_dtype",
                field="input_folder",
                message=f"{path.name} must be a structured event array with fields x, y, p, t.",
                suggestion="Use dtype fields x, y, p, and t for event-camera recordings.",
            )
        )
        return

    if events.size == 0:
        issues.append(
            PreflightIssue(
                severity="warning",
                code="empty_recording",
                field="input_folder",
                message=f"{path.name} contains no events.",
            )
        )
        return

    sample = events[: min(events.size, sample_events_per_file)]
    _check_polarity(path, sample, issues)
    _check_coordinates(config, path, sample, issues)
    _check_timestamps(config, path, sample, events.size, issues)


def _check_polarity(
    path: Path, sample: np.ndarray, issues: list[PreflightIssue]
) -> None:
    unique_polarities = set(np.unique(sample["p"]).tolist())
    if unique_polarities.issubset({0, 1, False, True}):
        return
    if unique_polarities.issubset({-1, 1}):
        issues.append(
            PreflightIssue(
                severity="warning",
                code="convertible_polarity_convention",
                field="p",
                message=f"{path.name} uses -1/1 polarity values; convert to 0/1 before running PeakLoc.",
                suggestion="Map negative polarity to 0 and positive polarity to 1.",
            )
        )
        return
    issues.append(
        PreflightIssue(
            severity="error",
            code="invalid_polarity_values",
            field="p",
            message=f"{path.name} has unsupported polarity values: {sorted(unique_polarities)}",
            suggestion="PeakLoc expects polarity values encoded as 0/1.",
        )
    )


def _check_coordinates(
    config: PeakLocConfig,
    path: Path,
    sample: np.ndarray,
    issues: list[PreflightIssue],
) -> None:
    x = np.asarray(sample["x"])
    y = np.asarray(sample["y"])
    if np.any(x < 0) or np.any(x >= config.sensor_width):
        issues.append(
            PreflightIssue(
                severity="error",
                code="x_out_of_bounds",
                field="sensor_width",
                message=f"{path.name} contains x coordinates outside [0, {config.sensor_width - 1}].",
            )
        )
    if np.any(y < 0) or np.any(y >= config.sensor_height):
        issues.append(
            PreflightIssue(
                severity="error",
                code="y_out_of_bounds",
                field="sensor_height",
                message=f"{path.name} contains y coordinates outside [0, {config.sensor_height - 1}].",
            )
        )


def _check_timestamps(
    config: PeakLocConfig,
    path: Path,
    sample: np.ndarray,
    event_count: int,
    issues: list[PreflightIssue],
) -> None:
    timestamps = np.asarray(sample["t"])
    if np.any(timestamps < 0):
        issues.append(
            PreflightIssue(
                severity="error",
                code="negative_timestamp",
                field="t",
                message=f"{path.name} contains negative timestamps.",
            )
        )
        return

    if timestamps.size > 1 and np.any(np.diff(timestamps.astype(np.float64)) < 0):
        issues.append(
            PreflightIssue(
                severity="warning",
                code="non_monotonic_timestamps",
                field="t",
                message=f"{path.name} timestamps are not monotonic in the sampled events.",
                suggestion="PeakLoc can sort local neighborhoods, but monotonic input is safer for auditing.",
            )
        )

    duration = float(np.max(timestamps) - np.min(timestamps))
    if event_count > sample.size:
        issues.append(
            PreflightIssue(
                severity="info",
                code="sampled_event_file",
                field="input_folder",
                message=f"{path.name} preflight sampled {sample.size} of {event_count} events.",
            )
        )
    if duration > 0 and config.slice_duration >= duration:
        issues.append(
            PreflightIssue(
                severity="warning",
                code="single_slice_recording",
                field="slice_duration",
                message=(
                    f"slice_duration={config.slice_duration:g} us is greater than or comparable "
                    f"to the sampled recording duration {duration:g} us; the run will likely produce one slice."
                ),
            )
        )


def _check_config_consistency(
    config: PeakLocConfig,
    strict_mode: bool,
    issues: list[PreflightIssue],
) -> None:
    if config.sigma_psf_px is not None:
        expected_fwhm = FWHM_FROM_SIGMA * config.sigma_psf_px
        relative_error = abs(config.dataset_fwhm - expected_fwhm) / expected_fwhm
        if relative_error > 0.05:
            issues.append(
                PreflightIssue(
                    severity="warning",
                    code="inconsistent_psf_fwhm",
                    field="dataset_fwhm",
                    message=(
                        f"dataset_fwhm={config.dataset_fwhm:g} px differs from "
                        f"2.354820045 * sigma_psf_px={expected_fwhm:g} px by "
                        f"{relative_error:.1%}."
                    ),
                    suggestion="Use dataset_fwhm only when it matches the configured PSF sigma.",
                )
            )

        min_roi_radius = ceil(3 * config.sigma_psf_px)
        if config.roi_radius < min_roi_radius:
            issues.append(
                PreflightIssue(
                    severity="warning",
                    code="roi_truncates_psf",
                    field="roi_radius",
                    message=(
                        f"roi_radius={config.roi_radius} is smaller than ceil(3 * sigma_psf_px)="
                        f"{min_roi_radius}; the fitted PSF may be truncated."
                    ),
                )
            )

        min_peak_neighbors = ceil(2 * config.sigma_psf_px)
        if config.peak_neighbors < min_peak_neighbors:
            issues.append(
                PreflightIssue(
                    severity="warning",
                    code="peak_neighbors_below_psf_scale",
                    field="peak_neighbors",
                    message=(
                        f"peak_neighbors={config.peak_neighbors} is below ceil(2 * sigma_psf_px)="
                        f"{min_peak_neighbors}; nearby PSF-scale peaks may duplicate."
                    ),
                )
            )

    if config.convolution_roi_radius > config.roi_radius:
        issues.append(
            PreflightIssue(
                severity="error",
                code="convolution_radius_exceeds_roi",
                field="convolution_roi_radius",
                message="convolution_roi_radius must be less than or equal to roi_radius.",
            )
        )

    roi_pixel_count = (2 * config.roi_radius + 1) ** 2
    if config.min_valid_pixels > roi_pixel_count:
        issues.append(
            PreflightIssue(
                severity="error",
                code="min_valid_pixels_exceeds_roi",
                field="min_valid_pixels",
                message=(
                    f"min_valid_pixels={config.min_valid_pixels} exceeds the ROI pixel count "
                    f"{roi_pixel_count}."
                ),
            )
        )

    _check_uncertainty_thresholds(config, issues)
    _check_missing_calibration_policy(config, strict_mode, issues)
    if config.cleanup_temp_outputs:
        issues.append(
            PreflightIssue(
                severity="warning",
                code="cleanup_temp_outputs_enabled",
                field="cleanup_temp_outputs",
                message="Per-slice temporary arrays will be removed after concatenation.",
                suggestion="For debug/tuning runs set cleanup_temp_outputs=false.",
            )
        )


def _check_uncertainty_thresholds(
    config: PeakLocConfig, issues: list[PreflightIssue]
) -> None:
    px_from_nm = None
    if config.max_localization_uncertainty_nm is not None:
        px_from_nm = (
            config.max_localization_uncertainty_nm / config.optical_pixel_size_nm
        )
        issues.append(
            PreflightIssue(
                severity="info",
                code="uncertainty_nm_threshold_px",
                field="max_localization_uncertainty_nm",
                message=(
                    f"max_localization_uncertainty_nm={config.max_localization_uncertainty_nm:g} nm "
                    f"equals {px_from_nm:.4g} px at {config.optical_pixel_size_nm:g} nm/px."
                ),
            )
        )

    if config.max_localization_uncertainty_px is not None and px_from_nm is not None:
        px = config.max_localization_uncertainty_px
        relative_difference = abs(px - px_from_nm) / max(
            px_from_nm, np.finfo(float).eps
        )
        if relative_difference > 0.05:
            issues.append(
                PreflightIssue(
                    severity="warning",
                    code="inconsistent_uncertainty_thresholds",
                    field="max_localization_uncertainty_px",
                    message=(
                        f"max_localization_uncertainty_px={px:g} differs from the nm threshold "
                        f"converted to pixels ({px_from_nm:.4g}) by {relative_difference:.1%}."
                    ),
                )
            )


def _check_missing_calibration_policy(
    config: PeakLocConfig,
    strict_mode: bool,
    issues: list[PreflightIssue],
) -> None:
    if (
        config.background_mode != "local_only"
        and config.calibration_path is None
        and config.allow_uncalibrated
    ):
        issues.append(
            PreflightIssue(
                severity="error" if strict_mode else "warning",
                code="missing_calibration",
                field="calibration_path",
                message=(
                    f"background_mode={config.background_mode!r} has no calibration_path; "
                    "PeakLoc will use null calibration."
                ),
                suggestion=(
                    "For publication-quality real data, provide dark and blank calibration maps "
                    "or use local_only explicitly for exploratory tuning."
                ),
            )
        )


def _check_calibration(
    config: PeakLocConfig,
    strict_mode: bool,
    issues: list[PreflightIssue],
) -> None:
    try:
        calibration = load_calibration(
            config.calibration_path,
            config.sensor_shape,
            allow_uncalibrated=config.allow_uncalibrated,
        )
    except Exception as exc:
        issues.append(
            PreflightIssue(
                severity="error",
                code="invalid_calibration",
                field="calibration_path",
                message=str(exc),
            )
        )
        return

    pixel_count = np.prod(calibration.sensor_shape)
    hot_pixel_fraction = float(
        np.count_nonzero(calibration.hot_pixel_mask) / pixel_count
    )
    valid_pixel_fraction = float(
        np.count_nonzero(calibration.valid_pixel_mask) / pixel_count
    )
    issues.append(
        PreflightIssue(
            severity="info",
            code="calibration_status",
            field="calibration_path",
            message=(
                f"calibration_id={calibration.calibration_id}, calibrated={calibration.calibrated}, "
                f"hot_pixel_fraction={hot_pixel_fraction:.4g}, "
                f"valid_pixel_fraction={valid_pixel_fraction:.4g}."
            ),
        )
    )
    if calibration.pixel_size_nm is not None:
        relative_difference = (
            abs(calibration.pixel_size_nm - config.optical_pixel_size_nm)
            / config.optical_pixel_size_nm
        )
        if relative_difference > 0.01:
            issues.append(
                PreflightIssue(
                    severity="warning" if not strict_mode else "error",
                    code="calibration_pixel_size_mismatch",
                    field="optical_pixel_size",
                    message=(
                        f"Calibration pixel size {calibration.pixel_size_nm:g} nm differs from "
                        f"config optical_pixel_size={config.optical_pixel_size_nm:g} nm."
                    ),
                )
            )


def _add_output_info(
    event_files: tuple[Path, ...], issues: list[PreflightIssue]
) -> None:
    for event_file in event_files:
        issues.append(
            PreflightIssue(
                severity="info",
                code="resolved_output_folder",
                field="input_folder",
                message=f"{event_file.name} output folder: {event_file.with_suffix('').resolve()}",
            )
        )


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
