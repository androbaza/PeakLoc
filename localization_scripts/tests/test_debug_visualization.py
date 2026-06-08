from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from localization_scripts.debug_visualization import (
    DebugVisualizationConfig,
    TruthPoint,
    build_interactive_spacetime_point_cloud_figure,
    match_truth_to_localizations,
    prepare_debug_output_dir,
    save_synthetic_localization_debug_artifacts,
    save_xy_summary_figure,
)


EVENT_DTYPE = np.dtype(
    [
        ("x", np.uint16),
        ("y", np.uint16),
        ("p", np.int8),
        ("t", np.uint64),
    ]
)

LOCALIZATION_DTYPE = np.dtype(
    [
        ("id", np.uint64),
        ("t_peak", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("E_total", np.uint64),
        ("E_total_n", np.uint64),
        ("sigma_x", np.float64),
        ("sigma_y", np.float64),
        ("cov_xy", np.float64),
        ("fit_success", np.bool_),
        ("fit_cond", np.float64),
    ]
)

ATTEMPTED_DTYPE = LOCALIZATION_DTYPE


def test_prepare_debug_output_dir_overwrites_previous_debug_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "scenario"
    output_dir.mkdir()
    (output_dir / ".peakloc_debug_artifacts").touch()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    prepare_debug_output_dir(output_dir, overwrite=True)

    assert not (output_dir / "stale.txt").exists()
    assert (output_dir / ".peakloc_debug_artifacts").is_file()


def test_prepare_debug_output_dir_refuses_to_delete_unmarked_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "scenario"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_debug_output_dir(output_dir, overwrite=True)


def test_match_truth_to_localizations_selects_nearest_valid_spacetime_match() -> None:
    truth = [
        TruthPoint(x_px=10.0, y_px=12.0, peak_us=100, label="truth_0"),
        TruthPoint(x_px=30.0, y_px=35.0, peak_us=500, label="truth_1"),
    ]
    localizations = np.asarray(
        [
            _localization_row(0, x=9.9, y=12.1, t_peak=102),
            _localization_row(1, x=28.0, y=35.0, t_peak=500),
            _localization_row(2, x=30.1, y=35.1, t_peak=510),
        ],
        dtype=LOCALIZATION_DTYPE,
    )

    matches = match_truth_to_localizations(
        truth,
        localizations,
        max_spatial_error_px=1.0,
        max_abs_time_error_us=20,
        min_events_per_polarity=100,
    )

    assert [match.localization_index for match in matches] == [0, 2]
    assert matches[0].passed_spatial
    assert matches[0].passed_time
    assert matches[0].passed_event_counts
    assert matches[1].spatial_error_px == pytest.approx(np.sqrt(0.02))


def test_save_synthetic_localization_debug_artifacts_writes_expected_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "debug"
    events = np.asarray(
        [
            (8, 9, 1, 80),
            (10, 12, 1, 90),
            (10, 12, 0, 110),
            (13, 14, 0, 120),
        ],
        dtype=EVENT_DTYPE,
    )
    localizations = np.asarray(
        [_localization_row(0, x=10.1, y=12.1, t_peak=101)],
        dtype=LOCALIZATION_DTYPE,
    )
    truth = [TruthPoint(x_px=10.0, y_px=12.0, peak_us=100, label="truth_0")]

    result = save_synthetic_localization_debug_artifacts(
        events=events,
        localizations=localizations,
        truth=truth,
        config=DebugVisualizationConfig(
            output_dir=output_dir,
            scenario_name="tiny",
            sensor_shape=(24, 24),
            optical_pixel_size_nm=67.0,
            max_spatial_error_px=1.0,
            max_abs_time_error_us=20,
            min_events_per_polarity=100,
            max_events_for_interactive=10,
        ),
        test_status="pre_assertion",
    )

    assert result.match_table_path == output_dir / "matches.csv"
    assert (output_dir / "matches.csv").is_file()
    assert (output_dir / "matches.json").is_file()
    assert (output_dir / "debug_report.md").is_file()
    assert (output_dir / "01_xy_detection_summary.png").is_file()
    assert (output_dir / "06_spacetime_point_cloud.html").is_file()


def test_static_summary_uses_xy_overlay_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scatter_calls: list[tuple[np.ndarray, np.ndarray]] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def scatter_spy(self, x, y, *args, **kwargs):
        scatter_calls.append((np.asarray(x), np.asarray(y)))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", scatter_spy)
    truth = [TruthPoint(x_px=3.0, y_px=7.0, peak_us=100, label="truth_0")]
    localizations = np.asarray(
        [_localization_row(0, x=4.0, y=8.0, t_peak=100)],
        dtype=LOCALIZATION_DTYPE,
    )
    matches = match_truth_to_localizations(
        truth,
        localizations,
        max_spatial_error_px=2.0,
        max_abs_time_error_us=10,
    )
    config = DebugVisualizationConfig(
        output_dir=tmp_path,
        scenario_name="xy",
        sensor_shape=(12, 12),
        optical_pixel_size_nm=67.0,
        max_spatial_error_px=2.0,
        max_abs_time_error_us=10,
        save_pdf=False,
        save_svg=False,
    )

    save_xy_summary_figure(
        events=None,
        density_images=(
            np.zeros((12, 12), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
        ),
        localizations=localizations,
        truth=truth,
        matches=matches,
        rois=None,
        metrics={
            "scenario_name": "xy",
            "matched_count": 1,
            "expected_count": 1,
            "max_spatial_error_px": 1.0,
            "max_spatial_error_nm": 67.0,
            "max_time_error_us": 0.0,
            "median_uncertainty_px": 0.2,
            "total_events": 0,
            "roi_count": 0,
            "localization_count": 1,
            "attempted_localization_count": 1,
        },
        status="PASS",
        config=config,
    )

    assert any(
        x_values.tolist() == [3.0] and y_values.tolist() == [7.0]
        for x_values, y_values in scatter_calls
    )
    assert any(
        x_values.tolist() == [4.0] and y_values.tolist() == [8.0]
        for x_values, y_values in scatter_calls
    )


def test_static_summary_overlays_failed_and_rejected_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scatter_calls: list[tuple[str | None, str | None, np.ndarray, np.ndarray]] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def scatter_spy(self, x, y, *args, **kwargs):
        scatter_calls.append(
            (
                kwargs.get("label"),
                kwargs.get("marker"),
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
            )
        )
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", scatter_spy)
    truth = [TruthPoint(x_px=3.0, y_px=7.0, peak_us=100, label="truth_0")]
    localizations = np.asarray(
        [_localization_row(0, x=4.0, y=8.0, t_peak=100)],
        dtype=LOCALIZATION_DTYPE,
    )
    attempted = np.asarray(
        [
            _localization_row(0, x=4.0, y=8.0, t_peak=100),
            _localization_row(1, x=5.0, y=9.0, t_peak=105, fit_success=False),
            _localization_row(2, x=6.0, y=10.0, t_peak=110),
        ],
        dtype=ATTEMPTED_DTYPE,
    )
    matches = match_truth_to_localizations(
        truth,
        localizations,
        max_spatial_error_px=2.0,
        max_abs_time_error_us=10,
    )
    config = DebugVisualizationConfig(
        output_dir=tmp_path,
        scenario_name="attempts",
        sensor_shape=(12, 12),
        optical_pixel_size_nm=67.0,
        max_spatial_error_px=2.0,
        max_abs_time_error_us=10,
        save_pdf=False,
        save_svg=False,
    )

    save_xy_summary_figure(
        events=None,
        density_images=(
            np.zeros((12, 12), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
            np.zeros((12, 12), dtype=np.float32),
        ),
        localizations=localizations,
        attempted_localizations=attempted,
        truth=truth,
        matches=matches,
        rois=None,
        metrics={
            "scenario_name": "attempts",
            "matched_count": 1,
            "expected_count": 1,
            "max_spatial_error_px": 1.0,
            "max_spatial_error_nm": 67.0,
            "max_time_error_us": 0.0,
            "median_uncertainty_px": 0.2,
            "total_events": 0,
            "roi_count": 0,
            "localization_count": 1,
            "attempted_localization_count": 3,
        },
        status="PASS",
        config=config,
    )

    failed = [
        (x_values, y_values)
        for label, marker, x_values, y_values in scatter_calls
        if label == "failed fit attempt" and marker == "^"
    ]
    rejected = [
        (x_values, y_values)
        for label, marker, x_values, y_values in scatter_calls
        if label == "filtered/rejected attempt" and marker == "s"
    ]
    assert failed and failed[0][0].tolist() == [5.0]
    assert failed[0][1].tolist() == [9.0]
    assert rejected and rejected[0][0].tolist() == [6.0]
    assert rejected[0][1].tolist() == [10.0]


def test_spacetime_event_traces_are_markers_not_lines() -> None:
    events = np.asarray(
        [
            (8, 9, 1, 80),
            (10, 12, 1, 90),
            (10, 12, 0, 110),
            (13, 14, 0, 120),
        ],
        dtype=EVENT_DTYPE,
    )
    localizations = np.asarray(
        [_localization_row(0, x=10.1, y=12.1, t_peak=101)],
        dtype=LOCALIZATION_DTYPE,
    )
    truth = [TruthPoint(x_px=10.0, y_px=12.0, peak_us=100, label="truth_0")]
    matches = match_truth_to_localizations(
        truth,
        localizations,
        max_spatial_error_px=1.0,
        max_abs_time_error_us=20,
    )
    config = DebugVisualizationConfig(
        output_dir=Path("unused"),
        scenario_name="tiny",
        sensor_shape=(24, 24),
        optical_pixel_size_nm=67.0,
        max_spatial_error_px=1.0,
        max_abs_time_error_us=20,
    )

    fig = build_interactive_spacetime_point_cloud_figure(
        events=events,
        localizations=localizations,
        truth=truth,
        matches=matches,
        config=config,
    )

    event_traces = [trace for trace in fig.data if "events" in str(trace.name)]
    assert event_traces
    assert all(trace.mode == "markers" for trace in event_traces)
    assert not any(
        trace.mode == "lines" and getattr(trace, "visible", True) is True
        for trace in fig.data
    )


def _localization_row(
    loc_id: int,
    *,
    x: float,
    y: float,
    t_peak: float,
    fit_success: bool = True,
) -> tuple[object, ...]:
    return (
        loc_id,
        t_peak,
        x,
        y,
        500,
        450,
        0.2,
        0.3,
        0.01,
        fit_success,
        100.0,
    )
