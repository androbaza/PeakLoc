import csv

import numpy as np
import pytest

from localization_scripts.config_sweep import (
    compute_synthetic_truth_metrics,
    load_sweep_spec,
    run_config_sweep,
)
from localization_scripts.localization_fitting import localization_qc_dtype
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import RecordingResult, SliceResult


def test_sweep_with_two_parameter_values_creates_two_result_rows(tmp_path):
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text('{"prominence": [8, 10]}', encoding="utf-8")

    artifacts = run_config_sweep(
        PeakLocConfig(input_folder=str(tmp_path), qc_enabled=False),
        sweep_path,
        output_dir=tmp_path / "sweep",
        runner=_runner(tmp_path),
    )

    csv_path = tmp_path / "sweep" / "sweep_results.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert [row["prominence"] for row in rows] == ["8", "10"]
    assert csv_path in artifacts
    assert (tmp_path / "sweep" / "pareto_localizations_vs_uncertainty.png").is_file()
    assert (tmp_path / "sweep" / "rejection_reason_heatmap.png").is_file()
    assert (tmp_path / "sweep" / "parameter_effects.html").is_file()


def test_invalid_sweep_field_fails_before_running(tmp_path):
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text('{"not_a_config_field": [1]}', encoding="utf-8")
    called = False

    def runner(config):
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="Unknown sweep config field"):
        run_config_sweep(
            PeakLocConfig(input_folder=str(tmp_path)),
            sweep_path,
            output_dir=tmp_path / "sweep",
            runner=runner,
        )
    assert not called


def test_synthetic_truth_metrics_are_computed_when_truth_is_passed():
    localizations = np.asarray(
        [(0.0, 0.0), (10.0, 10.0)],
        dtype=[("x", np.float64), ("y", np.float64)],
    )
    truth = np.asarray([(0.5, 0.0)], dtype=[("x", np.float64), ("y", np.float64)])

    metrics = compute_synthetic_truth_metrics(localizations, truth, max_distance_px=1.0)

    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["false_discovery_rate"] == 0.5
    assert metrics["median_spatial_error_px"] == 0.5


def test_load_sweep_spec_requires_non_empty_lists(tmp_path):
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text('{"prominence": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list"):
        load_sweep_spec(sweep_path)


def _runner(tmp_path):
    def run(config: PeakLocConfig):
        output_folder = tmp_path / f"run_{config.prominence:g}"
        output_folder.mkdir(parents=True, exist_ok=True)
        qc_table = np.zeros(2, dtype=localization_qc_dtype())
        qc_table["id"] = [0, 1]
        qc_table["accepted"] = [True, False]
        qc_table["fit_success"] = True
        qc_table["finite_position"] = True
        qc_table["finite_uncertainty"] = True
        qc_table["positive_uncertainty"] = True
        qc_table["fit_cond_ok"] = True
        qc_table["valid_pixels_ok"] = True
        qc_table["uncertainty_px"] = [0.2, 0.6]
        qc_table["uncertainty_nm"] = qc_table["uncertainty_px"] * 67.0
        qc_table["uncertainty_ok"] = [True, False]
        qc_table["fit_cond"] = 10.0
        qc_table["valid_pixel_count"] = 25
        qc_table["nll_per_event"] = [1.1, 1.4]
        qc_table["E_total"] = [10, 12]
        qc_table["E_total_n"] = [8, 9]
        qc_table["primary_rejection_reason"] = ["accepted", "uncertainty"]
        qc_path = output_folder / "localization_qc.npy"
        np.save(qc_path, qc_table)
        recording = RecordingResult(
            input_file=tmp_path / "recording.npy",
            output_folder=output_folder,
            event_count=20,
            time_min=0,
            time_max=100,
            slice_results=[
                SliceResult(
                    time_slice=100,
                    event_count=20,
                    unique_peak_count=3,
                    roi_count=2,
                    localization_count=1,
                    elapsed_seconds=0.1,
                    rejected_localization_count=1,
                )
            ],
            artifacts=[qc_path],
            elapsed_seconds=0.2,
        )
        return [recording]

    return run
