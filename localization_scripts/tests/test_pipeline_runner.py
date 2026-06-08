import json

import numpy as np

from localization_scripts import pipeline_runner
from localization_scripts.calibration import NullCalibration
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import (
    RecordingResult,
    calibration_to_metadata,
    save_processed_plots,
    summarize_fit_qc,
    write_structured_array_csv,
    write_run_report,
    write_effective_run_settings,
)


def test_write_effective_run_settings_includes_calibration_metadata(tmp_path):
    output_path = tmp_path / "reports" / "settings.json"
    calibration_metadata: dict[str, object] = {
        "calibration_id": "none",
        "calibrated": False,
    }

    write_effective_run_settings(
        PeakLocConfig(input_folder="data", num_cores=1),
        calibration_metadata,
        output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input_folder"] == "data"
    assert payload["sensor_height"] == 720
    assert payload["sensor_width"] == 1280
    assert payload["calibration"] == calibration_metadata


def test_null_calibration_metadata_uses_configured_sensor_shape():
    config = PeakLocConfig(sensor_height=720, sensor_width=1280)
    calibration = NullCalibration(config.sensor_shape)

    metadata = calibration_to_metadata(calibration)

    assert metadata["sensor_shape"] == [720, 1280]


def test_process_recording_loads_calibration_with_configured_sensor_shape(
    tmp_path, monkeypatch
):
    events = np.zeros(
        1,
        dtype=[("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")],
    )
    events["x"] = [10]
    events["y"] = [20]
    events["p"] = [1]
    events["t"] = [100]
    input_path = tmp_path / "recording.npy"
    np.save(input_path, events)
    observed_shapes = []

    def load_calibration_spy(calibration_path, sensor_shape, *, allow_uncalibrated):
        observed_shapes.append(sensor_shape)
        return NullCalibration(sensor_shape)

    monkeypatch.setattr(pipeline_runner, "load_calibration", load_calibration_spy)
    monkeypatch.setattr(pipeline_runner, "process_time_slice", lambda *args: None)
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        sensor_height=720,
        sensor_width=1280,
    )

    pipeline_runner.process_recording(input_path, config, "20260607_120000")

    assert observed_shapes == [(720, 1280)]


def test_process_recording_offsets_slice_localization_ids_without_duplicates(
    tmp_path, monkeypatch
):
    events = np.zeros(
        3,
        dtype=[("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")],
    )
    events["x"] = [10, 10, 10]
    events["y"] = [20, 20, 20]
    events["p"] = [1, 1, 1]
    events["t"] = [10, 110, 210]
    input_path = tmp_path / "recording.npy"
    np.save(input_path, events)
    output_folder = input_path.with_suffix("")
    temp_folder = output_folder / "temp_files"
    temp_folder.mkdir(parents=True)
    loc_dtype = [("id", np.uint64), ("x", np.float64), ("y", np.float64)]
    qc_dtype = [
        ("id", np.uint64),
        ("accepted", np.bool_),
        ("primary_rejection_reason", "U64"),
    ]
    roi_dtype = [("roi", np.uint32, (3, 3))]
    np.save(
        temp_folder / "localizations_time_slice_100.npy",
        np.asarray([(0, 1.0, 2.0), (1, 3.0, 4.0)], dtype=loc_dtype),
    )
    np.save(
        temp_folder / "attempted_localizations_time_slice_100.npy",
        np.asarray([(0, 1.0, 2.0), (1, 3.0, 4.0)], dtype=loc_dtype),
    )
    np.save(
        temp_folder / "localizations_time_slice_200.npy",
        np.asarray([(0, 5.0, 6.0), (1, 7.0, 8.0)], dtype=loc_dtype),
    )
    np.save(
        temp_folder / "attempted_localizations_time_slice_200.npy",
        np.asarray([(0, 5.0, 6.0), (1, 7.0, 8.0)], dtype=loc_dtype),
    )
    np.save(
        temp_folder / "localization_qc_time_slice_100.npy",
        np.asarray([(0, True, "accepted"), (1, False, "fit_failed")], dtype=qc_dtype),
    )
    np.save(
        temp_folder / "localization_qc_time_slice_200.npy",
        np.asarray([(0, True, "accepted"), (1, False, "uncertainty")], dtype=qc_dtype),
    )
    np.save(
        temp_folder / "rois_time_slice_100.npy",
        np.zeros(1, dtype=roi_dtype),
    )
    np.save(
        temp_folder / "rois_time_slice_200.npy",
        np.zeros(1, dtype=roi_dtype),
    )

    monkeypatch.setattr(
        pipeline_runner,
        "load_calibration",
        lambda calibration_path, sensor_shape, *, allow_uncalibrated: NullCalibration(
            sensor_shape
        ),
    )
    monkeypatch.setattr(pipeline_runner, "process_time_slice", lambda *args: None)
    monkeypatch.setattr(pipeline_runner, "save_processed_plots", lambda *args: [])
    config = PeakLocConfig(
        input_folder=str(tmp_path),
        slice_duration=100,
        cleanup_temp_outputs=False,
    )

    pipeline_runner.process_recording(input_path, config, "20260607_120000")

    output_path = output_folder / (
        f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    localizations = np.load(output_path)
    assert list(localizations["id"]) == [0, 1, 2, 3]
    attempted_output_path = output_folder / (
        f"attempted_localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    attempted_localizations = np.load(attempted_output_path)
    assert list(attempted_localizations["id"]) == [0, 1, 2, 3]
    qc_output_path = output_folder / (
        f"localization_qc_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}.npy"
    )
    localization_qc = np.load(qc_output_path)
    assert list(localization_qc["id"]) == [0, 1, 2, 3]
    assert qc_output_path.with_suffix(".csv").is_file()


def test_summarize_fit_qc_handles_poisson_fields():
    localizations = np.zeros(
        (2,),
        dtype=[
            ("fit_success", np.bool_),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("nll_per_event", np.float64),
            ("hot_pixel_count", np.uint32),
            ("valid_pixel_count", np.uint32),
        ],
    )
    localizations["fit_success"] = [True, False]
    localizations["sigma_x"] = [0.3, 0.4]
    localizations["sigma_y"] = [0.4, 0.3]
    localizations["nll_per_event"] = [1.0, 2.0]
    localizations["hot_pixel_count"] = [1, 0]
    localizations["valid_pixel_count"] = [100, 100]

    summary = summarize_fit_qc(localizations, roi_count=3)

    assert summary["fit_success_fraction"] == 0.5
    assert summary["median_uncertainty_px"] == 0.5
    assert summary["median_nll_per_event"] == 1.5
    assert summary["hot_pixel_fraction"] == 0.005
    assert summary["rejected_localization_count"] == 1

    summary = summarize_fit_qc(
        localizations, roi_count=3, filtered_localization_count=1
    )
    assert summary["fit_success_fraction"] == 0.5
    assert summary["rejected_localization_count"] == 2


def test_write_structured_array_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "localization_qc.csv"
    array = np.asarray(
        [(1, True, "accepted"), (2, False, "fit_failed")],
        dtype=[
            ("id", np.uint64),
            ("accepted", np.bool_),
            ("primary_rejection_reason", "U64"),
        ],
    )

    write_structured_array_csv(array, path)

    assert path.read_text(encoding="utf-8").splitlines() == [
        "id,accepted,primary_rejection_reason",
        "1,True,accepted",
        "2,False,fit_failed",
    ]


def test_save_processed_plots_keeps_attempted_montages_when_no_fits_are_accepted(
    tmp_path, monkeypatch
):
    loc_dtype = [("id", np.uint64)]
    localizations = np.empty(0, dtype=loc_dtype)
    attempted = np.zeros(1, dtype=loc_dtype)
    qc_table = np.zeros(1, dtype=[("id", np.uint64), ("accepted", np.bool_)])
    montage_path = tmp_path / "figures" / "uncertainty_highest_36_combined.png"
    calls = []

    def save_montage_spy(*args, **kwargs):
        calls.append((args, kwargs))
        return [montage_path]

    monkeypatch.setattr(pipeline_runner, "save_uncertainty_montages", save_montage_spy)

    artifacts = save_processed_plots(
        localizations,
        tmp_path,
        tmp_path / "localizations.npy",
        PeakLocConfig(plot_result=True),
        "20260608_120000",
        attempted,
        qc_table,
    )

    assert artifacts == [montage_path]
    assert calls


def test_write_run_report_includes_peak_interpolation_cutoff(tmp_path):
    recording = RecordingResult(
        input_file=tmp_path / "recording.npy",
        output_folder=tmp_path / "recording",
        event_count=0,
        time_min=None,
        time_max=None,
    )
    config = PeakLocConfig(peak_min_event_count=7)

    report_path = write_run_report(recording, config, "20260607_120000")

    assert "- Peak interpolation min events: `7`" in report_path.read_text(
        encoding="utf-8"
    )


def test_write_run_report_includes_scientific_validation_summary(tmp_path):
    output_folder = tmp_path / "recording"
    qc_dir = output_folder / "qc"
    qc_dir.mkdir(parents=True)
    summary_path = qc_dir / "run_qc_summary.json"
    frc_path = qc_dir / "frc_summary.json"
    preflight_path = qc_dir / "preflight_report.md"
    index_path = qc_dir / "index.html"
    summary_path.write_text(
        json.dumps(
            {
                "attempted_fit_count": 4,
                "accepted_from_qc_count": 2,
                "detection_funnel": {
                    "events_loaded": 10,
                    "peak_candidates": 3,
                    "rois_generated": 4,
                },
                "median_uncertainty_px": 0.2,
                "median_uncertainty_nm": 13.4,
                "p90_uncertainty_px": 0.4,
                "p90_uncertainty_nm": 26.8,
                "rejection_reasons": {"accepted": 2, "uncertainty": 2},
                "warnings": ["No accepted localizations were produced."],
            }
        ),
        encoding="utf-8",
    )
    frc_path.write_text(
        json.dumps(
            {
                "resolution_nm": 45.0,
                "threshold": 1 / 7,
                "warning": None,
                "drift_method": "binned_median",
            }
        ),
        encoding="utf-8",
    )
    preflight_path.write_text("- Status: `passed`\n", encoding="utf-8")
    index_path.write_text("<html></html>\n", encoding="utf-8")
    recording = RecordingResult(
        input_file=tmp_path / "recording.npy",
        output_folder=output_folder,
        event_count=10,
        time_min=0,
        time_max=100,
        calibration_metadata={"calibration_id": "none", "calibrated": False},
        artifacts=[summary_path, frc_path, preflight_path, index_path],
    )

    report_path = write_run_report(
        recording,
        PeakLocConfig(background_mode="local_only", calibration_path=None),
        "20260608_120000",
    )
    text = report_path.read_text(encoding="utf-8")

    assert "## Scientific Validation" in text
    assert "- Preflight status: `passed`" in text
    assert "- Attempted fits: `4`" in text
    assert "- Accepted fits: `2`" in text
    assert "- Rejection reasons: `accepted=2, uncertainty=2`" in text
    assert "- FRC resolution: `45 nm`" in text
    assert "background_mode=local_only and calibration_path=None" in text
