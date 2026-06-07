import json

import numpy as np

from localization_scripts import pipeline_runner
from localization_scripts.calibration import NullCalibration
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import (
    RecordingResult,
    calibration_to_metadata,
    summarize_fit_qc,
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
    roi_dtype = [("roi", np.uint32, (3, 3))]
    np.save(
        temp_folder / "localizations_time_slice_100.npy",
        np.asarray([(0, 1.0, 2.0), (1, 3.0, 4.0)], dtype=loc_dtype),
    )
    np.save(
        temp_folder / "localizations_time_slice_200.npy",
        np.asarray([(0, 5.0, 6.0), (1, 7.0, 8.0)], dtype=loc_dtype),
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
