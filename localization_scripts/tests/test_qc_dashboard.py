import json

import numpy as np

from localization_scripts.localization_fitting import localization_qc_dtype
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import RecordingResult, SliceResult
from localization_scripts.qc_dashboard import save_run_qc_dashboard


def test_dashboard_writes_index_summary_and_png(tmp_path):
    events = _events(6)
    input_path = tmp_path / "recording.npy"
    np.save(input_path, events)
    recording = _recording(input_path, events)
    attempted = _localizations(3)
    qc_table = _qc_table(attempted, accepted=[True, False, True])
    localizations = attempted[qc_table["accepted"]]
    config = _config(tmp_path)

    artifacts = save_run_qc_dashboard(
        recording=recording,
        config=config,
        localizations=localizations,
        attempted_localizations=attempted,
        localization_qc=qc_table,
        rois=np.zeros(2, dtype=[("roi", np.uint32, (3, 3))]),
        events=events,
        timestamp="20260608_120000",
    )

    qc_dir = recording.output_folder / "qc"
    assert qc_dir.joinpath("index.html").is_file()
    assert qc_dir.joinpath("run_qc_summary.json").is_file()
    assert qc_dir.joinpath("01_event_density_total.png").is_file()
    assert any(path.suffix == ".png" for path in artifacts)

    summary = json.loads(qc_dir.joinpath("run_qc_summary.json").read_text())
    assert summary["attempted_fit_count"] == 3
    assert summary["accepted_localization_count"] == 2
    assert summary["accepted_from_qc_count"] == 2


def test_dashboard_handles_empty_localization_table_with_warning(tmp_path):
    events = _events(0)
    input_path = tmp_path / "recording.npy"
    np.save(input_path, events)
    recording = _recording(input_path, events)
    empty_locs = _localizations(0)
    empty_qc = np.empty(0, dtype=localization_qc_dtype())

    save_run_qc_dashboard(
        recording=recording,
        config=_config(tmp_path),
        localizations=empty_locs,
        attempted_localizations=empty_locs,
        localization_qc=empty_qc,
        rois=np.zeros(0, dtype=[("roi", np.uint32, (3, 3))]),
        events=events,
        timestamp="20260608_120000",
    )

    summary = json.loads(
        recording.output_folder.joinpath("qc", "run_qc_summary.json").read_text()
    )
    assert "No accepted localizations were produced." in summary["warnings"]
    assert recording.output_folder.joinpath("qc", "04_detection_funnel.png").is_file()


def test_dashboard_counts_attempted_and_accepted_from_qc_table(tmp_path):
    events = _events(3)
    input_path = tmp_path / "recording.npy"
    np.save(input_path, events)
    attempted = _localizations(4)
    qc_table = _qc_table(attempted, accepted=[True, False, False, True])
    recording = _recording(input_path, events)

    save_run_qc_dashboard(
        recording=recording,
        config=_config(tmp_path),
        localizations=attempted[qc_table["accepted"]],
        attempted_localizations=attempted,
        localization_qc=qc_table,
        rois=np.zeros(4, dtype=[("roi", np.uint32, (3, 3))]),
        events=events,
        timestamp="20260608_120000",
    )

    summary = json.loads(
        recording.output_folder.joinpath("qc", "run_qc_summary.json").read_text()
    )
    assert summary["detection_funnel"]["attempted_fits"] == 4
    assert summary["detection_funnel"]["accepted_fits"] == 2
    assert summary["rejection_reasons"]["uncertainty"] == 2


def _config(tmp_path) -> PeakLocConfig:
    return PeakLocConfig(
        input_folder=str(tmp_path),
        sensor_height=16,
        sensor_width=16,
        background_mode="local_only",
        qc_static_dpi=70,
        qc_generate_interactive=False,
        qc_uncertainty_montage_n=2,
    )


def _recording(input_path, events: np.ndarray) -> RecordingResult:
    output_folder = input_path.with_suffix("")
    output_folder.mkdir(parents=True, exist_ok=True)
    return RecordingResult(
        input_file=input_path,
        output_folder=output_folder,
        event_count=int(events.size),
        time_min=int(events["t"].min()) if events.size else None,
        time_max=int(events["t"].max()) if events.size else None,
        slice_results=[
            SliceResult(
                time_slice=100,
                event_count=int(events.size),
                unique_peak_count=2,
                roi_count=2,
                localization_count=1,
                elapsed_seconds=0.1,
            )
        ],
    )


def _events(count: int) -> np.ndarray:
    events = np.zeros(
        count,
        dtype=[("x", np.uint16), ("y", np.uint16), ("p", np.int8), ("t", np.uint64)],
    )
    if count:
        events["x"] = np.arange(count) % 16
        events["y"] = np.arange(count) % 16
        events["p"] = np.arange(count) % 2
        events["t"] = np.arange(count) * 10
    return events


def _localizations(count: int) -> np.ndarray:
    roi_shape = (5, 5)
    localizations = np.zeros(
        count,
        dtype=[
            ("id", np.uint64),
            ("x", np.float64),
            ("y", np.float64),
            ("t_peak", np.float64),
            ("sub_x", np.float64),
            ("sub_y", np.float64),
            ("sigma_x", np.float64),
            ("sigma_y", np.float64),
            ("cov_xy", np.float64),
            ("A_pos", np.float64),
            ("A_neg", np.float64),
            ("bg_pos_local", np.float64),
            ("bg_neg_local", np.float64),
            ("E_total", np.uint64),
            ("E_total_n", np.uint64),
            ("nll_per_event", np.float64),
            ("fit_cond", np.float64),
            ("fit_success", np.bool_),
            ("hot_pixel_count", np.uint32),
            ("roi", np.uint32, roi_shape),
            ("roi_n", np.uint32, roi_shape),
        ],
    )
    if count:
        localizations["id"] = np.arange(count)
        localizations["x"] = np.arange(count) + 1.0
        localizations["y"] = np.arange(count) + 2.0
        localizations["t_peak"] = np.arange(count) * 10
        localizations["sub_x"] = 2.0
        localizations["sub_y"] = 2.0
        localizations["sigma_x"] = np.linspace(0.2, 0.6, count)
        localizations["sigma_y"] = np.linspace(0.2, 0.6, count)
        localizations["cov_xy"] = 0.0
        localizations["A_pos"] = 5.0
        localizations["A_neg"] = 4.0
        localizations["bg_pos_local"] = 0.5
        localizations["bg_neg_local"] = 0.5
        localizations["E_total"] = 10
        localizations["E_total_n"] = 8
        localizations["nll_per_event"] = 1.5
        localizations["fit_cond"] = 10.0
        localizations["fit_success"] = True
        localizations["roi"] = 1
        localizations["roi_n"] = 2
    return localizations


def _qc_table(localizations: np.ndarray, accepted: list[bool]) -> np.ndarray:
    qc_table = np.zeros(localizations.size, dtype=localization_qc_dtype())
    qc_table["id"] = localizations["id"]
    qc_table["accepted"] = accepted
    qc_table["fit_success"] = True
    qc_table["finite_position"] = True
    qc_table["finite_uncertainty"] = True
    qc_table["positive_uncertainty"] = True
    qc_table["fit_cond_ok"] = True
    qc_table["valid_pixels_ok"] = True
    qc_table["uncertainty_px"] = localizations["sigma_x"]
    qc_table["uncertainty_nm"] = localizations["sigma_x"] * 67.0
    qc_table["uncertainty_ok"] = accepted
    qc_table["fit_cond"] = localizations["fit_cond"]
    qc_table["valid_pixel_count"] = 25
    qc_table["nll_per_event"] = localizations["nll_per_event"]
    qc_table["E_total"] = localizations["E_total"]
    qc_table["E_total_n"] = localizations["E_total_n"]
    qc_table["primary_rejection_reason"] = np.where(
        qc_table["accepted"], "accepted", "uncertainty"
    )
    return qc_table
