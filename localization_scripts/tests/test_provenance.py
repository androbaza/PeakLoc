import csv
import json

import numpy as np

from localization_scripts.localization_fitting import localization_qc_dtype
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import RecordingResult
from localization_scripts.provenance import save_portable_outputs
from localization_scripts.preflight import effective_config_hash


def test_portable_outputs_write_metadata_and_csv_row_counts(tmp_path):
    recording = RecordingResult(
        input_file=tmp_path / "recording.npy",
        output_folder=tmp_path / "recording",
        event_count=2,
        time_min=10,
        time_max=20,
        calibration_metadata={"calibration_id": "none", "calibrated": False},
    )
    accepted = _localizations(2)
    attempted = _localizations(3)
    qc_table = np.zeros(3, dtype=localization_qc_dtype())
    qc_table["id"] = [0, 1, 2]
    qc_table["accepted"] = [True, True, False]

    artifacts = save_portable_outputs(
        recording=recording,
        config=PeakLocConfig(input_folder=str(tmp_path)),
        accepted_localizations=accepted,
        attempted_localizations=attempted,
        localization_qc=qc_table,
        timestamp="20260608_120000",
    )

    assert recording.output_folder.joinpath("run_metadata.json").is_file()
    assert recording.output_folder.joinpath("software_versions.json").is_file()
    assert recording.output_folder.joinpath("config_effective.json").is_file()
    assert recording.output_folder.joinpath("config_hash.txt").is_file()
    assert recording.output_folder.joinpath("localizations_accepted.csv") in artifacts
    assert _csv_row_count(recording.output_folder / "localizations_accepted.csv") == 2

    metadata = json.loads(
        recording.output_folder.joinpath("run_metadata.json").read_text()
    )
    assert metadata["input_event_count"] == 2
    assert metadata["calibration_id"] == "none"


def test_config_hash_changes_when_config_field_changes():
    first = PeakLocConfig(prominence=8.0)
    second = PeakLocConfig(prominence=9.0)

    assert effective_config_hash(first) != effective_config_hash(second)


def _localizations(count: int) -> np.ndarray:
    localizations = np.zeros(
        count,
        dtype=[
            ("id", np.uint64),
            ("x", np.float64),
            ("y", np.float64),
            ("roi", np.uint32, (3, 3)),
        ],
    )
    localizations["id"] = np.arange(count)
    localizations["x"] = np.arange(count)
    localizations["y"] = np.arange(count)
    return localizations


def _csv_row_count(path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))
