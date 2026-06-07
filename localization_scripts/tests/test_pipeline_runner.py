import json

import numpy as np

from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.pipeline_runner import (
    summarize_fit_qc,
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
    assert payload["calibration"] == calibration_metadata


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
