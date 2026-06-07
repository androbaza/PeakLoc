import json

import pytest

from localization_scripts.pipeline_config import (
    PeakLocConfig,
    load_peakloc_config,
    write_effective_config,
)


def test_load_peakloc_config_from_json_and_environment_override(tmp_path):
    config_path = tmp_path / "peakloc.json"
    config_path.write_text(
        json.dumps(
            {
                "input_folder": "from-config",
                "slice_start": 10,
                "slice_duration": 20,
                "num_cores": 2,
            }
        ),
        encoding="utf-8",
    )

    config = load_peakloc_config(
        config_path,
        environ={
            "PEAKLOC_INPUT_FOLDER": "from-env",
            "PEAKLOC_SLICE_START": "30",
            "PEAKLOC_SLICE_DURATION": "40",
        },
    )

    assert config.input_folder == "from-env"
    assert config.slice_start == 30
    assert config.slice_duration == 40
    assert config.num_cores == 2


def test_load_peakloc_config_uses_root_config_by_default(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"input_folder": "from-default-config", "num_cores": 1}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = load_peakloc_config(environ={})

    assert config.input_folder == "from-default-config"
    assert config.num_cores == 1


def test_peakloc_config_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown PeakLoc config setting"):
        PeakLocConfig.from_mapping({"not_a_setting": 1})


def test_peakloc_config_validates_scientific_parameters():
    with pytest.raises(ValueError, match="dataset_fwhm must be positive"):
        PeakLocConfig.from_mapping({"dataset_fwhm": 0})

    with pytest.raises(ValueError, match="optical_pixel_size must be positive"):
        PeakLocConfig.from_mapping({"optical_pixel_size": 0})

    with pytest.raises(ValueError, match="spline_smooth must be between 0 and 1"):
        PeakLocConfig.from_mapping({"spline_smooth": 1.5})


def test_peakloc_config_validates_boolean_parameters():
    with pytest.raises(ValueError, match="plot_result must be true or false"):
        PeakLocConfig.from_mapping({"plot_result": "yes"})


def test_peakloc_config_validates_event_model_settings():
    with pytest.raises(ValueError, match="fit_model must be"):
        PeakLocConfig.from_mapping({"fit_model": "not-a-model"})

    with pytest.raises(ValueError, match="calibration_path is required"):
        PeakLocConfig.from_mapping({"allow_uncalibrated": False})

    with pytest.raises(ValueError, match="sigma_psf_px must be positive"):
        PeakLocConfig.from_mapping({"sigma_psf_px": 0})

    with pytest.raises(ValueError, match="min_valid_pixels must be positive"):
        PeakLocConfig.from_mapping({"min_valid_pixels": 0})


def test_write_effective_config_is_human_readable_json(tmp_path):
    output_path = tmp_path / "reports" / "settings.json"
    config = PeakLocConfig(input_folder="data", num_cores=1)

    write_effective_config(config, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input_folder"] == "data"
    assert payload["num_cores"] == 1
    assert payload["optical_pixel_size"] == 67.0
    assert payload["fit_model"] == "poisson_joint"
    assert config.optical_pixel_size_nm == 67.0
    assert payload["plot_result"] is True
    assert output_path.read_text(encoding="utf-8").endswith("\n")
