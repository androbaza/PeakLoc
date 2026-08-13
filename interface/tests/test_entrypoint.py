from pathlib import Path

from PeakLocGUI import _loky_command_from_argv, parse_worker_args


def test_loky_command_is_detected_before_gui_argument_parsing():
    command = "from joblib.externals.loky.backend.resource_tracker import main; main(4424, False)"

    assert (
        _loky_command_from_argv(["PeakLoc.exe", "-B", "-S", "-c", command]) == command
    )


def test_normal_gui_arguments_keep_their_existing_parser_contract():
    args = parse_worker_args(["--pipeline-worker", "--config", "run.json"])

    assert args.pipeline_worker is True
    assert args.config == Path("run.json")
    assert _loky_command_from_argv(["PeakLoc.exe", "--pipeline-worker"]) is None
