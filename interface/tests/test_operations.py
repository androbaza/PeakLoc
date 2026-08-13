from loguru import logger

from interface import operations


def test_application_log_path_is_next_to_frozen_executable(tmp_path, monkeypatch):
    executable = tmp_path / "PeakLoc.exe"
    monkeypatch.setattr(operations.sys, "frozen", True, raising=False)
    monkeypatch.setattr(operations.sys, "executable", str(executable))

    assert operations.application_log_path() == tmp_path / "PeakLoc.log"


def test_configure_logging_writes_a_source_run_log(tmp_path, monkeypatch):
    log_path = tmp_path / "PeakLoc.log"
    monkeypatch.setattr(operations, "application_log_path", lambda: log_path)
    monkeypatch.setattr(operations, "_LOGGING_SINK_ID", None)

    operations.configure_logging()
    logger.info("focused logging check")
