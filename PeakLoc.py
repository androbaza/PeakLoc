import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import sysconfig


def _remove_sys_path_entry(entry: str) -> None:
    while entry in sys.path:
        sys.path.remove(entry)


def configure_worker_environment() -> None:
    """Keep the parent and spawned workers isolated to the active Pixi environment."""
    purelib = sysconfig.get_paths().get("purelib")
    if purelib is None:
        return

    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    for entry in inherited_pythonpath.split(os.pathsep):
        if entry:
            _remove_sys_path_entry(entry)
    for entry in tuple(sys.path):
        if entry != purelib and Path(entry).name in {"site-packages", "dist-packages"}:
            _remove_sys_path_entry(entry)

    project_root = str(Path(__file__).resolve().parent)
    _remove_sys_path_entry(project_root)
    _remove_sys_path_entry(purelib)
    sys.path[:0] = [project_root, purelib]
    os.environ["PYTHONPATH"] = os.pathsep.join([project_root, purelib])
    os.environ["PYTHONNOUSERSITE"] = "1"


configure_worker_environment()

from localization_scripts.config_sweep import run_config_sweep  # noqa: E402
from localization_scripts.pipeline_config import load_peakloc_config  # noqa: E402
from localization_scripts.pipeline_runner import run_batch  # noqa: E402
from localization_scripts.preflight import (  # noqa: E402
    run_preflight,
    write_preflight_report,
)

"""
if the system complains about memory, run the following command:
sudo echo 1 > /proc/sys/vm/overcommit_memory
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PeakLoc localization pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON PeakLoc configuration file",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Write a preflight report before processing and continue only if it passes",
    )
    parser.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Run preflight in publication-oriented strict mode",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write a preflight report and exit without processing recordings",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Path to a JSON parameter sweep specification",
    )
    return parser.parse_args()


def main() -> None:
    configure_worker_environment()
    args = parse_args()
    config = load_peakloc_config(args.config)
    if args.preflight or args.strict_preflight or args.preflight_only:
        report = run_preflight(
            config,
            config_path=args.config,
            strict_mode=args.strict_preflight,
        )
        report_path = (
            Path("reports") / f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        write_preflight_report(report, report_path)
        if report.has_errors:
            raise SystemExit(1)
        if args.preflight_only:
            return
    if args.sweep is not None:
        run_config_sweep(
            config,
            args.sweep,
            preflight=args.preflight or args.strict_preflight,
            strict_preflight=args.strict_preflight,
        )
        return
    run_batch(config)


if __name__ == "__main__":
    main()
