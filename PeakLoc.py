import argparse
from datetime import datetime
from pathlib import Path

from localization_scripts.config_sweep import run_config_sweep
from localization_scripts.pipeline_config import load_peakloc_config
from localization_scripts.pipeline_runner import run_batch
from localization_scripts.preflight import run_preflight, write_preflight_report

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
