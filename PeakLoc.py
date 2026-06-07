import argparse
from pathlib import Path

from localization_scripts.pipeline_config import load_peakloc_config
from localization_scripts.pipeline_runner import run_batch

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_peakloc_config(args.config)
    run_batch(config)


if __name__ == "__main__":
    main()
