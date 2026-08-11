from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PeakLoc desktop application")
    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument("--pipeline-worker", action="store_true")
    worker_group.add_argument("--calibration-worker", type=Path)
    worker_group.add_argument("--slice-worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_worker_args()
    if (
        args.slice_worker is not None
        or args.pipeline_worker
        or args.calibration_worker is not None
    ) and not getattr(sys, "frozen", False):
        from PeakLoc import configure_worker_environment

        configure_worker_environment()
    if args.slice_worker is not None:
        from localization_scripts.pipeline_runner import run_serialized_slice_worker

        run_serialized_slice_worker(args.slice_worker)
        return
    if args.pipeline_worker:
        if args.config is None:
            raise ValueError("--config is required for a pipeline worker")
        from interface.operations import run_pipeline_worker

        run_pipeline_worker(args.config, preflight_only=args.preflight_only)
        return
    if args.calibration_worker is not None:
        from interface.operations import run_calibration_worker

        run_calibration_worker(args.calibration_worker)
        return

    from interface.app import launch

    launch()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PeakLoc error: {error}", file=sys.stderr, flush=True)
        raise
