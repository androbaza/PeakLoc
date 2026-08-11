from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from localization_scripts.event_array_processing import raw_events_to_array


@dataclass(frozen=True)
class RateMaps:
    rate_pos: np.ndarray
    rate_neg: np.ndarray
    hot_pixel_mask: np.ndarray
    valid_pixel_mask: np.ndarray


def build_rate_maps(
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    hot_pixel_quantile: float = 0.999,
) -> RateMaps:
    counts_pos = np.zeros(sensor_shape, dtype=np.float64)
    counts_neg = np.zeros(sensor_shape, dtype=np.float64)
    if events.size == 0:
        valid_pixel_mask = np.ones(sensor_shape, dtype=bool)
        hot_pixel_mask = np.zeros(sensor_shape, dtype=bool)
        return RateMaps(counts_pos, counts_neg, hot_pixel_mask, valid_pixel_mask)

    duration_s = _recording_duration_s(events)
    positive = events["p"] > 0
    negative = ~positive
    np.add.at(counts_pos, (events["y"][positive], events["x"][positive]), 1)
    np.add.at(counts_neg, (events["y"][negative], events["x"][negative]), 1)
    rate_pos = counts_pos / duration_s
    rate_neg = counts_neg / duration_s
    total_rate = rate_pos + rate_neg
    threshold = np.quantile(total_rate, hot_pixel_quantile)
    hot_pixel_mask = total_rate > threshold
    valid_pixel_mask = np.ones(sensor_shape, dtype=bool)
    return RateMaps(rate_pos, rate_neg, hot_pixel_mask, valid_pixel_mask)


def write_event_calibration(
    output_path: Path,
    *,
    dark_maps: RateMaps,
    blank_maps: RateMaps,
    pixel_size_nm: float,
    sensor_model: str,
    calibration_id: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hot_pixel_mask = dark_maps.hot_pixel_mask | blank_maps.hot_pixel_mask
    valid_pixel_mask = dark_maps.valid_pixel_mask & blank_maps.valid_pixel_mask
    np.savez(
        output_path,
        dark_rate_pos=dark_maps.rate_pos,
        dark_rate_neg=dark_maps.rate_neg,
        blank_rate_pos=blank_maps.rate_pos,
        blank_rate_neg=blank_maps.rate_neg,
        hot_pixel_mask=hot_pixel_mask,
        valid_pixel_mask=valid_pixel_mask,
        pixel_size_nm=np.asarray(pixel_size_nm),
        sensor_model=np.asarray(sensor_model),
        calibration_id=np.asarray(calibration_id),
    )


def build_event_calibration(
    dark_path: Path,
    blank_path: Path,
    output_path: Path,
    *,
    pixel_size_nm: float,
    sensor_model: str,
    calibration_id: str,
    height: int | None = None,
    width: int | None = None,
    max_events: int = 1_000_000_000,
) -> Path:
    """Build calibration maps from dark and laser-on blank recordings."""
    if not dark_path.is_file():
        raise FileNotFoundError(f"Dark recording does not exist: {dark_path}")
    if not blank_path.is_file():
        raise FileNotFoundError(f"Blank recording does not exist: {blank_path}")
    if (height is None) != (width is None):
        raise ValueError(
            "Sensor height and width must both be provided or both omitted"
        )

    print(f"Reading dark recording: {dark_path}", flush=True)
    dark_events = raw_events_to_array(str(dark_path), max_events=max_events)
    print(f"Read {dark_events.size:,} dark events", flush=True)
    print(f"Reading laser-on blank recording: {blank_path}", flush=True)
    blank_events = raw_events_to_array(str(blank_path), max_events=max_events)
    print(f"Read {blank_events.size:,} blank events", flush=True)
    sensor_shape = resolve_sensor_shape(
        dark_events,
        blank_events,
        height=height,
        width=width,
    )
    print(
        f"Building {sensor_shape[1]} x {sensor_shape[0]} calibration maps", flush=True
    )
    dark_maps = build_rate_maps(dark_events, sensor_shape)
    blank_maps = build_rate_maps(blank_events, sensor_shape)
    write_event_calibration(
        output_path,
        dark_maps=dark_maps,
        blank_maps=blank_maps,
        pixel_size_nm=pixel_size_nm,
        sensor_model=sensor_model,
        calibration_id=calibration_id,
    )
    print(f"Wrote calibration to {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PeakLoc dark and laser-on blank calibration maps."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("calibration_event_model.npz")
    )
    parser.add_argument("--pixel-size-nm", type=float, default=77.0)
    parser.add_argument("--sensor-model", default="unknown")
    parser.add_argument("--calibration-id", default="event-model-calibration")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=1_000_000_000)
    args = parser.parse_args()

    dark_path = Path(input("Dark .raw recording path: ").strip()).expanduser()
    blank_path = Path(
        input("Laser-on blank .raw recording path: ").strip()
    ).expanduser()
    build_event_calibration(
        dark_path,
        blank_path,
        args.output,
        pixel_size_nm=args.pixel_size_nm,
        sensor_model=args.sensor_model,
        calibration_id=args.calibration_id,
        height=args.height,
        width=args.width,
        max_events=args.max_events,
    )


def resolve_sensor_shape(
    dark_events: np.ndarray,
    blank_events: np.ndarray,
    *,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    if height is not None and width is not None:
        return height, width
    events = np.concatenate([dark_events, blank_events])
    if events.size == 0:
        raise ValueError(
            "Provide --height and --width when calibration recordings are empty"
        )
    return int(events["y"].max()) + 1, int(events["x"].max()) + 1


def _recording_duration_s(events: np.ndarray) -> float:
    duration_us = int(events["t"].max()) - int(events["t"].min()) + 1
    return max(duration_us, 1) * 1e-6


if __name__ == "__main__":
    main()
