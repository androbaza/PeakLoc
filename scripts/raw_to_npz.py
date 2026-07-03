from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import numpy as np

from scripts.raw_to_video import (
    DEFAULT_MAX_EVENTS_BUFFER,
    RawEventReader,
    RawReaderFactory,
    discover_raw_paths,
    open_raw_reader,
)


DEFAULT_READ_WINDOW_US = 10_000_000
ARRAY_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RawArrayConversionResult:
    raw_path: Path
    array_path: Path
    event_count: int
    sensor_shape: tuple[int, int]
    time_start_us: int | None
    time_end_us: int | None

    @property
    def duration_us(self) -> int | None:
        if self.time_start_us is None or self.time_end_us is None:
            return None
        return max(0, self.time_end_us - self.time_start_us)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .raw event-camera recordings to compressed NumPy archives."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A .raw recording or a folder containing .raw recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional folder for converted arrays. Defaults to each raw file folder.",
    )
    parser.add_argument(
        "--read-window-us",
        type=int,
        default=DEFAULT_READ_WINDOW_US,
        help="RawReader time window used while streaming events. Default: 10000000.",
    )
    parser.add_argument(
        "--max-events-buffer",
        type=int,
        default=DEFAULT_MAX_EVENTS_BUFFER,
        help="RawReader event buffer size. Increase this for very dense recordings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = convert_raw_path_to_npz(
        args.input_path,
        output_dir=args.output_dir,
        read_window_us=args.read_window_us,
        max_events_buffer=args.max_events_buffer,
        overwrite=args.overwrite,
    )
    logger.info("Converted {} raw recording(s)", len(results))


def convert_raw_path_to_npz(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    read_window_us: int = DEFAULT_READ_WINDOW_US,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    overwrite: bool = False,
    reader_factory: RawReaderFactory | None = None,
) -> list[RawArrayConversionResult]:
    raw_paths = list(discover_raw_paths(input_path))
    if not raw_paths:
        raise FileNotFoundError(f"No .raw recordings found under {input_path}")

    return [
        convert_raw_to_npz(
            raw_path,
            output_dir=output_dir,
            read_window_us=read_window_us,
            max_events_buffer=max_events_buffer,
            overwrite=overwrite,
            reader_factory=reader_factory,
        )
        for raw_path in raw_paths
    ]


def convert_raw_to_npz(
    raw_path: Path,
    *,
    output_dir: Path | None = None,
    read_window_us: int = DEFAULT_READ_WINDOW_US,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    overwrite: bool = False,
    reader_factory: RawReaderFactory | None = None,
) -> RawArrayConversionResult:
    raw_path = raw_path.expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw recording does not exist: {raw_path}")
    if raw_path.suffix.lower() != ".raw":
        raise ValueError(f"Expected a .raw recording, got {raw_path}")
    if read_window_us <= 0:
        raise ValueError("read_window_us must be positive")

    output_path = npz_path(raw_path, output_dir)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace it."
        )

    reader = (
        open_raw_reader(raw_path, max_events_buffer)
        if reader_factory is None
        else reader_factory(raw_path, max_events_buffer)
    )
    sensor_shape = reader.get_size()

    logger.info("Converting {} to {}", raw_path, output_path)
    events = read_all_events(reader, read_window_us)
    metadata = build_metadata(raw_path, output_path, events, sensor_shape)
    time_start_us = event_time_start_us(events)
    time_end_us = event_time_end_us(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        events=events,
        sensor_shape=np.array(sensor_shape, dtype=np.int32),
        metadata=json.dumps(metadata, sort_keys=True),
    )

    logger.info(
        "Saved {} events to {} ({:.2f} MB)",
        events.size,
        output_path,
        output_path.stat().st_size / 1_000_000,
    )
    return RawArrayConversionResult(
        raw_path=raw_path,
        array_path=output_path,
        event_count=int(events.size),
        sensor_shape=sensor_shape,
        time_start_us=time_start_us,
        time_end_us=time_end_us,
    )


def read_all_events(reader: RawEventReader, read_window_us: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    while not reader.is_done():
        events = reader.load_delta_t(read_window_us)
        if events.size:
            chunks.append(events.copy())

    if not chunks:
        return np.empty(0, dtype=event_dtype())
    return np.concatenate(chunks)


def event_dtype() -> np.dtype:
    return np.dtype([("x", "u2"), ("y", "u2"), ("p", "i1"), ("t", "u8")])


def event_time_start_us(events: np.ndarray) -> int | None:
    return int(events["t"].min()) if events.size else None


def event_time_end_us(events: np.ndarray) -> int | None:
    return int(events["t"].max()) if events.size else None


def build_metadata(
    raw_path: Path,
    output_path: Path,
    events: np.ndarray,
    sensor_shape: tuple[int, int],
) -> dict[str, int | str | None]:
    time_start_us = event_time_start_us(events)
    time_end_us = event_time_end_us(events)
    duration_us = (
        None
        if time_start_us is None or time_end_us is None
        else time_end_us - time_start_us
    )
    positive_event_count = int(np.count_nonzero(events["p"] > 0)) if events.size else 0
    negative_event_count = int(np.count_nonzero(events["p"] <= 0)) if events.size else 0
    return {
        "format_version": ARRAY_FORMAT_VERSION,
        "raw_path": str(raw_path),
        "array_path": str(output_path),
        "event_count": int(events.size),
        "sensor_height": int(sensor_shape[0]),
        "sensor_width": int(sensor_shape[1]),
        "time_start_us": time_start_us,
        "time_end_us": time_end_us,
        "duration_us": duration_us,
        "positive_event_count": positive_event_count,
        "negative_event_count": negative_event_count,
    }


def npz_path(raw_path: Path, output_dir: Path | None = None) -> Path:
    parent = raw_path.parent if output_dir is None else output_dir.expanduser()
    return parent / f"{raw_path.stem}_events.npz"


if __name__ == "__main__":
    main()
