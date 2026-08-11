from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from loguru import logger

from scripts.raw_to_video import (
    DEFAULT_INTEGRATION_TIME_MS,
    DEFAULT_MAX_EVENTS_BUFFER,
    RawReaderFactory,
    discover_raw_paths,
    events_to_uint8_frame,
    integration_time_ms_to_us,
    open_raw_reader,
)


@dataclass(frozen=True)
class RawTiffStackConversionResult:
    raw_path: Path
    tiff_path: Path
    frame_count: int
    integration_time_us: int
    sensor_shape: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .raw event-camera recordings to 8-bit TIFF stacks."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A .raw recording or a folder containing .raw recordings.",
    )
    parser.add_argument(
        "--integration-time-ms",
        type=float,
        default=DEFAULT_INTEGRATION_TIME_MS,
        help="Frame integration time in milliseconds. Default: 50.",
    )
    parser.add_argument(
        "--max-events-buffer",
        type=int,
        default=DEFAULT_MAX_EVENTS_BUFFER,
        help="RawReader event buffer size. Increase this for very dense recordings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = convert_raw_path_to_tiff_stack(
        args.input_path,
        integration_time_ms=args.integration_time_ms,
        max_events_buffer=args.max_events_buffer,
    )
    logger.info("Converted {} raw recording(s)", len(results))


def convert_raw_path_to_tiff_stack(
    input_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
) -> list[RawTiffStackConversionResult]:
    raw_paths = list(discover_raw_paths(input_path))
    if not raw_paths:
        raise FileNotFoundError(f"No .raw recordings found under {input_path}")

    return [
        convert_raw_to_tiff_stack(
            raw_path,
            integration_time_ms=integration_time_ms,
            max_events_buffer=max_events_buffer,
            reader_factory=reader_factory,
        )
        for raw_path in raw_paths
    ]


def convert_raw_to_tiff_stack(
    raw_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
) -> RawTiffStackConversionResult:
    raw_path = raw_path.expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw recording does not exist: {raw_path}")
    if raw_path.suffix.lower() != ".raw":
        raise ValueError(f"Expected a .raw recording, got {raw_path}")

    integration_time_us = integration_time_ms_to_us(integration_time_ms)
    reader = (
        open_raw_reader(raw_path, max_events_buffer)
        if reader_factory is None
        else reader_factory(raw_path, max_events_buffer)
    )
    sensor_shape = reader.get_size()
    output_path = tiff_stack_path(raw_path, integration_time_ms)

    frame_count = 0
    logger.info(
        "Converting {} to {} with {} us integration windows",
        raw_path,
        output_path,
        integration_time_us,
    )
    writer = create_tiff_stack_writer(output_path)
    try:
        while not reader.is_done():
            events = reader.load_delta_t(integration_time_us)
            frame = events_to_uint8_frame(events, sensor_shape)
            write_tiff_frame(writer, frame)
            frame_count += 1

        if frame_count == 0:
            frame = np.zeros(sensor_shape, dtype=np.uint8)
            write_tiff_frame(writer, frame)
            frame_count = 1
    finally:
        writer.close()

    logger.info("Saved {} frame(s) to {}", frame_count, output_path)
    return RawTiffStackConversionResult(
        raw_path=raw_path,
        tiff_path=output_path,
        frame_count=frame_count,
        integration_time_us=integration_time_us,
        sensor_shape=sensor_shape,
    )


def create_tiff_stack_writer(output_path: Path) -> tifffile.TiffWriter:
    return tifffile.TiffWriter(output_path, bigtiff=True)


def write_tiff_frame(writer: tifffile.TiffWriter, frame: np.ndarray) -> None:
    writer.write(frame, photometric="minisblack")


def tiff_stack_path(raw_path: Path, integration_time_ms: float) -> Path:
    integration_label = f"{integration_time_ms:g}".replace(".", "p")
    return raw_path.with_name(f"{raw_path.stem}_dt{integration_label}ms_stack.tiff")


if __name__ == "__main__":
    main()
