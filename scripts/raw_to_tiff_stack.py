from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol

from loguru import logger
import numpy as np
import tifffile

from localization_scripts.event_array_processing import add_openeb_system_site_packages


DEFAULT_INTEGRATION_TIME_MS = 50.0
DEFAULT_MAX_EVENTS_BUFFER = 1_000_000


class RawEventReader(Protocol):
    current_time: int

    def get_size(self) -> tuple[int, int]: ...

    def is_done(self) -> bool: ...

    def load_delta_t(self, delta_t: int) -> np.ndarray: ...


@dataclass(frozen=True)
class RawTiffConversionResult:
    raw_path: Path
    tiff_path: Path
    frame_count: int
    integration_time_us: int
    sensor_shape: tuple[int, int]


RawReaderFactory = Callable[[Path, int], RawEventReader]


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
    results = convert_raw_path(
        args.input_path,
        integration_time_ms=args.integration_time_ms,
        max_events_buffer=args.max_events_buffer,
    )
    logger.info("Converted {} raw recording(s)", len(results))


def convert_raw_path(
    input_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
) -> list[RawTiffConversionResult]:
    raw_paths = list(discover_raw_paths(input_path))
    if not raw_paths:
        raise FileNotFoundError(f"No .raw recordings found under {input_path}")

    results = []
    for raw_path in raw_paths:
        results.append(
            convert_raw_to_tiff_stack(
                raw_path,
                integration_time_ms=integration_time_ms,
                max_events_buffer=max_events_buffer,
                reader_factory=reader_factory,
            )
        )
    return results


def discover_raw_paths(input_path: Path) -> Iterable[Path]:
    path = input_path.expanduser()
    if path.is_file():
        if path.suffix.lower() != ".raw":
            raise ValueError(f"Expected a .raw file, got {path}")
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    yield from sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.suffix.lower() == ".raw"
        ),
        key=lambda candidate: (
            len(candidate.relative_to(path).parts),
            candidate.relative_to(path).as_posix().lower(),
        ),
    )


def convert_raw_to_tiff_stack(
    raw_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
) -> RawTiffConversionResult:
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
    with tifffile.TiffWriter(output_path) as stack_writer:
        while not reader.is_done():
            events = reader.load_delta_t(integration_time_us)
            frame = events_to_uint8_frame(events, sensor_shape)
            stack_writer.write(
                frame,
                photometric="minisblack",
                contiguous=True,
                metadata=_tiff_metadata(integration_time_ms)
                if frame_count == 0
                else None,
            )
            frame_count += 1

    if frame_count == 0:
        frame = np.zeros(sensor_shape, dtype=np.uint8)
        tifffile.imwrite(
            output_path,
            frame,
            photometric="minisblack",
            metadata=_tiff_metadata(integration_time_ms),
        )
        frame_count = 1

    logger.info("Saved {} frame(s) to {}", frame_count, output_path)
    return RawTiffConversionResult(
        raw_path=raw_path,
        tiff_path=output_path,
        frame_count=frame_count,
        integration_time_us=integration_time_us,
        sensor_shape=sensor_shape,
    )


def open_raw_reader(raw_path: Path, max_events_buffer: int) -> RawEventReader:
    if max_events_buffer <= 0:
        raise ValueError("max_events_buffer must be positive")

    add_openeb_system_site_packages()
    raw_reader = import_module("metavision_core.event_io.raw_reader").RawReader
    return raw_reader(str(raw_path), max_events=max_events_buffer)


def integration_time_ms_to_us(integration_time_ms: float) -> int:
    if not np.isfinite(integration_time_ms) or integration_time_ms <= 0:
        raise ValueError("integration_time_ms must be a finite positive value")
    return max(1, int(round(integration_time_ms * 1_000)))


def tiff_stack_path(raw_path: Path, integration_time_ms: float) -> Path:
    integration_label = f"{integration_time_ms:g}".replace(".", "p")
    return raw_path.with_name(
        f"{raw_path.stem}_dt{integration_label}ms_8bit_stack.tiff"
    )


def _tiff_metadata(integration_time_ms: float) -> dict[str, str | float]:
    return {
        "axes": "YX",
        "IntegrationTime": integration_time_ms,
        "IntegrationTimeUnit": "ms",
    }


def events_to_uint8_frame(
    events: np.ndarray, sensor_shape: tuple[int, int]
) -> np.ndarray:
    frame_counts = np.zeros(sensor_shape, dtype=np.uint32)
    if events.size == 0:
        return frame_counts.astype(np.uint8)

    height, width = sensor_shape
    event_x = events["x"].astype(np.intp, copy=False)
    event_y = events["y"].astype(np.intp, copy=False)
    in_bounds = (event_x >= 0) & (event_x < width) & (event_y >= 0) & (event_y < height)
    np.add.at(frame_counts, (event_y[in_bounds], event_x[in_bounds]), 1)
    return np.minimum(frame_counts, np.iinfo(np.uint8).max).astype(np.uint8)


if __name__ == "__main__":
    main()
