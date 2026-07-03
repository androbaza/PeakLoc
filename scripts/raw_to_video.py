from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
from loguru import logger
import numpy as np

from localization_scripts.event_array_processing import add_openeb_system_site_packages


DEFAULT_INTEGRATION_TIME_MS = 50.0
DEFAULT_MAX_EVENTS_BUFFER = 1_000_000
DEFAULT_CODEC = "libx264"
DEFAULT_CRF = 23
DEFAULT_PRESET = "medium"


class RawEventReader(Protocol):
    current_time: int

    def get_size(self) -> tuple[int, int]: ...

    def is_done(self) -> bool: ...

    def load_delta_t(self, delta_t: int) -> np.ndarray: ...


class VideoWriter(Protocol):
    def append_data(self, im: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RawVideoConversionResult:
    raw_path: Path
    video_path: Path
    frame_count: int
    integration_time_us: int
    fps: float
    sensor_shape: tuple[int, int]
    codec: str


RawReaderFactory = Callable[[Path, int], RawEventReader]
VideoWriterFactory = Callable[[Path, float, str, int, str], VideoWriter]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .raw event-camera recordings to 8-bit MP4 videos."
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
        "--fps",
        type=float,
        default=None,
        help="Video frame rate. Defaults to real-time playback from integration time.",
    )
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help="FFmpeg video codec. Default: libx264.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help="H.264 quality value; lower is larger/better. Default: 23.",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help="FFmpeg encoder preset. Default: medium.",
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
        fps=args.fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        max_events_buffer=args.max_events_buffer,
    )
    logger.info("Converted {} raw recording(s)", len(results))


def convert_raw_path(
    input_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    fps: float | None = None,
    codec: str = DEFAULT_CODEC,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
    video_writer_factory: VideoWriterFactory | None = None,
) -> list[RawVideoConversionResult]:
    raw_paths = list(discover_raw_paths(input_path))
    if not raw_paths:
        raise FileNotFoundError(f"No .raw recordings found under {input_path}")

    results = []
    for raw_path in raw_paths:
        results.append(
            convert_raw_to_video(
                raw_path,
                integration_time_ms=integration_time_ms,
                fps=fps,
                codec=codec,
                crf=crf,
                preset=preset,
                max_events_buffer=max_events_buffer,
                reader_factory=reader_factory,
                video_writer_factory=video_writer_factory,
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


def convert_raw_to_video(
    raw_path: Path,
    *,
    integration_time_ms: float = DEFAULT_INTEGRATION_TIME_MS,
    fps: float | None = None,
    codec: str = DEFAULT_CODEC,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    max_events_buffer: int = DEFAULT_MAX_EVENTS_BUFFER,
    reader_factory: RawReaderFactory | None = None,
    video_writer_factory: VideoWriterFactory | None = None,
) -> RawVideoConversionResult:
    raw_path = raw_path.expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw recording does not exist: {raw_path}")
    if raw_path.suffix.lower() != ".raw":
        raise ValueError(f"Expected a .raw recording, got {raw_path}")

    integration_time_us = integration_time_ms_to_us(integration_time_ms)
    video_fps = video_fps_from_integration_time(integration_time_ms, fps)
    reader = (
        open_raw_reader(raw_path, max_events_buffer)
        if reader_factory is None
        else reader_factory(raw_path, max_events_buffer)
    )
    sensor_shape = reader.get_size()
    output_path = video_path(raw_path, integration_time_ms, codec)

    frame_count = 0
    logger.info(
        "Converting {} to {} with {} us integration windows at {:.3g} fps",
        raw_path,
        output_path,
        integration_time_us,
        video_fps,
    )
    writer_factory = (
        create_video_writer if video_writer_factory is None else video_writer_factory
    )
    writer = writer_factory(output_path, video_fps, codec, crf, preset)
    try:
        while not reader.is_done():
            events = reader.load_delta_t(integration_time_us)
            frame = events_to_uint8_frame(events, sensor_shape)
            writer.append_data(frame_to_video_rgb(frame))
            frame_count += 1

        if frame_count == 0:
            frame = np.zeros(sensor_shape, dtype=np.uint8)
            writer.append_data(frame_to_video_rgb(frame))
            frame_count = 1
    finally:
        writer.close()

    logger.info("Saved {} frame(s) to {}", frame_count, output_path)
    return RawVideoConversionResult(
        raw_path=raw_path,
        video_path=output_path,
        frame_count=frame_count,
        integration_time_us=integration_time_us,
        fps=video_fps,
        sensor_shape=sensor_shape,
        codec=codec,
    )


def create_video_writer(
    output_path: Path,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
) -> VideoWriter:
    return imageio.get_writer(
        output_path,
        mode="I",
        fps=fps,
        codec=codec,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-movflags",
            "+faststart",
        ],
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


def video_fps_from_integration_time(
    integration_time_ms: float, fps: float | None
) -> float:
    if fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive value")
        return float(fps)
    return 1_000.0 / integration_time_ms


def video_path(raw_path: Path, integration_time_ms: float, codec: str) -> Path:
    integration_label = f"{integration_time_ms:g}".replace(".", "p")
    codec_label = codec_label_for_filename(codec)
    return raw_path.with_name(
        f"{raw_path.stem}_dt{integration_label}ms_{codec_label}.mp4"
    )


def codec_label_for_filename(codec: str) -> str:
    if codec == "libx264":
        return "h264"
    return codec.removeprefix("lib").replace("_", "-")


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


def frame_to_video_rgb(frame: np.ndarray) -> np.ndarray:
    even_frame = pad_to_even_shape(frame)
    return np.repeat(even_frame[:, :, np.newaxis], 3, axis=2)


def pad_to_even_shape(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape
    padded_height = height + height % 2
    padded_width = width + width % 2
    if (padded_height, padded_width) == frame.shape:
        return frame

    padded = np.zeros((padded_height, padded_width), dtype=frame.dtype)
    padded[:height, :width] = frame
    return padded


if __name__ == "__main__":
    main()
