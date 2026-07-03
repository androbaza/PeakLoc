from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.raw_to_tiff_stack import (
    convert_raw_path,
    convert_raw_to_tiff_stack,
    discover_raw_paths,
    events_to_uint8_frame,
    integration_time_ms_to_us,
    tiff_stack_path,
)


EVENT_DTYPE = [("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")]


class FakeRawReader:
    current_time = 0

    def __init__(self, chunks: list[np.ndarray], sensor_shape: tuple[int, int]) -> None:
        self.chunks = chunks
        self.sensor_shape = sensor_shape
        self.done = False
        self.loaded_delta_t: list[int] = []

    def get_size(self) -> tuple[int, int]:
        return self.sensor_shape

    def is_done(self) -> bool:
        return self.done

    def load_delta_t(self, delta_t: int) -> np.ndarray:
        self.loaded_delta_t.append(delta_t)
        if not self.chunks:
            self.done = True
            return np.empty(0, dtype=EVENT_DTYPE)
        events = self.chunks.pop(0)
        self.done = not self.chunks
        return events


def test_convert_raw_to_tiff_stack_writes_8bit_frames_next_to_raw(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "sample.raw"
    raw_path.touch()
    first_chunk = np.array(
        [(2, 1, 1, 10)] * 260 + [(0, 0, 0, 20), (99, 0, 1, 30)],
        dtype=EVENT_DTYPE,
    )
    second_chunk = np.array([(3, 2, 1, 50)], dtype=EVENT_DTYPE)
    reader = FakeRawReader([first_chunk, second_chunk], sensor_shape=(3, 4))

    result = convert_raw_to_tiff_stack(
        raw_path,
        integration_time_ms=12.5,
        reader_factory=lambda _path, _buffer: reader,
    )

    assert result.tiff_path == raw_path.with_name("sample_dt12p5ms_8bit_stack.tiff")
    assert result.frame_count == 2
    assert result.integration_time_us == 12_500
    assert reader.loaded_delta_t == [12_500, 12_500]

    stack = tifffile.imread(result.tiff_path)
    assert stack.dtype == np.uint8
    assert stack.shape == (2, 3, 4)
    assert stack[0, 1, 2] == 255
    assert stack[0, 0, 0] == 1
    assert stack[0, 0, 3] == 0
    assert stack[1, 2, 3] == 1


def test_convert_raw_path_recurses_through_subfolders(tmp_path: Path) -> None:
    root_raw = tmp_path / "root.raw"
    nested_raw = tmp_path / "nested" / "child.RAW"
    nested_raw.parent.mkdir()
    root_raw.touch()
    nested_raw.touch()

    readers = [
        FakeRawReader([np.empty(0, dtype=EVENT_DTYPE)], sensor_shape=(1, 1)),
        FakeRawReader([np.empty(0, dtype=EVENT_DTYPE)], sensor_shape=(1, 1)),
    ]

    results = convert_raw_path(
        tmp_path,
        reader_factory=lambda _path, _buffer: readers.pop(0),
    )

    assert [result.raw_path for result in results] == [root_raw, nested_raw]
    assert all(result.tiff_path.is_file() for result in results)


def test_discover_raw_paths_rejects_non_raw_file(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a recording", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .raw file"):
        list(discover_raw_paths(text_path))


def test_integration_time_ms_to_us_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        integration_time_ms_to_us(0)


def test_events_to_uint8_frame_ignores_out_of_bounds_events() -> None:
    events = np.array([(1, 1, 1, 0), (5, 1, 1, 0)], dtype=EVENT_DTYPE)

    frame = events_to_uint8_frame(events, sensor_shape=(2, 2))

    assert frame.tolist() == [[0, 0], [0, 1]]


def test_tiff_stack_path_uses_safe_integration_time_label(tmp_path: Path) -> None:
    assert tiff_stack_path(tmp_path / "recording.raw", 50.5).name == (
        "recording_dt50p5ms_8bit_stack.tiff"
    )
