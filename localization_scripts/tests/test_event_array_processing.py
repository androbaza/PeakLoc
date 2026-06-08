import numpy as np
import pytest

from localization_scripts import event_array_processing
from localization_scripts.event_array_processing import array_to_time_map


def test_array_to_time_map_preserves_simultaneous_same_pixel_events():
    events = np.zeros(
        3,
        dtype=[("x", "uint16"), ("y", "uint16"), ("p", "byte"), ("t", "uint64")],
    )
    events["x"] = [4, 4, 4]
    events["y"] = [5, 5, 5]
    events["p"] = [1, 0, 1]
    events["t"] = [100, 100, 101]

    events_t_p_dict = array_to_time_map(events)

    assert list(events_t_p_dict[(np.int32(5), np.int32(4))]) == [
        (100, 1),
        (100, 0),
        (101, 1),
    ]


@pytest.mark.parametrize(
    ("num_coordinates", "expected_chunk_lengths"),
    [
        (0, []),
        (1, [1]),
        (23, [23]),
        (24, [24]),
        (25, [24, 1]),
        (48, [24, 24]),
    ],
)
def test_create_signal_processes_all_coordinate_chunks(
    monkeypatch: pytest.MonkeyPatch,
    num_coordinates: int,
    expected_chunk_lengths: list[int],
) -> None:
    coords = np.arange(num_coordinates * 2, dtype=np.int32).reshape(num_coordinates, 2)
    processed_chunk_lengths = []

    def process_chunk_stub(
        _dict_events: object, coords_split: np.ndarray, _max_len: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.int32, np.int32]]]:
        processed_chunk_lengths.append(len(coords_split))
        output_times = [np.array([row[0]], dtype=np.uint64) for row in coords_split]
        output_cumsum = [np.array([row[1]], dtype=np.int32) for row in coords_split]
        output_coords = [tuple(row) for row in coords_split]
        return output_times, output_cumsum, output_coords

    monkeypatch.setattr(
        event_array_processing, "process_conv_list_parallel", process_chunk_stub
    )

    times, cumsum, coordinates = event_array_processing.create_signal(
        {}, coords, max_len=3
    )

    assert processed_chunk_lengths == expected_chunk_lengths
    assert len(times) == num_coordinates
    assert len(cumsum) == num_coordinates
    assert coordinates == [tuple(row) for row in coords]
