from types import SimpleNamespace
import sys

from joblib import Parallel, delayed
import numpy as np
import pytest

from localization_scripts import event_array_processing
from localization_scripts.event_array_processing import array_to_time_map


def _worker_has_sys_path(path: str) -> bool:
    return path in sys.path


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
        _dict_events: object,
        coords_split: np.ndarray,
        _max_len: int,
        *,
        roi_rad: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.int32, np.int32]]]:
        assert roi_rad == 1
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


def test_raw_reader_system_path_does_not_leak_to_loky_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    openeb_site = tmp_path / "openeb_site"
    openeb_site.mkdir()
    event_dtype = np.dtype(
        [("x", np.uint16), ("y", np.uint16), ("p", np.int8), ("t", np.uint64)]
    )
    event_chunk = np.asarray([(1, 2, 1, 3)], dtype=event_dtype)
    import_paths: list[str] = []

    class FakeRawReader:
        def __init__(self, _filename: str, *, max_events: int) -> None:
            assert max_events == 7
            self.finished = False

        def is_done(self) -> bool:
            return self.finished

        def load_delta_t(self, _duration_us: int) -> np.ndarray:
            self.finished = True
            return event_chunk

    def import_module_stub(module_name: str) -> SimpleNamespace:
        import_paths.append(module_name)
        assert str(openeb_site) in sys.path
        if module_name == "metavision_core.event_io.raw_reader":
            return SimpleNamespace(RawReader=FakeRawReader)
        if module_name == "metavision_sdk_base":
            return SimpleNamespace(EventCD=event_dtype)
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(
        event_array_processing, "OPENEB_SYSTEM_SITE_PACKAGES", openeb_site
    )
    monkeypatch.setattr(event_array_processing, "import_module", import_module_stub)

    events = event_array_processing.raw_events_to_array("fixture.raw", max_events=7)

    assert import_paths == [
        "metavision_core.event_io.raw_reader",
        "metavision_sdk_base",
    ]
    assert events.tolist() == event_chunk.tolist()
    assert str(openeb_site) not in sys.path
    worker_paths = Parallel(n_jobs=2, backend="loky", reuse=False)(
        delayed(_worker_has_sys_path)(str(openeb_site)) for _ in range(2)
    )
    assert worker_paths == [False, False]


def test_openeb_site_packages_uses_windows_activation_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    openeb_site = tmp_path / "metavision_site"
    openeb_site.mkdir()

    monkeypatch.setenv(
        event_array_processing.OPENEB_SITE_PACKAGES_ENV_VAR, str(openeb_site)
    )

    assert event_array_processing.openeb_site_packages() == [openeb_site]
