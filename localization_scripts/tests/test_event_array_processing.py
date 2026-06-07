import numpy as np

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
