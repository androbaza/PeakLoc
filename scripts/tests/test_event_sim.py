import numpy as np

from scripts import event_sim


def test_event_simulator_emits_pipeline_event_conventions() -> None:
    simulator = event_sim.EventSimulator(W=1, H=1)
    base_image = np.ones((1, 1), dtype=np.float64)
    bright_image = np.exp(np.full((1, 1), event_sim.TOL * 2, dtype=np.float64))

    _, initialized_events = simulator.image_callback(base_image, 0.001)
    _, positive_events = simulator.image_callback(bright_image, 0.002)
    _, negative_events = simulator.image_callback(base_image, 0.003)

    assert initialized_events is None
    assert np.issubdtype(event_sim.EVENT_TYPE["t"], np.unsignedinteger)
    assert positive_events["t"].tolist() == [1_000]
    assert positive_events["p"].tolist() == [1]
    assert negative_events["t"].tolist() == [2_000]
    assert negative_events["p"].tolist() == [0]
