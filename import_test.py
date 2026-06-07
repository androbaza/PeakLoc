from pathlib import Path

from localization_scripts.event_array_processing import add_openeb_system_site_packages


RAW_FILE = Path("data/AF647_coverslip.raw")
EVENT_COUNT = 10
MAX_EVENTS = 1_000_000


def read_events(filename: Path, event_count: int = EVENT_COUNT):
    add_openeb_system_site_packages()

    from metavision_core.event_io.raw_reader import RawReader

    reader = RawReader(str(filename), max_events=MAX_EVENTS)
    return reader.load_n_events(event_count)


def main() -> None:
    events = read_events(RAW_FILE)
    print(f"Loaded {events.size} events from {RAW_FILE}")
    print(events)


if __name__ == "__main__":
    main()
