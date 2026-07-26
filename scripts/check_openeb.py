"""Validate the Prophesee/OpenEB bindings required to decode RAW recordings."""

import sys
from importlib import import_module

from localization_scripts.event_array_processing import add_openeb_system_site_packages


def main() -> None:
    add_openeb_system_site_packages()
    raw_reader = import_module("metavision_core.event_io.raw_reader").RawReader
    event_cd = import_module("metavision_sdk_base").EventCD
    print(
        f"OpenEB bindings OK on Python {sys.version_info.major}.{sys.version_info.minor}"
    )
    print(f"RawReader: {raw_reader}")
    print(f"EventCD: {event_cd}")


if __name__ == "__main__":
    main()
