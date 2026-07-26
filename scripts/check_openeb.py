"""Validate the Prophesee/OpenEB bindings required to decode RAW recordings."""

import sys
from importlib import import_module

from localization_scripts.event_array_processing import add_openeb_system_site_packages
from localization_scripts.pipeline_runner import _available_memory_bytes


def main() -> None:
    add_openeb_system_site_packages()
    raw_reader = import_module("metavision_core.event_io.raw_reader").RawReader
    event_cd = import_module("metavision_sdk_base").EventCD
    available_memory_gib = _available_memory_bytes() / 2**30
    print(
        f"OpenEB bindings OK on Python {sys.version_info.major}.{sys.version_info.minor}"
    )
    print(f"RawReader: {raw_reader}")
    print(f"EventCD: {event_cd}")
    print(f"Available RAM: {available_memory_gib:.1f} GiB")


if __name__ == "__main__":
    main()
