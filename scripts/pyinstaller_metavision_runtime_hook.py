"""Expose the default Windows Metavision installation to a frozen PeakLoc app."""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    metavision_root = Path(
        os.environ.get("PEAKLOC_METAVISION_ROOT", r"C:\Program Files\Prophesee")
    )
    site_packages = metavision_root / "lib" / "python3" / "site-packages"
    if site_packages.is_dir():
        site_packages_string = str(site_packages)
        if site_packages_string not in sys.path:
            sys.path.append(site_packages_string)

    dll_directories = [
        metavision_root / "bin",
        metavision_root / "third_party" / "bin",
    ]
    existing_dll_directories = [
        path for path in dll_directories if site_packages.is_dir() and path.is_dir()
    ]
    if existing_dll_directories:
        dll_directory_entries = [str(path) for path in existing_dll_directories]
        os.environ["PATH"] = os.pathsep.join(
            [*dll_directory_entries, os.environ.get("PATH", "")]
        )
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES = [
                os.add_dll_directory(str(path)) for path in existing_dll_directories
            ]
