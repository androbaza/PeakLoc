"""Expose the Windows Metavision Python bindings to a frozen PeakLoc app."""

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
