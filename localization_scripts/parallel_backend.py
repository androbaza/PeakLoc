from __future__ import annotations

import sys


def resolve_joblib_backend(requested_backend: str) -> str:
    """Use threads in frozen builds so Joblib does not re-enter the PyInstaller exe."""
    if getattr(sys, "frozen", False) and requested_backend == "loky":
        return "threading"
    return requested_backend
