# Build and deliver the Windows desktop application

The supported end-user artifact is a PyInstaller one-folder distribution:

    PeakLoc/
    ├── PeakLoc.exe
    ├── config.json
    ├── PeakLoc User Guide.md
    └── _internal/

The recipient needs neither Python, Pixi, nor this repository. The entire folder is required;
PeakLoc.exe is not a standalone file.

## Build-PC requirements

- 64-bit Windows
- Pixi
- Metavision Studio / SDK in C:\Program Files\Prophesee
- CPython 3.9-compatible Metavision bindings

From PowerShell in the repository root:

    pixi install
    pixi run check-openeb
    pixi run -e dev build-gui

The build task creates a windowed dist\PeakLoc\PeakLoc.exe, copies the portable starter config
beside it, includes the user guide, and bundles the Python scientific runtime in _internal.
The one-folder layout is deliberate: native scientific dependencies are more reliable and easier
to audit in this layout than in a self-extracting one-file build.

## Release validation

Test the release folder rather than the source checkout:

1. Double-click dist\PeakLoc\PeakLoc.exe.
2. Select a small RAW or NumPy event recording.
3. Select **Check setup** and confirm that expected prerequisites pass.
4. Build a calibration from small representative dark and blank recordings.
5. Run one short slice and inspect its output folder.
6. Cancel a disposable run and confirm the interface returns to Ready state.
7. Save and reopen a config.

For a clean target-PC test, copy dist\PeakLoc to another Windows machine that has the matching
Metavision SDK. Do not test only from the build checkout.

## Target-PC startup

The user double-clicks PeakLoc.exe and follows PeakLoc User Guide.md. The application discovers
config.json beside the executable. A missing Metavision installation does not prevent the GUI
from opening; **Check setup** reports that RAW decoding is unavailable. NumPy event recordings
remain usable without Metavision.

## Rebuild checklist

- Build with the Windows Pixi environment (Python 3.9).
- Confirm pixi run check-openeb before packaging RAW support.
- Use the supplied runtime hook.
- Deliver PeakLoc.exe, config.json, the user guide, and _internal together.
- Repeat calibration, preflight, short-run, cancellation, and config round-trip checks.
