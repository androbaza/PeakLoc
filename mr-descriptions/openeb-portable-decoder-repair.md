# Fix portable OpenEB decoder and GUI log timestamps

## Motivation

The existing `dist/PeakLoc` bundle contained CPython 3.10 extensions while the installed
Metavision SDK provides CPython 3.9 extensions. The GUI preflight therefore could not import
`metavision_sdk_base_paths_internal` for RAW decoding. Captured worker output was also appended
directly to `PeakLoc.log` without timestamps.

## Changes

- Format captured GUI worker records with millisecond timestamps matching the Loguru file format.
- Keep Prophesee DLL directories out of global Pixi activation; add them only after `h5py` loads
  inside the decoder context.
- Require CPython 3.9 in the Windows GUI build script.
- Make the RAW-reader path-isolation test valid when the Windows activation override is set.

## Validation

- `pixi run -e dev pytest interface/tests/test_operations.py localization_scripts/tests/test_event_array_processing.py -q` (12 passed)
- `pixi run check-openeb` (Metavision `RawReader` and `EventCD` imported on CPython 3.9)
- Built an isolated CPython 3.9 GUI distribution with PyInstaller.
- Ran `pixi run gui -- --pipeline-worker --config <disposable-config>` on one RAW interval from
  `1,000,000` to `1,300,000` microseconds. Preflight passed, the run exited 0, and produced a
  run report plus PNG/TIFF reconstructions.

## Release note

The normal `dist/PeakLoc` folder could not be refreshed during validation because an already-open
`dist/PeakLoc/PeakLoc.exe` held the build output. Close that application and run
`pixi run -e dev build-gui` to replace the release folder with the verified CPython 3.9 build.
