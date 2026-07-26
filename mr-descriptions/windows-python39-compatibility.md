# Windows Python 3.9 and Prophesee compatibility

## Motivation

The installed Prophesee SDK supplies CPython 3.9 extension modules on Windows.
The previous Windows Pixi environment used Python 3.12, so importing
`metavision_sdk_base` could not load its matching binary dependency.

## Changes

- Pin the `win-64` environment to Python 3.9 and constrain Windows-only
  packages to Python-3.9-compatible releases.
- Keep the Linux environment on Python 3.12, including its notebook,
  Kaleido, and type-checking tools.
- Add a Windows activation script that exposes the installed Prophesee SDK
  while preserving Pixi's DLL directory first on `PATH`.
- Replace Python 3.10+/3.11+ runtime features used by the pipeline with
  compatible equivalents: postponed annotations, a strict-zip helper, and
  ordinary dataclass construction.
- Add `pixi run check-openeb` to verify that RAW-reading bindings can load.
- Make the CPU-budget unit-test fixture portable to Windows.

## Validation

On Windows with the regenerated Pixi environments:

- `pixi run check-openeb` imports `RawReader` and `EventCD` successfully on
  Python 3.9.
- `pixi run peakloc --help` imports the application entry point successfully.
- Focused compatibility tests pass: `21 passed`.
- `ruff check` and `ruff format --check` pass for the new helpers, and
  `python -m compileall -q PeakLoc.py localization_scripts scripts` succeeds.

On Linux:

- `pixi run -e all ty check` succeeds.
