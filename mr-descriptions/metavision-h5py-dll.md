# Repair Windows Metavision and h5py DLL import order

## Motivation

On Windows, importing `metavision_core.event_io` loads Prophesee's bundled,
older `hdf5.dll` before it imports Pixi's `h5py`. The `h5py` extension was
built against Pixi's HDF5 1.14 runtime, so binding it to the already-loaded
Prophesee DLL fails with `ImportError: DLL load failed ... defs`.

## Changes

- Preload Pixi's `h5py` in the temporary OpenEB import context before a
  Metavision native module can load its DLLs.
- Make `check-openeb` use the production OpenEB context, including the safe
  HDF5 load order.

## Validation

- `pixi run check-openeb` succeeds on Windows with the installed Prophesee
  SDK and reports `RawReader` and `EventCD`.
- A direct import through `temporary_openeb_system_site_packages` succeeds.
- `ruff format --check` passes for the two changed Python modules. Focused
  `ruff check` retains three existing auto-fix suggestions in
  `event_array_processing.py`, unrelated to this change.

The repository-wide lint command still reports unrelated existing violations.
`pixi run -e all ty check` is unavailable on Windows because `ty` is declared
