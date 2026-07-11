# Loky Worker Environment Stability

## Motivation

Prevent RAW runs from crashing when Loky workers import Numba after OpenEB's system
site-packages directory exposes an incompatible host `coverage` package.

## Changes

- Scope OpenEB's `/usr/lib/python3/dist-packages` path to RAW decoding only.
- Replace inherited `PYTHONPATH` at PeakLoc startup with the repository root and the
  active Pixi site-packages directory.
- Remove foreign `site-packages` and `dist-packages` entries before pipeline imports.
- Add a regression test that verifies a mocked RAW reader's temporary path does not
  reach fresh Loky workers.

## Validation

- Loaded 20,770 events from a real 50 ms RAW chunk while the OpenEB path was active;
  the path was absent after decoding.
- With `PYTHONPATH=/usr/lib/python3/dist-packages`, the parent and two fresh Loky
  workers found no `coverage` module and imported Numba `0.65.1` successfully.
- Under the same contaminated environment, a two-worker `find_peaks_parallel` probe
  completed successfully.
- `PYTHONNOUSERSITE=1 pixi run -e dev pytest localization_scripts/tests/test_event_array_processing.py localization_scripts/tests/test_pipeline_runner.py localization_scripts/tests/test_qc_dashboard.py localization_scripts/tests/test_pipeline_config.py`
  passed: 28 tests.
