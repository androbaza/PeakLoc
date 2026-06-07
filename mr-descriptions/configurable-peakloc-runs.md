# Configurable PeakLoc Runs

## Motivation

PeakLoc run parameters were stored as module constants in `PeakLoc.py`, which made
the exact settings behind a generated localization file hard to audit later. This
change moves run settings into a validated dataclass that can be loaded from JSON
and records the effective settings and summary results with the generated
artifacts.

## Changes

- Added `PeakLocConfig` in `localization_scripts/pipeline_config.py`.
- Added `--config <path>` and `PEAKLOC_CONFIG` support for JSON configuration.
- Preserved existing environment overrides for `PEAKLOC_INPUT_FOLDER`,
  `PEAKLOC_SLICE_START`, and `PEAKLOC_SLICE_DURATION`.
- Refactored `PeakLoc.py` to pass configuration into peak finding, ROI
  generation, localization fitting, and plotting.
- Added per-recording Markdown reports in `reports/` with settings, counts,
  timings, and artifact paths.
- Added effective settings snapshots in `reports/` as pretty-printed JSON.
- Added `peakloc_config.example.json` and focused config tests.

## Assumptions

- JSON is used instead of YAML to avoid adding a parser dependency.
- A stdlib dataclass is used instead of Pydantic because the current settings are
  flat and the validation rules are simple.
- Temporary per-slice localization and ROI arrays are omitted from the report when
  `cleanup_temp_outputs` removes them.

## Validation

```bash
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e all ty check
pixi run -e dev pytest
pixi run peakloc
```

- `pytest`: 4 passed.
- `peakloc`: completed without warnings on `data/AF647_coverslip.raw`.
- Example report:
  `data/AF647_coverslip/reports/peakloc_report_20260607_161717.md`
- Example figures:
  `data/AF647_coverslip/figures/roi_fits_20260607_161717.png`
  `data/AF647_coverslip/figures/roi_event_times_20260607_161717.png`
