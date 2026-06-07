# Root Cleanup And Default Config

## Motivation

The repository root was accumulating helper scripts and `PeakLoc.py` had grown
into a mixed CLI, orchestration, plotting, and reporting module. The default
configuration file was also only an example, so normal runs did not load it
unless it was passed explicitly.

## Changes

- Moved helper scripts into `scripts/`:
  - `scripts/clean_temp_files.py`
  - `scripts/import_test.py`
  - `scripts/peaks_dict_to_locs.py`
- Added `scripts/__init__.py` and switched pixi helper tasks to `python -m`.
- Extracted batch orchestration, slice processing, report writing, event loading,
  and plotting artifact saving into `localization_scripts/pipeline_runner.py`.
- Kept `PeakLoc.py` as a thin CLI entry point.
- Renamed `peakloc_config.example.json` to root `config.json`.
- Made `config.json` load by default when `--config` and `PEAKLOC_CONFIG` are not
  provided.
- Updated README, AGENTS, pixi tasks, and MR notes for the new paths and default
  config behavior.
- Added `pytest.ini` so `pixi run -e dev pytest` can import package modules
  directly.

## Validation

```bash
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e all ty check
pixi run -e dev pytest
pixi run python -m py_compile PeakLoc.py scripts/*.py localization_scripts/*.py
pixi run import-test
pixi run peakloc
```

- `pytest`: 5 passed.
- `import-test`: loaded 10 events from `data/AF647_coverslip.raw`.
- `peakloc`: completed without warnings using default root `config.json`.
- Example report:
  `data/AF647_coverslip/reports/peakloc_report_20260607_164208.md`
