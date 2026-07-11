# Temporal Blink QC and Baseline Sweep

## Motivation

Make full-run QC useful for locating temporal variation across the reconstructed
biological structure, while retaining a conservative detection baseline for the
configured rapid-switching recording.

## Changes

- Added static temporal spatial maps, timing distributions, per-spatial-bin CSV/JSON
  statistics, and a bounded Plotly 3D timing map to the run QC dashboard.
- Marked the temporal quantities as ROI-window proxies rather than direct molecular
  transition times.
- Bound the 3D artifact to `qc_max_events_for_interactive` with deterministic sampling.
- Wired `convolution_roi_radius` into the convolved signal creation path.
- Cleared stale per-slice localization, ROI, and QC arrays before aggregation and
  removed QC arrays during normal temporary-file cleanup.
- Set the RAW reader rolling buffer to 100,000,000 events and documented the sweep
  evidence and remaining policy-only settings in `CONFIG_TUNING.md`.

## Validation

- Compared dense 10-12 s and moderate 30-32 s slices. The existing detection defaults
  were the best quality/throughput compromise; relaxed grouping caused 4-5% close
  space/time duplicates, while conservative grouping lost 27-41% of accepted fits.
- Generated the dashboard for the completed 10-15 s result. It produced all temporal
  artifacts, linked them from `qc/index.html`, and reported 9,777 valid temporal
  localizations with 258 qualified spatial bins.
- `PYTHONNOUSERSITE=1 pixi run -e dev pytest localization_scripts/tests/test_event_array_processing.py localization_scripts/tests/test_pipeline_runner.py localization_scripts/tests/test_qc_dashboard.py localization_scripts/tests/test_pipeline_config.py`
  passed: 27 tests.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff check --fix .` and
  `PYTHONNOUSERSITE=1 pixi run -e dev ruff format .` passed.
- Focused `ty check` for the changed Python modules passed. Full `ty check` still has
  eight existing diagnostics in `notebooks/raw_event_npz_explorer.ipynb`.
