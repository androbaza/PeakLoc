# Filter diffuse flash intervals before localization

## Motivation

The `Normal_vs_Rapid` recordings contain broad illumination changes that generate
apparently precise false localizations. The previous ROI-local compactness gate reported
zero rejected candidates because its result depended on ROI extent and spatial masking.
The affected illumination can remain active for seconds between its positive and negative
event transitions, so rejecting only a transition ROI is insufficient.

## Changes

- Detect broad same-polarity activity in bounded full-sensor time bins.
- Merge nearby transition bins and exclude the complete padded illumination interval before
  spatial-mask calibration, event indexing, convolution, peak finding, and ROI generation.
- Remove the ROI-local `diffuse_flash_max_local_fraction` and
  `diffuse_flash_min_positive_events` settings and their candidate-level gate.
- Record excluded intervals, duration, transition-bin count, and event count in a dedicated
  JSON artifact and in run/slice reports.
- Keep QC event-density and detection-funnel counts aligned with the retained event stream.

## Tuned parameters and assumptions

Full RAW profiling used 5 ms bins. Ordinary bins activated about 0.2–1.6% of the sensor;
flash transitions activated 10–100%. Positive and negative transitions were at most about
3.1 seconds apart in the profiled recordings.

- `diffuse_flash_bin_duration_us = 5000`
- `diffuse_flash_min_events_per_polarity = 100000`
- `diffuse_flash_min_active_pixel_fraction = 0.1`
- `diffuse_flash_max_gap_us = 5000000`
- `diffuse_flash_padding_us = 50000`

The filter intentionally discards all events—including real blinks—inside a detected flash
interval. It does not attempt to salvage signal while diffuse illumination is active.

## Validation

- `pixi run -e dev ruff check --fix .`
- `pixi run -e dev ruff format .`
- `pixi run -e dev pytest localization_scripts/tests/test_pipeline_config.py localization_scripts/tests/test_roi_generation.py localization_scripts/tests/test_pipeline_runner.py -q` (`23 passed`)
- Targeted `pixi run -e all ty check` for all changed Python modules (`passed`)
- Full `pixi run -e all ty check` still reports eight pre-existing notebook diagnostics in
  `notebooks/raw_event_npz_explorer.ipynb`.
- Full-sensor profiling found 4, 10, 11, and 11 excluded intervals in the rapid-only,
  normal, mixed, and 405-induced recordings, respectively.

## Artifact refresh

The existing `20260716_193311_378386` result arrays and visual/QC artifacts were refreshed
in place by removing rows whose `t_peak` falls in a detected interval. This post-filter
refresh avoids repeating peak detection and fitting; the next full recording run will
exercise the new early event-stream filter end to end.
