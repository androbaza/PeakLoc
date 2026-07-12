# Sparse Spatial-Mask Processing

## Motivation

PeakLoc previously expanded every pixel inside a slice's rectangular event bounds into
per-pixel dictionaries and convolved traces. For sparse biological structures, this
spends most memory and time on background-only sensor area.

## Changes

- Added an opt-in first-minute spatial calibration mask with density thresholding,
  connected-component filtering, and a configurable morphology margin.
- Added a second support mask, dilated by the larger of the convolution and ROI
  radii. Peak centres are only processed in the target mask, while every required
  neighbor event remains available from the support mask.
- Updated per-slice polarity/time maps to ignore events outside the support mask,
  avoiding a selected-event copy and avoiding empty-pixel dictionaries/traces.
- Saved target/support masks and JSON metadata to `reports/`, and included coverage
  and fallback state in the run report.
- Added a full-frame fallback for disabled masks, no qualifying components, and
  support masks that cover too much of the sensor.
- Restored the rapid-switching RAW configuration's 100,000,000-event OpenEB buffer:
  direct measurement showed lower values fail before decoding due to a peak batch of
  nearly nine million events in 50 ms. The disk-backed cache keeps that reader buffer
  separate from per-slice pipeline memory.

## Measurement on the current recording

The first 60 seconds contained 137,837,220 events. With the configured threshold of
400 events/pixel, minimum component size of 20 pixels, and a 12-pixel target margin:

- Target coverage: 25.17% (231,975 pixels)
- Support coverage: 32.36% (298,262 pixels)

This removes about 74.8% of convolution targets and 67.6% of support-map coordinates
while preserving the full convolution and ROI neighborhood for every permitted centre.

## Scientific safeguards

- The feature is disabled by default; it is enabled only in the current root config.
- An unmasked comparison can be run with `PEAKLOC_SPATIAL_MASK_ENABLED=false`.
- The mask is an intentional spatial selection, so the saved report artifacts should
  be reviewed before drawing quantitative conclusions, especially if structures drift
  or become active after the calibration minute.

## Validation

- Focused existing tests: 44 passed across pipeline config/runner, event processing,
  ROI generation, peak finding, localization fitting, and QC dashboard modules.
- Existing synthetic end-to-end pipeline test passed.
- A masked synthetic end-to-end probe retained one localization while reducing target
  coverage to 4.2% and support coverage to 12.4%.
- A direct first-minute RAW probe produced the coverage statistics above.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff check --fix .` and
  `PYTHONNOUSERSITE=1 pixi run -e dev ruff format .` passed.
- `PYTHONNOUSERSITE=1 pixi run -e all ty check localization_scripts` passed.
