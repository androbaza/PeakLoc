# Event Model Poisson Migration

## Motivation

PeakLoc previously localized event-count ROIs with an unweighted least-squares
Gaussian path named `localize_MLE`. This migration adds the planned SMLM-style
joint positive/negative Poisson fitter, calibration fallback objects, and the
runner wiring needed to track calibration and fit QC provenance.

## Changes

- Added event-model config fields, with `poisson_joint` as the default and
  `legacy_lsq` retained as a selectable regression path.
- Added `EventCalibration`, `NullCalibration`, NPZ calibration loading, and ROI
  calibrated-background extraction with hot-pixel masking.
- Added calibration utilities for dark/blank rate-map generation and bead
  `sigma_psf_px` estimation.
- Fixed ROI count dtypes, boundary checks, signed coordinate arrays, ROI origin
  metadata, and first/last event timestamp summaries.
- Added a pixel-integrated Gaussian PSF model and fixed-sigma joint Poisson ROI
  fitter with local backgrounds, Fisher covariance, condition reporting, and
  calibration provenance fields.
- Routed the pipeline through `localize_rois`, loaded calibration once per
  recording, and added fit QC summaries to run reports.

## Parameters and Assumptions

- `allow_uncalibrated=True` keeps the pipeline runnable before calibration
  recordings exist.
- `sigma_psf_px=None` falls back to `dataset_fwhm / 2.35`; bead calibration can
  provide an empirical fixed sigma later.
- `fit_sigma=True` is intentionally rejected until bead validation supports
  fitting sigma.
- Hot pixels are masked out of the likelihood.
- `polarity_time_gate_us=5e3` preserves the old timing gate by default while
  making it configurable.

## Validation

- `pixi run -e dev ruff check --fix .` passed.
- `pixi run -e dev ruff format .` passed with no files changed.
- `pixi run -e dev pytest` passed: 22 tests.
- `pixi run -e dev ty check` passed.
- `pixi run -e all ty check` could not run because `pixi.toml` does not define
  an `all` environment.
