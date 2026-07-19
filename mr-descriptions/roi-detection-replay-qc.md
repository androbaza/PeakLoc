# Add ROI detection-replay QC montage

## Motivation

The uncertainty quantile montage exposed the fitted ROI image but not the temporal decision that
materialized it. Reviewers could not tell which detected peak, positive window, and negative window
produced a displayed ROI.

## Changes

- Add `roi_detection_replay_quantile_samples.png` next to the uncertainty quantile montage, using
  the identical quantile and failed-fit sample selection.
- Give each sample a spatial ROI panel with detector-seed and fitted-center markers, plus a timing
  panel. The timing panel marks the cycle peak, the original detector peak where distinct, and the
  selected positive/negative temporal support windows.
- Carry temporal seed coordinates into attempted localization records so a new run can show the
  original spatial seed after ROI-center refinement.
- Reuse the new figure in the QC dashboard when it was already produced in `figures/`.

## Validation

- Generated the montage for the supplied 2026-07-17 Clathrin run at
  `figures/roi_detection_replay_quantile_samples.png`.
- Focused fit-review/localization tests passed (11 tests), and focused dashboard/pipeline tests
  passed (14 tests).
- Ruff and scoped type checking of the changed modules passed.
