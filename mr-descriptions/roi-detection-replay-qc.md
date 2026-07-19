# Add ROI detection-replay QC montage

## Motivation

The uncertainty quantile montage exposed the fitted ROI image but not the temporal decision that
materialized it. Reviewers could not tell which detected peak, positive window, and negative window
produced a displayed ROI.

## Changes

- Add `roi_detection_replay_quantile_samples.png` next to the uncertainty quantile montage, using
  the identical quantile and failed-fit sample selection.
- Give each sample a spatial ROI panel with detector-seed and fitted-center markers, plus a timing
  panel. The timing panel shows compact binned cumulative positive/negative ROI event traces,
  marks the cycle peak and original detector peak where distinct, and brackets the exact selected
  positive/negative temporal support windows.
- Preserve 128-bin, per-polarity selected-event histograms in temporally segmented ROI records so
  the cumulative traces are based on the events that actually form the displayed pixel images.
- Carry temporal seed coordinates into attempted localization records so a new run can show the
  original spatial seed after ROI-center refinement.
- Reuse the new figure in the QC dashboard when it was already produced in `figures/`.

## Validation

- Generated the montage for the supplied 2026-07-17 Clathrin run at
  `figures/roi_detection_replay_quantile_samples.png`.
- Focused fit-review/localization tests passed (11 tests), and focused dashboard/pipeline tests
  passed (14 tests).
- Ruff and scoped type checking of the changed modules passed.

## Follow-up refinement

- Replaced the initially ambiguous shaded timing bars with thin start/stop brackets and cumulative
  event curves. A detector-peak line is now only drawn when it differs from the cycle peak, so the
  figure never promises a separate line that is coincident and therefore invisible.
