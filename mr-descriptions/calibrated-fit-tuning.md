# Motivation

Calibrated single-emitter fits could converge on unrelated structure at the edge of
a 17x17 ROI even when the detected PSF was visibly centered. Full six-parameter
Fisher conditioning also rejected converged, positionally precise fits because
background and amplitude nuisance parameters were poorly scaled.

# Changes

- Restrict the fit center to a configurable neighborhood around the detected peak.
- Fit from corrected center-of-mass and detected-center initializations, selecting
  the best converged finite-likelihood result.
- Evaluate Fisher conditioning on the marginalized positional covariance block.
- Tune the default calibrated configuration for the supplied recording and camera.
- Prevent disabled plotting/QC modes from rendering fit-review artifacts.
- Reuse uncertainty montages in the QC dashboard instead of rendering duplicates.
- Isolate spawned workers to the active pixi Python environment.
- Add configuration guidance and a plan for larger deferred speedups.

# Validation

On the calibrated 10-15 second slice with 9,495,871 events and 32 cores:

- Baseline: 9,725 / 9,799 accepted, 67 fit failures, 119 fits within two pixels of
  an ROI edge.
- Final: 9,777 / 9,799 accepted, 3 positional-condition failures, 19 uncertainty
  rejections, and no fits within two pixels of an ROI edge.
- For 83 centered candidates that previously migrated to an edge, 82 moved closer
  to the event center of mass; median error improved from 6.18 px to 1.87 px.
- On an independent 30-32 second slice, all 11 analogous cases improved; median
  error improved from 5.06 px to 0.91 px.
- Final fast-review runtime was 130.86 seconds versus 147.63 seconds before output
  gating and 235.17 seconds for the full baseline QC run.

# Assumptions

- Calibration and experiment use matching bias settings.
- The detected temporal peak identifies the intended emitter within three pixels.
- The 50 nm uncertainty threshold remains the absolute-confidence filter.

# Remaining Work

Bounded RAW streaming, explicit single-slice semantics, sparse active-coordinate
detection, and ROI worker data reuse are described in `plans/pipeline_speedups.md`.
