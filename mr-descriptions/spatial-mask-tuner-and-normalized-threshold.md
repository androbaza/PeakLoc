# Add a spatial-mask tuner and normalized density threshold

## Motivation

The previous `spatial_mask_min_events` setting was a raw event count. It changed
meaning when acquisition duration, bias, or global illumination changed, and there was
no direct way to inspect its mask before a full localization run. The component and
support-coverage safeguards were also difficult to interpret from their names alone.

## Changes

- Replaced `spatial_mask_min_events` with
  `spatial_mask_min_density_quotient`.
  - The raw seed cutoff is now `calibration events / sensor pixels × quotient`.
  - Mask metadata and run reports save the mean density and derived raw cutoff for
    auditability.
  - Old raw-count configurations fail with an explicit migration message instead of
    silently changing their selection behavior.
- Added `pixi run spatial-mask-tuner [recording.raw]`.
  - It reads only the configured calibration duration directly from RAW decoder chunks
    (or memory-mapped normalized NPY events).
  - It exposes sample duration, density quotient, minimum seed-component pixels,
    target margin, and maximum support coverage.
  - **Render sum + mask** shows the log density sum and seed/retained/target/support
    outlines, reports coverage/fallback state, and prints a copyable JSON snippet.
- Expanded configuration/tuning documentation, including the exact meaning of
  `spatial_mask_min_component_pixels` and `spatial_mask_max_support_coverage`.

## Measurements

For the rapid-switching recording's first minute (137,842,777 events):

- Quotient `2.7` derives a 403.84-event/pixel seed threshold.
- It retained 24.92% target and 32.25% support coverage, closely matching the prior
  400-event cutoff's 25.17% / 32.36% result.

For the current local recording, the existing 100-event cutoff converts to quotient
`1.36`: its measured mean is 73.75 events/pixel, deriving a 100.30-event threshold and
producing active 30.84% target / 34.04% support coverage.

## Validation

- `pixi run spatial-mask-tuner --help` exposes the new task and arguments.
- Headless synthetic tuner preview rendered both density and target/support overlays.
- A synthetic pipeline-level mask probe produced an active mask and recorded normalized
  threshold metadata.
- Focused existing tests: 28 passed across configuration, pipeline runner, ROI
  generation, and peak finding; the synthetic blink end-to-end test also passed.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff check --fix .` and
  `PYTHONNOUSERSITE=1 pixi run -e dev ruff format .` passed.
- `PYTHONNOUSERSITE=1 pixi run -e all ty check localization_scripts
  scripts/spatial_mask_tuner.py` passed.
