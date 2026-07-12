# Reject diffuse focus-light flashes before ROI output

## Motivation

The rapid-switching recording accepted broad focus-light changes as point
localizations. In particular, accepted IDs `81487` and `221482` were spatially
uniform background bursts rather than compact blinks.

## Changes

- Added a conservative pre-output ROI gate for diffuse positive-event flashes.
- The gate skips a candidate only when all of the following are true:
  - at least 1,000 positive events are in the ROI;
  - at least 90% of ROI pixels are active; and
  - the brightest 3×3 patch contains no more than 10% of positive events.
- The rejected candidate never enters the returned ROI array, ROI artifact, or fitting
  queue. The run report records per-slice and total skip counts.
- Added validated configuration fields and documented the audit disable switch.

## Evidence and safeguards

On the completed 5-minute rapid-switching run:

- ID `81487`: 3,162 positive events, 100% active pixels, 4.0% in its brightest 3×3
  patch → rejected by the new gate.
- ID `221482`: 3,995 positive events, 100% active pixels, 4.6% in its brightest 3×3
  patch → rejected by the new gate.
- Replaying the rule against saved ROI data marks 1,713 of 246,605 candidates (0.69%),
  concentrated at the periodic illumination-change times.
- A high-count compact control (1,233 events, 74.2% in its brightest 3×3 patch) is
  retained. The gate cannot affect candidates below 1,000 positive events.

## Validation

- Focused tests: 28 passed across ROI generation, configuration, pipeline runner, and
  peak finding.
- Existing synthetic blink end-to-end test passed.
- Direct ROI-generation probe: a 1,156-event uniform burst was skipped, while a
  1,350-event compact blink was retained; disabling the gate retained the uniform
  burst for audit comparison.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff check --fix .` passed.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff format .` passed.
- `PYTHONNOUSERSITE=1 pixi run -e all ty check localization_scripts` passed.
- Full `ty check` remains blocked only by 8 pre-existing diagnostics in the modified
  `notebooks/raw_event_npz_explorer.ipynb`.
