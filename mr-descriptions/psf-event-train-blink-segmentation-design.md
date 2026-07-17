# Design PSF event-train blink segmentation

## Motivation

The current temporal ROI bounds are derived from spline interpolation indices and then populated by
one-sided polarity gates. Their physical width varies with event density, so background events and
neighbouring blink cycles can set ROI endpoints hundreds of milliseconds or seconds away from the
detected peak.

## Changes

- Document a two-pass temporal-segmentation design based on compact PSF-core event trains.
- Require a positive ON-transition train and a spatially matching negative OFF-transition train.
- Specify ordered matching that emits a separate ROI for every accepted blink cycle.
- Define strict initial background, polarity, spatial compactness, duration, and convergence gates.
- Separate AF647 ON-state kinetics from the event sensor's transition-train duration and use
  literature kinetics only as a condition-specific soft prior.
- Define an annotation, tuning, locked-validation, and Nature-quality diagnostic protocol.

## Evidence from the reference examples

- Rapid-blinking `blink_05` contains a dense positive train around -40 to 0 ms, its paired negative
  train after the peak, and an unrelated later positive cluster that should be excluded.
- 2025 `blink_02` contains one compact ON/OFF pair plus remote events extending beyond +/-2 s.
- 2025 `blink_01` contains at least three temporally ordered positive/negative cycles in the compact
  PSF core and should yield separate ROI records.

## Validation

- Reconstructed 1--20 ms polarity bins from the existing `raw_roi_events.csv` files and compared
  compact-core event timing for the three named references.
- Inspected the current spline boundary, peak suppression, ROI population, and polarity-gate logic.
- Cross-checked the duration-prior interpretation against primary AF647 switching literature.
- No runtime behavior was changed; implementation and quantitative validation remain explicit next
  stages in the design.
