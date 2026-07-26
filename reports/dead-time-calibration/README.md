# Event-camera dead-time calibration at 50 Hz

## Recommendation

Use the camera's **Maximum dead-time** setting (`bias_refr = -20`) as the
starting setting for blinking-fluorophore recordings. In this bright-bead
calibration it recovered every expected cycle with the legacy workflow, gave
the most accurate transition-train boundary spacing, and reduced event load by
about 2.1-fold relative to Default.

Before committing to a long fluorophore acquisition, run a short representative
slice with Maximum and Default. Keep Maximum if dim emitters remain detectable;
use Default if Maximum loses weak blinks. Do not use Setting 127 or Minimum for
routine blinking recordings: neither improved cycle recovery, while they
increased event load by 8.4-fold and 30-fold, respectively, relative to
Maximum.

For the normal PeakLoc extraction path, keep the simpler legacy ROI workflow.
Use transition-train segmentation when physically interpretable ON/OFF
boundaries are needed for QC or photophysics. The two workflows have comparable
cadence recovery after the shared peak-suppression fix, while the legacy path
is simpler and recovered more complete cycles.

## Experimental reference

- Sample: 100 nm TetraSpeck beads in one field of view.
- Excitation: 50 Hz square wave, 50% duty cycle.
- Analyzed interval: 1.000–5.000 s from each recording.
- Ground truth per recording: 200 cycles, 20 ms period, nominal 10 ms bright
  interval.
- Spatial reference: eight repeatedly detected beads, giving 1,600 expected
  bead-cycles per recording.

The function-generator frequency and duty cycle constrain cadence and duration,
but absolute optical phase was not recorded. Phase was therefore estimated
independently for each recording from its phase-locked peak train.

## What initially failed

Both production extraction workflows initially produced only 7–8 ROIs from a
recording that contains 200 flashes per bead. The failure occurred before ROI
extraction.

The shared peak filter built connected components from candidates less than
`peak_time_threshold` apart. With the configured 40 ms threshold, successive
20 ms laser cycles formed a transitive chain: cycle 1 linked to cycle 2, cycle
2 to cycle 3, and so on. The entire four-second train then collapsed to one
winner per spatial bead group.

Peak filtering now uses deterministic strongest-first local suppression.
A candidate is suppressed only by a directly nearby peak that has already been
retained. A chain of periodic peaks can no longer merge end-to-end.

For this 50 Hz calibration, `peak_time_threshold = 10,000 µs` and
`peak_neighbors = 9` were used. The temporal radius matches the nominal
transition width and is half the 20 ms cycle; the spatial radius suppresses
multiple candidates across each bead PSF. These are calibration-specific
values, not replacements for the 40 ms baseline used for slower fluorophore
data.

## Results

| Dead-time setting | `bias_refr` | Events, 1–5 s | Load vs Maximum | Legacy matched cycles | Transition-train matched cycles | Transition spacing, median (10th–90th) | Median absolute duration error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maximum | -20 | 0.461 M | 1.0× | 1,600/1,600 (100.0%) | 1,571/1,600 (98.2%) | 10.077 ms (9.787–10.445) | 0.164 ms |
| Default | 0 | 0.968 M | 2.1× | 1,600/1,600 (100.0%) | 1,568/1,600 (98.0%) | 10.345 ms (10.044–10.754) | 0.349 ms |
| Setting 127 | 127 | 3.887 M | 8.4× | 1,598/1,600 (99.9%) | 1,510/1,600 (94.4%) | 10.343 ms (10.079–10.912) | 0.343 ms |
| Minimum | 235 | 13.866 M | 30.1× | 1,592/1,600 (99.5%) | 1,444/1,600 (90.3%) | 10.330 ms (10.074–10.811) | 0.330 ms |

Maximum had the lowest event load and the best physical boundary estimate. Its
transition-train median absolute duration error was 0.164 ms and its 90th
percentile absolute error was 0.498 ms. Maximum also retained a median of 123
events per legacy ROI and localized all 1,600 legacy ROIs, so reduced event
load did not compromise this bright reference.

Minimum produced 13.87 million events in four seconds, missed eight legacy
bead-cycles, and had the lowest transition-pair recovery. The high rate
increased processing time from about 17 s at Maximum to about 79 s at Minimum
without improving the known 50 Hz result.

## Workflow comparison

### Legacy spline-window extraction

- Recovered 99.5–100% of expected bead-cycles.
- Is the preferred production path here because it is simpler and more
  complete.
- Its stored first-to-last event span is constrained by the ±5 ms polarity
  gates. The close-to-10 ms legacy span is therefore not an independent
  validation of optical endpoints.

### Transition-train segmentation

- Recovered 90.3–98.2% of expected bead-cycles.
- Provides the physically meaningful quantity used above: first event of the
  positive transition to first event of the matched negative transition.
- Correctly measured approximately 10 ms bright intervals.
- Lost cycles at the slice-context edges and rejected additional pairs at the
  high-rate Setting 127 and Minimum settings.

For a 50 Hz timing-QC run, the successful transition-train settings were:

```json
{
  "peak_time_threshold": 10000.0,
  "peak_neighbors": 9,
  "temporal_context_pre_us": 30000,
  "temporal_context_post_us": 30000,
  "temporal_max_cycle_span_us": 30000,
  "temporal_max_train_duration_us": 15000,
  "temporal_max_on_end_after_seed_us": 5000,
  "temporal_max_off_start_before_seed_us": 15000
}
```

These limits encode the known 20 ms calibration cycle and should not be copied
unchanged to fluorophores with unknown or slower blinking kinetics.

## Production changes

1. Peak candidate grouping was changed from transitive connected components to
   strongest-first local suppression. This is the key blink-recovery fix.
2. RAW decoding now seeks to `slice_start` and stops at `slice_end` (or at the
   requested `slice_count` boundary). Minimum previously decoded all 251 million
   events in the one-minute file just to analyze 13.9 million events from 1–5 s.
3. The currently selected calibration config uses
   `peak_time_threshold = 10,000 µs`, `peak_neighbors = 9`, and the simpler
   legacy workflow.

## Limitations

- The beads are much brighter and more repeatable than single fluorophores.
  This experiment calibrates timing and event-rate cost, not the detection
  sensitivity of Maximum dead time for dim emitters.
- No photodiode or oscilloscope trace was recorded with the event stream.
  Frequency and duty cycle are known, but absolute boundary offsets cannot be
  measured independently.
- Only one field of view and one four-second interval were compared.
- `bias_refr` is reported as the camera bias value. No unsupported conversion
  from that device-specific value to an absolute refractory time is made.

## Reproduction and artifacts

Regenerate the tables and figures after calibrated PeakLoc runs with:

```bash
pixi run python -m scripts.dead_time_calibration
```

- [Calibration figure (PNG)](dead_time_calibration.png)
- [Calibration figure (vector PDF)](dead_time_calibration.pdf)
- [Summary source data](dead_time_calibration_summary.csv)
- [Per-ROI figure source data](dead_time_calibration_measurements.csv)
- [Ground truth, bead centers, phases, and run manifest](dead_time_calibration_manifest.json)
