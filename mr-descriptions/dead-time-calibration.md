# Calibrate blink extraction and camera dead time at 50 Hz

## Motivation

The pulsed 100 nm bead recordings contain 200 known laser cycles per bead in
the selected 1–5 s interval, but both PeakLoc extraction paths initially
returned only 7–8 ROIs. Peak candidate grouping used transitive connected
components, so adjacent 20 ms cycles linked into one four-second group under
the previous 40 ms threshold.

## Changes

- Replace transitive peak grouping with deterministic strongest-first local
  suppression.
- Make RAW decoding seek to `slice_start` and stop at `slice_end` or the
  requested `slice_count` boundary.
- Calibrate the selected 50 Hz config to a 10 ms temporal and 9-pixel spatial
  suppression radius while retaining the simpler legacy ROI workflow.
- Add a reproducible dead-time analysis command that discovers matching runs,
  aligns them to the 50 Hz reference, assigns eight bead positions, writes
  per-ROI and summary source data, and exports PNG/PDF figures.
- Document method performance, limitations, and the camera recommendation.

## Findings and recommendation

- Maximum dead time (`bias_refr = -20`) recovered 1,600/1,600 legacy
  bead-cycles using 0.461 million events.
- Its transition-train path recovered 1,571/1,600 bead-cycles and measured a
  10.077 ms median positive-to-negative transition spacing, with 0.164 ms
  median absolute error from the 10 ms reference.
- Default used 2.1-fold more events; Setting 127 and Minimum used 8.4-fold and
  30.1-fold more without improving cadence or boundary accuracy.
- Use Maximum as the starting setting, but compare it with Default on a short
  fluorophore slice because bright beads do not measure dim-emitter
  sensitivity. Avoid Setting 127 and Minimum without separate sensitivity
  evidence.
- Prefer legacy extraction for routine localization. Use transition trains when
  physically interpretable ON/OFF boundaries are required.

## Validation

- `pixi run python -m scripts.dead_time_calibration` — passed and regenerated
  all committed tables and figures.
- Real Minimum RAW bounded decode — 13,865,974 events, timestamps
  1,000,000–4,999,999 µs.
- Focused pytest set — 27 passed:
  `test_peak_finding.py`, `test_event_array_processing.py`,
  `test_pipeline_config.py`, and the non-overlapping slice-bound task test.
- Full focused pipeline-runner file — 8 passed, 3 pre-existing failures in an
  outdated slice stub, removed montage symbol, and old QC-report layout.
- `pixi run -e dev ruff check --fix .` — passed.
- `pixi run -e dev ruff format .` — completed.
- Focused `ty check` on all new/changed analysis logic — passed.
- Full `pixi run -e all ty check` — 23 pre-existing diagnostics, including
  platform-specific `pipeline_runner.py` types and notebook typing; none remain
  in the new analysis, bounded loader, or peak suppression modules.

## Visual output

The new four-panel figure compares event load, matched 50 Hz bead-cycles,
detected bright intervals, and phase-locked peak timing across all four
dead-time settings and both extraction methods. A 3289 × 2482 px, 450 dpi PNG
and a one-page vector PDF are saved with the plotted source data in
`reports/dead-time-calibration/`.
