# Full-minute legacy dead-time comparison

## Motivation

Validate the earlier 1–5 s dead-time recommendation over the complete 0–60 s bead recordings and
quantify how Maximum versus Default dead time changes positive, negative, and total ROI event loads.

## Changes

- Added a reproducible legacy-only Maximum/Default report generator.
- Added deterministic five-blink cumulative-polarity detector replays per setting.
- Added setting-level and ROI-level source tables, including polarity-resolved event-count
  distributions.
- Added publication-ready PNG/PDF comparison and trace figures.
- Added a full report and manifest tied to the exact completed run directories and random seeds.

## Parameters and assumptions

- Ground truth: 50 Hz, 50% duty cycle, 20 ms period, 10 ms bright interval.
- Interval: `[0, 60 s)`.
- Eight dominant bead clusters; 24,000 expected bead-cycles per recording.
- Legacy extraction only, with `peak_time_threshold = 10 ms` and `peak_neighbors = 9`.
- Trace samples use fixed seeds 50,647 (Default) and 50,648 (Maximum).

## Validation

- The generator completed against both full-minute runs.
- Focused Ruff formatting/lint and ty checks pass for the new script.
- Generated tables contain 48,087 ROI rows and ten reproducibly selected blink traces.

## Result

Maximum and Default both recover 23,992/24,000 unique bead-cycles (99.967%). Maximum reduces RAW
events by 49.2%, processing time by 30.1%, and peak memory by 32.4%, while increasing accepted fit
yield from 96.62% to 99.80%. Median total events per ROI fall from 175 to 123 but remain sufficient
for the configured joint positive/negative fit.
