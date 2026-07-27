# Full-minute Maximum vs Default dead-time comparison

## Result

Using the calibrated **legacy-only** workflow on exactly 0–60 s, both settings recovered essentially
all of the expected 50 Hz bead cycles. Maximum dead time reduced RAW event traffic by
**49.2%** and completed the slice **1.43× faster**, while preserving
99.967% unique bead-cycle recovery and a
99.80% localization fit yield.

## Direct comparison

| Metric | Default | Maximum |
|---|---:|---:|
| RAW events, 0–60 s | 13,407,884 | 6,806,496 |
| Retained legacy ROIs | 24,024 | 24,063 |
| Accepted localizations | 23,211 | 24,014 |
| Unique matched bead-cycles / 24,000 | 23,992 | 23,992 |
| Bead-cycle recall | 99.967% | 99.967% |
| Duplicate detections | 0 | 1 |
| Unassigned detections | 32 | 70 |
| Fit yield | 96.62% | 99.80% |
| Median absolute 50 Hz phase error | 66 µs | 72 µs |
| Median legacy first-to-last window | 9.92 ms | 9.90 ms |
| Processing time | 492.1 s | 344.1 s |
| Peak resident memory | 7.81 GiB | 5.28 GiB |

## Events per ROI

Whisker ranges below are the 10th–90th percentiles; means and all additional quartiles are in
`full_recording_summary.csv`.

| Polarity count | Default median (P10–P90) | Maximum median (P10–P90) |
|---|---:|---:|
| Positive events | 119 (79–141) | 76 (61–89) |
| Negative events | 56 (14–71) | 45 (26–58) |
| Total events | 175 (93–209) | 123 (88–145) |

Maximum therefore lowers the typical positive-lobe event count more strongly than Default, but the
remaining counts are sufficient for the configured joint positive/negative fit. The separate
nonzero-ROI counts are included in the summary table so zero-count tails are not hidden.

## How legacy start/end extraction is visualized

For each setting, five ROIs were sampled reproducibly at random from all non-edge legacy detections
with both polarities present. Each trace plot shows:

1. the raw cumulative polarity step trace (+1 for a positive event, −1 for a negative event);
2. the linearly interpolated cumulative trace used to locate a prominent maximum;
3. the spline-curvature-derived start/stop interval (blue shading);
4. the retained peak (red line); and
5. the first and last events actually stored in the final gated ROI (green dashed lines).

The spline interval proposes the temporal window. Legacy ROI generation then expands it when needed
to include the ±5 ms polarity gates, and `t_1st`/`t_last` are the first/last events found inside that
counting window. Consequently, the stored first-to-last duration is an event-window proxy, not a
direct optical measurement of the 10 ms laser-bright interval. The replay uses a local ±250 ms RAW
context, so its spline shading is a diagnostic reconstruction; the stored retained peak and stored
`t_1st`/`t_last` values remain the authoritative full-slice outputs.

## Recommendation

Use **Maximum dead time (`bias_refr = -20`)** for blinking-fluorophore acquisition under comparable
event rates. Across the full minute it keeps essentially complete 50 Hz recovery, has the higher fit
yield, halves event traffic, reduces peak memory, and shortens processing. Keep the calibrated
legacy settings (`peak_time_threshold = 10 ms`, `peak_neighbors = 9`) for this acquisition regime.
No new blink-extraction method is justified by this comparison.

The exact 50 Hz recovery result is specific to these bright, periodically driven 100 nm beads.
Before making Maximum the permanent fluorophore default, confirm it on a short sparse-fluorophore
recording because real blinks are dimmer, non-periodic, and lack this ground truth.

## Reproduction

```bash
pixi run python -m scripts.dead_time_full_recording_comparison
```

## Artifacts

- `full_recording_comparison.png` / `.pdf`: acquisition, recovery, polarity-resolved ROI counts, and
  blink-window comparison.
- `legacy_cumulative_traces_default.png` / `.pdf`: five random Default legacy replays.
- `legacy_cumulative_traces_maximum.png` / `.pdf`: five random Maximum legacy replays.
- `full_recording_summary.csv`: setting-level statistics.
- `roi_measurements.csv`: one row per ROI, including positive, negative, and total events.
- `trace_sample_selection.csv`: sampled ROI identities and extracted boundaries.
- `legacy_cumulative_trace_data.csv`: plotted cumulative-trace source values.
- `full_recording_manifest.json`: run paths, settings, ground truth, and random seeds.
