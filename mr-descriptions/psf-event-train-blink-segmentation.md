# Segment compact ON/OFF event trains for blink timing

## Motivation

Spline-derived ROI bounds admit background events and neighbouring cycles, inflating the temporal
proxy. This change adds a conservative event-level estimator while retaining the legacy path.

## Changes

- Add two-pass transition-train segmentation with 4.25 px discovery and 3.5 px refinement cores.
- Use polarity-specific interevent gaps: 3 ms for ON and 8 ms for OFF.
- Gate trains by event count, active pixels, polarity purity, radial background density, and signed
  root Poisson deviance.
- Select zero or one seed-anchored, spatially matched ON/OFF pair and reject missing or ambiguous
  evidence explicitly.
- Materialize the full rectangular fit ROI within polarity-specific temporal supports.
- Persist segmented timing provenance and prefer it in temporal QC, with a labelled legacy fallback.
- Extend photophysics deconstruction with exact context events, detection curves, train/interval
  tables, selected fit events, nearby-seed diagnostics, and PNG/PDF figures.
- Keep `temporal_segmentation_enabled=false` until slice halos, duplicate ownership, and held-out
  validation are complete.

The former shared `temporal_max_interevent_gap_us` configuration key becomes
`temporal_max_on_interevent_gap_us` and `temporal_max_off_interevent_gap_us`.

## Real-data subsample validation

- Saved deconstructions: 10/15 accepted; median selected cycle 142.5685 ms versus 523.067 ms for
  the accepted samples' legacy spans, with 74.678% median per-candidate reduction.
- Fresh deterministic sample: 14/18 accepted across low/middle/high event-count strata
  (5/6, 5/6, and 4/6); median selected cycle 93.4 ms.
- Rapid `blink_05`: ON begins 36.214 ms before the seed; selected cycle is 147.547 ms.
- 2025 `blink_02`: ON begins 36.316 ms before the seed; selected cycle is 144.272 ms.
- 2025 `blink_01`: the weak central seed is rejected; persisted seeds at -269.623 ms and
  +358.177 ms form separate accepted intervals.
- Exact legacy RAW count/timestamp reconstruction passes for 15/15 samples.
- Context-event flags reproduce selected train counts, and selected-event rows reproduce every
  interval count and overlap/new/excluded statistic.
- No full recording or full pipeline run was performed.

## Performance

Warm direct segmentation took a median 0.425 ms/candidate (p95 0.670 ms). Warm
index/extract/segment/materialize processing took 1.15 ms/candidate on the saved fixtures. These
subsampled timings exclude nonlinear fitting, peak memory, dense-slice contention, and full-run
scheduling.

## Validation

- `39 passed` across the focused segmentation, exact QC crop, configuration, pipeline wiring,
  localization provenance, and temporal-QC tests.
- `ruff check --fix .` and `ruff format .` pass.
- `ty check` passes for all changed feature modules. Repository-wide `ty check` still reports eight
  pre-existing diagnostics in `notebooks/raw_event_npz_explorer.ipynb`, which this change does not
  modify.
- All three five-sample photophysics QC sets were regenerated using bounded per-sample RAW reads.

## Limitations

The 33 candidates are event-rich, not expert-labelled, and partly informed threshold tuning. The
results establish conservative, shorter train-aligned temporal selection on these samples, not
molecular lifetime accuracy, localization improvement, or population-level sensitivity. Segmented
maps were not refitted. Slice halos, duplicate pair ownership, weak/crowded labels, all-pair-edge
exports, held-out recording validation, and memory/full-run benchmarks remain required before the
feature can be enabled by default.

## Figures and source data

The detailed figures include legacy and segmented timing in the same panel for a direct comparison:

- `data/2026_07_15_Microtubule_Recordings/Normal_vs_Rapid/Rapid_Blinking_Only/recording_2026-07-15_14-01-32/20260716_193311_378386/qc/photophysics_deconstruction/blink_05_deconstruction.png`
- `data/2025_rec/MT_5May_S2_reduced_bias/20260716_210951_071893/qc/photophysics_deconstruction/blink_02_deconstruction.png`
- `data/2025_rec/MT_5May_S2_reduced_bias/20260716_210951_071893/qc/photophysics_deconstruction/blink_01_deconstruction.png`
- `data/2025_rec/MT_5May_S2_reduced_bias/20260716_210951_071893/qc/photophysics_deconstruction/nearby_seed_intervals.png`

Each QC directory contains vector PDFs plus `segmentation_context_events.csv`,
`detection_trace.csv`, `transition_trains.csv`, `blink_intervals.csv`,
`segmented_fit_events.csv`, `nearby_seed_intervals.csv`, and `analysis_summary.json`.
