# PeakLoc Next Crucial Features

## Motivation

PeakLoc runs needed stronger scientific auditability: pre-run configuration
checks, explicit fit rejection reasons, reviewable uncertainty diagnostics,
run-level QC outputs, systematic tuning tools, post-processing resolution
metrics, realistic synthetic benchmarks, and portable provenance metadata.

## Changes

- Added preflight configuration and input audits with CLI modes for preflight,
  strict preflight, and preflight-only execution.
- Saved full attempted-localization QC tables with primary rejection reasons in
  NumPy and CSV formats.
- Added uncertainty-ranked fit review montages for low, high, quantile, and
  polarity-specific fit review.
- Added run-level QC summaries, static figures, and HTML dashboard generation.
- Added parameter sweep support with JSON/CSV result tables and tuning figures.
- Added tested drift correction, FRC resolution estimation, and post-processing
  QC artifacts.
- Added a realistic synthetic event-camera generator for blink benchmark tests.
- Added duplicate localization merging and conservative overlap/multi-emitter
  flags.
- Added portable accepted/attempted localization CSV outputs and run provenance
  metadata.
- Centralized plotting standards and removed hard-coded helper-script paths.
- Extended run reports with scientific validation status, warnings, QC links,
  drift/FRC status, and rejection summaries.

## Assumptions

- CSV is used for portable localization tables before adding an optional Parquet
  dependency.
- The first drift and FRC implementations are conservative global diagnostics;
  local FRC can be added after the global path is stable on real data.
- Multi-emitter handling starts with explicit flags and duplicate suppression
  rather than silently reporting a validated two-emitter fit.
- Local-only background without calibration is treated as exploratory unless the
  run configuration explicitly justifies it.

## Validation

```bash
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e all ty check
pixi run -e dev pytest
```

- `ruff check --fix`: passed.
- `ruff format`: 61 files left unchanged.
- `ty check`: passed before and after the final merge.
- `pytest`: 110 passed, 1 xfailed.
