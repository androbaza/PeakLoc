# Artifact-output overhaul

## Motivation

PeakLoc runs previously mixed collaborator-facing images, technical QC, raw arrays,
and provenance at the run root. The July 17 Clathrin run showed the practical result:
ambiguous numbered files, duplicated figures, transparent sparse temporal maps, and
drift-correction outputs that should not be part of the standard hand-off.

## Changes

- Establish `share/{figures,statistics,metadata}` for collaborator hand-off and
  `debug/{arrays,qc,reports,provenance,temp_files}` for audit artifacts.
- Add a cropped high-resolution SMLM reconstruction, summary figure, temporal dynamics
  figures/statistics, FRC result, README, and HTML index to `share/`.
- Add descriptive debug QC filenames and fit-review montages that screen uncertainty
  quantiles, failed fits, and single-pixel-dominated ROI candidates.
- Render opaque, units-labelled temporal spatial maps and cap timing-display histograms
  at 1000 ms while retaining outlier counts in statistics.
- Remove drift correction and its rendered outputs; FRC now evaluates the accepted
  localizations directly.
- Migrate provenance, photophysics, plotting, configuration, and documentation to the
  new layout while retaining legacy-read fallbacks where needed.

## Validation

- `pixi run -e dev ruff check --fix` on changed production files: passed.
- `pixi run -e dev ruff format` on changed production files: passed.
- Targeted `pixi run -e all ty check` on changed production files: passed.
- Full `ty check` still reports eight pre-existing diagnostics in
  `notebooks/raw_event_npz_explorer.ipynb`.
- No plot tests were written or run, and the historical July 17 output was not modified.
