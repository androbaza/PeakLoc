# Add reproducible blink-photophysics deconstruction

## Motivation

The temporal QC distribution can show ROI first-to-last event spans far longer than reported AF647
ON-state lifetimes under high-intensity SMLM conditions. The existing plot correctly calls these
values proxies, but it does not expose which raw events set the endpoints or how the detector window
and polarity gates produce them.

## Changes

- Add a Nature-family publication-quality standard for scientific figures to `AGENTS.md`.
- Add `notebooks/blink_photophysics_deconstruction.ipynb` with AF647/event-camera context, exact
  timing definitions, deterministic sampling, RAW timestamp previews, figures, and run reports.
- Add a reusable analysis module that:
  - samples five accepted localizations from the 60th-90th event-count percentiles with seed 647;
  - replays the cumulative-polarity interpolation, prominence detection, and spline window;
  - recovers the exact ROI count window from stored positive/negative exposures;
  - streams only required RAW time/space regions while honoring diffuse-flash and spatial masks;
  - verifies first/last timestamps and polarity counts against saved ROI records; and
  - exports 450 dpi PNGs, vector PDFs, JSON/Markdown reports, and plotted source-data CSV files.

## Assumptions and interpretation

- The first-to-last ROI event span is not interpreted as a direct fluorophore ON-state lifetime.
- Positive and negative polarity gates are one-sided classifiers, not a symmetric 10 ms crop.
- AF647 comparisons are qualitative unless excitation intensity, buffer, event-camera calibration,
  and a molecular kinetic model are matched.

## Validation

- Executed every notebook cell from a clean kernel with `nbconvert`.
- Reconstructed 15 blinks across the three requested recordings; all 15 matched saved `t_1st`,
  `t_last`, positive fit-event counts, and negative fit-event counts.
- Observed median ROI proxies: 324.5 ms (rapid blinking), 550.9 ms (405-induced switching), and
  1,858.2 ms (2025 recording).
- `pixi run -e dev ruff check --fix .`: pass.
- `pixi run -e dev ruff format .`: pass.
- Focused `ty` check for the new module and notebook: pass.
- Repository-wide `pixi run -e all ty check`: blocked by eight pre-existing diagnostics in
  `notebooks/raw_event_npz_explorer.ipynb`; no new-file diagnostics remain.

## Visual output

Before: only aggregate timing distributions and spatial maps were available.

After: each QC output contains five-panel blink deconstructions, a five-sample timing/count overview,
and 3D x-y-time event clouds in PNG and PDF. Each plotted value has a corresponding CSV source-data
artifact in `qc/photophysics_deconstruction/`.
