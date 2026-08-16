# Live progress tab

## Outcome

Add a sixth desktop-GUI tab that observes an active PeakLoc pipeline run without sharing
mutable state with, blocking, or being able to fail the measurement subprocess. The tab shows
the full recording range and selected slice range, completed/active/pending slices, accumulated
localizations, sampled cumulative-sum peak traces, and sampled blink ROIs with adjustable event
time bounds.

## Design

- The GUI creates a unique progress directory for every non-preflight pipeline launch and passes
  it through `PEAKLOC_LIVE_PROGRESS_DIR`.
- Pipeline code treats the progress channel as optional. Every write is best-effort, catches
  filesystem/serialization failures, and logs a warning without changing the scientific result.
- A recording manifest and per-slice state/snapshot files are written with replace-based atomic
  updates. Concurrent slice workers only write uniquely named files.
- Snapshots are bounded: localization positions contain only `x`/`y`; peak traces and ROI event
  records use deterministic sample limits and downsampling. The GUI bins positions into a sensor
  image and releases each loaded slice payload.
- The Tk thread polls at a throttled interval. It never reads partially written files, never
  touches worker-owned arrays, and catches malformed/missing payloads inside the monitor.

## User interface

- A recording timeline distinguishes unselected, pending, active, completed, and failed slice
  regions and reports processed-slice and localization counts.
- An accumulated localization density image uses `image[y, x]` and physical x/y labels. Standard
  Matplotlib navigation provides pan/zoom; clicking selects the nearest sampled ROI.
- A peak-trace selector displays sampled cumulative event sums for retained peak-center pixels,
  with peak and extracted interval markers.
- An ROI selector and two time sliders filter the exact sampled events. The ON, OFF, and signed
  emitter matrices and temporal event counts update immediately. Invalid slider ordering is
  normalized without affecting pipeline data.

## Validation

- Exercise loader/model code against absent, partial, malformed, and complete temporary progress
  directories and confirm errors remain monitor-local.
- Launch the GUI far enough to construct all six tabs and drive one synthetic progress refresh.
- Run focused existing interface and pipeline-runner tests only.
- Run `pixi run -e dev ruff check --fix .`, `pixi run -e dev ruff format .`, and
  `pixi run -e all ty check`.

## Scope boundary

This change observes and samples existing intermediate arrays. It does not change peak selection,
ROI generation, localization fitting, final artifacts, cancellation semantics, or configuration
meaning.
