# Add a failure-isolated live measurement tab

## Motivation

PeakLoc's desktop application previously exposed setup and worker logs but no visual view of an
active measurement. Users could not see which recording slices were selected or completed,
watch localizations accumulate, inspect extracted peak traces, or manually review the event-time
window of representative blink ROIs.

## Changes

- Add a sixth **Live measurement** tab with a full-recording slice timeline, accumulated
  localization density, selectable cumulative-sum peak traces, sampled blink ROIs, and ON/OFF
  event timing.
- Add interactive Matplotlib pan/zoom, localization-click ROI selection, peak/ROI dropdowns, and
  first/last event sliders that recompute the displayed signed emitter map from copied raw events.
- Add an optional `PEAKLOC_LIVE_PROGRESS_DIR` protocol. Pipeline and concurrent slice workers
  publish atomic manifests, state JSON, and allow-pickle-free NPZ snapshots only for GUI runs.
- Isolate every progress-publication and monitor-read failure from the authoritative processing
  subprocess. Bound peak traces, ROI events, displayed samples, sensor dimensions, and retained
  GUI data; compress snapshot files and clean their unique temporary directory when replaced or
  the application closes.
- Bundle Matplotlib's Tk backend in the Windows build and document the new workflow.

The observer uses the established `image[y, x]` convention. Localization density is accumulated
at sensor-pixel resolution for a stable live preview; final fitted coordinates and saved
scientific outputs are unchanged. Manual ROI slider adjustments affect copied diagnostic samples
only and are not written back into fitting or pipeline artifacts.

## Validation

- `pixi run -e dev pytest interface/tests -q` (5 passed)
- Focused `test_pipeline_runner.py` selection (8 passed, 3 known baseline failures deselected)
- Focused Ruff check/format on all changed Python modules (passed)
- Python compilation of all changed Python modules (passed)
- Synthetic protocol and GUI smoke coverage for six-tab construction, missing/partial/completed
  progress, localization accumulation, peak selection, ROI slider updates, and malformed data
  isolation (passed)
- `git diff --check` (passed)

Repository-wide validation remains limited by existing project state:

- `pixi run -e dev ruff check --fix .` reports 47 pre-existing violations outside this change.
- `pixi run -e all ty check` cannot run on Windows because `ty` is declared only for `linux-64`.
- The full focused pipeline-runner file has 13 passing and 3 pre-existing failing tests whose
  expectations already disagree with `HEAD`: the old temp-artifact layout, a removed
  `save_uncertainty_montages` symbol, and the old `run_qc_summary.json` filename.

## Visual review

Before: the application ended at tab 5 and had no live visualization.

After: `figures/live-measurement-tab-preview.png` (high-resolution raster) and
`figures/live-measurement-tab-preview.svg` (vector) show the implemented monitoring canvas with
synthetic progress. `figures/live-measurement-tab-preview-source.npz` contains the plotted source
data.
