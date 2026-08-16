# feat: refine live measurement data exploration

## Motivation

The original sixth tab placed its controls, timeline, reconstruction, peak trace, ROI map, and
event timing on one wide canvas. This made ROI exploration distant from its controls and caused the
slice legend to collide with plot content in compact windows. Polarity-specific inspection and
plain-language live quality context were also missing.

## Changes

- Separate the live workspace into Progress, Reconstruction, and Signals & ROI subpages.
- Move the progress key into a responsive two-row Tk layout outside the Matplotlib axes.
- Add Both, Only ON, and Only OFF inspection modes for the ROI map and timing histogram.
- Keep peak/ROI selectors and manual event-window sliders directly above their plots.
- Explain the yellow extracted-interval shading and detected-peak line in the interface.
- Add bounded descriptive summaries for slice state, reconstruction occupancy/density, peak
  prominence, extracted interval, manual window duration, event count, and ON fraction.
- Open Signals & ROI automatically when a sampled ROI is selected from the reconstruction.
- Default `roi_radius` to 5 px (11 x 11 pixels).
- Default CPU and per-slice worker limits to `max(CPU_COUNT - 1, 1)` while preserving explicit
  configuration and affinity/resource caps.
- Update the portable config, settings help, desktop/configuration documentation, and live preview.

## Scientific and stability assumptions

- Both mode displays signed ON-minus-OFF counts; single-polarity modes display nonnegative counts.
- The yellow trace background is the algorithm-extracted ON-to-OFF interval, not a user selection.
- Live controls operate only on bounded copied diagnostic samples and never update fit results or
  saved outputs.
- Monitor snapshot validation, allow-pickle-free loading, and the GUI exception boundary remain in
  place, so malformed live data cannot terminate measurement processing.
- Descriptive summaries are for exploration and are explicitly not acceptance thresholds.

## Validation

- `pixi run -e dev ruff check interface/live_monitor.py interface/config_catalog.py
  localization_scripts/pipeline_config.py`
- `pixi run -e dev ruff format --check interface/live_monitor.py interface/config_catalog.py
  localization_scripts/pipeline_config.py`
- `pixi run python -m py_compile interface/live_monitor.py interface/config_catalog.py
  localization_scripts/pipeline_config.py`
- `pixi run -e dev pytest localization_scripts/tests/test_pipeline_config.py
  interface/tests/test_entrypoint.py -q` -> 13 passed
- Focused Tk/Matplotlib smoke -> dynamic defaults, three subpages, 900/1120 px content widths,
  three polarity modes, summary content, and malformed snapshot isolation passed.
- Direct `pixi run -e all ty check` and the `typecheck` task were attempted; this Windows Pixi
  environment does not provide the `ty` executable.

## Visual review

- `figures/live-measurement-tab-preview.png`
- `figures/live-measurement-tab-preview.svg`
- `figures/live-measurement-tab-preview-source.npz`

The updated preview shows the Signals & ROI scientific plots. The prior combined preview in Git
history provides the before view; the desktop guide documents the new page and control structure.
