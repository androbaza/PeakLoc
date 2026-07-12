# Full-resolution Temporal Blink Visualization

## Motivation

The final SMLM preview was smaller than its quantitative TIFF, and the temporal
blink QC combined several coarse maps into one static figure. The interactive
spatial timing view also auto-ranged a sampled trace, which made the final
fractions of the five-minute recording hard to inspect.

## Changes

- Render the PNG preview on the same pixel canvas as the TIFF, apply display-only
  gamma 2.0 correction, and reduce the scale-bar length from 20% to 10%.
- Replace the combined temporal spatial PNG with three tick-free native-sensor
  maps: turn-on, on-duration, and turn-off. Each map uses per-pixel median timing
  values and no fixed 300-unit color limit.
- Make the existing spatial 3D Plotly visualization use full-data time/color axis
  ranges, while retaining deterministic point sampling only for responsive orbiting.
- Add self-contained Plotly artifacts for:
  - a full-data 2D spatial timing map with continuous color-range, size, and opacity
    controls;
  - median temporal values per occupied sensor pixel in the 5x SMLM-render
    coordinate system;
  - time-binned temporal-dynamics trends through the acquisition; and
  - all accepted localizations on x/y/peak-time axes.
- Add z-range, color-range, marker, opacity, and camera controls to the 3D views.

## Generated recording artifacts

Generated directly from the completed localization table at
`data/5_minute_Recording_Rapid_Switching_From_Power_Increase/recording_2026-07-09_15-01-56/20260712_200539_675732`;
the RAW pipeline was not rerun.

- Final PNG/TIFF dimensions: `5895 x 3585`.
- Sensor temporal maps: `1280 x 720` each.
- Full turn-on range: `0` to `300.736018 s`; full turn-off range: `0.763744` to
  `300.931288 s`.
- Occupied native sensor pixels used by the median plot: `84,445`.

## Validation

- `pixi run -e dev pytest localization_scripts/tests/test_smlm_visualization.py localization_scripts/tests/test_qc_dashboard.py`
  — 6 passed.
- `pixi run -e dev ruff check --fix .` — passed.
- `pixi run -e dev ruff format .` — 70 files unchanged.
- `pixi run -e all ty check localization_scripts/blink_temporal_qc.py localization_scripts/smlm_visualization.py`
  — passed.
- Full `pixi run -e all ty check` still reports eight existing diagnostics in
  `notebooks/raw_event_npz_explorer.ipynb`; no changed module is implicated.
