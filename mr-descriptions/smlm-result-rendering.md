# SMLM Result Rendering

## Motivation

PeakLoc generated diagnostic ROI fit plots but did not produce a publication-style
SMLM visualization of the final localization table. The output also lacked an
explicit optical pixel size setting for physical scale bars and downstream
coordinate conversion.

## Changes

- Added `optical_pixel_size_nm` to `config.json` and `PeakLocConfig`, defaulting
  to 67 nm.
- Added `plot_result` to `config.json` and `PeakLocConfig`, defaulting to `true`.
- Added reusable SMLM rendering helpers in
  `localization_scripts/smlm_visualization.py`.
- Added standalone CLI script `scripts/plot_localizations.py`.
- Added pixi task `plot-result`.
- Integrated SMLM result rendering into the pipeline when `plot_result` is true.
- Saves:
  - 8-bit PNG preview with scalebar.
  - 12-bit-valued TIFF render stored as `uint16` and capped at 4095.
- Added tests for config validation and SMLM rendering.
- Documented output `.npy` files and downstream processing in README.

## Validation

```bash
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e dev typecheck
pixi run -e dev pytest
pixi run python -m py_compile scripts/plot_localizations.py localization_scripts/smlm_visualization.py localization_scripts/pipeline_runner.py localization_scripts/pipeline_config.py
pixi run plot-result data/AF647_coverslip/localizations_prominence_fwhm_6_prominence_12.npy
pixi run peakloc
```

- `pytest`: 9 passed.
- Standalone plot task saved PNG and TIFF outputs.
- `peakloc`: completed and wrote SMLM outputs to `data/AF647_coverslip/figures/`.
- Verified TIFF output was `uint16` with max value `4095`.
