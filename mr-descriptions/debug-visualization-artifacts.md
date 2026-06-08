# Publication-quality synthetic debug artifacts

## Motivation

The synthetic blink tests now produce deterministic visual artifacts that make
localization outcomes auditable beyond log output. Each scenario writes a fixed
artifact tree with matching tables, static figures, density TIFFs, an
interactive spacetime HTML plot, and a short Markdown report.

## Changes

- Added `localization_scripts/debug_visualization.py` with reusable debug
  dataclasses, safe marked-directory overwrite handling, truth/localization
  matching, CSV/JSON serialization, figure bundles, TIFF density outputs, and
  Plotly HTML spacetime output.
- Integrated artifact writing into
  `localization_scripts/tests/test_synthetic_blinks_pipeline.py` before the
  final numerical assertions, so failing and xfailed scenarios still leave
  diagnostics.
- Added focused tests for safe overwrite behavior, matching, artifact creation,
  and the `scatter(x, y)` overlay convention.
- Added `plotly` and conda-forge `python-kaleido` dependencies for interactive
  and export-capable visualization support.

## Parameters and assumptions

- Artifact filenames are deterministic and overwritten only inside directories
  marked with `.peakloc_debug_artifacts`.
- By default pytest writes artifacts under
  `debug_artifacts/synthetic_blinks/<scenario>/`, which is ignored by git.
  Set `PEAKLOC_DEBUG_ARTIFACT_DIR=/path/to/output` to write under a different
  persistent root.
- Image-array convention remains `image[y, x]`, and overlays use
  `scatter(x, y)`.
- The test assertions remain the oracle; figures are diagnostics and
  publication aids.

## Validation

```bash
pixi run -e dev pytest localization_scripts/tests/test_debug_visualization.py
pixi run -e dev pytest localization_scripts/tests/test_synthetic_blinks_pipeline.py
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e all ty check
```

Results:

- `test_debug_visualization.py`: 5 passed.
- `test_synthetic_blinks_pipeline.py`: 9 passed, 1 xfailed.
- Ruff check, Ruff format, and ty check passed.
