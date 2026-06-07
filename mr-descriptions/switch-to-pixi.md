# Switch PeakLoc to Pixi

## Summary
- Replace the exported conda environment with a curated pixi workspace.
- Add runtime, dev, UI, and all-feature pixi environments.
- Add `pixi run peakloc` and `pixi run import-test` workflows.
- Make RAW loading avoid the previous multi-billion-event preallocation.
- Allow `PeakLoc.py` input folder and slicing parameters to be configured with environment variables.
- Update documentation for the pixi workflow and Ubuntu OpenEB prerequisite.

## Validation
Passed:

```bash
pixi install
pixi run import-test
pixi run python -c "import PeakLoc; print('peakloc imports ok')"
pixi run python -m py_compile PeakLoc.py scripts/*.py localization_scripts/*.py
pixi run peakloc
```

Known residuals:

```bash
pixi run -e dev test
```

collects zero tests and exits with pytest code 5.

```bash
pixi run -e dev lint
pixi run -e dev ruff format --check .
pixi run -e all typecheck
```

still fail on pre-existing legacy style and typing issues, including wildcard imports, unused imports, formatting drift, unresolved exploratory UI references, Numba `prange` typing, and the untyped OpenEB extension module.

## Notes
- `pixi.toml` keeps Python at 3.12 because Ubuntu `metavision-openeb` installs Python 3.12 bindings under `/usr/lib/python3/dist-packages`.
- The default `peakloc` task reads from root `config.json` for local smoke runs. Use `PEAKLOC_INPUT_FOLDER=/path/to/raw/files pixi run peakloc` for quick real-data overrides.
