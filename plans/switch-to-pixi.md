# Switch PeakLoc Fully to Pixi

## Goal
Make `PeakLoc.py` run through pixi as the supported project workflow, preferably with:

```bash
pixi run peakloc
```

`pixi run python PeakLoc.py` should also continue to work. The current `environment.yml` appears to be a full exported conda environment, so the migration should replace it with a curated pixi manifest and lockfile.

## Current State
- `pixi.toml` is a minimal Metavision RAW reader smoke-test environment.
- Ubuntu 24 provides `metavision-openeb` Python bindings under `/usr/lib/python3/dist-packages`, so pixi currently needs Python 3.12 and `PYTHONPATH=/usr/lib/python3/dist-packages`.
- `environment.yml` pins many transitive packages and unrelated tools such as conda, Jupyter, Qt console packages, and notebook server dependencies.
- Core pipeline imports indicate these likely runtime dependencies:
  - `numpy`
  - `scipy`
  - `numba`
  - `joblib`
  - `scikit-image`
  - `awkward`
  - `csaps`
  - `interpolation`
  - `matplotlib`
  - `matplotlib-scalebar`
  - `tifffile`
  - `natsort`
  - `cryptography`
  - `h5py` for OpenEB import side effects
- Optional UI scripts additionally use `customtkinter` and `tkinter-tooltip`.

## Migration Plan

### 1. Define the pixi dependency model
Replace the smoke-test-only `pixi.toml` with a package-oriented manifest:

- default runtime environment for `PeakLoc.py`
- optional `dev` feature for `pytest`, `ruff`, and `ty`
- keep `python = "3.12.*"` while using Ubuntu OpenEB bindings

Do not import the whole `environment.yml`; add only direct dependencies with `pixi add`.

### 2. Add pixi tasks
Add tasks that encode the supported workflows:

```toml
[tasks]
peakloc = "python PeakLoc.py"
import-test = "python import_test.py"
test = "pytest"
lint = "ruff check ."
format = "ruff format ."
typecheck = "ty check"
```

If `PeakLoc.py` remains path-configured internally, document that `folder` or input settings must be updated before running.

### 3. Validate dependency completeness incrementally
Use import-level checks before running expensive localization:

```bash
pixi run python -c "import PeakLoc"
pixi run python -m py_compile PeakLoc.py peaks_dict_to_locs.py localization_scripts/*.py
pixi run import-test
```

For each missing package, add the package explicitly to pixi and rerun the checks. Keep a note of packages needed only by `interface/` so they can stay optional.

### 4. Make RAW loading safe enough for smoke tests
The existing full loader uses `RawReader(..., max_events=4e9)`, which can allocate about 60 GiB. Keep `import_test.py` bounded and add a small reader smoke task so OpenEB can be verified without running the full pipeline.

For `PeakLoc.py`, decide whether to keep workstation-scale full-file loading or add a configurable smoke/sample mode. Do not change scientific behavior in the pixi migration commit unless required.

### 5. Replace conda documentation
Update README and AGENTS references from:

```bash
conda env create -f environment.yml
```

to:

```bash
pixi install
pixi run peakloc
```

Document the Ubuntu OpenEB prerequisite and the current Python 3.12 constraint.

### 6. Retire `environment.yml`
After `pixi run peakloc` reaches the same import/runtime point as the conda environment, remove `environment.yml` or keep it only temporarily with a clear deprecation note. Prefer deletion once `pixi.lock` is committed and validated.

## Validation Checklist
- `pixi install`
- `pixi run import-test`
- `pixi run python -c "import PeakLoc"`
- `pixi run python -m py_compile PeakLoc.py peaks_dict_to_locs.py clean_temp_files.py localization_scripts/*.py`
- `pixi run peakloc` against a known small RAW file or a short sliced dataset
- `pixi run ruff check .`
- `pixi run ruff format --check .`
- `pixi run ty check`, allowing documented legacy failures if needed

## Focused, Atomic Commits
1. `chore: define pixi runtime dependencies`
   - Curate `pixi.toml`, regenerate `pixi.lock`, keep OpenEB bridge explicit.

2. `chore: add pixi tasks`
   - Add `peakloc`, `import-test`, `lint`, `format`, `typecheck`, and test tasks.

3. `fix: make raw reader smoke test bounded`
   - Keep `import_test.py` memory-safe and verify OpenEB RAW reading.

4. `docs: document pixi workflow`
   - Update README/AGENTS with `pixi install` and `pixi run peakloc`.

5. `chore: retire conda environment export`
   - Remove or deprecate `environment.yml` after validation.

6. `test: add lightweight import smoke tests`
   - Add pytest tests next to modules for dependency/import sanity if desired.

## Risks and Decisions
- Fully pixi-managed OpenEB may not be available from conda-forge. If not, keep Ubuntu `metavision-openeb` as a documented system prerequisite and bridge it through pixi activation. Installation: https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html#chapter-installation-linux-openeb-with-packages
- Some legacy code may fail `ruff` or `ty` initially. Treat style/type cleanup as separate tidy commits, not part of the environment migration unless it blocks execution.
