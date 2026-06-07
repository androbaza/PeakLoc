# Repository Guidelines

## Project Structure & Module Organization
PeakLoc is a Python event-camera SMLM pipeline. `PeakLoc.py` is the main batch workflow. `localization_scripts/` contains event parsing, peak detection, ROI generation, fitting, plotting, and simulation code. `interface/` contains exploratory UI scripts. `figures/` stores README assets. Helper scripts such as `peaks_dict_to_locs.py` and `clean_temp_files.py` live in `scripts/` and should stay thin.

Keep raw recordings, `.bias` files, generated `.npy`/`.pkl` outputs, temp folders, and machine-specific paths out of commits.

Coordinate convention:
- Image arrays are indexed as image[y, x] == image[row, col].
- Matplotlib imshow overlays must use:
    scatter(sub_x, sub_y)

## Environment & Commands
Use pixi. Add dependencies with `pixi add <package>`.

```bash
pixi install
pixi run peakloc
pixi run peaks-dict-to-locs
pixi run -e dev pytest
pixi run -e dev ruff check --fix .
pixi run -e dev ruff format .
pixi run -e all ty check
```

`PeakLoc.py` runs the full pipeline using root `config.json` by default; set `PEAKLOC_INPUT_FOLDER`, `PEAKLOC_SLICE_START`, or `PEAKLOC_SLICE_DURATION` when needed. Review `input_dir` before using `scripts/clean_temp_files.py`; it deletes generated arrays.

## Coding Policies & Style
Target Python 3.12. Type every function signature and use modern generics such as `list[int]`. Prefer pure functions; isolate plotting, file writes, and other side effects. If a function returns multiple values, create a dataclass, Pydantic `BaseModel`, or other named type instead of returning large tuples.

Use 4-space indentation and keep `.py` lines at 100 characters or less. Use `snake_case` for functions and variables, `UPPER_CASE` for constants, and clear names: prefer `coefficients` over `coeffs`. Sort imports with ruff/isort. Comments should explain why, not what. Preserve event fields `x`, `y`, `p`, and `t` unless all consumers are updated together.

## Testing & Validation
Use pytest. Place tests next to the module they cover, for example `localization_scripts/tests/test_peak_finding.py`; do not add a top-level `tests/` directory. New features need tests using tiny synthetic arrays or `localization_scripts/event_sim.py`. Tests must not require absolute lab paths or large raw recordings.

Before finishing, run `pixi run -e dev ruff check --fix .`, `pixi run -e dev ruff format .`, and `pixi run -e all ty check`. Run `pixi run -e dev pytest` when tests are relevant; note any known broken tests in the PR.

## Workflow, Commits & PRs
For substantial planning, write the plan as Markdown in `plans/` and include focused, atomic commits. After executing a plan, write a merge-request description in `mr-descriptions/`.

Use conventional commits such as `fix: repair ROI bounds` or `feat: add sliced localization`. Apply tidy-first: separate structural cleanup from behavior changes, prefix tidy commits with `tidy:`, and do not mix tidy and behavior edits. PRs should include motivation, changed parameters or assumptions, validation results, and before/after figures for visual output changes. Always commit the changes, unless asked not to.
