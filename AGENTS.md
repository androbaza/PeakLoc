# Repository Guidelines

## Project Structure & Module Organization
PeakLoc is a Python 3.10 event-camera SMLM analysis pipeline. `PeakLoc.py` is the main batch entry point and writes per-recording outputs into a sibling folder plus `temp_files/`. `localization_scripts/` contains reusable processing code for event loading, peak finding, ROI generation, localization fitting, plotting, and simulation. `interface/` holds exploratory UI scripts. `figures/` stores README images. `peaks_dict_to_locs.py` reruns ROI generation/localization from an existing peaks pickle, and `clean_temp_files.py` removes generated localization/ROI arrays for a configured data directory.

Keep raw recordings, `.bias` files, generated `.npy`/`.pkl` outputs, and machine-specific data folders out of commits.

## Build, Test, and Development Commands
```bash
conda env create -f environment.yml -n peakloc
conda activate peakloc
python PeakLoc.py
python peaks_dict_to_locs.py
python clean_temp_files.py
python -m compileall PeakLoc.py peaks_dict_to_locs.py clean_temp_files.py localization_scripts interface
```
Use the conda commands to create and activate the pinned environment. `PeakLoc.py` runs the full pipeline; update the script-level `folder` or `INPUT_FILE` values before launching. `peaks_dict_to_locs.py` expects `INPUT_FILE` and `INPUT_FILE_PEAKS` to point to local data. Review `input_dir` before running `clean_temp_files.py`, because it deletes generated arrays. `compileall` is the current lightweight syntax check.

## Coding Style & Naming Conventions
Use 4-space indentation and PEP 8-style Python. Follow existing naming: `snake_case` for functions and variables, uppercase constants for tunable pipeline parameters such as `PROMINENCE`, `ROI_RADIUS`, and `PEAK_TIME_THRESHOLD`. Keep reusable logic in `localization_scripts/`; root-level scripts should mainly orchestrate workflows and define run-specific configuration. Preserve structured event array fields `x`, `y`, `p`, and `t` unless updating all consumers.

## Testing Guidelines
No automated test suite is currently committed. Before submitting changes, run `python -m compileall ...` and validate affected code on a small or sliced event sample. For new tests, add a `tests/` directory with `pytest` files named `test_<module>.py`; prefer tiny synthetic arrays or helpers from `localization_scripts/event_sim.py` so tests do not depend on lab-specific absolute paths.

## Commit & Pull Request Guidelines
The history uses short imperative summaries such as `add segmentation`, `fix bugs`, and `fix typo`. Keep commits concise, but make them specific enough to identify the changed behavior, for example `fix ROI bounds handling`.

Pull requests should include a short description, changed parameters or data assumptions, validation commands/results, and before/after figures or screenshots for visualization changes. Link related issues when available and call out any required local data paths.
