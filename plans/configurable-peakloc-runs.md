# Configurable PeakLoc Runs

## Motivation

`PeakLoc.py` currently keeps scientific and runtime parameters as module-level
constants. That makes a run hard to reproduce unless the exact script state is
also tracked. The pipeline should load a validated configuration from JSON,
preserve the existing environment-variable workflow, and write a readable run
report next to the generated arrays and figures.

## Plan

1. Add a dataclass-backed configuration module.
   - Use a stdlib dataclass rather than Pydantic because the settings are flat
     and JSON is sufficient for the current workflow.
   - Validate scientific parameters such as FWHM, prominence, ROI radii, peak
     neighborhood size, slice duration, and core count.
   - Support `--config <path>` with JSON input and keep existing environment
     overrides for `PEAKLOC_INPUT_FOLDER`, `PEAKLOC_SLICE_START`, and
     `PEAKLOC_SLICE_DURATION`.

2. Refactor `PeakLoc.py` to consume the configuration object.
   - Replace module constants with `PeakLocConfig` fields.
   - Pass peak-finding, ROI, fitting, plotting, and core-count settings through
     function arguments.
   - Keep raw event slicing behavior unchanged.

3. Save a human-readable run report with artifacts.
   - Write Markdown reports under each recording output folder in `reports/`.
   - Include the settings used, input file details, per-slice counts and
     timings, aggregate localization/ROI counts, and saved artifact paths.
   - Use the same run timestamp in report and figure filenames.

4. Add an example JSON configuration.
   - Keep the example portable and pointed at the sample `data` folder.

5. Validate and commit atomically.
   - Run ruff check/fix, ruff format, ty check, and the PeakLoc smoke run.
   - Run pytest if relevant.
   - Commit the plan, config/report feature, example config/tests, and any
     validation-only fixes separately.
