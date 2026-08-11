# PeakLoc desktop interface implementation plan

## Goal

Provide a lightweight cross-platform desktop application that takes a first-time user from
calibration recordings to a validated PeakLoc run without requiring a terminal or direct JSON
editing. Keep presentation and orchestration in `interface/`; continue to use the existing
scientific configuration, preflight, calibration, and pipeline functions as the source of truth.

## User workflow

1. Choose one recording or a folder of recordings.
2. Optionally create an event calibration from dark and laser-on blank `.raw` files.
3. Review the small set of basic parameters, with units and plain-language explanations.
4. Expand advanced settings only when required; every supported config field remains editable.
5. Check the setup with PeakLoc preflight.
6. Start processing, monitor logs, and cancel a running child process if needed.
7. Save or reopen the resulting JSON configuration for repeatable runs.

## Architecture

- `PeakLocGUI.py`: minimal application/worker entry point suitable for source and frozen runs.
- `interface/app.py`: Tk/ttk views, state binding, dialogs, accessibility, and task monitoring.
- `interface/config_catalog.py`: declarative labels, tiers, groups, units, choices, and help text.
- `interface/operations.py`: non-visual worker commands for preflight, processing, and calibration.
- Existing `localization_scripts/` and `calibration_scripts/`: scientific implementation.

Long operations run in a child process. This prevents Tk from freezing, makes cancellation
possible, captures both stdout and stderr in the application log, and uses the same worker path
from source and from a PyInstaller executable.

## Pipeline integration

- Add an optional `input_file` config field so the pipeline and preflight can process exactly one
  user-selected recording without copying multi-gigabyte data or relying on platform-specific
  links.
- Make frozen slice workers re-enter the executable instead of trying to execute an extracted
  source path.
- Expose a callable calibration build function; the CLI and GUI use the same implementation.

## Packaging

- Add a Windows PyInstaller one-folder, windowed build task named `build-gui`.
- Bundle a portable starter configuration and keep the existing Metavision runtime hook.
- Document building, release-folder testing, target-PC prerequisites, and delivery.

## Validation

- Import/compile the interface and exercise config conversion without starting the event pipeline.
- Run focused existing tests for configuration, preflight, calibration, recording discovery, and
  pipeline runner.
- Run repository formatting, linting, and type checks as required by `AGENTS.md`.
- Build the Windows distribution and launch its worker help/smoke path where the environment
  permits.
