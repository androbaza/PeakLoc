# feat: add guided PeakLoc desktop application

## Motivation

PeakLoc previously required terminal use and direct JSON editing. This made calibration,
configuration, preflight, and processing inaccessible to users accustomed to GUI applications.

## Changes

- Add a cross-platform Tk/ttk desktop application with a five-step Data, Calibration, Basic,
  Advanced, and Run workflow.
- Bind all 91 PeakLoc configuration fields: five workflow fields and 86 documented basic or
  advanced controls with units and plain-language explanations.
- Run preflight, processing, calibration, and frozen slice workers in child processes so the UI
  remains responsive, streams logs, and can cancel the process tree.
- Support processing exactly one selected recording through the new optional input_file setting,
  while preserving folder and recursive batch discovery.
- Expose event-calibration building as a callable operation shared by the CLI and GUI.
- Check Metavision/OpenEB availability during RAW preflight while allowing the packaged GUI and
  NumPy event workflows to start without the SDK.
- Add a portable starter configuration, desktop user guide, Windows packaging guide, Pixi GUI
  tasks, and a fail-fast PyInstaller build script.
- Make frozen slice workers re-enter the executable instead of referencing extracted source.

## Assumptions and parameters

- The UI uses standard-library Tk/ttk for low dependency overhead and the same source on Windows
  and Linux.
- Windows RAW decoding targets the CPython 3.9 Metavision installation under
  C:\Program Files\Prophesee.
- The shipped starter config uses a 10-second, one-slice exploratory run and local-only
  background until the user creates or selects calibration.
- PyInstaller uses a one-folder layout. The final tested release is approximately 677 MB because
  it contains the complete scientific runtime; the UI itself adds no third-party dependency.

## Validation

- 21 focused existing config, preflight, calibration, and bead-sigma tests pass.
- Config catalog import proves 86 editable setting specs plus five workflow fields cover all 91
  dataclass fields.
- Temporary single-file preflight selects only the requested recording.
- Source GUI hidden startup and full widget/config binding pass.
- Source GUI worker subprocess preflight passes and streams issues.
- pixi run -e dev build-gui completes successfully.
- Frozen preflight worker and empty-recording processing worker both return exit code 0.
- Frozen windowed GUI remains alive through the startup smoke interval.
- Focused Ruff format and check pass for all changed feature files.
- Repository-required Ruff autofix/format commands were run. The wider repository retains 47
  pre-existing non-autofix lint findings.
- The required ty command is unavailable on Windows because ty is declared only for Linux.
- Three unrelated existing pipeline-runner tests still fail in stale-artifact and QC-report paths;
  no tests were added or modified for this feature.

## Delivery

Deliver the complete dist\PeakLoc folder, including PeakLoc.exe, config.json, PeakLoc User
Guide.md, and _internal. Validate calibration and a short real RAW run on a clean target PC before
general distribution.
