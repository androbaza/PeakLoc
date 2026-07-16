# Resource-aware parallel slice execution

## Motivation

PeakLoc previously processed time slices serially while briefly launching up to 30
workers for peak interpolation and localization. Long serial gaps left most CPU cores
idle, and overlapping unchanged slices would have oversubscribed CPU and memory.

## Changes

- Record per-stage wall time, event bytes, process peak RSS, temporary disk use, and PID
  as JSON/CSV and summarize them in the run report.
- Run a bounded number of isolated slice subprocesses. Workers reopen the NPY or RAW
  cache read-only, receive only small task descriptors, and return results in time order.
- Enforce a global CPU budget with a leased parallel stage. On the selected workstation,
  one slice can use 15 workers while the other uses the remaining CPU for serial work.
- Pin Numba explicitly and limit BLAS/OpenMP threads inside Loky workers.
- Add RAM and disk admission checks, conservative disk preflight, per-slice joblib
  directories, atomic artifacts, manifests, failure cancellation, and descendant cleanup.
- Build the Numba ROI event index once per slice and share it read-only across ROI
  threads instead of rebuilding the full index in every worker.
- Add `slice_count` for bounded benchmark runs and document all resource settings.
- Select two slice lanes, a 16-CPU budget, a 15-worker stage lease, 16 GiB RAM reserve,
  and 10 GiB disk reserve in the current workstation configuration.

## Benchmark

Input was a 6-second NPY extract of the currently selected RAW recording. The benchmark
processed the same two adjacent 0.5-second high-activity slices containing 2,338,762 and
1,927,273 events. Scientific settings, calibration, and spatial mask were unchanged.

| Configuration | Wall time | Peak RSS | Swap |
| --- | ---: | ---: | ---: |
| Original serial, 15 workers | 107.00 s | 5.83 GiB | 0 |
| Serial, 30 workers | 156.98 s | 10.46 GiB | 0 |
| Optimized serial, 15 workers | 61.17 s | 1.64 GiB | 0 |
| Two lanes, 15-worker lease | 56.55 s | 1.28 GiB/process | 0 |
| Two lanes repeat | 56.77 s | 1.28 GiB/process | 0 |

The selected configuration is 47.1% faster than the original measured baseline and
7.5% faster than the optimized serial run. The parallel repeats differ by 0.22 seconds.
Thirty workers were slower and used substantially more memory, so SMT saturation was
not selected.

The real selected-recording preflight resolved 2 lanes, 15 leased workers, and a 16-CPU
budget. It estimated 23.7 GiB required disk headroom with 43.7 GiB free.

## Scientific equivalence

The optimized serial run and both parallel runs produced byte-identical attempted
localizations, accepted localizations, localization QC, and ROI NPY arrays. SHA-256
hashes matched for every aggregate artifact, so no floating-point tolerance is needed.

## Validation

- 42 focused scheduler, configuration, preflight, ROI, peak, and localization tests pass.
- Ruff formatting and focused lint pass.
- `ty check` passes for every modified Python file.
- The required full `ty check` still reports 8 pre-existing diagnostics in
  `notebooks/raw_event_npz_explorer.ipynb`; this change does not touch that notebook.
- A deliberately interrupted benchmark terminated slice process groups and Loky
  descendants without leaving running workers.
- The selected two-lane benchmark completed twice with no swap activity.

Full five-minute processing was intentionally not run during tuning because the task
requested small slices and the workstation has limited free disk. The complete recording
remains the next operational soak run.

## GPU decision

The GPU is not placed on the critical path. Current work is dominated by irregular event
indexing, splines, and small SciPy operations, while fitting is only a small wall-time
fraction. A GPU backend would require a scientifically equivalent batched PSF, objective,
optimizer, and covariance implementation; individual ROI transfers would not pay off.
