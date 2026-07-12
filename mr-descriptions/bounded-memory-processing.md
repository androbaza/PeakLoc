# Bounded-Memory Long-Recording Processing

## Motivation

Long RAW recordings could exhaust RAM and swap because PeakLoc held the full event
array, sent large inputs to many persistent Loky workers, and repeatedly concatenated
all per-slice ROI/localization arrays in memory at the end of the run.

## Changes

- Decode RAW input into a temporary disk-backed event cache and memory-map `.npy`
  input instead of loading an entire recording into RAM.
- Read chronological slices with `searchsorted`; retain only the active slice.
- Build event-density QC incrementally in bounded chunks, with only a capped sample
  retained for optional interactive QC.
- Assemble final ROI, attempted-localization, accepted-localization, and QC arrays
  directly into memory-mapped `.npy` files rather than using repeated
  `np.concatenate` calls.
- Bound expensive parallel stages through `max_parallel_workers` (default `4`), use
  disk-backed Joblib transport, and retire Loky workers after each stage.
- Run ROI generation in shared-memory threads, remove unnecessary deep copies, and
  avoid retaining padded convolution buffers through row views.
- Explicitly release per-slice Python/native allocations and trim free glibc arenas
  between slices on Linux.
- Set the repository long-run configuration to a 1,000,000-event RAW reader buffer
  and a four-worker memory cap.

## Assumptions and trade-offs

- Event timestamps are normally chronological. Non-monotonic `.npy` input falls back
  to the previous boolean-mask selection behavior.
- RAW staging uses temporary disk space in the recording's `temp_files/` directory;
  it is removed after a successful run when `cleanup_temp_outputs` is enabled.
- Four workers is deliberately conservative. Raise `max_parallel_workers` only after
  a complete recording has shown stable memory use.

## Validation

- `PYTHONNOUSERSITE=1 pixi run -e dev pytest localization_scripts/tests/test_pipeline_config.py localization_scripts/tests/test_pipeline_runner.py localization_scripts/tests/test_event_array_processing.py localization_scripts/tests/test_roi_generation.py localization_scripts/tests/test_localization_fitting.py localization_scripts/tests/test_qc_dashboard.py localization_scripts/tests/test_peak_finding.py` — 44 passed.
- `PYTHONNOUSERSITE=1 pixi run -e dev pytest localization_scripts/tests/test_synthetic_blinks_pipeline.py::test_synthetic_blinks_are_detected_and_localized_end_to_end` — passed.
- A mocked RAW reader probe materialized three normalized events into a disk-backed
  cache and reopened it as a memory map.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff check --fix .` — passed.
- `PYTHONNOUSERSITE=1 pixi run -e dev ruff format .` — passed.
- `PYTHONNOUSERSITE=1 pixi run -e all ty check localization_scripts` — passed.
- Repository-wide `ty check` still reports eight pre-existing diagnostics in the
  modified `notebooks/raw_event_npz_explorer.ipynb`; the production package has no
  diagnostics.
