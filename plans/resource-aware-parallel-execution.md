# Resource-aware parallel execution

## Goal

Increase recording throughput by overlapping independent time slices while preserving
PeakLoc's bounded-memory behavior, deterministic scientific outputs, and worker
isolation. Optimize for sustained throughput rather than for a brief 100% CPU reading.

The first production target is the current workstation and 50,000,000 us slices:

- AMD Ryzen 9 7950X: 16 physical cores / 32 hardware threads
- 96 GiB RAM
- NVIDIA GeForce RTX 4070 Ti SUPER: 16 GiB VRAM
- NVMe storage, with only about 45 GiB free at the time of this plan

## Current execution model and bottlenecks

`run_batch()` processes recordings serially, and `process_recording()` processes every
time slice serially. A slice alternates between very different execution modes:

1. `array_to_polarity_map()` and `array_to_time_map*()` scan and index the slice on one
   core. Their `prange` calls do not create parallel execution because the functions are
   not compiled with `parallel=True`.
2. `process_conv_list_parallel()` is Numba-parallel and currently sees all 32 Numba
   threads, independently of `max_parallel_workers`.
3. Peak interpolation uses a temporary Loky pool with up to 30 processes.
4. ROI generation uses up to 30 threads.
5. Poisson localization creates another temporary Loky pool with up to 30 processes.
6. Peak filtering, object conversion, output writes, and cleanup are serial.

This explains the observed short 30-worker bursts and long low-utilization intervals.
It also means that starting two unchanged slices would be unsafe: two slices could each
start 30 Loky workers or 32 Numba threads and oversubscribe the machine.

The large outputs make storage part of the stability problem. Recent results contain
individual 1-2 GiB arrays, while per-slice files, aggregate files, a RAW cache, and
joblib transport files can coexist until cleanup.

## Target architecture

```text
recording mmap / RAW cache + calibration + spatial mask
                         |
              precomputed SliceTask bounds
                         |
       CPU + RAM + disk admission controller
                  /                 \
       slice process A          slice process B
       CPU quota: 8             CPU quota: 8
       private temp dir         private temp dir
                  \                 /
          ordered SliceResult collection
                         |
           aggregate arrays -> QC -> plots
```

The parent remains the owner of recording-level setup, final concatenation, QC, and
reporting. Slice workers receive only a small task descriptor: input/cache path, array
dtype and shape, start/stop indices, slice end time, configuration, and output path.
Each worker reopens the recording read-only with `mmap_mode="r"`; it must not receive a
large event array through process pickling.

Completed futures may arrive out of order, but results and aggregate files are always
sorted by slice end time before the existing ID-offset concatenation. Scientific row
order therefore remains the same as in serial execution.

## Configuration and resource contract

Add explicit settings without changing the current one-slice default:

- `max_concurrent_slices`: outer slice-process cap; default `1` for compatibility.
- `cpu_worker_budget`: maximum total CPU workers/threads used by slice work. Resolve
  `auto` from CPU affinity, but keep the value visible in effective settings.
- `max_workers_per_slice`: cap for a parallel stage inside one slice. Migrate the
  current `max_parallel_workers` meaning with a deprecation period.
- `memory_reserve_gib`: memory that the scheduler must leave available to the OS,
  parent, aggregation, and QC.
- `disk_reserve_gib`: free-space floor below which no new slice is admitted.

At any instant, enforce

```text
active slices * effective workers per slice <= cpu_worker_budget
```

Set Numba's thread count explicitly in every slice process. Set BLAS/OpenMP child
thread counts to one for Loky workers so that an eight-process stage cannot silently
become eight multi-threaded BLAS teams. Use a joblib inner-thread limit as a second
guard. Record all resolved values in the run report.

For this workstation, start benchmarking at `cpu_worker_budget=16`,
`max_concurrent_slices=2`, and `max_workers_per_slice=8`. Sixteen physical cores are a
safer baseline than 30 SMT threads; 24 and 30 workers remain benchmark candidates, not
defaults.

The scheduler must use a bounded submission window rather than queue every slice. It
admits a new slice only when a slot is free and RAM/disk remain above their reserves.
Use measured peak private memory from representative slices, scaled by event count
with a safety factor, for the initial per-slice memory estimate. If no trusted estimate
exists, use the configured two-slice cap and a conservative reserve instead of guessing
that all slices fit. Fall back to one slice when a slice is unusually large or when
timestamps are non-monotonic and require a copied boolean slice.

## Implementation sequence

### 1. Add measurements before changing scheduling

Add a named stage-timing record to `SliceResult` and time at least event indexing,
convolution, peak interpolation, local-maximum filtering, ROI generation, localization,
artifact writing, and memory release. Record slice event bytes, peak private RSS/high
water, total run wall time, worker-pool startup time, and temporary disk high water.

Persist the metrics as JSON/CSV beside the run report and summarize them in the report.
The current cumulative log timestamps are not sufficient to distinguish stage work
from pool creation or cleanup. Warm Numba kernels before collecting benchmark numbers
and report warm-up separately.

Use an external sampler during the benchmark to capture whole-process-tree CPU,
memory, swap, disk throughput, and context switches. This avoids adding a runtime
monitoring dependency solely for development profiling.

### 2. Make a slice an isolated, deterministic process task

Create named `SliceTask` and `SliceExecutionConfig` types rather than passing a large
tuple. Precompute monotonic slice indices with `searchsorted`. For `.raw`, finish the
existing on-disk event cache before starting workers; for `.npy`, let workers reopen the
source directly. Retain the current serial fallback for non-monotonic inputs initially.

Give each slice a directory such as `temp_files/slices/<time_slice>/`, including its own
joblib directory. Write every output to a `.partial` path and atomically rename only
after all arrays and the slice manifest are complete. This prevents joblib collisions
and stops a failed slice from looking complete. The parent validates the manifest
before adding artifacts to `RecordingResult`.

Use a spawn-safe, bounded process executor. A worker failure cancels pending work,
preserves completed slice artifacts for diagnosis, and prevents final aggregation.
Always close worker memory maps in `finally` blocks. Recycle slice processes after a
small number of tasks if high-water measurements show that Numba/native allocations do
not return to a stable baseline.

### 3. Introduce two-level CPU budgeting

Route every parallel stage through the resolved per-slice worker count rather than
reading `config.parallel_workers` independently. Apply the same quota to Numba,
peak-finding Loky, ROI threads, and localization Loky. Do not allow nested libraries to
create additional teams.

Keep the first version conservative: at most two slice processes and no more than the
global CPU budget across both inner stages. Reuse a slice-local Loky executor between
peak detection and localization only if measurement shows meaningful startup cost and
tests confirm that the OpenEB path isolation remains intact. Otherwise retain
`reuse=False`; worker stability is more important than optimizing the short bursts.

### 4. Add RAM, disk, and back-pressure safeguards

Before submitting a slice, check its estimated working set against available RAM plus
`memory_reserve_gib`. Track the actual high water returned by completed workers and
increase the estimate immediately when a slice exceeds it. Never respond to pressure
by starting more workers, and treat swap activity as a failed benchmark configuration.

Preflight must estimate coexistence of the RAW cache, per-slice outputs, final aggregate
arrays, joblib transport, and `disk_reserve_gib`. Serialize the short artifact-commit
section if concurrent writes reduce throughput or create large I/O latency. Fail early
with a required/free-space message instead of discovering a full disk during final
concatenation.

### 5. Parallelize or replace the remaining serial event index only after profiling

First benchmark overlapping the two independent Numba `nogil` event-map builders with
two threads. Keep it only if wall time improves without unacceptable memory-bandwidth
contention or peak-memory growth.

If event indexing remains the dominant stage, replace the two duplicated typed
dictionaries with one compact, read-only per-pixel event index: contiguous timestamp
and polarity arrays plus pixel offsets. Build it with sharded/private state and a
deterministic merge, then make convolution and ROI generation consume that shared
representation. This is a separate structural performance change with equivalence
tests. Do not simply add `parallel=True` to the current shared-dictionary mutations;
that would introduce races.

After the representation is compact and shareable, consider a single long-lived global
worker pool for peak/ROI/fit chunks. That is a later simplification, not a prerequisite
for two-slice execution.

### 6. Tune on representative 50-second slices

Benchmark the same adjacent, nontrivial slices with unchanged scientific settings.
Exclude the tiny trailing slice and keep the input in the same cache state. Compare at
least:

| Slice lanes | Workers/lane | CPU budget | Purpose |
| ---: | ---: | ---: | --- |
| 1 | 30 | 30 | Current baseline |
| 1 | 16 | 16 | Physical-core baseline |
| 2 | 8 | 16 | Recommended starting point |
| 2 | 12 | 24 | Test moderate SMT use |
| 3 | 5 | 15 | More coverage of serial stages |
| 4 | 4 | 16 | Throughput candidate if RAM allows |

Run the winning candidates more than once and then validate the winner on the complete
recording. Choose the configuration with the best stable events/second and
localizations/second, not the highest instantaneous CPU use.

## Validation and acceptance criteria

Add focused tests next to the affected modules:

- Configuration validation and CPU-budget resolution in
  `localization_scripts/tests/test_pipeline_config.py`.
- Bounded in-flight task count, out-of-order future completion, deterministic result
  sorting, worker failure, partial-file handling, and the non-monotonic fallback in
  `localization_scripts/tests/test_pipeline_runner.py`.
- A tiny three-slice synthetic recording processed with one and two slice lanes. Compare
  peak counts, ROI records, attempted/accepted localizations, QC records, and artifact
  order. Require exact equality where current serial execution is exact; define and
  justify tight tolerances for floating-point fields only if backend ordering changes.
- A worker-isolation test confirming that OpenEB system paths still do not leak into
  Loky workers.

The production configuration is accepted only when:

- total worker/thread concurrency never exceeds the resolved CPU budget;
- peak memory stays below the configured ceiling with no swap growth or OOM recovery;
- disk free space never crosses the reserve and no partial artifact is aggregated;
- serial and parallel scientific outputs are equivalent and repeatable;
- the full recording is materially faster than the current baseline; and
- Ctrl-C or a slice exception shuts down descendants and leaves a diagnosable run.

Run only the focused tests requested by repository policy, followed by Ruff formatting
and checks and the full `ty check` command.

## GPU decision

Do not put the GPU on the critical path for the first implementation.

The current expensive data preparation is dominated by irregular typed dictionaries,
variable-length traces, per-coordinate splines, and small SciPy operations. These are
not drop-in CUDA workloads. The localization stage uses independent SciPy L-BFGS-B
fits on 13 x 13 ROIs and currently accounts for only about 5-10% of total processing.
Even an infinitely fast localization backend would therefore improve end-to-end time
by only about 5-11% before transfer and launch overhead (Amdahl's law).

Retain a later GPU experiment behind a backend flag only if post-CPU profiling shows
that batched localization becomes at least roughly 25% of wall time. Such an experiment
must implement the complete PSF, Poisson objective, optimizer, and covariance path as a
batched GPU operation; copying individual ROIs to the GPU is not useful. Compare fit
status, positions, uncertainties, NLL, and acceptance decisions against the CPU backend
on synthetic and real slices. Adopt it only if the complete recording is faster and
scientific equivalence is demonstrated.

The RTX 4070 Ti SUPER is capable enough for that prototype, but CPU slice overlap is the
higher-value and much lower-risk optimization now.

## Commit sequence for implementation

Keep structural and behavioral work separate:

1. `perf: record per-stage pipeline resource metrics`
2. `tidy: introduce slice task and execution result types`
3. `feat: add bounded parallel slice execution`
4. `fix: enforce global thread and memory budgets`
5. `perf: compact and parallelize slice event indexing` only if profiling justifies it
6. `docs: report parallel tuning and selected defaults`

After executing the plan, add the required merge-request description under
`mr-descriptions/` with benchmark results, resolved resource settings, validation, and
any output-equivalence tolerances.
