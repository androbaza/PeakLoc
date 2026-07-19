# PeakLoc

PeakLoc is an event-camera localization pipeline for experimental Single-Molecule Localization Microscopy-like analysis.

It reads event-camera `.raw` recordings, detects blinking-like event peaks, extracts local regions of interest, fits event-count models, filters uncertain localizations, and can render localization tables into SMLM-style images.

This repository is currently best treated as a research pipeline, not as a polished black-box end-user application. For publication-grade use, calibrated acquisition data, careful parameter validation, and independent quality control are required.

## What PeakLoc does

PeakLoc converts event-camera recordings into localization outputs.

A typical workflow is:

1. Read `.raw` event-camera files.
2. Convert events into per-pixel time traces.
3. Detect candidate peaks in cumulative event signals.
4. Merge nearby peak candidates in space and time.
5. Extract positive and negative polarity event-count ROIs.
6. Fit a pixel-integrated Gaussian event model.
7. Estimate localization uncertainty.
8. Filter bad or uncertain fits.
9. Save localization tables, ROI arrays, QC tables, reports, figures, and optional rendered SMLM images.

## Who this documentation is for

The documentation assumes the reader has no prior understanding of event cameras, SMLM, or this repository.

Start here:

- [Documentation overview](docs/index.md)
- [Data preparation](docs/data-preparation.md)
- [Configuration guide](docs/configuration.md)
- [Run modes](docs/run-modes.md)
- [Output interpretation](docs/output-interpretation.md)
- [Use cases and limitations](docs/use-cases-and-limitations.md)

## Installation

PeakLoc uses [Pixi](https://pixi.sh/) for the Python environment.

Clone the repository:

```bash
git clone https://github.com/androbaza/PeakLoc.git
cd PeakLoc
```

Install the environment:

```bash
pixi install
```

Run a basic import test:

```bash
pixi run import-test
```

## Required external dependency: OpenEB / Metavision bindings

PeakLoc reads `.raw` files using Prophesee/OpenEB Python bindings.

On Ubuntu, the bindings are expected under:

```text
/usr/lib/python3/dist-packages
```

Installation steps: [docs.prophesee.ai](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
The repository currently bridges this system path into the Pixi Python 3.12 environment.



## Quick start

Edit `config.json` so that `input_folder` points to the folder containing your `.raw` files.

Then run:

```bash
pixi run peakloc
```

The pipeline creates a timestamped output folder for every run of each input recording.

Example:

```text
data/
├── AF647_coverslip.raw
└── AF647_coverslip/
    └── 20260712_143015_123456/
        ├── share/                         # send this directory to collaborators
        │   ├── README.md                   # run summary and hand-off guide
        │   ├── figures/                    # final labelled PNG and PDF figures
        │   ├── statistics/                 # compact CSV and JSON summaries
        │   └── metadata/                   # effective settings and software provenance
        └── debug/                          # technical audit trail; do not hand off by default
            ├── arrays/                     # localization, ROI, and attempted-fit arrays
            ├── qc/                         # fit montages and diagnostic figures
            ├── reports/                    # slice, mask, and run diagnostics
            ├── provenance/                 # detailed audit tables and preflight material
            └── temp_files/                 # transient per-slice worker output
```

Use `share/` as the collaborator-ready bundle. Its figures include the cropped SMLM
reconstruction, detection and fit summary, temporal dynamics, spatial timing maps with
units-bearing colorbars, and FRC when available. Timing-distribution panels display the
physically relevant 0–1000 ms range and report longer values in the accompanying
statistics rather than expanding the axis until the data disappear.

## Minimal smoke run

A smoke run should use a short time slice and low memory load.

In `config.json`:

```json
{
  "input_folder": "data",
  "slice_start": 0,
  "slice_duration": 10000000,
  "max_raw_events": 1000000,
  "num_cores": 4,
  "plot_result": true
}
```

Then run:

```bash
pixi run python PeakLoc.py --preflight
```

## Preflight checks

Run a preflight check and exit without processing:

```bash
pixi run python PeakLoc.py --preflight-only
```

## Parameter sweep

Create a sweep specification, for example:

```json
{
  "prominence": [8.0, 12.0, 16.0],
  "max_localization_uncertainty_nm": [30.0, 50.0, 80.0]
}
```

Save it as:

```text
sweep/prominence_uncertainty_sweep.json
```

Run:

```bash
pixi run python PeakLoc.py --config config.json --sweep sweep/prominence_uncertainty_sweep.json --preflight
```

Sweep outputs are written to:

```text
sweep/
├── sweep_results.csv
├── sweep_results.json
├── pareto_localizations_vs_uncertainty.png
├── rejection_reason_heatmap.png
└── parameter_effects.html
```

## Rendering an existing localization file

To render an existing localization `.npy` file:

```bash
pixi run plot-result /path/to/localizations.npy
```

If no path is provided, the script prompts for one.

## Hardware and RAM expectations

PeakLoc can be memory-intensive because event streams are expanded into per-pixel or per-ROI intermediate structures.

A practical starting point is:

- Small smoke tests: laptop or workstation, short slices only.
- Full-chip 1280 × 720 recordings: high-memory workstation.
- Long recordings: 128 GB RAM is recommended.
- Example reference workload: full-chip 1280 × 720, 600 seconds, about 10 minutes on a 24-core machine, assuming suitable data and configuration.

Use short `slice_duration` values first. Do not start with a full 600-second recording when validating a new dataset.

## Resource-aware parallel runs

PeakLoc can overlap the serial portions of adjacent slices without allowing two
memory-intensive worker bursts to oversubscribe the machine. For the 16-core reference
workstation, the measured starting point is:

```json
{
  "max_concurrent_slices": 2,
  "cpu_worker_budget": 16,
  "max_workers_per_slice": 15,
  "memory_reserve_gib": 16.0,
  "disk_reserve_gib": 10.0
}
```

One slice receives a 15-worker lease during Numba, peak, ROI, and fit stages while the
other slice can use the remaining CPU for serial work. The bounded scheduler checks RAM
and disk headroom before admitting work. Run reports include per-stage timing, peak RSS,
temporary disk usage, and the resolved resource settings. See
[`docs/configuration.md`](docs/configuration.md) for every setting.

The GPU is not currently used. The dominant stages use irregular event indexing and
small SciPy operations, while localization is too small a fraction of wall time to
justify GPU transfer and a separate fitting implementation.

## Current important limitations

PeakLoc is under active development. Important limitations are:

- `fit_sigma=true` is not currently the recommended or implemented production path. The current model uses a fixed `sigma_psf_px`.
- Simultaneous overlapping emitters are not fully resolved as independent emitters.
- Uncalibrated mode can be useful for exploration, but is not publication-grade.
- The default calibration-free settings are not a substitute for dark and blank calibration recordings.
- Parameter values are dataset-dependent. Defaults are starting points, not universal microscope settings.

## License

This repository is licensed under GPL-3.0. See [LICENSE](LICENSE).
