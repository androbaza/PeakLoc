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

The pipeline creates one output folder per input recording.

Example:

```text
data/
├── AF647_coverslip.raw
└── AF647_coverslip/
    ├── localizations_*.npy
    ├── rois_*.npy
    ├── localization_qc_*.npy
    ├── figures/
    ├── reports/
    └── qc/
```

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

## Current important limitations

PeakLoc is under active development. Important limitations are:

- `fit_sigma=true` is not currently the recommended or implemented production path. The current model uses a fixed `sigma_psf_px`.
- Simultaneous overlapping emitters are not fully resolved as independent emitters.
- Uncalibrated mode can be useful for exploration, but is not publication-grade.
- The default calibration-free settings are not a substitute for dark and blank calibration recordings.
- Legacy drift helper functionality exists, but should not be presented as the recommended final drift-correction path.
- Parameter values are dataset-dependent. Defaults are starting points, not universal microscope settings.

## License

This repository is licensed under GPL-3.0. See [LICENSE](LICENSE).
