# PeakLoc documentation

PeakLoc is a research pipeline for converting event-camera `.raw` recordings into localization tables and SMLM-style rendered images.

This documentation is written for a user who has no prior knowledge of the method.

## What is an event-camera recording?

A conventional camera stores full image frames. An event camera stores asynchronous pixel-level changes.

Each event usually contains:

```text
x, y, polarity, timestamp
```

In PeakLoc:

- `x` is the horizontal sensor coordinate.
- `y` is the vertical sensor coordinate.
- `p` is polarity, usually positive or negative.
- `t` is timestamp, usually in microseconds.

PeakLoc uses these events to identify blinking-like signals and estimate where the underlying emitter was located.

## What is a localization?

A localization is one fitted estimate of an emitter position.

In PeakLoc, a localization row typically contains:

```text
x, y, t_peak, t_1st, t_last, FWHM, E_total, E_total_n, sigma_x, sigma_y, cov_xy
```

The most important fields are:

- `x`: fitted emitter x-position in camera pixels.
- `y`: fitted emitter y-position in camera pixels.
- `t_peak`: peak time of the event signal.
- `E_total`: number of positive polarity events in the ROI.
- `E_total_n`: number of negative polarity events in the ROI.
- `sigma_x`, `sigma_y`, `cov_xy`: localization uncertainty information.
- `fit_success`: whether the numerical fit reported success.
- `fit_status`: text status returned by the fitting code.

## Recommended first path

For a new user, follow this order:

1. Read [Data preparation](data-preparation.md).
2. Prepare one short `.raw` recording in `data/`.
3. Run preflight-only mode.
4. Run a short smoke test.
5. Inspect the output report and QC figures.
6. Only then run a full batch.
7. Use parameter sweeps when tuning `prominence`, `roi_radius`, or uncertainty filters.

## Documentation files

- [Data preparation](data-preparation.md): how to arrange `.raw`, `.bias`, calibration, and folder layout.
- [Configuration guide](configuration.md): explanation of important `config.json` fields.
- [Run modes](run-modes.md): smoke run, preflight, strict preflight, full batch, and parameter sweep.
- [Output interpretation](output-interpretation.md): how to read `.npy`, QC, figures, reports, and rendered images.
- [Use cases and limitations](use-cases-and-limitations.md): what the pipeline is suitable for and where it is currently limited.

## Minimal command summary

Install:

```bash
pixi install
```

Run preflight only:

```bash
pixi run python PeakLoc.py --preflight-only
```

Run full pipeline:

```bash
pixi run peakloc
```

Run with explicit config:

```bash
pixi run python PeakLoc.py --config config.json
```

Render an existing localization file:

```bash
pixi run plot-result /path/to/localizations.npy
```
