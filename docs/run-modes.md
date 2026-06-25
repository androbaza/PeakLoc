# Run modes

PeakLoc supports several run modes. Use them in increasing order of risk.

## 1. Smoke run

A smoke run tests whether the repository, OpenEB bindings, config, and a small recording slice work.

Use a short slice:

```json
{
  "slice_start": 0,
  "slice_duration": 10000000,
  "num_cores": 4
}
```

Then run:

```bash
pixi run python PeakLoc.py --preflight
```

This runs preflight first and then processes only if preflight passes.

Use smoke runs when:

- installing the repository,
- testing a new workstation,
- testing a new camera dataset,
- changing OpenEB or Pixi environments,
- changing major config values.

## 2. Preflight-only

Preflight-only mode checks the config and writes a report without processing recordings.

Command:

```bash
pixi run python PeakLoc.py --preflight-only
```

With explicit config:

```bash
pixi run python PeakLoc.py --config config.json --preflight-only
```

Preflight reports are written to:

```text
reports/preflight_YYYYMMDD_HHMMSS.md
```

Use this before every full batch.

## 3. Strict preflight

Strict preflight is intended for more careful or publication-oriented runs.

Command:

```bash
pixi run python PeakLoc.py --strict-preflight --preflight-only
```

Strict mode should be used when:

- preparing final analysis,
- comparing biological or physical conditions,
- using uncertainty thresholds,
- producing figures for reports or publications,
- validating calibrated data.

A strict preflight failure should usually be fixed, not ignored.

## 4. Preflight and continue

This mode runs preflight first. If preflight passes, it continues to processing.

Command:

```bash
pixi run python PeakLoc.py --preflight
```

With explicit config:

```bash
pixi run python PeakLoc.py --config config.json --preflight
```

Use this for routine processing after your config is stable.

## 5. Full batch run

A full batch run processes all matching recordings in `input_folder`.

Command:

```bash
pixi run peakloc
```

Equivalent:

```bash
pixi run python PeakLoc.py
```

With explicit config:

```bash
pixi run python PeakLoc.py --config config.json
```

Use full batch only after:

- OpenEB import works,
- preflight passes,
- a smoke run produced plausible outputs,
- QC figures look reasonable,
- memory usage is acceptable.

## 6. Parameter sweep

A parameter sweep runs multiple configs generated from a base config and a sweep JSON file.

Example sweep file:

```json
{
  "prominence": [8.0, 12.0, 16.0],
  "max_localization_uncertainty_nm": [30.0, 50.0, 80.0],
  "roi_radius": [6, 8]
}
```

Save as:

```text
sweep/example_sweep.json
```

Run:

```bash
pixi run python PeakLoc.py --config config.json --sweep sweep/example_sweep.json --preflight
```

Strict sweep:

```bash
pixi run python PeakLoc.py --config config.json --sweep sweep/example_sweep.json --strict-preflight
```

Sweep output includes:

```text
sweep/sweep_results.csv
sweep/sweep_results.json
sweep/pareto_localizations_vs_uncertainty.png
sweep/rejection_reason_heatmap.png
sweep/parameter_effects.html
```

Use sweeps to tune:

- `prominence`,
- `roi_radius`,
- `min_events_pos`,
- `min_events_neg`,
- `max_localization_uncertainty_nm`,
- `max_localization_uncertainty_px`,
- `max_fit_cond`.

## 7. Render-only mode

To render an existing localization `.npy` file:

```bash
pixi run plot-result /path/to/localizations.npy
```

This is useful when:

- the localization step already finished,
- you changed only visualization settings,
- you want a quick preview,
- you want to inspect filtered localization tables.

## Suggested workflow for new data

Use this sequence:

```bash
pixi install
pixi run import-test
pixi run python PeakLoc.py --config config.json --preflight-only
pixi run python PeakLoc.py --config config.json --preflight
pixi run peakloc
pixi run plot-result /path/to/localizations.npy
```

Do not start with a full batch on a long recording.
