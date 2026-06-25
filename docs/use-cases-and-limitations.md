# Use cases and limitations

PeakLoc is a research pipeline. It is useful for exploration, localization extraction, QC experiments, and method development. It is not yet a fully validated turnkey publication pipeline.

## Use case 1: quick visualization

Goal:

```text
Check whether a recording contains meaningful event-localization structure.
```

Recommended settings:

```json
{
  "slice_start": 0,
  "slice_duration": 10000000,
  "allow_uncalibrated": true,
  "background_mode": "local_only",
  "plot_result": true,
  "qc_enabled": true
}
```

Run:

```bash
pixi run python PeakLoc.py --preflight
```

Use this for:

- checking whether data loads,
- checking whether the sample emits detectable events,
- generating a first SMLM-style preview,
- debugging installation.

Do not use this alone for publication-level quantification.

## Use case 2: localization extraction

Goal:

```text
Extract accepted localization coordinates from one or more recordings.
```

Recommended path:

1. Prepare `.raw`, `.bias`, and calibration files.
2. Run strict preflight.
3. Process a short slice.
4. Inspect QC.
5. Run full batch.
6. Use `localizations_*.npy` for downstream work.

Example:

```bash
pixi run python PeakLoc.py --config config.json --strict-preflight
```

Main output:

```text
localizations_*.npy
```

Important fields:

```text
x
y
t_peak
E_total
E_total_n
sigma_x
sigma_y
cov_xy
fit_success
fit_status
nll_per_event
```

## Use case 3: QC and refitting experiments

Goal:

```text
Test whether changes to fitting, ROI size, PSF width, or uncertainty filters improve results.
```

Use:

```text
rois_*.npy
localization_qc_*.npy
qc/
```

Useful parameters to sweep:

```json
{
  "prominence": [8.0, 12.0, 16.0],
  "roi_radius": [6, 8, 10],
  "sigma_psf_px": [1.5, 1.7, 1.9],
  "max_localization_uncertainty_nm": [30.0, 50.0, 80.0]
}
```

Run:

```bash
pixi run python PeakLoc.py --config config.json --sweep sweep/qc_sweep.json --preflight
```

Inspect:

```text
sweep/sweep_results.csv
sweep/pareto_localizations_vs_uncertainty.png
sweep/rejection_reason_heatmap.png
sweep/parameter_effects.html
```

## Use case 4: uncertainty filtering

Goal:

```text
Keep only localizations with acceptable estimated positional uncertainty.
```

Use:

```json
{
  "max_localization_uncertainty_nm": 50.0
}
```

or:

```json
{
  "max_localization_uncertainty_px": 0.75
}
```

Use nanometer filtering when `optical_pixel_size` is correct.

Use pixel filtering when comparing configurations independent of physical scale.

Check the QC table:

```text
uncertainty_px
uncertainty_nm
uncertainty_ok
primary_rejection_reason
```

If most fits are rejected by uncertainty, possible causes include:

- `sigma_psf_px` is wrong,
- event counts are too low,
- `prominence` is too permissive,
- `roi_radius` is inappropriate,
- calibration/background handling is poor,
- sample is too dense or noisy.

## Use case 5: drift and FRC postprocessing

PeakLoc contains modules for drift and FRC-related postprocessing.

Use drift correction carefully.

The current drift helper should be treated as a basic helper, not as the recommended final publication-grade drift workflow.

A safer workflow is:

1. Export accepted localization table.
2. Apply independently validated drift correction.
3. Render before and after drift correction.
4. Estimate FRC using split localizations.
5. Confirm that visual structure and FRC behavior are consistent.

FRC requires enough localizations. Sparse data can produce unstable or uninformative FRC estimates.

## Use case 6: synthetic validation

The repository contains synthetic-event and debug visualization utilities.

Synthetic validation is useful for testing:

- whether known emitters are recovered,
- whether peak detection merges nearby events correctly,
- whether uncertainty filters reject poor fits,
- whether parameter sweeps improve precision/recall,
- whether changes to fitting code break expected behavior.

Synthetic validation is not a substitute for real calibration data, but it is useful before changing core algorithms.

## Known limitation: `fit_sigma=true` is not the production path

The current recommended path uses:

```json
{
  "fit_sigma": false,
  "sigma_psf_px": 1.703
}
```

This means the PSF width is fixed during localization fitting.

This is a limitation because the model cannot adapt to:

- defocus,
- field-dependent PSF variation,
- aberrations,
- different fluorophore or optical conditions,
- unusually broad or narrow emitters.

However, fixed sigma is currently safer because event-camera ROIs can be sparse and noisy. Free sigma fitting can become unstable and can trade off against intensity and background.

Recommended current solution:

1. Estimate `sigma_psf_px` from isolated beads.
2. Use that value consistently.
3. Keep `fit_sigma=false`.
4. Use QC montages to check whether the assumed PSF width is plausible.

## Known limitation: overlapping simultaneous emitters

PeakLoc has helper logic related to possible multi-emitter or overlap flags, but simultaneous overlapping emitters are not fully resolved as independent emitters in the current recommended path.

This means:

- two emitters active at nearly the same time and position may be fitted as one,
- asymmetric ROIs may indicate unresolved overlap,
- dense samples can produce biased localizations,
- apparent high localization density is not automatically reliable.

Use lower-density validation data before interpreting dense samples.

## Known limitation: uncalibrated defaults are exploratory

The default config can run without calibration:

```json
{
  "allow_uncalibrated": true,
  "calibration_path": null,
  "background_mode": "local_only"
}
```

This is useful for testing, but it is not publication-grade.

For serious analysis, use:

```json
{
  "allow_uncalibrated": false,
  "calibration_path": "calibration_event_model.npz",
  "background_mode": "calibrated_plus_local"
}
```

and provide matching dark and blank recordings.

## Known limitation: calibration is sensor- and setting-specific

A calibration file is valid only for the sensor geometry and acquisition regime it represents.

Do not reuse calibration blindly across:

- different cameras,
- different sensor sizes,
- different bias settings,
- substantially different laser conditions,
- different temperature/noise regimes.

Store calibration metadata with the outputs.

## Known limitation: default parameters are not universal

Default parameters are starting values.

The following are especially dataset-dependent:

```text
prominence
dataset_fwhm
roi_radius
sigma_psf_px
min_events_pos
min_events_neg
max_localization_uncertainty_nm
background_mode
```

Use sweeps and QC rather than assuming defaults are correct.

## Known limitation: legacy drift helper

The repository contains a drift helper, but it should not be documented as the recommended final drift-correction method.

Use it only as:

- a basic diagnostic,
- a starting point for method development,
- a temporary helper for exploratory runs.

For final analysis, use a validated drift-correction workflow and document it separately.

## Practical decision guide

Use PeakLoc for:

- exploratory event-camera SMLM visualization,
- localization table extraction,
- QC of event-localization fits,
- parameter sweeps,
- testing fixed-PSF event fitting,
- synthetic validation,
- method development.

Be cautious when using PeakLoc for:

- publication-grade absolute localization precision,
- dense simultaneous emitters,
- uncalibrated quantitative comparisons,
- biological conclusions from default parameters,
- drift-sensitive resolution claims,
- comparisons across different bias/camera settings.

## Minimum quality evidence for serious use

Before trusting a dataset, collect and keep:

- raw `.raw` recordings,
- `.bias` files,
- dark calibration recording,
- laser-on blank calibration recording,
- event calibration `.npz`,
- effective config JSON,
- preflight report,
- localization `.npy`,
- QC table,
- QC montages,
- rendered preview,
- postprocessing notebook or script,
- final filtering criteria.

A result without these artifacts should be considered exploratory.
