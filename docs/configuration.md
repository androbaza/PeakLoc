# Configuration guide

PeakLoc reads settings from `config.json` by default.

You can also pass another config file:

```bash
pixi run python PeakLoc.py --config /path/to/config.json
```

Environment overrides are available for quick changes:

```bash
PEAKLOC_INPUT_FOLDER=/path/to/raw/files pixi run peakloc
PEAKLOC_SLICE_START=0 pixi run peakloc
PEAKLOC_SLICE_DURATION=10000000 pixi run peakloc
```

## Minimal example

```json
{
  "input_folder": "data",
  "slice_start": 0,
  "slice_duration": 100000000,
  "num_cores": 4,
  "prominence": 12.0,
  "dataset_fwhm": 4.01,
  "roi_radius": 8,
  "optical_pixel_size": 67.0,
  "sensor_height": 720,
  "sensor_width": 1280,
  "fit_model": "poisson_joint",
  "allow_uncalibrated": true,
  "calibration_path": null,
  "sigma_psf_px": 1.703,
  "fit_sigma": false,
  "psf_model": "pixel_integrated_gaussian",
  "background_mode": "local_only",
  "max_localization_uncertainty_nm": 50.0,
  "qc_enabled": true,
  "plot_result": true
}
```

## Input and slicing

### `input_folder`

Folder containing `.raw` event-camera recordings.

Example:

```json
{
  "input_folder": "data"
}
```

PeakLoc processes recordings in this folder and creates output folders next to them.

### `slice_start`

Start time of the processed slice, in microseconds.

Example:

```json
{
  "slice_start": 0
}
```

Use this to skip the beginning of a recording.

Examples:

```text
0         = start at recording beginning
1000000   = start at 1 second
60000000  = start at 60 seconds
```

### `slice_duration`

Duration of the processed slice, in microseconds.

Example:

```json
{
  "slice_duration": 100000000
}
```

Examples:

```text
10000000   = 10 seconds
100000000  = 100 seconds
600000000  = 600 seconds
```

For first tests, use a short duration.

### `max_raw_events`

Maximum number of events loaded per OpenEB chunk/read setting.

Example:

```json
{
  "max_raw_events": 1000000
}
```

This is not a scientific threshold. It controls raw event reading behavior and memory pressure.

## Parallelization

### `num_cores`

Number of CPU cores used by parallel parts of the pipeline.

Example:

```json
{
  "num_cores": 4
}
```

Higher values can speed up processing but can also increase memory use.

Use conservative values for first runs.

## Peak detection

### `prominence`

Minimum peak prominence in the cumulative event signal.

Example:

```json
{
  "prominence": 12.0
}
```

Lower values detect more candidate peaks but increase false positives.

Higher values detect fewer peaks but may miss weak emitters.

Typical tuning strategy:

```text
too many noisy localizations -> increase prominence
too few detections -> decrease prominence
```

Use parameter sweeps instead of guessing.

### `peak_min_event_count`

Minimum number of events required before attempting peak interpolation.

Example:

```json
{
  "peak_min_event_count": 2
}
```

Very low values are permissive. Higher values suppress weak event traces.

### `peak_time_threshold`

Maximum temporal separation used when merging nearby peak candidates.

Example:

```json
{
  "peak_time_threshold": 40000.0
}
```

Unit: microseconds.

This means candidates close in space and within about 40 ms can be considered the same event group.

### `peak_neighbors`

Spatial neighborhood radius used when merging local peak candidates.

Example:

```json
{
  "peak_neighbors": 9
}
```

Larger values merge candidates over a broader region.

### `interpolation_coefficient`

Controls how finely each cumulative event trace is interpolated before peak detection.

Example:

```json
{
  "interpolation_coefficient": 5
}
```

Higher values increase temporal sampling of the interpolated curve, but can increase compute time.

### `spline_smooth`

Smoothing parameter used for spline-based ON/OFF timing estimation.

Example:

```json
{
  "spline_smooth": 0.7
}
```

Allowed range:

```text
0 to 1
```

Lower values follow data more closely. Higher values smooth more strongly.

## ROI generation

### `roi_radius`

Radius of the extracted ROI around a detected peak.

Example:

```json
{
  "roi_radius": 8
}
```

The ROI side length is:

```text
2 × roi_radius + 1
```

For `roi_radius = 8`, the ROI is:

```text
17 × 17 pixels
```

Choose a radius large enough to include the PSF and local background, but not so large that neighboring emitters and background dominate.

### `convolution_roi_radius`

Radius used when collecting neighboring events for convolved event traces before peak detection.

Example:

```json
{
  "convolution_roi_radius": 1
}
```

This is separate from the fitting ROI radius.

## Optical scale and sensor geometry

### `optical_pixel_size`

Camera pixel size at the sample plane, in nanometers.

Example:

```json
{
  "optical_pixel_size": 67.0
}
```

This is used for:

- uncertainty conversion from pixels to nm,
- scalebars,
- physical interpretation,
- FRC postprocessing.

### `sensor_height`

Sensor height in pixels.

Example:

```json
{
  "sensor_height": 720
}
```

### `sensor_width`

Sensor width in pixels.

Example:

```json
{
  "sensor_width": 1280
}
```

The expected array convention is:

```text
image[y, x]
```

## Fitting model

### `fit_model`

Currently expected value:

```json
{
  "fit_model": "poisson_joint"
}
```

This uses a joint Poisson model for positive and negative polarity event-count ROIs.

### `psf_model`

Currently expected value:

```json
{
  "psf_model": "pixel_integrated_gaussian"
}
```

The model assumes a Gaussian PSF integrated over sensor pixels.

### `dataset_fwhm`

Approximate full width at half maximum of the PSF in pixels.

Example:

```json
{
  "dataset_fwhm": 4.01
}
```

This value is used as a dataset-level PSF width estimate and as a practical reference for localization and visualization.

The Gaussian conversion is approximately:

```text
sigma = FWHM / 2.35482
```

### `sigma_psf_px`

Fixed Gaussian sigma of the PSF in pixels.

Example:

```json
{
  "sigma_psf_px": 1.703
}
```

This is currently one of the most important fitting parameters.

The current production path uses a fixed PSF sigma. This is intentional because fitting sigma freely can destabilize low-count event ROIs.

A practical way to estimate this value is to use isolated bead data and `calibration_scripts/estimate_bead_sigma.py`.

### `fit_sigma`

Current recommended value:

```json
{
  "fit_sigma": false
}
```

This field exists in the config, but `fit_sigma=true` is not currently the recommended production path.

At present, the robust path is:

```text
estimate sigma from calibration/beads -> set sigma_psf_px -> keep fit_sigma false
```

### Why fixed `sigma_psf_px` is used instead of fitting sigma

Fixed sigma has several advantages:

- fewer free fit parameters,
- more stable optimization,
- more stable uncertainty estimates,
- reduced degeneracy between intensity, background, and width,
- better behavior for sparse event-count ROIs,
- easier comparison between recordings.

A future `fit_sigma=true` path would be useful for diagnosing optical aberrations, defocus, or heterogeneous PSFs, but it needs careful constraints and QC before it should be used as a default.

## Calibration and background

### `allow_uncalibrated`

Allows PeakLoc to run without a calibration file.

Example:

```json
{
  "allow_uncalibrated": true
}
```

Use this for exploration only.

For publication-oriented analysis, prefer:

```json
{
  "allow_uncalibrated": false,
  "calibration_path": "calibration_event_model.npz"
}
```

### `calibration_path`

Path to an event-model calibration `.npz` file.

Example:

```json
{
  "calibration_path": "calibration_event_model.npz"
}
```

Use `null` when no calibration file is provided:

```json
{
  "calibration_path": null
}
```

### `background_mode`

Controls background handling.

Allowed values are:

```text
calibrated_only
calibrated_plus_local
local_only
```

Recommended interpretation:

- `local_only`: useful for exploratory uncalibrated runs.
- `calibrated_only`: uses calibration-derived background.
- `calibrated_plus_local`: combines calibration with local background handling.

If no calibration file is available, use `local_only`.

### `hot_pixel_policy`

Current expected value:

```json
{
  "hot_pixel_policy": "mask"
}
```

Hot pixels from calibration are masked during fitting.

## Event-count and valid-pixel filters

### `min_events_pos`

Minimum positive-polarity event count required for fitting.

Example:

```json
{
  "min_events_pos": 10
}
```

### `min_events_neg`

Minimum negative-polarity event count required for fitting.

Example:

```json
{
  "min_events_neg": 10
}
```

These filters remove ROIs with insufficient polarity information.

### `min_valid_pixels`

Minimum number of valid, unmasked pixels required in the ROI.

Example:

```json
{
  "min_valid_pixels": 50
}
```

If too many ROI pixels are invalid or hot-pixel-masked, the fit is rejected.

## Uncertainty filters

PeakLoc estimates localization uncertainty from the fitted covariance information.

### `max_localization_uncertainty_px`

Maximum accepted localization uncertainty in pixels.

Example:

```json
{
  "max_localization_uncertainty_px": 0.75
}
```

Use `null` to disable this pixel-based filter:

```json
{
  "max_localization_uncertainty_px": null
}
```

### `max_localization_uncertainty_nm`

Maximum accepted localization uncertainty in nanometers.

Example:

```json
{
  "max_localization_uncertainty_nm": 50.0
}
```

This depends on `optical_pixel_size`.

If:

```text
optical_pixel_size = 67 nm / px
max_localization_uncertainty_nm = 50 nm
```

then the corresponding pixel threshold is:

```text
50 / 67 = 0.746 px
```

### `max_fit_cond`

Maximum accepted numerical condition value from the fit.

Example:

```json
{
  "max_fit_cond": 10000000000.0
}
```

Very high condition values can indicate unstable parameter estimation.

## QC options

### `qc_enabled`

Enables QC output generation.

Example:

```json
{
  "qc_enabled": true
}
```

### `qc_output_dirname`

Name of the QC output directory.

Example:

```json
{
  "qc_output_dirname": "qc"
}
```

### `qc_static_dpi`

DPI for static QC figures.

Example:

```json
{
  "qc_static_dpi": 450
}
```

### `qc_save_vector`

Save vector graphics such as SVG/PDF when supported.

Example:

```json
{
  "qc_save_vector": false
}
```

Keep this false unless you specifically need vector files.

### `qc_max_events_for_interactive`

Maximum number of events used in interactive QC plots.

Example:

```json
{
  "qc_max_events_for_interactive": 50000
}
```

Large values produce heavier HTML files.

### `qc_uncertainty_montage_n`

Number of low/high uncertainty examples shown in QC montages.

Example:

```json
{
  "qc_uncertainty_montage_n": 36
}
```

### `qc_generate_html`

Generate HTML QC summaries where supported.

Example:

```json
{
  "qc_generate_html": true
}
```

### `qc_generate_interactive`

Generate interactive QC plots where supported.

Example:

```json
{
  "qc_generate_interactive": true
}
```

### `qc_keep_intermediates`

Keep intermediate QC files.

Example:

```json
{
  "qc_keep_intermediates": false
}
```

Set true only when debugging.

## Plotting

### `plot_result`

Render SMLM-style images after localization.

Example:

```json
{
  "plot_result": true
}
```

If true, the pipeline writes rendered images to the output `figures/` directory.

### `plot_subplotsize`

Controls plotting size for some diagnostic figures.

Example:

```json
{
  "plot_subplotsize": 6
}
```

## Temporary output cleanup

### `cleanup_temp_outputs`

Remove temporary intermediate outputs after successful processing.

Example:

```json
{
  "cleanup_temp_outputs": true
}
```

Set false when debugging intermediate arrays.

## Recommended beginner configs

### Smoke test

```json
{
  "input_folder": "data",
  "slice_start": 0,
  "slice_duration": 10000000,
  "num_cores": 4,
  "prominence": 12.0,
  "dataset_fwhm": 4.01,
  "roi_radius": 8,
  "optical_pixel_size": 67.0,
  "sensor_height": 720,
  "sensor_width": 1280,
  "max_raw_events": 1000000,
  "cleanup_temp_outputs": true,
  "fit_model": "poisson_joint",
  "allow_uncalibrated": true,
  "calibration_path": null,
  "sigma_psf_px": 1.703,
  "fit_sigma": false,
  "psf_model": "pixel_integrated_gaussian",
  "background_mode": "local_only",
  "hot_pixel_policy": "mask",
  "min_events_pos": 10,
  "min_events_neg": 10,
  "min_valid_pixels": 50,
  "max_localization_uncertainty_px": null,
  "max_localization_uncertainty_nm": 50.0,
  "max_fit_cond": 10000000000.0,
  "qc_enabled": true,
  "qc_output_dirname": "qc",
  "qc_static_dpi": 450,
  "qc_save_vector": false,
  "qc_max_events_for_interactive": 50000,
  "qc_uncertainty_montage_n": 36,
  "qc_generate_html": true,
  "qc_generate_interactive": true,
  "qc_keep_intermediates": false,
  "plot_result": true
}
```

### More publication-oriented calibrated start

```json
{
  "input_folder": "data",
  "slice_start": 0,
  "slice_duration": 100000000,
  "num_cores": 8,
  "prominence": 12.0,
  "dataset_fwhm": 4.01,
  "roi_radius": 8,
  "optical_pixel_size": 67.0,
  "sensor_height": 720,
  "sensor_width": 1280,
  "max_raw_events": 1000000,
  "cleanup_temp_outputs": true,
  "fit_model": "poisson_joint",
  "allow_uncalibrated": false,
  "calibration_path": "calibration_event_model.npz",
  "sigma_psf_px": 1.703,
  "fit_sigma": false,
  "psf_model": "pixel_integrated_gaussian",
  "background_mode": "calibrated_plus_local",
  "hot_pixel_policy": "mask",
  "min_events_pos": 10,
  "min_events_neg": 10,
  "min_valid_pixels": 50,
  "max_localization_uncertainty_px": null,
  "max_localization_uncertainty_nm": 50.0,
  "max_fit_cond": 10000000000.0,
  "qc_enabled": true,
  "qc_output_dirname": "qc",
  "qc_static_dpi": 450,
  "qc_save_vector": false,
  "qc_max_events_for_interactive": 50000,
  "qc_uncertainty_montage_n": 36,
  "qc_generate_html": true,
  "qc_generate_interactive": true,
  "qc_keep_intermediates": false,
  "plot_result": true
}
```
