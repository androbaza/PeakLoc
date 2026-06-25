# Output interpretation

PeakLoc writes outputs next to each input `.raw` recording.

Example input:

```text
data/AF647_coverslip.raw
```

Example output folder:

```text
data/AF647_coverslip/
```

Typical contents:

```text
AF647_coverslip/
├── localizations_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
├── rois_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
├── localization_qc_*.npy
├── figures/
│   ├── *_smlm_*.png
│   └── *_smlm_*_12bit.tiff
├── reports/
│   ├── effective_config_*.json
│   └── report_*.md
└── qc/
    ├── uncertainty_lowest_36_combined.png
    ├── uncertainty_highest_36_combined.png
    ├── uncertainty_quantile_samples.png
    └── other QC outputs
```

## Localization `.npy`

Main output:

```text
localizations_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
```

This is a structured NumPy array. Each row corresponds to one accepted localization.

Important fields:

```text
id
t_peak
x
y
x2
y2
double
I
FWHM
E_total
E_total_n
sub_x
sub_y
t_1st
t_last
sigma_x
sigma_y
cov_xy
A_pos
A_neg
bg_pos_local
bg_neg_local
bg_pos_cal_sum
bg_neg_cal_sum
sigma_psf_px
nll
nll_per_event
fit_success
fit_status
fit_cond
calibration_id
calibrated_background
uncertainty_mode
hot_pixel_count
valid_pixel_count
```

### `x` and `y`

Fitted localization position in camera pixels.

Coordinate convention:

```text
image[y, x]
```

Convert to nanometers:

```text
x_nm = x × optical_pixel_size
y_nm = y × optical_pixel_size
```

### `t_peak`

Peak time in microseconds.

Useful for:

- temporal filtering,
- drift correction,
- odd/even FRC splitting,
- checking acquisition stability.

### `t_1st` and `t_last`

Estimated ON/OFF-like timing boundaries from peak interpolation.

These are useful for studying event timing, but should be treated carefully because timing estimates depend on peak detection and interpolation settings.

### `E_total` and `E_total_n`

Event counts in the positive and negative polarity ROIs.

Low event counts usually imply weaker fits and larger uncertainty.

### `FWHM`

Fitted or model-derived PSF width estimate in pixels.

Because the current recommended path uses fixed `sigma_psf_px`, FWHM is strongly tied to the configured PSF width.

### `sigma_x`, `sigma_y`, `cov_xy`

Uncertainty-related values from the fit covariance.

PeakLoc computes a worst-axis 1-sigma uncertainty from these fields.

### `fit_success`

Boolean fit status from the optimizer.

Accepted localizations should normally have:

```text
fit_success == true
```

### `fit_status`

Text status from the fitting step.

Use this for debugging failed or suspicious fits.

### `fit_cond`

Condition value of the fit. Very large values suggest numerical instability.

Controlled by:

```json
{
  "max_fit_cond": 10000000000.0
}
```

### `nll_per_event`

Negative log-likelihood normalized per event.

This is useful for relative QC within the same dataset and model settings.

Do not compare it blindly across very different configurations or calibration states.

## ROI `.npy`

Main ROI output:

```text
rois_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
```

This stores the extracted event-count ROIs used for fitting.

Typical ROI fields include:

```text
roi
roi_n
roi_x0
roi_y0
t_peak
t_1st
t_last
total_events_roi
total_neg_events_roi
roi_event_times
roi_event_times_n
```

Use ROI files for:

- debugging,
- refitting experiments,
- checking whether the ROI radius is appropriate,
- inspecting positive and negative polarity event distributions,
- developing improved fit models.

For routine downstream analysis, start from the localization `.npy`, not the ROI `.npy`.

## QC table `.npy`

QC output is usually named like:

```text
localization_qc_*.npy
```

This table contains attempted fits and filter decisions.

Important fields:

```text
id
accepted
fit_success
finite_position
finite_uncertainty
positive_uncertainty
fit_cond_ok
valid_pixels_ok
uncertainty_px
uncertainty_nm
uncertainty_ok
fit_cond
valid_pixel_count
nll_per_event
E_total
E_total_n
primary_rejection_reason
```

### `accepted`

Whether the localization passed all configured filters.

### `primary_rejection_reason`

Main reason a localization was rejected.

Common values:

```text
accepted
fit_failed
invalid_position
invalid_uncertainty
fit_condition
valid_pixels
uncertainty
```

Use this field to understand why a parameter setting is too strict or too permissive.

## Figures

The `figures/` directory contains diagnostic and rendered outputs.

When `plot_result` is true, expected SMLM render files include:

```text
*_smlm_<datetime>.png
*_smlm_<datetime>_12bit.tiff
```

### PNG render

The PNG is a visual preview. It may contain annotations such as a scalebar.

Use it for:

- quick inspection,
- lab notebook screenshots,
- sanity checks.

### 12-bit TIFF render

The TIFF render is more suitable for downstream image analysis.

Use it for:

- external image tools,
- quantitative image workflows,
- figure preparation after validation.

## Reports

The `reports/` folder stores run reports and effective settings.

Typical contents:

```text
effective_config_*.json
report_*.md
```

The effective config is important because it records the actual parameters used for that run.

Always keep reports together with localization outputs.

## QC directory

The `qc/` folder contains detailed diagnostic plots.

Common outputs include:

```text
uncertainty_lowest_36_combined.png
uncertainty_highest_36_combined.png
uncertainty_lowest_36_positive.png
uncertainty_highest_36_positive.png
uncertainty_lowest_36_negative.png
uncertainty_highest_36_negative.png
uncertainty_quantile_samples.png
uncertainty_failed_fits.png
```

Use these plots to answer:

- Are low-uncertainty fits visually plausible?
- Are high-uncertainty fits weak, asymmetric, clipped, or noisy?
- Are many fits failing because of invalid covariance?
- Does the configured `sigma_psf_px` match the observed event-count ROIs?
- Is `roi_radius` too small or too large?

## Loading outputs in Python

Example:

```python
import numpy as np

locs = np.load("localizations_prominence_fwhm_4.01_prominence_12.0.npy")
print(locs.dtype.names)
print(locs.shape)

x = locs["x"]
y = locs["y"]
t = locs["t_peak"]
```

Convert to nanometers:

```python
optical_pixel_size_nm = 67.0

x_nm = locs["x"] * optical_pixel_size_nm
y_nm = locs["y"] * optical_pixel_size_nm
```

Filter by time:

```python
start_us = 0
stop_us = 100_000_000

mask = (locs["t_peak"] >= start_us) & (locs["t_peak"] < stop_us)
locs_slice = locs[mask]
```

Filter by event count:

```python
mask = (locs["E_total"] >= 10) & (locs["E_total_n"] >= 10)
locs_filtered = locs[mask]
```

Load QC table:

```python
qc = np.load("localization_qc_example.npy")
accepted = qc["accepted"]
reasons = qc["primary_rejection_reason"]
```

Count rejection reasons:

```python
import numpy as np

unique, counts = np.unique(qc["primary_rejection_reason"], return_counts=True)
for reason, count in zip(unique, counts):
    print(reason, count)
```

## Interpreting a good first run

A plausible first run should usually show:

- nonzero accepted localizations,
- localization positions inside the sensor area,
- finite uncertainty values,
- not all fits rejected by the same reason,
- QC montages where accepted fits visually align with ROI event density,
- rendered image with structure consistent with the sample.

A suspicious run may show:

- zero accepted localizations,
- almost all fits rejected by uncertainty,
- many invalid covariance values,
- localizations on ROI borders,
- strong hot-pixel patterns,
- rendered image dominated by isolated noisy pixels,
- very high localization count from obvious background noise.

## Recommended downstream starting point

For most analysis, start from:

```text
localizations_*.npy
```

Then apply:

1. Fit-success filtering.
2. Uncertainty filtering.
3. Event-count filtering.
4. Time filtering.
5. Optional drift correction.
6. Optional FRC resolution estimate.
7. Final rendering.
