# Output interpretation

PeakLoc writes each run into a timestamped subfolder next to its input `.raw` recording.

Example input:

```text
data/AF647_coverslip.raw
```

Example output folder:

```text
data/AF647_coverslip/20260712_143015_123456/
```

Typical contents:

```text
AF647_coverslip/
└── 20260712_143015_123456/
    ├── share/                         # concise collaborator-ready bundle
    │   ├── README.md
    │   ├── figures/
    │   │   ├── smlm_reconstruction.png
    │   │   ├── detection_and_fit_summary.png
    │   │   └── temporal_blink_*.png
    │   ├── statistics/
    │   └── metadata/
    └── debug/                          # detailed technical audit trail
        ├── arrays/
        │   ├── localizations_*.npy
        │   ├── rois_*.npy
        │   └── localization_qc_*.npy
        ├── qc/
        │   ├── roi_detection_decision_replay.png
        │   ├── fit_uncertainty_quantile_montage.png
        │   └── fit_hot_pixel_dominated_rois.png
        ├── reports/
        ├── provenance/
        └── temp_files/
```

Send the complete `share/` directory to collaborators. It deliberately contains only
the final reconstruction, concise statistical evidence, and provenance needed to
interpret it. Keep `debug/` with the run owner: it contains large arrays and technical
diagnostics needed to audit choices, diagnose failures, or reproduce the hand-off.

## Localization `.npy`

Main output:

```text
debug/arrays/localizations_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
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
debug/arrays/rois_prominence_fwhm_<fwhm>_prominence_<prominence>.npy
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
debug/arrays/localization_qc_*.npy
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

## Collaborator bundle: `share/`

The `share/` directory is the complete hand-off package. It is intentionally small,
readable without code, and separated from arrays and implementation diagnostics.
Start with `share/README.md`, then inspect its figures and accompanying statistics.

### Final figures

`share/figures/` contains named, publication-sized outputs. Static charts are written
as 450 dpi PNGs and as PDFs where a vector representation is meaningful.

- `smlm_reconstruction.png` and `smlm_reconstruction_12bit.tiff` are the cropped
  final reconstruction preview and quantitative raster, respectively.
- `detection_and_fit_summary.png` summarizes the detection funnel, fit uncertainty,
  spatial localization density, and the hot-pixel screen.
- `temporal_blink_timing_estimates.png` and
  `temporal_blink_dynamics_over_recording.png` summarize temporal dynamics.
- `temporal_blink_spatial_maps.png` shows spatial timing summaries with a labelled
  colorbar and physical time units for every panel.
- `frc_resolution.png`, when enough localizations are available, shows the FRC result.

The spatial maps use an opaque neutral value for bins without enough localizations and
a colorbar for each time scale. A sparse dataset therefore remains visibly sparse
instead of becoming an apparently empty white square.

The timing-estimate distribution is displayed from 0 to 1000 ms. Longer intervals are
not discarded: the figure annotates their number and the complete distribution remains
in the statistics JSON.

### Statistics and metadata

`share/statistics/` contains the data underlying the collaborator figures in compact,
machine-readable form:

- `run_summary.json` records headline counts and fit-quality summaries.
- `temporal_blink_statistics.json` records timing quantiles and the number outside the
  0–1000 ms display range.
- `temporal_blink_spatial_bin_statistics.csv` and
  `temporal_blink_dynamics_over_recording.csv` preserve plotted temporal summaries.
- `frc_summary.json`, when available, records the resolution estimate and warning.

`share/metadata/` stores the effective configuration, run metadata, software versions,
and configuration hash. Keep it with the figures whenever sharing or archiving a run.

## Technical audit trail: `debug/`

`debug/` contains the detailed evidence needed to diagnose, reproduce, or challenge a
pipeline decision. It is not the default collaborator hand-off.

- `debug/arrays/` contains accepted localizations, attempted fits, ROIs, and QC arrays.
- `debug/qc/roi_detection_decision_replay.png` shows example ROI event signals and
  their cumulative selected-event decision trace.
- `debug/qc/fit_uncertainty_quantile_montage.png` samples fits through uncertainty
  quantiles, so a permissive setting cannot hide poor ROIs in an average summary.
- `debug/qc/fit_hot_pixel_dominated_rois.png` highlights ROIs with a large single-pixel
  event fraction, including rejected or accepted cases that need manual review.
- `debug/reports/` and `debug/provenance/` retain slice diagnostics, preflight results,
  detailed tables, and configuration provenance.

Use the decision replay and fit montages before changing peak, ROI, or fit thresholds.
They distinguish a plausible PSF-like event from a single hot pixel or a weak,
background-dominated ROI.

## Loading outputs in Python

Example:

```python
import numpy as np

locs = np.load("debug/arrays/localizations_prominence_fwhm_4.01_prominence_12.0.npy")
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
qc = np.load("debug/arrays/localization_qc_example.npy")
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
debug/arrays/localizations_*.npy
```

Then apply:

1. Fit-success filtering.
2. Uncertainty filtering.
3. Event-count filtering.
4. Time filtering.
5. Optional FRC resolution estimate.
6. Final rendering.
