# PeakLoc Configuration Tuning

This guide covers the settings most likely to need adjustment after inspecting
ROI-fit montages and QC artifacts. Change one group at a time and compare the
attempted-fit table, rejection reasons, and uncertainty distributions.

## Fitted Center Moves Away From the Visible PSF

`max_fit_center_offset_px` limits how far the fitted center may move from the
detected peak at the center of the ROI.

- Default: `3.0` pixels for a `roi_radius` of 8.
- Reduce it when fits move toward competing emitters or structure near an ROI edge.
- Increase it when valid emitters are consistently detected more than three pixels
  from the ROI center.
- Set it to `null` only for diagnostics; full-ROI fitting can converge on unrelated
  edge structure.

Keep `roi_radius` large enough to show background around the PSF. Increasing the ROI
without constraining the center gives competing structures more influence.

## Background Absorbs the PSF

Use `background_mode: "calibrated_plus_local"` when the calibration matches the
sensor and bias settings. The calibrated map handles spatial structure while the
local term handles recording-to-recording offsets.

- Inspect `A_pos`, `A_neg`, `bg_pos_local`, `bg_neg_local`, and calibrated background
  sums in `localizations_attempted.csv`.
- Large background-to-amplitude ratios are a warning, especially when the fitted
  center also reaches its allowed offset.
- Use `calibrated_only` only when calibration and experiment illumination are well
  matched; otherwise it cannot absorb a global background offset.
- Use `local_only` as a diagnostic comparison, not as the calibrated production
  default.

Do not increase `min_events_pos` or `min_events_neg` to solve bad fits. Those settings
remove low-event ROIs before fitting and can hide rather than correct model failures.

## Fit Rejections

`max_fit_cond` now evaluates the 2x2 positional covariance condition, not the full
amplitude/background parameter matrix.

- Default: `100`.
- Lower it if strongly directional or collapsed position covariances are accepted.
- Raise it only after inspecting the rejected ROI montage.

`max_localization_uncertainty_nm` controls absolute positional confidence. The
default of 50 nm rejects weak fits even when their x/y covariance has a reasonable
condition number. Tune this against simulated ground truth or a registered reference,
not accepted counts alone.

## Peak and ROI Generation

- Increase `prominence` when temporal noise produces too many candidates.
- Decrease `prominence` only when visible blinking events are absent before fitting.
- Increase `peak_time_threshold` when the same blink is detected repeatedly nearby.
- `polarity_time_gate_us` must include both positive and negative lobes. Increase it
  when ROI montages show one polarity truncated in time; reduce it when unrelated
  events dominate the ROI.
- Keep `sigma_psf_px` tied to bead calibration. Do not tune it merely to increase
  acceptance.

## Runtime and QC

- `num_cores` controls peak, ROI, and fit parallelism. Use the available logical cores
  when memory permits.
- Set `plot_result` to `false` for parameter sweeps that do not need rendered outputs.
- Set `qc_enabled` to `false` for fast iterations. Attempted, accepted, ROI, and QC
  arrays are still written.
- Default `qc_static_dpi` is 200. Use 450 for publication exports.
- `qc_generate_interactive` is off by default because Plotly artifacts add substantial
  runtime and disk usage. Enable it for interactive review runs.
- Keep `cleanup_temp_outputs` false while diagnosing individual slices; restore true
  for routine batch runs.

The current `slice_duration` creates repeated bins from `slice_start` through the end
of a recording. It does not limit a RAW run to one bin. Use a pre-sliced `.npy` for a
single diagnostic interval until bounded RAW processing is implemented.
