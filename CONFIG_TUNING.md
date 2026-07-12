# PeakLoc Configuration Tuning

This guide covers the settings most likely to need adjustment after inspecting
ROI-fit montages and QC artifacts. Change one group at a time and compare the
attempted-fit table, rejection reasons, and uncertainty distributions.

## Current Baseline Status

The root `config.json` detection defaults were compared on dense 10-12 s and
moderate 30-32 s slices from the configured rapid-switching recording.

- Keep `prominence: 12`, `peak_time_threshold: 40000`, and `peak_neighbors: 9`.
  Relaxing grouping produced 4-5% close space/time duplicates and more boundary
  fits; stronger grouping discarded 27-41% of accepted localizations.
- Keep `convolution_roi_radius: 1`. Radius 2 made about four times as many raw
  candidates, ran 37-53% longer, and gave worse uncertainty and fit rejection rates.
- Keep `interpolation_coefficient: 5` and `spline_smooth: 0.7`. Higher values made
  ROI timing windows shorter but did not improve acceptance or uncertainty.
- `sigma_psf_px: 1.703`, `dataset_fwhm: 4.01`, geometry, and pixel size are
  calibration-derived values. Do not optimize them from acceptance counts.

The remaining fit gates are deliberately conservative policy thresholds, not values
proven optimal by these slices. Change them only with a reference sample, simulation,
or explicit ROI-fit review.

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
  when ROI montages show one polarity truncated in time. It expands narrow spline
  windows but does not cap broad ones, so reducing it will not remove unrelated events
  from a long ROI window. A maximum ROI timing window would be a separate model change
  that must be validated against blink-duration ground truth.
- Keep `sigma_psf_px` tied to bead calibration. Do not tune it merely to increase
  acceptance.

`min_events_pos`, `min_events_neg`, `min_valid_pixels`,
`max_localization_uncertainty_nm`, `max_fit_cond`, and
`max_fit_center_offset_px` are quality-policy gates. The current 10-15 s diagnostic
had 283-289 valid pixels per attempted ROI, only 23 pre-fit event-count drops, 19
uncertainty rejections, and 3 condition rejections. These observations support the
current values but do not justify loosening them to increase accepted counts.

## Calibration and Fixed Implementation Settings

- `fit_sigma: true` is not implemented. Keep it `false`.
- `fit_model`, `psf_model`, and `hot_pixel_policy` currently have one production
  implementation each. Treat them as provenance fields, not sweep parameters.
- `qc_keep_intermediates` is reserved and currently has no runtime effect.
- The configured calibration has valid coverage for the selected sensor and should be
  regenerated only when sensor bias, optics, or camera geometry changes.

## Runtime and QC

- `num_cores` is the available CPU ceiling; `max_parallel_workers` is the safety cap
  used for memory-intensive peak, ROI, and fit stages. Start with four workers even on
  larger machines, then raise the cap only after a full run is stable.
- `max_raw_events` is the OpenEB RAW-reader rolling buffer size, not a processing cap.
  It must exceed the peak decoder batch for the recording. This rapid-switching
  recording requires 100,000,000; lower values make OpenEB abort before PeakLoc can
  stream events to disk. The disk-backed event cache prevents that buffer from being
  multiplied across slices or workers.
- Set `plot_result` to `false` for parameter sweeps that do not need rendered outputs.
- Set `qc_enabled` to `false` for fast iterations. Attempted, accepted, ROI, and QC
  arrays are still written.
- Default `qc_static_dpi` is 200. Use 450 for publication exports.
- `qc_generate_interactive` is off by default because Plotly artifacts add substantial
  runtime and disk usage. Enable it for interactive review runs.
- `qc_generate_temporal_3d` separately controls the temporal Plotly artifact. It is
  enabled for full QC runs and samples at most `qc_max_events_for_interactive`
  localizations, while the statistics and static maps use all accepted localizations.
- The sparse-structure spatial mask is calibrated from the first 60 seconds. For the
  current rapid-switching recording, `spatial_mask_min_events: 400`,
  `spatial_mask_min_component_pixels: 20`, and a 12-pixel margin retained a sparse
  high-event target map while preserving an ROI/convolution support halo. The measured
  137,837,220-event calibration yielded 25.17% target coverage and 32.36% support
  coverage, avoiding about three quarters of convolution targets. Inspect the saved
  report masks before relying on it for quantitative comparisons.
- Keep `cleanup_temp_outputs` false while diagnosing individual slices; restore true
  for routine batch runs. A new run now clears stale per-slice localization, ROI, and
  QC arrays before aggregation, so retained intermediates cannot contaminate it.

## Temporal Blink Artifacts

The `qc/` folder contains temporal maps, a per-spatial-bin CSV, summary JSON/Markdown,
timing distributions, and `temporal_blink_spatial_3d.html`. The maps visualize the
accepted localizations over the final image using the first ROI event, first-to-last
ROI event duration, and last ROI event.

These are ROI-window timing proxies, not direct molecular turn-on or turn-off times:
they include all events in the fitted ROI window and can reflect background, hot pixels,
and the peak-detection timing window. Use spatial patterns as a lead for review, then
validate any biological claim with signal-associated event timing or an independent
measurement.

The current `slice_duration` creates repeated bins from `slice_start` through the end
of a recording. It does not limit a RAW run to one bin. RAW inputs are staged on disk
and processed one time slice at a time, so long recordings no longer require the full
event array to remain resident.
