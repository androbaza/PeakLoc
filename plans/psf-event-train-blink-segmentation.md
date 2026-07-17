# PSF event-train blink segmentation

## Goal

Replace the current permissive spline-derived ROI timing window with a physically constrained,
event-level segmentation stage. A valid localization must contain a compact positive-polarity
turn-on train followed by a spatially matching negative-polarity turn-off train. Each matched pair
becomes one blink ROI; unrelated background events and additional blink cycles must not enter that
ROI.

The initial implementation should be conservative. A candidate without two convincing transition
trains is rejected rather than rescued by widening its time window.

## What the three reference examples establish

All offsets below are relative to the saved detection peak and were measured from the exported
`raw_roi_events.csv`. Counts refer to the compact region within 4.5 pixels of the fitted centre, not
the full square ROI.

| Reference | Event-train evidence | Required result |
| --- | --- | --- |
| Rapid-blinking `blink_05` | Positive train is concentrated from about -40 to 0 ms, with 231 positive events in that interval. The principal negative train begins around +30 ms and extends as a weaker tail. A new positive cluster around +120 to +140 ms is spatially displaced. | Keep the central ON/OFF pair. Start near the dense -40 to 0 ms train, end after its matched negative train, and exclude the later positive cluster. |
| 2025 `blink_02` | Positive events form one compact train over roughly -100 to -10 ms; its densest late portion is near the user's expected -40 to -20 ms region. Negative events occur mainly from about +10 to +110 ms. Events extending beyond +/-2 s set the current proxy endpoints but are not part of the compact transition. | Keep one pair and discard the remote events. Treat the exact onset boundary as an annotation/calibration target rather than forcing all pre-peak activity into the ROI. |
| 2025 `blink_01` | The compact core contains at least three ordered positive/negative cycles: approximately -340 to -160 ms, -140 to +200 ms, and +260 to +500 ms. | Emit three independently timed ROI records if all three pairs pass quality gates. Never fit the full -677 to +849 ms interval as one blink. |

These examples also expose a limitation of the current `peak_time_threshold = 40 ms`: temporal
non-maximum suppression must not destroy a second physical blink before event-train segmentation
can inspect it.

## Photophysical interpretation

Three timescales must remain distinct:

1. Fluorescence lifetime is on the nanosecond scale and is not measured by this pipeline.
2. AF647 ON-state dwell time is condition dependent. Lin et al. report exponential ON-time
   distributions and ON-state lifetimes of only a few milliseconds at 31--97 kW cm^-2 excitation.
3. An event-camera transition train is the thresholded, pixel-by-pixel response of a spatial PSF.
   Its extent can include pixel threshold dispersion, spatial sampling, and detector/electronics
   effects, so it need not equal the dye's ON-state dwell time.

AF647 kinetics therefore constrain ordering and plausible duration, but literature values must be
used as a soft prior until laser irradiance, buffer, camera bias, and sparse-molecule calibration
match the recording. The final prior should be learned separately for each acquisition condition.

Primary references:

- Lin et al., *PLOS ONE* 10, e0128135 (2015),
  <https://doi.org/10.1371/journal.pone.0128135>.
- Diekmann et al., *Nature Methods* 17, 909--912 (2020),
  <https://doi.org/10.1038/s41592-020-0918-5>.
- van de Linde et al., *Nature Protocols* 6, 991--1009 (2011),
  <https://doi.org/10.1038/nprot.2011.336>.

## Proposed pipeline

### 1. Preserve temporal candidates

Use the present peak finder only to seed a local spatial-temporal context. Do not use its cubic
spline bounds as blink boundaries.

- Retain all local temporal maxima within a connected spatial candidate.
- Replace early 40 ms destructive merging with a short duplicate-peak suppression window, initially
  3 ms, or defer suppression until event-train pairs are known.
- Read a fixed physical-time context around each candidate, initially 150 ms before and 200 ms
  after. This is a search context only: no event enters a fit merely because it is inside it.
- If a transition train touches a context edge, extend the context once by 100 ms. Reject a second
  edge contact rather than allowing unbounded expansion.

This removes the present dependence on interpolation-array index distance, whose physical duration
changes with event density.

### 2. Build a compact PSF-core signal

Temporal segmentation should use the part of the ROI in which events are likely to come from the
candidate PSF.

1. Estimate a provisional centre from the peak image using a robust weighted centroid or the
   existing fixed-sigma PSF template.
2. Set the core radius to `clip(2.5 * psf_sigma_px, 3 px, 5 px)`. With the current
   `psf_sigma = 1.703 px`, this is 4.26 px and closely matches the 4.5 px diagnostic used above.
3. Use the remaining fit ROI as a background annulus. Estimate background independently for each
   polarity from spatial-annulus events and the temporal context flanks.
4. Assign a Gaussian spatial weight to every core event. Keep raw event identity and fields
   `x`, `y`, `p`, and `t`; do not aggregate away timestamps needed for the final ROI.

The segmentation signal is then a 1 ms time series for each polarity containing:

- PSF-weighted event count;
- number of distinct active pixels;
- polarity purity;
- weighted centroid and radial spread; and
- excess over the expected annular-background count.

The 1 ms series may be smoothed only with a 2--3 ms causal or symmetric kernel for scoring. Raw
timestamps determine reported boundaries.

### 3. Detect transition trains with background-relative hysteresis

Detect positive and negative trains separately. A train starts when the background-normalized
score exceeds a strict entry threshold and continues at a lower threshold. Bridge only short gaps.
This can first be implemented as a deterministic hysteresis detector, then replaced by a constrained
hidden semi-Markov model if validation shows a clear benefit.

Initial, deliberately strict settings:

| Setting | Initial value | Purpose |
| --- | ---: | --- |
| `temporal_bin_us` | 1,000 | Resolve transition structure without sparse microsecond bins. |
| `burst_enter_z` | 5.0 | Require clear excess over local background. |
| `burst_continue_z` | 2.0 | Retain the lower-density tail of a real train. |
| `burst_max_gap_us` | 8,000 | Bridge threshold gaps inside a train, but not long background intervals. |
| `burst_min_events` | 12 | Reject isolated background events. |
| `burst_min_active_pixels` | 5 | Require a spatial PSF response rather than one hot pixel. |
| `burst_min_polarity_purity` | 0.80 | Require positive ON and negative OFF evidence. |
| `burst_min_duration_us` | 2,000 | Reject single-bin impulses. |
| `burst_max_duration_us` | 120,000 | Prevent permissive trains while retaining the observed 2025 reference for calibration. |

The thresholds are starting hypotheses, not final constants. The reference examples are too few to
estimate reliable operating characteristics.

For an accepted train, set its start and end to the first and last raw event assigned to bins above
the continuation threshold. Do not use the first or last event in the search context.

### 4. Pair ON and OFF trains as one physical cycle

Construct candidate edges from every positive train to every later negative train in the same local
context. Score an edge using:

- correct polarity order;
- centre separation, initially no more than 1.25 px;
- compatible radial spread and PSF-template correlation;
- unexplained background between and around the trains;
- ON-state interval between the end of the positive train and start of the negative train; and
- a condition-specific photophysical duration prior.

Use dynamic programming or minimum-cost ordered matching to choose a non-overlapping set of pairs.
The matching objective should prefer several strong, compact cycles over one long interval spanning
multiple cycles. A new positive train before the chosen negative train creates an ambiguity penalty;
if both possible pairs remain plausible, reject them instead of merging them.

Initial physical guards:

| Setting | Initial value | Interpretation |
| --- | ---: | --- |
| `pair_min_on_state_us` | 1,000 | Below this, timing is likely unresolved. |
| `pair_max_on_state_us` | 150,000 | Hard guard, intentionally broader than the literature prior. |
| `pair_max_cycle_span_us` | 250,000 | Maximum from first ON-train event to last OFF-train event. |
| `pair_max_centroid_distance_px` | 1.25 | Both transitions must describe the same spatial emitter. |
| `pair_duration_prior_mode_us` | 30,000 | Soft starting prior centred on the observed 20--40 ms lead, not a hard crop. |

The mode must be fitted from accepted sparse AF647 calibration data for each laser/buffer/bias
condition. A broad log-normal or gamma prior is safer than a symmetric Gaussian because switching
times are right-skewed.

### 5. Split and assign events without overlap

Each selected ON/OFF pair creates one `BlinkInterval` and one ROI record.

The temporal extent is the union of its two transition trains plus the intervening ON-state interval.
Events enter the ROI only when both conditions hold:

1. their timestamp lies inside that pair's explicit interval; and
2. their spatial likelihood under the provisional PSF exceeds their likelihood under the local
   background/competing-PSF models.

When two pairs are close, assign each event to the pair with maximum posterior probability and keep
an assignment margin. Do not duplicate an event across ROIs. Reject events below the margin as
background. This replaces the current one-sided polarity gates, which admit arbitrarily early
positive and arbitrarily late negative events.

### 6. Fit, refine once, and stop

1. Fit the provisional pair using only its assigned events.
2. Recompute the PSF core, spatial likelihoods, train boundaries, and pair score around the fitted
   centre.
3. Refit once if the centre moved by at least 0.1 px or either boundary moved by at least 1 ms.
4. Accept when the second pass is stable. Reject non-convergent candidates; do not widen their
   windows repeatedly.

This two-pass design provides the fitted centre needed for clean segmentation without making the
initial segmentation depend on a fit that already contains background or neighbouring blinks.

### 7. Apply explicit acceptance and rejection gates

A blink must pass all of the following:

- one positive train and one later negative train are matched;
- both trains pass event-count, active-pixel, polarity-purity, and background-excess gates;
- transition centroids and widths are compatible with one PSF;
- fitted PSF likelihood is better than a diffuse/background model;
- the fitted width or residual pattern is not out-of-focus;
- the pair satisfies hard cycle-duration guards;
- no strong unassigned transition train lies inside the final interval; and
- iterative boundaries and centre converge.

Store stable machine-readable rejection reasons such as `missing_on_train`, `missing_off_train`,
`diffuse_train`, `centroid_mismatch`, `ambiguous_pairing`, `cycle_too_long`, and
`segmentation_not_converged`. Blinks without clear rapid trains should fail one of the first four
gates rather than receive a larger ROI.

## Data model and configuration changes

Return named types instead of enlarging existing tuples:

```python
@dataclass(frozen=True)
class TransitionTrain:
    polarity: int
    first_event_us: int
    last_event_us: int
    event_indices: np.ndarray
    weighted_centroid_x: float
    weighted_centroid_y: float
    event_count: int
    active_pixel_count: int
    polarity_purity: float
    background_z_score: float


@dataclass(frozen=True)
class BlinkInterval:
    seed_peak_us: int
    on_train: TransitionTrain
    off_train: TransitionTrain
    assigned_event_indices: np.ndarray
    pair_score: float
    iteration_count: int
```

Add a `temporal_segmentation` configuration section containing the settings above. Keep current
`find_on_off` settings available behind a temporary compatibility flag, but mark spline bounds and
`polarity_time_gate` as legacy. Saved ROI/localization records should include train boundaries,
pair scores, background scores, iteration count, parent seed identifier, split-cycle identifier,
and rejection reason.

## Tuning protocol

The settings should be tuned iteratively, but not by repeatedly inspecting only the three named
examples.

1. Annotate the three references with ON-train start/end, OFF-train start/end, cycle count, and
   accept/reject status. The user-proposed -20 to -40 ms onset should be recorded as an annotation,
   while the earlier activity in 2025 `blink_02` remains visible for adjudication.
2. Build a stratified annotation set of at least 50--100 candidates spanning all acquisition
   conditions, event counts, focus quality, crowded regions, and diffuse backgrounds.
3. Split by recording, not by event, into tuning and locked validation sets.
4. Optimize in this order: background/train thresholds, spatial compactness, ON/OFF pairing, then
   duration priors. This avoids using the photophysical prior to hide a poor train detector.
5. Score boundary error, event-assignment precision/recall, blink accept/reject precision, and
   split/merge error. Give false merges and background contamination more weight than rejected weak
   blinks.
6. Start strict and relax one parameter family at a time only when validation recall improves
   without materially increasing merge or contamination rates.
7. Lock thresholds per acquisition condition and confirm on an untouched recording.

Recommended primary metrics:

- median absolute ON/OFF boundary error in milliseconds;
- fraction of assigned events within expert-labelled trains;
- false-merge rate per 1,000 candidates;
- missed-split rate per 1,000 candidates;
- accepted-blink precision and yield; and
- localization shift relative to expert-cleaned event assignments.

## Required diagnostic output

Every validation candidate should get a publication-ready deconstruction containing:

- raw polarity raster with every timestamp available in source data;
- PSF-weighted 1 ms train scores and entry/continuation thresholds;
- core versus annulus event rates;
- detected trains and matched pair edges;
- final assigned versus rejected events in x-y-time;
- first-pass and refined boundaries;
- fit residual or PSF-versus-diffuse likelihood; and
- a concise acceptance/rejection table.

Export high-resolution PNG and vector PDF, and write plotted source-data CSV files beside them in
the run's `qc/photophysics_deconstruction/` directory.

## Implementation sequence

1. Add pure feature-extraction, train-detection, ordered-pairing, and assignment functions with
   synthetic focused, weak, background-contaminated, and multi-cycle examples.
2. Add `BlinkInterval` provenance to ROI generation while retaining a legacy compatibility path.
3. Add the two-pass fit/segmentation loop and stable rejection reasons.
4. Extend the deconstruction notebook so thresholds and pair decisions are inspectable.
5. Tune on annotated examples, freeze condition-specific profiles, and run the locked validation.
6. Compare localization yield, split/merge rate, timing distributions, and final reconstruction
   quality against the current pipeline before changing the default.

## Definition of done

- Rapid `blink_05` selects the compact central pair and excludes the later positive cluster.
- 2025 `blink_02` excludes the multi-second flank events and exposes any disagreement between the
  algorithmic and expert onset boundaries.
- 2025 `blink_01` becomes independently fitted, non-overlapping blink ROIs rather than one long ROI.
- Candidates lacking compact, polarity-ordered trains are rejected with an explicit reason.
- No boundary depends on interpolation sample index or the first/last event in a permissive window.
- The locked validation set meets thresholds chosen before it is evaluated.
- Nature-quality diagnostics and their source data are written for all reference and validation
  examples.
