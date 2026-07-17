# PSF event-train blink segmentation

## Goal and current decision

Replace the permissive spline-derived ROI timing proxy with a conservative event-level
segmentation stage. A valid localization must contain a spatially compact positive-polarity
turn-on train and a matching negative-polarity turn-off train anchored to the retained detection
peak. The fit then uses the complete rectangular ROI, but only positive events in the explicit ON
support and negative events in the explicit OFF support.

The real-data prototype supports this design, with important corrections to the original plan:

- The frequently observed 20--40 ms lead is a soft calibration prior, not a boundary rule.
- In the inspected samples, a 3.5 px refined core separated the transition more cleanly than the
  previous approximately 4.25 px diagnostic core, while a wider discovery pass protected against a
  slightly displaced seed.
- Polarity-specific gaps were needed for the named ideals: 3 ms isolated the concentrated positive
  ON burst, while the weaker negative OFF response needed 8 ms to avoid fragmentation. A shared
  8 ms gap chained sparse early positive activity into the user-identified 2025 ideal blink.
- One retained peak should produce at most one anchored ON/OFF pair. Distinct cycle candidates are
  often already represented by separate retained peaks and must not be emitted repeatedly from
  every neighbouring seed.
- Compact-core events are evidence for timing, not a spatial crop for fitting. Spatially cropping
  the final ROI risks biasing the PSF fit.
- The first release remains feature-gated and disabled by default. Current samples are sufficient
  for a conservative prototype, but not for changing production defaults.

## Real-data prototype review

### Scope

The hypothesis was tested without a full recording run:

- all 15 previously exported photophysics deconstructions from the 2025, 405-induced, and rapid-
  blinking conditions (7,537 raw ROI events); and
- 18 fresh deterministic accepted-fit samples (seed 7647), stratified by low, middle, and high
  event count in the rapid-blinking and 405-induced recordings. Only a 500 ms context around
  each seed was read for this second set (8,799 cropped ROI events).

The combined prototype therefore covered 33 candidates and 16,336 events. It deliberately
contains different event yields and acquisition conditions, but it is not an expert-labelled
locked validation set.

### Evidence

| Observation | Result | Design consequence |
| --- | --- | --- |
| Fresh stratified candidates | 14/18 had an anchored train pair; low/middle/high strata contributed 5/6, 5/6, and 4/6 accepted pairs. All four rejections were `no_anchored_pair`. | Compact trains occurred beyond the named examples in this two-recording sample and were not confined to the highest-count fits. |
| Ten accepted saved-sample pairs | The median segmented cycle span was 142.5685 ms versus a 523.067 ms median legacy span; the median per-candidate reduction was 74.678%. | The legacy proxy was much longer than the compact-train selection in these samples, consistent with temporal-flank contamination but not proof of exact physical boundaries. |
| Eighteen fresh candidates | The median selected cycle span was 93.4 ms (range 50.2--151.1 ms). | A fixed narrow duration would reject some algorithmically accepted candidates. Retain hard safety bounds plus measured train evidence. |
| Visual-inspection onset hypothesis | In the fresh accepted set, the median onset lead was 33.5 ms and 7/14 starts lay 20--40 ms before the seed. The range was 5.9--59.3 ms. | The 20--40 ms observation is a useful empirical prior, but measured train evidence still needs broader support. |
| Annular background | Expected core background was only 0.004--0.324 event/ms in inspected intervals. | A normal z-score is unstable in this sparse regime. Use an area-corrected density ratio and a shrinkage-stabilized root Poisson-deviance score. The score is a gate, not a calibrated significance test. |
| Core radius | Approximately 3.5 px separated trains more cleanly than 4.26 px in the inspected examples, while both named ideals retained ample transition events at 3 px. | Use a wider discovery core followed by a 3.5 px refined core. |
| Polarity gap sweep | A shared 3 ms gap fragmented sparse OFF responses; a shared 8 ms gap merged the 2025 ideal ON activity from -101 to -13 ms. ON=3 ms and OFF=8 ms accepted both ideals with ON starts at -36.214 and -36.316 ms. | Use polarity-specific gaps and keep both as validation parameters. |

With the final strict prototype defaults, 10/15 saved examples are accepted, and all 15 exact RAW
reconstruction checks for stored first/last timestamps and positive/negative counts pass. Both
user-identified ideal blinks are accepted with ON starts at -36.214 ms (rapid `blink_05`) and
-36.316 ms (2025 `blink_02`) relative to their retained seeds. The displayed central seed of 2025
`blink_01` is rejected because its terminal ON group has only five core events, below the 12-event
ON-train gate. Persisted neighbouring seeds at -269.623 ms and +358.177 ms relative to the displayed
seed are independently accepted instead of being merged into its interval. These are algorithmic
interval decisions, not expert-confirmed physical cycle identities. Other rejections are explicit
(`missing_on_train`, `missing_off_train`, or `no_anchored_pair`) rather than being rescued with a
wider ROI.

The persisted neighbouring seeds support selecting at most one anchored pair per retained seed
while keeping the current 40 ms non-maximum suppression for this implementation. In a separate
exploratory sweep, reducing suppression to 3 ms turned the 15 saved contexts into 46 seed groups.
Without labels and duplicate ownership, that increase establishes neither additional physical
cycles nor improved recall and would add repeated segmentation and assignment work.

### What the prototype does not prove

- The 33 candidates are not expert-labelled for exact first/last transition events. The same small
  collection informed parameter iteration and the reported comparison, so the results are in-sample.
- The candidates were drawn from previously accepted, event-rich fits. Acceptance rates are
  conditional on that selection and do not estimate raw-candidate sensitivity or specificity.
- The fresh 18-candidate set covers only the rapid and 405-induced recordings; the 2025 condition is
  represented by its five saved deconstructions.
- Samples do not establish sensitivity for weak, diffuse, out-of-focus, or very crowded emitters.
- Acceptance yield is not yet a suitable optimization objective: a conservative rejection can be
  preferable to a falsely merged blink.
- Timing agreement and shorter selected spans do not prove that boundaries are physically exact or
  that localization is less biased. Segmented maps were not refitted; fit-centre shift and residuals
  still require comparison against expert-cleaned ROIs.
- Slice-boundary recall is intentionally reduced because the initial integration has no temporal
  halo.

## Photophysical interpretation

Keep three timescales distinct:

1. Fluorescence lifetime is on the nanosecond scale and is not measured here.
2. AF647 ON-state dwell time depends on irradiance, buffer, and switching chemistry.
3. An event-camera transition train is the thresholded, pixel-by-pixel response of a spatial PSF.
   Its duration also includes threshold dispersion, spatial sampling, and detector/electronics
   effects, so it is not a direct dye-lifetime measurement.

AF647 kinetics motivate the expected polarity order and provide guards against implausibly long
cycles, but literature values are a soft prior until irradiance, buffer, bias, and detector response
match the recording. Thresholds must ultimately be validated per acquisition condition.

Primary references:

- Lin et al., *PLOS ONE* 10, e0128135 (2015),
  <https://doi.org/10.1371/journal.pone.0128135>.
- Diekmann et al., *Nature Methods* 17, 909--912 (2020),
  <https://doi.org/10.1038/s41592-020-0918-5>.
- van de Linde et al., *Nature Protocols* 6, 991--1009 (2011),
  <https://doi.org/10.1038/nprot.2011.336>.

## Corrected implementation design

### 1. Keep peak detection as the seed generator

Use each retained spatial-temporal peak only to identify a local context. Do not use spline bounds
as blink endpoints. For the first implementation, retain the existing 40 ms non-maximum
suppression because the inspected multi-cycle context retained separate candidate peaks. Revisit
peak suppression only with a labelled missed-cycle study; a shorter suppression window is not a
free accuracy improvement.

Each seed can select zero or one anchored train pair. Nearby retained seeds may initially select
the same physical pair, so duplicate-pair resolution remains an integration requirement before the
feature becomes the default.

### 2. Read bounded context and reject incomplete edges

Read a fixed 250 ms before and 250 ms after each seed. This is search context, not fit support.
Reject candidates whose ROI crosses a spatial sensor boundary or whose full context crosses a
temporal slice boundary. Do not infer a train boundary from truncated data or expand repeatedly.

This deliberately sacrifices candidates near slice edges. Production defaulting requires owned
temporal halos: read neighbouring context, assign a candidate to exactly one slice by seed time,
and deduplicate overlap. Until that ownership rule exists, `context_before_slice` and
`context_after_slice` are correct rejection reasons.

### 3. Detect trains in a compact core with sparse-background statistics

Use two cheap segmentation passes:

1. Detect provisional trains within a 4.25 px discovery core around the integer seed.
2. Compute an event-count-weighted centroid from the provisional ON/OFF pair and redetect within a
   3.5 px core.

The background region is the inscribed radial annulus from the core boundary to
`roi_radius`; the square corners of the rectangular fit ROI are not included. For each polarity
and candidate support, measure:

- raw event count and distinct active pixels;
- polarity purity among all core events in that support;
- exact lattice-pixel-area-corrected core/annulus density ratio;
- root Poisson-deviance score over the shrinkage-stabilized annular expectation; and
- centroid and radial spread.

Use a small background pseudocount because an empty annulus must not imply infinite certainty. Do
not use Gaussian z-scores at the observed sparse expected counts or interpret the deviance gate as
a calibrated hypothesis-test significance.

### 4. Form concentrated transition groups

Sort same-polarity core events by raw timestamp. Split positive events at gaps greater than 3 ms
and negative events at gaps greater than 8 ms. A group becomes a train only if it passes
polarity-specific event-count and active-pixel gates, polarity purity, maximum duration,
core/annulus density ratio, and the Poisson-deviance score.

Initial strict settings supported by the prototype are:

| Setting | Value |
| --- | ---: |
| `temporal_bin_us` | 1,000 |
| `temporal_max_on_interevent_gap_us` | 3,000 |
| `temporal_max_off_interevent_gap_us` | 8,000 |
| `temporal_min_on_events` / `temporal_min_off_events` | 12 / 8 |
| `temporal_min_on_active_pixels` / `temporal_min_off_active_pixels` | 6 / 5 |
| `temporal_min_polarity_purity` | 0.80 |
| `temporal_max_train_duration_us` | 150,000 |
| `temporal_min_core_density_ratio` | 1.5 |
| `temporal_min_interval_deviance` | 2.0 |

The first and last raw events in the accepted group are diagnostic extrema. The fit support is the
containing half-open 1 ms interval `[support_start_us, support_stop_us)`. Store both; never replace
the raw timestamps with bin edges in reported photophysics.

The 3/8 ms rules are intentionally strict concentrated-component definitions. If labelled weak
trains show systematic fragmentation, compare them with significant-bin hysteresis that bridges
only supported gaps. Do not relax the ON gap merely to increase yield: doing so reintroduces the
early events the user-identified boundary excludes.

### 5. Select one seed-anchored ON/OFF pair

Form all provisional positive/negative edges and retain only pairs satisfying:

- the ON train starts no later than the seed and ends no more than 30 ms after it;
- the OFF train ends no earlier than the seed and starts no more than 30 ms before it;
- ON/OFF endpoints overlap by at most 20 ms, which accommodates pixel-level transition overlap in
  the rapid ideal example;
- total first-ON to last-OFF span is at most 300 ms; and
- ON/OFF centroid distance is at most 1.75 px.

Rank admissible pairs first by temporal anchor cost, then centroid distance, then event support.
If two distinct pairs have anchor costs within 5 ms, reject as `ambiguous_pairing`. The observed
20--40 ms onset lead can be reported and later added as a weak condition-specific prior, but it
must not override train evidence.

### 6. Refine timing before the expensive fit

Redetect once around the event-weighted pair centroid and stop. This is a two-pass segmentation,
not two full PSF fits. Run the existing Poisson localization once on the accepted, materialized
ROI. This avoids doubling the dominant nonlinear-fit cost and avoids making segmentation depend on
a fit already contaminated by permissive timing.

The current fixed PSF width and segmentation gates do not provide a validated focus-quality
decision. Train compactness, density, and deviance reject some diffuse evidence, but explicit
focus classification and post-segmentation fit-likelihood/residual validation remain future work.

### 7. Materialize full rectangular, polarity-specific fit supports

The compact core determines whether a transition is real; it does not determine which spatial
pixels the PSF fit may see. After refinement:

- accumulate every positive event in the full rectangular ROI during the ON half-open support;
- accumulate every negative event in the full rectangular ROI during the OFF half-open support;
- exclude the quiet interval from both polarity maps unless it overlaps an explicit support; and
- never discard a fit event only because it lies outside the 3.5 px detection core.

The accepted production ROI record stores ON/OFF support starts and stops, raw first/last event
timestamps, counts, active pixels, centroids, density/deviance scores, pair score, endpoint overlap,
quiet dwell, cycle span, parent seed, and refinement count. Train radial spread and the internal
pair cost remain available only during segmentation/QC and are not production ROI fields.
Downstream temporal summaries use explicit segmented fields when present and retain a clearly
labelled legacy fallback for old artifacts.

### 8. Reject explicitly and conservatively

Stable reasons include `empty_context`, `missing_on_train`, `missing_off_train`,
`no_anchored_pair`, `ambiguous_pairing`, refined-pass variants, sensor/ROI boundary failures,
and slice-context failures. Blinks without a concentrated multi-pixel event train are weak/diffuse
evidence and should be rejected rather than assigned a wider timing window.

## Configuration and rollout

Expose typed, validated `temporal_*` settings in the existing flat pipeline configuration. Keep
`temporal_segmentation_enabled = false` by default while legacy ROI generation remains available.
The earlier shared `temporal_max_interevent_gap_us` key is replaced by the explicit ON and OFF
keys shown above; external configs using the shared key must migrate because unknown keys are
rejected. Do not tune an acquisition profile silently during development. Validation settings must
be recorded with acquisition-condition provenance.

Before enabling the feature by default:

1. implement slice halos and deterministic ownership;
2. resolve duplicate pair selection across nearby retained seeds;
3. label weak/diffuse and crowded cases, not only ideal accepted fits;
4. validate timing and localization bias on held-out recordings; and
5. establish condition-specific settings or demonstrate that one profile generalizes.

## Performance review

The design bounds work per seed to one rectangular ROI over a 500 ms context. Build the per-pixel
time/polarity index once per slice; use binary searches and compiled extraction to gather each
context. The two segmentation passes operate only on gathered events and require four temporal
sorts (two polarities at two centres), followed by one nonlinear localization fit for accepted
candidates.

Primary performance risks are:

- repeatedly gathering overlapping contexts for dense neighbouring seeds;
- sorting the same local events for duplicate seeds;
- increased candidate count if peak suppression is relaxed;
- Python object allocation for train records; and
- QC code scanning an entire long slice when only a few exact sample intervals are requested.

Mitigations, in order, are to keep current peak suppression, reuse the slice event index, read QC
events by exact per-sample intervals, benchmark candidate extraction separately from segmentation
and fitting, and only then consider caching overlapping contexts or compiled grouping. Current QC
may reread overlapping sample intervals; interval merging remains a future optimization. Do not
optimize by spatially cropping fit maps or by removing provenance.

Real-data performance validation must remain subsampled. Report candidates/s, events/s, median and
95th-percentile time per candidate, peak memory, acceptance fraction, and the fraction of runtime
spent in extraction, segmentation, and fitting. Benchmark cold and warm compiled paths separately;
never extrapolate a first-call compilation time to a full run.

The final 3/8 ms implementation was timed on the 15 saved real-data fixtures only. Direct
segmentation took a median 0.425 ms per candidate (95th percentile 0.670 ms) and sustained about
2,782 candidates/s or 1.40 million cropped events/s when warm. Rebuilding the small per-recording
event index, extracting contexts, segmenting, and materializing accepted ROIs together took
1.15 ms/candidate when warm (871 candidates/s). The first recording call in a fresh process took
0.79 s and included one-time compiled-extraction initialization; subsequent recording fixtures took
4.6--7.4 ms. These local exploratory timings do not preserve enough hardware and repetition
metadata for cross-machine comparison. They also exclude nonlinear fitting and do not characterize
peak memory or dense full-slice contention, so they support the subsampled implementation but not a
portable or full-run runtime claim.

## Accuracy validation protocol

The prototype is hypothesis evidence, not threshold certification. Continue as follows:

1. Export the exact 33-candidate manifest, selection metadata, and benchmark method. The current
   plan records seed 7647 and aggregate counts, but this is not yet a locked machine-readable
   validation artifact.
2. Expert-label exact ON/OFF raw-event boundaries, cycle identity, compact/diffuse status, and
   accept/reject status for the named references plus a balanced weak/crowded sample.
3. Split by recording, not by event or blink, into tuning and locked validation sets.
4. Tune in order: gap/group evidence, spatial compactness/background gates, anchored pairing, then
   optional duration priors. This prevents photophysical priors from hiding a poor detector.
5. Weight false merges and background contamination more heavily than conservative weak-blink
   rejection.
6. Freeze profiles before testing an untouched recording.

Primary accuracy metrics are:

- median absolute ON/OFF boundary error in milliseconds;
- event-assignment precision/recall against expert trains;
- false-merge and missed-split rates per 1,000 candidates;
- accepted-blink precision and yield with confidence intervals;
- cycle-span and quiet-dwell distribution shift versus the legacy proxy; and
- fitted-centre displacement and residual change versus expert-cleaned ROIs.

For the current small validation, always report denominators and individual candidates alongside
summaries; do not imply population-level generalization from medians alone.

## Current diagnostics and remaining validation outputs

The three regenerated five-sample QC sets currently provide:

- legacy ROI event maps and raw legacy-window timestamps;
- a bounded detection trace and a +/-250 ms polarity raster, with every plotted trace point and
  context event exported row by row;
- provisional/refined transition-train tables with acceptance metrics;
- the selected interval, pair score, explicit timing arithmetic, and rejection reason;
- full-ROI segmented fit-event tables distinguished from compact-core evidence;
- 1 ms core/annulus activity tables and three-dimensional event views; and
- nearby retained-seed intervals for separating cycle candidates.

Use the shared publication style, physical units, colourblind-safe encoding, honest axes, and
legible single-/double-column typography. Export high-resolution PNG and vector PDF where
practical. Current machine-readable artifacts include legacy ROI events, transition trains, the
selected pair/interval, exact segmentation-context events, raw/interpolated detection curves,
binned context activity, segmented fit events, nearby-seed decisions, timing comparison, and
settings/provenance.

The segmented-to-legacy fit-event count ratio is descriptive, not event-assignment recall: the
segmented ROI can be recentered and can include events outside the legacy time window. Legacy
reconstruction consistency checks likewise verify replay fidelity, not that the legacy temporal
boundary is physically correct.

A locked validation package must additionally export every admissible pair edge with anchor cost
and ambiguity margin, and post-segmentation fit residual or PSF-versus-diffuse evidence. These
outputs remain future work and are required before claiming event-assignment precision/recall or
localization improvement.

## Implementation sequence and gates

1. **Prototype evidence -- complete:** test the train hypothesis on 33 candidates/16,336 events,
   sweep core radius and grouping gap, and retain the documented in-sample corrections above.
2. **Pure segmentation -- complete for feature-gated use:** deterministic train detection,
   sparse-background gates, anchored pairing, ambiguity rejection, centroid refinement, and
   focused synthetic edge cases.
3. **ROI integration -- implemented behind a disabled flag:** per-slice indexed extraction,
   explicit QC records, full rectangular polarity-specific support materialization, and temporal
   provenance through localization.
4. **Photophysics QC -- complete for the current diagnostic scope:** all three five-blink sample
   sets were regenerated from exact per-sample RAW intervals. All 15 legacy timestamp/count replay
   checks pass, and train, selected-pair, timing, fit-event, activity, and nearby-seed artifacts were
   written. Row-level context events and detection curves reproduce the plotted data; all-pair-edge
   and post-refit outputs listed above remain future validation requirements.
5. **Subsampled real-data comparison -- complete for timing selection:** rerun deterministic
   candidates and compare selected train timing with legacy outputs without a full recording.
   Post-segmentation localization refits were not performed, so this is not a localization-accuracy
   result.
6. **Held-out validation and production hardening -- future gate:** expert labels, slice halos,
   duplicate ownership, weak/diffuse assessment, exact validation source artifacts, locked
   recording-level validation, and only then consideration of enabling the feature by default.

## Definition of done for this implementation

- Rapid `blink_05` selects its compact central pair with an ON start 36.214 ms before its seed and
  excludes the later displaced activity.
- 2025 `blink_02` selects an ON start 36.316 ms before its seed and reports raw versus support
  boundaries rather than using remote flank events as endpoints.
- Persisted seeds 269.623 ms before and 358.177 ms after the displayed 2025 `blink_01` seed form
  separate accepted intervals; its weak central seed is explicitly rejected.
- Candidates lacking compact, polarity-ordered trains are rejected with explicit reasons.
- No accepted timing boundary depends on a spline sample index or a permissive context extremum.
- Production ROI materialization retains the full rectangular spatial support within explicit
  ON/OFF time windows; no real-data post-segmentation refit was used for the comparison.
- Across the ten accepted saved samples, median selected cycle span is 142.5685 ms versus a
  523.067 ms median legacy span, with a 74.678% median per-candidate reduction. This demonstrates
  stricter, train-aligned selection on the sample, not ground-truth temporal or localization
  accuracy.
- All 15 exact RAW reconstruction checks pass without a full recording run.
- Nature-style regenerated diagnostics and the current machine-readable train, interval, activity,
  and event tables are written for all 15 reference samples. The additional locked-validation
  source artifacts identified above remain future work.
- The feature stays disabled by default until held-out accuracy, edge ownership, and duplicate
  handling meet the production gates above.
