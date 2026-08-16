from __future__ import annotations

from dataclasses import dataclass, fields

from localization_scripts.pipeline_config import PeakLocConfig


@dataclass(frozen=True)
class SettingSpec:
    name: str
    label: str
    description: str
    group: str
    tier: str = "advanced"
    unit: str = ""
    choices: tuple[str, ...] = ()


WORKFLOW_FIELDS = frozenset(
    {
        "input_folder",
        "input_file",
        "recursive_input",
        "calibration_path",
        "allow_uncalibrated",
    }
)

BASIC_FIELDS = frozenset(
    {
        "slice_start",
        "slice_end",
        "slice_duration",
        "slice_count",
        "num_cores",
        "max_parallel_workers",
        "spatial_mask_enabled",
        "prominence",
        "dataset_fwhm",
        "roi_radius",
        "optical_pixel_size",
        "sigma_psf_px",
        "max_localization_uncertainty_nm",
        "qc_enabled",
        "plot_result",
        "cleanup_temp_outputs",
    }
)


_DETAILS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "slice_start": (
        "Start time",
        "Skip this many microseconds at the beginning. Use 0 for a first run.",
        "µs",
        (),
    ),
    "slice_end": (
        "End time",
        "Optional exclusive end time. Leave blank to continue to the recording end.",
        "µs",
        (),
    ),
    "slice_duration": (
        "Slice duration",
        "Processing chunk length. Start with 10,000,000 µs (10 s) to limit memory use.",
        "µs",
        (),
    ),
    "slice_count": (
        "Maximum slices",
        "Optional number of chunks to process. Leave blank to process the selected range; it is unavailable when an end time is set.",
        "slices",
        (),
    ),
    "num_cores": (
        "CPU cores",
        "Upper CPU count PeakLoc may use. The default reserves one logical CPU.",
        "cores",
        (),
    ),
    "max_parallel_workers": (
        "Parallel workers",
        "Maximum worker processes used inside one slice. The default reserves one logical CPU.",
        "workers",
        (),
    ),
    "max_concurrent_slices": (
        "Concurrent slices",
        "How many slices may overlap. Values above 1 increase memory and disk pressure.",
        "slices",
        (),
    ),
    "cpu_worker_budget": (
        "Global CPU budget",
        "Optional total worker budget shared by concurrent slices. Blank uses the CPU-core limit.",
        "workers",
        (),
    ),
    "max_workers_per_slice": (
        "Workers per slice",
        "Optional hard cap for one slice. Blank uses the parallel-worker limit.",
        "workers",
        (),
    ),
    "memory_reserve_gib": (
        "Memory reserve",
        "PeakLoc waits instead of starting more work when available RAM falls below this reserve.",
        "GiB",
        (),
    ),
    "disk_reserve_gib": (
        "Disk reserve",
        "PeakLoc waits instead of starting more work when free output-disk space is too low.",
        "GiB",
        (),
    ),
    "spatial_mask_enabled": (
        "Limit processing to active regions",
        "Detect illuminated sample regions and ignore most inactive sensor pixels.",
        "",
        (),
    ),
    "spatial_mask_sample_duration_us": (
        "Mask sample duration",
        "Recording duration used to estimate active spatial regions.",
        "µs",
        (),
    ),
    "spatial_mask_min_density_quotient": (
        "Mask density threshold",
        "Minimum event density relative to the sensor-wide mean for mask support.",
        "× mean",
        (),
    ),
    "spatial_mask_min_component_pixels": (
        "Smallest mask region",
        "Connected mask regions smaller than this pixel count are discarded.",
        "pixels",
        (),
    ),
    "spatial_mask_margin_px": (
        "Mask margin",
        "Pixels added around detected active regions to avoid cropping useful signal.",
        "px",
        (),
    ),
    "spatial_mask_max_support_coverage": (
        "Maximum mask coverage",
        "Disable an unhelpful mask if it covers more than this fraction of the sensor.",
        "fraction",
        (),
    ),
    "prominence": (
        "Peak prominence",
        "Minimum strength of a detected peak. Higher values reduce false candidates but may miss dim emitters.",
        "events",
        (),
    ),
    "dataset_fwhm": (
        "Expected PSF width (FWHM)",
        "Expected point-spread-function width used during peak detection.",
        "px",
        (),
    ),
    "peak_time_threshold": (
        "Peak merge time",
        "Candidate peaks closer than this time may be merged.",
        "µs",
        (),
    ),
    "polarity_time_gate_us": (
        "Polarity time gate",
        "Time window used to associate positive and negative event trains.",
        "µs",
        (),
    ),
    "peak_neighbors": (
        "Peak neighbor radius",
        "Spatial neighborhood used to merge duplicate peak candidates.",
        "px",
        (),
    ),
    "peak_min_event_count": (
        "Minimum peak events",
        "Reject peak candidates with fewer associated events.",
        "events",
        (),
    ),
    "roi_radius": (
        "ROI radius",
        "Half-width of the square fit region. The side length is 2 × radius + 1.",
        "px",
        (),
    ),
    "convolution_roi_radius": (
        "Convolution radius",
        "Spatial radius used to combine neighboring event traces before peak detection.",
        "px",
        (),
    ),
    "temporal_segmentation_enabled": (
        "Temporal blink segmentation",
        "Experimental separation of positive and negative blink event trains.",
        "",
        (),
    ),
    "temporal_context_pre_us": (
        "Context before seed",
        "Extra event context loaded before a candidate for temporal segmentation.",
        "µs",
        (),
    ),
    "temporal_context_post_us": (
        "Context after seed",
        "Extra event context loaded after a candidate for temporal segmentation.",
        "µs",
        (),
    ),
    "temporal_discovery_core_radius_px": (
        "Discovery core radius",
        "Large inner spatial radius used to discover temporal event trains.",
        "px",
        (),
    ),
    "temporal_core_radius_px": (
        "Temporal core radius",
        "Inner radius used to distinguish emitter events from local background.",
        "px",
        (),
    ),
    "temporal_bin_us": (
        "Temporal bin width",
        "Time-bin width for temporal segmentation diagnostics.",
        "µs",
        (),
    ),
    "temporal_max_on_interevent_gap_us": (
        "Maximum ON-event gap",
        "Largest positive-event gap still treated as one ON train.",
        "µs",
        (),
    ),
    "temporal_max_off_interevent_gap_us": (
        "Maximum OFF-event gap",
        "Largest negative-event gap still treated as one OFF train.",
        "µs",
        (),
    ),
    "temporal_min_on_events": (
        "Minimum ON events",
        "Minimum positive events required for an ON train.",
        "events",
        (),
    ),
    "temporal_min_off_events": (
        "Minimum OFF events",
        "Minimum negative events required for an OFF train.",
        "events",
        (),
    ),
    "temporal_min_on_active_pixels": (
        "Minimum ON active pixels",
        "Minimum distinct pixels required in an ON train.",
        "pixels",
        (),
    ),
    "temporal_min_off_active_pixels": (
        "Minimum OFF active pixels",
        "Minimum distinct pixels required in an OFF train.",
        "pixels",
        (),
    ),
    "temporal_min_polarity_purity": (
        "Minimum polarity purity",
        "Required fraction of events with the expected polarity in a temporal train.",
        "fraction",
        (),
    ),
    "temporal_max_train_duration_us": (
        "Maximum train duration",
        "Longest accepted duration of one ON or OFF event train.",
        "µs",
        (),
    ),
    "temporal_min_core_density_ratio": (
        "Minimum core density ratio",
        "Required event-density enrichment in the temporal core over background.",
        "ratio",
        (),
    ),
    "temporal_min_interval_deviance": (
        "Minimum interval deviance",
        "Required temporal contrast between the candidate train and its background.",
        "",
        (),
    ),
    "temporal_max_endpoint_overlap_us": (
        "Maximum endpoint overlap",
        "Allowed overlap between paired ON and OFF train endpoints.",
        "µs",
        (),
    ),
    "temporal_max_cycle_span_us": (
        "Maximum blink-cycle span",
        "Longest allowed time from the ON train to its paired OFF train.",
        "µs",
        (),
    ),
    "temporal_max_centroid_distance_px": (
        "Maximum centroid distance",
        "Largest spatial separation accepted between paired ON and OFF trains.",
        "px",
        (),
    ),
    "temporal_max_on_end_after_seed_us": (
        "ON end after seed",
        "Latest accepted ON-train end relative to its detection seed.",
        "µs",
        (),
    ),
    "temporal_max_off_start_before_seed_us": (
        "OFF start before seed",
        "Earliest accepted OFF-train start relative to its detection seed.",
        "µs",
        (),
    ),
    "temporal_ambiguity_margin_us": (
        "Temporal ambiguity margin",
        "Time margin used to flag ambiguous train pairings.",
        "µs",
        (),
    ),
    "temporal_background_pseudocount": (
        "Background pseudocount",
        "Small stabilizing count used in temporal density ratios.",
        "events",
        (),
    ),
    "diffuse_flash_rejection_enabled": (
        "Reject diffuse flashes",
        "Exclude full-sensor light transitions that can create widespread false peaks.",
        "",
        (),
    ),
    "diffuse_flash_bin_duration_us": (
        "Flash bin duration",
        "Time-bin width used to detect diffuse light transitions.",
        "µs",
        (),
    ),
    "diffuse_flash_min_events_per_polarity": (
        "Flash event threshold",
        "Minimum events of each polarity required to label a diffuse transition.",
        "events",
        (),
    ),
    "diffuse_flash_min_active_pixel_fraction": (
        "Flash active-pixel fraction",
        "Minimum sensor fraction active during a diffuse transition.",
        "fraction",
        (),
    ),
    "diffuse_flash_max_gap_us": (
        "Maximum flash gap",
        "Nearby diffuse intervals closer than this are merged.",
        "µs",
        (),
    ),
    "diffuse_flash_padding_us": (
        "Flash exclusion padding",
        "Extra time excluded before and after each diffuse transition.",
        "µs",
        (),
    ),
    "interpolation_coefficient": (
        "Interpolation factor",
        "Temporal interpolation density used during peak detection.",
        "",
        (),
    ),
    "spline_smooth": (
        "Spline smoothing",
        "Smoothing strength for interpolated event traces, from 0 to 1.",
        "fraction",
        (),
    ),
    "plot_subplotsize": (
        "Diagnostic panel size",
        "Base size of legacy diagnostic plot panels.",
        "in",
        (),
    ),
    "plot_result": (
        "Render localization image",
        "Create the final SMLM-style reconstruction after localization.",
        "",
        (),
    ),
    "optical_pixel_size": (
        "Optical pixel size",
        "Sample-plane size represented by one sensor pixel. Must match calibration metadata.",
        "nm/px",
        (),
    ),
    "sensor_height": (
        "Sensor height",
        "Event-camera sensor height used for calibration and bounds checks.",
        "px",
        (),
    ),
    "sensor_width": (
        "Sensor width",
        "Event-camera sensor width used for calibration and bounds checks.",
        "px",
        (),
    ),
    "max_raw_events": (
        "Raw read buffer",
        "OpenEB read-buffer size. Increase if decoding reports that its buffer is too small.",
        "events",
        (),
    ),
    "cleanup_temp_outputs": (
        "Remove temporary arrays",
        "Delete large per-slice intermediates after final outputs are assembled.",
        "",
        (),
    ),
    "fit_model": (
        "Fit model",
        "Production event-count likelihood model. Only poisson_joint is supported.",
        "",
        ("poisson_joint",),
    ),
    "sigma_psf_px": (
        "Fixed PSF sigma",
        "Calibrated Gaussian sigma used by the localization fit. Leave blank only for unsupported workflows.",
        "px",
        (),
    ),
    "fit_sigma": (
        "Fit sigma per emitter",
        "Experimental option; fixed calibrated sigma is the production path.",
        "",
        (),
    ),
    "psf_model": (
        "PSF model",
        "Spatial model integrated over sensor pixels.",
        "",
        ("pixel_integrated_gaussian",),
    ),
    "background_mode": (
        "Background model",
        "Use calibration maps, a local fitted background, or both.",
        "",
        ("calibrated_plus_local", "calibrated_only", "local_only"),
    ),
    "hot_pixel_policy": (
        "Hot-pixel policy",
        "Treatment of pixels marked hot by calibration. Only masking is supported.",
        "",
        ("mask",),
    ),
    "min_events_pos": (
        "Minimum positive events",
        "Minimum positive-polarity count required before fitting an ROI.",
        "events",
        (),
    ),
    "min_events_neg": (
        "Minimum negative events",
        "Minimum negative-polarity count required before fitting an ROI.",
        "events",
        (),
    ),
    "min_valid_pixels": (
        "Minimum valid ROI pixels",
        "Minimum unmasked pixels required in a fit region.",
        "pixels",
        (),
    ),
    "max_fit_cond": (
        "Maximum fit condition",
        "Reject numerically ill-conditioned fits above this condition estimate.",
        "",
        (),
    ),
    "max_fit_center_offset_px": (
        "Maximum center offset",
        "Reject fits whose center moves too far from the detected peak. Blank disables this filter.",
        "px",
        (),
    ),
    "max_localization_uncertainty_px": (
        "Maximum uncertainty (pixels)",
        "Optional pixel-unit uncertainty cutoff. Blank disables this cutoff.",
        "px",
        (),
    ),
    "max_localization_uncertainty_nm": (
        "Maximum uncertainty",
        "Reject localizations with estimated uncertainty above this value. Blank disables it.",
        "nm",
        (),
    ),
    "qc_enabled": (
        "Generate quality-control output",
        "Create diagnostic tables, figures, and reports for reviewing the run.",
        "",
        (),
    ),
    "qc_output_dirname": (
        "QC folder name",
        "Name of the quality-control subdirectory inside debug output.",
        "",
        (),
    ),
    "qc_static_dpi": (
        "QC raster resolution",
        "Resolution of saved PNG quality-control figures.",
        "dpi",
        (),
    ),
    "qc_save_vector": (
        "Save optional SVG QC figures",
        "Also save SVG versions of applicable QC figures; disabled by default.",
        "",
        (),
    ),
    "qc_max_events_for_interactive": (
        "Interactive event limit",
        "Maximum sampled events included in interactive QC views.",
        "events",
        (),
    ),
    "qc_uncertainty_montage_n": (
        "Uncertainty montage size",
        "Number of fits shown in uncertainty review montages.",
        "fits",
        (),
    ),
    "qc_generate_html": (
        "Generate HTML dashboard",
        "Create a browser-readable quality-control dashboard.",
        "",
        (),
    ),
    "qc_generate_interactive": (
        "Generate interactive plots",
        "Create additional interactive plots; these can be slower and larger.",
        "",
        (),
    ),
    "qc_generate_temporal_3d": (
        "Generate temporal 3D plot",
        "Create an optional three-dimensional temporal event view.",
        "",
        (),
    ),
    "qc_keep_intermediates": (
        "Keep QC intermediates",
        "Retain intermediate QC data for debugging at the cost of disk space.",
        "",
        (),
    ),
}


def _group_for(name: str) -> str:
    if name.startswith("temporal_"):
        return "Temporal segmentation"
    if name.startswith("spatial_mask_"):
        return "Spatial mask"
    if name.startswith("diffuse_flash_"):
        return "Diffuse flash rejection"
    if name.startswith("qc_"):
        return "Quality control"
    if name in {
        "num_cores",
        "max_parallel_workers",
        "max_concurrent_slices",
        "cpu_worker_budget",
        "max_workers_per_slice",
        "memory_reserve_gib",
        "disk_reserve_gib",
        "max_raw_events",
    }:
        return "Resources"
    if name.startswith("slice_"):
        return "Processing range"
    if name in {
        "prominence",
        "dataset_fwhm",
        "peak_time_threshold",
        "polarity_time_gate_us",
        "peak_neighbors",
        "peak_min_event_count",
        "interpolation_coefficient",
        "spline_smooth",
    }:
        return "Peak detection"
    if name in {"roi_radius", "convolution_roi_radius"}:
        return "Regions of interest"
    if name in {"optical_pixel_size", "sensor_height", "sensor_width"}:
        return "Optics and sensor"
    if name in {"plot_result", "plot_subplotsize", "cleanup_temp_outputs"}:
        return "Outputs"
    return "Localization fit"


def _build_catalog() -> tuple[SettingSpec, ...]:
    config_fields = {field.name for field in fields(PeakLocConfig)}
    missing_details = sorted(config_fields - WORKFLOW_FIELDS - set(_DETAILS))
    if missing_details:
        raise RuntimeError("Missing GUI setting help: " + ", ".join(missing_details))

    specs = []
    for field in fields(PeakLocConfig):
        if field.name in WORKFLOW_FIELDS:
            continue
        label, description, unit, choices = _DETAILS[field.name]
        specs.append(
            SettingSpec(
                name=field.name,
                label=label,
                description=description,
                group=_group_for(field.name),
                tier="basic" if field.name in BASIC_FIELDS else "advanced",
                unit=unit,
                choices=choices,
            )
        )
    return tuple(specs)


SETTINGS = _build_catalog()


def settings_for_tier(tier: str) -> tuple[SettingSpec, ...]:
    return tuple(spec for spec in SETTINGS if spec.tier == tier)
