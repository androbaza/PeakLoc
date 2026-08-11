"""Interactive review tool for normalized spatial-mask configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

from localization_scripts.event_array_processing import (
    RAW_READ_DURATION_US,
    temporary_openeb_system_site_packages,
)
from localization_scripts.pipeline_config import PeakLocConfig, load_peakloc_config
from localization_scripts.spatial_mask import (
    SpatialMaskPreview,
    accumulate_event_density_in_time_window,
    preview_spatial_mask,
)

RAW_SUFFIX = ".raw"
NPY_SUFFIX = ".npy"
TUNER_DENSITY_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class SpatialMaskSample:
    """One bounded calibration density image for interactive parameter review."""

    density: np.ndarray
    event_count: int
    start_us: int
    stop_us: int

    @property
    def duration_us(self) -> int:
        return self.stop_us - self.start_us


class SpatialMaskTuner:
    """Matplotlib control panel for previewing a recording's spatial mask."""

    def __init__(self, recording_path: Path, config: PeakLocConfig) -> None:
        self.recording_path = recording_path
        self.config = config
        self.sample_duration_us = config.spatial_mask_sample_duration_us
        self.sample = load_spatial_mask_sample(
            recording_path,
            config,
            duration_us=self.sample_duration_us,
        )
        self.display_density = _display_density(self.sample.density)
        self.figure, (self.sum_axis, self.overlay_axis) = plt.subplots(
            1,
            2,
            figsize=(14, 8),
            sharex=True,
            sharey=True,
        )
        self.figure.subplots_adjust(left=0.08, right=0.97, bottom=0.47, top=0.88)
        self.status_text = self.figure.text(0.08, 0.435, "", va="top", fontsize=9)
        self.duration_box = TextBox(
            self.figure.add_axes((0.08, 0.06, 0.13, 0.04)),
            "Sample s",
            initial=f"{self.sample_duration_us / 1e6:g}",
        )
        self.reload_button = Button(
            self.figure.add_axes((0.24, 0.06, 0.14, 0.04)),
            "Reload sample",
        )
        self.render_button = Button(
            self.figure.add_axes((0.41, 0.06, 0.2, 0.04)),
            "Render sum + mask",
        )
        self.quotient_slider = Slider(
            self.figure.add_axes((0.25, 0.33, 0.65, 0.025)),
            "Density quotient",
            0.1,
            max(20.0, config.spatial_mask_min_density_quotient * 2),
            valinit=config.spatial_mask_min_density_quotient,
            valstep=0.1,
        )
        self.component_slider = Slider(
            self.figure.add_axes((0.25, 0.28, 0.65, 0.025)),
            "Min seed pixels",
            1,
            max(1_000, config.spatial_mask_min_component_pixels * 2),
            valinit=config.spatial_mask_min_component_pixels,
            valstep=1,
        )
        self.margin_slider = Slider(
            self.figure.add_axes((0.25, 0.23, 0.65, 0.025)),
            "Target margin px",
            0,
            max(100, config.spatial_mask_margin_px * 2),
            valinit=config.spatial_mask_margin_px,
            valstep=1,
        )
        self.coverage_slider = Slider(
            self.figure.add_axes((0.25, 0.18, 0.65, 0.025)),
            "Max support coverage",
            0.05,
            1.0,
            valinit=config.spatial_mask_max_support_coverage,
            valstep=0.01,
        )
        self.reload_button.on_clicked(self._reload_sample)
        self.render_button.on_clicked(self._render)
        self._render(None)

    def show(self) -> None:
        """Show the tuner window until the user closes it."""
        plt.show()

    def _reload_sample(self, _event: Any) -> None:
        try:
            duration_seconds = float(self.duration_box.text)
            if duration_seconds <= 0:
                raise ValueError("Sample duration must be positive")
            self.sample_duration_us = int(round(duration_seconds * 1e6))
            self.sample = load_spatial_mask_sample(
                self.recording_path,
                self.config,
                duration_us=self.sample_duration_us,
            )
            self.display_density = _display_density(self.sample.density)
            self._render(None)
        except (OSError, ValueError) as error:
            self.status_text.set_text(f"Could not load sample: {error}")
            self.figure.canvas.draw_idle()

    def _render(self, _event: Any) -> None:
        """Render the current density image and morphology-expanded mask preview."""
        preview = preview_spatial_mask(
            self.sample.density,
            calibration_event_count=self.sample.event_count,
            min_density_quotient=float(self.quotient_slider.val),
            min_component_pixels=int(self.component_slider.val),
            margin_px=int(self.margin_slider.val),
            support_margin_px=max(
                self.config.roi_radius,
                self.config.convolution_roi_radius,
            ),
            max_support_coverage=float(self.coverage_slider.val),
        )
        self._draw_density_sum()
        self._draw_overlay(preview)
        self.status_text.set_text(self._status_text(preview))
        print(self._config_snippet())
        self.figure.canvas.draw_idle()

    def _draw_density_sum(self) -> None:
        self.sum_axis.clear()
        self.sum_axis.imshow(
            self.display_density, cmap="magma", interpolation="nearest"
        )
        self.sum_axis.set_title("Calibration event-density sum (log1p)")
        self.sum_axis.set(xlabel="Sensor x", ylabel="Sensor y")

    def _draw_overlay(self, preview: SpatialMaskPreview) -> None:
        self.overlay_axis.clear()
        self.overlay_axis.imshow(
            self.display_density,
            cmap="magma",
            interpolation="nearest",
        )
        if np.any(preview.seed_mask):
            self.overlay_axis.contour(
                preview.seed_mask,
                levels=[0.5],
                colors=["#ffdd00"],
                linewidths=0.5,
            )
        if np.any(preview.retained_seed_mask):
            self.overlay_axis.contour(
                preview.retained_seed_mask,
                levels=[0.5],
                colors=["#ff7f0e"],
                linewidths=0.8,
            )
        if preview.support_mask is not None:
            self.overlay_axis.contour(
                preview.support_mask,
                levels=[0.5],
                colors=["#00d5ff"],
                linewidths=0.8,
            )
        if preview.target_mask is not None:
            self.overlay_axis.contour(
                preview.target_mask,
                levels=[0.5],
                colors=["#00ff86"],
                linewidths=1.0,
            )
        self.overlay_axis.set_title(
            "Mask overlay: seed yellow, retained orange, support cyan, target green"
        )
        self.overlay_axis.set(xlabel="Sensor x", ylabel="Sensor y")

    def _status_text(self, preview: SpatialMaskPreview) -> str:
        target_coverage = _coverage(preview.target_mask)
        support_coverage = _coverage(preview.support_mask)
        support_margin_px = max(
            self.config.roi_radius,
            self.config.convolution_roi_radius,
        )
        lines = [
            f"Sample: {self.sample.duration_us / 1e6:.3g} s, "
            f"{self.sample.event_count:,} in-bounds events.",
            f"Mean density: {preview.mean_events_per_pixel:.3g} events/pixel; "
            f"seed threshold: {preview.seed_threshold_events:.3g} events/pixel.",
            f"Seed pixels: {preview.seed_pixel_count:,}; retained seed pixels: "
            f"{preview.retained_seed_pixel_count:,}; target: {_format_coverage(target_coverage)}; "
            f"support: {_format_coverage(support_coverage)}.",
            f"Support halo: {support_margin_px} px (max ROI/convolution radius).",
        ]
        if preview.fallback_reason is not None:
            lines.append(f"Full-sensor fallback: {preview.fallback_reason}")
        else:
            lines.append(
                "Preview is safe to use. The JSON snippet was printed to the terminal."
            )
        return "\n".join(lines)

    def _config_snippet(self) -> str:
        return json.dumps(
            {
                "spatial_mask_enabled": True,
                "spatial_mask_sample_duration_us": self.sample_duration_us,
                "spatial_mask_min_density_quotient": round(
                    float(self.quotient_slider.val),
                    4,
                ),
                "spatial_mask_min_component_pixels": int(self.component_slider.val),
                "spatial_mask_margin_px": int(self.margin_slider.val),
                "spatial_mask_max_support_coverage": round(
                    float(self.coverage_slider.val),
                    4,
                ),
            },
            indent=2,
        )


def load_spatial_mask_sample(
    recording_path: Path,
    config: PeakLocConfig,
    *,
    duration_us: int,
) -> SpatialMaskSample:
    """Load only one calibration interval into a native-resolution density image."""
    if duration_us <= 0:
        raise ValueError("Sample duration must be positive")
    if not recording_path.is_file():
        raise FileNotFoundError(f"Recording does not exist: {recording_path}")
    if recording_path.suffix.lower() == RAW_SUFFIX:
        return _load_raw_spatial_mask_sample(recording_path, config, duration_us)
    if recording_path.suffix.lower() == NPY_SUFFIX:
        return _load_npy_spatial_mask_sample(recording_path, config, duration_us)
    raise ValueError(
        "Spatial-mask tuner accepts .raw or normalized event .npy recordings"
    )


def main(argv: list[str] | None = None) -> None:
    """Start the desktop spatial-mask tuner."""
    args = _parse_args(argv)
    config = load_peakloc_config(args.config)
    recording_path = _resolve_recording_path(args.recording, config)
    tuner = SpatialMaskTuner(recording_path, config)
    tuner.show()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively preview PeakLoc spatial-mask parameters"
    )
    parser.add_argument(
        "recording",
        type=Path,
        nargs="?",
        help="RAW or normalized event NPY recording; inferred when config has one RAW file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="PeakLoc JSON configuration path (defaults to config.json)",
    )
    return parser.parse_args(argv)


def _resolve_recording_path(
    requested_path: Path | None,
    config: PeakLocConfig,
) -> Path:
    if requested_path is not None:
        return requested_path
    input_folder = Path(config.input_folder)
    recordings = sorted(input_folder.glob(f"*{RAW_SUFFIX}"))
    if len(recordings) == 1:
        return recordings[0]
    if not recordings:
        raise FileNotFoundError(
            f"No RAW recording found in configured input folder: {input_folder}"
        )
    raise ValueError(
        "Configured input folder contains multiple RAW recordings; pass the recording path."
    )


def _load_npy_spatial_mask_sample(
    recording_path: Path,
    config: PeakLocConfig,
    duration_us: int,
) -> SpatialMaskSample:
    events = np.load(recording_path, mmap_mode="r", allow_pickle=False)
    try:
        _require_event_fields(events)
        if events.size == 0:
            raise ValueError("Recording contains no events")
        timestamps = events["t"]
        timestamps_monotonic = _timestamps_are_monotonic(timestamps)
        sample_start = (
            int(timestamps[0]) if timestamps_monotonic else int(np.min(timestamps))
        )
        sample_end = (
            int(timestamps[-1]) if timestamps_monotonic else int(np.max(timestamps))
        )
        sample_stop = min(sample_start + duration_us, sample_end + 1)
        density, event_count = accumulate_event_density_in_time_window(
            events,
            config.sensor_shape,
            start_us=sample_start,
            stop_us=sample_stop,
            timestamps_monotonic=timestamps_monotonic,
        )
        return SpatialMaskSample(
            density=density,
            event_count=event_count,
            start_us=sample_start,
            stop_us=sample_stop,
        )
    finally:
        _close_memory_map(events)


def _load_raw_spatial_mask_sample(
    recording_path: Path,
    config: PeakLocConfig,
    duration_us: int,
) -> SpatialMaskSample:
    density = np.zeros(config.sensor_shape, dtype=np.uint32)
    sample_start: int | None = None
    requested_stop: int | None = None
    last_timestamp: int | None = None
    event_count = 0

    with temporary_openeb_system_site_packages():
        RawReader = import_module("metavision_core.event_io.raw_reader").RawReader
        reader = RawReader(str(recording_path), max_events=config.max_raw_events)
        while not reader.is_done():
            try:
                events = reader.load_delta_t(RAW_READ_DURATION_US)
            except ValueError as error:
                if "buffer size too small" not in str(error):
                    raise
                raise ValueError(
                    "RAW reader buffer is too small for this recording. Increase "
                    "max_raw_events before using the spatial-mask tuner."
                ) from error
            if events.size == 0:
                continue
            timestamps = events["t"]
            if sample_start is None:
                sample_start = int(timestamps[0])
                requested_stop = sample_start + duration_us
            if requested_stop is None:
                raise RuntimeError("Spatial-mask sample has no stop time")
            event_count += _accumulate_raw_chunk(
                density,
                events,
                config.sensor_shape,
                start_us=sample_start,
                stop_us=requested_stop,
            )
            last_timestamp = int(timestamps[-1])
            if last_timestamp >= requested_stop:
                break

    if sample_start is None or last_timestamp is None or requested_stop is None:
        raise ValueError("RAW recording contains no events")
    return SpatialMaskSample(
        density=density,
        event_count=event_count,
        start_us=sample_start,
        stop_us=min(requested_stop, last_timestamp + 1),
    )


def _accumulate_raw_chunk(
    density: np.ndarray,
    events: np.ndarray,
    sensor_shape: tuple[int, int],
    *,
    start_us: int,
    stop_us: int,
) -> int:
    """Accumulate a decoder chunk with at most one million-element temporary masks."""
    event_count = 0
    height, width = sensor_shape
    for start in range(0, events.size, TUNER_DENSITY_CHUNK_SIZE):
        chunk = events[start : start + TUNER_DENSITY_CHUNK_SIZE]
        x = chunk["x"]
        y = chunk["y"]
        timestamps = chunk["t"]
        valid = (
            (timestamps >= start_us)
            & (timestamps < stop_us)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        np.add.at(density, (y[valid], x[valid]), 1)
        event_count += int(np.count_nonzero(valid))
    return event_count


def _timestamps_are_monotonic(timestamps: np.ndarray) -> bool:
    if timestamps.size < 2:
        return True
    previous = int(timestamps[0])
    for start in range(0, timestamps.size, TUNER_DENSITY_CHUNK_SIZE):
        chunk = timestamps[start : start + TUNER_DENSITY_CHUNK_SIZE]
        if chunk.size == 0:
            continue
        if int(chunk[0]) < previous or np.any(chunk[1:] < chunk[:-1]):
            return False
        previous = int(chunk[-1])
    return True


def _require_event_fields(events: np.ndarray) -> None:
    names = events.dtype.names or ()
    required = {"x", "y", "t"}
    if not required.issubset(names):
        raise ValueError("Event array must provide x, y, and t fields")


def _close_memory_map(array: np.ndarray) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def _coverage(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    return float(np.mean(mask))


def _format_coverage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _display_density(density: np.ndarray) -> np.ndarray:
    """Return the compact log-density image shared by both preview panels."""
    return np.log1p(density.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()
