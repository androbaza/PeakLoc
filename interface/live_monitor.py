"""Failure-isolated Tk/Matplotlib view of optional pipeline progress snapshots."""

from __future__ import annotations

import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from localization_scripts.live_progress import LIVE_PROGRESS_SCHEMA_VERSION
from localization_scripts.plot_style import (
    EVENT_DENSITY_CMAP,
    PLOT_COLORS,
    style_publication_axis,
)

POLL_INTERVAL_MS = 500
DISPLAY_SAMPLE_LIMIT = 48
STATE_COLORS = {
    "pending": "#B8C2CC",
    "active": PLOT_COLORS["orange"],
    "completed": PLOT_COLORS["green"],
    "skipped": "#E5E7EB",
    "failed": PLOT_COLORS["vermillion"],
}
REQUIRED_SNAPSHOT_KEYS = {
    "schema_version",
    "slice_start_us",
    "slice_stop_us",
    "localization_image",
}


@dataclass(frozen=True)
class SliceWindow:
    start_us: int
    stop_us: int


@dataclass(frozen=True)
class ProgressManifest:
    recording_id: str
    recording_name: str
    recording_start_us: int
    recording_stop_us: int
    selected_start_us: int
    selected_stop_us: int
    sensor_height: int
    sensor_width: int
    slices: list[SliceWindow]
    state: str
    error: str | None = None


def load_progress_manifest(path: Path) -> ProgressManifest:
    """Load and validate one atomically published recording manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("progress manifest must be a JSON object")
    if int(payload.get("schema_version", -1)) != LIVE_PROGRESS_SCHEMA_VERSION:
        raise ValueError("unsupported live-progress schema")
    raw_slices = payload.get("slices")
    if not isinstance(raw_slices, list):
        raise TypeError("progress manifest slices must be a list")
    slices = [
        SliceWindow(int(item["start_us"]), int(item["stop_us"]))
        for item in raw_slices
        if isinstance(item, dict)
    ]
    manifest = ProgressManifest(
        recording_id=str(payload["recording_id"]),
        recording_name=str(payload["recording_name"]),
        recording_start_us=int(payload["recording_start_us"]),
        recording_stop_us=int(payload["recording_stop_us"]),
        selected_start_us=int(payload["selected_start_us"]),
        selected_stop_us=int(payload["selected_stop_us"]),
        sensor_height=int(payload["sensor_height"]),
        sensor_width=int(payload["sensor_width"]),
        slices=slices,
        state=str(payload.get("state", "running")),
        error=str(payload["error"]) if payload.get("error") else None,
    )
    if manifest.recording_stop_us < manifest.recording_start_us:
        raise ValueError("recording progress has inverted time bounds")
    if manifest.sensor_height <= 0 or manifest.sensor_width <= 0:
        raise ValueError("recording progress has invalid sensor dimensions")
    if manifest.sensor_height > 20_000 or manifest.sensor_width > 20_000:
        raise ValueError("recording progress sensor dimensions are unreasonably large")
    if manifest.sensor_height * manifest.sensor_width > 20_000_000:
        raise ValueError("recording progress sensor image is unreasonably large")
    if any(window.stop_us <= window.start_us for window in manifest.slices):
        raise ValueError("recording progress contains an invalid slice window")
    return manifest


def load_progress_snapshot(path: Path) -> dict[str, np.ndarray]:
    """Load an allow-pickle-free snapshot into independent arrays."""
    with np.load(path, allow_pickle=False) as snapshot:
        missing = REQUIRED_SNAPSHOT_KEYS.difference(snapshot.files)
        if missing:
            raise ValueError(f"progress snapshot is missing {sorted(missing)}")
        schema_version = int(np.asarray(snapshot["schema_version"]).reshape(-1)[0])
        if schema_version != LIVE_PROGRESS_SCHEMA_VERSION:
            raise ValueError("unsupported live-progress snapshot schema")
        return {name: np.asarray(snapshot[name]).copy() for name in snapshot.files}


class LiveProgressMonitor(ttk.Frame):
    """Read-only progress monitor whose failures remain inside the GUI tab."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="Page.TFrame", padding=(20, 14))
        self.progress_directory: Path | None = None
        self.recording_directory: Path | None = None
        self.manifest: ProgressManifest | None = None
        self.slice_states: dict[int, str] = {}
        self.loaded_snapshots: set[Path] = set()
        self.snapshot_errors: dict[Path, int] = {}
        self.localization_image: np.ndarray | None = None
        self.localization_count = 0
        self.peak_samples: list[dict[str, Any]] = []
        self.roi_samples: list[dict[str, Any]] = []
        self.process_finished = False
        self.process_succeeded: bool | None = None
        self.monitor_warning: str | None = None
        self._updating_roi_controls = False
        self._poll_after_id: str | None = None

        self.status_text = tk.StringVar(
            value="Start processing to populate live measurement progress."
        )
        self.peak_selection = tk.StringVar()
        self.roi_selection = tk.StringVar()
        self.roi_start = tk.DoubleVar(value=0.0)
        self.roi_stop = tk.DoubleVar(value=1.0)
        self.roi_window_text = tk.StringVar(value="No sampled blink selected")

        self._build_layout()
        self._render_all()
        self._schedule_poll()

    def _build_layout(self) -> None:
        heading = ttk.Frame(self, style="Page.TFrame")
        heading.pack(fill="x")
        ttk.Label(heading, text="Live measurement", style="Title.TLabel").pack(
            side="left"
        )
        ttk.Label(
            heading,
            textvariable=self.status_text,
            style="Muted.TLabel",
            wraplength=760,
            justify="right",
        ).pack(side="right", padx=(16, 0))

        controls = ttk.Frame(self, style="Surface.TFrame", padding=(12, 8))
        controls.pack(fill="x", pady=(10, 6))
        ttk.Label(controls, text="Peak trace", style="Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.peak_selector = ttk.Combobox(
            controls, textvariable=self.peak_selection, state="readonly", width=33
        )
        self.peak_selector.grid(row=0, column=1, sticky="ew", padx=(8, 18))
        self.peak_selector.bind("<<ComboboxSelected>>", self._selection_changed)
        ttk.Label(controls, text="Blink ROI", style="Card.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        self.roi_selector = ttk.Combobox(
            controls, textvariable=self.roi_selection, state="readonly", width=33
        )
        self.roi_selector.grid(row=0, column=3, sticky="ew", padx=(8, 0))
        self.roi_selector.bind("<<ComboboxSelected>>", self._roi_selection_changed)

        ttk.Label(controls, text="First event", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(7, 0)
        )
        self.roi_start_slider = ttk.Scale(
            controls, variable=self.roi_start, command=self._roi_window_changed
        )
        self.roi_start_slider.grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(7, 0)
        )
        ttk.Label(controls, text="Last event", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )
        self.roi_stop_slider = ttk.Scale(
            controls, variable=self.roi_stop, command=self._roi_window_changed
        )
        self.roi_stop_slider.grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(4, 0)
        )
        ttk.Label(
            controls, textvariable=self.roi_window_text, style="CardMuted.TLabel"
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(5, 0))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        figure_card = ttk.Frame(self, style="Surface.TFrame", padding=6)
        figure_card.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(11, 5.0), dpi=100, constrained_layout=True)
        grid = self.figure.add_gridspec(
            2, 4, height_ratios=(0.8, 3.2), width_ratios=(1.35, 1.0, 1.0, 1.0)
        )
        self.timeline_axis = self.figure.add_subplot(grid[0, :])
        self.localization_axis = self.figure.add_subplot(grid[1, 0])
        self.peak_axis = self.figure.add_subplot(grid[1, 1])
        self.roi_axis = self.figure.add_subplot(grid[1, 2])
        self.temporal_axis = self.figure.add_subplot(grid[1, 3])
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_card)
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, figure_card, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas.mpl_connect("button_press_event", self._localization_clicked)

    def watch(self, progress_directory: Path) -> None:
        """Start observing a new unique directory and discard the previous view."""
        self.progress_directory = progress_directory
        self.process_finished = False
        self.process_succeeded = None
        self._reset_recording()
        self.status_text.set(
            "Waiting for the measurement worker to publish recording bounds…"
        )
        self._render_all()

    def mark_process_finished(self, succeeded: bool) -> None:
        self.process_finished = True
        self.process_succeeded = succeeded
        self._update_status()
        self._render_timeline()
        self.canvas.draw_idle()

    def close(self) -> None:
        if self._poll_after_id is not None:
            self.after_cancel(self._poll_after_id)
            self._poll_after_id = None

    def _schedule_poll(self) -> None:
        if self.winfo_exists():
            self._poll_after_id = self.after(POLL_INTERVAL_MS, self._poll_once)

    def _poll_once(self) -> None:
        try:
            if self.progress_directory is not None:
                self._read_current_recording()
        except Exception as error:  # noqa: BLE001 - the monitor must isolate all failures
            self.status_text.set(
                f"Live monitor warning: {error}. Processing is unaffected."
            )
        finally:
            self._schedule_poll()

    def _read_current_recording(self) -> None:
        if self.progress_directory is None:
            return
        current_path = self.progress_directory / "current.json"
        if not current_path.is_file():
            return
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise TypeError("current progress pointer is malformed")
        recording_id = str(current.get("recording_id", ""))
        if not recording_id or recording_id != Path(recording_id).name:
            raise ValueError("current progress pointer has no recording id")
        recording_directory = self.progress_directory / "recordings" / recording_id
        manifest = load_progress_manifest(recording_directory / "manifest.json")
        recording_changed = (
            self.manifest is None or manifest.recording_id != self.manifest.recording_id
        )
        manifest_changed = manifest != self.manifest
        if recording_changed:
            self._reset_recording()
            self.recording_directory = recording_directory
            self.manifest = manifest
            self.localization_image = np.zeros(
                (manifest.sensor_height, manifest.sensor_width), dtype=np.uint64
            )
        else:
            self.manifest = manifest
        changed = (
            manifest_changed | self._read_slice_states() | self._read_new_snapshots()
        )
        self._update_status()
        if changed:
            self._update_selectors()
            self._render_all()

    def _read_slice_states(self) -> bool:
        if self.recording_directory is None:
            return False
        states = {}
        for path in sorted((self.recording_directory / "states").glob("slice_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                states[int(payload["slice_stop_us"])] = str(payload["state"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
        changed = states != self.slice_states
        self.slice_states = states
        return changed

    def _read_new_snapshots(self) -> bool:
        if self.recording_directory is None:
            return False
        changed = False
        for path in sorted(
            (self.recording_directory / "snapshots").glob("slice_*.npz")
        ):
            if path in self.loaded_snapshots:
                continue
            modified_ns = path.stat().st_mtime_ns
            if self.snapshot_errors.get(path) == modified_ns:
                continue
            try:
                snapshot = load_progress_snapshot(path)
                self._merge_snapshot(snapshot)
            except (OSError, TypeError, ValueError, KeyError) as error:
                self.snapshot_errors[path] = modified_ns
                self.monitor_warning = (
                    f"Ignored malformed monitor snapshot {path.name}: {error}"
                )
                continue
            self.loaded_snapshots.add(path)
            self.snapshot_errors.pop(path, None)
            changed = True
        return changed

    def _merge_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        if self.localization_image is None:
            raise ValueError("snapshot arrived before recording dimensions")
        image = np.asarray(snapshot["localization_image"], dtype=np.uint64)
        if image.shape != self.localization_image.shape:
            raise ValueError(
                f"localization image shape {image.shape} does not match "
                f"{self.localization_image.shape}"
            )
        slice_stop = int(np.asarray(snapshot["slice_stop_us"]).reshape(-1)[0])
        peak_count = len(self.peak_samples)
        roi_count = len(self.roi_samples)
        try:
            self._merge_peak_samples(snapshot, slice_stop)
            self._merge_roi_samples(snapshot, slice_stop)
        except (TypeError, ValueError, KeyError, IndexError):
            del self.peak_samples[peak_count:]
            del self.roi_samples[roi_count:]
            raise
        self.localization_image += image
        self.localization_count += int(image.sum())

    def _merge_peak_samples(
        self, snapshot: dict[str, np.ndarray], slice_stop: int
    ) -> None:
        sample_count = len(snapshot.get("peak_y", ()))
        required = (
            "peak_x",
            "peak_time_us",
            "peak_on_us",
            "peak_off_us",
            "peak_prominence",
            "peak_trace_time_us",
            "peak_trace_cumsum",
        )
        if sample_count and any(name not in snapshot for name in required):
            raise ValueError("peak sample arrays are incomplete")
        for index in range(sample_count):
            if len(self.peak_samples) >= DISPLAY_SAMPLE_LIMIT:
                break
            self.peak_samples.append(
                {
                    "slice_stop": slice_stop,
                    "y": int(snapshot["peak_y"][index]),
                    "x": int(snapshot["peak_x"][index]),
                    "peak_time": float(snapshot["peak_time_us"][index]),
                    "on_time": float(snapshot["peak_on_us"][index]),
                    "off_time": float(snapshot["peak_off_us"][index]),
                    "prominence": float(snapshot["peak_prominence"][index]),
                    "times": snapshot["peak_trace_time_us"][index],
                    "values": snapshot["peak_trace_cumsum"][index],
                }
            )

    def _merge_roi_samples(
        self, snapshot: dict[str, np.ndarray], slice_stop: int
    ) -> None:
        sample_count = len(snapshot.get("roi_center_y", ()))
        if not sample_count:
            return
        required = (
            "roi_center_x",
            "roi_t_first_us",
            "roi_t_peak_us",
            "roi_t_last_us",
            "roi_positive",
            "roi_negative",
            "roi_event_offsets",
            "roi_event_y",
            "roi_event_x",
            "roi_event_p",
            "roi_event_t_us",
        )
        if any(name not in snapshot for name in required):
            raise ValueError("ROI sample arrays are incomplete")
        offsets = np.asarray(snapshot["roi_event_offsets"], dtype=np.int64)
        if offsets.size != sample_count + 1 or np.any(np.diff(offsets) < 0):
            raise ValueError("ROI event offsets are invalid")
        event_count = len(snapshot["roi_event_t_us"])
        if int(offsets[-1]) != event_count:
            raise ValueError("ROI event offsets do not match event arrays")
        for index in range(sample_count):
            if len(self.roi_samples) >= DISPLAY_SAMPLE_LIMIT:
                break
            start, stop = int(offsets[index]), int(offsets[index + 1])
            self.roi_samples.append(
                {
                    "slice_stop": slice_stop,
                    "center_y": int(snapshot["roi_center_y"][index]),
                    "center_x": int(snapshot["roi_center_x"][index]),
                    "first_time": int(snapshot["roi_t_first_us"][index]),
                    "peak_time": int(snapshot["roi_t_peak_us"][index]),
                    "last_time": int(snapshot["roi_t_last_us"][index]),
                    "positive": snapshot["roi_positive"][index],
                    "negative": snapshot["roi_negative"][index],
                    "event_y": snapshot["roi_event_y"][start:stop],
                    "event_x": snapshot["roi_event_x"][start:stop],
                    "event_p": snapshot["roi_event_p"][start:stop],
                    "event_t": snapshot["roi_event_t_us"][start:stop],
                }
            )

    def _reset_recording(self) -> None:
        self.recording_directory = None
        self.manifest = None
        self.slice_states.clear()
        self.loaded_snapshots.clear()
        self.snapshot_errors.clear()
        self.localization_image = None
        self.localization_count = 0
        self.monitor_warning = None
        self.peak_samples.clear()
        self.roi_samples.clear()
        self.peak_selection.set("")
        self.roi_selection.set("")

    def _update_status(self) -> None:
        if self.manifest is None:
            return
        completed = sum(
            state in {"completed", "skipped"} for state in self.slice_states.values()
        )
        total = len(self.manifest.slices)
        state = self.manifest.state
        if self.process_finished and not self.process_succeeded:
            state = "failed"
        status = (
            f"{self.manifest.recording_name} — {state}; {completed}/{total} slices; "
            f"{self.localization_count:,} localizations"
        )
        if self.manifest.error:
            status += f". {self.manifest.error}"
        if self.monitor_warning:
            status += f". {self.monitor_warning}; processing is unaffected"
        self.status_text.set(status)

    def _update_selectors(self) -> None:
        peak_values = [
            self._peak_label(index) for index in range(len(self.peak_samples))
        ]
        roi_values = [self._roi_label(index) for index in range(len(self.roi_samples))]
        self.peak_selector.configure(values=peak_values)
        self.roi_selector.configure(values=roi_values)
        if peak_values and self.peak_selection.get() not in peak_values:
            self.peak_selection.set(peak_values[0])
        if roi_values and self.roi_selection.get() not in roi_values:
            self.roi_selection.set(roi_values[0])
            self._configure_roi_sliders(0)

    def _peak_label(self, index: int) -> str:
        sample = self.peak_samples[index]
        return (
            f"{index + 1}: x={sample['x']}, y={sample['y']}, "
            f"t={sample['peak_time'] * 1e-6:.3f} s"
        )

    def _roi_label(self, index: int) -> str:
        sample = self.roi_samples[index]
        return (
            f"{index + 1}: x={sample['center_x']}, y={sample['center_y']}, "
            f"t={sample['peak_time'] * 1e-6:.3f} s"
        )

    def _selected_index(self, selection: str) -> int | None:
        if not selection or ":" not in selection:
            return None
        try:
            return int(selection.split(":", maxsplit=1)[0]) - 1
        except ValueError:
            return None

    def _selection_changed(self, _event: tk.Event[Any] | None = None) -> None:
        self._render_peak()
        self.canvas.draw_idle()

    def _roi_selection_changed(self, _event: tk.Event[Any] | None = None) -> None:
        index = self._selected_index(self.roi_selection.get())
        if index is not None and 0 <= index < len(self.roi_samples):
            self._configure_roi_sliders(index)
        self._render_roi()
        self.canvas.draw_idle()

    def _configure_roi_sliders(self, index: int) -> None:
        sample = self.roi_samples[index]
        first_time = float(sample["first_time"])
        last_time = float(sample["last_time"])
        if last_time <= first_time:
            last_time = first_time + 1.0
        self._updating_roi_controls = True
        self.roi_start_slider.configure(from_=first_time, to=last_time)
        self.roi_stop_slider.configure(from_=first_time, to=last_time)
        self.roi_start.set(first_time)
        self.roi_stop.set(last_time)
        self._updating_roi_controls = False
        self._update_roi_window_text(sample)

    def _roi_window_changed(self, _value: str) -> None:
        if self._updating_roi_controls:
            return
        index = self._selected_index(self.roi_selection.get())
        if index is not None and 0 <= index < len(self.roi_samples):
            self._update_roi_window_text(self.roi_samples[index])
        self._render_roi()
        self.canvas.draw_idle()

    def _update_roi_window_text(self, sample: dict[str, Any]) -> None:
        peak_time = float(sample["peak_time"])
        first_ms = (min(self.roi_start.get(), self.roi_stop.get()) - peak_time) / 1000.0
        last_ms = (max(self.roi_start.get(), self.roi_stop.get()) - peak_time) / 1000.0
        self.roi_window_text.set(
            f"Manual window relative to peak: {first_ms:+.3f} to {last_ms:+.3f} ms"
        )

    def _localization_clicked(self, event: Any) -> None:
        if event.inaxes is not self.localization_axis or not self.roi_samples:
            return
        if event.xdata is None or event.ydata is None:
            return
        distances = [
            (sample["center_x"] - event.xdata) ** 2
            + (sample["center_y"] - event.ydata) ** 2
            for sample in self.roi_samples
        ]
        index = int(np.argmin(distances))
        self.roi_selection.set(self._roi_label(index))
        self._configure_roi_sliders(index)
        self._render_roi()
        self.canvas.draw_idle()

    def _render_all(self) -> None:
        self._render_timeline()
        self._render_localizations()
        self._render_peak()
        self._render_roi()
        self.canvas.draw_idle()

    def _render_timeline(self) -> None:
        axis = self.timeline_axis
        axis.clear()
        manifest = self.manifest
        if manifest is None:
            axis.text(
                0.5, 0.5, "Waiting for recording metadata", ha="center", va="center"
            )
            axis.set_axis_off()
            return
        axis.set_axis_on()
        origin = manifest.recording_start_us
        duration = max(manifest.recording_stop_us - origin, 1) * 1e-6
        axis.broken_barh([(0, duration)], (0.1, 0.8), facecolors="#E5E7EB")
        selected_start = (manifest.selected_start_us - origin) * 1e-6
        selected_width = (
            max(manifest.selected_stop_us - manifest.selected_start_us, 0) * 1e-6
        )
        axis.broken_barh(
            [(selected_start, selected_width)],
            (0.1, 0.8),
            facecolors="#DCEEF7",
        )
        for window in manifest.slices:
            state = self.slice_states.get(window.stop_us, "pending")
            if manifest.state == "completed" and state == "pending":
                state = "skipped"
            if (
                self.process_finished
                and not self.process_succeeded
                and state == "active"
            ):
                state = "failed"
            axis.broken_barh(
                [
                    (
                        (window.start_us - origin) * 1e-6,
                        (window.stop_us - window.start_us) * 1e-6,
                    )
                ],
                (0.1, 0.8),
                facecolors=STATE_COLORS.get(state, STATE_COLORS["pending"]),
                edgecolors="white",
                linewidth=0.5,
            )
        axis.set_xlim(0, duration)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlabel("Recording time (s)")
        axis.set_title("Measurement slices")
        legend_states = ("pending", "active", "completed", "skipped", "failed")
        axis.legend(
            handles=[
                Patch(facecolor=STATE_COLORS[state], label=state.capitalize())
                for state in legend_states
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.4),
            ncol=len(legend_states),
            frameon=False,
            fontsize=7,
        )
        style_publication_axis(axis)

    def _render_localizations(self) -> None:
        axis = self.localization_axis
        previous_limits = None
        if axis.images:
            previous_limits = (axis.get_xlim(), axis.get_ylim())
        axis.clear()
        if self.localization_image is None:
            axis.text(0.5, 0.5, "No localizations yet", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            height, width = self.localization_image.shape
            axis.imshow(
                np.log1p(self.localization_image),
                origin="upper",
                extent=(0, width, height, 0),
                interpolation="nearest",
                cmap=EVENT_DENSITY_CMAP,
                aspect="equal",
            )
            if self.roi_samples:
                axis.scatter(
                    [sample["center_x"] for sample in self.roi_samples],
                    [sample["center_y"] for sample in self.roi_samples],
                    s=10,
                    facecolors="none",
                    edgecolors=PLOT_COLORS["sky_blue"],
                    linewidths=0.7,
                )
            axis.set_xlabel("x (sensor px)")
            axis.set_ylabel("y (sensor px)")
            if previous_limits is not None:
                axis.set_xlim(previous_limits[0])
                axis.set_ylim(previous_limits[1])
        axis.set_title(f"Accumulated localizations (n={self.localization_count:,})")
        style_publication_axis(axis)

    def _render_peak(self) -> None:
        axis = self.peak_axis
        axis.clear()
        index = self._selected_index(self.peak_selection.get())
        if index is None or not 0 <= index < len(self.peak_samples):
            axis.text(0.5, 0.5, "No peak samples yet", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            sample = self.peak_samples[index]
            times = np.asarray(sample["times"], dtype=np.float64)
            values = np.asarray(sample["values"], dtype=np.float64)
            valid = np.isfinite(times) & np.isfinite(values)
            relative_ms = (times[valid] - sample["peak_time"]) / 1000.0
            axis.plot(
                relative_ms,
                values[valid],
                color=PLOT_COLORS["blue"],
                linewidth=1.1,
            )
            axis.axvline(0, color=PLOT_COLORS["vermillion"], linewidth=0.9)
            axis.axvspan(
                (sample["on_time"] - sample["peak_time"]) / 1000.0,
                (sample["off_time"] - sample["peak_time"]) / 1000.0,
                color=PLOT_COLORS["orange"],
                alpha=0.2,
            )
            axis.set_xlabel("Time from peak (ms)")
            axis.set_ylabel("Cumulative event sum")
        axis.set_title("Extracted peak-center trace")
        style_publication_axis(axis)

    def _render_roi(self) -> None:
        self.roi_axis.clear()
        self.temporal_axis.clear()
        index = self._selected_index(self.roi_selection.get())
        if index is None or not 0 <= index < len(self.roi_samples):
            for axis, message in (
                (self.roi_axis, "No ROI samples yet"),
                (self.temporal_axis, "No ROI events yet"),
            ):
                axis.text(0.5, 0.5, message, ha="center", va="center")
                axis.set_xticks([])
                axis.set_yticks([])
                style_publication_axis(axis)
            self.roi_axis.set_title("Manually windowed emitter")
            self.temporal_axis.set_title("ROI event timing")
            return
        sample = self.roi_samples[index]
        event_t = np.asarray(sample["event_t"], dtype=np.float64)
        event_p = np.asarray(sample["event_p"], dtype=np.int8)
        event_y = np.asarray(sample["event_y"], dtype=np.int64)
        event_x = np.asarray(sample["event_x"], dtype=np.int64)
        first_time = min(self.roi_start.get(), self.roi_stop.get())
        last_time = max(self.roi_start.get(), self.roi_stop.get())
        selected = (event_t >= first_time) & (event_t <= last_time)
        shape = np.asarray(sample["positive"]).shape
        positive = np.zeros(shape, dtype=np.int32)
        negative = np.zeros(shape, dtype=np.int32)
        radius_y, radius_x = shape[0] // 2, shape[1] // 2
        relative_y = event_y[selected] - (sample["center_y"] - radius_y)
        relative_x = event_x[selected] - (sample["center_x"] - radius_x)
        within = (
            (relative_y >= 0)
            & (relative_y < shape[0])
            & (relative_x >= 0)
            & (relative_x < shape[1])
        )
        selected_polarity = event_p[selected][within]
        on = selected_polarity == 1
        off = ~on
        np.add.at(positive, (relative_y[within][on], relative_x[within][on]), 1)
        np.add.at(negative, (relative_y[within][off], relative_x[within][off]), 1)
        signed = positive - negative
        limit = max(int(np.max(np.abs(signed))) if signed.size else 0, 1)
        self.roi_axis.imshow(
            signed,
            origin="upper",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        self.roi_axis.scatter(radius_x, radius_y, marker="+", s=35, color="black")
        self.roi_axis.set_xlabel("ROI x (px)")
        self.roi_axis.set_ylabel("ROI y (px)")
        self.roi_axis.set_title(
            f"Emitter: ON {int(positive.sum())}, OFF {int(negative.sum())}"
        )
        style_publication_axis(self.roi_axis)

        if event_t.size:
            relative_time_ms = (event_t - sample["peak_time"]) / 1000.0
            bin_count = min(max(int(math.sqrt(event_t.size)), 8), 40)
            bins = np.linspace(
                relative_time_ms.min(), relative_time_ms.max(), bin_count + 1
            )
            if np.allclose(bins[0], bins[-1]):
                bins = np.linspace(bins[0] - 0.5, bins[0] + 0.5, bin_count + 1)
            self.temporal_axis.hist(
                relative_time_ms[event_p == 1],
                bins=bins,
                histtype="step",
                color=PLOT_COLORS["green"],
                label="ON",
            )
            self.temporal_axis.hist(
                relative_time_ms[event_p != 1],
                bins=bins,
                histtype="step",
                color=PLOT_COLORS["vermillion"],
                label="OFF",
            )
            self.temporal_axis.axvspan(
                (first_time - sample["peak_time"]) / 1000.0,
                (last_time - sample["peak_time"]) / 1000.0,
                color=PLOT_COLORS["sky_blue"],
                alpha=0.2,
            )
            self.temporal_axis.legend(frameon=False, fontsize=7)
        self.temporal_axis.set_xlabel("Time from peak (ms)")
        self.temporal_axis.set_ylabel("Events per bin")
        self.temporal_axis.set_title("ROI event timing")
        style_publication_axis(self.temporal_axis)
