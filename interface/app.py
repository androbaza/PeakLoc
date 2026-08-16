from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from interface.config_catalog import SettingSpec, settings_for_tier
from interface.operations import (
    CalibrationRequest,
    application_directory,
    format_captured_output,
    startup_config_path,
    worker_command,
)
from localization_scripts.pipeline_config import PeakLocConfig, write_effective_config

APP_TITLE = "PeakLoc"
APP_SUBTITLE = "Event-camera localization, guided from calibration to results"
BACKGROUND = "#F3F6FA"
SURFACE = "#FFFFFF"
INK = "#152333"
MUTED = "#5E6C7B"
ACCENT = "#0D9488"
ACCENT_DARK = "#0F766E"
DANGER = "#C2413B"
BORDER = "#D9E1E8"
OPTIONAL_TYPES: dict[str, type] = {
    "input_file": str,
    "slice_end": int,
    "slice_count": int,
    "cpu_worker_budget": int,
    "max_workers_per_slice": int,
    "calibration_path": str,
    "sigma_psf_px": float,
    "max_fit_center_offset_px": float,
    "max_localization_uncertainty_px": float,
    "max_localization_uncertainty_nm": float,
}


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self,
            background=BACKGROUND,
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(
            self.canvas, style="Page.TFrame", padding=(24, 18, 32, 32)
        )
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self, _event: tk.Event[Any]) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event[Any]) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event[Any]) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event[Any]) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event[Any]) -> None:
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")


class PeakLocApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE} — Desktop")
        self.root.geometry("1180x780")
        self.root.minsize(960, 680)
        self.root.configure(background=BACKGROUND)
        self._configure_style()

        self.config = self._load_startup_config()
        self.config_path = startup_config_path()
        self.config_vars = self._create_config_variables(self.config)
        self.input_mode = tk.StringVar(
            value="file" if self.config.input_file is not None else "folder"
        )
        selected_input = self.config.input_file or self.config.input_folder
        self.input_path = tk.StringVar(value=selected_input)
        self.status_text = tk.StringVar(value="Ready")
        self.readiness_text = tk.StringVar()

        self.dark_path = tk.StringVar()
        self.blank_path = tk.StringVar()
        self.calibration_output = tk.StringVar(
            value=str(application_directory() / "calibration_event_model.npz")
        )
        self.sensor_model = tk.StringVar(value="Prophesee event camera")
        self.calibration_id = tk.StringVar(value="event-model-calibration")

        self.process: subprocess.Popen[str] | None = None
        self.process_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.task_kind: str | None = None
        self.task_temp_directory: Path | None = None
        self.pending_calibration_output: Path | None = None
        self.log_path = application_directory() / "PeakLoc.log"

        self._build_layout()
        self._refresh_readiness()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        default_font = ("Segoe UI", 10) if os.name == "nt" else ("TkDefaultFont", 10)
        self.root.option_add("*Font", default_font)
        style.configure(".", background=BACKGROUND, foreground=INK)
        style.configure("Page.TFrame", background=BACKGROUND)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Header.TFrame", background=INK)
        style.configure(
            "HeaderTitle.TLabel",
            background=INK,
            foreground="#FFFFFF",
            font=(default_font[0], 20, "bold"),
        )
        style.configure(
            "HeaderSubtitle.TLabel",
            background=INK,
            foreground="#C8D3DE",
            font=(default_font[0], 10),
        )
        style.configure(
            "Title.TLabel",
            background=BACKGROUND,
            foreground=INK,
            font=(default_font[0], 18, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background=BACKGROUND,
            foreground=INK,
            font=(default_font[0], 12, "bold"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=SURFACE,
            foreground=INK,
            font=(default_font[0], 11, "bold"),
        )
        style.configure("Card.TLabel", background=SURFACE, foreground=INK)
        style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED)
        style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED)
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(18, 10),
            font=(default_font[0], 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#9EBDB9")],
        )
        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(14, 9),
        )
        style.configure("TButton", padding=(12, 8))
        style.configure("TEntry", fieldbackground="#FFFFFF", padding=7)
        style.configure("TCombobox", fieldbackground="#FFFFFF", padding=6)
        style.configure("TCheckbutton", background=BACKGROUND)
        style.configure("Card.TCheckbutton", background=SURFACE)
        style.configure("TRadiobutton", background=SURFACE)
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#E5EBF1",
            foreground=MUTED,
            padding=(18, 11),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", SURFACE)],
            foreground=[("selected", INK)],
        )

    def _load_startup_config(self) -> PeakLocConfig:
        path = startup_config_path()
        if not path.is_file():
            return PeakLocConfig()
        try:
            return PeakLocConfig.from_json(path)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showwarning(
                "Configuration not loaded",
                f"PeakLoc could not load {path}:\n\n{error}\n\nDefaults will be used.",
            )
            return PeakLocConfig()

    def _create_config_variables(self, config: PeakLocConfig) -> dict[str, tk.Variable]:
        variables: dict[str, tk.Variable] = {}
        for field in fields(PeakLocConfig):
            value = getattr(config, field.name)
            if isinstance(value, bool):
                variables[field.name] = tk.BooleanVar(value=value)
            else:
                variables[field.name] = tk.StringVar(
                    value="" if value is None else str(value)
                )
        return variables

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(28, 18))
        header.pack(fill="x")
        title_area = ttk.Frame(header, style="Header.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(title_area, text=APP_TITLE, style="HeaderTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            title_area,
            text=APP_SUBTITLE,
            style="HeaderSubtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        actions = ttk.Frame(header, style="Header.TFrame")
        actions.pack(side="right")
        ttk.Button(actions, text="Open config…", command=self._open_config).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="Save config…", command=self._save_config_as).pack(
            side="left", padx=4
        )

        body = ttk.Frame(self.root, style="Page.TFrame", padding=(24, 20, 24, 10))
        body.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill="both", expand=True)
        self.data_page = ttk.Frame(self.notebook, style="Page.TFrame")
        self.calibration_page = ttk.Frame(self.notebook, style="Page.TFrame")
        self.basic_page = ttk.Frame(self.notebook, style="Page.TFrame")
        self.advanced_page = ttk.Frame(self.notebook, style="Page.TFrame")
        self.run_page = ttk.Frame(self.notebook, style="Page.TFrame")
        self.notebook.add(self.data_page, text="1  Data")
        self.notebook.add(self.calibration_page, text="2  Calibration")
        self.notebook.add(self.basic_page, text="3  Basic settings")
        self.notebook.add(self.advanced_page, text="4  Advanced")
        self.notebook.add(self.run_page, text="5  Run")

        self._build_data_page()
        self._build_calibration_page()
        self._build_settings_page(self.basic_page, "basic")
        self._build_settings_page(self.advanced_page, "advanced")
        self._build_run_page()
        self.notebook.bind(
            "<<NotebookTabChanged>>", lambda _event: self._refresh_readiness()
        )

        status_bar = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 8))
        status_bar.pack(fill="x", side="bottom")
        ttk.Label(
            status_bar, textvariable=self.status_text, style="CardMuted.TLabel"
        ).pack(side="left")
        ttk.Label(
            status_bar,
            text="Settings are validated before every task",
            style="CardMuted.TLabel",
        ).pack(side="right")

    def _page_intro(
        self,
        parent: tk.Misc,
        title: str,
        description: str,
    ) -> ttk.Frame:
        container = ttk.Frame(parent, style="Page.TFrame", padding=(28, 24))
        container.pack(fill="both", expand=True)
        ttk.Label(container, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text=description,
            style="Muted.TLabel",
            wraplength=850,
            justify="left",
        ).pack(anchor="w", pady=(6, 20))
        return container

    def _build_data_page(self) -> None:
        container = self._page_intro(
            self.data_page,
            "Choose recordings",
            "Process one .raw/.npy recording or every recording in a folder. "
            "PeakLoc writes each run beside its source recording.",
        )
        card = ttk.Frame(container, style="Surface.TFrame", padding=24)
        card.pack(fill="x")
        ttk.Label(
            card, text="What do you want to process?", style="CardTitle.TLabel"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Radiobutton(
            card,
            text="One recording",
            variable=self.input_mode,
            value="file",
            command=self._input_mode_changed,
        ).grid(row=1, column=0, sticky="w", padx=(0, 24))
        ttk.Radiobutton(
            card,
            text="A folder of recordings",
            variable=self.input_mode,
            value="folder",
            command=self._input_mode_changed,
        ).grid(row=1, column=1, sticky="w")
        ttk.Entry(card, textvariable=self.input_path).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(18, 6)
        )
        ttk.Button(card, text="Browse…", command=self._browse_input).grid(
            row=2, column=2, sticky="ew", padx=(10, 0), pady=(18, 6)
        )
        self.recursive_check = ttk.Checkbutton(
            card,
            text="Include recordings in subfolders",
            variable=self.config_vars["recursive_input"],
            style="Card.TCheckbutton",
            command=self._refresh_readiness,
        )
        self.recursive_check.grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        card.columnconfigure(1, weight=1)

        hint = ttk.Frame(container, style="Surface.TFrame", padding=20)
        hint.pack(fill="x", pady=(16, 0))
        ttk.Label(hint, text="Good first run", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            hint,
            text=(
                "Choose one representative recording, use a short slice duration, then visit "
                "Run and select Check setup. Expand to a whole folder after reviewing the first output."
            ),
            style="CardMuted.TLabel",
            wraplength=850,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

    def _input_mode_changed(self) -> None:
        self.input_path.set("")
        self.recursive_check.configure(
            state="normal" if self.input_mode.get() == "folder" else "disabled"
        )
        self._refresh_readiness()

    def _browse_input(self) -> None:
        if self.input_mode.get() == "file":
            selected = filedialog.askopenfilename(
                title="Choose a PeakLoc recording",
                filetypes=[
                    ("Event recordings", "*.raw *.npy"),
                    ("RAW recordings", "*.raw"),
                    ("NumPy event arrays", "*.npy"),
                ],
            )
        else:
            selected = filedialog.askdirectory(title="Choose a recordings folder")
        if selected:
            self.input_path.set(str(Path(selected).resolve()))
            self._refresh_readiness()

    def _build_calibration_page(self) -> None:
        scroll = ScrollableFrame(self.calibration_page)
        scroll.pack(fill="both", expand=True)
        container = scroll.content
        ttk.Label(container, text="Prepare calibration", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            container,
            text=(
                "A dark recording measures sensor noise. A laser-on blank recording measures "
                "background without emitters. PeakLoc combines them into one reusable .npz file."
            ),
            style="Muted.TLabel",
            wraplength=880,
            justify="left",
        ).pack(anchor="w", pady=(6, 20))

        build_card = ttk.Frame(container, style="Surface.TFrame", padding=22)
        build_card.pack(fill="x")
        ttk.Label(
            build_card,
            text="Create a calibration file",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))
        self._path_row(
            build_card,
            row=1,
            label="Dark recording",
            variable=self.dark_path,
            command=lambda: self._browse_calibration_recording(self.dark_path),
        )
        self._path_row(
            build_card,
            row=2,
            label="Laser-on blank",
            variable=self.blank_path,
            command=lambda: self._browse_calibration_recording(self.blank_path),
        )
        self._path_row(
            build_card,
            row=3,
            label="Calibration output",
            variable=self.calibration_output,
            command=self._browse_calibration_output,
        )
        ttk.Label(build_card, text="Calibration name", style="Card.TLabel").grid(
            row=4, column=0, sticky="w", pady=6
        )
        ttk.Entry(build_card, textvariable=self.calibration_id).grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=6
        )
        ttk.Label(build_card, text="Sensor model", style="Card.TLabel").grid(
            row=5, column=0, sticky="w", pady=6
        )
        ttk.Entry(build_card, textvariable=self.sensor_model).grid(
            row=5, column=1, columnspan=2, sticky="ew", pady=6
        )
        calibration_note = (
            "Pixel size, sensor width/height, and raw read-buffer values come from Settings. "
            "The current values are shown in the readiness summary before calibration starts."
        )
        ttk.Label(
            build_card,
            text=calibration_note,
            style="CardMuted.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 10))
        self.calibrate_button = ttk.Button(
            build_card,
            text="Build calibration",
            style="Accent.TButton",
            command=self._start_calibration,
        )
        self.calibrate_button.grid(row=7, column=0, columnspan=3, sticky="w")
        build_card.columnconfigure(1, weight=1)

        use_card = ttk.Frame(container, style="Surface.TFrame", padding=22)
        use_card.pack(fill="x", pady=(16, 0))
        ttk.Label(
            use_card,
            text="Calibration used for processing",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Entry(
            use_card,
            textvariable=self.config_vars["calibration_path"],
        ).grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(
            use_card,
            text="Choose existing…",
            command=self._browse_existing_calibration,
        ).grid(row=1, column=2, padx=(10, 0))
        ttk.Checkbutton(
            use_card,
            text="Allow running without calibration (exploratory use only)",
            variable=self.config_vars["allow_uncalibrated"],
            style="Card.TCheckbutton",
            command=self._refresh_readiness,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Label(
            use_card,
            text=(
                "For publication-oriented work, build or select calibration and turn off "
                "uncalibrated mode. The setup check validates the file and sensor dimensions."
            ),
            style="CardMuted.TLabel",
            wraplength=820,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        use_card.columnconfigure(1, weight=1)

    def _path_row(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Any,
    ) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 12)
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=6
        )
        ttk.Button(parent, text="Browse…", command=command).grid(
            row=row, column=2, padx=(10, 0), pady=6
        )

    def _browse_calibration_recording(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a calibration recording",
            filetypes=[("RAW recordings", "*.raw"), ("All files", "*.*")],
        )
        if selected:
            variable.set(str(Path(selected).resolve()))

    def _browse_calibration_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save calibration",
            defaultextension=".npz",
            filetypes=[("PeakLoc calibration", "*.npz")],
            initialfile=Path(self.calibration_output.get()).name,
        )
        if selected:
            self.calibration_output.set(str(Path(selected).resolve()))

    def _browse_existing_calibration(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a PeakLoc calibration",
            filetypes=[("PeakLoc calibration", "*.npz")],
        )
        if selected:
            self.config_vars["calibration_path"].set(str(Path(selected).resolve()))
            self.config_vars["allow_uncalibrated"].set(False)
            self._refresh_readiness()

    def _build_settings_page(self, parent: ttk.Frame, tier: str) -> None:
        scroll = ScrollableFrame(parent)
        scroll.pack(fill="both", expand=True)
        content = scroll.content
        title = "Basic settings" if tier == "basic" else "Advanced settings"
        description = (
            "Start here. These settings control the processing range, detection sensitivity, "
            "fit scale, resources, and primary outputs."
            if tier == "basic"
            else "These controls expose every remaining PeakLoc option. Keep defaults unless "
            "your acquisition protocol or quality-control review gives a reason to change them."
        )
        ttk.Label(content, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            content,
            text=description,
            style="Muted.TLabel",
            wraplength=880,
            justify="left",
        ).pack(anchor="w", pady=(6, 20))

        current_group = ""
        group_card: ttk.Frame | None = None
        row = 0
        for spec in settings_for_tier(tier):
            if spec.group != current_group:
                current_group = spec.group
                group_card = ttk.Frame(content, style="Surface.TFrame", padding=20)
                group_card.pack(fill="x", pady=(0, 14))
                ttk.Label(
                    group_card,
                    text=current_group,
                    style="CardTitle.TLabel",
                ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
                group_card.columnconfigure(1, weight=1)
                row = 1
            if group_card is None:
                continue
            self._setting_row(group_card, row, spec)
            row += 1

    def _setting_row(self, parent: ttk.Frame, row: int, spec: SettingSpec) -> None:
        label_text = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
        label = ttk.Label(parent, text=label_text, style="Card.TLabel")
        label.grid(row=row, column=0, sticky="nw", padx=(0, 18), pady=8)
        variable = self.config_vars[spec.name]
        if isinstance(variable, tk.BooleanVar):
            control: tk.Widget = ttk.Checkbutton(
                parent,
                text="Enabled",
                variable=variable,
                style="Card.TCheckbutton",
                command=self._refresh_readiness,
            )
        elif spec.choices:
            control = ttk.Combobox(
                parent,
                textvariable=variable,
                values=spec.choices,
                state="readonly",
                width=30,
            )
        else:
            control = ttk.Entry(parent, textvariable=variable, width=28)
        control.grid(row=row, column=1, sticky="new", pady=6)
        ttk.Label(
            parent,
            text=spec.description,
            style="CardMuted.TLabel",
            wraplength=430,
            justify="left",
        ).grid(row=row, column=2, sticky="nw", padx=(18, 0), pady=8)

    def _build_run_page(self) -> None:
        container = self._page_intro(
            self.run_page,
            "Check and run PeakLoc",
            "Check setup first. PeakLoc validates paths, calibration, sensor geometry, "
            "resource headroom, and scientific consistency before processing.",
        )
        summary = ttk.Frame(container, style="Surface.TFrame", padding=20)
        summary.pack(fill="x")
        ttk.Label(summary, text="Current setup", style="CardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            summary,
            textvariable=self.readiness_text,
            style="CardMuted.TLabel",
            wraplength=860,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        actions = ttk.Frame(container, style="Page.TFrame")
        actions.pack(fill="x", pady=(16, 10))
        self.preflight_button = ttk.Button(
            actions,
            text="Check setup",
            command=lambda: self._start_pipeline(preflight_only=True),
        )
        self.preflight_button.pack(side="left")
        self.run_button = ttk.Button(
            actions,
            text="Start processing",
            style="Accent.TButton",
            command=lambda: self._start_pipeline(preflight_only=False),
        )
        self.run_button.pack(side="left", padx=(10, 0))
        self.cancel_button = ttk.Button(
            actions,
            text="Cancel",
            style="Danger.TButton",
            command=self._cancel_task,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(10, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        log_card = ttk.Frame(container, style="Surface.TFrame", padding=12)
        log_card.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_card,
            height=18,
            background="#0F1D2B",
            foreground="#DDE7F0",
            insertbackground="#FFFFFF",
            selectbackground=ACCENT_DARK,
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Consolas", 9) if os.name == "nt" else ("TkFixedFont", 9),
            wrap="word",
            state="disabled",
        )
        log_scroll = ttk.Scrollbar(
            log_card,
            orient="vertical",
            command=self.log_text.yview,
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _collect_config(self) -> PeakLocConfig:
        defaults = PeakLocConfig()
        payload: dict[str, object] = {}
        for field in fields(PeakLocConfig):
            variable = self.config_vars[field.name]
            if isinstance(variable, tk.BooleanVar):
                payload[field.name] = bool(variable.get())
                continue
            raw_value = str(variable.get()).strip()
            if field.name in OPTIONAL_TYPES and not raw_value:
                payload[field.name] = None
                continue
            target_type = OPTIONAL_TYPES.get(
                field.name, type(getattr(defaults, field.name))
            )
            try:
                payload[field.name] = target_type(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{field.name} has an invalid value: {raw_value!r}"
                ) from error

        selected_path = self.input_path.get().strip()
        if self.input_mode.get() == "file":
            payload["input_file"] = selected_path or None
            if selected_path:
                payload["input_folder"] = str(Path(selected_path).parent)
        else:
            payload["input_file"] = None
            payload["input_folder"] = selected_path
        return PeakLocConfig.from_mapping(payload)

    def _apply_config(self, config: PeakLocConfig, path: Path) -> None:
        self.config = config
        self.config_path = path
        for field in fields(PeakLocConfig):
            value = getattr(config, field.name)
            variable = self.config_vars[field.name]
            variable.set(
                value
                if isinstance(value, bool)
                else ("" if value is None else str(value))
            )
        if config.input_file is not None:
            self.input_mode.set("file")
            self.input_path.set(config.input_file)
        else:
            self.input_mode.set("folder")
            self.input_path.set(config.input_folder)
        self.recursive_check.configure(
            state="normal" if self.input_mode.get() == "folder" else "disabled"
        )
        self.status_text.set(f"Loaded {path.name}")
        self._refresh_readiness()

    def _open_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open PeakLoc configuration",
            filetypes=[("PeakLoc configuration", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        try:
            config = PeakLocConfig.from_json(path)
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("Configuration error", str(error))
            return
        self._apply_config(config, path)

    def _save_config_as(self) -> None:
        try:
            config = self._collect_config()
        except ValueError as error:
            messagebox.showerror("Configuration error", str(error))
            return
        selected = filedialog.asksaveasfilename(
            title="Save PeakLoc configuration",
            defaultextension=".json",
            filetypes=[("PeakLoc configuration", "*.json")],
            initialfile=self.config_path.name,
        )
        if not selected:
            return
        path = Path(selected)
        write_effective_config(config, path)
        self.config = config
        self.config_path = path
        self.status_text.set(f"Saved {path.name}")

    def _start_calibration(self) -> None:
        if self.process is not None:
            return
        dark_path = Path(self.dark_path.get().strip())
        blank_path = Path(self.blank_path.get().strip())
        output_path = Path(self.calibration_output.get().strip())
        if not dark_path.is_file() or dark_path.suffix.lower() != ".raw":
            messagebox.showerror(
                "Dark recording required",
                "Choose an existing .raw dark recording.",
            )
            return
        if not blank_path.is_file() or blank_path.suffix.lower() != ".raw":
            messagebox.showerror(
                "Blank recording required",
                "Choose an existing .raw laser-on blank recording.",
            )
            return
        if dark_path.resolve() == blank_path.resolve():
            messagebox.showerror(
                "Two recordings required",
                "Dark and laser-on blank recordings must be different files.",
            )
            return
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
            self.calibration_output.set(str(output_path))
        try:
            request = CalibrationRequest(
                dark_path=str(dark_path.resolve()),
                blank_path=str(blank_path.resolve()),
                output_path=str(output_path.resolve()),
                pixel_size_nm=float(self.config_vars["optical_pixel_size"].get()),
                sensor_model=self.sensor_model.get().strip() or "unknown",
                calibration_id=(
                    self.calibration_id.get().strip() or "event-model-calibration"
                ),
                height=int(self.config_vars["sensor_height"].get()),
                width=int(self.config_vars["sensor_width"].get()),
                max_events=int(self.config_vars["max_raw_events"].get()),
            )
        except ValueError as error:
            messagebox.showerror(
                "Calibration settings error",
                f"Check pixel size, sensor dimensions, and raw read buffer.\n\n{error}",
            )
            return

        temp_directory = Path(tempfile.mkdtemp(prefix="peakloc_gui_calibration_"))
        request_path = temp_directory / "calibration_request.json"
        request.write_json(request_path)
        self.pending_calibration_output = output_path.resolve()
        self.notebook.select(self.run_page)
        self._start_task(
            worker_command("--calibration-worker", str(request_path)),
            kind="calibration",
            temp_directory=temp_directory,
        )

    def _start_pipeline(self, *, preflight_only: bool) -> None:
        if self.process is not None:
            return
        try:
            config = self._collect_config()
        except ValueError as error:
            messagebox.showerror("Configuration error", str(error))
            return
        if config.input_file is not None:
            selected_path = Path(config.input_file)
            if not selected_path.is_file():
                messagebox.showerror(
                    "Recording required",
                    "Choose an existing .raw or .npy recording on the Data tab.",
                )
                self.notebook.select(self.data_page)
                return
        elif not Path(config.input_folder).is_dir():
            messagebox.showerror(
                "Recording folder required",
                "Choose an existing recordings folder on the Data tab.",
            )
            self.notebook.select(self.data_page)
            return

        temp_directory = Path(tempfile.mkdtemp(prefix="peakloc_gui_run_"))
        config_path = temp_directory / "peakloc_run_config.json"
        write_effective_config(config, config_path)
        command = worker_command("--pipeline-worker", "--config", str(config_path))
        if preflight_only:
            command.append("--preflight-only")
        self.config = config
        self._start_task(
            command,
            kind="preflight" if preflight_only else "pipeline",
            temp_directory=temp_directory,
        )

    def _start_task(
        self,
        command: list[str],
        *,
        kind: str,
        temp_directory: Path,
    ) -> None:
        self._append_log(f"\n[{kind.upper()}] Starting…\n")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command,
                cwd=application_directory(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                start_new_session=os.name != "nt",
            )
        except Exception:
            shutil.rmtree(temp_directory, ignore_errors=True)
            raise
        self.task_kind = kind
        self.task_temp_directory = temp_directory
        self._set_busy(True)
        reader = threading.Thread(target=self._read_process_output, daemon=True)
        reader.start()
        self.root.after(100, self._poll_process_queue)

    def _read_process_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.process_queue.put(("line", line))
        return_code = process.wait()
        self.process_queue.put(("done", return_code))

    def _poll_process_queue(self) -> None:
        while True:
            try:
                event, value = self.process_queue.get_nowait()
            except queue.Empty:
                break
            if event == "line":
                self._append_log(str(value))
            elif event == "done":
                self._finish_task(int(value))
                return
        if self.process is not None:
            self.root.after(100, self._poll_process_queue)

    def _append_log(self, text: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(format_captured_output(text))
        except OSError:
            # Logging must not prevent the GUI from displaying or completing a run.
            pass
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.preflight_button.configure(state=state)
        self.run_button.configure(state=state)
        self.calibrate_button.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start(12)
            self.status_text.set("Working — follow progress in the Run log")
        else:
            self.progress.stop()

    def _finish_task(self, return_code: int) -> None:
        kind = self.task_kind or "task"
        succeeded = return_code == 0
        self._append_log(
            f"\n[{kind.upper()}] {'Completed' if succeeded else 'Stopped with errors'} "
            f"(exit code {return_code}).\n"
        )
        if (
            succeeded
            and kind == "calibration"
            and self.pending_calibration_output is not None
        ):
            self.config_vars["calibration_path"].set(
                str(self.pending_calibration_output)
            )
            self.config_vars["allow_uncalibrated"].set(False)
            self.calibration_output.set(str(self.pending_calibration_output))
            self.status_text.set("Calibration built and selected")
        elif succeeded and kind == "preflight":
            self.status_text.set("Setup check passed")
        elif succeeded:
            self.status_text.set("PeakLoc processing completed")
        else:
            self.status_text.set(f"{kind.capitalize()} failed — review the log")

        temp_directory = self.task_temp_directory
        self.process = None
        self.task_kind = None
        self.task_temp_directory = None
        self.pending_calibration_output = None
        if temp_directory is not None:
            shutil.rmtree(temp_directory, ignore_errors=True)
        self._set_busy(False)
        self._refresh_readiness()

    def _cancel_task(self) -> None:
        if self.process is None:
            return
        if not messagebox.askyesno(
            "Cancel current task?",
            "PeakLoc will stop the current operation. Partial run outputs may remain for review.",
        ):
            return
        self.status_text.set("Stopping current task…")
        self._terminate_process_tree(self.process)

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

    def _refresh_readiness(self) -> None:
        if not hasattr(self, "readiness_text"):
            return
        selected = self.input_path.get().strip() or "No recording selected"
        mode = (
            "Single recording"
            if self.input_mode.get() == "file"
            else "Recording folder"
        )
        calibration = str(self.config_vars["calibration_path"].get()).strip()
        if calibration:
            calibration_status = f"Calibration: {Path(calibration).name}"
        elif bool(self.config_vars["allow_uncalibrated"].get()):
            calibration_status = "Calibration: exploratory uncalibrated mode"
        else:
            calibration_status = "Calibration: required but not selected"
        self.readiness_text.set(
            f"{mode}: {selected}\n"
            f"{calibration_status}\n"
            f"Range: {self.config_vars['slice_start'].get()} to "
            f"{self.config_vars['slice_end'].get() or 'recording end'} µs; "
            f"{self.config_vars['slice_duration'].get()} µs per slice\n"
            f"Sensor: {self.config_vars['sensor_width'].get()} × "
            f"{self.config_vars['sensor_height'].get()} px at "
            f"{self.config_vars['optical_pixel_size'].get()} nm/px"
        )

    def _on_close(self) -> None:
        if self.process is not None:
            should_close = messagebox.askyesno(
                "PeakLoc is still working",
                "Stop the current task and close PeakLoc?",
            )
            if not should_close:
                return
            self._terminate_process_tree(self.process)
        self.root.destroy()


def launch() -> None:
    root = tk.Tk()
    PeakLocApp(root)
    root.mainloop()
