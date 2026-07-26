from __future__ import annotations

import csv
import ctypes
import errno
import gc
import os
if os.name == "nt":
    import msvcrt
else:
    import fcntl
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
import json
import pickle
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TextIO

try:
    import resource
except ModuleNotFoundError:
    resource = None

import matplotlib
import numpy as np
from loguru import logger
from natsort import natsorted

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.artifact_layout import ArtifactLayout
from localization_scripts.calibration import EventCalibration, load_calibration
from localization_scripts.event_array_processing import (
    array_to_polarity_map,
    array_to_time_map,
    array_to_time_map_for_coords,
    create_convolved_signals,
    EVENT_DTYPE,
    materialize_raw_events,
    save_dict,
)
from localization_scripts.diffuse_flash import (
    DiffuseFlashDetection,
    TimeInterval,
    detect_diffuse_flash_intervals,
    exclude_time_intervals,
    iter_retained_event_spans,
)
from localization_scripts.fit_review_diagnostics import save_fit_review_diagnostics
from localization_scripts.localization_fitting import (
    localization_uncertainty_px,
    localization_qc_dtype,
    localize_rois_with_attempts,
)
from localization_scripts.peak_finding import (
    create_peak_lists,
    find_local_max_peak,
    find_peaks_parallel,
    group_timestamps_by_coordinate,
)
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import PREVIEW_DPI, save_publication_figure
from localization_scripts.plotting_functions import plot_rois_from_locs
from localization_scripts.provenance import save_portable_outputs
from localization_scripts.qc_dashboard import EventQCAccumulator, save_run_qc_dashboard
from localization_scripts.recording_discovery import find_recording_files
from localization_scripts.roi_generation import (
    generate_coord_lists,
    generate_rois,
)
from localization_scripts.smlm_visualization import save_smlm_visualization
from localization_scripts.spatial_mask import (
    SpatialMask,
    accumulate_event_density_in_time_window,
    build_spatial_mask,
    disabled_spatial_mask,
)
from localization_scripts.temporal_roi_generation import (
    generate_temporally_segmented_rois,
)
from localization_scripts.temporal_segmentation import temporal_settings_from_config


SLICE_TEMP_ARTIFACT_PREFIXES = (
    "attempted_localizations",
    "localizations",
    "localization_qc",
    "temporal_segmentation_qc",
    "rois",
    "unique_peaks",
)
JOBLIB_TEMP_DIRNAME = "joblib"
SLICE_MANIFEST_DIRNAME = "slices"
SLICE_REQUEST_DIRNAME = "slice_requests"
GIB = 1024**3
SLICE_MEMORY_EVENT_MULTIPLIER = 8
SLICE_DISK_EVENT_MULTIPLIER = 8
PARALLEL_STAGE_LOCK_FILENAME = "parallel_stage.lock"

_SLICE_WORKER_CONTEXT: SliceExecutionConfig | None = None
_SLICE_WORKER_CALIBRATION: EventCalibration | None = None


@dataclass
class SliceStageMetrics:
    event_index_seconds: float = 0.0
    convolution_seconds: float = 0.0
    peak_interpolation_seconds: float = 0.0
    peak_filter_seconds: float = 0.0
    roi_generation_seconds: float = 0.0
    localization_seconds: float = 0.0
    artifact_write_seconds: float = 0.0
    memory_release_seconds: float = 0.0


@dataclass
class SliceResult:
    time_slice: int
    event_count: int
    unique_peak_count: int
    roi_count: int
    localization_count: int
    elapsed_seconds: float
    fit_success_fraction: float | None = None
    median_uncertainty_px: float | None = None
    median_nll_per_event: float | None = None
    hot_pixel_fraction: float | None = None
    rejected_localization_count: int = 0
    diffuse_flash_excluded_event_count: int = 0
    event_bytes: int = 0
    peak_rss_bytes: int = 0
    temp_disk_bytes: int = 0
    process_id: int = 0
    stage_metrics: SliceStageMetrics = field(default_factory=SliceStageMetrics)
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class RecordingResult:
    input_file: Path
    output_folder: Path
    event_count: int
    time_min: int | None
    time_max: int | None
    slice_results: list[SliceResult] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    calibration_metadata: dict[str, object] = field(default_factory=dict)
    spatial_mask_metadata: dict[str, object] = field(default_factory=dict)
    diffuse_flash_intervals: tuple[TimeInterval, ...] = ()
    diffuse_flash_excluded_event_count: int = 0
    diffuse_flash_transition_bin_count: int = 0


@dataclass(frozen=True)
class SliceTask:
    slice_start: int
    time_slice: int
    start_index: int
    stop_index: int
    event_bytes: int


@dataclass(frozen=True)
class SliceWorkerRequest:
    context: SliceExecutionConfig
    task: SliceTask
    result_path: Path


@dataclass(frozen=True)
class SliceExecutionConfig:
    filename: Path
    event_path: Path
    event_format: Literal["npy", "raw_cache"]
    event_count: int
    output_folder: Path
    config: PeakLocConfig
    spatial_mask: SpatialMask | None
    diffuse_flash_intervals: tuple[TimeInterval, ...] = ()
    parallel_stage_lock_path: Path | None = None


@dataclass
class EventStore:
    """A recording-backed array and the optional temporary RAW cache behind it."""

    events: np.ndarray
    cache_path: Path | None = None
    timestamps_monotonic: bool = True


class SliceResourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimeSlice:
    start: int
    stop: int


def build_time_slices(config: PeakLocConfig, time_max: int) -> list[TimeSlice]:
    """Build contiguous processing windows, including a bounded final window."""
    if config.slice_end is None:
        stop = time_max + config.slice_duration + 1
        if config.slice_count is not None:
            stop = min(
                stop,
                config.slice_start + config.slice_duration * (config.slice_count + 1),
            )
        return [
            TimeSlice(time_slice - config.slice_duration, time_slice)
            for time_slice in range(
                config.slice_start + config.slice_duration,
                stop,
                config.slice_duration,
            )
        ]

    stop = min(config.slice_end, time_max + 1)
    if config.slice_count is not None:
        stop = min(stop, config.slice_start + config.slice_duration * config.slice_count)
    return [
        TimeSlice(start, min(start + config.slice_duration, stop))
        for start in range(config.slice_start, stop, config.slice_duration)
    ]


def build_slice_tasks(
    events: np.ndarray,
    time_slices: Iterable[TimeSlice | int],
    *,
    slice_duration: int | None = None,
) -> list[SliceTask]:
    """Build parallel tasks for explicit windows or legacy slice end times."""
    timestamps = events["t"]
    itemsize = int(events.dtype.itemsize)
    tasks = []
    for time_slice in time_slices:
        if isinstance(time_slice, TimeSlice):
            window = time_slice
        else:
            if slice_duration is None:
                raise ValueError("slice_duration is required for integer time slices")
            window = TimeSlice(time_slice - slice_duration, time_slice)
        start_index = int(np.searchsorted(timestamps, window.start, side="left"))
        stop_index = int(np.searchsorted(timestamps, window.stop, side="left"))
        tasks.append(
            SliceTask(
                slice_start=window.start,
                time_slice=window.stop,
                start_index=start_index,
                stop_index=stop_index,
                event_bytes=max(stop_index - start_index, 0) * itemsize,
            )
        )
    return tasks


def execute_slice_tasks(
    tasks: list[SliceTask], context: SliceExecutionConfig
) -> list[SliceResult]:
    non_empty_tasks = [task for task in tasks if task.stop_index > task.start_index]
    if not non_empty_tasks:
        return []

    max_workers = min(context.config.effective_concurrent_slices, len(non_empty_tasks))
    results = []
    next_task_index = 0
    active: dict[int, tuple[subprocess.Popen[bytes], SliceTask, Path, Path]] = {}
    try:
        while next_task_index < len(non_empty_tasks) or active:
            while next_task_index < len(non_empty_tasks) and len(active) < max_workers:
                task = non_empty_tasks[next_task_index]
                active_tasks = [entry[1] for entry in active.values()]
                if not _resources_allow_submission(
                    task, active_tasks, context.config, context.output_folder
                ):
                    if not active:
                        raise SliceResourceError(
                            _resource_failure_message(task, context)
                        )
                    break
                process, request_path, result_path = _start_slice_process(task, context)
                active[process.pid] = (process, task, request_path, result_path)
                next_task_index += 1

            completed_pids = [
                pid for pid, entry in active.items() if entry[0].poll() is not None
            ]
            if not completed_pids:
                time.sleep(0.1)
                continue

            for pid in completed_pids:
                process, task, request_path, result_path = active.pop(pid)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Slice worker for {task.time_slice} exited with "
                        f"status {process.returncode}; request preserved at "
                        f"{request_path}"
                    )
                result = _read_slice_worker_result(result_path)
                request_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)
                if result is not None:
                    _validate_slice_result(result)
                    results.append(result)
    except BaseException:
        for process, _, _, _ in active.values():
            _terminate_process_group(process)
        raise

    return sorted(results, key=lambda result: result.time_slice)


def _start_slice_process(
    task: SliceTask, context: SliceExecutionConfig
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    request_folder = context.output_folder / "temp_files" / SLICE_REQUEST_DIRNAME
    request_folder.mkdir(parents=True, exist_ok=True)
    request_path = request_folder / f"request_{task.time_slice}.pkl"
    result_path = request_folder / f"result_{task.time_slice}.pkl"
    partial_path = _partial_artifact_path(request_path)
    request = SliceWorkerRequest(context=context, task=task, result_path=result_path)
    with partial_path.open("wb") as output_file:
        pickle.dump(request, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    partial_path.replace(request_path)

    logger.info(
        "Submitting slice ending at {} ({} events, {} workers)",
        task.time_slice,
        task.stop_index - task.start_index,
        context.config.parallel_workers,
    )
    entrypoint = Path(__file__).resolve().parents[1] / "PeakLoc.py"
    environment = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, str(entrypoint), "--slice-worker", str(request_path)],
        cwd=entrypoint.parent,
        env=environment,
        start_new_session=True,
    )
    return process, request_path, result_path


def run_serialized_slice_worker(request_path: Path) -> None:
    with request_path.open("rb") as input_file:
        request: SliceWorkerRequest = pickle.load(input_file)
    _initialize_slice_worker(request.context)
    result = _run_slice_task(request.task)
    partial_path = _partial_artifact_path(request.result_path)
    with partial_path.open("wb") as output_file:
        pickle.dump(result, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    partial_path.replace(request.result_path)


def _read_slice_worker_result(result_path: Path) -> SliceResult | None:
    if not result_path.is_file():
        raise RuntimeError(f"Slice worker did not write its result: {result_path}")
    with result_path.open("rb") as input_file:
        result: SliceResult | None = pickle.load(input_file)
    return result


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _initialize_slice_worker(context: SliceExecutionConfig) -> None:
    global _SLICE_WORKER_CONTEXT, _SLICE_WORKER_CALIBRATION
    _SLICE_WORKER_CONTEXT = context
    _configure_numerical_threads(context.config.parallel_workers)
    _SLICE_WORKER_CALIBRATION = load_calibration(
        context.config.calibration_path,
        context.config.sensor_shape,
        allow_uncalibrated=context.config.allow_uncalibrated,
    )


def _run_slice_task(task: SliceTask) -> SliceResult | None:
    context = _SLICE_WORKER_CONTEXT
    calibration = _SLICE_WORKER_CALIBRATION
    if context is None or calibration is None:
        raise RuntimeError("Slice worker was not initialized")
    events = _open_worker_event_store(context)
    try:
        event_slice = np.asarray(events[task.start_index : task.stop_index])
        result = process_time_slice(
            event_slice,
            task.time_slice,
            context.filename,
            context.config,
            calibration,
            context.output_folder,
            spatial_mask=context.spatial_mask,
            diffuse_flash_intervals=context.diffuse_flash_intervals,
            parallel_stage_lock_path=context.parallel_stage_lock_path,
            slice_start=task.slice_start,
        )
        if result is not None:
            _write_slice_manifest(result, context.output_folder)
        return result
    finally:
        _shutdown_loky_workers()
        _close_memory_map(events)
        release_unused_memory()


def _shutdown_loky_workers() -> None:
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True, kill_workers=True)


def _open_worker_event_store(context: SliceExecutionConfig) -> np.ndarray:
    if context.event_format == "npy":
        return np.load(context.event_path, mmap_mode="r", allow_pickle=False)
    if context.event_format == "raw_cache":
        return np.memmap(
            context.event_path,
            dtype=EVENT_DTYPE,
            mode="r",
            shape=(context.event_count,),
        )
    raise ValueError(f"Unsupported worker event format: {context.event_format}")


def _configure_numerical_threads(worker_count: int) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    from numba import set_num_threads

    set_num_threads(worker_count)


def _resources_allow_submission(
    task: SliceTask,
    active_tasks: Iterable[SliceTask],
    config: PeakLocConfig,
    output_folder: Path,
) -> bool:
    active_event_bytes = sum(active.event_bytes for active in active_tasks)
    projected_memory = (
        active_event_bytes + task.event_bytes
    ) * SLICE_MEMORY_EVENT_MULTIPLIER
    memory_ok = (
        _available_memory_bytes()
        >= int(config.memory_reserve_gib * GIB) + projected_memory
    )
    projected_disk = (
        active_event_bytes + task.event_bytes
    ) * SLICE_DISK_EVENT_MULTIPLIER
    disk_ok = (
        shutil.disk_usage(output_folder).free
        >= int(config.disk_reserve_gib * GIB) + projected_disk
    )
    return memory_ok and disk_ok


def _available_memory_bytes() -> int:
    if os.name == "nt":
        return _windows_available_memory_bytes()
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0
    return int(page_size * available_pages)


def _windows_available_memory_bytes() -> int:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.ullAvailPhys)


def _resource_failure_message(task: SliceTask, context: SliceExecutionConfig) -> str:
    available_gib = _available_memory_bytes() / GIB
    free_disk_gib = shutil.disk_usage(context.output_folder).free / GIB
    return (
        f"Cannot admit slice ending at {task.time_slice}: "
        f"available RAM={available_gib:.1f} GiB with "
        f"reserve={context.config.memory_reserve_gib:.1f} GiB; "
        f"free disk={free_disk_gib:.1f} GiB with "
        f"reserve={context.config.disk_reserve_gib:.1f} GiB."
    )


def _write_slice_manifest(result: SliceResult, output_folder: Path) -> None:
    manifest_folder = (
        output_folder / "temp_files" / SLICE_MANIFEST_DIRNAME / str(result.time_slice)
    )
    manifest_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_folder / "manifest.json"
    partial_path = manifest_folder / ".manifest.json.partial"
    payload = {
        "time_slice": result.time_slice,
        "event_count": result.event_count,
        "artifacts": [str(path) for path in result.artifacts],
    }
    partial_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial_path.replace(manifest_path)
    result.artifacts.append(manifest_path)


def _validate_slice_result(result: SliceResult) -> None:
    missing = [path for path in result.artifacts if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Slice completed without all declared artifacts: "
            + ", ".join(str(path) for path in missing)
        )


def run_batch(config: PeakLocConfig) -> list[RecordingResult]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = Path(config.input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}. Set PEAKLOC_INPUT_FOLDER "
            "or provide input_folder in the JSON config."
        )

    results = []
    input_files = find_recording_files(folder, recursive=config.recursive_input)
    for filename in natsorted(input_files):
        logger.info("Processing {}", filename)
        recording = process_recording(filename, config, run_timestamp)
        layout = ArtifactLayout.from_run_directory(recording.output_folder)
        layout.ensure_directories()
        settings_path = (
            layout.debug_reports_dir / f"peakloc_settings_{run_timestamp}.json"
        )
        write_effective_run_settings(
            config, recording.calibration_metadata, settings_path
        )
        recording.artifacts.append(settings_path)
        write_run_report(recording, config, run_timestamp)
        results.append(recording)
    return results


def _partial_artifact_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial")


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    partial_path = _partial_artifact_path(path)
    partial_path.unlink(missing_ok=True)
    with partial_path.open("wb") as output_file:
        np.save(output_file, array, allow_pickle=False)
    partial_path.replace(path)


def _atomic_save_dict(path: Path, payload: dict) -> None:
    partial_path = _partial_artifact_path(path)
    partial_path.unlink(missing_ok=True)
    save_dict(payload, str(partial_path))
    partial_path.replace(path)


def _atomic_write_structured_array_csv(array: np.ndarray, path: Path) -> None:
    partial_path = _partial_artifact_path(path)
    partial_path.unlink(missing_ok=True)
    write_structured_array_csv(array, partial_path)
    partial_path.replace(path)


def _acquire_parallel_stage(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+", encoding="utf-8")
    if os.name == "nt":
        lock_file.seek(0)
        if not lock_file.read(1):
            lock_file.write("\0")
            lock_file.flush()
        lock_file.seek(0)
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno != errno.EACCES and getattr(error, "winerror", None) not in {
                    32,
                    33,
                }:
                    lock_file.close()
                    raise
                time.sleep(0.1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _release_parallel_stage(lock_file: TextIO | None) -> None:
    if lock_file is None:
        return
    if os.name == "nt":
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def process_time_slice(
    event_slice: np.ndarray,
    time_slice: int,
    filename: Path,
    config: PeakLocConfig,
    calibration: EventCalibration,
    output_folder: Path,
    spatial_mask: SpatialMask | None = None,
    diffuse_flash_intervals: tuple[TimeInterval, ...] = (),
    parallel_stage_lock_path: Path | None = None,
    slice_start: int | None = None,
) -> SliceResult | None:
    events, diffuse_flash_excluded_event_count = exclude_time_intervals(
        event_slice, diffuse_flash_intervals
    )
    if events.size == 0:
        logger.info(
            "No events found in time slice ending at {} for {}", time_slice, filename
        )
        return None

    start_time = time.perf_counter()
    stage_metrics = SliceStageMetrics()
    event_count = int(events.size)
    event_bytes = int(events.nbytes)
    if diffuse_flash_excluded_event_count:
        logger.info(
            "Excluded {} events in diffuse flash intervals before processing",
            diffuse_flash_excluded_event_count,
        )
    temp_files_localization = output_folder / "temp_files"
    joblib_temp_folder = temp_files_localization / JOBLIB_TEMP_DIRNAME / str(time_slice)
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_files_localization.mkdir(parents=True, exist_ok=True)
    joblib_temp_folder.mkdir(parents=True, exist_ok=True)
    worker_count = config.parallel_workers

    if spatial_mask is not None and spatial_mask.is_active:
        target_coords = spatial_mask.target_coords
        support_coords = spatial_mask.support_coords
        if target_coords is None or support_coords is None:
            raise RuntimeError("Active spatial mask has no coordinate arrays")
        logger.info(
            "Using spatial mask with {} target and {} support pixels",
            len(target_coords),
            len(support_coords),
        )
    else:
        min_x = events["x"].min()
        min_y = events["y"].min()
        max_x = events["x"].max()
        max_y = events["y"].max()
        target_coords = generate_coord_lists(min_y, max_y, min_x, max_x)
        support_coords = target_coords

    logger.info("Analyzing the data using {} bounded workers", worker_count)
    logger.info(
        "Converting events to dictionaries; elapsed time: {:.2f} seconds",
        time.perf_counter() - start_time,
    )
    stage_start = time.perf_counter()
    dict_events, max_len = array_to_polarity_map(events, support_coords)
    if max_len == 0:
        logger.info("No events fall inside the spatial support mask for this slice")
        del dict_events, max_len, target_coords, support_coords, events, event_slice
        _release_and_measure(stage_metrics)
        return None

    events_t_p_dict = (
        array_to_time_map_for_coords(events, support_coords)
        if spatial_mask is not None and spatial_mask.is_active
        else array_to_time_map(events)
    )
    del events, event_slice
    _release_and_measure(stage_metrics)
    stage_metrics.event_index_seconds = time.perf_counter() - stage_start

    logger.info(
        "Creating convolved signals; elapsed time: {:.2f} seconds",
        time.perf_counter() - start_time,
    )
    stage_start = time.perf_counter()
    stage_lock = _acquire_parallel_stage(parallel_stage_lock_path)
    max_len = int(max_len * 2 * (config.convolution_roi_radius * 2 + 1) ** 2)
    times, cumsum, coordinates = create_convolved_signals(
        dict_events,
        target_coords,
        max_len,
        worker_count,
        convolution_roi_radius=config.convolution_roi_radius,
    )
    _release_parallel_stage(stage_lock)
    del dict_events, max_len, target_coords, support_coords
    _release_and_measure(stage_metrics)
    stage_metrics.convolution_seconds = time.perf_counter() - stage_start

    logger.info(
        "Finding peaks; elapsed time: {:.2f} seconds", time.perf_counter() - start_time
    )
    stage_start = time.perf_counter()
    stage_lock = _acquire_parallel_stage(parallel_stage_lock_path)
    peak_list = find_peaks_parallel(
        times,
        cumsum,
        coordinates,
        worker_count,
        prominence=config.prominence,
        interpolation_coefficient=config.interpolation_coefficient,
        cutoff_event_count=config.peak_min_event_count,
        spline_smooth=config.spline_smooth,
        joblib_temp_folder=joblib_temp_folder,
        backend="loky",
    )
    _release_parallel_stage(stage_lock)
    peaks, prominences, on_times, coordinates_peaks = create_peak_lists(peak_list)
    del times, cumsum, coordinates, peak_list
    _release_and_measure(stage_metrics)
    peaks_dict = group_timestamps_by_coordinate(
        coordinates_peaks, peaks, prominences, on_times
    )
    del peaks, prominences, on_times, coordinates_peaks
    stage_metrics.peak_interpolation_seconds = time.perf_counter() - stage_start

    logger.info(
        "Filtering peaks; elapsed time: {:.2f} seconds",
        time.perf_counter() - start_time,
    )
    stage_start = time.perf_counter()
    unique_peaks = find_local_max_peak(
        peaks_dict,
        threshold=config.peak_time_threshold,
        neighbors=config.peak_neighbors,
    )
    del peaks_dict

    unique_peaks_path = (
        temp_files_localization
        / f"unique_peaks_fwhm_{config.dataset_fwhm:g}_prominence_{config.prominence:g}"
        f"_time_slice_{time_slice}.pkl"
    )
    _atomic_save_dict(unique_peaks_path, unique_peaks)
    unique_peak_count = sum(len(values) for values in unique_peaks.values())
    stage_metrics.peak_filter_seconds = time.perf_counter() - stage_start

    logger.info(
        "Generating ROIs; elapsed time: {:.2f} seconds",
        time.perf_counter() - start_time,
    )
    stage_start = time.perf_counter()
    stage_lock = _acquire_parallel_stage(parallel_stage_lock_path)
    temporal_segmentation_qc = None
    if config.temporal_segmentation_enabled:
        temporal_result = generate_temporally_segmented_rois(
            unique_peaks,
            events_t_p_dict,
            roi_radius=config.roi_radius,
            min_x=0,
            min_y=0,
            max_x=config.sensor_width - 1,
            max_y=config.sensor_height - 1,
            slice_start_us=(
                time_slice - config.slice_duration if slice_start is None else slice_start
            ),
            slice_stop_us=time_slice,
            settings=temporal_settings_from_config(config),
        )
        rois = temporal_result.rois
        temporal_segmentation_qc = temporal_result.qc
    else:
        rois = generate_rois(
            unique_peaks,
            events_t_p_dict,
            roi_rad=config.roi_radius,
            min_x=0,
            min_y=0,
            num_cores=worker_count,
            max_x=config.sensor_width - 1,
            max_y=config.sensor_height - 1,
            polarity_time_gate_us=config.polarity_time_gate_us,
        )
    _release_parallel_stage(stage_lock)
    del events_t_p_dict, unique_peaks
    _release_and_measure(stage_metrics)
    stage_metrics.roi_generation_seconds = time.perf_counter() - stage_start

    logger.info(
        "Performing localization; elapsed time: {:.2f} seconds",
        time.perf_counter() - start_time,
    )
    stage_start = time.perf_counter()
    stage_lock = _acquire_parallel_stage(parallel_stage_lock_path)
    localization_tables = localize_rois_with_attempts(
        rois,
        config,
        calibration,
        joblib_temp_folder=joblib_temp_folder,
    )
    _release_parallel_stage(stage_lock)
    attempted_localizations = localization_tables.attempted
    localizations = localization_tables.filtered
    localization_qc = localization_tables.qc_table
    stage_metrics.localization_seconds = time.perf_counter() - stage_start

    logger.info(
        "Finished; total elapsed time: {:.2f} seconds", time.perf_counter() - start_time
    )
    stage_start = time.perf_counter()
    attempted_localizations_path = (
        temp_files_localization
        / f"attempted_localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    localizations_path = (
        temp_files_localization
        / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    rois_path = (
        temp_files_localization / f"rois_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    localization_qc_path = (
        temp_files_localization
        / f"localization_qc_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    localization_qc_csv_path = localization_qc_path.with_suffix(".csv")
    temporal_segmentation_qc_path = (
        temp_files_localization
        / f"temporal_segmentation_qc_prominence_fwhm_{config.dataset_fwhm:g}"
        f"_prominence_{config.prominence:g}_time_slice_{time_slice}.npy"
    )
    temporal_segmentation_qc_csv_path = temporal_segmentation_qc_path.with_suffix(
        ".csv"
    )
    _atomic_save_array(attempted_localizations_path, attempted_localizations)
    _atomic_save_array(localizations_path, localizations)
    _atomic_save_array(localization_qc_path, localization_qc)
    _atomic_write_structured_array_csv(localization_qc, localization_qc_csv_path)
    _atomic_save_array(rois_path, rois)
    if temporal_segmentation_qc is not None:
        _atomic_save_array(temporal_segmentation_qc_path, temporal_segmentation_qc)
        _atomic_write_structured_array_csv(
            temporal_segmentation_qc,
            temporal_segmentation_qc_csv_path,
        )
    stage_metrics.artifact_write_seconds = time.perf_counter() - stage_start

    fit_qc = summarize_fit_qc(
        attempted_localizations,
        roi_count=len(rois),
        filtered_localization_count=len(localizations),
    )
    rejected_localization_count = fit_qc["rejected_localization_count"]
    if not isinstance(rejected_localization_count, int):
        rejected_localization_count = 0
    slice_result = SliceResult(
        time_slice=time_slice,
        event_count=event_count,
        unique_peak_count=unique_peak_count,
        roi_count=len(rois),
        localization_count=len(localizations),
        elapsed_seconds=time.perf_counter() - start_time,
        fit_success_fraction=fit_qc["fit_success_fraction"],
        median_uncertainty_px=fit_qc["median_uncertainty_px"],
        median_nll_per_event=fit_qc["median_nll_per_event"],
        hot_pixel_fraction=fit_qc["hot_pixel_fraction"],
        rejected_localization_count=rejected_localization_count,
        diffuse_flash_excluded_event_count=diffuse_flash_excluded_event_count,
        event_bytes=event_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        temp_disk_bytes=_directory_size_bytes(temp_files_localization),
        process_id=os.getpid(),
        stage_metrics=stage_metrics,
        artifacts=[
            unique_peaks_path,
            attempted_localizations_path,
            localizations_path,
            localization_qc_path,
            localization_qc_csv_path,
            rois_path,
        ],
    )
    if temporal_segmentation_qc is not None:
        slice_result.artifacts.extend(
            [
                temporal_segmentation_qc_path,
                temporal_segmentation_qc_csv_path,
            ]
        )
    del temporal_segmentation_qc
    del (
        localization_tables,
        attempted_localizations,
        localizations,
        localization_qc,
        rois,
    )
    _release_and_measure(stage_metrics)
    slice_result.elapsed_seconds = time.perf_counter() - start_time
    return slice_result


def process_recording(
    filename: Path, config: PeakLocConfig, run_timestamp: str
) -> RecordingResult:
    recording_start = time.time()
    run_directory = filename.with_suffix("") / run_timestamp
    layout = ArtifactLayout.from_run_directory(run_directory)
    layout.ensure_directories()
    temp_files_localization = layout.temp_files_dir
    clear_stale_slice_artifacts(temp_files_localization)
    event_store = open_event_store(filename, config, temp_files_localization)
    events = event_store.events
    try:
        time_min, time_max = event_time_bounds(events)
        recording = RecordingResult(
            input_file=filename,
            output_folder=run_directory,
            event_count=int(events.size),
            time_min=time_min,
            time_max=time_max,
        )

        if events.size == 0:
            logger.info("No events found for {}", filename)
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        calibration = load_calibration(
            config.calibration_path,
            config.sensor_shape,
            allow_uncalibrated=config.allow_uncalibrated,
        )
        recording.calibration_metadata = calibration_to_metadata(calibration)

        if time_max is None:
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        flash_detection = detect_recording_diffuse_flashes(
            events,
            config,
            timestamps_monotonic=event_store.timestamps_monotonic,
        )
        recording.diffuse_flash_intervals = flash_detection.intervals
        recording.diffuse_flash_excluded_event_count = (
            flash_detection.excluded_event_count
        )
        recording.diffuse_flash_transition_bin_count = (
            flash_detection.transition_bin_count
        )
        recording.artifacts.append(
            save_diffuse_flash_intervals(recording, config, run_timestamp)
        )
        if flash_detection.intervals:
            logger.info(
                "Excluding {} diffuse flash intervals containing {} events",
                len(flash_detection.intervals),
                flash_detection.excluded_event_count,
            )
        time_slices = build_time_slices(config, time_max)
        if len(time_slices) == 0:
            logger.info("No time slices to process for {}", filename)
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        spatial_mask = build_recording_spatial_mask(
            events,
            config,
            time_min=time_min,
            time_max=time_max,
            timestamps_monotonic=event_store.timestamps_monotonic,
            excluded_intervals=flash_detection.intervals,
        )
        recording.artifacts.extend(
            save_spatial_mask_artifacts(
                recording,
                spatial_mask,
                config,
                run_timestamp,
            )
        )
        if spatial_mask.is_active:
            logger.info(
                "Spatial mask retains {:.1%} target and {:.1%} support coverage",
                spatial_mask.target_coverage,
                spatial_mask.support_coverage,
            )
        else:
            logger.info(
                "Spatial mask fallback: {}",
                spatial_mask.fallback_reason,
            )

        event_qc = None
        if config.qc_enabled:
            event_qc = EventQCAccumulator.create(
                config.sensor_shape,
                expected_event_count=(
                    int(events.size) - flash_detection.excluded_event_count
                ),
                sample_limit=(
                    config.qc_max_events_for_interactive
                    if config.qc_generate_interactive
                    else 0
                ),
            )
            for retained_events in iter_retained_event_spans(
                events, flash_detection.intervals
            ):
                event_qc.add(retained_events)

        active_spatial_mask = spatial_mask if spatial_mask.is_active else None
        _configure_numerical_threads(config.parallel_workers)
        if config.effective_concurrent_slices > 1 and event_store.timestamps_monotonic:
            event_path = (
                filename if filename.suffix == ".npy" else event_store.cache_path
            )
            if event_path is None:
                raise RuntimeError("Parallel RAW processing requires an event cache")
            tasks = build_slice_tasks(events, time_slices)
            execution_context = SliceExecutionConfig(
                filename=filename,
                event_path=event_path,
                event_format="npy" if filename.suffix == ".npy" else "raw_cache",
                event_count=int(events.size),
                output_folder=layout.debug_dir,
                config=config,
                spatial_mask=active_spatial_mask,
                diffuse_flash_intervals=flash_detection.intervals,
                parallel_stage_lock_path=(
                    temp_files_localization / PARALLEL_STAGE_LOCK_FILENAME
                ),
            )
            recording.slice_results.extend(
                execute_slice_tasks(tasks, execution_context)
            )
            for result in recording.slice_results:
                recording.artifacts.extend(result.artifacts)
        else:
            if (
                config.effective_concurrent_slices > 1
                and not event_store.timestamps_monotonic
            ):
                logger.warning(
                    "Parallel slices require monotonic timestamps; using serial fallback"
                )
            for time_slice, event_slice in iter_event_slices(
                events,
                time_slices,
                timestamps_monotonic=event_store.timestamps_monotonic,
            ):
                try:
                    slice_bounds = (
                        {"slice_start": time_slice.start}
                        if config.slice_end is not None
                        else {}
                    )
                    if flash_detection.intervals:
                        slice_result = process_time_slice(
                            event_slice,
                            time_slice.stop,
                            filename,
                            config,
                            calibration,
                            layout.debug_dir,
                            active_spatial_mask,
                            flash_detection.intervals,
                            **slice_bounds,
                        )
                    else:
                        slice_result = process_time_slice(
                            event_slice,
                            time_slice.stop,
                            filename,
                            config,
                            calibration,
                            layout.debug_dir,
                            active_spatial_mask,
                            **slice_bounds,
                        )
                finally:
                    del event_slice
                    release_unused_memory()
                if slice_result is not None:
                    _write_slice_manifest(slice_result, layout.debug_dir)
                    _validate_slice_result(slice_result)
                    recording.slice_results.append(slice_result)
                    recording.artifacts.extend(slice_result.artifacts)

        recording.slice_results.sort(key=lambda result: result.time_slice)
        recording.artifacts.extend(write_slice_metrics(recording, run_timestamp))

        if not temp_files_localization.is_dir():
            logger.info("No temporary localization folder found for {}", filename)
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        sorted_names = natsorted(
            path.name for path in temp_files_localization.iterdir()
        )
        loc_names = [name for name in sorted_names if name.startswith("localizations")]
        attempted_loc_names = [
            name for name in sorted_names if name.startswith("attempted_localizations")
        ]
        roi_names = [name for name in sorted_names if name.startswith("rois")]
        localization_qc_names = [
            name
            for name in sorted_names
            if name.startswith("localization_qc") and name.endswith(".npy")
        ]
        temporal_segmentation_qc_names = [
            name
            for name in sorted_names
            if name.startswith("temporal_segmentation_qc") and name.endswith(".npy")
        ]
        if not loc_names or not roi_names:
            logger.info("No localization outputs found for {}", filename)
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        localizations_path = (
            layout.debug_arrays_dir
            / f"localizations_prominence_fwhm_{config.dataset_fwhm:g}"
            f"_prominence_{config.prominence:g}.npy"
        )
        attempted_localizations_path = (
            layout.debug_arrays_dir
            / f"attempted_localizations_prominence_fwhm_{config.dataset_fwhm:g}"
            f"_prominence_{config.prominence:g}.npy"
        )
        rois_path = (
            layout.debug_arrays_dir / f"rois_prominence_fwhm_{config.dataset_fwhm:g}"
            f"_prominence_{config.prominence:g}.npy"
        )
        localization_qc_path = (
            layout.debug_arrays_dir
            / f"localization_qc_prominence_fwhm_{config.dataset_fwhm:g}"
            f"_prominence_{config.prominence:g}.npy"
        )
        localization_qc_csv_path = localization_qc_path.with_suffix(".csv")
        temporal_segmentation_qc_path = (
            layout.debug_arrays_dir
            / f"temporal_segmentation_qc_prominence_fwhm_{config.dataset_fwhm:g}"
            f"_prominence_{config.prominence:g}.npy"
        )
        temporal_segmentation_qc_csv_path = temporal_segmentation_qc_path.with_suffix(
            ".csv"
        )

        localizations = concatenate_slice_arrays_to_disk(
            temp_files_localization,
            loc_names,
            localizations_path,
            offset_ids=True,
        )
        attempted_localizations = concatenate_slice_arrays_to_disk(
            temp_files_localization,
            attempted_loc_names,
            attempted_localizations_path,
            offset_ids=True,
        )
        localization_qc = concatenate_slice_arrays_to_disk(
            temp_files_localization,
            localization_qc_names,
            localization_qc_path,
            offset_ids=True,
        )
        temporal_segmentation_qc = concatenate_slice_arrays_to_disk(
            temp_files_localization,
            temporal_segmentation_qc_names,
            temporal_segmentation_qc_path,
            offset_ids=True,
        )
        rois = concatenate_slice_arrays_to_disk(
            temp_files_localization,
            roi_names,
            rois_path,
            offset_ids=False,
        )

        if localizations is None or rois is None:
            logger.info("No localization outputs found for {}", filename)
            recording.elapsed_seconds = time.time() - recording_start
            return recording

        if attempted_localizations is not None:
            recording.artifacts.append(attempted_localizations_path)
        if localization_qc is not None:
            write_structured_array_csv(localization_qc, localization_qc_csv_path)
            recording.artifacts.extend([localization_qc_path, localization_qc_csv_path])
        if temporal_segmentation_qc is not None:
            write_structured_array_csv(
                temporal_segmentation_qc,
                temporal_segmentation_qc_csv_path,
            )
            recording.artifacts.extend(
                [temporal_segmentation_qc_path, temporal_segmentation_qc_csv_path]
            )
        recording.artifacts.extend([localizations_path, rois_path])

        attempted_table = (
            attempted_localizations
            if attempted_localizations is not None
            else np.empty(0, dtype=localizations.dtype)
        )
        qc_table = (
            localization_qc
            if localization_qc is not None
            else np.empty(0, dtype=localization_qc_dtype())
        )
        recording.artifacts.extend(
            save_portable_outputs(
                recording=recording,
                config=config,
                accepted_localizations=localizations,
                attempted_localizations=attempted_table,
                localization_qc=qc_table,
                timestamp=run_timestamp,
            )
        )
        recording.artifacts.extend(
            save_processed_plots(
                localizations,
                layout,
                localizations_path,
                config,
                run_timestamp,
                attempted_localizations,
                localization_qc,
            )
        )
        if config.qc_enabled:
            recording.artifacts.extend(
                save_run_qc_dashboard(
                    recording=recording,
                    config=config,
                    localizations=localizations,
                    attempted_localizations=attempted_table,
                    localization_qc=qc_table,
                    rois=rois,
                    events=None,
                    timestamp=run_timestamp,
                    event_qc=event_qc,
                )
            )

        if config.cleanup_temp_outputs:
            remove_temp_artifacts(recording, temp_files_localization, sorted_names)

        recording.elapsed_seconds = time.time() - recording_start
        return recording
    finally:
        del events
        close_event_store(event_store, remove_cache=config.cleanup_temp_outputs)
        release_unused_memory()


def open_event_store(
    filename: Path,
    config: PeakLocConfig,
    temp_folder: Path,
) -> EventStore:
    """Open a recording without loading the complete event stream into RAM."""
    if filename.suffix == ".raw":
        cache_path = temp_folder / "raw_event_cache.dat"
        return EventStore(
            events=materialize_raw_events(
                filename,
                cache_path,
                max_events=config.max_raw_events,
            ),
            cache_path=cache_path,
            timestamps_monotonic=True,
        )
    if filename.suffix == ".npy":
        events = np.load(filename, mmap_mode="r", allow_pickle=False)
        return EventStore(
            events=events,
            timestamps_monotonic=timestamps_are_monotonic(events),
        )
    raise ValueError(f"Unsupported input file: {filename}")


def close_event_store(event_store: EventStore, *, remove_cache: bool) -> None:
    _close_memory_map(event_store.events)
    if remove_cache and event_store.cache_path is not None:
        event_store.cache_path.unlink(missing_ok=True)


def event_time_bounds(events: np.ndarray) -> tuple[int | None, int | None]:
    if events.size == 0:
        return None, None
    timestamps = events["t"]
    return int(np.min(timestamps)), int(np.max(timestamps))


def detect_recording_diffuse_flashes(
    events: np.ndarray,
    config: PeakLocConfig,
    *,
    timestamps_monotonic: bool,
) -> DiffuseFlashDetection:
    if not config.diffuse_flash_rejection_enabled:
        return DiffuseFlashDetection((), 0, 0)
    if not timestamps_monotonic:
        logger.warning(
            "Diffuse flash interval filtering requires monotonic timestamps; "
            "leaving this recording unchanged"
        )
        return DiffuseFlashDetection((), 0, 0)
    return detect_diffuse_flash_intervals(
        events,
        config.sensor_shape,
        bin_duration_us=config.diffuse_flash_bin_duration_us,
        min_events_per_polarity=config.diffuse_flash_min_events_per_polarity,
        min_active_pixel_fraction=config.diffuse_flash_min_active_pixel_fraction,
        max_gap_us=config.diffuse_flash_max_gap_us,
        padding_us=config.diffuse_flash_padding_us,
    )


def save_diffuse_flash_intervals(
    recording: RecordingResult, config: PeakLocConfig, timestamp: str
) -> Path:
    report_folder = ArtifactLayout.from_run_directory(
        recording.output_folder
    ).debug_reports_dir
    report_folder.mkdir(parents=True, exist_ok=True)
    path = report_folder / f"diffuse_flash_intervals_{timestamp}.json"
    payload = {
        "enabled_in_config": config.diffuse_flash_rejection_enabled,
        "transition_bin_count": recording.diffuse_flash_transition_bin_count,
        "excluded_event_count": recording.diffuse_flash_excluded_event_count,
        "excluded_duration_us": sum(
            interval.duration_us for interval in recording.diffuse_flash_intervals
        ),
        "intervals": [
            {
                "start_us": interval.start_us,
                "stop_us": interval.stop_us,
                "duration_us": interval.duration_us,
            }
            for interval in recording.diffuse_flash_intervals
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def build_recording_spatial_mask(
    events: np.ndarray,
    config: PeakLocConfig,
    *,
    time_min: int | None,
    time_max: int | None,
    timestamps_monotonic: bool,
    excluded_intervals: tuple[TimeInterval, ...] = (),
) -> SpatialMask:
    """Calibrate a sparse processing region without retaining a second event array."""
    if not config.spatial_mask_enabled:
        return disabled_spatial_mask(
            "Spatial masking is disabled in the configuration."
        )
    if time_min is None or time_max is None:
        return disabled_spatial_mask(
            "Recording has no timestamp range for mask calibration."
        )

    sample_start = time_min
    sample_stop = min(
        sample_start + config.spatial_mask_sample_duration_us,
        time_max + 1,
    )
    density, calibration_event_count = accumulate_event_density_in_time_window(
        events,
        config.sensor_shape,
        start_us=sample_start,
        stop_us=sample_stop,
        timestamps_monotonic=timestamps_monotonic,
        excluded_intervals=excluded_intervals,
    )
    try:
        return build_spatial_mask(
            density,
            sample_start_us=sample_start,
            sample_stop_us=sample_stop,
            calibration_event_count=calibration_event_count,
            min_density_quotient=config.spatial_mask_min_density_quotient,
            min_component_pixels=config.spatial_mask_min_component_pixels,
            margin_px=config.spatial_mask_margin_px,
            support_margin_px=max(
                config.roi_radius,
                config.convolution_roi_radius,
            ),
            max_support_coverage=config.spatial_mask_max_support_coverage,
        )
    finally:
        del density
        release_unused_memory()


def save_spatial_mask_artifacts(
    recording: RecordingResult,
    spatial_mask: SpatialMask,
    config: PeakLocConfig,
    timestamp: str,
) -> list[Path]:
    """Save auditable spatial-mask diagnostics without retaining event-density data."""
    report_folder = ArtifactLayout.from_run_directory(
        recording.output_folder
    ).debug_reports_dir
    report_folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        **spatial_mask.metadata(),
        "enabled_in_config": config.spatial_mask_enabled,
        "sample_duration_us": config.spatial_mask_sample_duration_us,
        "min_density_quotient": config.spatial_mask_min_density_quotient,
        "min_component_pixels": config.spatial_mask_min_component_pixels,
        "margin_px": config.spatial_mask_margin_px,
        "support_margin_px": max(
            config.roi_radius,
            config.convolution_roi_radius,
        ),
        "max_support_coverage": config.spatial_mask_max_support_coverage,
    }
    recording.spatial_mask_metadata = metadata
    metadata_path = report_folder / f"spatial_mask_{timestamp}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [metadata_path]
    if not spatial_mask.is_active:
        return artifacts

    target_mask_path = report_folder / f"spatial_mask_target_{timestamp}.npy"
    support_mask_path = report_folder / f"spatial_mask_support_{timestamp}.npy"
    if spatial_mask.target_mask is None or spatial_mask.support_mask is None:
        raise RuntimeError("Active spatial mask has no mask arrays")
    np.save(target_mask_path, spatial_mask.target_mask)
    np.save(support_mask_path, spatial_mask.support_mask)
    artifacts.extend([target_mask_path, support_mask_path])
    return artifacts


def timestamps_are_monotonic(
    events: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> bool:
    """Check ordering in bounded chunks before using fast searchsorted slices."""
    if events.size < 2:
        return True
    timestamps = events["t"]
    previous = int(timestamps[0])
    for start in range(0, events.size, chunk_size):
        chunk = timestamps[start : start + chunk_size]
        if chunk.size == 0:
            continue
        if int(chunk[0]) < previous or np.any(chunk[1:] < chunk[:-1]):
            return False
        previous = int(chunk[-1])
    return True


def iter_event_slices(
    events: np.ndarray,
    time_slices: Iterable[TimeSlice],
    *,
    timestamps_monotonic: bool,
) -> Iterator[tuple[TimeSlice, np.ndarray]]:
    """Yield one time slice at a time, avoiding repeated full-array boolean copies."""
    timestamps = events["t"]
    if not timestamps_monotonic:
        logger.warning(
            "Timestamps are not monotonic; using a bounded-copy slice fallback"
        )
    for time_slice in time_slices:
        if timestamps_monotonic:
            start_index = int(np.searchsorted(timestamps, time_slice.start, side="left"))
            stop_index = int(np.searchsorted(timestamps, time_slice.stop, side="left"))
            yield time_slice, np.asarray(events[start_index:stop_index])
        else:
            mask = (timestamps >= time_slice.start) & (timestamps < time_slice.stop)
            yield time_slice, np.asarray(events[mask])


def concatenate_slice_arrays_to_disk(
    temp_folder: Path,
    slice_names: list[str],
    output_path: Path,
    *,
    offset_ids: bool,
) -> np.ndarray | None:
    """Append slice arrays into a final `.npy` memory map without RAM concatenation."""
    if not slice_names:
        return None

    first_path = temp_folder / slice_names[0]
    first_array = np.load(first_path, mmap_mode="r", allow_pickle=False)
    dtype = first_array.dtype
    _close_memory_map(first_array)

    total_size = 0
    for slice_name in slice_names:
        slice_path = temp_folder / slice_name
        slice_array = np.load(slice_path, mmap_mode="r", allow_pickle=False)
        try:
            if slice_array.dtype != dtype:
                raise ValueError(
                    f"Slice dtype does not match {first_path}: {slice_path}"
                )
            if offset_ids and (
                slice_array.dtype.names is None or "id" not in slice_array.dtype.names
            ):
                raise ValueError(
                    f"Localization file has no structured 'id' field: {slice_path}"
                )
            total_size += int(slice_array.size)
        finally:
            _close_memory_map(slice_array)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    partial_path.unlink(missing_ok=True)
    output_array = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=dtype,
        shape=(total_size,),
    )
    next_id = 0
    start = 0
    try:
        for slice_name in slice_names:
            slice_path = temp_folder / slice_name
            slice_array = np.load(slice_path, mmap_mode="r", allow_pickle=False)
            try:
                stop = start + int(slice_array.size)
                output_array[start:stop] = slice_array
                if offset_ids and slice_array.size:
                    output_array["id"][start:stop] += next_id
                    next_id = int(np.max(output_array["id"][start:stop])) + 1
                start = stop
            finally:
                _close_memory_map(slice_array)
        output_array.flush()
    finally:
        _close_memory_map(output_array)

    partial_path.replace(output_path)
    return np.load(output_path, mmap_mode="r", allow_pickle=False)


def _close_memory_map(array: np.ndarray) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def release_unused_memory() -> None:
    """Collect Python/native allocations and return free glibc arenas between slices."""
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        return


def _release_and_measure(metrics: SliceStageMetrics) -> None:
    started = time.perf_counter()
    release_unused_memory()
    metrics.memory_release_seconds += time.perf_counter() - started


def _peak_rss_bytes() -> int:
    if resource is None:
        return 0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024 if sys.platform.startswith("linux") else 1
    return int(usage.ru_maxrss * scale)


def _directory_size_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


def write_slice_metrics(recording: RecordingResult, timestamp: str) -> list[Path]:
    report_folder = ArtifactLayout.from_run_directory(
        recording.output_folder
    ).debug_reports_dir
    report_folder.mkdir(parents=True, exist_ok=True)
    json_path = report_folder / f"slice_metrics_{timestamp}.json"
    csv_path = report_folder / f"slice_metrics_{timestamp}.csv"
    rows = []
    for result in recording.slice_results:
        row = {
            "time_slice": result.time_slice,
            "event_count": result.event_count,
            "event_bytes": result.event_bytes,
            "unique_peak_count": result.unique_peak_count,
            "roi_count": result.roi_count,
            "localization_count": result.localization_count,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
            "temp_disk_bytes": result.temp_disk_bytes,
            "process_id": result.process_id,
            **asdict(result.stage_metrics),
        }
        rows.append(row)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0]) if rows else ["time_slice"]
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return [json_path, csv_path]


def write_effective_run_settings(
    config: PeakLocConfig,
    calibration_metadata: dict[str, object],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["calibration"] = calibration_metadata
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def write_structured_array_csv(array: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = array.dtype.names
    if field_names is None:
        raise ValueError("CSV output requires a structured NumPy array")
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(field_names)
        for row in array:
            writer.writerow(
                [_csv_scalar(row[field_name]) for field_name in field_names]
            )


def _csv_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def calibration_to_metadata(calibration: EventCalibration) -> dict[str, object]:
    return {
        "calibration_id": calibration.calibration_id,
        "calibrated": calibration.calibrated,
        "sensor_shape": list(calibration.sensor_shape),
        "sensor_model": calibration.sensor_model,
        "pixel_size_nm": calibration.pixel_size_nm,
    }


def summarize_fit_qc(
    localizations: np.ndarray,
    *,
    roi_count: int,
    filtered_localization_count: int | None = None,
) -> dict[str, float | int | None]:
    accepted_count = (
        int(localizations.size)
        if filtered_localization_count is None
        else filtered_localization_count
    )
    if localizations.size == 0 or localizations.dtype.names is None:
        return {
            "fit_success_fraction": None,
            "median_uncertainty_px": None,
            "median_nll_per_event": None,
            "hot_pixel_fraction": None,
            "rejected_localization_count": max(roi_count - accepted_count, 0),
        }
    names = set(localizations.dtype.names)
    fit_success_fraction = None
    median_uncertainty_px = None
    median_nll_per_event = None
    hot_pixel_fraction = None
    if "fit_success" in names:
        fit_success_fraction = float(np.mean(localizations["fit_success"]))
    if {"sigma_x", "sigma_y", "cov_xy"}.issubset(names):
        uncertainty = localization_uncertainty_px(localizations)
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if finite_uncertainty.size:
            median_uncertainty_px = float(np.median(finite_uncertainty))
    elif {"sigma_x", "sigma_y"}.issubset(names):
        uncertainty = np.sqrt(
            np.maximum(localizations["sigma_x"], 0) ** 2
            + np.maximum(localizations["sigma_y"], 0) ** 2
        )
        finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
        if finite_uncertainty.size:
            median_uncertainty_px = float(np.median(finite_uncertainty))
    if "nll_per_event" in names:
        finite_nll = localizations["nll_per_event"][
            np.isfinite(localizations["nll_per_event"])
        ]
        if finite_nll.size:
            median_nll_per_event = float(np.median(finite_nll))
    if {"hot_pixel_count", "valid_pixel_count"}.issubset(names):
        valid_count = int(np.sum(localizations["valid_pixel_count"]))
        if valid_count > 0:
            hot_pixel_fraction = float(
                np.sum(localizations["hot_pixel_count"]) / valid_count
            )
    return {
        "fit_success_fraction": fit_success_fraction,
        "median_uncertainty_px": median_uncertainty_px,
        "median_nll_per_event": median_nll_per_event,
        "hot_pixel_fraction": hot_pixel_fraction,
        "rejected_localization_count": max(roi_count - accepted_count, 0),
    }


def save_processed_plots(
    localizations: np.ndarray,
    layout: ArtifactLayout,
    localizations_path: Path,
    config: PeakLocConfig,
    timestamp: str,
    attempted_localizations: np.ndarray | None = None,
    localization_qc: np.ndarray | None = None,
) -> list[Path]:
    share_figure_dir = layout.share_figures_dir
    debug_figure_dir = layout.debug_qc_dir
    share_figure_dir.mkdir(parents=True, exist_ok=True)
    debug_figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    if config.plot_result and localizations.size == 0:
        logger.info(
            "Skipping accepted-localization plots because no fits were accepted"
        )
    elif config.plot_result:
        roi_fit_figure = plot_rois_from_locs(
            localizations,
            subplotsize=config.plot_subplotsize,
            dataset_FWHM=config.dataset_fwhm,
        )
        if roi_fit_figure is not None:
            roi_fit_path = debug_figure_dir / "roi_fit_overview.png"
            artifacts.extend(
                save_publication_figure(
                    roi_fit_figure,
                    roi_fit_path,
                    dpi=max(config.qc_static_dpi, PREVIEW_DPI),
                    save_vector=config.qc_save_vector,
                )
            )
            plt.close(roi_fit_figure)
            logger.info("Saved ROI fit plot to {}", roi_fit_path)

    if config.plot_result and localizations.size:
        result = save_smlm_visualization(
            localizations,
            localizations_path,
            share_figure_dir,
            optical_pixel_size_nm=config.optical_pixel_size_nm,
            timestamp=timestamp,
            sensor_shape=config.sensor_shape,
            output_stem="smlm_reconstruction",
            crop_to_data=False,
        )
        if result is not None:
            artifacts.extend([result.png_path, result.tiff_path])
            logger.info("Saved SMLM result PNG to {}", result.png_path)
            logger.info("Saved SMLM result TIFF to {}", result.tiff_path)

    if (
        config.qc_enabled
        and attempted_localizations is not None
        and localization_qc is not None
    ):
        review_paths = save_fit_review_diagnostics(
            attempted_localizations,
            localizations,
            localization_qc,
            debug_figure_dir,
            config=config,
            n=getattr(config, "qc_uncertainty_montage_n", 36),
            dpi=getattr(config, "qc_static_dpi", 450),
        )
        artifacts.extend(review_paths)
        logger.info("Saved {} fit-review diagnostic(s)", len(review_paths))

    return artifacts


def remove_temp_artifacts(
    recording: RecordingResult, temp_folder: Path, sorted_names: list[str]
) -> None:
    removed_artifacts = set()
    for loc_file in sorted_names:
        if is_slice_temp_artifact(loc_file):
            temp_artifact = temp_folder / loc_file
            temp_artifact.unlink(missing_ok=True)
            removed_artifacts.add(temp_artifact)
    ephemeral_roots = [
        temp_folder / JOBLIB_TEMP_DIRNAME,
        temp_folder / SLICE_MANIFEST_DIRNAME,
        temp_folder / SLICE_REQUEST_DIRNAME,
    ]
    recording.artifacts = [
        artifact
        for artifact in recording.artifacts
        if artifact not in removed_artifacts
        and not any(
            root == artifact or root in artifact.parents for root in ephemeral_roots
        )
    ]
    for root in ephemeral_roots:
        shutil.rmtree(root, ignore_errors=True)
    (temp_folder / PARALLEL_STAGE_LOCK_FILENAME).unlink(missing_ok=True)


def clear_stale_slice_artifacts(temp_folder: Path) -> None:
    """Remove prior per-slice arrays before aggregating a new recording run."""
    if not temp_folder.is_dir():
        return
    shutil.rmtree(temp_folder / JOBLIB_TEMP_DIRNAME, ignore_errors=True)
    shutil.rmtree(temp_folder / SLICE_MANIFEST_DIRNAME, ignore_errors=True)
    shutil.rmtree(temp_folder / SLICE_REQUEST_DIRNAME, ignore_errors=True)
    (temp_folder / PARALLEL_STAGE_LOCK_FILENAME).unlink(missing_ok=True)
    for path in temp_folder.iterdir():
        if path.is_file() and is_slice_temp_artifact(path.name):
            path.unlink()


def is_slice_temp_artifact(filename: str) -> bool:
    return filename.startswith(SLICE_TEMP_ARTIFACT_PREFIXES)


def write_run_report(
    recording: RecordingResult, config: PeakLocConfig, timestamp: str
) -> Path:
    report_folder = ArtifactLayout.from_run_directory(recording.output_folder).share_dir
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / "run_report.md"
    if report_path not in recording.artifacts:
        recording.artifacts.append(report_path)

    total_unique_peaks = sum(
        result.unique_peak_count for result in recording.slice_results
    )
    total_rois = sum(result.roi_count for result in recording.slice_results)
    total_localizations = sum(
        result.localization_count for result in recording.slice_results
    )

    scheduling_mode = (
        "serial"
        if config.effective_concurrent_slices == 1
        else "bounded lanes with leased parallel stages"
    )

    lines = [
        "# PeakLoc Run Report",
        "",
        f"- Run timestamp: `{timestamp}`",
        f"- Input file: `{recording.input_file}`",
        f"- Output folder: `{recording.output_folder}`",
        f"- Input events: `{recording.event_count}`",
        "- Diffuse flash intervals excluded before processing: "
        f"`{len(recording.diffuse_flash_intervals)}`",
        "- Events excluded with diffuse flash intervals: "
        f"`{recording.diffuse_flash_excluded_event_count}`",
        "- Events retained for processing: "
        f"`{recording.event_count - recording.diffuse_flash_excluded_event_count}`",
        f"- Event time range: `{recording.time_min}` to `{recording.time_max}`",
        f"- Processed slices: `{len(recording.slice_results)}`",
        f"- Total unique peaks: `{total_unique_peaks}`",
        f"- Total ROIs: `{total_rois}`",
        f"- Total localizations: `{total_localizations}`",
        f"- Elapsed time: `{recording.elapsed_seconds:.2f} s`",
        f"- Peak interpolation min events: `{config.peak_min_event_count}`",
        f"- Slice scheduling: `{scheduling_mode}`",
        f"- Concurrent slice lanes: `{config.effective_concurrent_slices}`",
        f"- Resolved CPU worker budget: `{config.resolved_cpu_worker_budget}`",
        f"- Workers per leased parallel stage: `{config.parallel_workers}`",
        f"- Memory reserve: `{config.memory_reserve_gib:g} GiB`",
        f"- Disk reserve: `{config.disk_reserve_gib:g} GiB`",
        f"- Calibration ID: `{recording.calibration_metadata.get('calibration_id')}`",
        f"- Calibrated background: `{recording.calibration_metadata.get('calibrated')}`",
        *_spatial_mask_report_lines(recording),
        "",
        *_scientific_validation_lines(recording, config),
        "",
        "## Settings",
        "",
        "```json",
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        "```",
        "",
        "## Slice Results",
        "",
    ]

    if recording.slice_results:
        lines.extend(
            [
                "| Time slice | Events | Flash events excluded | Unique peaks | ROIs | "
                "Localizations | Success | Unc. px | NLL/event | Hot px | Rejected | Seconds |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in recording.slice_results:
            lines.append(
                f"| {result.time_slice} | {result.event_count} | "
                f"{result.diffuse_flash_excluded_event_count} | "
                f"{result.unique_peak_count} | "
                f"{result.roi_count} | "
                f"{result.localization_count} | "
                f"{_format_optional_float(result.fit_success_fraction)} | "
                f"{_format_optional_float(result.median_uncertainty_px)} | "
                f"{_format_optional_float(result.median_nll_per_event)} | "
                f"{_format_optional_float(result.hot_pixel_fraction)} | "
                f"{result.rejected_localization_count} | {result.elapsed_seconds:.2f} |"
            )
    else:
        lines.append("No time slices produced localizations.")

    lines.extend(["", "## Slice Stage Metrics", ""])
    if recording.slice_results:
        lines.extend(
            [
                "| Time slice | Index | Convolution | Peaks | Filter | ROIs | Fit | "
                "Write | Release | Peak RSS GiB | Temp GiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: |",
            ]
        )
        for result in recording.slice_results:
            metrics = result.stage_metrics
            lines.append(
                f"| {result.time_slice} | {metrics.event_index_seconds:.2f} | "
                f"{metrics.convolution_seconds:.2f} | "
                f"{metrics.peak_interpolation_seconds:.2f} | "
                f"{metrics.peak_filter_seconds:.2f} | "
                f"{metrics.roi_generation_seconds:.2f} | "
                f"{metrics.localization_seconds:.2f} | "
                f"{metrics.artifact_write_seconds:.2f} | "
                f"{metrics.memory_release_seconds:.2f} | "
                f"{result.peak_rss_bytes / 2**30:.2f} | "
                f"{result.temp_disk_bytes / 2**30:.2f} |"
            )
    else:
        lines.append("No slice metrics were recorded.")

    lines.extend(["", "## Artifacts", ""])
    if recording.artifacts:
        for artifact in recording.artifacts:
            lines.append(f"- `{artifact}`")
    else:
        lines.append("No output artifacts were generated.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved run report to {}", report_path)
    return report_path


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def _spatial_mask_report_lines(recording: RecordingResult) -> list[str]:
    metadata = recording.spatial_mask_metadata
    if not metadata:
        return []
    if not metadata.get("active"):
        return [
            "- Spatial processing mask: `inactive`",
            f"- Spatial mask fallback: `{metadata.get('fallback_reason')}`",
        ]
    return [
        "- Spatial processing mask: `active`",
        f"- Spatial target coverage: `{_format_json_float(metadata.get('target_coverage'))}`",
        f"- Spatial support coverage: `{_format_json_float(metadata.get('support_coverage'))}`",
        f"- Spatial calibration events: `{metadata.get('calibration_event_count')}`",
        f"- Spatial mean density: `{_format_json_float(metadata.get('mean_events_per_pixel'))}` events/pixel",
        f"- Spatial density quotient: `{_format_json_float(metadata.get('min_density_quotient'))}`",
        f"- Spatial seed threshold: `{_format_json_float(metadata.get('seed_threshold_events'))}` events/pixel",
    ]


def _scientific_validation_lines(
    recording: RecordingResult, config: PeakLocConfig
) -> list[str]:
    qc_summary = _load_named_json_artifact(recording, "run_summary.json")
    frc_summary = _load_named_json_artifact(recording, "frc_summary.json")
    qc_index = _find_artifact(recording, "index.html")
    preflight_report = _find_artifact(recording, "preflight_report.md")
    preflight_status = _preflight_status(preflight_report)
    warnings = _validation_warnings(qc_summary, config)

    lines = [
        "## Scientific Validation",
        "",
        f"- Preflight status: `{preflight_status}`",
        f"- Calibration status: `{_calibration_status(recording)}`",
    ]
    if qc_summary:
        detection_funnel = _dict_field(qc_summary, "detection_funnel")
        lines.extend(
            [
                f"- Events loaded: `{detection_funnel.get('events_loaded', 'n/a')}`",
                f"- Peak candidates: `{detection_funnel.get('peak_candidates', 'n/a')}`",
                f"- ROIs generated: `{detection_funnel.get('rois_generated', 'n/a')}`",
                f"- Attempted fits: `{qc_summary.get('attempted_fit_count', 'n/a')}`",
                f"- Accepted fits: `{qc_summary.get('accepted_from_qc_count', 'n/a')}`",
                f"- Median uncertainty: `{_format_json_float(qc_summary.get('median_uncertainty_px'))} px / "
                f"{_format_json_float(qc_summary.get('median_uncertainty_nm'))} nm`",
                f"- 90th percentile uncertainty: `{_format_json_float(qc_summary.get('p90_uncertainty_px'))} px / "
                f"{_format_json_float(qc_summary.get('p90_uncertainty_nm'))} nm`",
            ]
        )
        rejection_reasons = _dict_field(qc_summary, "rejection_reasons")
        if rejection_reasons:
            reason_text = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(rejection_reasons.items())
            )
            lines.append(f"- Rejection reasons: `{reason_text}`")
        else:
            lines.append("- Rejection reasons: `none recorded`")
    else:
        lines.append("- Collaborator summary: `not available`")

    if frc_summary:
        lines.append(
            f"- FRC resolution: `{_format_json_float(frc_summary.get('resolution_nm'))} nm`"
        )
        if frc_summary.get("warning"):
            warnings.append(f"FRC warning: {frc_summary['warning']}")
    else:
        lines.append("- FRC resolution: `not available`")

    if qc_index is not None:
        lines.append(f"- Collaborator index: `{qc_index}`")
    lines.extend(["", "### Warnings Requiring Attention", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    return lines


def _validation_warnings(
    qc_summary: dict[str, Any] | None, config: PeakLocConfig
) -> list[str]:
    warnings = []
    if qc_summary:
        raw_warnings = qc_summary.get("warnings", [])
        if isinstance(raw_warnings, list):
            warnings.extend(str(warning) for warning in raw_warnings)
    if config.background_mode == "local_only" and config.calibration_path is None:
        warnings.append(
            "background_mode=local_only and calibration_path=None. This is acceptable "
            "for exploratory tuning, but publication-grade real-data analysis should "
            "use dark and laser-on blank calibration maps or explicitly justify "
            "local-only background."
        )
    return warnings


def _calibration_status(recording: RecordingResult) -> str:
    calibration_id = recording.calibration_metadata.get("calibration_id")
    calibrated = recording.calibration_metadata.get("calibrated")
    return f"calibration_id={calibration_id}, calibrated={calibrated}"


def _preflight_status(preflight_report: Path | None) -> str:
    if preflight_report is None or not preflight_report.is_file():
        return "not available"
    text = preflight_report.read_text(encoding="utf-8")
    if "- Status: `passed`" in text:
        return "passed"
    if "- Status: `failed`" in text:
        return "failed"
    return "available"


def _load_named_json_artifact(
    recording: RecordingResult, filename: str
) -> dict[str, Any] | None:
    artifact = _find_artifact(recording, filename)
    if artifact is None or not artifact.is_file():
        return None
    return json.loads(artifact.read_text(encoding="utf-8"))


def _dict_field(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name, {})
    return value if isinstance(value, dict) else {}


def _find_artifact(recording: RecordingResult, filename: str) -> Path | None:
    for artifact in recording.artifacts:
        artifact_path = Path(artifact)
        if artifact_path.name == filename:
            return artifact_path
    return None


def _format_json_float(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.3g}"
    return str(value)
