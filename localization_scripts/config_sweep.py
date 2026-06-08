from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from localization_scripts.localization_fitting import localization_qc_dtype
from localization_scripts.pipeline_config import PeakLocConfig
from localization_scripts.plot_style import PLOT_COLORS, PREVIEW_DPI, SEQUENTIAL_CMAP
from localization_scripts.preflight import run_preflight, write_preflight_report


Runner = Callable[[PeakLocConfig], list[Any]]


@dataclass(frozen=True)
class SweepRun:
    index: int
    parameters: dict[str, Any]
    config: PeakLocConfig


def load_sweep_spec(path: str | Path) -> dict[str, tuple[Any, ...]]:
    sweep_path = Path(path)
    with sweep_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a JSON object")

    allowed_fields = {field.name for field in fields(PeakLocConfig)}
    invalid_fields = sorted(set(payload) - allowed_fields)
    if invalid_fields:
        raise ValueError("Unknown sweep config field(s): " + ", ".join(invalid_fields))

    spec: dict[str, tuple[Any, ...]] = {}
    for field_name, values in payload.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep field {field_name!r} must be a non-empty list")
        spec[field_name] = tuple(values)
    return spec


def iter_sweep_runs(
    base_config: PeakLocConfig, sweep_spec: Mapping[str, Iterable[Any]]
) -> list[SweepRun]:
    field_names = list(sweep_spec)
    value_lists = [tuple(sweep_spec[field_name]) for field_name in field_names]
    runs = []
    for index, values in enumerate(itertools.product(*value_lists)):
        parameters = dict(zip(field_names, values))
        config = PeakLocConfig(**{**base_config.to_dict(), **parameters})
        config.validate()
        runs.append(SweepRun(index=index, parameters=parameters, config=config))
    return runs


def run_config_sweep(
    base_config: PeakLocConfig,
    sweep_path: str | Path,
    *,
    output_dir: str | Path = "sweep",
    runner: Runner | None = None,
    preflight: bool = False,
    strict_preflight: bool = False,
    truth: np.ndarray | None = None,
) -> list[Path]:
    if runner is None:
        from localization_scripts.pipeline_runner import run_batch

        runner = run_batch

    sweep_spec = load_sweep_spec(sweep_path)
    runs = iter_sweep_runs(base_config, sweep_spec)
    sweep_output_dir = Path(output_dir)
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sweep_run in runs:
        if preflight or strict_preflight:
            report = run_preflight(sweep_run.config, strict_mode=strict_preflight)
            report_path = sweep_output_dir / f"preflight_{sweep_run.index:04d}.md"
            write_preflight_report(report, report_path)
            if report.has_errors:
                raise ValueError(
                    f"Sweep config {sweep_run.index} failed preflight: {report_path}"
                )

        results = runner(sweep_run.config)
        row = {
            "sweep_index": sweep_run.index,
            **sweep_run.parameters,
            **_metrics_from_results(results, truth=truth),
        }
        rows.append(row)

    artifacts = write_sweep_outputs(rows, sweep_output_dir)
    return artifacts


def write_sweep_outputs(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    csv_path = output_dir / "sweep_results.csv"
    json_path = output_dir / "sweep_results.json"
    _write_rows_csv(rows, csv_path)
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = [csv_path, json_path]
    artifacts.append(
        _save_pareto_plot(rows, output_dir / "pareto_localizations_vs_uncertainty.png")
    )
    artifacts.append(
        _save_rejection_heatmap(rows, output_dir / "rejection_reason_heatmap.png")
    )
    artifacts.append(
        _write_parameter_effects_html(rows, output_dir / "parameter_effects.html")
    )
    return artifacts


def _metrics_from_results(
    results: list[Any], *, truth: np.ndarray | None
) -> dict[str, Any]:
    event_count = int(
        sum(getattr(recording, "event_count", 0) for recording in results)
    )
    elapsed_seconds = float(
        sum(getattr(recording, "elapsed_seconds", 0.0) for recording in results)
    )
    slice_results = [
        slice_result
        for recording in results
        for slice_result in getattr(recording, "slice_results", [])
    ]
    unique_peaks = int(
        sum(getattr(result, "unique_peak_count", 0) for result in slice_results)
    )
    rois = int(sum(getattr(result, "roi_count", 0) for result in slice_results))
    accepted_from_slices = int(
        sum(getattr(result, "localization_count", 0) for result in slice_results)
    )
    rejected_from_slices = int(
        sum(
            getattr(result, "rejected_localization_count", 0)
            for result in slice_results
        )
    )

    qc_table = _load_qc_from_artifacts(results)
    localizations = _load_localizations_from_artifacts(results)
    if qc_table.size:
        attempted_fits = int(qc_table.size)
        accepted_fits = int(np.count_nonzero(qc_table["accepted"]))
        uncertainty_px = np.asarray(qc_table["uncertainty_px"], dtype=np.float64)
        uncertainty_nm = np.asarray(qc_table["uncertainty_nm"], dtype=np.float64)
        nll_per_event = np.asarray(qc_table["nll_per_event"], dtype=np.float64)
        event_counts_pos = np.asarray(qc_table["E_total"], dtype=np.float64)
        event_counts_neg = np.asarray(qc_table["E_total_n"], dtype=np.float64)
        rejection_counts = Counter(
            str(reason) for reason in qc_table["primary_rejection_reason"]
        )
    else:
        attempted_fits = accepted_from_slices + rejected_from_slices
        accepted_fits = accepted_from_slices
        uncertainty_px = np.asarray([], dtype=np.float64)
        uncertainty_nm = np.asarray([], dtype=np.float64)
        nll_per_event = np.asarray([], dtype=np.float64)
        event_counts_pos = np.asarray([], dtype=np.float64)
        event_counts_neg = np.asarray([], dtype=np.float64)
        rejection_counts = Counter()

    metrics: dict[str, Any] = {
        "events_processed": event_count,
        "peak_candidates": unique_peaks,
        "unique_peaks": unique_peaks,
        "rois": rois,
        "attempted_fits": attempted_fits,
        "accepted_localizations": accepted_fits,
        "median_uncertainty_px": _nan_percentile(uncertainty_px, 50),
        "p90_uncertainty_px": _nan_percentile(uncertainty_px, 90),
        "median_uncertainty_nm": _nan_percentile(uncertainty_nm, 50),
        "p90_uncertainty_nm": _nan_percentile(uncertainty_nm, 90),
        "median_nll_per_event": _nan_percentile(nll_per_event, 50),
        "median_events_pos": _nan_percentile(event_counts_pos, 50),
        "median_events_neg": _nan_percentile(event_counts_neg, 50),
        "runtime_seconds": elapsed_seconds,
    }
    for reason, count in rejection_counts.items():
        if reason == "accepted":
            continue
        metrics[f"fraction_rejected_{reason}"] = count / max(attempted_fits, 1)
    if truth is not None and localizations.size:
        metrics.update(compute_synthetic_truth_metrics(localizations, truth))
    return metrics


def compute_synthetic_truth_metrics(
    localizations: np.ndarray,
    truth: np.ndarray,
    *,
    max_distance_px: float = 2.0,
) -> dict[str, Any]:
    if localizations.size == 0 or truth.size == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "false_discovery_rate": 0.0,
            "median_spatial_error_px": None,
        }
    if not _has_fields(localizations, {"x", "y"}) or not _has_fields(truth, {"x", "y"}):
        raise ValueError("Truth and localization arrays must have x and y fields")

    loc_xy = np.column_stack([localizations["x"], localizations["y"]])
    truth_xy = np.column_stack([truth["x"], truth["y"]])
    distances = np.linalg.norm(loc_xy[:, None, :] - truth_xy[None, :, :], axis=2)
    matches: list[float] = []
    used_locs: set[int] = set()
    used_truth: set[int] = set()
    for flat_index in np.argsort(distances, axis=None):
        loc_index, truth_index = np.unravel_index(flat_index, distances.shape)
        distance = float(distances[loc_index, truth_index])
        if distance > max_distance_px:
            break
        if loc_index in used_locs or truth_index in used_truth:
            continue
        used_locs.add(int(loc_index))
        used_truth.add(int(truth_index))
        matches.append(distance)

    true_positives = len(matches)
    precision = true_positives / max(localizations.size, 1)
    recall = true_positives / max(truth.size, 1)
    return {
        "recall": recall,
        "precision": precision,
        "false_discovery_rate": 1.0 - precision,
        "median_spatial_error_px": None if not matches else float(np.median(matches)),
        "p95_spatial_error_px": None
        if not matches
        else float(np.percentile(matches, 95)),
    }


def _load_qc_from_artifacts(results: list[Any]) -> np.ndarray:
    paths = _artifact_paths(results, prefix="localization_qc", suffix=".npy")
    arrays = _load_arrays(_prefer_full_outputs(paths))
    if not arrays:
        return np.empty(0, dtype=localization_qc_dtype())
    return np.concatenate(arrays)


def _load_localizations_from_artifacts(results: list[Any]) -> np.ndarray:
    paths = [
        path
        for path in _artifact_paths(results, prefix="localizations", suffix=".npy")
        if not path.name.startswith("localization_qc")
    ]
    arrays = _load_arrays(_prefer_full_outputs(paths))
    if not arrays:
        return np.asarray([])
    return np.concatenate(arrays)


def _artifact_paths(results: list[Any], *, prefix: str, suffix: str) -> list[Path]:
    return [
        Path(artifact)
        for recording in results
        for artifact in getattr(recording, "artifacts", [])
        if Path(artifact).name.startswith(prefix) and Path(artifact).suffix == suffix
    ]


def _prefer_full_outputs(paths: list[Path]) -> list[Path]:
    full_outputs = [path for path in paths if path.parent.name != "temp_files"]
    return full_outputs or paths


def _load_arrays(paths: list[Path]) -> list[np.ndarray]:
    arrays = []
    for path in paths:
        if path.is_file():
            arrays.append(np.load(path, allow_pickle=False))
    return arrays


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    field_names = sorted({field_name for row in rows for field_name in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _save_pareto_plot(rows: list[dict[str, Any]], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if rows:
        x = np.asarray(
            [row.get("accepted_localizations", 0) for row in rows], dtype=np.float64
        )
        y = np.asarray(
            [
                row.get("median_uncertainty_nm")
                if row.get("median_uncertainty_nm") is not None
                else row.get("median_uncertainty_px", np.nan)
                for row in rows
            ],
            dtype=np.float64,
        )
        axis.scatter(x, y, color=PLOT_COLORS["blue"])
    axis.set_title("Pareto: localizations vs uncertainty")
    axis.set_xlabel("accepted localizations")
    axis.set_ylabel("median uncertainty")
    fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_rejection_heatmap(rows: list[dict[str, Any]], path: Path) -> Path:
    reason_fields = sorted(
        field_name
        for row in rows
        for field_name in row
        if field_name.startswith("fraction_rejected_")
    )
    fig, axis = plt.subplots(
        figsize=(max(5, len(reason_fields)), 4), constrained_layout=True
    )
    if rows and reason_fields:
        matrix = np.asarray(
            [[row.get(field, 0.0) for field in reason_fields] for row in rows]
        )
        image = axis.imshow(matrix, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0, vmax=1)
        axis.set_xticks(
            np.arange(len(reason_fields)),
            [field.replace("fraction_rejected_", "") for field in reason_fields],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(rows)), [str(row["sweep_index"]) for row in rows])
        plt.colorbar(image, ax=axis, fraction=0.046, label="fraction")
    else:
        axis.text(
            0.5,
            0.5,
            "No rejection data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    axis.set_title("Rejection reason heatmap")
    fig.savefig(path, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_parameter_effects_html(rows: list[dict[str, Any]], path: Path) -> Path:
    headers = sorted({field_name for row in rows for field_name in row})
    body_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{row.get(header, '')}</td>" for header in headers)
        + "</tr>"
        for row in rows
    )
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>PeakLoc Sweep</title></head>
<body>
<h1>PeakLoc Parameter Sweep</h1>
<table border="1">
<thead><tr>{header_html}</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def _nan_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _has_fields(array: np.ndarray, names: set[str]) -> bool:
    return names.issubset(array.dtype.names or ())
