from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RoiCalibration:
    bg_pos: np.ndarray
    bg_neg: np.ndarray
    valid_mask: np.ndarray
    hot_pixel_mask: np.ndarray

    @property
    def hot_pixel_count(self) -> int:
        return int(np.count_nonzero(self.hot_pixel_mask))

    @property
    def valid_pixel_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))


@dataclass(frozen=True)
class EventCalibration:
    dark_rate_pos: np.ndarray
    dark_rate_neg: np.ndarray
    blank_rate_pos: np.ndarray
    blank_rate_neg: np.ndarray
    hot_pixel_mask: np.ndarray
    valid_pixel_mask: np.ndarray
    pixel_size_nm: float | None = None
    sensor_model: str = "unknown"
    calibration_id: str = "none"
    calibrated: bool = False
    metadata: dict[str, Any] | None = None

    @property
    def sensor_shape(self) -> tuple[int, int]:
        return int(self.dark_rate_pos.shape[0]), int(self.dark_rate_pos.shape[1])

    def validate_sensor_shape(self, sensor_shape: tuple[int, int]) -> None:
        if self.sensor_shape != sensor_shape:
            raise ValueError(
                f"Calibration sensor shape {self.sensor_shape} does not match "
                f"recording sensor shape {sensor_shape}"
            )

    def get_roi_background(
        self,
        global_y0: int,
        global_x0: int,
        roi_radius: int,
        dt_pos_s: float,
        dt_neg_s: float,
    ) -> RoiCalibration:
        roi_side = roi_radius * 2 + 1
        y1 = global_y0 + roi_side
        x1 = global_x0 + roi_side
        if global_y0 < 0 or global_x0 < 0 or y1 > self.sensor_shape[0]:
            raise ValueError("ROI y range is outside the calibration map")
        if x1 > self.sensor_shape[1]:
            raise ValueError("ROI x range is outside the calibration map")

        bg_rate_pos = _background_rate(self.dark_rate_pos, self.blank_rate_pos)
        bg_rate_neg = _background_rate(self.dark_rate_neg, self.blank_rate_neg)
        roi_slice = np.s_[global_y0:y1, global_x0:x1]
        bg_pos = dt_pos_s * bg_rate_pos[roi_slice]
        bg_neg = dt_neg_s * bg_rate_neg[roi_slice]
        hot_pixels = self.hot_pixel_mask[roi_slice]
        valid_mask = self.valid_pixel_mask[roi_slice] & ~hot_pixels
        return RoiCalibration(
            bg_pos=np.asarray(bg_pos, dtype=np.float64),
            bg_neg=np.asarray(bg_neg, dtype=np.float64),
            valid_mask=np.asarray(valid_mask, dtype=bool),
            hot_pixel_mask=np.asarray(hot_pixels, dtype=bool),
        )


def load_calibration(
    calibration_path: str | Path | None,
    sensor_shape: tuple[int, int],
    *,
    allow_uncalibrated: bool,
) -> EventCalibration:
    if calibration_path is None:
        if not allow_uncalibrated:
            raise ValueError("Calibration path is required for calibrated fitting")
        return NullCalibration(sensor_shape)

    path = Path(calibration_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    calibration = _load_npz_calibration(path)
    calibration.validate_sensor_shape(sensor_shape)
    return calibration


def NullCalibration(sensor_shape: tuple[int, int]) -> EventCalibration:
    zeros = np.zeros(sensor_shape, dtype=np.float64)
    hot_pixels = np.zeros(sensor_shape, dtype=bool)
    valid_pixels = np.ones(sensor_shape, dtype=bool)
    return EventCalibration(
        dark_rate_pos=zeros.copy(),
        dark_rate_neg=zeros.copy(),
        blank_rate_pos=zeros.copy(),
        blank_rate_neg=zeros.copy(),
        hot_pixel_mask=hot_pixels,
        valid_pixel_mask=valid_pixels,
        calibration_id="none",
        calibrated=False,
        metadata={},
    )


def _load_npz_calibration(path: Path) -> EventCalibration:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "dark_rate_pos",
            "dark_rate_neg",
            "blank_rate_pos",
            "blank_rate_neg",
            "hot_pixel_mask",
            "valid_pixel_mask",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(
                f"Calibration file {path} is missing required array(s): "
                + ", ".join(missing)
            )

        dark_rate_pos = np.asarray(payload["dark_rate_pos"], dtype=np.float64)
        dark_rate_neg = np.asarray(payload["dark_rate_neg"], dtype=np.float64)
        blank_rate_pos = np.asarray(payload["blank_rate_pos"], dtype=np.float64)
        blank_rate_neg = np.asarray(payload["blank_rate_neg"], dtype=np.float64)
        hot_pixel_mask = np.asarray(payload["hot_pixel_mask"], dtype=bool)
        valid_pixel_mask = np.asarray(payload["valid_pixel_mask"], dtype=bool)

        _validate_common_shape(
            dark_rate_pos,
            dark_rate_neg,
            blank_rate_pos,
            blank_rate_neg,
            hot_pixel_mask,
            valid_pixel_mask,
        )
        return EventCalibration(
            dark_rate_pos=dark_rate_pos,
            dark_rate_neg=dark_rate_neg,
            blank_rate_pos=blank_rate_pos,
            blank_rate_neg=blank_rate_neg,
            hot_pixel_mask=hot_pixel_mask,
            valid_pixel_mask=valid_pixel_mask,
            pixel_size_nm=_optional_float(payload, "pixel_size_nm"),
            sensor_model=_optional_str(payload, "sensor_model", default="unknown"),
            calibration_id=_optional_str(payload, "calibration_id", default=path.stem),
            calibrated=True,
            metadata={},
        )


def _background_rate(dark_rate: np.ndarray, blank_rate: np.ndarray) -> np.ndarray:
    laser_excess_rate = np.maximum(blank_rate - dark_rate, 0.0)
    return dark_rate + laser_excess_rate


def _validate_common_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(
            f"Calibration arrays must have one shape, got {sorted(shapes)}"
        )
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError(f"Calibration maps must be 2D arrays, got shape {shape}")


def _optional_float(payload: Any, key: str) -> float | None:
    if key not in payload.files:
        return None
    return float(np.asarray(payload[key]).item())


def _optional_str(payload: Any, key: str, *, default: str) -> str:
    if key not in payload.files:
        return default
    return str(np.asarray(payload[key]).item())
