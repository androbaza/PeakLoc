# Data preparation

PeakLoc depends strongly on correct input data. Most failed or misleading results come from wrong sensor geometry, missing OpenEB bindings, wrong time slicing, wrong calibration assumptions, or overly large first runs.

## Input file type

PeakLoc expects event-camera `.raw` files readable through OpenEB / Metavision.

The pipeline reads `.raw` files using:

```python
metavision_core.event_io.raw_reader.RawReader
metavision_sdk_base.EventCD
```

If these bindings are unavailable, `.raw` reading will fail before localization starts.

## Required OpenEB / Metavision bindings

On Ubuntu, the repository expects OpenEB / Metavision Python packages under:

```text
/usr/lib/python3/dist-packages
```

PeakLoc adds this path at runtime if it exists.

Install the OpenEB / Metavision packages outside Pixi first. Then use Pixi for the PeakLoc Python environment.

Check whether the bindings are visible:

```bash
pixi run python -c "from metavision_core.event_io.raw_reader import RawReader; from metavision_sdk_base import EventCD; print('OpenEB OK')"
```

If this fails, fix the OpenEB installation before running PeakLoc.

## Expected sensor geometry

The default `config.json` assumes a sensor of:

```json
{
  "sensor_width": 1280,
  "sensor_height": 720
}
```

This corresponds to image arrays shaped as:

```text
height × width = 720 × 1280
```

Coordinate convention:

```text
image[y, x] == image[row, column]
```

Use this convention when checking overlays, ROI positions, and rendered images.

If your camera has a different geometry, update both fields:

```json
{
  "sensor_width":  YOUR_SENSOR_WIDTH,
  "sensor_height": YOUR_SENSOR_HEIGHT
}
```

Do not ignore geometry mismatch. Wrong geometry can produce invalid ROIs, wrong masks, failed calibration validation, or incorrect rendered images.

## RAW file folder layout

Use a simple input folder first.

Recommended minimal layout:

```text
PeakLoc/
├── config.json
└── data/
    ├── sample_001.raw
    ├── sample_001.bias
    ├── sample_002.raw
    └── sample_002.bias
```

In `config.json`:

```json
{
  "input_folder": "data"
}
```

PeakLoc creates one output directory per `.raw` file:

```text
data/
├── sample_001.raw
├── sample_001.bias
└── sample_001/
    ├── localizations_*.npy
    ├── rois_*.npy
    ├── localization_qc_*.npy
    ├── figures/
    ├── reports/
    └── qc/
```

## `.bias` files

A `.bias` file stores event-camera bias/acquisition settings.

PeakLoc does not currently use the `.bias` file as a direct numerical input to the localization fit. You should still keep the `.bias` file next to the `.raw` file because it documents the camera state during acquisition.

For reproducibility, keep:

```text
sample.raw
sample.bias
```

together.

The `.bias` file is important because event rates, noise, hot pixels, and apparent blinking dynamics can depend strongly on bias settings.

## Calibration files

PeakLoc supports an event-model calibration `.npz` file through:

```json
{
  "calibration_path": "path/to/calibration_event_model.npz",
  "allow_uncalibrated": false
}
```

The calibration file is expected to contain maps such as:

```text
dark_rate_pos
dark_rate_neg
blank_rate_pos
blank_rate_neg
hot_pixel_mask
valid_pixel_mask
```

The calibration maps must match the sensor shape.

For example, with the default sensor:

```text
720 × 1280
```

A calibration file built for a different sensor size should not be used.

## Building an event calibration file

The repository contains:

```text
calibration_scripts/build_event_calibration.py
```

This script expects two recordings:

1. A dark recording.
2. A laser-on blank recording.

The goal is to estimate background event rates and hot pixels.

Example command:

```bash
pixi run python -m calibration_scripts.build_event_calibration \
  --output calibration_event_model.npz \
  --pixel-size-nm 67.0 \
  --sensor-model "your-camera-model" \
  --calibration-id "YYYYMMDD_standard_bias" \
  --height 720 \
  --width 1280
```

The script prompts for the dark and blank `.raw` paths.

## Uncalibrated mode

The default config can allow uncalibrated processing:

```json
{
  "allow_uncalibrated": true,
  "calibration_path": null,
  "background_mode": "local_only"
}
```

This is acceptable for:

- smoke tests,
- debugging,
- exploratory visualization,
- checking whether a recording contains detectable events.

It is not sufficient for strong quantitative claims. For publication-grade work, use calibration recordings and run strict preflight.

## Optical pixel size

The config field:

```json
{
  "optical_pixel_size": 67.0
}
```

defines the physical size of one camera pixel in nanometers at the sample plane.

PeakLoc stores fitted `x` and `y` in camera pixels. Convert to nanometers with:

```text
x_nm = x_px × optical_pixel_size
y_nm = y_px × optical_pixel_size
```

Wrong optical pixel size gives wrong physical distances, wrong uncertainty in nanometers, wrong scalebars, and wrong FRC scale.

## RAM expectations

PeakLoc can use substantial RAM because event streams are transformed into intermediate per-pixel and per-ROI structures.

A practical expectation:

- Short smoke run: use a small `slice_duration`.
- Full-chip recording: use a high-memory workstation.
- Long 1280 × 720 recordings: 128 GB RAM is recommended.
- First validation should never be done on the full recording.

Start with:

```json
{
  "slice_start": 0,
  "slice_duration": 10000000,
  "max_raw_events": 1000000,
  "num_cores": 4
}
```

Then increase `slice_duration` only after outputs look reasonable.

## Data checklist before running

Before running a full batch, check:

- The `.raw` file can be opened by OpenEB.
- The `.bias` file is stored next to the `.raw` file.
- `sensor_width` and `sensor_height` match the camera.
- `optical_pixel_size` matches the microscope configuration.
- `slice_start` and `slice_duration` select a valid time range.
- Calibration file is either provided or uncalibrated mode is intentionally allowed.
- The input folder contains only recordings you want to process.
- You have enough RAM for the selected slice duration.
- Preflight passes.
