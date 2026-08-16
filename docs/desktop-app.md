# PeakLoc desktop application

The PeakLoc desktop application is the recommended route for users who do not work from a
terminal. It exposes the same validated configuration and scientific pipeline as the command-line
workflow while keeping JSON, Python, and Pixi out of the normal acquisition-to-processing path.

## Before the first run

For Prophesee RAW recordings, install Metavision Studio / SDK using its default location:

    C:\Program Files\Prophesee

The SDK must provide bindings for the Python version used by the application (currently CPython
3.9 for the Windows build). NumPy event arrays do not require the RAW decoder. **Check setup**
reports a clear error if the decoder is unavailable.

Keep the complete delivered folder together. Do not move only PeakLoc.exe away from _internal.

## Guided workflow

### 1. Data

Choose **One recording** to test a single RAW or NumPy event file. Choose **A folder of
recordings** for a batch, and enable subfolders only when the hierarchy is intentional. PeakLoc
writes timestamped results beside each source recording, so ensure that location is writable.

### 2. Calibration

For calibrated processing:

1. Record a dark acquisition with the same camera settings as the experiment.
2. Record a laser-on blank acquisition without emitters.
3. Choose both RAW files and an output NPZ location.
4. Confirm optical pixel size and sensor dimensions in Settings.
5. Select **Build calibration** and follow progress in the Run log.

The new calibration becomes the selected processing calibration automatically. Existing NPZ
calibrations can be selected directly. Uncalibrated mode is for exploratory tuning and is not a
substitute for calibration in publication-oriented work.

### 3. Basic settings

Start with a short processing range. A 10,000,000 microsecond slice is 10 seconds. Basic settings
expose the controls most often changed between datasets: processing range, CPU use, spatial
masking, peak prominence, PSF width, ROI radius, optical scale, fit uncertainty, and outputs.

Every control displays its unit and a plain-language explanation. Defaults are starting points,
not universal microscope settings.

### 4. Advanced settings

Advanced contains every remaining supported PeakLoc configuration field, grouped by purpose.
Keep these defaults until QC output or a documented acquisition change motivates adjustment.

### 5. Run

Select **Check setup** before processing. This checks the recording selection, RAW decoder,
calibration arrays, optical and sensor consistency, scientific parameter relationships, RAM, and
disk headroom. Errors must be corrected before processing; warnings deserve review.

Select **Start processing** after the check passes. The application stays responsive and streams
pipeline output into the log. **Cancel** stops the process tree; partial output may remain.
The same streamed output is appended to `PeakLoc.log` beside `PeakLoc.exe` for troubleshooting.

### 6. Live measurement

Starting a processing run opens **Live measurement** automatically. Its timeline shows the full
recording, the selected processing interval, and pending, active, completed, skipped, or failed
slices. The tab is divided into **Progress**, **Reconstruction**, and **Signals & ROI** pages so the
timeline key and plots remain readable in smaller windows. **Progress** reports selected duration,
slice completion, active/failed slices, and localization yield. **Reconstruction** shows occupied
sensor pixels, peak density, and mean localizations per occupied pixel. These are descriptive live
indicators, not scientific acceptance thresholds. Use the Matplotlib toolbar to pan or zoom.

PeakLoc also retains a bounded sample of extracted cumulative-sum peak traces and blink ROIs for
interactive review. Choose samples from the dropdowns, or click near a marked ROI in the
localization image. In **Signals & ROI**, the yellow peak-trace shading marks the
algorithm-extracted ON-to-OFF blink interval and the red line marks the detected peak center.
Select **Both**, **Only ON**, or **Only OFF** to filter the ROI map and event-time histogram. In
**Both**, the emitter map is signed ON minus OFF counts; the single-polarity modes show nonnegative
event counts. Move **First event** and **Last event** to inspect how the manual time window changes
the sampled emitter. These controls inspect copied diagnostic samples only: they do not alter the
measurement, fitted localizations, or saved scientific output. If monitoring data is missing or
malformed, the warning remains in this tab and processing continues independently.

## Saving repeatable settings

Use **Save config** to write a JSON configuration for later reuse. **Open config** loads a saved
configuration into all workflow and settings pages. PeakLoc also writes effective settings into
each run's debug metadata.

## Running from source

On Windows or Linux with Pixi installed:

    pixi install
    pixi run gui

The interface uses standard-library Tk/ttk and the same application code on both platforms.
Executable builds must be created on their target platform.

## Building the Windows application

On the Windows build PC, install the matching Metavision SDK, then run:

    pixi install
    pixi run check-openeb
    pixi run -e dev build-gui

The release is written to dist\PeakLoc. The build explicitly bundles h5py and its native HDF5
dependencies because RAW decoding imports h5py before Metavision. Test PeakLoc.exe, **Check
setup**, calibration with small real recordings, and a short processing run from that folder
before delivery.
