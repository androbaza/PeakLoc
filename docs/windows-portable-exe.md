# Build a portable Windows PeakLoc application

This guide creates a `PeakLoc.exe` distribution for another Windows PC. The user only
needs to install Metavision Studio / SDK in its default location. They do not need Pixi,
Python, this repository, or any environment variables.

The released folder must keep these items together:

```text
PeakLoc/
├── PeakLoc.exe
├── config.json
└── _internal/                 # created by PyInstaller; do not change or remove
```

`PeakLoc.exe` is a console application. Double-clicking it opens a terminal window and
prints the same PeakLoc logs that `pixi run peakloc` prints during a source run. It uses
the `config.json` in the same folder as the executable, regardless of the directory
from which it was started. Relative paths in that configuration, such as
`"input_folder": "data"`, are also relative to the application folder.

## Target-PC requirement

Install the Windows Metavision Studio / SDK on the target computer before copying the
PeakLoc folder. Accept the installer default location:

```text
C:\Program Files\Prophesee
```

The bundled application loads the Metavision Python bindings and DLLs from that
location. No Pixi, Python, or `PEAKLOC_METAVISION_ROOT` setting is needed on the target
PC. The installed SDK must supply CPython 3.9 bindings, matching PeakLoc's Windows
runtime.

## Build the distribution on Windows

Build on a 64-bit Windows PC with Metavision installed in the default location. Pixi is
needed only on this build PC.

1. Open **PowerShell** in the cloned PeakLoc repository and create the Windows
   environment:

   ```powershell
   pixi install
   pixi run check-openeb
   ```

   `check-openeb` must report that `RawReader` and `EventCD` load successfully. Resolve
   that problem before building; otherwise the executable cannot decode `.raw` files.

2. Set `config.json` to the configuration to deliver. Keep portable paths relative to
   the release folder. For example, use `"input_folder": "data"` and later create a
   `data` folder next to `PeakLoc.exe` for the recipient's `.raw` recordings. Do not use
   a build-PC-specific absolute path.

3. Build a console, one-folder application. Run this command from the repository root:

   ```powershell
   pixi run --with pyinstaller pyinstaller `
     --noconfirm `
     --clean `
     --onedir `
     --console `
     --name PeakLoc `
     --runtime-hook scripts\pyinstaller_metavision_runtime_hook.py `
     --collect-submodules localization_scripts `
     --collect-all numba `
     --collect-all plotly `
     PeakLoc.py
   ```

   Keep `--console`: `--windowed` or `--noconsole` suppresses the terminal and hides the
   logs. `--onedir` is deliberate: scientific packages are more reliable in this layout
   than as a single self-extracting executable, and `config.json` remains visibly beside
   the executable.

4. Copy the chosen configuration beside the executable:

   ```powershell
   Copy-Item config.json dist\PeakLoc\config.json -Force
   ```

5. Test the release folder itself, not the source checkout:

   ```powershell
   Set-Location dist\PeakLoc
   .\PeakLoc.exe --preflight-only
   ```

   This must open a console, identify the configuration beside `PeakLoc.exe`, and either
   finish preflight or report expected data/configuration issues. An error about a missing
   Metavision installation means the default SDK installation is absent or incomplete.

## Deliver and run on another PC

Copy the entire `dist\PeakLoc` folder to the target PC; do not copy only
`PeakLoc.exe`. Keep `_internal` and `config.json` next to it. If the delivered
configuration uses the portable `data` path, place recordings in:

```text
PeakLoc\data\
```

Then double-click `PeakLoc.exe`. The console remains visible for the duration of the
run and contains PeakLoc's logs and any error message. Do not move `config.json` away
from the executable or rename `_internal`.

## Rebuild checklist

- Build with the Windows Pixi environment, which uses Python 3.9.
- Confirm `pixi run check-openeb` before packaging.
- Use the supplied runtime hook so the frozen process finds the default Metavision SDK.
- Copy `config.json` beside `PeakLoc.exe` after every clean build.
- Copy the whole `dist\PeakLoc` directory to the target computer.
