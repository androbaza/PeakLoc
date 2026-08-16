$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repositoryRoot

$pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.9") {
    throw "The Windows GUI must be built with the Pixi CPython 3.9 environment required by Metavision. Run 'pixi run -e dev build-gui'."
}

$pyinstallerArguments = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", "PeakLoc",
    "--runtime-hook", "scripts\pyinstaller_metavision_runtime_hook.py",
    "--collect-submodules", "interface",
    "--collect-submodules", "localization_scripts",
    "--collect-submodules", "calibration_scripts",
    "--collect-all", "numba",
    # The RAW decoder imports h5py before loading Metavision. GUI worker imports are lazy,
    # so explicitly bundle h5py and its native HDF5 dependencies in the frozen app.
    "--collect-all", "h5py",
    "--exclude-module", "localization_scripts.tests",
    "--exclude-module", "calibration_scripts.test_estimate_bead_sigma",
    "--exclude-module", "numba.tests",
    "--exclude-module", "numba.cuda",
    "--exclude-module", "pytest",
    "--hidden-import", "tkinter",
    "--hidden-import", "matplotlib.backends.backend_svg",
    "PeakLocGUI.py"
)
& pyinstaller @pyinstallerArguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$releaseDirectory = Join-Path $repositoryRoot "dist\PeakLoc"
Copy-Item "config.portable.json" (Join-Path $releaseDirectory "config.json") -Force
Copy-Item "docs\desktop-app.md" (Join-Path $releaseDirectory "PeakLoc User Guide.md") -Force

Write-Host ""
Write-Host "PeakLoc desktop distribution is ready:"
Write-Host "  $releaseDirectory"
Write-Host "Deliver the entire folder, including _internal."
