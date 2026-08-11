$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repositoryRoot

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
    "--exclude-module", "localization_scripts.tests",
    "--exclude-module", "calibration_scripts.test_estimate_bead_sigma",
    "--exclude-module", "numba.tests",
    "--exclude-module", "numba.cuda",
    "--exclude-module", "pytest",
    "--hidden-import", "tkinter",
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
