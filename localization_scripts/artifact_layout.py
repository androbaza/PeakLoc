"""Canonical filesystem layout for artifacts produced by one PeakLoc run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Named locations for collaborator-facing and diagnostic run artifacts.

    ``share`` contains the concise, portable hand-off bundle. ``debug`` retains
    arrays, diagnostics, reports, provenance, and temporary worker artifacts
    needed to audit or reproduce that bundle.
    """

    run_directory: Path
    share_dir: Path
    share_figures_dir: Path
    share_statistics_dir: Path
    share_metadata_dir: Path
    debug_dir: Path
    debug_arrays_dir: Path
    debug_qc_dir: Path
    debug_reports_dir: Path
    debug_provenance_dir: Path
    temp_files_dir: Path

    @classmethod
    def from_run_directory(cls, run_directory: str | Path) -> ArtifactLayout:
        """Build the standard layout without creating any directories."""
        root = Path(run_directory)
        share_dir = root / "share"
        debug_dir = root / "debug"
        return cls(
            run_directory=root,
            share_dir=share_dir,
            share_figures_dir=share_dir / "figures",
            share_statistics_dir=share_dir / "statistics",
            share_metadata_dir=share_dir / "metadata",
            debug_dir=debug_dir,
            debug_arrays_dir=debug_dir / "arrays",
            debug_qc_dir=debug_dir / "qc",
            debug_reports_dir=debug_dir / "reports",
            debug_provenance_dir=debug_dir / "provenance",
            temp_files_dir=debug_dir / "temp_files",
        )

    def ensure_directories(self) -> None:
        """Create the complete run-artifact tree, including an empty temp leaf."""
        for directory in (
            self.share_figures_dir,
            self.share_statistics_dir,
            self.share_metadata_dir,
            self.debug_arrays_dir,
            self.debug_qc_dir,
            self.debug_reports_dir,
            self.debug_provenance_dir,
            self.temp_files_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
