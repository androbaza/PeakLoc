from __future__ import annotations

from pathlib import Path
from typing import Final

from matplotlib.axes import Axes
from matplotlib.figure import Figure

PUBLICATION_DPI: Final[int] = 450
PREVIEW_DPI: Final[int] = 300
SINGLE_COLUMN_WIDTH_IN: Final[float] = 3.5
DOUBLE_COLUMN_WIDTH_IN: Final[float] = 7.2
PANEL_LABEL_SIZE: Final[float] = 9.0
AXIS_LABEL_SIZE: Final[float] = 8.0
TICK_LABEL_SIZE: Final[float] = 7.0

PLOT_COLORS: Final[dict[str, str]] = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#999999",
    "black": "#000000",
}

SEQUENTIAL_CMAP: Final[str] = "cividis"
EVENT_DENSITY_CMAP: Final[str] = "magma"


def style_publication_axis(axis: Axes, *, show_grid: bool = False) -> None:
    """Apply the common compact styling used by run-report figures."""
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=0.8,
        length=3.0,
        pad=2.0,
    )
    axis.set_xlabel(axis.get_xlabel(), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(axis.get_ylabel(), fontsize=AXIS_LABEL_SIZE)
    axis.set_title(axis.get_title(), fontsize=PANEL_LABEL_SIZE)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        axis.spines[spine].set_linewidth(0.8)
    if show_grid:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)


def save_publication_figure(
    figure: Figure,
    path: Path,
    *,
    dpi: int,
    save_vector: bool,
) -> list[Path]:
    """Persist a publication-ready raster and, when requested, a PDF companion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    paths = [path]
    if save_vector:
        vector_path = path.with_suffix(".pdf")
        figure.savefig(vector_path, bbox_inches="tight", facecolor="white")
        paths.append(vector_path)
    return paths
