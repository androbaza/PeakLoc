# Live measurement insight refinement

## Goal

Make the sixth desktop tab a responsive, insight-first workspace for judging live data and
settings without changing pipeline results or allowing monitor failures to affect processing.

## Scope

- Split the crowded live view into Progress, Reconstruction, and Signals & ROI subpages.
- Put selectors, polarity display controls, and manual event-window sliders beside the plots they
  affect.
- Add Both, Only ON, and Only OFF ROI display modes and explain the extracted peak-trace shading.
- Add bounded, descriptive summaries from already-copied monitor snapshots; do not recalculate or
  mutate pipeline output.
- Keep the progress key outside the Matplotlib axes so it remains readable at the 960 x 680
  minimum window size.
- Change code and portable-config defaults to a 5 px ROI radius and one fewer worker than the
  available logical CPU count, with a floor of one. Explicit user settings continue to win.

## Stability constraints

- Preserve allow-pickle-free, bounded snapshot loading and the existing monitor exception boundary.
- Keep UI callbacks read-only and redraw only the affected canvas.
- Preserve `image[y, x]` and plot coordinates as `(x, y)`.
- Do not modify the user's root `config.json` edit.

## Completion evidence

- Focused live-monitor and configuration tests plus changed-file Ruff and compile checks pass.
- Headless GUI smoke checks cover 960 x 680 and 1180 x 780, all three subpages, all polarity modes,
  manual ROI windowing, localization selection, and malformed snapshot isolation.
- Updated PNG/SVG/source-data preview artifacts and desktop documentation describe the interface.
- Changes are committed with a conventional commit and summarized in `mr-descriptions/`.
