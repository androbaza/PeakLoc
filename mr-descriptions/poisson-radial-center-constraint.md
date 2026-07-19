# Enforce radial Poisson-fit center constraints

## Motivation

The Clathrin Minimum Dead Time run showed healthy-looking ROI signal while 163 fits were marked
as failed. The optimizer had converged, but it frequently selected a corner of the existing
axis-aligned `(x, y)` bound. This placed a fitted center as much as 4.24 px from the ROI seed
despite `max_fit_center_offset_px=3.0`, causing unstable Fisher estimates and accepting a small
number of obviously misplaced centers.

## Changes

- Interpret `max_fit_center_offset_px` as the radial limit its name promises, using a stiff
  center penalty in the L-BFGS-B objective and a final numerical projection to the circle.
- Set the default and active Clathrin configuration to a 1.75 px displacement, matching the
  temporal-segmentation centroid tolerance.
- Add a synthetic corner-at-the-axis-limit regression test and stabilize an exact floating-point
  test comparison.

## Validation

- The saved 2026-07-17 Clathrin run had 163 condition-based rejections. Re-fitting those ROIs
  with the new 1.75 px setting recovered 159 at `max_fit_cond=100`; every recovered center was
  within 1.75 px of its ROI seed.
- Re-fitting the 36 most displaced previous fits reduced their maximum center displacement from
  4.24 px to 1.75 px.
- `pixi run -e dev pytest localization_scripts/tests/test_poisson_fitting.py -q` passed (8 tests).
- Ruff check and formatting passed. Full `ty check` remains blocked by 8 pre-existing diagnostics
  in `notebooks/raw_event_npz_explorer.ipynb`; scoped type checking of the changed Python files
  passed.
