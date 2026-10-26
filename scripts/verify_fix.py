# -*- coding: utf-8 -*-
"""
Phase-1 fix verification (read-only). Confirms two things about the gated run:

  (1) Consistency: results_fixed/ differs from results/ ONLY at the previously
      undecidable cell-years (decidable days == 0), which changed from 0/0/0 to
      missing (-9999 / NaN). Every other cell-year is bit-identical.
  (2) The fixed summary numbers, recomputed as UNWEIGHTED grid-cell arithmetic
      means (each 0.25x0.25 cell counts equally, NOT area-weighted).

Writes nothing.
"""
import os

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(ROOT, "results")
NEW = os.path.join(ROOT, "results_fixed")
FILL_I = -9999

old = xr.open_dataset(os.path.join(OLD, "hw_annual_1961-2022.nc"))
new = xr.open_dataset(os.path.join(NEW, "hw_annual_1961-2022.nc"))
vm = xr.open_dataset(os.path.join(NEW, "valid_mask.nc"))
valid = vm["valid_mask"].values.astype(bool)
vm.close()

years = new["year"].values
status = new["data_status"].values
dec = new["decidable_days"].values

undef = (status == 3)        # decidable == 0
complete = (status == 1)     # decidable == 153
incomplete = (status == 2)   # 0 < decidable < 153


def g(ds, name):
    a = ds[name].values.astype(float)
    return np.where(a == FILL_I, np.nan, a)


print("=" * 72)
print("DATA-STATUS TALLY (fixed output)")
print("=" * 72)
print(f"  ocean      (status 0): {int((status == 0).sum()):>8d}")
print(f"  complete   (status 1): {int(complete.sum()):>8d}")
print(f"  incomplete (status 2): {int(incomplete.sum()):>8d}")
print(f"  undecidable(status 3): {int(undef.sum()):>8d}")

inc_valid = incomplete & valid[None, :, :]
print()
print("  incomplete (valid) cell-years decidable-day histogram:")
uv, uc = np.unique(dec[inc_valid], return_counts=True)
for v, c in zip(uv, uc):
    print(f"      decidable == {int(v):3d}: {int(c)} cell-years")

print()
print("=" * 72)
print("(1) CONSISTENCY fixed vs original")
print("=" * 72)
for name in ["hw_count", "hw_days", "hw_max_dur", "hw_intensity"]:
    o = g(old, name)
    n = g(new, name)
    changed = ~np.isclose(o, n, equal_nan=True)
    n_changed = int(changed.sum())
    outside = int((changed & ~undef).sum())
    flag = "OK" if outside == 0 else "!! NOT OK"
    print(f"  {name:<12}: changed={n_changed:6d}  changed-outside-undecidable={outside:6d}  [{flag}]")
    if n_changed and name != "hw_intensity":
        print(f"             old values at changed cells: {np.unique(o[changed])}  "
              f"(new = NaN/{-9999})")

print()
print("  (intensity was already NaN at undecidable cell-years, so it never changes.)")

print()
print("=" * 72)
print("(2) SUMMARY NUMBERS -- UNWEIGHTED grid-cell arithmetic mean")
print("=" * 72)
count = g(new, "hw_count")
days = g(new, "hw_days")
intens = g(new, "hw_intensity")
n_valid = int(valid.sum())

# climatological days: per-cell mean over 62 years, then mean over valid cells
cell_clim_days = np.where(valid, np.nanmean(days, axis=0), np.nan)
mean_days = np.nanmean(cell_clim_days)
n_days_cells = int(np.isfinite(cell_clim_days).sum())

iy = int(np.where(years == 2022)[0][0])
d22 = np.where(valid, days[iy], np.nan)
mean_d22 = np.nanmean(d22)
n_d22_cells = int(np.isfinite(d22).sum())

cell_clim_intens = np.where(valid, np.nanmean(intens, axis=0), np.nan)
mean_intens = np.nanmean(cell_clim_intens)
n_intens_cells = int(np.isfinite(cell_clim_intens).sum())

print(f"  climatological heatwave days = {mean_days:.4f}  (over {n_days_cells} cells; "
      f"each year nanmean'd, no-heatwave years counted as 0)")
print(f"  2022 heatwave days           = {mean_d22:.4f}  (over {n_d22_cells} cells; "
      f"45 undecidable cells dropped)")
print(f"  2022 / climatology ratio     = {mean_d22 / mean_days:.3f}")
print(f"  mean exceedance intensity    = {mean_intens:.4f} degC  (over {n_intens_cells} "
      f"cells with >=1 heatwave year; no-heatwave years DROPPED, not 0)")
print()
print("  These are UNWEIGHTED arithmetic means over grid cells (each 0.25x0.25 cell")
print("  counts equally). NOT area-weighted, NOT a national aggregate -- no China")
print("  boundary or area weight is used.")
print()
print("  Intensity definition: for each cell, mean exceedance intensity is averaged")
print("  over YEARS THAT HAVE >=1 heatwave day only (nanmean over years), then those")
print("  per-cell values are averaged over cells. So the intensity denominator is")
print("  smaller than the days denominator (days counts heatwave-free years as 0).")

old.close()
new.close()
