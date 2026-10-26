# -*- coding: utf-8 -*-
"""
Verification (read-only): aggregation trace and figure/memory checks.

Point 3: reproduce exactly how the summary numbers (8.3 d, 27.6 d, 3.3x, 1.5 C)
         were computed, and state the cell/year selection.
Point 4: colorbar saturation (fraction of cells above the map vmax) and the
         memory layout used by hw_full.py.
"""
import os

import numpy as np
import xarray as xr

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

ds = xr.open_dataset(os.path.join(RES, "hw_annual_1961-2022.nc"))
vm = xr.open_dataset(os.path.join(RES, "valid_mask.nc"))
valid = vm["valid_mask"].values.astype(bool)
vm.close()
years = ds["year"].values
FILL_I = -9999


def grab(name):
    a = ds[name].values.astype(float)
    return np.where(a == FILL_I, np.nan, a)


count = grab("hw_count")
days = grab("hw_days")
maxd = grab("hw_max_dur")
intens = grab("hw_intensity")
n_valid = int(valid.sum())

print("=" * 72)
print("CHECK 3 -- aggregation trace")
print("=" * 72)

# --- 8.3 days ---
cell_clim_days = np.nanmean(days, axis=0)              # per-cell mean over 62 years
cell_clim_days = np.where(valid, cell_clim_days, np.nan)
mean_days = np.nanmean(cell_clim_days)
print("8.3 days  = nanmean over 62 years per cell, then arithmetic mean over")
print(f"            valid cells.  n_valid_cells={n_valid}, value={mean_days:.4f}")
print(f"            (denominator = ALL {n_valid} valid cells, incl. cells with 0 heatwave days)")

# --- 27.6 days ---
iy = int(np.where(years == 2022)[0][0])
d22 = np.where(valid, days[iy], np.nan)
mean_d22 = np.nanmean(d22)
print(f"27.6 days = arithmetic mean over valid cells of the 2022 field only.")
print(f"            value={mean_d22:.4f}  (n_valid_cells={n_valid})")

# --- 3.3x ---
print(f"3.3x      = 27.62 / 8.34 = {mean_d22/mean_days:.3f}  (ratio of the two")
print("            unweighted means above, same fixed valid-cell set)")

# --- 1.5 C ---
cell_clim_intens = np.nanmean(intens, axis=0)          # nanmean over years
cell_clim_intens = np.where(valid, cell_clim_intens, np.nan)
mean_intens = np.nanmean(cell_clim_intens)
n_intens_cells = int(np.isfinite(cell_clim_intens).sum())
print("1.5 C     = per-cell nanmean of annual intensity over years, then")
print(f"            arithmetic mean over valid cells.")
print(f"            value={mean_intens:.4f}  (n cells with >=1 finite annual value = {n_intens_cells})")
print("            NOTE: nanmean over years DROPS no-heatwave years (not counted")
print("            as 0); and cells with no heatwave in any year are dropped too.")
print()
print("All four numbers are UNWEIGHTED arithmetic means over grid cells (each")
print("0.25x0.25 cell counts equally). They are NOT area-weighted and NOT a")
print("national/regional aggregate -- no China boundary or area weight was used.")
print()

print("=" * 72)
print("CHECK 4a -- colorbar saturation")
print("=" * 72)
VMAP = {"hw_count": 4, "hw_days": 20, "hw_max_dur": 10, "hw_intensity": 3}
for name in ["hw_count", "hw_days", "hw_max_dur", "hw_intensity"]:
    a = grab(name)
    clim = np.where(valid, np.nanmean(a, axis=0), np.nan)
    y22 = np.where(valid, a[iy], np.nan)
    vmax = VMAP[name]
    frac_clim = float(np.nanmean(clim > vmax))
    frac_22 = float(np.nanmean(y22 > vmax))
    print(f"  {name:<12} vmax={vmax:3d} | clim max={np.nanmax(clim):6.2f} "
          f"frac_above_vmax={frac_clim*100:5.2f}% | 2022 max={np.nanmax(y22):6.2f} "
          f"frac_above_vmax={frac_22*100:5.2f}%")

print()
print("  -> the 2022 maps reuse the climatology vmax, so 2022 is heavily clipped.")
print()
print("=" * 72)
print("CHECK 4b -- memory layout used by hw_full.py")
print("=" * 72)
print("  data (full record)         : (22645,163,283) float32 = 4.18 GB (in RAM)")
print("  Xb = data[baseline_idx]    : fancy indexing by an int64 index ARRAY")
print("                                (np.where -> ndarray), so this is a COPY of")
print("                                (10957,163,283) = 2.02 GB, NOT a view.")
print("  thr2d (thresholds)         : (366, 46129) float64 = 0.135 GB, then cast to")
print("                                float32 = 0.067 GB")
print("  per-slot percentile sample : (<=150, 46129) transient = ~0.055 GB")
print("  peak approx                : 4.18 + 2.02 + 0.135 + 0.055 ~ 6.4 GB")
print("  event-detection loop       : per-cell, small temporary arrays only")
print("  available RAM at run time  : ~12.9 GB (psutil reported)")
print()
print("  Evidence that data[baseline_idx] copies (fancy indexing):")
small = np.arange(12.0).reshape(4, 3)
idx = np.array([1, 2, 3])
sub = small[idx]
print("    np.shares_memory(small, small[array_index]) =",
      np.shares_memory(small, sub),
      "(False => copy)")

ds.close()
