# -*- coding: utf-8 -*-
"""
Verification (read-only): per-cell-year data completeness vs saved 0/0/0 outputs.

Point 1 of the phase-1 audit. For every valid grid cell and every year it
counts, over the May-Sep season (153 days):
  * valid Tmax days        (finite Tmax)
  * valid threshold days   (finite calendar-day threshold)
  * decidable days         (both finite)
  * max consecutive missing Tmax run
and cross-references cells whose saved hw_count/hw_days/hw_max_dur are all 0
against the "all-missing / threshold-invalid / no-decidable-day" conditions.

Reads source + results only; writes nothing.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from numba import njit

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_lib as hw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")
RES = os.path.join(ROOT, "results")


@njit
def max_consec_missing(mask):
    """mask: (nt, ncells) bool (True = missing). -> max run length per cell."""
    nt, nc = mask.shape
    out = np.zeros(nc, dtype=np.int32)
    for c in range(nc):
        run = 0
        best = 0
        for t in range(nt):
            if mask[t, c]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        out[c] = best
    return out


# ---- calendar
nc = Dataset(NC, "r")
var = nc.variables["tmax"]
t = nc.variables["time"][:]
origin = pd.Timestamp("1961-01-01 00:00:00")
dates = origin + pd.to_timedelta(t, unit="h")
month = dates.month.values.astype(np.int64)
day = dates.day.values.astype(np.int64)
year = dates.year.values.astype(np.int64)
years = np.arange(int(year.min()), int(year.max()) + 1)
n_years = len(years)
slot_all = hw.slot_index(month, day)
season = hw.season_mask(month, day)
season_idx = np.where(season)[0]
slot_season = slot_all[season_idx]

# ---- thresholds (fixed over years)
thr_ds = xr.open_dataset(os.path.join(RES, "hw_threshold_1981-2010.nc"))
thr366 = thr_ds["threshold"].values.astype(float)          # (366, lat, lon)
thr_ds.close()
season_slots_1yr = slot_season[:153]                        # same 153 slots every year
thr_season = thr366[season_slots_1yr]                       # (153, lat, lon)
thr_missing = np.isnan(thr_season)

# ---- saved outputs
out_ds = xr.open_dataset(os.path.join(RES, "hw_annual_1961-2022.nc"))
count = out_ds["hw_count"].values
days = out_ds["hw_days"].values
maxd = out_ds["hw_max_dur"].values
out_ds.close()
vm_ds = xr.open_dataset(os.path.join(RES, "valid_mask.nc"))
valid = vm_ds["valid_mask"].values.astype(bool)
vm_ds.close()

FILL_I = -9999
nlat, nlon = valid.shape

# ---- accumulate per (year, cell)
n_valid_cellyears = 0
n_zero_output = 0
n_zero_and_nomissing = 0            # zero output AND decidable>0 (genuine)
n_zero_undef = 0                    # zero output AND no decidable day
n_zero_allmissing = 0               # zero output AND all Tmax missing
n_zero_thrinvalid = 0               # zero output AND all thresholds invalid
hist_decidable = np.zeros(154, dtype=np.int64)
n_any_gap_over = {5: 0, 10: 0, 15: 0, 30: 0}   # cell-years with a missing run > k

for yi, yv in enumerate(years):
    sel = np.where(season & (year == yv))[0]
    i0, i1 = int(sel[0]), int(sel[-1]) + 1
    arr = var[i0:i1]                                   # (153, lat, lon)
    data = np.ma.filled(arr.astype(np.float32), np.float32(np.nan))
    miss = np.isnan(data)                              # (153, lat, lon)

    valid_tmax = (~miss).sum(axis=0)                   # (lat, lon)
    valid_thr = (~thr_missing).sum(axis=0)
    decidable = (~miss & ~thr_missing).sum(axis=0)
    maxmiss = max_consec_missing(miss.reshape(153, -1)).reshape(nlat, nlon)

    # only valid (land) cells
    vt = valid_tmax[valid]
    dd = decidable[valid]
    mm = maxmiss[valid]
    is_zero = (count[yi][valid] == 0) & (days[yi][valid] == 0) & (maxd[yi][valid] == 0)

    n_valid_cellyears += valid.sum()
    n_zero_output += int(is_zero.sum())
    n_zero_and_nomissing += int((is_zero & (dd > 0)).sum())
    n_zero_undef += int((is_zero & (dd == 0)).sum())
    n_zero_allmissing += int((is_zero & (vt == 0)).sum())
    n_zero_thrinvalid += int((is_zero & (valid_thr[valid] == 0)).sum())

    for v in np.unique(dd):
        hist_decidable[int(v)] += int((dd == v).sum())
    for k in n_any_gap_over:
        n_any_gap_over[k] += int((mm > k).sum())

nc.close()

print("=" * 72)
print("CHECK 1 -- per-cell-year completeness (May-Sep, 153 days)")
print("=" * 72)
print(f"valid cells           : {int(valid.sum())}")
print(f"valid cell-years      : {n_valid_cellyears}  (= {int(valid.sum())} x {n_years})")
print()
print("saved 0/0/0 outputs among valid cell-years:")
print(f"  hw_count=0 & hw_days=0 & hw_max_dur=0 : {n_zero_output}")
print(f"    - with decidable days > 0 (genuine 'no heatwave'): {n_zero_and_nomissing}")
print(f"    - with decidable days = 0 (UNDEFINED, should be NaN): {n_zero_undef}")
print(f"      of which all Tmax missing   : {n_zero_allmissing}")
print(f"      of which all thresholds NaN : {n_zero_thrinvalid}")
print()
print("distribution of decidable days over valid cell-years (value: count):")
for v in [0, 1, 5, 10, 20, 50, 76, 100, 130, 140, 150, 152, 153]:
    print(f"    decidable == {v:3d}: {hist_decidable[v]}")
print(f"    decidable >= 150   : {int(hist_decidable[150:].sum())}")
print(f"    decidable == 153   : {hist_decidable[153]}")
print()
print("max consecutive missing-Tmax run (within season) over valid cell-years:")
for k, v in n_any_gap_over.items():
    print(f"    gap > {k:2d} days : {v} cell-years")
print()
print("Interpretation:")
print("  - A cell-year with decidable==0 cannot support any conclusion; the saved")
print("    0/0/0 there is indistinguishable from a genuine heatwave-free year.")
print("  - A cell-year with a long missing run can hide a >=3-day event inside the")
print("    gap (events are not connected across it, and the hidden days are not")
print("    counted), so 0/0/0 or a low count may under-report.")
