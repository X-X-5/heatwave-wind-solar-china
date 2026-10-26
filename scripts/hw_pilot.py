# -*- coding: utf-8 -*-
"""
Phase 1 pilot: validate the heatwave algorithm BEFORE running the full dataset.

Part A -- synthetic unit tests (deterministic, no data file needed) covering:
    A1  2 consecutive hot days  -> NOT a heatwave
    A2  3 consecutive hot days  -> one heatwave (duration 3)
    A3  4 consecutive hot days  -> one heatwave (duration 4)
    A4  two separated events    -> count=2, total days = sum
    A5  a missing day breaks a run (events are NOT connected across a gap)
    A6  threshold percentile + min-sample rule + NaN handling
    A7  cross-month run (Jul 30 - Aug 1) -> one event
    A8  season boundary: April tail is excluded; May 1 start is respected
Part B -- real small-region smoke test (well-covered box, a few years) to
          confirm the pipeline runs end-to-end and produces plausible values.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_lib as hw

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def events1d(hot, excess=None):
    """Convenience wrapper: run _events_2d on a single-year 1D hot mask."""
    hot = np.asarray(hot, dtype=bool).reshape(1, -1)
    if excess is None:
        excess = np.zeros_like(hot, dtype=float)
    else:
        excess = np.asarray(excess, dtype=float).reshape(1, -1)
    return hw._events_2d(hot, excess, 3)


print("=" * 70)
print("PART A -- synthetic unit tests")
print("=" * 70)

# A1: 2 consecutive -> no event
ne, td, md, se, hd = events1d([1, 1, 0, 0, 0])
check("A1 two-day run is not a heatwave", ne[0] == 0 and td[0] == 0 and md[0] == 0,
      f"ne={ne[0]} td={td[0]}")

# A2: 3 consecutive -> one event, duration 3
ne, td, md, se, hd = events1d([1, 1, 1, 0, 0])
check("A2 three-day run -> 1 event / 3 days", ne[0] == 1 and td[0] == 3 and md[0] == 3,
      f"ne={ne[0]} td={td[0]} md={md[0]}")

# A3: 4 consecutive -> one event, duration 4
ne, td, md, se, hd = events1d([1, 1, 1, 1, 0])
check("A3 four-day run -> 1 event / 4 days", ne[0] == 1 and td[0] == 4 and md[0] == 4,
      f"ne={ne[0]} td={td[0]} md={md[0]}")

# A4: two separated events
ne, td, md, se, hd = events1d([1, 1, 1, 0, 0, 1, 1, 1, 1, 0])
check("A4 two events -> count 2 / 7 days / max 4",
      ne[0] == 2 and td[0] == 7 and md[0] == 4,
      f"ne={ne[0]} td={td[0]} md={md[0]}")

# A5: missing day in the middle breaks the run (missing encoded as not-hot)
#     [hot,hot,missing,hot,hot,hot] -> left run len2 (no event), right run len3 (event)
ne, td, md, se, hd = events1d([1, 1, 0, 1, 1, 1])
check("A5 missing day breaks run (no cross-gap connection)",
      ne[0] == 1 and td[0] == 3 and md[0] == 3,
      f"ne={ne[0]} td={td[0]} md={md[0]}")

# A5b: missing -> only a 2-day run remains on each side -> no event
ne, td, md, se, hd = events1d([1, 1, 0, 1, 1])
check("A5b gap leaves two 2-day runs -> no heatwave",
      ne[0] == 0 and td[0] == 0, f"ne={ne[0]} td={td[0]}")

# A5c: intensity -- only heatwave days contribute, missing day never does
ne, td, md, se, hd = events1d([1, 1, 1, 0, 0], excess=[2.0, 4.0, 6.0, 0.0, 0.0])
check("A5c intensity sums only heatwave days", se[0] == 12.0 and hd[0] == 3,
      f"se={se[0]} hd={hd[0]}")

# A6: threshold percentile + min-sample + NaN
# Build a single cell, 366 slots, 30 "years" with a 5-day window.
# Use a controlled value per slot so the pooled window percentile is exact.
rng = np.random.default_rng(0)
n_slots = 366
n_years = 30
# value for each slot grows linearly: v = slot (0..365)
v = np.arange(n_slots, dtype=float)
X = np.full((n_years * n_slots, 1), np.nan)
slot = np.zeros(n_years * n_slots, dtype=np.int64)
row = 0
for y in range(n_years):
    for s in range(n_slots):
        X[row, 0] = v[s]
        slot[row] = s
        row += 1
thr = hw.compute_thresholds(X, slot, q=90.0, min_samples=100)
# Reference: pooled sample for slot s = n_years copies of each of the 5 window
# values {s-2..s+2}; the 90th percentile is over that 150-value sample.
def ref_pooled(s):
    vals = []
    for k in range(s - 2, s + 3):
        vals += [v[k % n_slots]] * n_years
    return np.array(vals, dtype=float)

ok_slots = True
for s in range(2, 364):
    if not np.allclose(thr[s, 0], np.percentile(ref_pooled(s), 90.0)):
        ok_slots = False
        break
check("A6a percentile of pooled 5-day window matches numpy reference", ok_slots)

# A6b: min-sample rule -> set one cell to mostly missing
X2 = X.copy()
X2[:, 0] = np.nan  # all missing -> threshold must be NaN everywhere
thr2 = hw.compute_thresholds(X2, slot, q=90.0, min_samples=100)
check("A6b all-missing cell -> NaN thresholds", bool(np.all(np.isnan(thr2))))

# A6c: NaN values are excluded from the percentile sample
X3 = X.copy()
# poison one of the 5 window members (slot 100) with NaN for the cell
X3[slot == 100, 0] = np.nan
thr3 = hw.compute_thresholds(X3, slot, q=90.0, min_samples=100)
# window for slot 102 = {100,101,102,103,104}; slot 100 is now NaN -> 4 valid
ref = []
for k in [101, 102, 103, 104]:
    ref += [v[k]] * n_years
expected_102 = np.percentile(np.array(ref, dtype=float), 90.0)
check("A6c NaN excluded from percentile sample", bool(np.allclose(thr3[102, 0], expected_102)))

print()
print("=" * 70)
print("PART A2 -- calendar / season integration (synthetic 2-cell series)")
print("=" * 70)

# Build a tiny fake 2-year daily series (2010-2011) for one cell with a
# constant threshold, inject hot days, and verify cross-month + season boundary.
import pandas as pd

dates = pd.date_range("2010-01-01", "2011-12-31", freq="D")
month = dates.month.values.astype(np.int64)
day = dates.day.values.astype(np.int64)
year = dates.year.values.astype(np.int64)
slot_all = hw.slot_index(month, day)
season = hw.season_mask(month, day)
season_idx = np.where(season)[0]
slot_season = slot_all[season_idx]
n_years = 2
days_per_year = 153  # May..Sep inclusive

# constant threshold 30 degC, Tmax 31 everywhere -> NOT hot only where we set below
tmax = np.full(len(dates), 29.0)          # below threshold by default
thr_366 = np.full(366, 30.0)

# ---- A7: cross-month event Jul 30 / 31 / Aug 1 (2010) ----
for d in dates:
    if (d.year == 2010) and (d.month == 7 and d.day >= 30) or (d.month == 8 and d.day == 1):
        tmax[(dates == d)] = 31.0
# but this sets Jul 30,31, Aug 1 -> 3 consecutive hot days crossing month boundary
ne, td, md, intensity = hw.detect_cell_metrics(tmax, thr_366, season_idx, slot_season, n_years, days_per_year)
# 2010 should have exactly one 3-day event; 2011 zero
check("A7 cross-month (Jul30-Aug1) -> 1 event / 3 days in 2010",
      ne[0] == 1 and td[0] == 3 and md[0] == 3, f"ne0={ne[0]} td0={td[0]}")
check("A7 2011 clean (no events)", ne[1] == 0 and td[1] == 0, f"ne1={ne[1]}")

# ---- A8: season boundary ----
# April 29-30 hot + May 1-2 hot, but Apr is outside season -> only May 1-2
# remain -> 2-day run -> NOT a heatwave. May 1-3 hot -> event.
tmax2 = np.full(len(dates), 29.0)
for d in dates:
    if d.year == 2010 and ((d.month == 4 and d.day >= 29) or (d.month == 5 and d.day <= 3)):
        tmax2[(dates == d)] = 31.0
ne, td, md, intensity = hw.detect_cell_metrics(tmax2, thr_366, season_idx, slot_season, n_years, days_per_year)
# April 29,30 dropped; remaining May 1,2,3 -> 3-day event
check("A8 season boundary: Apr tail excluded, May1-3 -> 1 event / 3 days",
      ne[0] == 1 and td[0] == 3, f"ne0={ne[0]} td0={td[0]}")

# Sep 30 boundary: Sep 29,30 + Oct 1,2 hot -> only Sep 29,30 remain (2-day, no event)
tmax3 = np.full(len(dates), 29.0)
for d in dates:
    if d.year == 2010 and ((d.month == 9 and d.day >= 29) or (d.month == 10 and d.day <= 2)):
        tmax3[(dates == d)] = 31.0
ne, td, md, intensity = hw.detect_cell_metrics(tmax3, thr_366, season_idx, slot_season, n_years, days_per_year)
check("A8 season boundary: Sep29-30 + Oct1-2 -> Oct excluded -> no event",
      ne[0] == 0 and td[0] == 0, f"ne0={ne[0]} td0={td[0]}")

# intensity check on A8's event
ne, td, md, intensity = hw.detect_cell_metrics(tmax2, thr_366, season_idx, slot_season, n_years, days_per_year)
check("A8 intensity = mean exceedance = 1.0 degC", np.allclose(intensity[0], 1.0),
      f"intensity0={intensity[0]}")

print()
print("=" * 70)
print("PART B -- real small-region smoke test")
print("=" * 70)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")

from netCDF4 import Dataset

nc = Dataset(NC, "r")
var = nc.variables["tmax"]
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]

# full time coordinate
t = nc.variables["time"][:]
origin = pd.Timestamp("1961-01-01 00:00:00")
dates = origin + pd.to_timedelta(t, unit="h")
month = dates.month.values.astype(np.int64)
day = dates.day.values.astype(np.int64)
year = dates.year.values.astype(np.int64)
slot_all = hw.slot_index(month, day)
season = hw.season_mask(month, day)
season_idx = np.where(season)[0]
slot_season = slot_all[season_idx]
n_years = 62
days_per_year = 153
baseline_idx = np.where((year >= 1981) & (year <= 2010))[0]

# sub-region: eastern China, well-covered (lon 112..121, lat 29..35)
lat_sel = (lat >= 29.0) & (lat <= 35.0)
lon_sel = (lon >= 112.0) & (lon <= 121.0)
sub = var[:, lat_sel, lon_sel]
data = np.ma.filled(sub.astype(np.float32), np.float32(np.nan))
n_lat, n_lon = data.shape[1], data.shape[2]
print(f"  region shape (time,lat,lon) = {data.shape}")

data2d = data.reshape(data.shape[0], -1)          # (n_days, n_lat*n_lon)
thr2d = hw.compute_thresholds(data2d[baseline_idx], slot_all[baseline_idx])
thr_full = thr2d.reshape(366, n_lat, n_lon)

# run detection for every cell in the sub-region
cnt = np.zeros((n_years, n_lat, n_lon), np.int32)
days = np.zeros((n_years, n_lat, n_lon), np.int32)
maxd = np.zeros((n_years, n_lat, n_lon), np.int32)
intens = np.full((n_years, n_lat, n_lon), np.nan)
for j in range(n_lat):
    for k in range(n_lon):
        col = data[:, j, k]
        if np.isnan(col).all():
            continue
        ne, td, md, it = hw.detect_cell_metrics(col, thr_full[:, j, k],
                                                season_idx, slot_season,
                                                n_years, days_per_year)
        cnt[:, j, k] = ne
        days[:, j, k] = td
        maxd[:, j, k] = md
        intens[:, j, k] = it

finite_cells = int(np.isfinite(days).sum())
max_annual_days = float(np.nanmax(days))
print(f"  active cells (any valid data): {int((~np.isnan(data).all(axis=0)).sum())} / {n_lat*n_lon}")
print(f"  finite annual-day entries  : {finite_cells}")
print(f"  max annual heatwave days (any cell, any year) : {max_annual_days}")
print(f"  sanity range check         : all annual days in [0, 153] -> "
      f"{bool(np.nanmin(days) >= 0 and np.nanmax(days) <= 153)}")
print(f"  sample: mean annual heatwave days over region (2000) = "
      f"{float(np.nanmean(days[39])):.2f}  (1961=idx0 -> 2000=idx39)")

# print a small table for a few representative cells
print("  representative cells (lon, lat) -> mean annual days / max duration (1961-2022):")
for (jj, kk) in [(6, 18), (3, 15), (1, 10)]:   # a few interior cells
    ll = lon[lon_sel][kk]
    la = lat[lat_sel][jj]
    d = days[:, jj, kk]
    m = maxd[:, jj, kk]
    print(f"    lon={ll:.2f} lat={la:.2f} : mean_days={np.nanmean(d):.1f}  "
          f"max_dur={np.nanmax(m)}")

nc.close()

print()
print("=" * 70)
print("SUMMARY")
print(f"  PASS: {len(PASS)}   FAIL: {len(FAIL)}")
for f in FAIL:
    print("   FAILED ->", f)
print("=" * 70)
sys.exit(1 if FAIL else 0)
