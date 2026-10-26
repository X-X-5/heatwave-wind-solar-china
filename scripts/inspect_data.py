# -*- coding: utf-8 -*-
"""
Phase 1 — read-only data inspection for CN05.1 daily Tmax.

File: data/CN05.1_Tmax_1961-2022_daily_025x025.nc

This script:
  1. dumps dimensions / variables / attributes,
  2. decodes and validates the time coordinate (calendar, leap days,
     duplicates, gaps),
  3. validates the lon/lat coordinates (range, order, resolution),
  4. reads tmax in chunks and reports valid coverage and missing-data
     distribution without ever loading the full array.

It never modifies or overwrites the source NetCDF.
"""
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")

LINES = []
def log(s=""):
    print(s)
    LINES.append(s)

# ---------------------------------------------------------------- metadata
nc = Dataset(NC, "r")

log("=" * 70)
log("FILE")
log(f"  path      : {NC}")
log(f"  size      : {os.path.getsize(NC)/1e9:.3f} GB")
log("=" * 70)
log("DIMENSIONS")
for k, v in nc.dimensions.items():
    log(f"  {k:<8} size={len(v):>7}  unlimited={v.isunlimited()}")
log("VARIABLES")
for k, v in nc.variables.items():
    log(f"  {k:<8} dtype={str(v.dtype):<8} dims={v.dimensions} shape={v.shape}")
log("GLOBAL ATTRIBUTES")
for k in nc.ncattrs():
    log(f"  {k} = {nc.getncattr(k)}")
log("VARIABLE ATTRIBUTES")
for name in ["time", "lon", "lat", "tmax"]:
    v = nc.variables[name]
    for k in v.ncattrs():
        log(f"  {name}.{k} = {v.getncattr(k)}")

# ---------------------------------------------------------------- time
t = nc.variables["time"][:]
t_units = nc.variables["time"].units
t_cal = nc.variables["time"].calendar
log("=" * 70)
log("TIME")
log(f"  units    : {t_units}")
log(f"  calendar : {t_cal}")
log(f"  n        : {t.size}")
log(f"  first/last raw : {t[0]} .. {t[-1]}")

# Decode with pandas (calendar=standard == proleptic Gregorian; no leap seconds).
origin = pd.Timestamp("1961-01-01 00:00:00")
dates = origin + pd.to_timedelta(t, unit="h")
dates = dates.round("s")
log(f"  first date : {dates[0]}")
log(f"  last date  : {dates[-1]}")
log(f"  date dtype : {dates.dtype}")

# Expected continuous daily sequence
expected = pd.date_range(dates[0], dates[-1], freq="D")
log(f"  expected n (contiguous daily) : {expected.size}")
log(f"  actual n                       : {dates.size}")

dup = dates.duplicated()
missing_dates = expected.difference(dates)
extra_dates = dates.difference(expected)
log(f"  duplicate date count : {int(dup.sum())}")
log(f"  missing date count   : {len(missing_dates)}")
log(f"  extra date count     : {len(extra_dates)}")
if len(missing_dates):
    log("  missing dates sample: " + ", ".join(map(str, missing_dates[:10])))
if len(extra_dates):
    log("  extra dates sample: " + ", ".join(map(str, extra_dates[:10])))

# leap days
is_leap_day = (dates.month == 2) & (dates.day == 29)
log(f"  Feb-29 count        : {int(is_leap_day.sum())}")
leap_years = sorted(set(dates[is_leap_day].year))
log(f"  leap years present  : {leap_years}")

# time-step regularity
dt = np.diff(t)
log(f"  time step (h) unique: {np.unique(dt)}")

# ---------------------------------------------------------------- lon / lat
lon = nc.variables["lon"][:]
lat = nc.variables["lat"][:]
log("=" * 70)
log("LON / LAT")
log(f"  lon: n={lon.size}  min={lon.min()} max={lon.max()}")
log(f"  lat: n={lat.size}  min={lat.min()} max={lat.max()}")
log(f"  lon[:3]={lon[:3]}  lon[-3:]={lon[-3:]}")
log(f"  lat[:3]={lat[:3]}  lat[-3:]={lat[-3:]}")
log(f"  dlon unique={np.unique(np.round(np.diff(lon), 6))}")
log(f"  dlat unique={np.unique(np.round(np.diff(lat), 6))}")
log(f"  lon monotonic increasing : {bool(np.all(np.diff(lon) > 0))}")
log(f"  lat monotonic increasing : {bool(np.all(np.diff(lat) > 0))}")

# ---------------------------------------------------------------- tmax coverage
log("=" * 70)
log("TMAX DATA COVERAGE (chunked read over time)")
var = nc.variables["tmax"]
fill = var._FillValue if hasattr(var, "_FillValue") else None
log(f"  dtype      : {var.dtype}")
log(f"  _FillValue : {fill}")

nt, ny, nx = var.shape
# accumulate in chunks of one year to bound memory (~365*163*283*4 ~ 67 MB/chunk)
years = np.unique(dates.year)
log(f"  years      : {years[0]} .. {years[-1]} ({len(years)} years)")

total_valid = 0
total_missing = 0
sum_val = 0.0
vmin = np.inf
vmax = -np.inf
# per-grid-cell accumulation (over full record)
valid_count = np.zeros((ny, nx), dtype=np.int64)
missing_count = np.zeros((ny, nx), dtype=np.int64)
# per-year missing fraction
year_rows = []
# any day that is fully missing
fully_missing_days = 0

for yr in years:
    sel = dates.year == yr
    idx = np.where(sel)[0]
    i0, i1 = idx[0], idx[-1] + 1
    arr = var[i0:i1]  # masked array (auto-mask on fill value)
    data = np.ma.getdata(arr)
    mask = np.ma.getmaskarray(arr)
    n_valid = int((~mask).sum())
    n_missing = int(mask.sum())
    total_valid += n_valid
    total_missing += n_missing
    if n_valid > 0:
        vv = data[~mask]
        sum_val += float(vv.sum())
        vmin = min(vmin, float(vv.min()))
        vmax = max(vmax, float(vv.max()))
    valid_count += (~mask).sum(axis=0)
    missing_count += mask.sum(axis=0)
    # fully missing days in this year
    fully_missing_days += int(mask.all(axis=(1, 2)).sum())
    year_rows.append((int(yr), n_valid, n_missing, n_missing / (n_valid + n_missing)))

log(f"  total valid   : {total_valid}")
log(f"  total missing : {total_missing}")
log(f"  missing frac  : {total_missing/(total_valid+total_missing):.5f}")
log(f"  valid range   : [{vmin:.3f}, {vmax:.3f}]  (stored units, see below)")
log(f"  fully-missing days (all grid masked): {fully_missing_days}")

log("  per-year missing fraction:")
yr_df = pd.DataFrame(year_rows, columns=["year", "valid", "missing", "miss_frac"])
for _, r in yr_df.iterrows():
    log(f"    {int(r.year)}: missing={r.missing:>9}  frac={r.miss_frac:.5f}")

# spatial missing pattern
always_missing = missing_count == nt  # never any valid value
never_missing = missing_count == 0   # always valid
partial = (~always_missing) & (~never_missing)
log("  spatial missing pattern (over full record):")
log(f"    grid cells always valid      : {int(never_missing.sum())} / {ny*nx}")
log(f"    grid cells always missing    : {int(always_missing.sum())} / {ny*nx}")
log(f"    grid cells partially missing : {int(partial.sum())} / {ny*nx}")

if always_missing.any():
    ii, jj = np.where(always_missing)
    log(f"    always-missing cell lats    : {np.unique(lat[ii])}")
    log(f"    always-missing cell lons    : {np.unique(lon[jj])}")

# where are the partially missing cells (roughly)?
if partial.any():
    ii, jj = np.where(partial)
    log(f"    partial cells: lat range {lat[ii].min()}..{lat[ii].max()}, "
        f"lon range {lon[jj].min()}..{lon[jj].max()}")

nc.close()

log("=" * 70)
log("DONE (read-only; source file untouched)")

# write report
out = os.path.join(ROOT, "results", "data_check_report.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
log(f"Report written to: {out}")
