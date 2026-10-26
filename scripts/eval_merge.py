# -*- coding: utf-8 -*-
"""
Phase-1 diagnostic (read-only): heatwave SEGMENT-MERGING evaluation.

The formal definition (A) is: a heatwave = >= 3 CONSECUTIVE days with
Tmax strictly > threshold. This script does NOT change that definition; it only
produces a diagnostic comparison against an alternative grouping.

Definition B (diagnostic only): first identify the same >= 3-day hot segments,
then MERGE segments separated by <= 2 COMPLETE non-exceedance ("cool") days into
a single "process". Rules:
  * a missing day is never a mergeable gap (segments on either side stay apart);
  * the cool gap days are NOT counted as heatwave days;
  * process span = days from the first to the last hot day (includes cool gaps);
  * hot days = exceedance days only; gap days = cool days inside the process;
  * merging chains transitively (repeated <=2-day cool gaps form one process).

Outputs diagnostics only: 5 synthetic validation cases, then a small-region
comparison of 2022 vs a normal year (2000). Nothing is written to the formal
results.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_lib as hw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")
RES = os.path.join(ROOT, "results")          # thresholds are unchanged by the fix

MAX_GAP = 2


# ---------------------------------------------------------------- helpers
def find_segments(hot, min_run=3):
    """Maximal runs of consecutive hot days with length >= min_run -> [(s,e)]."""
    segs = []
    n = len(hot)
    i = 0
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            if j - i >= min_run:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def merge_processes(segs, cool, missing, max_gap=MAX_GAP):
    """Merge adjacent segments separated by <= max_gap complete cool days.

    Returns list of processes (start, end) inclusive (end = last hot day).
    A missing day anywhere in the gap forbids merging.
    """
    if not segs:
        return []
    procs = []
    cur_s, cur_e = segs[0]
    for (s, e) in segs[1:]:
        gap = list(range(cur_e + 1, s))
        gap_ok = (len(gap) <= max_gap
                  and all(cool[d] for d in gap)
                  and not any(missing[d] for d in gap))
        if gap_ok:
            cur_e = e
        else:
            procs.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    procs.append((cur_s, cur_e))
    return procs


def summarize(hot, cool, missing):
    segs = find_segments(hot)
    A = (len(segs),                                    # events
         sum(e - s + 1 for s, e in segs),              # hot days
         max((e - s + 1 for s, e in segs), default=0))  # max duration

    procs = merge_processes(segs, cool, missing)
    hot_days = sum(int(hot[s:e + 1].sum()) for s, e in procs)
    spans = [e - s + 1 for s, e in procs]
    gap_days = sum(int(cool[s:e + 1].sum()) for s, e in procs)
    B = (len(procs),                                   # processes
         hot_days,                                     # hot days (same as A)
         sum(spans),                                   # total process span
         max(spans, default=0),                        # longest process span
         gap_days)                                     # total gap days
    return A, B, segs, procs


def state_from_string(s):
    hot = np.array([c == "H" for c in s], dtype=bool)
    cool = np.array([c == "C" for c in s], dtype=bool)
    missing = np.array([c == "M" for c in s], dtype=bool)
    return hot, cool, missing


# ---------------------------------------------------------------- synthetic cases
print("=" * 72)
print("SYNTHETIC VALIDATION CASES  (H=hot, C=cool, M=missing)")
print("=" * 72)
print("  A = strict >=3 consecutive days   B = merge segments with <=2 cool-day gap")
print()

cases = [
    ("hot4-cool2-hot3", "HHHHCCHHH", None),
    ("hot2-cool2-hot2", "HHCCHH", None),
    ("hot3-miss1-hot3", "HHHMHHH", None),
    ("hot3-cool3-hot3", "HHHCCCHHH", None),
    ("hot3-c2-hot3-c2-hot3", "HHHCCHHHCCHHH", None),
]

for label, s, _ in cases:
    hot, cool, missing = state_from_string(s)
    A, B, segs, procs = summarize(hot, cool, missing)
    print(f"  {label:24s}  ({s})")
    print(f"      A: events={A[0]}, hot_days={A[1]}, max_dur={A[2]}")
    print(f"      B: processes={B[0]}, hot_days={B[1]}, total_span={B[2]}, "
          f"max_span={B[3]}, gap_days={B[4]}")
    print()

print("  expected:")
print("    hot4-cool2-hot3        -> A: 2 events (4,3)          B: 1 process span=9 hot=7 gap=2")
print("    hot2-cool2-hot2        -> A: 0 (both <3)             B: 0")
print("    hot3-miss1-hot3        -> A: 2 events                B: 2 processes (missing NOT mergeable)")
print("    hot3-cool3-hot3        -> A: 2 events                B: 2 processes (gap=3 > 2)")
print("    hot3-c2-hot3-c2-hot3   -> A: 3 events                B: 1 process span=13 hot=9 gap=4 (chained)")

# ---------------------------------------------------------------- small region
print()
print("=" * 72)
print("SMALL-REGION COMPARISON: 2022 vs 2000 (normal year)")
print("=" * 72)

nc = Dataset(NC, "r")
var = nc.variables["tmax"]
lat_full = nc.variables["lat"][:]
lon_full = nc.variables["lon"][:]
t = nc.variables["time"][:]
origin = pd.Timestamp("1961-01-01 00:00:00")
dates = origin + pd.to_timedelta(t, unit="h")
month = dates.month.values.astype(np.int64)
day = dates.day.values.astype(np.int64)
year = dates.year.values.astype(np.int64)
season = hw.season_mask(month, day)

thr_ds = xr.open_dataset(os.path.join(RES, "hw_threshold_1981-2010.nc"))
thr366 = thr_ds["threshold"].values.astype(float)      # (366, lat, lon)
thr_ds.close()

LAT0, LAT1 = 30.0, 33.0
LON0, LON1 = 112.0, 121.0
i0 = int(np.argmin(np.abs(lat_full - LAT0)))
i1 = int(np.argmin(np.abs(lat_full - LAT1))) + 1
j0 = int(np.argmin(np.abs(lon_full - LON0)))
j1 = int(np.argmin(np.abs(lon_full - LON1))) + 1
print(f"region: lat {lat_full[i0]:.2f}..{lat_full[i1-1]:.2f}N, "
      f"lon {lon_full[j0]:.2f}..{lon_full[j1-1]:.2f}E  "
      f"({i1-i0} lat x {j1-j0} lon = {(i1-i0)*(j1-j0)} cells)")

season_slots = np.arange(121, 274, dtype=np.int64)      # May 1 .. Sep 30


def cell_state(Y, ii, jj):
    sel = np.where(season & (year == Y))[0]
    arr = var[sel[0]:sel[-1] + 1]                        # (153, nlat, nlon)
    data = np.ma.filled(arr.astype(np.float32), np.float32(np.nan))
    tmax_s = data[:, ii, jj]
    thr_s = thr366[:, ii, jj][season_slots]
    finite = np.isfinite(tmax_s) & np.isfinite(thr_s)
    hot = finite & (tmax_s > thr_s)
    cool = finite & (tmax_s <= thr_s)
    missing = ~finite
    return hot, cool, missing, tmax_s


def region_stats(Y):
    n_ev_A = 0
    hot_A = 0
    max_dur_A = 0
    n_proc_B = 0
    span_B = 0
    gap_B = 0
    max_span_B = 0
    n_used = 0
    per_cell = []
    for ii in range(i0, i1):
        for jj in range(j0, j1):
            hot, cool, missing, _ = cell_state(Y, ii, jj)
            if (~missing).sum() == 0:            # skip fully-missing cell-year
                continue
            A, B, segs, procs = summarize(hot, cool, missing)
            n_ev_A += A[0]
            hot_A += A[1]
            max_dur_A = max(max_dur_A, A[2])
            n_proc_B += B[0]
            span_B += B[2]
            gap_B += B[4]
            max_span_B = max(max_span_B, B[3])
            n_used += 1
            per_cell.append((ii, jj, A, B))
    return dict(n_ev_A=n_ev_A, hot_A=hot_A, max_dur_A=max_dur_A,
                n_proc_B=n_proc_B, span_B=span_B, gap_B=gap_B,
                max_span_B=max_span_B), n_used, per_cell


for Y in [2000, 2022]:
    s, n_used, per_cell = region_stats(Y)
    print(f"\n--- year {Y}  ({n_used} valid cell-years) ---")
    print(f"  A (strict):   events={s['n_ev_A']:5d}   hot_days={s['hot_A']:5d}   "
          f"longest_event={s['max_dur_A']:3d} d")
    print(f"  B (merged):   processes={s['n_proc_B']:5d}   hot_days={s['hot_A']:5d}   "
          f"total_span={s['span_B']:5d} d   longest_process={s['max_span_B']:3d} d   "
          f"gap_days={s['gap_B']:5d}")
    print(f"  mean per cell: A events={s['n_ev_A']/n_used:.3f}, "
          f"B processes={s['n_proc_B']/n_used:.3f}, hot_days={s['hot_A']/n_used:.3f}")

# ---- a representative cell (max 2022 hot days) + its merged processes
print("\n--- representative cell: max 2022 hot days in region, process listing ---")
best = None
for (ii, jj, A, B) in region_stats(2022)[2]:
    if best is None or A[1] > best[2][1]:
        best = (ii, jj, A, B)
ii, jj, A, B = best
hot, cool, missing, tmax_s = cell_state(2022, ii, jj)
segs = find_segments(hot)
procs = merge_processes(segs, cool, missing)
d0 = pd.Timestamp(f"{2022}-05-01")
print(f"  cell lon={lon_full[jj]:.2f} lat={lat_full[ii]:.2f}  (A: {A[0]} events, "
      f"{A[1]} hot days, max_dur {A[2]};  B: {B[0]} processes, max_span {B[3]})")
print(f"  segments (A events): "
      + ", ".join(f"{(d0 + pd.Timedelta(s, 'D')).date()}~{(d0 + pd.Timedelta(e, 'D')).date()}"
                  f"({e - s + 1}d)" for s, e in segs))
for (s, e) in procs:
    span = e - s + 1
    hotn = int(hot[s:e + 1].sum())
    gapn = int(cool[s:e + 1].sum())
    print(f"  process (B): {(d0 + pd.Timedelta(s, 'D')).date()} ~ "
          f"{(d0 + pd.Timedelta(e, 'D')).date()}  span={span}d  hot={hotn}d  gap={gapn}d")

nc.close()
print()
print("DONE merge evaluation (diagnostic only; formal definition unchanged).")
