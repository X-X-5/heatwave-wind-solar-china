# -*- coding: utf-8 -*-
"""
Phase-2 pilot: merged heatwave definition (gap < 2 days) over the 2022
small-region box 30-33 N, 112-121 E.

Reuses the (unchanged) phase-1 calendar-day P90 thresholds from
results_fixed/hw_threshold_1981-2010.nc; only the event GROUPING changes.

Outputs (new, independent directory results_merge/):
  * hw_2022_daily_labels.nc      per-day classification codes (+ tmax/threshold)
  * hw_2022_events_merged.csv    per-cell merged-event catalog
  * hw_2022_region_summary.txt   region totals (strict / gap<2 / gap<=2)
  * fig_hw_2022_merge_cell.png   representative merged cell time series

Read-only w.r.t. source data and all existing results. Does not fill missing,
does not interpolate, does not download anything.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_merge_lib as m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")
THR_NC = os.path.join(ROOT, "results_fixed", "hw_threshold_1981-2010.nc")
OUT = os.path.join(ROOT, "results_merge")
os.makedirs(OUT, exist_ok=True)

LAT0, LAT1 = 30.0, 33.0
LON0, LON1 = 112.0, 121.0
YEAR = 2022

# ---------------------------------------------------------------- region + time
nc = Dataset(NC, "r")
var = nc.variables["tmax"]
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]
t = nc.variables["time"][:]
origin = pd.Timestamp("1961-01-01 00:00:00")
dates = origin + pd.to_timedelta(t, unit="h")
year = dates.year.values.astype(np.int64)
month = dates.month.values.astype(np.int64)
day = dates.day.values.astype(np.int64)

i0 = int(np.argmin(np.abs(lat - LAT0)))
i1 = int(np.argmin(np.abs(lat - LAT1))) + 1
j0 = int(np.argmin(np.abs(lon - LON0)))
j1 = int(np.argmin(np.abs(lon - LON1))) + 1
lat_r = lat[i0:i1]
lon_r = lon[j0:j1]
print(f"region: lat {lat_r[0]:.2f}..{lat_r[-1]:.2f} N, "
      f"lon {lon_r[0]:.2f}..{lon_r[-1]:.2f} E  ({len(lat_r)} x {len(lon_r)} "
      f"= {len(lat_r)*len(lon_r)} cells)")

season = (month >= 5) & (month <= 9)
seas_idx = np.where(season & (year == YEAR))[0]
assert len(seas_idx) == 153, f"expected 153 season days, got {len(seas_idx)}"
g0, g1 = int(seas_idx[0]), int(seas_idx[-1]) + 1
d0 = pd.Timestamp(f"{YEAR}-05-01")
d_season = [d0 + pd.Timedelta(days=d) for d in range(153)]
slot_season = np.array([(dt.month, dt.day) for dt in d_season])

# Tmax for the region x 2022 season (small: 153 x 13 x 37)
blk = var[g0:g1, i0:i1, j0:j1]
tmax = np.ma.filled(blk.astype(np.float32), np.float32(np.nan))
nc.close()

# thresholds (unchanged phase-1) for the region, indexed by calendar-day slot
thr_ds = xr.open_dataset(THR_NC)
thr366 = thr_ds["threshold"].values.astype(np.float64)   # (366, 163, 283)
thr_ds.close()
thr_region = thr366[:, i0:i1, j0:j1]                     # (366, 13, 37)

# slot indices 0..365 for each of the 153 season days
def slot_of(mo, dy):
    import hw_lib as hw
    return int(hw.slot_index(np.array([mo]), np.array([dy]))[0])

slots = np.array([slot_of(mo, dy) for (mo, dy) in slot_season], dtype=np.int64)

# ---------------------------------------------------------------- per-cell run
nlat, nlon = len(lat_r), len(lon_r)
labels_all = np.full((153, nlat, nlon), m.LABEL_MISSING, dtype=np.int8)
tmax_all = tmax.copy()
thr_all = np.full((153, nlat, nlon), np.nan, dtype=np.float32)

events_rows = []
# summary accumulators: strict (segments) vs gap<2 (formal) vs gap<=2 (diag)
summ = dict(n_valid=0,
            strict_events=0, strict_hot=0, strict_maxdur=0,
            m1_processes=0, m1_span=0, m1_gap=0, m1_maxspan=0, m1_hot=0,
            m2_processes=0, m2_span=0, m2_gap=0, m2_maxspan=0, m2_hot=0,
            n_cells_with_merge=0)

for ii in range(nlat):
    for jj in range(nlon):
        col = tmax[:, ii, jj]                      # (153,)
        if not np.isfinite(col).any():
            continue
        th = thr_region[slots, ii, jj]             # (153,)
        finite = np.isfinite(col) & np.isfinite(th)
        if not finite.any():
            continue
        summ["n_valid"] += 1
        hot = finite & (col > th)
        cool = finite & (col <= th)
        missing = ~finite
        excess = col - th

        segs = m.find_qualified_segments(hot)
        procs1 = m.merge_qualified_segments(segs, cool, missing, max_gap=1)
        procs2 = m.merge_qualified_segments(segs, cool, missing, max_gap=2)
        labels = m.classify_days(hot, cool, missing, segs, procs1)
        labels_all[:, ii, jj] = labels
        thr_all[:, ii, jj] = th.astype(np.float32)

        # strict (segments) summary
        summ["strict_events"] += len(segs)
        for (s, e) in segs:
            summ["strict_hot"] += e - s + 1
            summ["strict_maxdur"] = max(summ["strict_maxdur"], e - s + 1)
        # merged summaries
        summ["m1_processes"] += len(procs1)
        summ["m1_hot"] += sum(p["n_hot"] for p in procs1)
        summ["m1_span"] += sum(p["span"] for p in procs1)
        summ["m1_gap"] += sum(p["n_gap"] for p in procs1)
        summ["m1_maxspan"] = max(summ["m1_maxspan"],
                                 max((p["span"] for p in procs1), default=0))
        summ["m2_processes"] += len(procs2)
        summ["m2_hot"] += sum(p["n_hot"] for p in procs2)
        summ["m2_span"] += sum(p["span"] for p in procs2)
        summ["m2_gap"] += sum(p["n_gap"] for p in procs2)
        summ["m2_maxspan"] = max(summ["m2_maxspan"],
                                 max((p["span"] for p in procs2), default=0))
        if any(p["n_gap"] > 0 for p in procs1):
            summ["n_cells_with_merge"] += 1

        # per-event catalog (formal gap<2)
        seg_idx_by_pos = {}
        pos = 0
        for (s, e) in segs:
            seg_idx_by_pos[(s, e)] = np.arange(s, e + 1)
            pos += 1
        for p in procs1:
            idx = np.concatenate([np.arange(s, e + 1) for (s, e) in p["segments"]])
            mean_exc = float(np.nanmean(excess[idx]))
            events_rows.append(dict(
                lon=float(lon_r[jj]), lat=float(lat_r[ii]),
                start=(d0 + pd.Timedelta(days=int(p["start"]))).strftime("%Y-%m-%d"),
                end=(d0 + pd.Timedelta(days=int(p["end"]))).strftime("%Y-%m-%d"),
                n_segments=int(p["n_segments"]),
                heatwave_days=int(p["n_hot"]),
                gap_days=int(p["n_gap"]),
                span_days=int(p["span"]),
                mean_excess_C=round(mean_exc, 3),
            ))

# ---------------------------------------------------------------- region summary
print()
print("=" * 72)
print(f"REGION SUMMARY  {YEAR}  (30-33 N, 112-121 E)  valid cells={summ['n_valid']}")
print("=" * 72)
print(f"  strict (>=3d segments, no merge):")
print(f"    events={summ['strict_events']:5d}   hot_days={summ['strict_hot']:5d}   "
      f"longest={summ['strict_maxdur']:3d} d")
print(f"  merged gap<2  (formal, merge only 1-day cool gap):")
print(f"    processes={summ['m1_processes']:5d}   hot_days={summ['m1_hot']:5d}   "
      f"total_span={summ['m1_span']:5d} d   gap_days={summ['m1_gap']:5d}   "
      f"longest={summ['m1_maxspan']:3d} d")
print(f"  merged gap<=2 (diagnostic, matches old eval_merge):")
print(f"    processes={summ['m2_processes']:5d}   hot_days={summ['m2_hot']:5d}   "
      f"total_span={summ['m2_span']:5d} d   gap_days={summ['m2_gap']:5d}   "
      f"longest={summ['m2_maxspan']:3d} d")
print(f"  cells with >=1 merge under gap<2: {summ['n_cells_with_merge']} / "
      f"{summ['n_valid']}")
print(f"  (hot_days identical across groupings: {summ['strict_hot']} == "
      f"{summ['m1_hot']} == {summ['m2_hot']})")

# ---------------------------------------------------------------- write outputs
labels_da = xr.DataArray(
    labels_all, dims=("date", "lat", "lon"),
    coords={"date": pd.to_datetime(d_season), "lat": lat_r, "lon": lon_r},
    name="label",
)
labels_da.attrs = {
    "long_name": "daily heatwave classification (merged definition, gap<2)",
    "codes": "0=missing/undecidable; 1=cool day; 2=isolated exceedance; "
             "3=heatwave day; 4=gap day (1-day cool gap in merged process)",
}
tmax_da = xr.DataArray(
    tmax_all, dims=("date", "lat", "lon"),
    coords={"date": pd.to_datetime(d_season), "lat": lat_r, "lon": lon_r},
    name="tmax",
)
tmax_da.attrs = {"long_name": "daily max temperature", "units": "degC",
                 "note": "NaN = missing"}
thr_da = xr.DataArray(
    thr_all, dims=("date", "lat", "lon"),
    coords={"date": pd.to_datetime(d_season), "lat": lat_r, "lon": lon_r},
    name="threshold",
)
thr_da.attrs = {"long_name": "calendar-day P90 threshold", "units": "degC"}

ds = xr.Dataset({"label": labels_da, "tmax": tmax_da, "threshold": thr_da})
ds.attrs = {
    "title": "CN05.1 Tmax 2022 daily heatwave classification (merged, gap<2)",
    "region": "30-33 N, 112-121 E",
    "baseline_period": "1981-2010", "percentile": 90,
    "window": "5-day centred (day-2..day+2), cross-year wrap",
    "min_run_days": 3, "season": "May-Sep",
    "merge_rule": "adjacent >=3d segments merge iff complete gap < 2 days "
                  "(exactly 1 valid cool day); missing/isolated-exceedance never merge",
}
ds["label"].encoding = {"dtype": "int8", "_FillValue": np.int8(-1)}
out_labels = os.path.join(OUT, "hw_2022_daily_labels.nc")
ds.to_netcdf(out_labels)
print(f"wrote {out_labels}")

df = pd.DataFrame(events_rows)
out_csv = os.path.join(OUT, "hw_2022_events_merged.csv")
df.to_csv(out_csv, index=False)
print(f"wrote {out_csv}  ({len(df)} merged events)")

with open(os.path.join(OUT, "hw_2022_region_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"CN05.1 Tmax {YEAR} heatwave region summary (30-33N, 112-121E)\n")
    f.write(f"valid cells: {summ['n_valid']}\n")
    f.write(f"strict: events={summ['strict_events']} hot_days={summ['strict_hot']} "
            f"longest={summ['strict_maxdur']}d\n")
    f.write(f"merged gap<2 (formal): processes={summ['m1_processes']} "
            f"hot_days={summ['m1_hot']} total_span={summ['m1_span']} "
            f"gap_days={summ['m1_gap']} longest={summ['m1_maxspan']}d\n")
    f.write(f"merged gap<=2 (diag): processes={summ['m2_processes']} "
            f"hot_days={summ['m2_hot']} total_span={summ['m2_span']} "
            f"gap_days={summ['m2_gap']} longest={summ['m2_maxspan']}d\n")
    f.write(f"cells with >=1 merge (gap<2): {summ['n_cells_with_merge']}\n")

# ---------------------------------------------------------------- representative
# Prefer the phase-1 representative cell (114.50E, 31.75N) if it merges under
# gap<2; otherwise use the cell with the most merged gap days.
def cell_index(lo, la):
    jj = int(np.argmin(np.abs(lon_r - lo)))
    ii = int(np.argmin(np.abs(lat_r - la)))
    return ii, jj

def cell_result(ii, jj):
    col = tmax[:, ii, jj]
    th = thr_region[slots, ii, jj]
    finite = np.isfinite(col) & np.isfinite(th)
    hot = finite & (col > th)
    cool = finite & (col <= th)
    missing = ~finite
    excess = col - th
    segs = m.find_qualified_segments(hot)
    procs1 = m.merge_qualified_segments(segs, cool, missing, max_gap=1)
    labels = m.classify_days(hot, cool, missing, segs, procs1)
    return dict(col=col, th=th, hot=hot, cool=cool, missing=missing,
                excess=excess, segs=segs, procs=procs1, labels=labels)

rep_ii, rep_jj = cell_index(114.50, 31.75)
rep = cell_result(rep_ii, rep_jj)
rep_has_merge = any(p["n_gap"] > 0 for p in rep["procs"])
if not rep_has_merge:
    best = None
    for ii in range(nlat):
        for jj in range(nlon):
            if not np.isfinite(tmax[:, ii, jj]).any():
                continue
            r = cell_result(ii, jj)
            g = sum(p["n_gap"] for p in r["procs"])
            if g > 0 and (best is None or g > best[0]):
                best = (g, ii, jj)
    if best is None:
        print("\nNO merged event (gap<2) found anywhere in the pilot region; "
              "reporting honestly, no representative plot.")
        sys.exit(0)
    rep_ii, rep_jj = best[1], best[2]
    rep = cell_result(rep_ii, rep_jj)
    print(f"\nphase-1 cell 114.50E/31.75N had no gap<2 merge; using cell with "
          f"most gap days instead.")

print(f"\nREPRESENTATIVE CELL  lon={lon_r[rep_jj]:.2f}E  lat={lat_r[rep_ii]:.2f}N")
print(f"  strict: {len(rep['segs'])} segments, {sum(e-s+1 for s,e in rep['segs'])} "
      f"hot days, longest {max((e-s+1 for s,e in rep['segs']), default=0)}d")
print(f"  merged (gap<2): {len(rep['procs'])} processes, "
      f"gap_days={sum(p['n_gap'] for p in rep['procs'])}")
for p in rep["procs"]:
    s = (d0 + pd.Timedelta(days=int(p["start"]))).strftime("%m-%d")
    e = (d0 + pd.Timedelta(days=int(p["end"]))).strftime("%m-%d")
    print(f"    {s} ~ {e}  segs={p['n_segments']}  hot={p['n_hot']}d  "
          f"gap={p['n_gap']}d  span={p['span']}d")

# The representative-cell figure is drawn by scripts/hw_merge_plot.py, which
# reads the saved outputs above (labels nc + events csv) without recomputing.
print("\nDONE phase-2 merge pilot (figure via scripts/hw_merge_plot.py).")
