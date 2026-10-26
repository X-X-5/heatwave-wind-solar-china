# -*- coding: utf-8 -*-
"""
Phase 1 full run: heatwave detection over the whole CN05.1 Tmax record.

Loads the source NetCDF once in time-chunks (IO-friendly, never all-at-once),
then computes in memory:
  * per-cell calendar-day 90th-percentile thresholds (baseline 1981-2010),
  * annual heatwave metrics for 1961-2022 (count / total days / max duration /
    mean exceedance intensity),
and writes (phase-1 fix output directory, independent of the original results/):
  * results_fixed/hw_threshold_1981-2010.nc   (366 calendar days x lat x lon)
  * results_fixed/hw_annual_1961-2022.nc      (year x lat x lon: 4 metrics
      + decidable_days / max_missing_run / data_status)
  * results_fixed/valid_mask.nc               (1 = cell has any valid Tmax data)
  * results_fixed/hw_2022_events.csv          (top 2022 grid cells + their events)

Validity gate (phase-1 fix): a cell-year whose number of *decidable* season days
(both Tmax and threshold finite) is 0 is written as missing --
hw_count/hw_days/hw_max_dur = -9999, intensity = NaN -- never as 0. Re-running
this script reproduces the gated output (the gate lives in hw_lib.detect_cell_full).

Read-only w.r.t. the source file. See docs/heatwave_method.md for the
definition and all conventions.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_lib as hw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")
OUT = os.path.join(ROOT, "results_fixed")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- open + calendar
nc = Dataset(NC, "r")
var = nc.variables["tmax"]
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]
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
season_idx = np.where(season)[0]                    # indices of all May-Sep days
slot_season = slot_all[season_idx]
days_per_year = int(len(season_idx) // n_years)     # 153
baseline_idx = np.where((year >= hw.BASELINE[0]) & (year <= hw.BASELINE[1]))[0]

nt, nlat, nlon = var.shape
print(f"grid: {nlat} lat x {nlon} lon, {nt} days, {n_years} years, "
      f"{days_per_year} season days/year")

# ---------------------------------------------------------------- load data
data = np.empty((nt, nlat, nlon), dtype=np.float32)
CHUNK = 730  # ~2 years per read
for t0 in range(0, nt, CHUNK):
    t1 = min(t0 + CHUNK, nt)
    blk = var[t0:t1]
    data[t0:t1] = np.ma.filled(blk.astype(np.float32), np.float32(np.nan))
    if t1 % 3650 < CHUNK or t1 == nt:
        print(f"  loaded {t1}/{nt} days", flush=True)
nc.close()
print(f"data in memory: {data.nbytes/1e9:.2f} GB")

# ---------------------------------------------------------------- thresholds
FILL_I = -9999
thr_all = np.full((366, nlat, nlon), np.nan, dtype=np.float32)
print("computing thresholds (baseline 1981-2010)...")
Xb = data[baseline_idx]                              # fancy index (int64 array)
Xb2d = Xb.reshape(Xb.shape[0], nlat * nlon)          # -> 2.02 GB COPY, not a view
thr2d = hw.compute_thresholds(Xb2d, slot_all[baseline_idx])
thr_all = thr2d.reshape(366, nlat, nlon).astype(np.float32)
del Xb, Xb2d, thr2d
print("thresholds done")

# ---------------------------------------------------------------- events
count = np.full((n_years, nlat, nlon), FILL_I, dtype=np.int32)
days = np.full((n_years, nlat, nlon), FILL_I, dtype=np.int32)
maxd = np.full((n_years, nlat, nlon), FILL_I, dtype=np.int32)
intens = np.full((n_years, nlat, nlon), np.nan, dtype=np.float32)
decidable = np.full((n_years, nlat, nlon), FILL_I, dtype=np.int16)
maxmiss = np.full((n_years, nlat, nlon), FILL_I, dtype=np.int16)
status = np.zeros((n_years, nlat, nlon), dtype=np.int8)
valid_mask = np.zeros((nlat, nlon), dtype=np.int8)

n_active = 0
print("detecting heatwaves...")
for i in range(nlat):
    row = data[:, i, :]                              # (nt, nlon) view
    has_data = ~np.isnan(row).all(axis=0)            # (nlon,)
    valid_mask[i, :] = has_data.astype(np.int8)
    thr_row = thr_all[:, i, :]                       # (366, nlon)
    for j in np.where(has_data)[0]:
        ne, td, md, it, dec, mrun = hw.detect_cell_full(
            row[:, j], thr_row[:, j], season_idx, slot_season,
            n_years, days_per_year)
        count[:, i, j] = ne
        days[:, i, j] = td
        maxd[:, i, j] = md
        intens[:, i, j] = it.astype(np.float32)
        decidable[:, i, j] = dec
        maxmiss[:, i, j] = mrun
        # 0=ocean, 1=complete, 2=incomplete (>=1 missing day), 3=undecidable
        status[:, i, j] = np.where(dec == 0, 3,
                                   np.where(dec == days_per_year, 1, 2)).astype(np.int8)
        n_active += 1
    if (i + 1) % 20 == 0:
        print(f"  lat row {i+1}/{nlat} done", flush=True)
print(f"processed {n_active} active grid cells")

# ---------------------------------------------------------------- write NetCDF
import xarray as xr

ds = xr.Dataset(
    data_vars={
        "hw_count": (("year", "lat", "lon"), count),
        "hw_days": (("year", "lat", "lon"), days),
        "hw_max_dur": (("year", "lat", "lon"), maxd),
        "hw_intensity": (("year", "lat", "lon"), intens),
        "decidable_days": (("year", "lat", "lon"), decidable),
        "max_missing_run": (("year", "lat", "lon"), maxmiss),
        "data_status": (("year", "lat", "lon"), status),
    },
    coords={"year": years.astype(np.int32), "lat": lat, "lon": lon},
)
ds["hw_count"].attrs = {
    "long_name": "annual count of heatwave events", "units": "1",
    "comment": "event = >=3 consecutive season days with Tmax > calendar-day 90th pct",
}
ds["hw_days"].attrs = {
    "long_name": "annual total heatwave days", "units": "day",
    "comment": "sum of lengths of all heatwave events in the year",
}
ds["hw_max_dur"].attrs = {
    "long_name": "annual maximum heatwave duration", "units": "day",
    "comment": "longest event in the year (0 if none)",
}
ds["hw_intensity"].attrs = {
    "long_name": "annual mean exceedance intensity", "units": "degC",
    "comment": "mean of (Tmax - threshold) over heatwave days; NaN if no heatwave day",
}
ds["decidable_days"].attrs = {
    "long_name": "annual decidable season days", "units": "day",
    "comment": "count of May-Sep days where both Tmax and threshold are finite (0..153)",
}
ds["max_missing_run"].attrs = {
    "long_name": "annual max consecutive missing-Tmax run", "units": "day",
    "comment": "longest run of consecutive missing Tmax days within May-Sep",
}
ds["data_status"].attrs = {
    "long_name": "annual data status", "units": "1",
    "comment": "0=ocean/no data, 1=complete, 2=incomplete (>=1 missing day), 3=undecidable",
}
ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
ds.attrs = {
    "title": "CN05.1 Tmax heatwave metrics (research definition, not ETCCDI WSDI)",
    "source": "CN05.1_Tmax_1961-2022_daily_025x025.nc (Wu & Gao 2013)",
    "baseline_period": f"{hw.BASELINE[0]}-{hw.BASELINE[1]}",
    "percentile": hw.Q, "window": "5-day centred (day-2..day+2), cross-year wrap",
    "min_run_days": hw.MIN_RUN, "season": f"{hw.SEASON[0]}-{hw.SEASON[1]} (month)",
    "exceedance": "strict > threshold",
    "percentile_method": "numpy linear (Hyndman-Fan type 7)",
    "min_samples": hw.MIN_SAMPLES,
    "missing_handling": "excluded from threshold/intensity; breaks runs; no gap-filling",
    "validity_gate": "cell-year with 0 decidable days -> metrics = -9999/NaN (never 0)",
}
for v in ["hw_count", "hw_days", "hw_max_dur"]:
    ds[v].encoding = {"_FillValue": np.int32(FILL_I)}
for v in ["decidable_days", "max_missing_run"]:
    ds[v].encoding = {"_FillValue": np.int16(FILL_I)}
out_nc = os.path.join(OUT, "hw_annual_1961-2022.nc")
ds.to_netcdf(out_nc)
print(f"wrote {out_nc}")

ds_thr = xr.Dataset(
    data_vars={"threshold": (("doy", "lat", "lon"), thr_all)},
    coords={"doy": np.arange(1, 367, dtype=np.int32), "lat": lat, "lon": lon},
)
ds_thr["threshold"].attrs = {
    "long_name": "Tmax 90th percentile threshold", "units": "degC",
    "comment": "doy=1..366 leap-year frame; Feb 29 = doy 60",
}
ds_thr["threshold"].encoding = {"_FillValue": np.float32(np.nan)}
ds_thr.attrs = ds.attrs
out_thr = os.path.join(OUT, "hw_threshold_1981-2010.nc")
ds_thr.to_netcdf(out_thr)
print(f"wrote {out_thr}")

xr.Dataset(
    data_vars={"valid_mask": (("lat", "lon"), valid_mask)},
    coords={"lat": lat, "lon": lon},
    attrs={"comment": "1 = cell has any valid Tmax data over 1961-2022"},
).to_netcdf(os.path.join(OUT, "valid_mask.nc"))
print(f"wrote {os.path.join(OUT, 'valid_mask.nc')}")

# ---------------------------------------------------------------- 2022 check
Y2022 = 2022
iy = int(np.where(years == Y2022)[0][0])
seas_2022 = np.where(season & (year == Y2022))[0]     # 153 global indices
d2022 = dates[seas_2022]
slot_2022 = slot_all[seas_2022]                       # 153 slot indices for 2022


def cell_events(i, j):
    tmax_s = data[seas_2022, i, j]
    thr_s = thr_all[:, i, j][slot_2022]
    finite = np.isfinite(tmax_s) & np.isfinite(thr_s)
    hot = finite & (tmax_s > thr_s)
    excess = tmax_s - thr_s
    evs, run = [], []
    for k in range(len(hot)):
        if hot[k]:
            run.append(k)
        else:
            if len(run) >= hw.MIN_RUN:
                ex = excess[run]
                evs.append((d2022[run[0]], d2022[run[-1]], len(run),
                            float(np.nanmax(ex)), float(np.nanmean(ex))))
            run = []
    if len(run) >= hw.MIN_RUN:
        ex = excess[run]
        evs.append((d2022[run[0]], d2022[run[-1]], len(run),
                    float(np.nanmax(ex)), float(np.nanmean(ex))))
    return evs


d22 = np.where(valid_mask == 1, days[iy].astype(float), -1)
flat_idx = np.argsort(d22.ravel())[::-1]
ri, ci = np.unravel_index(flat_idx, d22.shape)
rows = []
top_cells = []
for (ii, jj) in zip(ri, ci):
    if valid_mask[ii, jj] == 0 or d22[ii, jj] <= 0:
        continue
    top_cells.append((ii, jj))
    if len(top_cells) >= 30:
        break

print(f"\nTop grid cells by 2022 heatwave days (top {len(top_cells)}):")
for (ii, jj) in top_cells:
    evs = cell_events(ii, jj)
    print(f"  lon={lon[jj]:6.2f} lat={lat[ii]:6.2f} | days={int(days[iy,ii,jj]):3d} "
          f"count={int(count[iy,ii,jj]):2d} maxdur={int(maxd[iy,ii,jj]):2d}")
    for e in evs:
        rows.append((float(lon[jj]), float(lat[ii]), e[0].strftime("%Y-%m-%d"),
                     e[1].strftime("%Y-%m-%d"), e[2], round(e[3], 2), round(e[4], 2)))
        print(f"      event {e[0].date()} -> {e[1].date()}  dur={e[2]}d  "
              f"peak_exc={e[3]:.2f}C  mean_exc={e[4]:.2f}C")

df = pd.DataFrame(rows, columns=["lon", "lat", "start", "end", "duration_days",
                                 "peak_excess_C", "mean_excess_C"])
csv_out = os.path.join(OUT, "hw_2022_events.csv")
df.to_csv(csv_out, index=False)
print(f"wrote {csv_out}")
print("\nDONE full heatwave run.")
