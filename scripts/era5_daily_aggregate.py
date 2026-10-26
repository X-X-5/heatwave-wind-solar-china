# -*- coding: utf-8 -*-
"""
Daily aggregation of the hourly wind/solar potential under TWO candidate day
boundaries (parameterized by --start/--end; August 2022 was the pilot).

Reads the existing hourly NetCDFs (Faiman trial + linear sensitivity variant) --
does NOT recompute any hourly quantity from daily means. Applies the window rules
validated by scripts/era5_daily_aggregation_test.py:

  - instantaneous (tair / wind100 / wpd): POINT samples in [T_start, T_end)
  - end-of-hour accumulations (ghi / pvpot / pvpot_lin): [t-1h,t] fully in window,
    i.e. timestamps in (T_start, T_end]

Day label D (CN05.1 date), Beijing = UTC+8:
  A: calendar 00-24 BJT   -> [D-1 16:00, D 16:00) UTC
  B: station 20-20 BJT    -> [D-1 12:00, D 12:00) UTC  (provisional primary;
     CN05.1 day boundary UNCONFIRMED)

Per gridpoint per day: 24 valid hours required, else that metric -> NaN; the
valid-hour count is saved. Cumulative metrics are sum(24 hourly W m-2) * 1 h / 1000
= kWh m-2 (NOT actual generation). Writes two independent daily NetCDFs; never
overwrites the hourly NetCDFs or existing heatwave results.
"""
import os
import sys
import json
import argparse

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era5_daily_window import WINDOWS, point_mask, accum_mask
from _nc_guard import write_guarded

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY = os.path.join(ROOT, "results_hourly")
OUT = os.path.join(ROOT, "results_daily")

HOUR_LEN_H = 1.0     # each hourly sample represents 1 h
KWH = 1000.0         # W h/m2 -> kWh/m2


def load_hourly_sorted(path):
    ds = xr.open_dataset(path).sortby("latitude")   # ascending 30..33 to match CN05.1
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faim", required=True, help="path to Faiman hourly NetCDF")
    ap.add_argument("--lin", required=True, help="path to linear hourly NetCDF")
    ap.add_argument("--start", default="2022-08-01", help="first date label (inclusive)")
    ap.add_argument("--end", default="2022-09-01", help="last date label (exclusive)")
    ap.add_argument("--tag", default="202208", help="output filename tag")
    args = ap.parse_args()

    for p in (args.faim, args.lin):
        if not os.path.exists(p):
            raise SystemExit(f"hourly input not found: {p}")

    faim = load_hourly_sorted(args.faim)
    lin = load_hourly_sorted(args.lin)

    time = faim["valid_time"].values.astype("datetime64[h]")
    lat = faim["latitude"].values
    lon = faim["longitude"].values
    nlat, nlon = lat.size, lon.size

    # hourly arrays (time, lat, lon), already nonlinear-computed
    tair = faim["tair"].values
    wind100 = faim["wind100"].values
    wpd = faim["wpd"].values
    ghi = faim["ghi"].values
    pvpot = faim["pvpot"].values                 # Faiman (current scheme)
    pvpot_lin = lin["pvpot_lin"].values          # linear sensitivity variant

    dates = np.arange(args.start, args.end, dtype="datetime64[D]")
    nday = dates.size
    print(f"aggregating {nday} days ({dates[0].astype('datetime64[D]')} .. "
          f"{dates[-1].astype('datetime64[D]')}), grid {nlat}x{nlon}")

    os.makedirs(OUT, exist_ok=True)

    for key, (label, wfn) in WINDOWS.items():
        s0, _ = wfn(dates[0])
        _, e_last = wfn(dates[-1])

        tair_mean = np.full((nday, nlat, nlon), np.nan)
        tair_max = np.full((nday, nlat, nlon), np.nan)
        wind100_mean = np.full((nday, nlat, nlon), np.nan)
        wpd_mean = np.full((nday, nlat, nlon), np.nan)
        ghi_sum = np.full((nday, nlat, nlon), np.nan)
        pvpot_mean = np.full((nday, nlat, nlon), np.nan)
        pvpot_sum = np.full((nday, nlat, nlon), np.nan)
        pvpot_lin_mean = np.full((nday, nlat, nlon), np.nan)
        pvpot_lin_sum = np.full((nday, nlat, nlon), np.nan)
        valid_hours = np.zeros((nday, nlat, nlon), dtype=np.int16)

        for i, D in enumerate(dates):
            s, e = wfn(D)
            pm = point_mask(time, s, e)
            am = accum_mask(time, s, e)
            assert int(pm.sum()) == 24 and int(am.sum()) == 24, (D, pm.sum(), am.sum())

            p_tair = tair[pm]; p_wind = wind100[pm]; p_wpd = wpd[pm]
            a_ghi = ghi[am]; a_pv = pvpot[am]; a_pvl = pvpot_lin[am]

            pv = np.isfinite(p_tair) & np.isfinite(p_wind) & np.isfinite(p_wpd)
            av = np.isfinite(a_ghi) & np.isfinite(a_pv) & np.isfinite(a_pvl)
            vh = np.minimum(pv.sum(0), av.sum(0))          # (lat, lon)
            valid_hours[i] = vh
            complete = vh == 24

            with np.errstate(invalid="ignore"):
                tair_mean[i] = np.where(complete, np.nanmean(p_tair, 0), np.nan)
                tair_max[i] = np.where(complete, np.nanmax(p_tair, 0), np.nan)
                wind100_mean[i] = np.where(complete, np.nanmean(p_wind, 0), np.nan)
                wpd_mean[i] = np.where(complete, np.nanmean(p_wpd, 0), np.nan)
                ghi_sum[i] = np.where(complete,
                                      np.nansum(a_ghi, 0) * HOUR_LEN_H / KWH, np.nan)
                pvpot_mean[i] = np.where(complete, np.nanmean(a_pv, 0), np.nan)
                pvpot_sum[i] = np.where(complete,
                                        np.nansum(a_pv, 0) * HOUR_LEN_H / KWH, np.nan)
                pvpot_lin_mean[i] = np.where(complete, np.nanmean(a_pvl, 0), np.nan)
                pvpot_lin_sum[i] = np.where(complete,
                                            np.nansum(a_pvl, 0) * HOUR_LEN_H / KWH, np.nan)

        ds = xr.Dataset(
            {
                "tair_mean": (("date", "latitude", "longitude"), tair_mean),
                "tair_max": (("date", "latitude", "longitude"), tair_max),
                "wind100_mean": (("date", "latitude", "longitude"), wind100_mean),
                "wpd_mean": (("date", "latitude", "longitude"), wpd_mean),
                "ghi_sum": (("date", "latitude", "longitude"), ghi_sum),
                "pvpot_mean": (("date", "latitude", "longitude"), pvpot_mean),
                "pvpot_sum": (("date", "latitude", "longitude"), pvpot_sum),
                "pvpot_lin_mean": (("date", "latitude", "longitude"), pvpot_lin_mean),
                "pvpot_lin_sum": (("date", "latitude", "longitude"), pvpot_lin_sum),
                "valid_hours": (("date", "latitude", "longitude"), valid_hours),
            },
            coords={
                "date": dates.astype("datetime64[D]"),
                "latitude": lat,
                "longitude": lon,
            },
        )
        for v, units, long in [
            ("tair_mean", "degC", "daily mean 2 m air temperature (hourly t2m)"),
            ("tair_max", "degC", "daily max 2 m air temperature (max of 24 hourly samples)"),
            ("wind100_mean", "m s-1", "daily mean 100 m wind speed"),
            ("wpd_mean", "W m-2", "daily mean wind power density 0.5*rho*V100^3"),
            ("ghi_sum", "kWh m-2", "daily accumulated GHI = sum(24 hourly ghi)*1h/1000"),
            ("pvpot_mean", "W m-2", "daily mean PV potential (Faiman cell temp)"),
            ("pvpot_sum", "kWh m-2", "daily accumulated PV potential (Faiman) = sum*1h/1000, NOT generation"),
            ("pvpot_lin_mean", "W m-2", "daily mean PV potential (linear cell temp)"),
            ("pvpot_lin_sum", "kWh m-2", "daily accumulated PV potential (linear) = sum*1h/1000, NOT generation"),
            ("valid_hours", "h", "number of valid hours used (24 = complete)"),
        ]:
            ds[v].attrs = {"units": units, "long_name": long}

        ds.attrs = {
            "title": f"ERA5 daily wind/solar potential, {args.start} .. {args.end} (exclusive), boundary {key}",
            "day_boundary": label,
            "day_window_utc": f"[{s0}, {e_last})",
            "point_samples_utc": f"{s0} .. {e_last - np.timedelta64(1, 'h')}",
            "accum_samples_utc": f"{s0 + np.timedelta64(1, 'h')} .. {e_last}",
            "n_days": nday,
            "valid_hours_requirement": "24 per gridpoint per day, else metric NaN",
            "cumulative_units_note": "sum(hourly W m-2) * 1 h / 1000 = kWh m-2; NOT actual generation",
            "aggregation_order": "hourly nonlinear metrics computed first, then daily aggregate",
            "cn05.1_day_boundary": "UNCONFIRMED (station convention is 20-20 BJT, not yet verified for CN05.1 itself)",
            "source_hourly": os.path.basename(args.faim) + ", " + os.path.basename(args.lin),
            "config_signature": json.dumps({
                "boundary_key": key,
                "start": args.start,
                "end": args.end,
                "tag": args.tag,
                "hour_len_h": HOUR_LEN_H,
                "kwh_divisor": KWH,
                "valid_hours_requirement": 24,
                "point_mask": "[T_start,T_end) instantaneous (tair/wind100/wpd)",
                "accum_mask": "(T_start,T_end] interval (ghi/pvpot/pvpot_lin)",
                "source_hourly": os.path.basename(args.faim) + ", " + os.path.basename(args.lin),
            }, sort_keys=True),
        }
        out_nc = os.path.join(OUT, f"era5_daily_potential_{args.tag}_{key}.nc")
        enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
               for v in ds.data_vars if v != "valid_hours"}
        enc["valid_hours"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
        write_guarded(ds, out_nc, enc)

        n_complete = int((valid_hours == 24).sum())
        print(f"[{key}] {label}")
        print(f"   UTC window [point {s0} .. {e_last - np.timedelta64(1, 'h')}; "
              f"accum {s0 + np.timedelta64(1, 'h')} .. {e_last}]")
        print(f"   valid_hours==24: {n_complete}/{valid_hours.size} "
              f"(min={int(valid_hours.min())}, max={int(valid_hours.max())})")
        print(f"   ghi_sum    min={np.nanmin(ghi_sum):.4g} max={np.nanmax(ghi_sum):.4g} kWh m-2")
        print(f"   pvpot_sum  min={np.nanmin(pvpot_sum):.4g} max={np.nanmax(pvpot_sum):.4g} kWh m-2")
        print(f"   wpd_mean   min={np.nanmin(wpd_mean):.4g} max={np.nanmax(wpd_mean):.4g} W m-2")


if __name__ == "__main__":
    main()
