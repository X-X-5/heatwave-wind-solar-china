# -*- coding: utf-8 -*-
"""
Synthetic cross-day-boundary test for the ERA5 hourly -> daily aggregation rule.

Confirms, for BOTH candidate CN05.1 day windows (A: calendar 00-24 Beijing;
B: 20-20 Beijing, the station convention), that:

  1. instantaneous variables (t2m / wind / wpd) are assigned by POINT timestamp:
     exactly 24 samples in [T_start, T_end);
  2. end-of-hour accumulated radiation (ssrd) and its derived ghi / pvpot are
     assigned by INTERVAL-END timestamp: exactly 24 samples in (T_start, T_end];
  3. the two classes are shifted by exactly 1 h at a boundary (why a uniform
     .resample over the whole NetCDF is wrong -- a single UTC date does NOT
     contain a full Beijing 20-20 day);
  4. consecutive days tile the hour axis with no duplication and no omission;
  5. daily sums of ssrd / ghi / pvpot equal the hourly accumulation over the same
     span (radiation & PV conservation).

Read-only: writes no daily product. A short real-data cross-check on the existing
Faiman trial NetCDF is appended (does not modify it). No download, no git.
"""
import os
import glob
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era5_daily_window import (H, window_calendar_day, window_2020_day,
                               point_mask, accum_mask)

FAILS = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


# --- candidate day windows & selection masks: imported from era5_daily_window ---


def build_synthetic():
    time = np.arange("2022-07-30T00", "2022-09-02T00", dtype="datetime64[h]")
    n = time.size
    rng = np.arange(n, dtype=float)
    return {
        "time": time,
        "t2m": rng,                       # unique per hour -> easy dup/omit detection
        "wpd": rng + 100.0,               # instantaneous (point) class
        "ssrd": rng + 0.5,                # end-of-hour accumulation
        "ghi": (rng + 0.5) / 3600.0,
        "pvpot": (rng + 0.5) / 3600.0 * 0.9,   # stands in for PR*ghi (label rule matters)
    }


def run_window_checks(name, data, window_fn, dates):
    time = data["time"]
    ssrd = data["ssrd"]; ghi = data["ghi"]; pvpot = data["pvpot"]
    n = time.size

    print(f"\n=== candidate {name}: {len(dates)} complete days "
          f"{dates[0].astype('datetime64[D]')} .. {dates[-1].astype('datetime64[D]')} ===")

    s0, _ = window_fn(dates[0])
    _, e_last = window_fn(dates[-1])

    point_cov = np.zeros(n, int)
    accum_cov = np.zeros(n, int)
    daily_ssrd = []; daily_ghi = []; daily_pv = []

    for D in dates:
        s, e = window_fn(D)
        pm = point_mask(time, s, e)
        am = accum_mask(time, s, e)
        check(f"{D.astype('datetime64[D]')} point n=24 / accum n=24",
              int(pm.sum()) == 24 and int(am.sum()) == 24)
        point_cov += pm.astype(int)
        accum_cov += am.astype(int)
        daily_ssrd.append(ssrd[am].sum())
        daily_ghi.append(ghi[am].sum())
        daily_pv.append(pvpot[am].sum())

    # 3) point/accum are shifted exactly 1 h at the boundary (sample day)
    s, e = window_fn(dates[len(dates) // 2])
    pm = point_mask(time, s, e); am = accum_mask(time, s, e)
    check("point min == T_start and max == T_end-1h",
          time[pm].min() == s and time[pm].max() == e - H)
    check("accum min == T_start+1h and max == T_end",
          time[am].min() == s + H and time[am].max() == e)

    # 4) no duplication / no omission: consecutive days tile the hour axis exactly
    in_point = (time >= s0) & (time < e_last)
    in_accum = (time >= s0 + H) & (time <= e_last)
    check("point: every hour in union covered exactly once",
          bool((point_cov[in_point] == 1).all() and (point_cov[~in_point] == 0).all()))
    check("accum: every hour in union covered exactly once",
          bool((accum_cov[in_accum] == 1).all() and (accum_cov[~in_accum] == 0).all()))

    # 5) daily sums == hourly accumulation over the same span (conservation)
    for label, daily, hour_arr in [
        ("ssrd", daily_ssrd, ssrd[in_accum].sum()),
        ("ghi",  daily_ghi,  ghi[in_accum].sum()),
        ("pvpot", daily_pv,  pvpot[in_accum].sum()),
    ]:
        check(f"{label}: sum(daily totals) == hourly sum over union",
              np.isclose(sum(daily), hour_arr))

    # a single UTC calendar date does NOT contain a full Beijing day (uniform-resample gotcha)
    s, e = window_fn(np.datetime64("2022-08-15"))
    nd_point = np.unique(time[point_mask(time, s, e)].astype("datetime64[D]")).size
    nd_accum = np.unique(time[accum_mask(time, s, e)].astype("datetime64[D]")).size
    check("day 2022-08-15 spans 2 UTC dates (point & accum) -> groupby(date) is wrong",
          nd_point == 2 and nd_accum == 2)


def real_data_cross_check():
    print("\n=== real-data cross-check (candidate B, read-only) ===")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results_hourly")
    ncs = sorted(glob.glob(os.path.join(out, "era5_hourly_potential_2022*.nc")))
    if not ncs:
        print("  [SKIP] no Faiman trial NetCDF found")
        return
    import xarray as xr
    ds = xr.open_dataset(ncs[-1])
    time = ds["valid_time"].values.astype("datetime64[h]")
    pvpot = ds["pvpot"].values

    dates = np.arange("2022-08-01", "2022-09-01", dtype="datetime64[D]")
    s0, _ = window_2020_day(dates[0])          # 07-31 12:00 UTC
    _, e_last = window_2020_day(dates[-1])     # 08-31 12:00 UTC

    check("downloaded span covers [s0-H, e_last] for candidate B",
          time.min() <= s0 - H and time.max() >= e_last)

    pc = np.zeros(time.size, int); ac = np.zeros(time.size, int)
    daily_pv = []
    for D in dates:
        s, e = window_2020_day(D)
        pc += point_mask(time, s, e).astype(int)
        ac += accum_mask(time, s, e).astype(int)
        daily_pv.append(pvpot[accum_mask(time, s, e)].sum())

    in_point = (time >= s0) & (time < e_last)
    in_accum = (time >= s0 + H) & (time <= e_last)
    check("real point tiling (no dup/omit) over August",
          bool((pc[in_point] == 1).all() and (pc[~in_point] == 0).all()))
    check("real accum tiling (no dup/omit) over August",
          bool((ac[in_accum] == 1).all() and (ac[~in_accum] == 0).all()))
    check("real pvpot: sum(daily) == hourly sum over union",
          np.isclose(sum(daily_pv), pvpot[in_accum].sum()))
    print(f"  complete days={len(dates)}, union hours(accum)={int(in_accum.sum())}, "
          f"daily pvpot sum={sum(daily_pv):.3f} W h m-2")


def main():
    data = build_synthetic()
    dates = np.arange("2022-07-31", "2022-09-01", dtype="datetime64[D]")

    # absolute window-position guard (catches a constant-offset bug in the
    # Beijing->UTC conversion that the tiling checks alone would miss)
    sA, eA = window_calendar_day(np.datetime64("2022-08-15"))
    sB, eB = window_2020_day(np.datetime64("2022-08-15"))
    check("A(2022-08-15) == [08-14 16:00, 08-15 16:00) UTC",
          sA == np.datetime64("2022-08-14T16") and eA == np.datetime64("2022-08-15T16"))
    check("B(2022-08-15) == [08-14 12:00, 08-15 12:00) UTC",
          sB == np.datetime64("2022-08-14T12") and eB == np.datetime64("2022-08-15T12"))

    run_window_checks("A calendar 00-24 BJT", data, window_calendar_day, dates)
    run_window_checks("B 20-20 BJT (station)", data, window_2020_day, dates)
    real_data_cross_check()

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
