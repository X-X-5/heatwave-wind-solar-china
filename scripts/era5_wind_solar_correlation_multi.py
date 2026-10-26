# -*- coding: utf-8 -*-
"""
Multi-year (2011-2022) wind-solar temporal-correlation in heatwave vs ordinary
days, pooled by CALENDAR MONTH across years.

This is a RESOURCE-LEVEL temporal-complementarity exploration. It is NOT a joint
generation stability assessment and NOT a causal attribution of heatwave effects.

Difference from the single-year 2022 script: instead of per-year-month samples,
each CALENDAR month (May..Sep) pools all days of that month across the complete
years. For each gridpoint and calendar month, two groups are compared:
  heatwave = label 3; ordinary = label 1 (label 0/2/4 excluded; gap days NOT
  folded into ordinary). Months are never mixed into one grand correlation.

Per gridpoint per calendar month, per group:
  * Pearson r and Spearman rho of (pvpot_sum, wpd_mean) over the paired valid
    days (both finite);
  * paired valid-day count n (pooled across years);
  * number of DISTINCT years contributing >= 1 day;
  * number of contiguous date segments and the longest segment (within-year;
    year boundaries break contiguity) -- sample concentration hint only;
  * sample std of PV and WPD (ddof=1).

Display thresholds (NOT sufficiency criteria): main = both groups >= 10 days;
sensitivity = both groups >= 5 days. A per-year-per-month product is also kept so
that "does pooling mask year differences" can be checked.

No significance tests, no p-values, no confidence intervals, no summing of WPD
and PV potential. Correlation change is never read as a heatwave causal effect.
"""
import os
import sys
import json
import csv
import glob
import re
import argparse
import calendar

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "results_multi")

MONTHS = [5, 6, 7, 8, 9]                     # calendar months pooled across years
MONTH_SHORT = [calendar.month_abbr[m] for m in MONTHS]
MAIN_MIN = 10
SENS_MIN = 5
ZERO_TOL = 1e-12

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def pct(x, p):
    return float(np.nanpercentile(x, p)) if x.size else float("nan")


def sign3(x, tol=ZERO_TOL):
    return np.where(x > tol, 1, np.where(x < -tol, -1, 0))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return np.nan
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0.0:
        return np.nan
    return float(np.clip((xm * ym).sum() / denom, -1.0, 1.0))


def rankdata_avg(a):
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[j]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    return pearson(rankdata_avg(x), rankdata_avg(y))


def contiguous_days(sel_bool, dates):
    """Contiguous segments of selected days, using REAL date differences.

    Two selected days belong to the same segment only when their dates differ by
    exactly 1 day. Cross-year adjacency (e.g. 2011-09-30 followed in the pooled
    array by 2012-05-01, or May 31 of one year followed by May 1 of the next)
    never forms a segment, because the true date gap is far larger than 1 day.

    sel_bool : 1-D bool over pooled (date-sorted) days of one calendar month.
    dates    : 1-D datetime64[D] per pooled day.
    Returns (n_segments, longest_segment_days).
    """
    idx = np.where(sel_bool)[0]
    if idx.size == 0:
        return 0, 0
    if idx.size == 1:
        return 1, 1
    d = dates[idx]
    contiguous = (d[1:] - d[:-1]) == np.timedelta64(1, "D")
    brk = ~contiguous
    n_seg = int(brk.sum()) + 1
    brk_pos = np.where(brk)[0]
    lengths = np.diff(np.concatenate(([-1], brk_pos, [idx.size - 1])))
    return int(n_seg), int(lengths.max())


def discover_complete_years(data_dir):
    """Years with all three daily products present (A, B, aligned labels)."""
    years = set()
    for p in glob.glob(os.path.join(data_dir, "era5_daily_potential_*_A.nc")):
        m = re.match(r"era5_daily_potential_(\d{4})0501_\d{4}0930_A\.nc",
                     os.path.basename(p))
        if m:
            years.add(int(m.group(1)))
    complete = []
    for y in sorted(years):
        a = os.path.join(data_dir, f"era5_daily_potential_{y}0501_{y}0930_A.nc")
        b = os.path.join(data_dir, f"era5_daily_potential_{y}0501_{y}0930_B.nc")
        l = os.path.join(data_dir, f"cn05_labels_{y}0501_{y}0930_aligned.nc")
        if all(os.path.exists(p) for p in (a, b, l)):
            complete.append(y)
    return complete


def load_all(data_dir, years):
    das_a, das_b, labs = [], [], []
    for y in years:
        da = xr.open_dataset(os.path.join(
            data_dir, f"era5_daily_potential_{y}0501_{y}0930_A.nc"))[["pvpot_sum", "wpd_mean"]]
        db = xr.open_dataset(os.path.join(
            data_dir, f"era5_daily_potential_{y}0501_{y}0930_B.nc"))[["pvpot_sum", "wpd_mean"]]
        la = xr.open_dataset(os.path.join(
            data_dir, f"cn05_labels_{y}0501_{y}0930_aligned.nc"))[["label"]]
        das_a.append(da); das_b.append(db); labs.append(la)
    A = xr.concat(das_a, dim="date")
    B = xr.concat(das_b, dim="date")
    L = xr.concat(labs, dim="date")
    return A, B, L


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DIR,
                    help="dir holding the per-year daily A/B + aligned labels")
    ap.add_argument("--out", default=None, help="output dir (default: --data-dir)")
    ap.add_argument("--years", default=None,
                    help="comma list to restrict years (default: auto-discover complete)")
    args = ap.parse_args()
    out = args.out or args.data_dir
    os.makedirs(out, exist_ok=True)

    if args.years:
        years = [int(x) for x in args.years.split(",") if x.strip()]
    else:
        years = discover_complete_years(args.data_dir)
    if not years:
        raise SystemExit("no complete years found; run the per-year pipeline first")
    all_years_expected = list(range(2011, 2023))
    missing_years = [y for y in all_years_expected if y not in years]
    print(f"complete years included: {years}")
    if missing_years:
        print(f"MISSING (excluded, reported honestly): {missing_years}")

    A, B, L = load_all(args.data_dir, years)
    dates = A["date"].values.astype("datetime64[D]")
    lat = A["latitude"].values
    lon = A["longitude"].values
    nlat, nlon = lat.size, lon.size

    # coordinate sanity across boundaries/labels (exact, no interpolation)
    assert np.array_equal(A["date"].values, B["date"].values), "A/B date mismatch"
    assert np.array_equal(A["date"].values, L["date"].values), "daily/label date mismatch"
    assert np.array_equal(A["latitude"].values, L["latitude"].values)
    assert np.array_equal(A["longitude"].values, L["longitude"].values)

    label = L["label"].values.astype(np.int8)
    pv = {"A": A["pvpot_sum"].values.astype(np.float64),
          "B": B["pvpot_sum"].values.astype(np.float64)}
    wpd = {"A": A["wpd_mean"].values.astype(np.float64),
           "B": B["wpd_mean"].values.astype(np.float64)}

    month_arr = (dates.astype("datetime64[M]").astype("int64") % 12) + 1
    year_arr = dates.astype("datetime64[Y]").astype("int64") + 1970
    year_set = np.array(sorted(years), dtype=np.int64)

    nmon = len(MONTHS)
    shp = (nmon, nlat, nlon)
    res = {}
    for bd in ("A", "B"):
        for g in ("hw", "ne"):
            res[f"r_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"rho_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"n_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"n_year_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"n_seg_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"max_seg_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"pv_sd_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"wpd_sd_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"flag_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
        res[f"delta_r_{bd}"] = np.full(shp, np.nan)
        res[f"delta_rho_{bd}"] = np.full(shp, np.nan)

    for mi, M in enumerate(MONTHS):
        tmask = month_arr == M
        lab_m = label[tmask]
        yr_m = year_arr[tmask]
        dates_m = dates[tmask]
        hw_mask = lab_m == 3
        ne_mask = lab_m == 1
        for bd in ("A", "B"):
            pv_m = pv[bd][tmask]
            wpd_m = wpd[bd][tmask]
            for g, gmask in (("hw", hw_mask), ("ne", ne_mask)):
                sel = gmask & np.isfinite(pv_m) & np.isfinite(wpd_m)
                r_arr = np.full((nlat, nlon), np.nan)
                rho_arr = np.full((nlat, nlon), np.nan)
                n_arr = np.zeros((nlat, nlon), np.int16)
                nyr_arr = np.zeros((nlat, nlon), np.int16)
                nseg_arr = np.zeros((nlat, nlon), np.int16)
                mseg_arr = np.zeros((nlat, nlon), np.int16)
                pv_sd_arr = np.full((nlat, nlon), np.nan)
                wpd_sd_arr = np.full((nlat, nlon), np.nan)
                flag_arr = np.zeros((nlat, nlon), np.int16)
                for i in range(nlat):
                    for j in range(nlon):
                        s = sel[:, i, j]
                        x = pv_m[s, i, j]
                        y = wpd_m[s, i, j]
                        n = x.size
                        n_arr[i, j] = n
                        nyr_arr[i, j] = np.unique(yr_m[s]).size if n else 0
                        nseg_arr[i, j], mseg_arr[i, j] = contiguous_days(s, dates_m)
                        if n >= 2:
                            pv_sd_arr[i, j] = float(np.std(x, ddof=1))
                            wpd_sd_arr[i, j] = float(np.std(y, ddof=1))
                        if n < 2:
                            flag_arr[i, j] = 1
                        elif np.ptp(x) == 0.0:
                            flag_arr[i, j] = 2
                        elif np.ptp(y) == 0.0:
                            flag_arr[i, j] = 3
                        else:
                            flag_arr[i, j] = 0
                            r_arr[i, j] = pearson(x, y)
                            rho_arr[i, j] = spearman(x, y)
                res[f"r_{g}_{bd}"][mi] = r_arr
                res[f"rho_{g}_{bd}"][mi] = rho_arr
                res[f"n_{g}_{bd}"][mi] = n_arr
                res[f"n_year_{g}_{bd}"][mi] = nyr_arr
                res[f"n_seg_{g}_{bd}"][mi] = nseg_arr
                res[f"max_seg_{g}_{bd}"][mi] = mseg_arr
                res[f"pv_sd_{g}_{bd}"][mi] = pv_sd_arr
                res[f"wpd_sd_{g}_{bd}"][mi] = wpd_sd_arr
                res[f"flag_{g}_{bd}"][mi] = flag_arr
            res[f"delta_r_{bd}"][mi] = res[f"r_hw_{bd}"][mi] - res[f"r_ne_{bd}"][mi]
            res[f"delta_rho_{bd}"][mi] = res[f"rho_hw_{bd}"][mi] - res[f"rho_ne_{bd}"][mi]

    # ---------------- per-year-per-month (year-difference check), boundary B ----
    shp_yr = (len(year_set), nmon, nlat, nlon)
    yr_res = {}
    for g in ("hw", "ne"):
        yr_res[f"r_{g}"] = np.full(shp_yr, np.nan)
        yr_res[f"rho_{g}"] = np.full(shp_yr, np.nan)
        yr_res[f"n_{g}"] = np.zeros(shp_yr, dtype=np.int16)
    yr_res["delta_r"] = np.full(shp_yr, np.nan)
    yr_res["delta_rho"] = np.full(shp_yr, np.nan)

    for yi, Y in enumerate(year_set):
        for mi, M in enumerate(MONTHS):
            tmask = (year_arr == Y) & (month_arr == M)
            lab_m = label[tmask]
            hw_mask = lab_m == 3
            ne_mask = lab_m == 1
            pv_m = pv["B"][tmask]
            wpd_m = wpd["B"][tmask]
            for g, gmask in (("hw", hw_mask), ("ne", ne_mask)):
                sel = gmask & np.isfinite(pv_m) & np.isfinite(wpd_m)
                r_arr = np.full((nlat, nlon), np.nan)
                rho_arr = np.full((nlat, nlon), np.nan)
                n_arr = np.zeros((nlat, nlon), np.int16)
                for i in range(nlat):
                    for j in range(nlon):
                        s = sel[:, i, j]
                        x = pv_m[s, i, j]
                        y = wpd_m[s, i, j]
                        n = x.size
                        n_arr[i, j] = n
                        if n >= 2 and np.ptp(x) > 0.0 and np.ptp(y) > 0.0:
                            r_arr[i, j] = pearson(x, y)
                            rho_arr[i, j] = spearman(x, y)
                yr_res[f"r_{g}"][yi, mi] = r_arr
                yr_res[f"rho_{g}"][yi, mi] = rho_arr
                yr_res[f"n_{g}"][yi, mi] = n_arr
            yr_res["delta_r"][yi, mi] = yr_res["r_hw"][yi, mi] - yr_res["r_ne"][yi, mi]
            yr_res["delta_rho"][yi, mi] = yr_res["rho_hw"][yi, mi] - yr_res["rho_ne"][yi, mi]

    # ---------------- NetCDF: pooled ----------------
    month_coord = np.array(MONTHS, dtype=np.int16)
    ds = xr.Dataset(
        {name: (("month", "latitude", "longitude"), arr) for name, arr in res.items()},
        coords={"month": month_coord, "latitude": lat, "longitude": lon},
    )
    ds["month"].attrs = {"long_name": "calendar month (pooled across complete years)",
                         "units": "month_of_year"}
    for bd in ("A", "B"):
        for g, glong in (("hw", "heatwave (label 3)"), ("ne", "ordinary non-exceedance (label 1)")):
            ds[f"r_{g}_{bd}"].attrs = {"units": "1", "long_name": f"Pearson r (PV,WPD), {glong}, boundary {bd}"}
            ds[f"rho_{g}_{bd}"].attrs = {"units": "1", "long_name": f"Spearman rho (PV,WPD), {glong}, boundary {bd}"}
            ds[f"n_{g}_{bd}"].attrs = {"units": "days", "long_name": f"paired valid-day count (pooled), {glong}, boundary {bd}"}
            ds[f"n_year_{g}_{bd}"].attrs = {"units": "years", "long_name": f"distinct years contributing >=1 day, {glong}, boundary {bd}"}
            ds[f"n_seg_{g}_{bd}"].attrs = {"units": "1", "long_name": f"contiguous date segments (year boundaries break), {glong}, boundary {bd}"}
            ds[f"max_seg_{g}_{bd}"].attrs = {"units": "days", "long_name": f"longest contiguous segment, {glong}, boundary {bd}"}
            ds[f"pv_sd_{g}_{bd}"].attrs = {"units": "kWh m-2", "long_name": f"sample std of PV (ddof=1), {glong}, boundary {bd}"}
            ds[f"wpd_sd_{g}_{bd}"].attrs = {"units": "W m-2", "long_name": f"sample std of WPD (ddof=1), {glong}, boundary {bd}"}
            ds[f"flag_{g}_{bd}"].attrs = {"long_name": f"reason flag, {glong}, boundary {bd}",
                                          "flag_codes": "0=ok,1=insufficient(n<2),2=PV const,3=WPD const"}
        ds[f"delta_r_{bd}"].attrs = {"units": "1", "long_name": f"r_heatwave - r_ordinary, boundary {bd}"}
        ds[f"delta_rho_{bd}"].attrs = {"units": "1", "long_name": f"rho_heatwave - rho_ordinary, boundary {bd}"}
    ds.attrs = {
        "title": f"Multi-year ({years[0]}-{years[-1]}) wind-solar temporal correlation, "
                 "heatwave vs ordinary, pooled by calendar month",
        "years_included": years,
        "years_excluded": missing_years,
        "pooling": "each calendar month pools all days of that month across the complete years; months never mixed",
        "correlation": "Pearson r and Spearman rho of (pvpot_sum, wpd_mean) over paired valid-day sample",
        "paired_valid_day": "both pvpot_sum (Faiman) and wpd_mean finite",
        "groups": "heatwave = label 3; ordinary = label 1; excluded = label 0/2/4; gap days NOT folded into ordinary",
        "nan_reasons": "1=insufficient pairs (n<2), 2=PV constant, 3=WPD constant",
        "std_ddof": 1,
        "main_min_days": MAIN_MIN,
        "sensitivity_min_days": SENS_MIN,
        "threshold_note": "display filter, NOT a statistical sufficiency criterion",
        "boundary": "B = 20-20 BJT (provisional primary); A = 00-24 BJT (sensitivity)",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "not_an_independence_measure": "days are autocorrelated and spatially dependent",
        "not_a_reliability_or_causal_statement": "descriptive; no causal attribution",
        "config_signature": json.dumps({
            "correlation": ["pearson", "spearman"],
            "spearman_rank_method": "average",
            "groups": {"heatwave": 3, "ordinary": 1, "excluded": [0, 2, 4]},
            "metrics": {"pv": "pvpot_sum", "wpd": "wpd_mean"},
            "boundaries": ["A", "B"],
            "pooling": "calendar_month_across_years",
            "main_min_days": MAIN_MIN, "sensitivity_min_days": SENS_MIN,
            "zero_tol": ZERO_TOL, "delta": "(hw - ordinary)",
        }, sort_keys=True),
    }
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.integer):
            enc[v] = {"zlib": True, "complevel": 4, "dtype": str(ds[v].dtype)}
        else:
            enc[v] = {"zlib": True, "complevel": 4, "_FillValue": np.nan}
    ytag = f"{years[0]}_{years[-1]}"
    nc_out = os.path.join(out, f"wind_solar_corr_multi_{ytag}.nc")
    write_guarded(ds, nc_out, enc)

    # ---------------- NetCDF: per-year-per-month ----------------
    ds_yr = xr.Dataset(
        {name: (("year", "month", "latitude", "longitude"), arr)
         for name, arr in yr_res.items()},
        coords={"year": year_set, "month": month_coord,
                "latitude": lat, "longitude": lon},
    )
    ds_yr["year"].attrs = {"long_name": "year"}
    ds_yr["month"].attrs = {"long_name": "calendar month"}
    for g, glong in (("hw", "heatwave (label 3)"), ("ne", "ordinary non-exceedance (label 1)")):
        ds_yr[f"r_{g}"].attrs = {"long_name": f"Pearson r (PV,WPD), {glong}, boundary B"}
        ds_yr[f"rho_{g}"].attrs = {"long_name": f"Spearman rho (PV,WPD), {glong}, boundary B"}
        ds_yr[f"n_{g}"].attrs = {"long_name": f"paired valid-day count, {glong}, boundary B"}
    ds_yr["delta_r"].attrs = {"long_name": "r_heatwave - r_ordinary, boundary B"}
    ds_yr["delta_rho"].attrs = {"long_name": "rho_heatwave - rho_ordinary, boundary B"}
    ds_yr.attrs = {
        "title": f"Per-year-per-month wind-solar correlation (boundary B), {years[0]}-{years[-1]}",
        "purpose": "retained to check whether pooling across years masks year differences",
        "config_signature": json.dumps({"boundary": "B", "groups": {"heatwave": 3, "ordinary": 1}},
                                       sort_keys=True),
    }
    enc_yr = {}
    for v in ds_yr.data_vars:
        if np.issubdtype(ds_yr[v].dtype, np.integer):
            enc_yr[v] = {"zlib": True, "complevel": 4, "dtype": str(ds_yr[v].dtype)}
        else:
            enc_yr[v] = {"zlib": True, "complevel": 4, "_FillValue": np.nan}
    nc_yr_out = os.path.join(out, f"wind_solar_corr_multi_per_year_{ytag}.nc")
    write_guarded(ds_yr, nc_yr_out, enc_yr)

    # ---------------- verification ----------------
    print("=" * 70)
    print("verification (pooled):")
    for bd in ("A", "B"):
        for g in ("hw", "ne"):
            r = res[f"r_{g}_{bd}"]; rho = res[f"rho_{g}_{bd}"]; flag = res[f"flag_{g}_{bd}"]
            n = res[f"n_{g}_{bd}"]; ny = res[f"n_year_{g}_{bd}"]
            ok_r = bool(np.all(np.isnan(r) | ((r >= -1) & (r <= 1))))
            ok_rho = bool(np.all(np.isnan(rho) | ((rho >= -1) & (rho <= 1))))
            ok_flag = bool(np.all(np.isfinite(r) == (flag == 0)))
            ok_n = bool(np.all((n >= 0) & (n <= 31 * len(years))))
            ok_ny = bool(np.all((ny >= 0) & (ny <= len(years))))
            print(f"  {bd}/{g}: r,rho in [-1,1] {ok_r and ok_rho}, finite<->flag0 {ok_flag}, "
                  f"n in range {ok_n}, n_year<=years {ok_ny}")

    # ---------------- summaries ----------------
    def monthly_summary(delta_r, r_hw, r_ne, delta_rho, rho_hw, rho_ne,
                        n_hw, n_ne, nyr_hw, nyr_ne, nseg_hw, nseg_ne, min_n):
        rows = []
        for mi, M in enumerate(MONTHS):
            common = (np.isfinite(delta_r[mi]) & (n_hw[mi] >= min_n) & (n_ne[mi] >= min_n)).ravel()
            dr = delta_r[mi].ravel()[common]
            drho = delta_rho[mi].ravel()[common]
            rhw = r_hw[mi].ravel()[common]; rne = r_ne[mi].ravel()[common]
            rhohw = rho_hw[mi].ravel()[common]; rhone = rho_ne[mi].ravel()[common]
            nhw = n_hw[mi].ravel()[common]; nne = n_ne[mi].ravel()[common]
            nyhw = nyr_hw[mi].ravel()[common]; nyne = nyr_ne[mi].ravel()[common]
            nghw = nseg_hw[mi].ravel()[common]; ngne = nseg_ne[mi].ravel()[common]
            n_valid = int(common.sum())
            dr_pos, dr_zero, dr_neg = (pos_zero_neg(dr) if n_valid else (0, 0, 0))
            drho_pos, drho_zero, drho_neg = (pos_zero_neg(drho) if n_valid else (0, 0, 0))
            rows.append({
                "month": calendar.month_abbr[M], "n_valid": n_valid,
                "n_hw_median": pct(nhw, 50), "n_hw_years_median": pct(nyhw, 50),
                "n_hw_seg_median": pct(nghw, 50),
                "n_ne_median": pct(nne, 50), "n_ne_years_median": pct(nyne, 50),
                "n_ne_seg_median": pct(ngne, 50),
                "r_hw_median": pct(rhw, 50), "r_ne_median": pct(rne, 50),
                "r_median_diff": (pct(rhw, 50) - pct(rne, 50)),
                "delta_r_median": pct(dr, 50), "delta_r_p25": pct(dr, 25), "delta_r_p75": pct(dr, 75),
                "delta_r_pos": dr_pos, "delta_r_zero": dr_zero, "delta_r_neg": dr_neg,
                "rho_hw_median": pct(rhohw, 50), "rho_ne_median": pct(rhone, 50),
                "delta_rho_median": pct(drho, 50), "delta_rho_p25": pct(drho, 25), "delta_rho_p75": pct(drho, 75),
                "delta_rho_pos": drho_pos, "delta_rho_zero": drho_zero, "delta_rho_neg": drho_neg,
            })
        return rows

    Bk = "B"
    main_rows = monthly_summary(res[f"delta_r_{Bk}"], res[f"r_hw_{Bk}"], res[f"r_ne_{Bk}"],
                                res[f"delta_rho_{Bk}"], res[f"rho_hw_{Bk}"], res[f"rho_ne_{Bk}"],
                                res[f"n_hw_{Bk}"], res[f"n_ne_{Bk}"],
                                res[f"n_year_hw_{Bk}"], res[f"n_year_ne_{Bk}"],
                                res[f"n_seg_hw_{Bk}"], res[f"n_seg_ne_{Bk}"], MAIN_MIN)
    filt_rows = monthly_summary(res[f"delta_r_{Bk}"], res[f"r_hw_{Bk}"], res[f"r_ne_{Bk}"],
                                res[f"delta_rho_{Bk}"], res[f"rho_hw_{Bk}"], res[f"rho_ne_{Bk}"],
                                res[f"n_hw_{Bk}"], res[f"n_ne_{Bk}"],
                                res[f"n_year_hw_{Bk}"], res[f"n_year_ne_{Bk}"],
                                res[f"n_seg_hw_{Bk}"], res[f"n_seg_ne_{Bk}"], SENS_MIN)

    write_csv(os.path.join(out, f"monthly_summary_multi_B_{ytag}.csv"),
              main_rows, list(main_rows[0].keys()))
    write_csv(os.path.join(out, f"sample_filter_summary_multi_B_{ytag}.csv"),
              filt_rows, list(filt_rows[0].keys()))

    # per-year spread of delta_r across years (year-difference check), boundary B
    yr_rows = []
    for mi, M in enumerate(MONTHS):
        for yi, Y in enumerate(year_set):
            d = yr_res["delta_r"][yi, mi].ravel()
            fin = np.isfinite(d) & (yr_res["n_hw"][yi, mi].ravel() >= 1) & (yr_res["n_ne"][yi, mi].ravel() >= 1)
            dd = d[fin]
            yr_rows.append({
                "year": int(Y), "month": calendar.month_abbr[M],
                "n_valid": int(fin.sum()),
                "delta_r_median": pct(dd, 50), "delta_r_p25": pct(dd, 25), "delta_r_p75": pct(dd, 75),
                "n_hw_median": pct(yr_res["n_hw"][yi, mi].ravel()[fin], 50),
                "n_ne_median": pct(yr_res["n_ne"][yi, mi].ravel()[fin], 50),
            })
    write_csv(os.path.join(out, f"per_year_monthly_summary_multi_B_{ytag}.csv"),
              yr_rows, list(yr_rows[0].keys()))

    # ---------------- figure ----------------
    fig_png = os.path.join(out, f"fig_wind_solar_corr_multi_monthly_B_{ytag}.png")
    make_fig(res[f"delta_r_B"], res[f"r_hw_B"], res[f"r_ne_B"],
             res[f"n_hw_B"], res[f"n_ne_B"], years, fig_png)

    # ---------------- report ----------------
    write_report(years, missing_years, main_rows, filt_rows, yr_rows, ytag,
                 os.path.join(out, f"REPORT_wind_solar_corr_multi_{ytag}.md"))

    # ---------------- headline print ----------------
    print("=" * 70)
    print(f"delta_r = r_heatwave - r_ordinary (boundary B, pooled calendar month, "
          f"both groups >= {MAIN_MIN} days, years {years[0]}-{years[-1]}):")
    for r in main_rows:
        print(f"  {r['month']}: n_valid={r['n_valid']}  n_hw_med={r['n_hw_median']} "
              f"(years {r['n_hw_years_median']}, seg {r['n_hw_seg_median']})  "
              f"n_ne_med={r['n_ne_median']} (years {r['n_ne_years_median']}, seg {r['n_ne_seg_median']})  "
              f"delta_r_med={r['delta_r_median']} (P25 {r['delta_r_p25']}, P75 {r['delta_r_p75']})  "
              f"pos/zero/neg={r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']}")
    print("DONE multi-year wind-solar correlation")


def pos_zero_neg(x, tol=ZERO_TOL):
    return (int((x > tol).sum()), int((np.abs(x) <= tol).sum()), int((x < -tol).sum()))


def write_csv(path, rows, fieldnames):
    if os.path.exists(path):
        print(f"SKIP (exists, not overwriting): {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def write_text_guarded(path, text):
    if os.path.exists(path):
        print(f"SKIP (exists, not overwriting): {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}")


def make_fig(delta_r_B, r_hw_B, r_ne_B, n_hw_B, n_ne_B, years, outfig):
    nmon = len(MONTHS)
    common = (np.isfinite(delta_r_B) & (n_hw_B >= MAIN_MIN) & (n_ne_B >= MAIN_MIN))
    hw_boxes, ne_boxes, delta_boxes, ns = [], [], [], []
    for mi in range(nmon):
        m = common[mi].ravel()
        hw_boxes.append(r_hw_B[mi].ravel()[m])
        ne_boxes.append(r_ne_B[mi].ravel()[m])
        delta_boxes.append(delta_r_B[mi].ravel()[m])
        ns.append(int(m.sum()))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.8))
    ax = axes[0]
    pos_hw = [mi + 1 - 0.22 for mi in range(nmon)]
    pos_ne = [mi + 1 + 0.22 for mi in range(nmon)]
    bp_hw = ax.boxplot(hw_boxes, positions=pos_hw, widths=0.28, showfliers=False,
                       patch_artist=True,
                       medianprops=dict(color="#a03a2e", lw=1.4),
                       whiskerprops=dict(color="#b07a72", lw=1.0),
                       capprops=dict(color="#b07a72", lw=1.0),
                       boxprops=dict(color="#a03a2e", lw=1.0))
    bp_ne = ax.boxplot(ne_boxes, positions=pos_ne, widths=0.28, showfliers=False,
                       patch_artist=True,
                       medianprops=dict(color="#1f6f9f", lw=1.4),
                       whiskerprops=dict(color="#5a7d95", lw=1.0),
                       capprops=dict(color="#5a7d95", lw=1.0),
                       boxprops=dict(color="#1f6f9f", lw=1.0))
    for bp in bp_hw["boxes"]:
        bp.set_facecolor("#f2c8c0")
    for bp in bp_ne["boxes"]:
        bp.set_facecolor("#bcd7e8")
    ax.axhline(0, color="#8a8a8a", ls="--", lw=1.0, zorder=2)
    ax.set_xticks(range(1, nmon + 1))
    ax.set_xticklabels(MONTH_SHORT, fontsize=8.5)
    ax.set_ylabel("Pearson r (wind–solar)", fontsize=9)
    ax.set_title(f"Pearson r — heatwave (red) vs ordinary (blue), "
                 f"calendar month pooled {years[0]}–{years[-1]}, both groups ≥{MAIN_MIN} days",
                 loc="left", fontsize=9.5)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.45)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#f2c8c0", edgecolor="#a03a2e", label="heatwave"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#bcd7e8", edgecolor="#1f6f9f", label="ordinary")],
              loc="upper right", fontsize=8, frameon=False)
    for mi, n in enumerate(ns):
        if n == 0:
            ax.text(mi + 1, 0, "no valid gridpoints", ha="center", va="center",
                    fontsize=8, color="#c62828")

    ax2 = axes[1]
    ax2.boxplot(delta_boxes, positions=range(1, nmon + 1), widths=0.6, showfliers=False,
                patch_artist=True,
                medianprops=dict(color="#1f6f9f", lw=1.4),
                whiskerprops=dict(color="#5a7d95", lw=1.0),
                capprops=dict(color="#5a7d95", lw=1.0),
                boxprops=dict(color="#5a7d95", lw=1.0))
    for bp in ax2.patches:
        bp.set_facecolor("#cfe0eb")
    ax2.axhline(0, color="#8a8a8a", ls="--", lw=1.0, zorder=2)
    ax2.set_xticks(range(1, nmon + 1))
    ax2.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(MONTH_SHORT, ns)], fontsize=8.5)
    ax2.set_xlabel("calendar month (pooled across years)", fontsize=9)
    ax2.set_ylabel("Δr = r_hw − r_ne", fontsize=9)
    ax2.set_title("Δr distribution (same common gridpoints as above)", loc="left", fontsize=9.5)
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.45)
    ax2.tick_params(axis="y", labelsize=8.5)
    for mi, n in enumerate(ns):
        if n == 0:
            ax2.text(mi + 1, 0, "no valid gridpoints", ha="center", va="center",
                     fontsize=8, color="#c62828")

    fig.suptitle(
        f"Wind–solar temporal correlation: heatwave vs ordinary days (boundary B)\n"
        f"calendar month pooled over {years[0]}–{years[-1]} ({len(years)} y); "
        "gridpoint count is NOT independent events; no significance testing",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if os.path.exists(outfig):
        print(f"SKIP (exists, not overwriting): {outfig}")
    else:
        fig.savefig(outfig, dpi=150, bbox_inches="tight")
        print(f"wrote {outfig}")
    plt.close(fig)


def write_report(years, missing_years, main_rows, filt_rows, yr_rows, ytag, out_md):
    L = []
    L.append("# Multi-year (2011–2022) wind–solar temporal correlation (heatwave vs ordinary days) exploration\n\n")
    L.append("A resource-level temporal-complementarity exploration. **Not** a joint-generation stability "
             "assessment, **not** a causal attribution of heatwave effects.\n\n")

    L.append("## Inputs and coverage\n\n")
    L.append(f"- Complete years actually included: {years} ({len(years)} years, each with a complete May–Sep "
             "warm season).\n")
    if missing_years:
        L.append(f"- **Missing / excluded years: {missing_years}** (data incomplete; not mixed in as complete "
             "warm seasons).\n")
    else:
        L.append("- All 2011–2022 years are fully included.\n")
    L.append("- Daily potential: per-year `era5_daily_potential_YYYY0501_YYYY0930_{A,B}.nc`;\n")
    L.append("- Aligned CN05.1 labels: per-year `cn05_labels_YYYY0501_YYYY0930_aligned.nc` (day boundary UNCONFIRMED).\n")
    L.append("- Main analysis: Faiman daily-total PV potential (`pvpot_sum`) + daily-mean WPD (`wpd_mean`), "
             "boundary B; boundary A as sensitivity.\n\n")

    L.append("## Method\n\n")
    L.append("- **Pooled by calendar month**: for each month (May–Sep) pool all dates of that month across "
             "2011–2022, and per gridpoint take heatwave days (label=3) and ordinary non-exceedance days "
             "(label=1); exclude label=0/2/4, and do not fold gap days into ordinary days. **The whole "
             "May–Sep period is not mixed directly into one grand correlation coefficient.**\n")
    L.append("- For each group use the paired day sample where PV and WPD are both finite, and compute Pearson r, "
             "Spearman rho (average rank), the paired valid-day count n, the number of contributing years, the "
             "number of contiguous date segments and the longest segment in days (within-year contiguous, "
             "broken across years; only a hint of sample concentration, not interpreted as an independent-event "
             "count).\n")
    L.append(f"- Main display: both groups ≥ {MAIN_MIN} days; exploratory sensitivity: both groups ≥ {SENS_MIN} "
             "days. The threshold is a display filter, **not** a statistical reliability criterion, and was "
             "not adjusted to the results.\n")
    L.append("- If the paired sample is insufficient (n<2) or either variable has no variation, r and rho are "
             "set to NaN and a reason is recorded (1=insufficient sample, 2=PV constant, 3=WPD constant).\n")
    L.append("- Between groups: delta_r = r_heatwave − r_ordinary, delta_rho = rho_heatwave − rho_ordinary, "
             "computed only on the common gridpoints where both groups are computable; both group coefficients "
             "and the differences are kept.\n")
    L.append("- Per-year-per-month results are kept separately (`wind_solar_corr_multi_per_year_*.nc` / "
             "`per_year_monthly_summary_*_B.csv`) to check whether pooling masks year differences.\n\n")

    L.append("## Main results (boundary B, both groups ≥ 10 days, pooled by calendar month)\n\n")
    L.append("| month | valid gridpoints | n_hw median(years/segments) | n_ne median(years/segments) | r_heatwave median | r_ordinary median | "
             "delta_r median(P25–P75) | delta_r pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for r in main_rows:
        if r["n_valid"]:
            hw = f"{r['n_hw_median']:.0f}({r['n_hw_years_median']:.0f}/{r['n_hw_seg_median']:.0f})"
            ne = f"{r['n_ne_median']:.0f}({r['n_ne_years_median']:.0f}/{r['n_ne_seg_median']:.0f})"
            L.append(f"| {r['month']} | {r['n_valid']} | {hw} | {ne} | {r['r_hw_median']:.3f} | "
                     f"{r['r_ne_median']:.3f} | {r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 (no gridpoints meeting the threshold; reported honestly) | — | — | — | — | — | — |\n")
    L.append("\n> \"n_hw/n_ne median(years/segments)\" is the median of that group's paired valid-day count; the "
             "parentheses give the median number of contributing years / the median number of contiguous date "
             "segments.\n")
    L.append("> The \"difference of group medians\" (r_heatwave median − r_ordinary median) and the \"median of "
             "per-gridpoint differences\" (delta_r median) are two different things; they are listed separately "
             "and not conflated.\n\n")

    L.append("### Spearman (same threshold)\n\n")
    L.append("| month | valid gridpoints | rho_heatwave median | rho_ordinary median | delta_rho median(P25–P75) | pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in main_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | {r['rho_hw_median']:.3f} | {r['rho_ne_median']:.3f} | "
                     f"{r['delta_rho_median']:.3f}({r['delta_rho_p25']:.3f}–{r['delta_rho_p75']:.3f}) | "
                     f"{r['delta_rho_pos']}/{r['delta_rho_zero']}/{r['delta_rho_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 | — | — | — | — |\n")
    L.append("\n")

    L.append(f"## Sample-size sensitivity (both groups ≥ {SENS_MIN} days)\n\n")
    L.append("| month | valid gridpoints | delta_r median(P25–P75) | pos/zero/neg |\n")
    L.append("|---|---|---|---|\n")
    for r in filt_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | "
                     f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 (no gridpoints after filtering) | — | — |\n")
    L.append("\n")

    L.append("## Per-year-per-month delta_r (checks whether pooling masks year differences, boundary B)\n\n")
    L.append("The table below gives the per-year-per-month delta_r distributions (median and quartiles across "
             "gridpoints), used to observe whether pooling to the calendar month masks differences between "
             "years.\n\n")
    L.append("| year | month | valid gridpoints | delta_r median(P25–P75) | n_hw median | n_ne median |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in yr_rows:
        if r["n_valid"]:
            L.append(f"| {r['year']} | {r['month']} | {r['n_valid']} | "
                     f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['n_hw_median']:.0f} | {r['n_ne_median']:.0f} |\n")
        else:
            L.append(f"| {r['year']} | {r['month']} | 0 | — | — | — |\n")
    L.append("\n")

    L.append("## Figure\n\n")
    L.append("- `fig_wind_solar_corr_multi_monthly_B_*.png` — top panel: Pearson r distributions of the two "
             "groups (red=heatwave, blue=ordinary); bottom panel: delta_r distribution; each month uses the "
             "same common gridpoint set and annotates n; the title notes the years actually included.\n\n")

    L.append("## Interpretation boundaries (important)\n\n")
    L.append("- A negative delta only means the correlation moves in a more negative direction; if both groups "
             "are positive, it cannot be directly called \"forming reverse complementarity\".\n")
    L.append("- A change in the correlation coefficient is not directly equivalent to an improvement in joint "
             "supply stability, nor can it be interpreted as a causal effect of heatwaves.\n")
    L.append("- Each group is a selected day-sample, **not** the correlation of a single continuous heatwave "
             "process.\n")
    L.append("- There are issues of temporal autocorrelation, spatial dependence, unequal group sample sizes, "
             "high-temperature-condition selection, and limited variable ranges.\n")
    L.append("- Pooling across years enlarges the sample, but different years may have different heatwave "
             "backgrounds and weather patterns, so the cross-year pooled results do not automatically represent "
             "a \"typical heatwave\" or a long-term pattern.\n")
    L.append("- No conventional independent-sample p-values, significance maps, or confidence intervals were "
             "done; WPD and PV potential are not summed directly.\n\n")

    write_text_guarded(out_md, "".join(L))


if __name__ == "__main__":
    main()
