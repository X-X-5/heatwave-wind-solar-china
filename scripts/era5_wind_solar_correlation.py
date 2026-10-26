# -*- coding: utf-8 -*-
"""
Single-year (May-Sep 2022) EXPLORATORY description of the wind-solar *temporal*
correlation (Pearson r and Spearman rho) in heatwave vs ordinary days.

This is a RESOURCE-LEVEL temporal-complementarity exploration. It is NOT a joint
generation stability assessment and NOT a causal attribution of heatwave effects.

Reads the EXISTING daily potential products (boundary A and B) and the aligned CN05.1
labels. No download, no recompute, no change to labels / formulas / heatwave definition.

Per gridpoint per month, for each group (heatwave = label 3, ordinary = label 1;
label 0/2/4 excluded, gap days NOT folded into ordinary):
  * paired valid-day sample = days with BOTH pvpot_sum (Faiman) and wpd_mean finite;
  * Pearson r and Spearman rho of (PV, WPD) over that sample;
  * paired valid-day count n, sample std of PV and WPD;
  * number of contiguous day segments and the longest segment (sample concentration
    hint only -- NOT treated as independent events).

Display thresholds (NOT sufficiency criteria): main = both groups >= 10 days;
exploratory sensitivity = both groups >= 5 days. Thresholds are never adjusted to
obtain more gridpoints or better-looking results.

If the paired sample is insufficient (n < 2) or either variable has zero variance,
the coefficients are NaN and a reason flag is recorded.

Group comparison (per gridpoint): delta_r = r_heatwave - r_ordinary,
delta_rho = rho_heatwave - rho_ordinary, computed only where both groups are
computable; both group coefficients and the deltas are kept.

Interpretation guardrails (do not over-read):
  * a negative delta only means the correlation shifted more negative;
  * if both groups are positive, that is NOT "reverse complementarity";
  * a correlation change is NOT a reliability change;
  * each group is a selected day-sample, not a single continuous heatwave episode.

No significance tests, no p-values, no confidence intervals, no concatenation of
months into one correlation.
"""
import os
import sys
import json
import csv
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
DEFAULT_OUT = os.path.join(ROOT, "results_complementarity")

MONTHS = ["2022-05", "2022-06", "2022-07", "2022-08", "2022-09"]
MONTH_SHORT = [calendar.month_abbr[int(m[5:7])] for m in MONTHS]

MAIN_MIN = 10       # main display: both groups >= 10 paired valid days
SENS_MIN = 5        # exploratory sensitivity: both groups >= 5 days
ZERO_TOL = 1e-12    # |delta| <= tol counted as "zero"

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


def contiguous(idx):
    """Number of contiguous runs and longest run for sorted 0-based day indices."""
    if idx.size == 0:
        return 0, 0
    if idx.size == 1:
        return 1, 1
    gaps = np.diff(idx)
    n_seg = int((gaps > 1).sum()) + 1
    brk = np.where(gaps > 1)[0]
    lengths = np.diff(np.concatenate(([-1], brk, [idx.size - 1])))
    return int(n_seg), int(lengths.max())


def write_text_guarded(path, text):
    if os.path.exists(path):
        print(f"SKIP (exists, not overwriting): {path}")
        return "skipped"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}")
    return "written"


def write_csv(path, rows, fieldnames):
    if os.path.exists(path):
        print(f"SKIP (exists, not overwriting): {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-a", required=True, help="daily potential NetCDF, boundary A")
    ap.add_argument("--daily-b", required=True, help="daily potential NetCDF, boundary B")
    ap.add_argument("--labels", required=True, help="aligned CN05.1 daily-label NetCDF")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()

    for p in (args.daily_a, args.daily_b, args.labels):
        if not os.path.exists(p):
            raise SystemExit(f"input not found: {p}")
    os.makedirs(args.out, exist_ok=True)

    daA = xr.open_dataset(args.daily_a)
    daB = xr.open_dataset(args.daily_b)
    lab = xr.open_dataset(args.labels)

    dates = daA["date"].values
    lat = daA["latitude"].values
    lon = daA["longitude"].values
    nlat, nlon = lat.size, lon.size
    nmon = len(MONTHS)

    label = lab["label"].values
    pv = {"A": daA["pvpot_sum"].values.astype(np.float64),
          "B": daB["pvpot_sum"].values.astype(np.float64)}
    wpd = {"A": daA["wpd_mean"].values.astype(np.float64),
           "B": daB["wpd_mean"].values.astype(np.float64)}

    shp = (nmon, nlat, nlon)
    res = {}
    for bd in ("A", "B"):
        for g in ("hw", "ne"):
            res[f"r_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"rho_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"n_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"pv_sd_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"wpd_sd_{g}_{bd}"] = np.full(shp, np.nan)
            res[f"n_seg_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"max_seg_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
            res[f"flag_{g}_{bd}"] = np.zeros(shp, dtype=np.int16)
        res[f"delta_r_{bd}"] = np.full(shp, np.nan)
        res[f"delta_rho_{bd}"] = np.full(shp, np.nan)

    for mi, mo in enumerate(MONTHS):
        tmask = dates.astype("datetime64[M]") == np.datetime64(mo)
        lab_m = label[tmask]
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
                pv_sd_arr = np.full((nlat, nlon), np.nan)
                wpd_sd_arr = np.full((nlat, nlon), np.nan)
                n_seg_arr = np.zeros((nlat, nlon), np.int16)
                max_seg_arr = np.zeros((nlat, nlon), np.int16)
                flag_arr = np.zeros((nlat, nlon), np.int16)
                for i in range(nlat):
                    for j in range(nlon):
                        s = sel[:, i, j]
                        x = pv_m[s, i, j]
                        y = wpd_m[s, i, j]
                        n = x.size
                        n_arr[i, j] = n
                        if n >= 2:
                            pv_sd_arr[i, j] = float(np.std(x, ddof=1))
                            wpd_sd_arr[i, j] = float(np.std(y, ddof=1))
                        n_seg_arr[i, j], max_seg_arr[i, j] = contiguous(np.where(s)[0])
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
                res[f"pv_sd_{g}_{bd}"][mi] = pv_sd_arr
                res[f"wpd_sd_{g}_{bd}"][mi] = wpd_sd_arr
                res[f"n_seg_{g}_{bd}"][mi] = n_seg_arr
                res[f"max_seg_{g}_{bd}"][mi] = max_seg_arr
                res[f"flag_{g}_{bd}"][mi] = flag_arr
            res[f"delta_r_{bd}"][mi] = res[f"r_hw_{bd}"][mi] - res[f"r_ne_{bd}"][mi]
            res[f"delta_rho_{bd}"][mi] = res[f"rho_hw_{bd}"][mi] - res[f"rho_ne_{bd}"][mi]

    # ---------------- assemble NetCDF ----------------
    month_coord = np.array([np.datetime64(mo + "-01") for mo in MONTHS], dtype="datetime64[D]")
    ds = xr.Dataset(
        {name: (("month", "latitude", "longitude"), arr) for name, arr in res.items()},
        coords={"month": month_coord, "latitude": lat, "longitude": lon},
    )
    for bd in ("A", "B"):
        for g, glong in (("hw", "heatwave (label 3)"), ("ne", "ordinary non-exceedance (label 1)")):
            ds[f"r_{g}_{bd}"].attrs = {"units": "1", "long_name": f"Pearson r (PV, WPD), {glong}, boundary {bd}"}
            ds[f"rho_{g}_{bd}"].attrs = {"units": "1", "long_name": f"Spearman rho (PV, WPD), {glong}, boundary {bd}"}
            ds[f"n_{g}_{bd}"].attrs = {"units": "days", "long_name": f"paired valid-day count, {glong}, boundary {bd}"}
            ds[f"pv_sd_{g}_{bd}"].attrs = {"units": "kWh m-2", "long_name": f"sample std of PV (ddof=1), {glong}, boundary {bd}"}
            ds[f"wpd_sd_{g}_{bd}"].attrs = {"units": "W m-2", "long_name": f"sample std of WPD (ddof=1), {glong}, boundary {bd}"}
            ds[f"n_seg_{g}_{bd}"].attrs = {"units": "1", "long_name": f"number of contiguous day segments, {glong}, boundary {bd}"}
            ds[f"max_seg_{g}_{bd}"].attrs = {"units": "days", "long_name": f"longest contiguous segment, {glong}, boundary {bd}"}
            ds[f"flag_{g}_{bd}"].attrs = {"long_name": f"reason flag, {glong}, boundary {bd}",
                                          "flag_codes": "0=ok, 1=insufficient pairs (n<2), 2=PV constant, 3=WPD constant"}
        ds[f"delta_r_{bd}"].attrs = {"units": "1", "long_name": f"r_heatwave - r_ordinary, boundary {bd}"}
        ds[f"delta_rho_{bd}"].attrs = {"units": "1", "long_name": f"rho_heatwave - rho_ordinary, boundary {bd}"}

    ds.attrs = {
        "title": "Single-year (2022) wind-solar temporal correlation, heatwave vs ordinary",
        "correlation": "Pearson r and Spearman rho of (pvpot_sum, wpd_mean) over the paired valid-day sample",
        "paired_valid_day": "both pvpot_sum (Faiman) and wpd_mean finite",
        "groups": "heatwave = label 3; ordinary = label 1; excluded = label 0/2/4; gap days NOT folded into ordinary",
        "nan_reasons": "1=insufficient pairs (n<2), 2=PV constant, 3=WPD constant",
        "std_ddof": 1,
        "main_min_days": MAIN_MIN,
        "sensitivity_min_days": SENS_MIN,
        "threshold_note": "display filter, NOT a statistical sufficiency criterion; never adjusted for more/better results",
        "boundary": "B = 20-20 BJT (provisional primary); A = 00-24 BJT (sensitivity)",
        "pv_scheme": "pvpot = Faiman (provisional primary); wpd = 0.5*rho*V100^3",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "not_an_independence_measure": "days are autocorrelated and spatially dependent; gridpoint count is NOT independent events",
        "not_a_reliability_or_causal_statement": "single-year, descriptive; no causal attribution, no reliability conclusion",
        "source_daily": f"{os.path.basename(args.daily_a)}, {os.path.basename(args.daily_b)}",
        "source_labels": os.path.basename(args.labels),
        "config_signature": json.dumps({
            "correlation": ["pearson", "spearman"],
            "spearman_rank_method": "average",
            "groups": {"heatwave": 3, "ordinary": 1, "excluded": [0, 2, 4]},
            "metrics": {"pv": "pvpot_sum", "wpd": "wpd_mean"},
            "boundaries": ["A", "B"],
            "main_min_days": MAIN_MIN,
            "sensitivity_min_days": SENS_MIN,
            "zero_tol": ZERO_TOL,
            "delta": "(hw - ordinary)",
        }, sort_keys=True),
    }
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.integer):
            enc[v] = {"zlib": True, "complevel": 4, "dtype": str(ds[v].dtype)}
        else:
            enc[v] = {"zlib": True, "complevel": 4, "_FillValue": np.nan}

    nc_out = os.path.join(args.out, "wind_solar_corr_20220501_20220930.nc")
    write_guarded(ds, nc_out, enc)

    # ---------------- verification ----------------
    print("=" * 70)
    print("verification:")
    for bd in ("A", "B"):
        for g in ("hw", "ne"):
            r = res[f"r_{g}_{bd}"]
            rho = res[f"rho_{g}_{bd}"]
            flag = res[f"flag_{g}_{bd}"]
            ok_r = bool(np.all(np.isnan(r) | ((r >= -1) & (r <= 1))))
            ok_rho = bool(np.all(np.isnan(rho) | ((rho >= -1) & (rho <= 1))))
            ok_flag = bool(np.all(np.isfinite(r) == (flag == 0)))
            n = res[f"n_{g}_{bd}"]
            ok_n = bool(np.all((n >= 0) & (n <= 31)))
            print(f"  {bd}/{g}: r in [-1,1] {ok_r}, rho in [-1,1] {ok_rho}, "
                  f"finite<->flag0 {ok_flag}, n in [0,31] {ok_n}")
        dr = res[f"delta_r_{bd}"]
        drho = res[f"delta_rho_{bd}"]
        print(f"  {bd} delta_r in [-2,2] {bool(np.all(np.isnan(dr) | ((dr >= -2) & (dr <= 2))))}, "
              f"delta_rho in [-2,2] {bool(np.all(np.isnan(drho) | ((drho >= -2) & (drho <= 2))))}")

    # ---------------- summaries ----------------
    def monthly_summary(delta_r, r_hw, r_ne, delta_rho, rho_hw, rho_ne, n_hw, n_ne, min_n):
        rows = []
        for mi, mo in enumerate(MONTHS):
            common = (np.isfinite(delta_r[mi]) & (n_hw[mi] >= min_n) & (n_ne[mi] >= min_n)).ravel()
            dr = delta_r[mi].ravel()[common]
            drho = delta_rho[mi].ravel()[common]
            rhw = r_hw[mi].ravel()[common]
            rne = r_ne[mi].ravel()[common]
            rhohw = rho_hw[mi].ravel()[common]
            rhone = rho_ne[mi].ravel()[common]
            nhw = n_hw[mi].ravel()[common]
            nne = n_ne[mi].ravel()[common]
            n_valid = int(common.sum())
            dr_pos, dr_zero, dr_neg = pos_zero_neg(dr) if n_valid else (0, 0, 0)
            drho_pos, drho_zero, drho_neg = pos_zero_neg(drho) if n_valid else (0, 0, 0)
            rows.append({
                "month": mo, "n_valid": n_valid,
                "n_hw_median": pct(nhw, 50) if n_valid else "",
                "n_ne_median": pct(nne, 50) if n_valid else "",
                "r_hw_median": pct(rhw, 50), "r_ne_median": pct(rne, 50),
                "r_median_diff": (pct(rhw, 50) - pct(rne, 50)),
                "delta_r_median": pct(dr, 50), "delta_r_p25": pct(dr, 25),
                "delta_r_p75": pct(dr, 75),
                "delta_r_pos": dr_pos, "delta_r_zero": dr_zero, "delta_r_neg": dr_neg,
                "rho_hw_median": pct(rhohw, 50), "rho_ne_median": pct(rhone, 50),
                "rho_median_diff": (pct(rhohw, 50) - pct(rhone, 50)),
                "delta_rho_median": pct(drho, 50), "delta_rho_p25": pct(drho, 25),
                "delta_rho_p75": pct(drho, 75),
                "delta_rho_pos": drho_pos, "delta_rho_zero": drho_zero,
                "delta_rho_neg": drho_neg,
            })
        return rows

    main_rows = monthly_summary(res["delta_r_B"], res["r_hw_B"], res["r_ne_B"],
                                res["delta_rho_B"], res["rho_hw_B"], res["rho_ne_B"],
                                res["n_hw_B"], res["n_ne_B"], MAIN_MIN)
    filt_rows = monthly_summary(res["delta_r_B"], res["r_hw_B"], res["r_ne_B"],
                                res["delta_rho_B"], res["rho_hw_B"], res["rho_ne_B"],
                                res["n_hw_B"], res["n_ne_B"], SENS_MIN)

    write_csv(os.path.join(args.out, "monthly_summary_B.csv"), main_rows, list(main_rows[0].keys()))
    write_csv(os.path.join(args.out, "sample_filter_summary_B.csv"), filt_rows, list(filt_rows[0].keys()))

    # Pearson vs Spearman direction consistency (boundary B, both filters).
    # The comparison is about the DIRECTION of delta_r vs delta_rho (the group
    # differences), not about the sign of the raw r/rho coefficients.
    def direction_counts(a, b):
        sa = sign3(a); sb = sign3(b)
        return (int(((sa == sb) & (sa != 0)).sum()),
                int(((sa != sb) & (sa != 0) & (sb != 0)).sum()),
                int(((sa == 0) & (sb == 0)).sum()),
                int(((sa == 0) != (sb == 0)).sum()))

    ps_rows = []
    for min_n in (MAIN_MIN, SENS_MIN):
        for mi, mo in enumerate(MONTHS):
            common = (np.isfinite(res["delta_r_B"][mi]) & np.isfinite(res["delta_rho_B"][mi])
                      & (res["n_hw_B"][mi] >= min_n) & (res["n_ne_B"][mi] >= min_n)).ravel()
            dr = res["delta_r_B"][mi].ravel()[common]
            drho = res["delta_rho_B"][mi].ravel()[common]
            n = int(common.sum())
            if n:
                same, opp, bz, oz = direction_counts(dr, drho)
            else:
                same = opp = bz = oz = 0
            ps_rows.append({"min_days": min_n, "month": mo, "n_common": n,
                            "n_same_sign": same, "n_opposite_sign": opp,
                            "n_both_zero": bz, "n_only_one_zero": oz})

    # A/B sensitivity on common valid gridpoints (both filters, both boundaries)
    ab_rows = []
    for min_n in (MAIN_MIN, SENS_MIN):
        for mi, mo in enumerate(MONTHS):
            ca = (np.isfinite(res["delta_r_A"][mi]) & (res["n_hw_A"][mi] >= min_n)
                  & (res["n_ne_A"][mi] >= min_n)).ravel()
            cb = (np.isfinite(res["delta_r_B"][mi]) & (res["n_hw_B"][mi] >= min_n)
                  & (res["n_ne_B"][mi] >= min_n)).ravel()
            shared = ca & cb
            a = res["delta_r_A"][mi].ravel()[shared]
            b = res["delta_r_B"][mi].ravel()[shared]
            n = int(shared.sum())
            delta = b - a
            if n:
                same, opp, bz, oz = direction_counts(a, b)
            else:
                same = opp = bz = oz = 0
            ab_rows.append({
                "min_days": min_n, "month": mo, "n_shared": n,
                "delta_r_A_median": pct(a, 50), "delta_r_A_p25": pct(a, 25),
                "delta_r_A_p75": pct(a, 75),
                "delta_r_B_median": pct(b, 50), "delta_r_B_p25": pct(b, 25),
                "delta_r_B_p75": pct(b, 75),
                "delta_ab_median": pct(delta, 50), "delta_ab_p25": pct(delta, 25),
                "delta_ab_p75": pct(delta, 75),
                "n_ab_same_sign": same, "n_ab_opposite_sign": opp,
                "n_ab_both_zero": bz, "n_ab_only_one_zero": oz,
            })
    write_csv(os.path.join(args.out, "ab_sensitivity.csv"), ab_rows, list(ab_rows[0].keys()))

    # July sample-filter split: retained (>=10) vs extra (5..9), boundary B
    jul_rows = []
    mi_jul = 2
    c10 = (np.isfinite(res["delta_r_B"][mi_jul]) & (res["n_hw_B"][mi_jul] >= MAIN_MIN)
           & (res["n_ne_B"][mi_jul] >= MAIN_MIN))
    c5 = (np.isfinite(res["delta_r_B"][mi_jul]) & (res["n_hw_B"][mi_jul] >= SENS_MIN)
          & (res["n_ne_B"][mi_jul] >= SENS_MIN))
    subset_ok = bool(np.all(c10 <= c5))
    extra = c5 & ~c10
    for part, m in (("retained", c10), ("extra", extra)):
        d = res["delta_r_B"][mi_jul][m]
        hw = res["n_hw_B"][mi_jul][m]
        ne = res["n_ne_B"][mi_jul][m]
        pos, zero, neg = pos_zero_neg(d) if d.size else (0, 0, 0)
        jul_rows.append({
            "part": part, "n": int(m.sum()),
            "n_hw_min": int(hw.min()) if hw.size else "",
            "n_hw_max": int(hw.max()) if hw.size else "",
            "n_ne_min": int(ne.min()) if ne.size else "",
            "n_ne_max": int(ne.max()) if ne.size else "",
            "delta_r_median": pct(d, 50), "delta_r_p25": pct(d, 25),
            "delta_r_p75": pct(d, 75),
            "delta_r_pos": pos, "delta_r_zero": zero, "delta_r_neg": neg,
        })

    # ---------------- figure ----------------
    fig_png = os.path.join(args.out, "fig_wind_solar_corr_monthly_B.png")
    make_fig(res["delta_r_B"], res["r_hw_B"], res["r_ne_B"], res["n_hw_B"], res["n_ne_B"],
             fig_png)

    # ---------------- report ----------------
    write_report(args, main_rows, filt_rows, ps_rows, ab_rows, jul_rows, subset_ok,
                 os.path.join(args.out, "REPORT_wind_solar_corr.md"))

    # ---------------- headline print ----------------
    print("=" * 70)
    print("delta_r = r_heatwave - r_ordinary (boundary B, both groups >=10 days):")
    for r in main_rows:
        print(f"  {r['month']}: n_valid={r['n_valid']}  r_hw_med={r['r_hw_median']} "
              f"r_ne_med={r['r_ne_median']}  delta_r_med={r['delta_r_median']} "
              f"(P25 {r['delta_r_p25']}, P75 {r['delta_r_p75']})  "
              f"pos/zero/neg={r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']}")
    print("DONE wind-solar correlation")


def pos_zero_neg(x, tol=ZERO_TOL):
    return (int((x > tol).sum()), int((np.abs(x) <= tol).sum()), int((x < -tol).sum()))


def make_fig(delta_r_B, r_hw_B, r_ne_B, n_hw_B, n_ne_B, outfig):
    nmon = len(MONTHS)
    common = (np.isfinite(delta_r_B) & (n_hw_B >= MAIN_MIN) & (n_ne_B >= MAIN_MIN))
    hw_boxes, ne_boxes, delta_boxes, ns = [], [], [], []
    for mi in range(nmon):
        m = common[mi].ravel()
        hw_boxes.append(r_hw_B[mi].ravel()[m])
        ne_boxes.append(r_ne_B[mi].ravel()[m])
        delta_boxes.append(delta_r_B[mi].ravel()[m])
        ns.append(int(m.sum()))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.8), sharex=False)

    # -- top: Pearson r, heatwave vs ordinary --
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
    ax.set_title(f"Pearson r — heatwave (red) vs ordinary (blue), common gridpoints (both groups ≥{MAIN_MIN} days)",
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

    # -- bottom: delta_r distribution --
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
    ax2.set_xlabel("2022 month", fontsize=9)
    ax2.set_ylabel("Δr = r_hw − r_ne", fontsize=9)
    ax2.set_title("Δr distribution (same common gridpoints as above)", loc="left", fontsize=9.5)
    ax2.grid(True, axis="y", ls=":", lw=0.5, alpha=0.45)
    ax2.tick_params(axis="y", labelsize=8.5)
    for mi, n in enumerate(ns):
        if n == 0:
            ax2.text(mi + 1, 0, "no valid gridpoints", ha="center", va="center",
                     fontsize=8, color="#c62828")

    fig.suptitle(
        "Wind–solar temporal correlation: heatwave vs ordinary days (boundary B)\n"
        "single-year exploratory; gridpoint count is NOT independent events; "
        "no significance testing",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if os.path.exists(outfig):
        print(f"SKIP (exists, not overwriting): {outfig}")
    else:
        fig.savefig(outfig, dpi=150, bbox_inches="tight")
        print(f"wrote {outfig}")
    plt.close(fig)


def write_report(args, main_rows, filt_rows, ps_rows, ab_rows, jul_rows, subset_ok, out_md):
    L = []
    L.append("# 2022 年风-光时间相关（热浪日 vs 普通日）探索\n\n")
    L.append("资源层面的时间互补性探索。**不是**联合发电稳定性评估，**不是**热浪因果归因。\n\n")

    L.append("## 输入\n\n")
    L.append(f"- 日尺度潜力（日界 A）：`{os.path.basename(args.daily_a)}`\n")
    L.append(f"- 日尺度潜力（日界 B，暂定主方案）：`{os.path.basename(args.daily_b)}`\n")
    L.append(f"- 对齐 CN05.1 标签：`{os.path.basename(args.labels)}`（日界 UNCONFIRMED）\n")
    L.append("- 主分析：Faiman 日累计 PV 潜力（`pvpot_sum`）+ 日平均 WPD（`wpd_mean`），日界 B；"
             "日界 A 作敏感性。\n\n")

    L.append("## 方法\n\n")
    L.append("- 逐格点、逐月，分别取热浪高温日（label=3）与普通非超阈值日（label=1）；"
             "排除 label=0/2/4，间隔日不混入普通日。\n")
    L.append("- 每组使用 PV 与 WPD 同时有限的成对日样本，计算 Pearson r、Spearman rho"
             "（秩的平均秩）、成对有效日数 n、两变量标准差（ddof=1）、"
             "连续日期段数与最长连续段天数（仅提示样本集中程度，不解释为独立事件数）。\n")
    L.append(f"- 主展示：两组各 ≥{MAIN_MIN} 天；探索性敏感性：两组各 ≥{SENS_MIN} 天。"
             "门槛为展示筛选，**不是**统计可靠性标准，未据结果调整。\n")
    L.append("- 若成对样本不足（n<2）或任一变量无变率，r 与 rho 记 NaN，并记录原因"
             "（1=样本不足、2=PV 常量、3=WPD 常量）。\n")
    L.append("- 组间：delta_r = r_热浪 − r_普通，delta_rho = rho_热浪 − rho_普通，"
             "只在两组均可计算的共同格点上计算；两组系数与差值都保留。\n\n")

    L.append("## 主结果（日界 B，两组各 ≥10 天）\n\n")
    L.append("| 月 | 有效格点 | r_热浪 中位数 | r_普通 中位数 | 两组中位数之差 | "
             "delta_r 中位数(P25–P75) | delta_r 正/零/负 |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for r in main_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | {r['r_hw_median']:.3f} | "
                     f"{r['r_ne_median']:.3f} | {r['r_median_diff']:.3f} | "
                     f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0（无满足门槛的格点，如实报告） | — | — | — | — | — |\n")
    L.append("\n> “两组中位数之差”（r_热浪中位数 − r_普通中位数）与“逐格点差值的中位数”"
             "（delta_r 中位数）是两回事，分别列出，不互相冒充。\n\n")

    L.append("### Spearman（同门槛）\n\n")
    L.append("| 月 | 有效格点 | rho_热浪 中位数 | rho_普通 中位数 | delta_rho 中位数(P25–P75) | 正/零/负 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in main_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | {r['rho_hw_median']:.3f} | "
                     f"{r['rho_ne_median']:.3f} | "
                     f"{r['delta_rho_median']:.3f}({r['delta_rho_p25']:.3f}–{r['delta_rho_p75']:.3f}) | "
                     f"{r['delta_rho_pos']}/{r['delta_rho_zero']}/{r['delta_rho_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 | — | — | — | — |\n")
    L.append("\n")

    L.append(f"## 样本量敏感性（两组各 ≥{SENS_MIN} 天）\n\n")
    L.append("| 月 | 有效格点 | delta_r 中位数(P25–P75) | 正/零/负 |\n")
    L.append("|---|---|---|---|\n")
    for r in filt_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | "
                     f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0（筛选后无格点） | — | — |\n")
    L.append("\n")

    L.append("## Pearson 与 Spearman 方向一致性（delta_r 与 delta_rho 的方向，日界 B）\n\n")
    L.append("> 本表比较的是**组间差值 delta_r 与 delta_rho 的方向**，"
             "不等同于原始相关系数 r 与 rho 的正负一致性。\n\n")
    for min_n in (MAIN_MIN, SENS_MIN):
        L.append(f"### 两组各 ≥{min_n} 天\n\n")
        L.append("| 月 | 共同格点 | 同向 | 反向 | 均零 | 仅一侧零 | 反向/共同 |\n")
        L.append("|---|---|---|---|---|---|---|\n")
        for r in ps_rows:
            if r["min_days"] != min_n:
                continue
            if r["n_common"]:
                ratio = f"{r['n_opposite_sign']}/{r['n_common']}"
                L.append(f"| {r['month']} | {r['n_common']} | {r['n_same_sign']} | "
                         f"{r['n_opposite_sign']} | {r['n_both_zero']} | "
                         f"{r['n_only_one_zero']} | {ratio} |\n")
            else:
                L.append(f"| {r['month']} | 0 | — | — | — | — | — |\n")
        L.append("\n")

    L.append("## 日界 A/B 敏感性（delta_r，共同有效格点）\n\n")
    L.append("> 比较 delta_r 在日界 A、B 下的方向与幅度；"
             "四类（同向/反向/均零/仅一侧零）之和等于共同格点数。\n\n")
    for min_n in (MAIN_MIN, SENS_MIN):
        L.append(f"### 两组各 ≥{min_n} 天\n\n")
        L.append("| 月 | 共同格点 | delta_r_A 中位数(P25–P75) | delta_r_B 中位数(P25–P75) | "
                 "delta(B−A) 中位数(P25–P75) | 同向/反向/均零/仅一侧零 | 反向/共同 |\n")
        L.append("|---|---|---|---|---|---|---|\n")
        for r in ab_rows:
            if r["min_days"] != min_n:
                continue
            if r["n_shared"]:
                ratio = f"{r['n_ab_opposite_sign']}/{r['n_shared']}"
                L.append(f"| {r['month']} | {r['n_shared']} | "
                         f"{r['delta_r_A_median']:.3f}({r['delta_r_A_p25']:.3f}–{r['delta_r_A_p75']:.3f}) | "
                         f"{r['delta_r_B_median']:.3f}({r['delta_r_B_p25']:.3f}–{r['delta_r_B_p75']:.3f}) | "
                         f"{r['delta_ab_median']:.3f}({r['delta_ab_p25']:.3f}–{r['delta_ab_p75']:.3f}) | "
                         f"{r['n_ab_same_sign']}/{r['n_ab_opposite_sign']}/{r['n_ab_both_zero']}/{r['n_ab_only_one_zero']} | "
                         f"{ratio} |\n")
            else:
                L.append(f"| {r['month']} | 0 | — | — | — | — | — |\n")
        L.append("\n")

    L.append("## 7 月样本筛选敏感性（≥10 的 31 格点 vs ≥5 额外纳入的 375 格点）\n\n")
    L.append(f"≥10 的 31 格点是否为 ≥5 的 406 格点的子集：{'是' if subset_ok else '否'}。\n\n")
    L.append("| 部分 | n | n_hw | n_ne | delta_r 中位数(P25–P75) | 正/零/负 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in jul_rows:
        label = "保留(≥10)" if r["part"] == "retained" else "额外(5–9)"
        hw_rng = f"{r['n_hw_min']}–{r['n_hw_max']}" if r["n"] else "—"
        ne_rng = f"{r['n_ne_min']}–{r['n_ne_max']}" if r["n"] else "—"
        L.append(f"| {label} | {r['n']} | {hw_rng} | {ne_rng} | "
                 f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                 f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
    L.append("\n")
    L.append("> 门槛改变同时改变了空间样本集合；目前不能区分“小样本不稳定”与“空间差异”各自的贡献，"
             "也不能据此断言样本量造成方向反转。\n\n")

    L.append("## 图\n\n")
    L.append("- `fig_wind_solar_corr_monthly_B.png` — 上图为两组 Pearson r 分布（红=热浪、蓝=普通），"
             "下图为 delta_r 分布；每月使用同一共同格点集合并标注 n。\n\n")

    L.append("## 解读边界（重要）\n\n")
    L.append("- delta 为负只表示相关关系向更负的方向移动；若两组均为正，不能直接称为“形成了反向互补”。\n")
    L.append("- 相关系数变化不直接等同于联合供能稳定性改善。\n")
    L.append("- 各组是选出的日样本集合，**不是**单次连续热浪过程的相关。\n")
    L.append("- 存在时间自相关、空间依赖、组间样本量不同、高温条件筛选及变量变幅受限等问题。\n")
    L.append("- 单年结果不能解释为热浪的因果效应或长期规律。\n")
    L.append("- 未做常规独立样本 p 值、显著性地图或置信区间。\n")
    L.append("- 上述“方向一致性”与 A/B 敏感性比较的是 **delta_r 的方向**，"
             "不等同于原始相关系数 r 的正负一致性。\n")

    write_text_guarded(out_md, "".join(L))


if __name__ == "__main__":
    main()
