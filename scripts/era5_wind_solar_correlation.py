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
    L.append("# 2022 wind–solar temporal correlation (heatwave vs ordinary days) exploration\n\n")
    L.append("A resource-level temporal-complementarity exploration. **Not** a joint-generation stability "
             "assessment, **not** a causal attribution of heatwave effects.\n\n")

    L.append("## Inputs\n\n")
    L.append(f"- Daily potential (boundary A): `{os.path.basename(args.daily_a)}`\n")
    L.append(f"- Daily potential (boundary B, provisional primary): `{os.path.basename(args.daily_b)}`\n")
    L.append(f"- Aligned CN05.1 labels: `{os.path.basename(args.labels)}` (day boundary UNCONFIRMED)\n")
    L.append("- Main analysis: Faiman daily-total PV potential (`pvpot_sum`) + daily-mean WPD (`wpd_mean`), "
             "boundary B; boundary A as sensitivity.\n\n")

    L.append("## Method\n\n")
    L.append("- Per gridpoint per month, take heatwave days (label=3) and ordinary non-exceedance days "
             "(label=1) separately; exclude label=0/2/4, and do not fold gap days into ordinary days.\n")
    L.append("- For each group use the paired day sample where PV and WPD are both finite, and compute "
             "Pearson r, Spearman rho (average rank), the paired valid-day count n, the two variables' "
             "standard deviations (ddof=1), the number of contiguous date segments and the longest segment "
             "in days (only a hint of sample concentration, not interpreted as an independent-event count).\n")
    L.append(f"- Main display: both groups ≥ {MAIN_MIN} days; exploratory sensitivity: both groups ≥ {SENS_MIN} "
             "days. The threshold is a display filter, **not** a statistical reliability criterion, and was "
             "not adjusted to the results.\n")
    L.append("- If the paired sample is insufficient (n<2) or either variable has no variation, r and rho are "
             "set to NaN and a reason is recorded (1=insufficient sample, 2=PV constant, 3=WPD constant).\n")
    L.append("- Between groups: delta_r = r_heatwave − r_ordinary, delta_rho = rho_heatwave − rho_ordinary, "
             "computed only on the common gridpoints where both groups are computable; both group coefficients "
             "and the differences are kept.\n\n")

    L.append("## Main results (boundary B, both groups ≥ 10 days)\n\n")
    L.append("| month | valid gridpoints | r_heatwave median | r_ordinary median | difference of group medians | "
             "delta_r median(P25–P75) | delta_r pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for r in main_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | {r['r_hw_median']:.3f} | "
                     f"{r['r_ne_median']:.3f} | {r['r_median_diff']:.3f} | "
                     f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                     f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 (no gridpoints meeting the threshold; reported honestly) | — | — | — | — | — |\n")
    L.append("\n> The \"difference of group medians\" (r_heatwave median − r_ordinary median) and the \"median "
             "of per-gridpoint differences\" (delta_r median) are two different things; they are listed "
             "separately and not conflated.\n\n")

    L.append("### Spearman (same threshold)\n\n")
    L.append("| month | valid gridpoints | rho_heatwave median | rho_ordinary median | delta_rho median(P25–P75) | pos/zero/neg |\n")
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

    L.append("## Pearson vs Spearman direction consistency (direction of delta_r vs delta_rho, boundary B)\n\n")
    L.append("> This table compares the **directions of the between-group differences delta_r and delta_rho**, "
             "not the sign consistency of the raw correlation coefficients r and rho.\n\n")
    for min_n in (MAIN_MIN, SENS_MIN):
        L.append(f"### Both groups ≥ {min_n} days\n\n")
        L.append("| month | common gridpoints | same | opposite | both-zero | only-one-zero | opposite/common |\n")
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

    L.append("## Boundary A/B sensitivity (delta_r, common valid gridpoints)\n\n")
    L.append("> Compares the direction and magnitude of delta_r under boundaries A and B; the four classes "
             "(same/opposite/both-zero/only-one-zero) sum to the common-gridpoint count.\n\n")
    for min_n in (MAIN_MIN, SENS_MIN):
        L.append(f"### Both groups ≥ {min_n} days\n\n")
        L.append("| month | common gridpoints | delta_r_A median(P25–P75) | delta_r_B median(P25–P75) | "
                 "delta(B−A) median(P25–P75) | same/opposite/both-zero/only-one-zero | opposite/common |\n")
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

    L.append("## July sample-filter sensitivity (≥10's 31 gridpoints vs ≥5's additionally-included 375 gridpoints)\n\n")
    L.append(f"Whether the ≥10's 31 gridpoints are a subset of the ≥5's 406 gridpoints: {'yes' if subset_ok else 'no'}.\n\n")
    L.append("| part | n | n_hw | n_ne | delta_r median(P25–P75) | pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in jul_rows:
        label = "retained(≥10)" if r["part"] == "retained" else "extra(5–9)"
        hw_rng = f"{r['n_hw_min']}–{r['n_hw_max']}" if r["n"] else "—"
        ne_rng = f"{r['n_ne_min']}–{r['n_ne_max']}" if r["n"] else "—"
        L.append(f"| {label} | {r['n']} | {hw_rng} | {ne_rng} | "
                 f"{r['delta_r_median']:.3f}({r['delta_r_p25']:.3f}–{r['delta_r_p75']:.3f}) | "
                 f"{r['delta_r_pos']}/{r['delta_r_zero']}/{r['delta_r_neg']} |\n")
    L.append("\n")
    L.append("> Changing the threshold also changes the spatial sample set; at present we cannot separate the "
             "contributions of \"small-sample instability\" and \"spatial differences\", nor therefore assert "
             "that the sample size caused a direction reversal.\n\n")

    L.append("## Figure\n\n")
    L.append("- `fig_wind_solar_corr_monthly_B.png` — top panel: Pearson r distributions of the two groups "
             "(red=heatwave, blue=ordinary); bottom panel: delta_r distribution; each month uses the same "
             "common gridpoint set and annotates n.\n\n")

    L.append("## Interpretation boundaries (important)\n\n")
    L.append("- A negative delta only means the correlation moves in a more negative direction; if both groups "
             "are positive, it cannot be directly called \"forming reverse complementarity\".\n")
    L.append("- A change in the correlation coefficient is not directly equivalent to an improvement in joint "
             "supply stability.\n")
    L.append("- Each group is a selected day-sample, **not** the correlation of a single continuous heatwave "
             "process.\n")
    L.append("- There are issues of temporal autocorrelation, spatial dependence, unequal group sample sizes, "
             "high-temperature-condition selection, and limited variable ranges.\n")
    L.append("- Single-year results cannot be interpreted as a causal effect of heatwaves or a long-term pattern.\n")
    L.append("- No conventional independent-sample p-values, significance maps, or confidence intervals were done.\n")
    L.append("- The \"direction consistency\" and A/B sensitivity above compare the **direction of delta_r**, "
             "not the sign consistency of the raw correlation coefficient r.\n")

    write_text_guarded(out_md, "".join(L))


if __name__ == "__main__":
    main()
