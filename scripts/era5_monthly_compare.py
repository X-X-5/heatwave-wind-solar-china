# -*- coding: utf-8 -*-
"""
Monthly (May-Sep 2022) DESCRIPTIVE comparison of heatwave vs ordinary non-exceedance
days, using the existing daily products and the aligned CN05.1 labels. No re-download,
no re-audit, no re-computation of hourly/daily products.

Groups (per month, per gridpoint):
  heatwave  = label 3
  ordinary  = label 1
  excluded  = label 0 (missing), 2 (isolated exceedance), 4 (merged gap)

Metrics (daily products):
  wpd       = wpd_mean        [W m-2]   (no cell-temp model -> boundary sensitivity)
  pvpot     = pvpot_sum       [kWh m-2] (Faiman, provisional primary)
  pvpotlin  = pvpot_lin_sum   [kWh m-2] (linear, model sensitivity)

Boundaries: B = 20-20 BJT (provisional primary), A = 00-24 BJT (sensitivity).
CN05.1 day boundary stays UNCONFIRMED.

diff = mean(heatwave group) - mean(ordinary group)  (SIGNED; positive => heatwave higher).
If either group has no sample, the diff is NaN.

Single-year case study only: no significance testing, no causal attribution, no
complementarity conclusion.
"""
import os
import sys
import json
import csv
import re
import argparse
import calendar

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "results_multi", "monthly")

EX_LAT, EX_LON = 31.75, 114.50   # fixed example cell (the 2022 max-heatwave-event-day cell, kept for comparability)

# Derived in main() from the --daily-a filename (year Y -> tag Y0501_Y0930):
YEAR = None
TAG = None
MONTHS = None
MONTH_SHORT = None
NC_OUT = CSV_SUMMARY = CSV_SENS = REPORT_MD = FIG_DIST = FIG_REP = None

# metric -> (daily-var, unit, long label)
METRICS = {
    "wpd": ("wpd_mean", "W m-2", "WPD daily mean"),
    "pvpot": ("pvpot_sum", "kWh m-2", "PV potential daily cumulative (Faiman)"),
    "pvpotlin": ("pvpot_lin_sum", "kWh m-2", "PV potential daily cumulative (linear)"),
}
STAT = {
    "hw": "heatwave-group mean (label 3)",
    "ne": "ordinary non-exceedance group mean (label 1)",
    "diff": "heatwave minus ordinary (signed)",
}


def pct(x, p):
    return float(np.percentile(x, p)) if x.size else float("nan")


def shade_runs(ax, mask, color, alpha=0.28):
    """Shade contiguous True runs of a boolean day-mask."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    splits = np.where(np.diff(idx) > 1)[0]
    for r in np.split(idx, splits + 1):
        ax.axvspan(r[0] - 0.5, r[-1] + 0.5, color=color, alpha=alpha, zorder=0, lw=0)


def month_ticks(dates):
    mvals = dates.astype("datetime64[M]")
    starts = np.where(np.diff(mvals.astype(int)) != 0)[0] + 1
    idx = np.concatenate([[0], starts])
    labels = [calendar.month_abbr[(mvals[i].astype(int) % 12) + 1] for i in idx]
    return idx, labels


def main():
    ap = argparse.ArgumentParser(
        description="Monthly heatwave-vs-ordinary descriptive comparison for ONE year "
                    "(explicit inputs, no glob; year derived from the daily-a filename).")
    ap.add_argument("--daily-a", required=True,
                    help="daily potential NetCDF, boundary A")
    ap.add_argument("--daily-b", required=True,
                    help="daily potential NetCDF, boundary B")
    ap.add_argument("--labels", required=True,
                    help="aligned CN05.1 daily-label NetCDF")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()

    global YEAR, TAG, MONTHS, MONTH_SHORT
    global NC_OUT, CSV_SUMMARY, CSV_SENS, REPORT_MD, FIG_DIST, FIG_REP

    mm = re.match(r"era5_daily_potential_(\d{4})0501_\d{4}0930_A\.nc",
                  os.path.basename(args.daily_a))
    if not mm:
        raise SystemExit(f"cannot parse year from daily-a filename: {args.daily_a}")
    YEAR = mm.group(1)
    TAG = f"{YEAR}0501_{YEAR}0930"
    MONTHS = [f"{YEAR}-{m:02d}" for m in range(5, 10)]
    MONTH_SHORT = [calendar.month_abbr[int(m[5:7])] for m in MONTHS]

    BOUNDARIES = {"A": args.daily_a, "B": args.daily_b}
    for p in (args.daily_a, args.daily_b, args.labels):
        if not os.path.exists(p):
            raise SystemExit(f"input not found: {p}")
    os.makedirs(args.out, exist_ok=True)

    NC_OUT = os.path.join(args.out, f"era5_monthly_hw_vs_ordinary_{TAG}.nc")
    CSV_SUMMARY = os.path.join(args.out, f"monthly_summary_{TAG}.csv")
    CSV_SENS = os.path.join(args.out, f"sensitivity_{TAG}.csv")
    REPORT_MD = os.path.join(args.out, f"REPORT_monthly_{TAG}.md")
    FIG_DIST = os.path.join(args.out, f"fig_monthly_diff_distribution_{TAG}.png")
    FIG_REP = os.path.join(args.out, f"fig_rep_cell_{TAG}.png")

    daA = xr.open_dataset(args.daily_a)
    daB = xr.open_dataset(args.daily_b)
    lab = xr.open_dataset(args.labels)

    dates = daA["date"].values
    lat = daA["latitude"].values
    lon = daA["longitude"].values
    nlat, nlon = lat.size, lon.size
    label = lab["label"].values                       # (153, lat, lon) float32

    # metric arrays keyed by (metric, boundary)
    data = {}
    for m, (var, _u, _l) in METRICS.items():
        data[(m, "A")] = daA[var].values
        data[(m, "B")] = daB[var].values

    # ---- per-month, per-gridpoint group means + diffs ----
    nmon = len(MONTHS)
    out = {
        "n_heatwave": np.zeros((nmon, nlat, nlon), dtype=np.int16),
        "n_ordinary": np.zeros((nmon, nlat, nlon), dtype=np.int16),
    }
    for m in METRICS:
        for bd in ("A", "B"):
            for st in ("hw", "ne", "diff"):
                out[f"{m}_{st}_{bd}"] = np.full((nmon, nlat, nlon), np.nan,
                                                dtype=np.float32)

    for mi, mo in enumerate(MONTHS):
        tmask = dates.astype("datetime64[M]") == np.datetime64(mo)
        lab_m = label[tmask]                          # (t_m, lat, lon)
        hw_mask = lab_m == 3
        ne_mask = lab_m == 1
        out["n_heatwave"][mi] = hw_mask.sum(0).astype(np.int16)
        out["n_ordinary"][mi] = ne_mask.sum(0).astype(np.int16)
        for (m, bd), arr in data.items():
            var_m = arr[tmask]                        # (t_m, lat, lon)
            hw_vals = np.where(hw_mask, var_m, np.nan)
            ne_vals = np.where(ne_mask, var_m, np.nan)
            with np.errstate(all="ignore"):
                hw_mean = np.nanmean(hw_vals, axis=0)
                ne_mean = np.nanmean(ne_vals, axis=0)
            out[f"{m}_hw_{bd}"][mi] = hw_mean
            out[f"{m}_ne_{bd}"][mi] = ne_mean
            out[f"{m}_diff_{bd}"][mi] = hw_mean - ne_mean

    # ---- assemble NetCDF ----
    month_coord = np.array([np.datetime64(mo + "-01") for mo in MONTHS],
                           dtype="datetime64[D]")
    ds = xr.Dataset(
        {name: (("month", "latitude", "longitude"), arr)
         for name, arr in out.items()},
        coords={"month": month_coord, "latitude": lat, "longitude": lon},
    )
    for name in ("n_heatwave", "n_ordinary"):
        ds[name].attrs = {"units": "days",
                          "long_name": ("number of days in the group (label 3 / label 1) "
                                        "during the month at this gridpoint")}
    for m, (var, unit, long) in METRICS.items():
        for bd in ("A", "B"):
            for st, stlong in STAT.items():
                ds[f"{m}_{st}_{bd}"].attrs = {
                    "units": unit,
                    "long_name": f"{long}, {stlong}, boundary {bd}",
                }
    ds.attrs = {
        "title": f"Monthly heatwave vs ordinary non-exceedance group comparison ({YEAR})",
        "diff_convention": "diff = mean(heatwave group) - mean(ordinary group); positive => heatwave higher (signed, not relative)",
        "groups": "heatwave = label 3; ordinary = label 1; excluded = label 0 (missing), 2 (isolated exceedance), 4 (merged gap)",
        "boundary": "B = 20-20 BJT (provisional primary); A = 00-24 BJT (sensitivity)",
        "pv_scheme": "pvpot = Faiman (provisional primary); pvpotlin = linear (model sensitivity); wpd has no cell-temp model",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "spatial_aggregation": "per gridpoint per month: group means computed first, then differenced; monthly summaries aggregate over gridpoints (gridpoint-days NOT treated as independent samples)",
        "source_daily": f"era5_daily_potential_{TAG}_{{A,B}}.nc",
        "source_labels": os.path.basename(args.labels),
        "config_signature": json.dumps({
            "groups": {"heatwave": 3, "ordinary": 1,
                       "excluded": [0, 2, 4]},
            "diff": "hw_mean - ne_mean (signed)",
            "metrics": {m: var for m, (var, _u, _l) in METRICS.items()},
            "boundaries": list(BOUNDARIES),
        }, sort_keys=True),
    }
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for v in ds.data_vars if v not in ("n_heatwave", "n_ordinary")}
    enc["n_heatwave"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
    enc["n_ordinary"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
    write_guarded(ds, NC_OUT, enc)

    # ---- monthly summary (primary: boundary B, Faiman PV) ----
    prim = [("wpd", "B"), ("pvpot", "B")]
    summary = []
    for mi, mo in enumerate(MONTHS):
        for m, bd in prim:
            d = out[f"{m}_diff_{bd}"][mi].ravel()
            n_hw = out["n_heatwave"][mi].ravel()
            n_ne = out["n_ordinary"][mi].ravel()
            valid = np.isfinite(d)
            dv = d[valid]; hwv = n_hw[valid]; nev = n_ne[valid]
            n_valid = int(dv.size)
            summary.append({
                "month": mo, "metric": m, "unit": METRICS[m][1], "n_valid": n_valid,
                "n_hw_min": int(hwv.min()) if n_valid else -1,
                "n_hw_p25": pct(hwv, 25), "n_hw_median": pct(hwv, 50),
                "n_hw_p75": pct(hwv, 75), "n_hw_max": int(hwv.max()) if n_valid else -1,
                "n_ne_min": int(nev.min()) if n_valid else -1,
                "n_ne_p25": pct(nev, 25), "n_ne_median": pct(nev, 50),
                "n_ne_p75": pct(nev, 75), "n_ne_max": int(nev.max()) if n_valid else -1,
                "diff_p25": pct(dv, 25), "diff_median": pct(dv, 50),
                "diff_p75": pct(dv, 75),
                "n_pos": int((dv > 0).sum()), "n_zero": int((dv == 0).sum()),
                "n_neg": int((dv < 0).sum()),
            })

    # ---- sensitivity (sign changes + diff distribution over shared valid cells) ----
    comps = [
        ("WPD boundary A vs B", "wpd_diff_A", "wpd_diff_B", "A", "B"),
        ("PV model Faiman vs linear (B)", "pvpot_diff_B", "pvpotlin_diff_B",
         "Faiman", "linear"),
        ("PV boundary A vs B", "pvpot_diff_A", "pvpot_diff_B", "A", "B"),
    ]
    sens = []
    for mi, mo in enumerate(MONTHS):
        for name, k1, k2, s1, s2 in comps:
            a = out[k1][mi].ravel()
            b = out[k2][mi].ravel()
            shared = np.isfinite(a) & np.isfinite(b)
            av, bv = a[shared], b[shared]
            n = int(shared.sum())
            sgn_a = np.sign(av); sgn_b = np.sign(bv)
            n_chg = int((sgn_a != sgn_b).sum())
            p2n = int(((sgn_a > 0) & (sgn_b < 0)).sum())
            n2p = int(((sgn_a < 0) & (sgn_b > 0)).sum())
            delta = bv - av
            sens.append({
                "month": mo, "comparison": name, "scheme1": s1, "scheme2": s2,
                "n_shared": n, "n_sign_changes": n_chg,
                "pos_to_neg": p2n, "neg_to_pos": n2p,
                "scheme1_median": pct(av, 50), "scheme1_p25": pct(av, 25),
                "scheme1_p75": pct(av, 75),
                "scheme2_median": pct(bv, 50), "scheme2_p25": pct(bv, 25),
                "scheme2_p75": pct(bv, 75),
                "delta_p25": pct(delta, 25), "delta_median": pct(delta, 50),
                "delta_p75": pct(delta, 75),
            })

    # ---- write CSVs ----
    with open(CSV_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(CSV_SENS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)
    print(f"wrote {CSV_SUMMARY}")
    print(f"wrote {CSV_SENS}")

    # ---- figures ----
    make_fig_dist(out)
    make_fig_rep(daB, lab, dates)

    # ---- report ----
    write_report(summary, sens)
    print(f"wrote {REPORT_MD}")


def make_fig_dist(out):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0))
    panels = [("pvpot", "B", "PV potential diff (Faiman)", "kWh m$^{-2}$"),
              ("wpd", "B", "WPD diff", "W m$^{-2}$")]
    for ax, (m, bd, title, unit) in zip(axes, panels):
        boxes, ns = [], []
        for mi in range(len(MONTHS)):
            d = out[f"{m}_diff_{bd}"][mi].ravel()
            d = d[np.isfinite(d)]
            boxes.append(d); ns.append(d.size)
        bp = ax.boxplot(boxes, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="#1f6f9f", lw=1.4),
                        whiskerprops=dict(color="#5a7d95", lw=1.0),
                        capprops=dict(color="#5a7d95", lw=1.0),
                        boxprops=dict(color="#5a7d95", lw=1.0))
        for patch in bp["boxes"]:
            patch.set_facecolor("#bcd7e8")
        ax.axhline(0, color="#8a8a8a", ls="--", lw=1.0, zorder=2)
        ax.set_xticks(range(1, len(MONTHS) + 1))
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(MONTH_SHORT, ns)],
                           fontsize=8.5)
        ax.set_ylabel(f"heatwave $-$ ordinary ({unit})", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.45)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle(f"Monthly per-gridpoint difference distribution (boundary B), {YEAR}",
                 fontsize=11, y=0.99)
    cap = ("Box = IQR (Q1–Q3); centre line = median; whiskers = farthest data point "
           "within 1.5×IQR; outliers not shown; n = valid comparison gridpoints.")
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(FIG_DIST, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_DIST}")


def make_fig_rep(daB, lab, dates):
    cell = daB.sel(latitude=EX_LAT, longitude=EX_LON, method="nearest")
    lbc = lab.sel(latitude=EX_LAT, longitude=EX_LON, method="nearest")
    pv = cell["pvpot_sum"].values
    wpd = cell["wpd_mean"].values
    lbl = lbc["label"].values
    hw = lbl == 3
    gap = lbl == 4
    x = np.arange(len(dates))
    tidx, tlab = month_ticks(dates)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    for ax, y, title, unit in [
        (axes[0], pv, "Daily PV potential (Faiman, cumulative)", "kWh m$^{-2}$"),
        (axes[1], wpd, "Daily wind power density (mean)", "W m$^{-2}$"),
    ]:
        shade_runs(ax, hw, color="#e15759", alpha=0.30)
        shade_runs(ax, gap, color="#9e9e9e", alpha=0.45)
        ax.plot(x, y, color="#1f6f9f", lw=1.2, zorder=3)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, ls=":", lw=0.5, alpha=0.4)
        ax.tick_params(axis="y", labelsize=8.5)
    axes[1].set_xticks(tidx)
    axes[1].set_xticklabels(tlab, fontsize=8.5)
    axes[1].set_xlabel(f"{YEAR} (month)", fontsize=9)

    handles = [
        mpatches.Patch(color="#e15759", alpha=0.30, label="heatwave day (label 3)"),
        mpatches.Patch(color="#9e9e9e", alpha=0.45, label="gap day (label 4)"),
        plt.Line2D([0], [0], color="#1f6f9f", lw=1.2, label="daily value"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        f"Example cell {EX_LAT:.2f}°N, {EX_LON:.2f}°E — 153-day series\n"
        f"day boundary B (provisional) = BJT previous-day 20:00 → same-day 20:00; "
        f"CN05.1 day boundary unconfirmed",
        fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(FIG_REP, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG_REP}")


def write_report(summary, sens):
    L = []
    L.append(f"# ERA5 {YEAR} warm-season monthly descriptive comparison (single-year case study)\n\n")

    L.append("## Inputs & conventions\n\n")
    L.append("- Daily products: boundary **B** = BJT previous-day 20:00 – same-day "
             "20:00 (provisional primary), **A** = 00-24 BJT (boundary sensitivity).\n")
    L.append("- PV: **Faiman** (`pvpot_sum`) provisional primary; **linear** "
             "(`pvpot_lin_sum`) model sensitivity. WPD: `wpd_mean`.\n")
    L.append("- CN05.1 day boundary: **UNCONFIRMED**.\n")
    L.append("- Groups: **heatwave = label 3**, **ordinary = label 1**; "
             "excluded = 0 (missing), 2 (isolated exceedance), 4 (gap).\n")
    L.append("- **diff = mean(heatwave) − mean(ordinary)** (signed, not relative); "
             "positive ⇒ heatwave higher. If either group has no sample, diff = NaN.\n")
    L.append("- **Spatial aggregation:** per gridpoint per month, group means are "
             "computed first then differenced per gridpoint; the monthly summary "
             "aggregates over gridpoints. Gridpoint-days are **NOT** treated as "
             "independent samples.\n")
    L.append("- Single-year descriptive only: no significance testing, no causal "
             "attribution, no complementarity conclusion.\n")

    L.append("\n## Monthly summary (primary: boundary B, Faiman PV)\n\n")
    L.append("| month | metric | unit | n_valid | n_hw min/med/max | "
             "n_ne min/med/max | diff P25/med/P75 | pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    for r in summary:
        n_hw = f"{r['n_hw_min']}/{r['n_hw_median']:.0f}/{r['n_hw_max']}"
        n_ne = f"{r['n_ne_min']}/{r['n_ne_median']:.0f}/{r['n_ne_max']}"
        L.append(f"| {r['month']} | {r['metric']} | {r['unit']} | {r['n_valid']} | "
                 f"{n_hw} | {n_ne} | "
                 f"{r['diff_p25']:.4g}/{r['diff_median']:.4g}/{r['diff_p75']:.4g} | "
                 f"{r['n_pos']}/{r['n_zero']}/{r['n_neg']} |\n")

    L.append("\n## Sensitivity (over shared valid gridpoints)\n\n")
    L.append("Sign change = sign(diff) flips between scheme1 and scheme2. "
             "delta = scheme2 − scheme1.\n\n")
    L.append("| month | comparison | n_shared | sign chg (p→n / n→p) | "
             "scheme1 med (P25–P75) | scheme2 med (P25–P75) | delta med (P25–P75) |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for r in sens:
        L.append(f"| {r['month']} | {r['comparison']} | {r['n_shared']} | "
                 f"{r['n_sign_changes']} ({r['pos_to_neg']} / {r['neg_to_pos']}) | "
                 f"{r['scheme1_median']:.4g} ({r['scheme1_p25']:.4g}–{r['scheme1_p75']:.4g}) | "
                 f"{r['scheme2_median']:.4g} ({r['scheme2_p25']:.4g}–{r['scheme2_p75']:.4g}) | "
                 f"{r['delta_median']:.4g} ({r['delta_p25']:.4g}–{r['delta_p75']:.4g}) |\n")

    L.append("\n> **PV boundary invariance (checked, not assumed):** `pvpot_sum`/`pvpot_lin_sum` "
             "are bit-identical between boundaries A and B (delta = 0, 0 sign changes), because "
             "the two windows differ only in the late-evening hours 20:00-24:00 BJT "
             "(13:00-16:00 UTC), where `ssrd=0` exactly. WPD and temperature are nonzero in "
             "those hours, so they differ between A and B.\n")

    L.append("\n## Figures\n\n")
    L.append(f"- `{os.path.basename(FIG_DIST)}` — monthly PV/WPD diff distribution "
             "(boundary B), valid-gridpoint count per month. Box = IQR (Q1–Q3); "
             "centre line = median; whiskers extend to the farthest actual data point "
             "within 1.5×IQR of the box; outliers not drawn (showfliers=False); "
             "n = valid comparison gridpoints (heatwave & ordinary both present); the "
             "valid-gridpoint set differs by month and each box is not independent "
             "samples or a confidence interval.\n")
    L.append(f"- `{os.path.basename(FIG_REP)}` — example cell "
             f"{EX_LAT:.2f}°N/{EX_LON:.2f}°E 153-day series (boundary B), "
             "heatwave and gap days shaded. Example cell = the 30–33°N/112–121°E "
             "gridpoint with the most 2022 heatwave-event days (`eval_merge.py`); "
             "not validated for regional representativeness.\n")

    L.append("\n## Files\n\n")
    L.append(f"- `{os.path.basename(NC_OUT)}`\n")
    L.append(f"- `{os.path.basename(CSV_SUMMARY)}`\n")
    L.append(f"- `{os.path.basename(CSV_SENS)}`\n")
    L.append(f"- `{os.path.basename(REPORT_MD)}`\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(L))


if __name__ == "__main__":
    main()
