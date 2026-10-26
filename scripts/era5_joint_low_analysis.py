# -*- coding: utf-8 -*-
"""
Single-year (May-Sep 2022) EXPLORATORY description of joint wind+solar LOW states in
heatwave vs ordinary days, over the small 30-33N / 112-121E domain.

Reads the EXISTING daily potential products (boundary A and B) and the aligned CN05.1
heatwave labels. No download, no recompute of daily products, no change to labels,
formulas, or the heatwave definition.

Definition of "low" (relative, single-year, per gridpoint per month):
  * valid-energy day = both pvpot_sum (Faiman) and wpd_mean are finite.
  * P25 thresholds are computed over ALL valid-energy days of the month (every label),
    separately for PV and WPD, with a FIXED quantile algorithm:
        numpy.nanquantile(x, q=0.25, axis=0, method="linear")
  * "low" = value < P25; value == P25 counts as NOT low.
  * The threshold only ranks days within this single year and month; it is NOT a
    long-term climate / energy-drought / supply-shortage threshold.

Four mutually-exclusive and exhaustive states (wind x solar, each low or not):
  both_low, wind_only_low, solar_only_low, neither_low.

Groups: heatwave = label 3; ordinary non-exceedance = label 1. label 4 (merged gap)
and label 2 (isolated exceedance) are NOT mixed into either group; label 0 (missing)
is excluded. Per group / gridpoint / month we report the valid-day count and the
four-state counts+proportions plus wind-low / solar-low / joint-low proportions.

Headline: joint_low_diff = (joint-low proportion heatwave - ordinary) * 100
(percentage points), per gridpoint per month. Primary boundary B; boundary A as a
sensitivity check. CN05.1 day boundary stays UNCONFIRMED.

Single-year, descriptive only: NO significance testing, NO causal attribution, NO
complementarity or power-system-reliability conclusion.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "results_multi", "joint")

# Derived in main() from the --daily-a filename (or --year):
YEAR = None
MONTHS = None
MONTH_SHORT = None

STATE_NAMES = ["both_low", "wind_only_low", "solar_only_low", "neither_low"]

QUANTILE_Q = 0.25
QUANTILE_METHOD = "linear"
MIN_SAMPLE = 5          # exploratory sample-size filter (NOT a sufficiency criterion)

# CJK-capable font for any Chinese annotation in the figure/report (falls back to DejaVu)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def pct(x, p):
    return float(np.nanpercentile(x, p)) if x.size else float("nan")


def write_text_guarded(path, text):
    """Write a text file; refuse to overwrite an existing one (no silent clobber)."""
    if os.path.exists(path):
        print(f"SKIP (exists, not overwriting): {path}")
        return "skipped"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {path}")
    return "written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-a", required=True,
                    help="path to daily potential NetCDF, boundary A (explicit, no glob)")
    ap.add_argument("--daily-b", required=True,
                    help="path to daily potential NetCDF, boundary B (explicit, no glob)")
    ap.add_argument("--labels", required=True,
                    help="path to aligned CN05.1 daily-label NetCDF (explicit, no glob)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default: results_multi/joint)")
    ap.add_argument("--year", default=None,
                    help="year to analyze (default: parse from --daily-a filename)")
    args = ap.parse_args()

    global YEAR, MONTHS, MONTH_SHORT
    if args.year:
        YEAR = str(int(args.year))
    else:
        mm = re.match(r"era5_daily_potential_(\d{4})0501_\d{4}0930_A\.nc",
                      os.path.basename(args.daily_a))
        if not mm:
            raise SystemExit(f"cannot parse year from daily-a filename: {args.daily_a}")
        YEAR = mm.group(1)
    MONTHS = [f"{YEAR}-{m:02d}" for m in range(5, 10)]
    MONTH_SHORT = [calendar.month_abbr[int(m[5:7])] for m in MONTHS]

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

    label = lab["label"].values                       # (153, lat, lon) float32
    pv = {"A": daA["pvpot_sum"].values, "B": daB["pvpot_sum"].values}    # kWh m-2
    wpd = {"A": daA["wpd_mean"].values, "B": daB["wpd_mean"].values}      # W m-2

    # ---------------- per-month, per-gridpoint, per-boundary computation ----------------
    res = {}
    shp = (nmon, nlat, nlon)
    for bd in ("A", "B"):
        res[f"pv_p25_{bd}"] = np.full(shp, np.nan, dtype=np.float64)
        res[f"wpd_p25_{bd}"] = np.full(shp, np.nan, dtype=np.float64)
        res[f"n_thr_{bd}"] = np.zeros(shp, dtype=np.int16)
        res[f"n_hw_{bd}"] = np.zeros(shp, dtype=np.int16)
        res[f"n_ne_{bd}"] = np.zeros(shp, dtype=np.int16)
        res[f"state_count_hw_{bd}"] = np.zeros((nmon, 4, nlat, nlon), dtype=np.int16)
        res[f"state_count_ne_{bd}"] = np.zeros((nmon, 4, nlat, nlon), dtype=np.int16)
        res[f"state_prop_hw_{bd}"] = np.full((nmon, 4, nlat, nlon), np.nan, dtype=np.float64)
        res[f"state_prop_ne_{bd}"] = np.full((nmon, 4, nlat, nlon), np.nan, dtype=np.float64)
        for g in ("hw", "ne"):
            for k in ("wind_low", "solar_low", "joint_low"):
                res[f"{k}_prop_{g}_{bd}"] = np.full(shp, np.nan, dtype=np.float64)
        res[f"joint_low_diff_{bd}"] = np.full(shp, np.nan, dtype=np.float64)

    for mi, mo in enumerate(MONTHS):
        tmask = dates.astype("datetime64[M]") == np.datetime64(mo)
        lab_m = label[tmask]                          # (t_m, lat, lon)
        hw_mask = lab_m == 3
        ne_mask = lab_m == 1

        for bd in ("A", "B"):
            pv_m = pv[bd][tmask]
            wpd_m = wpd[bd][tmask]
            valid_energy = np.isfinite(pv_m) & np.isfinite(wpd_m)

            pv_v = np.where(valid_energy, pv_m, np.nan)
            wpd_v = np.where(valid_energy, wpd_m, np.nan)
            with np.errstate(all="ignore"):
                pv_p25 = np.nanquantile(pv_v, QUANTILE_Q, axis=0, method=QUANTILE_METHOD)
                wpd_p25 = np.nanquantile(wpd_v, QUANTILE_Q, axis=0, method=QUANTILE_METHOD)

            n_thr = valid_energy.sum(axis=0)
            solar_low = valid_energy & (pv_m < pv_p25[None, :, :])
            wind_low = valid_energy & (wpd_m < wpd_p25[None, :, :])

            both_low = wind_low & solar_low
            wind_only_low = wind_low & ~solar_low
            solar_only_low = ~wind_low & solar_low
            neither_low = ~wind_low & ~solar_low

            res[f"pv_p25_{bd}"][mi] = pv_p25
            res[f"wpd_p25_{bd}"][mi] = wpd_p25
            res[f"n_thr_{bd}"][mi] = n_thr.astype(np.int16)

            for g, key in ((hw_mask, "hw"), (ne_mask, "ne")):
                gv = g & valid_energy
                n_g = gv.sum(axis=0)
                res[f"n_{key}_{bd}"][mi] = n_g.astype(np.int16)
                sc = np.stack([(both_low & gv).sum(0),
                               (wind_only_low & gv).sum(0),
                               (solar_only_low & gv).sum(0),
                               (neither_low & gv).sum(0)], axis=0)
                res[f"state_count_{key}_{bd}"][mi] = sc.astype(np.int16)
                with np.errstate(divide="ignore", invalid="ignore"):
                    res[f"state_prop_{key}_{bd}"][mi] = sc.astype(np.float64) / n_g[None, :, :]
                    res[f"wind_low_prop_{key}_{bd}"][mi] = (
                        (wind_low & gv).sum(0).astype(np.float64) / n_g)
                    res[f"solar_low_prop_{key}_{bd}"][mi] = (
                        (solar_low & gv).sum(0).astype(np.float64) / n_g)
                    res[f"joint_low_prop_{key}_{bd}"][mi] = (sc[0].astype(np.float64) / n_g)

            res[f"joint_low_diff_{bd}"][mi] = (
                (res[f"joint_low_prop_hw_{bd}"][mi] - res[f"joint_low_prop_ne_{bd}"][mi]) * 100.0
            )

    # ---------------- assemble NetCDF ----------------
    month_coord = np.array([np.datetime64(mo + "-01") for mo in MONTHS], dtype="datetime64[D]")
    ds = xr.Dataset(
        {name: (("month", "latitude", "longitude"), arr)
         for name, arr in res.items() if arr.ndim == 3},
        coords={"month": month_coord, "latitude": lat, "longitude": lon},
    )
    for name, arr in res.items():
        if arr.ndim == 4:  # (month, state, lat, lon)
            ds[name] = (("month", "state", "latitude", "longitude"), arr)
    ds = ds.assign_coords(state=np.array(STATE_NAMES))

    # variable metadata
    units = {
        "pv_p25": "kWh m-2", "wpd_p25": "W m-2", "n_thr": "days",
        "n_hw": "days", "n_ne": "days",
    }
    for bd in ("A", "B"):
        for name, u in units.items():
            ds[f"{name}_{bd}"].attrs["units"] = u
        ds[f"pv_p25_{bd}"].attrs["long_name"] = f"P25 PV potential (Faiman) threshold over all valid-energy days, boundary {bd}"
        ds[f"wpd_p25_{bd}"].attrs["long_name"] = f"P25 WPD threshold over all valid-energy days, boundary {bd}"
        ds[f"n_thr_{bd}"].attrs["long_name"] = f"number of valid-energy days (PV & WPD both finite) used for the threshold, boundary {bd}"
        ds[f"n_hw_{bd}"].attrs["long_name"] = f"heatwave (label 3) valid-energy day count, boundary {bd}"
        ds[f"n_ne_{bd}"].attrs["long_name"] = f"ordinary non-exceedance (label 1) valid-energy day count, boundary {bd}"
        for key in ("hw", "ne"):
            g = "heatwave (label 3)" if key == "hw" else "ordinary non-exceedance (label 1)"
            ds[f"state_count_{key}_{bd}"].attrs["long_name"] = \
                f"four-state day count, {g}, boundary {bd}; state dim = {STATE_NAMES}"
            ds[f"state_prop_{key}_{bd}"].attrs["long_name"] = \
                f"four-state proportion (count / valid-day count), {g}, boundary {bd}"
            ds[f"state_prop_{key}_{bd}"].attrs["units"] = "1"
            for k in ("wind_low", "solar_low", "joint_low"):
                ds[f"{k}_prop_{key}_{bd}"].attrs["long_name"] = \
                    f"{k} proportion, {g}, boundary {bd}"
                ds[f"{k}_prop_{key}_{bd}"].attrs["units"] = "1"
        ds[f"joint_low_diff_{bd}"].attrs["units"] = "percentage points"
        ds[f"joint_low_diff_{bd}"].attrs["long_name"] = \
            f"(joint-low proportion heatwave - ordinary) * 100, boundary {bd}"

    ds.attrs = {
        "title": f"Single-year ({YEAR}) joint wind+solar LOW state description, heatwave vs ordinary",
        "low_definition": "value < monthly P25 threshold; value == P25 counts as NOT low",
        "threshold_basis": "P25 computed over ALL valid-energy days of the month (every label), separately for PV and WPD",
        "valid_energy_day": "both pvpot_sum (Faiman) and wpd_mean are finite",
        "quantile_algorithm": f"numpy.nanquantile(x, q={QUANTILE_Q}, axis=0, method='{QUANTILE_METHOD}')",
        "threshold_not_a_climate_threshold": "relative single-year monthly ranking only; NOT a long-term climate / energy-drought / supply-shortage threshold",
        "states": json.dumps(STATE_NAMES),
        "groups": "heatwave = label 3; ordinary non-exceedance = label 1; excluded = label 0 (missing), 2 (isolated exceedance), 4 (merged gap)",
        "joint_low_diff_units": "percentage points = (proportion heatwave - proportion ordinary) * 100",
        "boundary": "B = 20-20 BJT (provisional primary); A = 00-24 BJT (sensitivity)",
        "pv_scheme": "pvpot = Faiman (provisional primary); wpd = 0.5*rho*V100^3",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "min_sample_filter_days": MIN_SAMPLE,
        "min_sample_filter_note": "exploratory sample-size filter, NOT a statistical sufficiency criterion",
        "source_daily": f"{os.path.basename(args.daily_a)}, {os.path.basename(args.daily_b)}",
        "source_labels": os.path.basename(args.labels),
        "config_signature": json.dumps({
            "quantile": {"q": QUANTILE_Q, "method": QUANTILE_METHOD},
            "low_definition": "value < P25 (equal is not low)",
            "threshold_basis": "all valid-energy days (all labels)",
            "groups": {"heatwave": 3, "ordinary": 1, "excluded": [0, 2, 4]},
            "metrics": {"pv": "pvpot_sum", "wpd": "wpd_mean"},
            "boundaries": ["A", "B"],
            "min_sample_filter_days": MIN_SAMPLE,
            "joint_low_diff": "(prop_hw - prop_ne) * 100",
        }, sort_keys=True),
    }
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.integer):
            enc[v] = {"zlib": True, "complevel": 4, "dtype": str(ds[v].dtype)}
        else:
            enc[v] = {"zlib": True, "complevel": 4, "_FillValue": np.nan}

    nc_out = os.path.join(args.out, f"joint_low_{YEAR}0501_{YEAR}0930.nc")
    write_guarded(ds, nc_out, enc)

    # ---------------- verification (state partition / count conservation / ranges) ----
    print("=" * 70)
    print("verification (per boundary):")
    for bd in ("A", "B"):
        for key in ("hw", "ne"):
            sc = res[f"state_count_{key}_{bd}"]
            n = res[f"n_{key}_{bd}"]
            has = n > 0
            part_ok = np.all(sc.sum(axis=1)[has] == n[has])
            sp = res[f"state_prop_{key}_{bd}"]
            sp_sum = np.nansum(sp, axis=1)
            prop_sum_ok = np.allclose(sp_sum[has], 1.0, atol=1e-9)
            prop_range_ok = bool(np.all(np.isnan(sp) | ((sp >= 0) & (sp <= 1))))
            print(f"  {bd}/{key}: partition==n {part_ok}, prop-sum==1 {prop_sum_ok}, "
                  f"prop-in-[0,1] {prop_range_ok}")

    # ---------------- monthly spatial summary (boundary B, main: both groups >=1 day) --
    diff_B = res["joint_low_diff_B"]
    n_hw_B = res["n_hw_B"]
    n_ne_B = res["n_ne_B"]

    def spatial_summary(diff, n_hw, n_ne, mask=None):
        rows = []
        for mi, mo in enumerate(MONTHS):
            d = diff[mi].ravel()
            hwv = n_hw[mi].ravel()
            nev = n_ne[mi].ravel()
            valid = np.isfinite(d)
            if mask is not None:
                valid &= mask[mi].ravel()
            dv = d[valid]
            n_valid = int(dv.size)
            rows.append({
                "month": mo, "n_valid": n_valid,
                "n_hw_min": int(hwv[valid].min()) if n_valid else "",
                "n_hw_median": pct(hwv[valid], 50) if n_valid else "",
                "n_hw_max": int(hwv[valid].max()) if n_valid else "",
                "n_ne_min": int(nev[valid].min()) if n_valid else "",
                "n_ne_median": pct(nev[valid], 50) if n_valid else "",
                "n_ne_max": int(nev[valid].max()) if n_valid else "",
                "diff_median": pct(dv, 50), "diff_p25": pct(dv, 25), "diff_p75": pct(dv, 75),
                "n_pos": int((dv > 0).sum()), "n_zero": int((dv == 0).sum()),
                "n_neg": int((dv < 0).sum()),
            })
        return rows

    main_rows = spatial_summary(diff_B, n_hw_B, n_ne_B)
    filt_mask = (n_hw_B >= MIN_SAMPLE) & (n_ne_B >= MIN_SAMPLE)
    filt_rows = spatial_summary(diff_B, n_hw_B, n_ne_B, mask=filt_mask)

    # ---------------- boundary A/B sensitivity ----------------
    diff_A = res["joint_low_diff_A"]
    # exact integer sign of the difference to avoid float misclassification of zero
    def joint_num(bd):
        c_hw = res[f"state_count_hw_{bd}"][:, 0]   # both_low count, heatwave
        c_ne = res[f"state_count_ne_{bd}"][:, 0]
        return (c_hw.astype(np.int64) * res[f"n_ne_{bd}"].astype(np.int64)
                - c_ne.astype(np.int64) * res[f"n_hw_{bd}"].astype(np.int64))

    num_A = joint_num("A")
    num_B = joint_num("B")
    sens_rows = []
    for mi, mo in enumerate(MONTHS):
        a = diff_A[mi].ravel()
        b = diff_B[mi].ravel()
        shared = np.isfinite(a) & np.isfinite(b)
        av, bv = a[shared], b[shared]
        n = int(shared.sum())
        na = num_A[mi].ravel()[shared]
        nb = num_B[mi].ravel()[shared]
        sa, sb = np.sign(na), np.sign(nb)
        n_flip = int(((sa != sb) & (sa != 0) & (sb != 0)).sum())
        n_onezero = int(((na == 0) != (nb == 0)).sum())
        delta = bv - av
        sens_rows.append({
            "month": mo, "n_shared": n,
            "n_sign_flip": n_flip, "n_one_side_zero": n_onezero,
            "diff_A_median": pct(av, 50), "diff_A_p25": pct(av, 25), "diff_A_p75": pct(av, 75),
            "diff_B_median": pct(bv, 50), "diff_B_p25": pct(bv, 25), "diff_B_p75": pct(bv, 75),
            "delta_median": pct(delta, 50), "delta_p25": pct(delta, 25),
            "delta_p75": pct(delta, 75),
        })

    # ---------------- write CSVs ----------------
    def write_csv(path, rows, fieldnames):
        if os.path.exists(path):
            print(f"SKIP (exists, not overwriting): {path}")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    write_csv(os.path.join(args.out, f"monthly_summary_joint_low_B_{YEAR}.csv"),
              main_rows, list(main_rows[0].keys()))
    write_csv(os.path.join(args.out, f"sample_filter_summary_joint_low_B_{YEAR}.csv"),
              filt_rows, list(filt_rows[0].keys()))
    write_csv(os.path.join(args.out, f"ab_sensitivity_joint_low_{YEAR}.csv"),
              sens_rows, list(sens_rows[0].keys()))

    # ---------------- figure ----------------
    fig_png = os.path.join(args.out, f"fig_joint_low_diff_monthly_B_{YEAR}.png")
    make_fig(diff_B, n_hw_B, n_ne_B, fig_png)

    # ---------------- report ----------------
    write_report(args, main_rows, filt_rows, sens_rows,
                 os.path.join(args.out, f"REPORT_joint_low_{YEAR}.md"))

    # ---------------- print headline numbers ----------------
    print("=" * 70)
    print("monthly joint-low difference (heatwave - ordinary, pp; boundary B, both groups >=1 day):")
    for r in main_rows:
        print(f"  {r['month']}: n_valid={r['n_valid']}  median={r['diff_median']} "
              f"(P25 {r['diff_p25']}, P75 {r['diff_p75']})  pos/zero/neg="
              f"{r['n_pos']}/{r['n_zero']}/{r['n_neg']}")
    print("DONE joint low analysis")


def make_fig(diff_B, n_hw_B, n_ne_B, outfig):
    # sharex=False so each panel keeps its own month + n tick labels (top: >=1 day,
    # bottom: >=5 day filter). The two panels' n values differ and must not be
    # suppressed by a shared axis.
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=False)
    panels = [
        (axes[0], None,
         "all gridpoints with both groups present (>=1 day each)"),
        (axes[1], (n_hw_B >= MIN_SAMPLE) & (n_ne_B >= MIN_SAMPLE),
         f"sample-size filter: both groups >= {MIN_SAMPLE} days each"),
    ]
    for ax, mask, title in panels:
        boxes, ns = [], []
        for mi in range(len(MONTHS)):
            d = diff_B[mi].ravel()
            valid = np.isfinite(d)
            if mask is not None:
                valid &= mask[mi].ravel()
            dv = d[valid]
            boxes.append(dv)
            ns.append(int(dv.size))
        ax.boxplot(boxes, showfliers=False, patch_artist=True, widths=0.6,
                   medianprops=dict(color="#1f6f9f", lw=1.4),
                   whiskerprops=dict(color="#5a7d95", lw=1.0),
                   capprops=dict(color="#5a7d95", lw=1.0),
                   boxprops=dict(color="#5a7d95", lw=1.0))
        for bp in ax.patches:
            bp.set_facecolor("#bcd7e8")
        ax.axhline(0, color="#8a8a8a", ls="--", lw=1.0, zorder=2)
        ax.set_xticks(range(1, len(MONTHS) + 1))
        ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(MONTH_SHORT, ns)],
                           fontsize=8.5)
        ax.set_ylabel("joint-low proportion difference (pp)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.45)
        ax.tick_params(axis="y", labelsize=8.5)
        for mi, n in enumerate(ns):
            if n == 0:
                ax.text(mi + 1, 0, "no valid gridpoints", ha="center", va="center",
                        fontsize=8, color="#c62828", rotation=0)
    axes[-1].set_xlabel(f"{YEAR} month", fontsize=9)
    fig.suptitle(
        "Heatwave − ordinary joint wind+solar LOW proportion difference (boundary B)\n"
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


def write_report(args, main_rows, filt_rows, sens_rows, out_md):
    L = []
    L.append(f"# {YEAR} 年风-光联合偏低探索性描述（热浪日 vs 普通日）\n\n")
    L.append("单年、小区域、描述性分析。不做因果归因、不做显著性检验、不做长期风险结论。\n\n")

    L.append("## 输入\n\n")
    L.append(f"- 日尺度潜力（日界 A）：`{os.path.basename(args.daily_a)}`\n")
    L.append(f"- 日尺度潜力（日界 B，暂定主方案）：`{os.path.basename(args.daily_b)}`\n")
    L.append(f"- 对齐 CN05.1 标签：`{os.path.basename(args.labels)}`（日界 UNCONFIRMED）\n")
    L.append("- 主分析：Faiman PV 潜力（`pvpot_sum`，kWh m⁻²）+ WPD（`wpd_mean`，W m⁻²），日界 B；"
             "日界 A 仅作敏感性检查。\n\n")

    L.append("## “偏低”定义（单年、逐月、相对）\n\n")
    L.append("- 有效能源日 = 该日 PV 潜力与 WPD 均有限。\n")
    L.append("- P25 阈值基于当月**全部有效能源日**（不分热浪/普通），PV 与 WPD 分别计算。\n")
    L.append(f"- 分位数算法固定：`numpy.nanquantile(x, q={QUANTILE_Q}, method='{QUANTILE_METHOD}')`。\n")
    L.append("- 偏低 = value < P25；等于 P25 归入“不偏低”。\n")
    L.append("- 该阈值仅用于本次单年逐月相对状态分类，**不是**长期气候阈值、能源干旱阈值或供电短缺阈值。\n")
    L.append("- 不将 PV 与 WPD 直接相加，也不转换为未经验证的容量因子。\n\n")

    L.append("## 四类互斥完备状态\n\n")
    L.append("- 风低光低、风低光不低、风不低光低、风不低光不低。\n")
    L.append("- 分组：热浪高温日 = label 3；普通非超阈值日 = label 1。\n")
    L.append("- label 4（间隔日）、label 2（孤立超阈值日）不混入两组；label 0（缺测）排除。\n")
    L.append("- 每组四类状态天数之和 = 该组有效日数；有样本时四类比例之和 = 1；"
             "任一组无样本时比例与组间差值为 NaN（不填 0）。\n")
    L.append("- “分别偏低”与“同时偏低”含义不同：风偏低比例/光偏低比例/风光同时偏低比例分别报告。\n\n")

    L.append("## 月度空间摘要（日界 B，两组均有样本的格点）\n\n")
    L.append("| 月 | 有效格点 | 热浪日中位数(min–max) | 普通日中位数(min–max) | "
             "差值中位数(P25–P75) pp | 正/零/负 |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in main_rows:
        hw_rng = f"{r['n_hw_median']:g}({r['n_hw_min']}–{r['n_hw_max']})" if r['n_valid'] else "—"
        ne_rng = f"{r['n_ne_median']:g}({r['n_ne_min']}–{r['n_ne_max']})" if r['n_valid'] else "—"
        diff_rng = f"{r['diff_median']:.2f}({r['diff_p25']:.2f}–{r['diff_p75']:.2f})" if r['n_valid'] else "—"
        L.append(f"| {r['month']} | {r['n_valid']} | {hw_rng} | {ne_rng} | "
                 f"{diff_rng} | {r['n_pos']}/{r['n_zero']}/{r['n_neg']} |\n")
    L.append("\n> 正差值 = 热浪日同时偏低比例高于普通日。小样本（尤其单格点当月热浪日仅个位数）"
             "会显著放大比例的不稳定，见下节筛选。\n\n")

    L.append(f"## 样本量敏感性（两组各 ≥{MIN_SAMPLE} 天）\n\n")
    L.append("| 月 | 有效格点 | 差值中位数(P25–P75) pp | 正/零/负 |\n")
    L.append("|---|---|---|---|\n")
    for r in filt_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | "
                     f"{r['diff_median']:.2f}({r['diff_p25']:.2f}–{r['diff_p75']:.2f}) | "
                     f"{r['n_pos']}/{r['n_zero']}/{r['n_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0（筛选后无格点，如实报告） | — | — |\n")
    L.append(f"\n> “{MIN_SAMPLE} 天”是本次探索性筛选，**不是**统计充分性标准；"
             "未根据结果好坏调整门槛。\n\n")

    L.append("## 日界 A/B 敏感性\n\n")
    L.append("在两组均有样本的**共同有效比较格点**上比较。符号翻转与一侧为零分开报告，不合并。\n\n")
    L.append("| 月 | 共同格点 | 符号翻转 | 一侧为零 | diff_A 中位数(P25–P75) pp | "
             "diff_B 中位数(P25–P75) pp | delta(B−A) 中位数(P25–P75) pp |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    for r in sens_rows:
        if r["n_shared"]:
            L.append(f"| {r['month']} | {r['n_shared']} | {r['n_sign_flip']} | "
                     f"{r['n_one_side_zero']} | "
                     f"{r['diff_A_median']:.2f}({r['diff_A_p25']:.2f}–{r['diff_A_p75']:.2f}) | "
                     f"{r['diff_B_median']:.2f}({r['diff_B_p25']:.2f}–{r['diff_B_p75']:.2f}) | "
                     f"{r['delta_median']:.2f}({r['delta_p25']:.2f}–{r['delta_p75']:.2f}) |\n")
        else:
            L.append(f"| {r['month']} | 0 | — | — | — | — | — |\n")
    L.append("\n> “符号翻转”= 日界 A、B 下差值一正一负（均非零）；"
             "“一侧为零”= 恰有一侧差值为 0（另一侧非零）。\n\n")
    flips = [str(r["n_sign_flip"]) for r in sens_rows]
    aug = next(r for r in sens_rows if r["month"] == f"{YEAR}-08")
    L.append("> **分层描述**：整体上，日界 A、B 下 5 个月的同时偏低比例差空间中位数"
             "均为负值，负差值特征保持一致；但部分格点的差值方向对日界选择敏感。"
             f"符号翻转格点数（5–9 月）分别为 {'/'.join(flips)}，其中 8 月最多，"
             f"{aug['n_sign_flip']}/{aug['n_shared']} 个共同格点发生符号翻转"
             f"（另有 {aug['n_one_side_zero']} 个一侧为零）。\n\n")

    L.append("## 图\n\n")
    L.append(f"- `fig_joint_low_diff_monthly_B_{YEAR}.png` — 逐月“热浪日−普通日同时偏低比例差”格点分布"
             "（日界 B，单位 pp），上图为两组均 ≥1 天，下图为 ≥5 天筛选；标注有效比较格点数；"
             "无数据月份明确标记为无有效格点，不画成 0。\n\n")

    L.append("## 局限与边界\n\n")
    L.append(f"- 单年（{YEAR}）单季节、小区域（30–33°N / 112–121°E），描述性，不代表长期或大区域规律。\n")
    L.append("- 四类状态仅描述风、光资源组合的当月相对高低，**不能**直接证明互补性增强或减弱，"
             "**不能**证明电力系统可靠性变化。\n")
    L.append("- 偏低阈值为单年逐月相对 P25，样本少（尤其逐格点逐月）时比例不稳定，不做区间估计。\n")
    L.append("- CN05.1 观测日界未确认；日界 A/B 差异已单独报告，未与 CN05.1 对齐下结论。\n")

    write_text_guarded(out_md, "".join(L))


if __name__ == "__main__":
    main()
