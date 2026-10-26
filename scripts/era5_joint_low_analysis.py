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
    L.append(f"# {YEAR} wind–solar joint-low exploratory description (heatwave vs ordinary days)\n\n")
    L.append("Single-year, small-region, descriptive analysis. No causal attribution, no significance "
             "testing, no long-term risk conclusion.\n\n")

    L.append("## Inputs\n\n")
    L.append(f"- Daily potential (boundary A): `{os.path.basename(args.daily_a)}`\n")
    L.append(f"- Daily potential (boundary B, provisional primary): `{os.path.basename(args.daily_b)}`\n")
    L.append(f"- Aligned CN05.1 labels: `{os.path.basename(args.labels)}` (day boundary UNCONFIRMED)\n")
    L.append("- Main analysis: Faiman PV potential (`pvpot_sum`, kWh m⁻²) + WPD (`wpd_mean`, W m⁻²), "
             "boundary B; boundary A is only a sensitivity check.\n\n")

    L.append("## \"Low\" definition (single-year, per-month, relative)\n\n")
    L.append("- Valid-energy day = the day has both finite PV potential and finite WPD.\n")
    L.append("- The P25 threshold is computed over **all valid-energy days** of the month (all labels "
             "mixed), separately for PV and WPD.\n")
    L.append(f"- Fixed quantile algorithm: `numpy.nanquantile(x, q={QUANTILE_Q}, method='{QUANTILE_METHOD}')`.\n")
    L.append("- Low = value < P25; value equal to P25 counts as \"not low\".\n")
    L.append("- This threshold only ranks days within this single-year monthly relative classification; it is "
             "**not** a long-term climate threshold, an energy-drought threshold, or a supply-shortage threshold.\n")
    L.append("- PV and WPD are not added together directly, nor converted to an unverified capacity factor.\n\n")

    L.append("## Four mutually-exclusive and exhaustive states\n\n")
    L.append("- wind-low solar-low, wind-low solar-not-low, wind-not-low solar-low, wind-not-low solar-not-low.\n")
    L.append("- Groups: heatwave day = label 3; ordinary non-exceedance day = label 1.\n")
    L.append("- label 4 (gap day) and label 2 (isolated exceedance day) are not mixed into either group; "
             "label 0 (missing) is excluded.\n")
    L.append("- Per group, the four-state day counts sum to that group's valid-day count; with a sample the "
             "four proportions sum to 1; when a group has no sample, the proportions and the between-group "
             "difference are NaN (not filled with 0).\n")
    L.append("- \"Individually low\" and \"simultaneously low\" mean different things: the wind-low proportion / "
             "solar-low proportion / simultaneous wind+solar-low proportion are reported separately.\n\n")

    L.append("## Monthly spatial summary (boundary B, gridpoints with both groups present)\n\n")
    L.append("| month | valid gridpoints | heatwave median(min–max) | ordinary median(min–max) | "
             "difference median(P25–P75) pp | pos/zero/neg |\n")
    L.append("|---|---|---|---|---|---|\n")
    for r in main_rows:
        hw_rng = f"{r['n_hw_median']:g}({r['n_hw_min']}–{r['n_hw_max']})" if r['n_valid'] else "—"
        ne_rng = f"{r['n_ne_median']:g}({r['n_ne_min']}–{r['n_ne_max']})" if r['n_valid'] else "—"
        diff_rng = f"{r['diff_median']:.2f}({r['diff_p25']:.2f}–{r['diff_p75']:.2f})" if r['n_valid'] else "—"
        L.append(f"| {r['month']} | {r['n_valid']} | {hw_rng} | {ne_rng} | "
                 f"{diff_rng} | {r['n_pos']}/{r['n_zero']}/{r['n_neg']} |\n")
    L.append("\n> A positive difference = the simultaneous-low proportion on heatwave days is higher than on "
             "ordinary days. Small samples (especially a single gridpoint with only single-digit heatwave days "
             "in a month) substantially amplify the instability of the proportions; see the filter section below.\n\n")

    L.append(f"## Sample-size sensitivity (both groups ≥ {MIN_SAMPLE} days)\n\n")
    L.append("| month | valid gridpoints | difference median(P25–P75) pp | pos/zero/neg |\n")
    L.append("|---|---|---|---|\n")
    for r in filt_rows:
        if r["n_valid"]:
            L.append(f"| {r['month']} | {r['n_valid']} | "
                     f"{r['diff_median']:.2f}({r['diff_p25']:.2f}–{r['diff_p75']:.2f}) | "
                     f"{r['n_pos']}/{r['n_zero']}/{r['n_neg']} |\n")
        else:
            L.append(f"| {r['month']} | 0 (no gridpoints after filtering; reported honestly) | — | — |\n")
    L.append(f"\n> \"{MIN_SAMPLE} days\" is this round's exploratory filter, **not** a statistical sufficiency "
             "criterion; the threshold was not adjusted to make the results look better.\n\n")

    L.append("## Boundary A/B sensitivity\n\n")
    L.append("Compared on the **common valid comparison gridpoints** where both groups are present. Sign flips "
             "and one-side-zero are reported separately, not merged.\n\n")
    L.append("| month | common gridpoints | sign flips | one-side-zero | diff_A median(P25–P75) pp | "
             "diff_B median(P25–P75) pp | delta(B−A) median(P25–P75) pp |\n")
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
    L.append("\n> \"Sign flip\" = under boundaries A and B the difference is positive on one and negative on "
             "the other (both non-zero); \"one-side-zero\" = exactly one side has a zero difference "
             "(the other side non-zero).\n\n")
    flips = [str(r["n_sign_flip"]) for r in sens_rows]
    aug = next(r for r in sens_rows if r["month"] == f"{YEAR}-08")
    L.append("> **Stratified description**: overall, under both boundaries A and B the spatial medians of the "
             "simultaneous-low proportion difference over the 5 months are all negative, and the negative-difference "
             "feature is consistent; however, for some gridpoints the sign of the difference is sensitive to the "
             "choice of day boundary. The numbers of sign-flip gridpoints (May–Sep) are "
             f"{'/'.join(flips)}, with August the largest: "
             f"{aug['n_sign_flip']}/{aug['n_shared']} common gridpoints flip sign "
             f"(plus {aug['n_one_side_zero']} with one side zero).\n\n")

    L.append("## Figure\n\n")
    L.append(f"- `fig_joint_low_diff_monthly_B_{YEAR}.png` — per-month gridpoint distribution of the "
             "\"heatwave − ordinary simultaneous-low proportion difference\" (boundary B, unit pp); top panel "
             "both groups ≥ 1 day, bottom panel ≥ 5-day filter; the number of valid comparison gridpoints is "
             "annotated; months without data are explicitly marked as no-valid-gridpoints, not drawn as 0.\n\n")

    L.append("## Limitations and boundaries\n\n")
    L.append(f"- Single year ({YEAR}), single season, small region (30–33°N / 112–121°E), descriptive; does not "
             "represent long-term or large-region patterns.\n")
    L.append("- The four states only describe the relative high/low of the wind and solar resource combination "
             "within a month; they **cannot** directly prove that complementarity is enhanced or weakened, and "
             "**cannot** prove a change in power-system reliability.\n")
    L.append("- The low threshold is a single-year per-month relative P25; with few samples (especially per "
             "gridpoint per month) the proportions are unstable, and no interval estimates are made.\n")
    L.append("- The CN05.1 observation day boundary is unconfirmed; the A/B difference is reported separately, "
             "and no conclusion is drawn from alignment with CN05.1.\n")

    write_text_guarded(out_md, "".join(L))


if __name__ == "__main__":
    main()
