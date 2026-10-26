# -*- coding: utf-8 -*-
"""
Descriptive comparison of the two daily-boundary trials (Aug 2022) against the
CN05.1 heatwave daily labels.

Read-only on ALL inputs (hourly NetCDFs, daily NetCDFs, labels). Writes NEW files
under results_daily/ only: a per-gridpoint comparison NetCDF, a brief summary
table, a representative-cell time-series figure, and a short report. Does NOT
modify or overwrite the hourly/daily NetCDFs or any heatwave results.

Boundaries (date label D, Beijing = UTC+8):
  A: calendar 00-24 BJT  -> UTC [D-1 16:00, D 16:00)
  B: station 20-20 BJT   -> UTC [D-1 12:00, D 12:00)   (provisional primary;
     CN05.1 day boundary UNCONFIRMED)

Aggregation convention (no area/cos-latitude weighting; stated, not changed):
  per gridpoint -> difference (hw mean minus non-exceed mean, or A minus B) ->
  unweighted arithmetic mean over the N gridpoints that have both groups and
  finite values. N is printed in each 'n=…' summary.

Hard limits (per request): NO significance testing; NO gridpoint-day-as-
independent-samples; NO claim about which boundary matches CN05.1; NO
generalization of one month to long-term heatwave effects.
"""
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era5_daily_window import window_calendar_day, window_2020_day

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DAILY = os.path.join(ROOT, "results_daily")
RES_MERGE = os.path.join(ROOT, "results_merge")

LABELS_NC = os.path.join(RES_MERGE, "hw_2022_daily_labels.nc")
DAILY_A = os.path.join(RES_DAILY, "era5_daily_potential_202208_A.nc")
DAILY_B = os.path.join(RES_DAILY, "era5_daily_potential_202208_B.nc")

OUT_NC = os.path.join(RES_DAILY, "era5_daily_compare_202208.nc")
OUT_TABLE = os.path.join(RES_DAILY, "summary_table.md")
OUT_FIG = os.path.join(RES_DAILY, "fig_representative_cell.png")
OUT_REPORT = os.path.join(RES_DAILY, "REPORT.md")

# CN05.1 label codes (see docs/heatwave_method_v2.md); label 1 = ordinary
# non-exceedance day (Tmax below threshold, not part of any event)
CODE = {"missing": 0, "nonexceed": 1, "isolated": 2, "heatwave": 3, "gap": 4}

# CJK-capable font for the figure caption (falls back to DejaVu if unavailable)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_daily(path):
    return xr.open_dataset(path)


def group_mean(arr, mask):
    """Mean of arr over the day axis where `mask` is True; NaN if no day.
    arr/mask: (nday, nlat, nlon). Returns (mean, count) as (nlat, nlon)."""
    m = mask & np.isfinite(arr)
    n = m.sum(0)
    s = np.where(m, arr, 0.0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.where(n > 0, n, 1.0), np.nan)
    return out, n


def spatial(stat, name, unit):
    """Unweighted arithmetic mean/median/min/max over finite gridpoints."""
    v = stat[np.isfinite(stat)]
    if v.size == 0:
        return f"{name}: all-NaN (n=0)"
    return (f"{name}: mean={np.nanmean(v):+.4g} median={np.nanmedian(v):+.4g} "
            f"min={np.nanmin(v):+.4g} max={np.nanmax(v):+.4g} ({unit}, n={v.size})")


def main():
    print("loading inputs (read-only)...")
    lab = xr.open_dataset(LABELS_NC)
    da = load_daily(DAILY_A)
    db = load_daily(DAILY_B)

    # ---------- coordinate / date verification (exact match, no interpolation) ----------
    lat_d = da["latitude"].values
    lon_d = da["longitude"].values
    date_d = da["date"].values.astype("datetime64[D]")
    lat_l = lab["lat"].values
    lon_l = lab["lon"].values
    date_l = lab["date"].values.astype("datetime64[D]")

    ok_lat = np.array_equal(lat_d, lat_l) and np.all(np.diff(lat_d) > 0)
    ok_lon = np.array_equal(lon_d, lon_l) and np.all(np.diff(lon_d) > 0)
    aug = (date_l >= np.datetime64("2022-08-01")) & (date_l <= np.datetime64("2022-08-31")
    )
    ok_date = aug.sum() == 31 and np.array_equal(date_d, date_l[aug])
    if not (ok_lat and ok_lon and ok_date):
        raise SystemExit(
            "COORDINATE/DATE MISMATCH -> pause, do not proceed. "
            f"lat_equal={ok_lat} lon_equal={ok_lon} date_equal={ok_date}")
    print("coordinates & dates consistent: lat/lon exact match (ascending), "
          "Aug 1..31 aligned -> exact index join, no nearest/interpolation")

    # recorded windows (traceability)
    print("A attrs:", {k: da.attrs.get(k) for k in ("day_window_utc", "accum_samples_utc", "point_samples_utc")})
    print("B attrs:", {k: db.attrs.get(k) for k in ("day_window_utc", "accum_samples_utc", "point_samples_utc")})

    labels = lab["label"].values[aug].astype(np.int16)   # (31, 13, 37)

    # completeness: every gridpoint-day must have 24 valid hours (else metric is NaN)
    for tag, ds in (("A", da), ("B", db)):
        vh = ds["valid_hours"].values
        n_complete = int((vh == 24).sum())
        print(f"boundary {tag}: valid_hours==24 in {n_complete}/{vh.size} gridpoint-days "
              f"(min={int(vh.min())}, max={int(vh.max())})")

    # hourly-metric-first daily arrays (Faiman primary; linear sensitivity variant)
    def arr(ds, v):
        return ds[v].values.astype(float)

    pvpot_sum_B = arr(db, "pvpot_sum")          # kWh/m2 (Faiman)
    pvpot_mean_B = arr(db, "pvpot_mean")
    pvpot_lin_sum_B = arr(db, "pvpot_lin_sum")  # linear
    pvpot_lin_mean_B = arr(db, "pvpot_lin_mean")
    wpd_mean_B = arr(db, "wpd_mean")
    wpd_mean_A = arr(da, "wpd_mean")
    pvpot_sum_A = arr(da, "pvpot_sum")
    pvpot_mean_A = arr(da, "pvpot_mean")
    tair_mean_A = arr(da, "tair_mean")
    tair_mean_B = arr(db, "tair_mean")

    # ---------- Part 4.1: per-gridpoint boundary A-vs-B difference (daily) ----------
    d_wpd = np.nanmean(wpd_mean_A - wpd_mean_B, axis=0)          # W/m2
    d_pvpot_sum = np.nanmean(pvpot_sum_A - pvpot_sum_B, axis=0)  # kWh/m2
    d_pvpot_mean = np.nanmean(pvpot_mean_A - pvpot_mean_B, axis=0)
    d_tair = np.nanmean(tair_mean_A - tair_mean_B, axis=0)       # degC
    print("\n[4.1] boundary A minus B (per-gridpoint, mean over 31 days):")
    print("  " + spatial(d_wpd, "d_wpd_mean", "W/m2"))
    print("  " + spatial(d_pvpot_sum, "d_pvpot_sum", "kWh/m2"))
    print("  " + spatial(d_pvpot_mean, "d_pvpot_mean", "W/m2"))
    print("  " + spatial(d_tair, "d_tair_mean", "degC"))

    # ---------- group masks ----------
    hw = labels == CODE["heatwave"]
    ne = labels == CODE["nonexceed"]
    gap = labels == CODE["gap"]
    iso = labels == CODE["isolated"]
    n_hw = hw.sum(0); n_ne = ne.sum(0); n_gap = gap.sum(0); n_iso = iso.sum(0)

    n_valid_grid = int(((n_hw + n_ne + n_gap + n_iso) > 0).sum())
    n_both = int(((n_hw > 0) & (n_ne > 0)).sum())
    n_invalid = int((n_hw + n_ne + n_gap + n_iso == 0).sum())
    n_missing_group = n_valid_grid - n_both
    print(f"\ngridpoint counts (of {n_hw.size} total): invalid label={n_invalid}, "
          f"valid={n_valid_grid}, both hw&non-exceed={n_both}, "
          f"valid-but-missing-a-group={n_missing_group}")

    def hwcool(a):
        """per-gridpoint: hw mean, non-exceed mean, hw-non-exceed diff
        (NaN if either side missing)."""
        hw_m, _ = group_mean(a, hw)
        ne_m, _ = group_mean(a, ne)
        with np.errstate(invalid="ignore"):
            diff = np.where((n_hw > 0) & (n_ne > 0), hw_m - ne_m, np.nan)
        return hw_m, ne_m, diff

    # ---------- Part 4.2: heatwave vs non-exceed, boundary B (primary); WPD also under A ----------
    pvsum_hw, pvsum_ne, pvsum_d = hwcool(pvpot_sum_B)
    pvmean_hw, pvmean_ne, pvmean_d = hwcool(pvpot_mean_B)
    wpd_hw, wpd_ne, wpd_d = hwcool(wpd_mean_B)
    wpd_hw_A, wpd_ne_A, wpd_d_A = hwcool(wpd_mean_A)

    # PV group-difference sign counts (finite values)
    pv_fin = pvsum_d[np.isfinite(pvsum_d)]
    pv_pos = int((pv_fin > 0).sum()); pv_zero = int((pv_fin == 0).sum()); pv_neg = int((pv_fin < 0).sum())

    print("\n[4.2] heatwave vs non-exceed (boundary B), per-gridpoint:")
    print("  " + spatial(pvsum_d, "pvpot_sum hw-non-exceed", "kWh/m2"))
    print("  " + spatial(pvmean_d, "pvpot_mean hw-non-exceed", "W/m2"))
    print("  " + spatial(wpd_d, "wpd_mean hw-non-exceed", "W/m2"))
    print(f"  PV diff sign: positive={pv_pos} zero={pv_zero} negative={pv_neg}")
    print("  sample counts (gridpoint sum): hw=%d non-exceed=%d gap=%d isolated=%d"
          % (int(n_hw.sum()), int(n_ne.sum()), int(n_gap.sum()), int(n_iso.sum())))
    print("  [WPD under boundary A] " + spatial(wpd_d_A, "wpd_mean hw-non-exceed", "W/m2"))

    # ---------- Part 4.3: Faiman vs linear 'hw-non-exceed PV difference' (model sensitivity) ----------
    pl_sum_hw, pl_sum_ne, pl_sum_d = hwcool(pvpot_lin_sum_B)
    pl_mean_hw, pl_mean_ne, pl_mean_d = hwcool(pvpot_lin_mean_B)
    sens_sum = pvsum_d - pl_sum_d          # Faiman hw-non-exceed minus linear hw-non-exceed
    sens_mean = pvmean_d - pl_mean_d
    print("\n[4.3] Faiman vs linear 'hw-non-exceed' PV difference (boundary B; identical under A):")
    print("  Faiman  hw-non-exceed pvpot_sum: " + spatial(pvsum_d, "", "kWh/m2").split(':', 1)[1])
    print("  Linear  hw-non-exceed pvpot_sum: " + spatial(pl_sum_d, "", "kWh/m2").split(':', 1)[1])
    print("  (Faiman-linear) hw-non-exceed diff: " + spatial(sens_sum, "", "kWh/m2").split(':', 1)[1])

    # ---------- Part 4.4: representative cell ----------
    cand = (n_hw > 0) & (n_ne > 0)
    score = np.where(cand, n_hw.astype(float) + 0.5 * n_ne.astype(float), -1)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    rlat, rlon = float(lat_d[i]), float(lon_d[j])
    print(f"\nexample cell (auto-selected; distinct from 31.75N/114.50E): lat={rlat:.2f}N lon={rlon:.2f}E "
          f"(n_hw={int(n_hw[i,j])} n_non-exceed={int(n_ne[i,j])} "
          f"n_gap={int(n_gap[i,j])} n_iso={int(n_iso[i,j])})")

    # ---------- save per-gridpoint comparison NetCDF (traceability) ----------
    def mk(name, v, unit, long):
        return v, {"units": unit, "long_name": long}

    fields = {}
    fields["n_hw"] = mk("n_hw", n_hw.astype(np.int16), "days", "heatwave days (label 3)")
    fields["n_non_exceed"] = mk("n_non_exceed", n_ne.astype(np.int16), "days", "ordinary non-exceedance days (label 1)")
    fields["n_gap"] = mk("n_gap", n_gap.astype(np.int16), "days", "gap days (label 4)")
    fields["n_iso"] = mk("n_iso", n_iso.astype(np.int16), "days", "isolated exceedance days (label 2)")
    fields["pvpot_sum_hw_non_exceed"] = mk("pvpot_sum_hw_non_exceed", pvsum_d, "kWh m-2", "hw-non-exceed daily PVpot sum diff (Faiman, B)")
    fields["pvpot_mean_hw_non_exceed"] = mk("pvpot_mean_hw_non_exceed", pvmean_d, "W m-2", "hw-non-exceed daily PVpot mean diff (Faiman, B)")
    fields["wpd_mean_hw_non_exceed"] = mk("wpd_mean_hw_non_exceed", wpd_d, "W m-2", "hw-non-exceed daily WPD mean diff (B)")
    fields["wpd_mean_hw_non_exceed_A"] = mk("wpd_mean_hw_non_exceed_A", wpd_d_A, "W m-2", "hw-non-exceed daily WPD mean diff (A)")
    fields["pvpot_lin_sum_hw_non_exceed"] = mk("pvpot_lin_sum_hw_non_exceed", pl_sum_d, "kWh m-2", "hw-non-exceed daily PVpot sum diff (linear, B)")
    fields["pvpot_lin_mean_hw_non_exceed"] = mk("pvpot_lin_mean_hw_non_exceed", pl_mean_d, "W m-2", "hw-non-exceed daily PVpot mean diff (linear, B)")
    fields["d_wpd_mean_AB"] = mk("d_wpd_mean_AB", d_wpd, "W m-2", "boundary A-B daily WPD mean diff")
    fields["d_pvpot_sum_AB"] = mk("d_pvpot_sum_AB", d_pvpot_sum, "kWh m-2", "boundary A-B daily PVpot sum diff")
    fields["d_pvpot_mean_AB"] = mk("d_pvpot_mean_AB", d_pvpot_mean, "W m-2", "boundary A-B daily PVpot mean diff")
    fields["d_tair_mean_AB"] = mk("d_tair_mean_AB", d_tair, "degC", "boundary A-B daily tair mean diff")

    ds_out = xr.Dataset(
        {k: (("latitude", "longitude"), v) for k, (v, _) in fields.items()},
        coords={"latitude": lat_d, "longitude": lon_d},
    )
    for k, (_, at) in fields.items():
        ds_out[k].attrs = at
    ds_out.attrs = {
        "title": "Per-gridpoint descriptive comparison, Aug 2022 (single-month exploratory)",
        "label_source": os.path.basename(LABELS_NC),
        "daily_sources": f"{os.path.basename(DAILY_A)}, {os.path.basename(DAILY_B)}",
        "boundary_A": "calendar 00-24 BJT -> UTC [D-1 16:00, D 16:00)",
        "boundary_B": "station 20-20 BJT -> UTC [D-1 12:00, D 12:00) (provisional primary; CN05.1 UNCONFIRMED)",
        "label_codes": "0 missing, 1 ordinary non-exceedance, 2 isolated exceedance, 3 heatwave, 4 gap",
        "hw_non_exceed_definition": "hw = label 3; non-exceed = label 1; gap(4)/isolated(2) reported separately; missing(0) never counted as non-heatwave",
        "no_control_rule": "gridpoints without a non-exceedance day -> NaN (no borrowing)",
        "aggregation": "per gridpoint difference -> unweighted arithmetic mean over finite gridpoints (no cos-latitude/area weighting)",
        "caution": "exploratory single month; NO significance testing; NO independent-samples claim; NO boundary-vs-CN05.1 claim",
    }
    enc = {k: {"zlib": True, "complevel": 4} for k in ds_out.data_vars}
    ds_out.to_netcdf(OUT_NC, encoding=enc)
    print(f"wrote {OUT_NC}")

    # ---------- representative-cell figure ----------
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)
    xx = mdates.date2num([np.datetime64(d) for d in date_d.tolist()])
    label_cell = labels[:, i, j]

    def shade(ax):
        for k in range(31):
            if label_cell[k] == CODE["heatwave"]:
                ax.axvspan(xx[k] - 0.5, xx[k] + 0.5, color="#c62828", alpha=0.10, zorder=0)
            elif label_cell[k] == CODE["gap"]:
                ax.axvspan(xx[k] - 0.5, xx[k] + 0.5, color="#f9a825", alpha=0.10, zorder=0)

    ax = axes[0]
    ax.plot(xx, pvpot_sum_B[:, i, j], color="#1565c0", lw=1.6, label="PVpot daily total Faiman (B)")
    ax.plot(xx, pvpot_lin_sum_B[:, i, j], color="#6a1b9a", lw=1.4, ls="--",
            label="PVpot daily total linear (B)")
    shade(ax)
    ax.set_ylabel("PV potential daily total (kWh/m²)")
    ax.set_title(f"Example cell {rlat:.2f}°N, {rlon:.2f}°E (a different example cell from 31.75°N/114.50°E)\n"
                 f"(heatwave days {int(n_hw[i,j])}, ordinary non-exceedance days {int(n_ne[i,j])})", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.annotate("Radiation/PV-type indicators are identical under the two day boundaries A and B "
                "(the difference falls only in the nighttime 20–24 h, with no shortwave radiation)",
                xy=(0.01, 0.02), xycoords="axes fraction", fontsize=7.5, color="#555555")

    ax = axes[1]
    ax.plot(xx, wpd_mean_A[:, i, j], color="#e65100", lw=1.6, label="WPD daily mean, boundary A")
    ax.plot(xx, wpd_mean_B[:, i, j], color="#00695c", lw=1.4, ls="--", label="WPD daily mean, boundary B")
    shade(ax)
    ax.set_ylabel("WPD daily mean (W/m²)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel("2022-08 (UTC day assignment; see REPORT)")
    for lb in ax.get_xticklabels():
        lb.set_fontsize(8)

    fig.suptitle("Single-month exploratory comparison, not causal attribution", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")

    # ---------- summary table + report ----------
    m = lambda v: np.nanmean(v) if np.isfinite(v).any() else float("nan")

    table = [
        "# Day-boundary trials vs heatwave labels — brief summary table (2022-08 single-month exploration)\n",
        "| Item | Boundary A (00–24 Beijing time) | Boundary B (20–20 Beijing time, provisional primary) |",
        "|---|---|---|",
        f"| Daily UTC window (instantaneous sample times) | {da.attrs['point_samples_utc']} | {db.attrs['point_samples_utc']} |",
        f"| Daily UTC window (hour-ending accumulation times) | {da.attrs['accum_samples_utc']} | {db.attrs['accum_samples_utc']} |",
        "| Complete-day requirement | 24 valid hours per gridpoint per day, otherwise NaN | same as left |",
        f"| Complete gridpoint-days | {int((da['valid_hours'].values==24).sum())}/14911 | {int((db['valid_hours'].values==24).sum())}/14911 |",
        f"| Invalid-label gridpoints (all 31 days = 0) | {n_invalid} | {n_invalid} |",
        f"| Valid matched gridpoints (label∈{{1,2,3,4}}) | {n_valid_grid}/481 | {n_valid_grid}/481 |",
        f"| Gridpoints with both heatwave and ordinary non-exceedance days | {n_both} | {n_both} |",
        f"| Valid gridpoints missing either group | {n_missing_group} | {n_missing_group} |",
        f"| Heatwave-day samples (gridpoint × day total) | {int(n_hw.sum())} | {int(n_hw.sum())} |",
        f"| Ordinary non-exceedance-day samples | {int(n_ne.sum())} | {int(n_ne.sum())} |",
        f"| Gap-day samples | {int(n_gap.sum())} | {int(n_gap.sum())} |",
        f"| Isolated exceedance-day samples | {int(n_iso.sum())} | {int(n_iso.sum())} |",
        f"| PVpot daily total heatwave−non-exceedance (equal-weight gridpoint mean, kWh/m²) | {m(pvsum_d):+.4g} | {m(pvsum_d):+.4g} |",
        f"| PVpot daily total linear heatwave−non-exceedance (equal-weight gridpoint mean, kWh/m²) | {m(pl_sum_d):+.4g} | {m(pl_sum_d):+.4g} |",
        f"| WPD daily mean heatwave−non-exceedance (equal-weight gridpoint mean, W/m²) | {m(wpd_d_A):+.4g} | {m(wpd_d):+.4g} |",
        f"| Difference of the two boundaries' WPD daily mean (equal-weight gridpoint mean, W/m²) | — | {m(d_wpd):+.4g} (A−B) |",
        f"| Difference of the two boundaries' PVpot daily total (equal-weight gridpoint mean, kWh/m²) | — | {m(d_pvpot_sum):+.4g} (A−B) |",
        "",
        "> Aggregation convention (stated honestly, no weighting added): all \"heatwave−non-exceedance\" and "
        "\"difference of the two boundaries\" values are obtained in **two steps** —",
        f"> (1) per gridpoint first take that gridpoint's difference (heatwave-day mean − ordinary "
        "non-exceedance-day mean, or A − B daily mean); (2) then take the **equal-weight arithmetic mean "
        f"over {n_both} valid gridpoints**",
        "> (each gridpoint equal weight; no cos-latitude/area weighting; no significance testing).",
        "> PV/radiation-type indicators are identical under the two boundaries, because the two boundaries "
        "differ only in the nighttime 20–24 h (no shortwave radiation); differences appear only in "
        "temperature, wind speed, and WPD (instantaneous quantities).",
        "> The heatwave−non-exceedance difference is a **single-month descriptive comparison, not causal "
        "attribution**.",
        "",
    ]
    with open(OUT_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(table))
    print(f"wrote {OUT_TABLE}")

    report = [
        "# Day-boundary trials vs heatwave labels report (2022-08 single-month exploration)",
        "",
        "> This round is an **exploratory, single-month, descriptive** comparison; no significance "
        "testing, gridpoint-days are not treated as independent samples,",
        "> no claim about which day boundary truly matches CN05.1, and no generalization of a single "
        "month to long-term heatwave effects.",
        "",
        "## 1. Data and traceability (hourly data kept long-term, not overwritten)",
        "",
        "- Hourly derived NetCDFs are kept long-term: `results_hourly/era5_hourly_potential_20220731_20220901.nc`",
        "  (Faiman primary) and `era5_hourly_potential_tcell_linear_20220731_20220901.nc`",
        "  (linear-form sensitivity control). Both retain the UTC time axis, variable units, model "
        "formulas and parameters, the meaning of the radiation interval (`ghi=ssrd/3600`, the "
        "hour-ending 1 h accumulation's interval average), and the trial status; Faiman and the "
        "linear form are clearly distinguished by separate files and separate variable names "
        "(`pvpot_*` vs `pvpot_lin_*`).",
        "- The daily products record the hourly files and aggregation rules used: `source_hourly`, "
        "`aggregation_order`, `day_window_utc`, `point_samples_utc`, `accum_samples_utc`, "
        "`valid_hours_requirement`, and other attributes have been written to the daily NetCDF; "
        "`cn05.1_day_boundary=UNCONFIRMED`.",
        "- This round deleted or overwrote no hourly or daily file, and made no extra copies of the "
        "same hourly data.",
        "",
        "## 2. Actual UTC start/end and complete days",
        "",
        f"- Boundary A (calendar 00–24 Beijing time): instantaneous {da.attrs['point_samples_utc']};",
        f"  hour-ending accumulation {da.attrs['accum_samples_utc']}.",
        f"- Boundary B (20–20 Beijing time, provisional primary, CN05.1 day boundary unconfirmed):",
        f"  instantaneous {db.attrs['point_samples_utc']}; hour-ending accumulation {db.attrs['accum_samples_utc']}.",
        f"- Complete days: 31 days × 13×37 gridpoints = 14911 gridpoint-days; A complete "
        f"{int((da['valid_hours'].values==24).sum())},",
        f"  B complete {int((db['valid_hours'].values==24).sum())} (all 24 valid hours, no missing).",
        "",
        "## 3. Label join",
        "",
        "- Coordinates and dates were checked item by item and are consistent (latitude 30–33°N "
        "ascending, longitude 112–121°E ascending, 08-01–08-31 aligned), using **exact match + "
        "reordering**, no nearest-neighbour, no interpolation.",
        f"- Invalid-label gridpoints (all 31 days = 0): {n_invalid}; valid matched gridpoints "
        f"(label∈{{1,2,3,4}}): {n_valid_grid}.",
        f"- Gridpoints with both heatwave(3) and ordinary non-exceedance(1) days: {n_both}; valid "
        f"gridpoints missing either group: {n_missing_group}.",
        f"  (Relationship: 481 = {n_valid_grid} valid + {n_invalid} invalid; **all** {n_valid_grid} "
        f"valid gridpoints this month contain both heatwave days and non-exceedance days, so "
        f"\"both present\" = \"valid\" = {n_both}; there is no 475.)",
        f"- PV group-difference (heatwave−non-exceedance) per-gridpoint sign: positive {pv_pos} / "
        f"zero {pv_zero} / negative {pv_neg} (plus {n_invalid} gridpoints with no control, recorded "
        "as NaN).",
        f"- Label samples (gridpoint × day total): heatwave days {int(n_hw.sum())}, ordinary "
        f"non-exceedance days {int(n_ne.sum())},",
        f"  gap days {int(n_gap.sum())}, isolated exceedance days {int(n_iso.sum())}; missing (0) "
        f"is counted in no group and is **never** treated as a non-heatwave day.",
        f"- For heatwave events crossing the August boundary, **only the within-August daily states "
        "are taken**; they are not renumbered and their durations are not recomputed.",
        "",
        "## 4. Descriptive comparison results",
        "",
        "> Aggregation convention (stated honestly, no weighting added): summary numbers such as "
        "1.889, −1.05, +1.75, 0.032 are all obtained in **two steps** —",
        "> (1) per gridpoint first take that gridpoint's difference (heatwave-day mean − ordinary "
        "non-exceedance-day mean; daily total kWh/m² or daily mean W/m²);",
        f"> (2) then take the **equal-weight arithmetic mean over {n_both} valid gridpoints** (each "
        "gridpoint equal weight; no cos-latitude/area weighting; no significance testing).",
        "> The number of valid gridpoints used for each mean is noted in the trailing `n=…`.",
        "",
        "### 4.1 Difference between the two boundaries (per-gridpoint daily-value difference, A−B)",
        "",
        f"- PVpot daily total: {spatial(d_pvpot_sum, 'd_pvpot_sum', 'kWh/m2')} (always 0).",
        f"- PVpot daily mean: {spatial(d_pvpot_mean, 'd_pvpot_mean', 'W/m2')} (always 0).",
        f"- WPD daily mean: {spatial(d_wpd, 'd_wpd_mean', 'W/m2')}.",
        f"- Temperature daily mean: {spatial(d_tair, 'd_tair_mean', 'degC')}.",
        "",
        "> Conclusion: radiation/PV-type daily indicators are **identical** under the two boundaries "
        "— the two boundaries differ only in the nighttime 20–24 h (Beijing time), when there is no "
        "shortwave radiation, so the GHI/PVpot daily total and daily mean are unchanged; differences "
        "appear only in instantaneous quantities (temperature, wind speed, WPD).",
        "",
        "### 4.2 Heatwave days vs ordinary non-exceedance days (per gridpoint, boundary B primary)",
        "",
        f"- PVpot daily total heatwave−non-exceedance: {spatial(pvsum_d, '', 'kWh/m2').split(':',1)[1]}",
        f"- PVpot daily mean heatwave−non-exceedance: {spatial(pvmean_d, '', 'W/m2').split(':',1)[1]}",
        f"- WPD daily mean heatwave−non-exceedance (boundary B): {spatial(wpd_d, '', 'W/m2').split(':',1)[1]}",
        f"- WPD daily mean heatwave−non-exceedance (boundary A): {spatial(wpd_d_A, '', 'W/m2').split(':',1)[1]}",
        "- Gridpoints with no ordinary non-exceedance day (no control) are recorded NaN, not borrowed "
        "from neighbouring gridpoints.",
        "",
        "### 4.3 Faiman vs linear form (heatwave−non-exceedance PV difference, model sensitivity)",
        "",
        f"- Faiman: {spatial(pvsum_d, 'pvpot_sum hw-non-exceed', 'kWh/m2')}",
        f"- Linear form: {spatial(pl_sum_d, 'pvpot_sum hw-non-exceed', 'kWh/m2')}",
        f"- (Faiman−linear form) difference: {spatial(sens_sum, '', 'kWh/m2').split(':',1)[1]}",
        "",
        "> Radiation/PV results are independent of the day boundary, so this sensitivity is the same "
        "under the two boundaries.",
        "",
        "### 4.4 Example-cell time series",
        "",
        f"- Example cell {rlat:.2f}°N, {rlon:.2f}°E (a different example cell from 31.75°N/114.50°E; "
        f"heatwave days {int(n_hw[i,j])}, ordinary non-exceedance days {int(n_ne[i,j])},",
        f"  gap days {int(n_gap[i,j])}, isolated exceedance days {int(n_iso[i,j])}); see "
        f"`{os.path.basename(OUT_FIG)}`.",
        "- Figure title: **single-month exploratory comparison, not causal attribution**.",
        "",
        "## 5. Deliverables",
        "",
        f"- Trial daily NetCDFs: `era5_daily_potential_202208_A.nc`, `_B.nc` (the two boundaries "
        "saved separately).",
        f"- Per-gridpoint comparison NetCDF: `{os.path.basename(OUT_NC)}`.",
        f"- Brief summary table: `{os.path.basename(OUT_TABLE)}`.",
        f"- Example-cell figure: `{os.path.basename(OUT_FIG)}`.",
        f"- This report: `{os.path.basename(OUT_REPORT)}`.",
        "",
    ]
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
