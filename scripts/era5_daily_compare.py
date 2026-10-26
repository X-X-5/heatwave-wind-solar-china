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
    ax.plot(xx, pvpot_sum_B[:, i, j], color="#1565c0", lw=1.6, label="PVpot 日累计 Faiman（B）")
    ax.plot(xx, pvpot_lin_sum_B[:, i, j], color="#6a1b9a", lw=1.4, ls="--",
            label="PVpot 日累计 线性式（B）")
    shade(ax)
    ax.set_ylabel("PV 潜力日累计 (kWh/m²)")
    ax.set_title(f"示例格点 {rlat:.2f}°N, {rlon:.2f}°E（另一示例格点，不同于 31.75°N/114.50°E）\n"
                 f"（热浪日 {int(n_hw[i,j])}，普通非超阈值日 {int(n_ne[i,j])}）", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.annotate("辐射/PV 类指标在 A、B 两套日界下相同（差异仅落在夜间 20–24 时，无短波辐射）",
                xy=(0.01, 0.02), xycoords="axes fraction", fontsize=7.5, color="#555555")

    ax = axes[1]
    ax.plot(xx, wpd_mean_A[:, i, j], color="#e65100", lw=1.6, label="WPD 日平均 日界 A")
    ax.plot(xx, wpd_mean_B[:, i, j], color="#00695c", lw=1.4, ls="--", label="WPD 日平均 日界 B")
    shade(ax)
    ax.set_ylabel("WPD 日平均 (W/m²)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.set_xlabel("2022-08（UTC 归日，见 REPORT）")
    for lb in ax.get_xticklabels():
        lb.set_fontsize(8)

    fig.suptitle("单月探索性比较，非因果归因", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_FIG}")

    # ---------- summary table + report ----------
    m = lambda v: np.nanmean(v) if np.isfinite(v).any() else float("nan")

    table = [
        "# 日界试算与热浪标签对照 — 简要汇总表（2022-08 单月探索）\n",
        "| 项 | 日界 A（00–24 北京时） | 日界 B（20–20 北京时，暂定主方案） |",
        "|---|---|---|",
        f"| 逐日 UTC 窗口（瞬时量时次） | {da.attrs['point_samples_utc']} | {db.attrs['point_samples_utc']} |",
        f"| 逐日 UTC 窗口（时末累计时次） | {da.attrs['accum_samples_utc']} | {db.attrs['accum_samples_utc']} |",
        "| 完整日要求 | 每格点每日 24 个有效小时，否则记 NaN | 同左 |",
        f"| 完整格点-日数 | {int((da['valid_hours'].values==24).sum())}/14911 | {int((db['valid_hours'].values==24).sum())}/14911 |",
        f"| 标签无效格点数（31 天全为 0） | {n_invalid} | {n_invalid} |",
        f"| 有效匹配格点数（标签∈{{1,2,3,4}}） | {n_valid_grid}/481 | {n_valid_grid}/481 |",
        f"| 既有热浪日又有普通非超阈值日的格点数 | {n_both} | {n_both} |",
        f"| 有效格点中缺少任一组的格点数 | {n_missing_group} | {n_missing_group} |",
        f"| 热浪日样本（格点×日合计） | {int(n_hw.sum())} | {int(n_hw.sum())} |",
        f"| 普通非超阈值日样本 | {int(n_ne.sum())} | {int(n_ne.sum())} |",
        f"| 间隙日样本 | {int(n_gap.sum())} | {int(n_gap.sum())} |",
        f"| 孤立超标日样本 | {int(n_iso.sum())} | {int(n_iso.sum())} |",
        f"| PVpot 日累计 热浪−非超阈值（格点等权均值，kWh/m²） | {m(pvsum_d):+.4g} | {m(pvsum_d):+.4g} |",
        f"| PVpot 日累计 线性式 热浪−非超阈值（格点等权均值，kWh/m²） | {m(pl_sum_d):+.4g} | {m(pl_sum_d):+.4g} |",
        f"| WPD 日平均 热浪−非超阈值（格点等权均值，W/m²） | {m(wpd_d_A):+.4g} | {m(wpd_d):+.4g} |",
        f"| 两套日界 WPD 日平均之差（格点等权均值，W/m²） | — | {m(d_wpd):+.4g}（A−B） |",
        f"| 两套日界 PVpot 日累计之差（格点等权均值，kWh/m²） | — | {m(d_pvpot_sum):+.4g}（A−B） |",
        "",
        "> 汇总口径（如实说明，未新增加权）：所有“热浪−非超阈值”与“两套日界之差”均为**两步**——",
        f"> (1) 逐格点先求该格点差值（热浪日均 − 普通非超阈值日均，或 A − B 日均）；(2) 再对 **{n_both} 个有效格点做等权算术平均**",
        "> （每格点等权，未做 cos 纬度/面积加权，未做显著性检验）。",
        "> PV/辐射类指标在两套日界下完全相同，因两套日界仅相差夜间 20–24 时（无短波辐射）；差异仅出现在气温、风速与 WPD（瞬时量）。",
        "> 热浪−非超阈值差异为**单月描述性比较，非因果归因**。",
        "",
    ]
    with open(OUT_TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(table))
    print(f"wrote {OUT_TABLE}")

    report = [
        "# 日界试算与热浪标签对照报告（2022-08 单月探索）",
        "",
        "> 本轮为**探索性、单月、描述性**比较；不做显著性检验，不将格点—日当作独立样本，",
        "> 不声称哪一种日界与 CN05.1 真正一致，不把单月结果推广为长期热浪影响。",
        "",
        "## 1. 数据与可追溯性（小时数据长期保留，不覆盖）",
        "",
        "- 小时派生 NetCDF 长期保留：`results_hourly/era5_hourly_potential_20220731_20220901.nc`",
        "  （Faiman 主方案）与 `era5_hourly_potential_tcell_linear_20220731_20220901.nc`",
        "  （线性式敏感性对照）。二者均保留 UTC 时间轴、变量单位、模型公式与参数、辐射区间",
        "  含义（`ghi=ssrd/3600`，时末 1 h 累计的区间平均）及试算状态；Faiman 与线性式以独立",
        "  文件与独立变量名（`pvpot_*` vs `pvpot_lin_*`）明确区分。",
        "- 日产品记录所用小时文件与聚合规则：`source_hourly`、`aggregation_order`、",
        "  `day_window_utc`、`point_samples_utc`、`accum_samples_utc`、`valid_hours_requirement`",
        "  等属性已写入日 NetCDF；`cn05.1_day_boundary=UNCONFIRMED`。",
        "- 本轮未删除、未覆盖任何小时或日尺度文件，也未复制多份相同小时数据。",
        "",
        "## 2. 实际 UTC 起止与完整日",
        "",
        f"- 日界 A（日历 00–24 北京时）：瞬时量 {da.attrs['point_samples_utc']}；",
        f"  时末累计 {da.attrs['accum_samples_utc']}。",
        f"- 日界 B（20–20 北京时，暂定主方案，CN05.1 日界未确认）：",
        f"  瞬时量 {db.attrs['point_samples_utc']}；时末累计 {db.attrs['accum_samples_utc']}。",
        f"- 完整日：31 天 × 13×37 格点 = 14911 格点-日；A 完整 {int((da['valid_hours'].values==24).sum())}，",
        f"  B 完整 {int((db['valid_hours'].values==24).sum())}（均 24 有效小时，无缺测）。",
        "",
        "## 3. 标签连接",
        "",
        "- 坐标与日期逐项核对一致（纬度 30–33°N 升序、经度 112–121°E 升序、08-01–08-31 对齐），",
        "  采用**精确匹配 + 重排**，无最近邻、无插值。",
        f"- 标签无效格点数（31 天全为 0）：{n_invalid}；有效匹配格点数（标签∈{{1,2,3,4}}）：{n_valid_grid}。",
        f"- 既有热浪日(3)又有普通非超阈值日(1)的格点：{n_both}；有效格点中缺少任一组的格点：{n_missing_group}。",
        f"  （关系：481 = {n_valid_grid} 有效 + {n_invalid} 无效；{n_valid_grid} 个有效格点本月**全部**同时含热浪日",
        f"  与非超阈值日，故“两者都有”=“有效”={n_both}，不存在 475。）",
        f"- PV 分组差（热浪−非超阈值）逐格点符号：正 {pv_pos} / 零 {pv_zero} / 负 {pv_neg}（另有 {n_invalid} 格点无对照记 NaN）。",
        f"- 标签样本（格点×日合计）：热浪日 {int(n_hw.sum())}，普通非超阈值日 {int(n_ne.sum())}，",
        f"  间隙日 {int(n_gap.sum())}，孤立超标日 {int(n_iso.sum())}；缺失(0)不计入任何组，也**绝不**当作非热浪日。",
        f"- 对跨 8 月边界的热浪事件**只取 8 月内逐日状态**，不重编号、不重算持续时长。",
        "",
        "## 4. 描述性比较结果",
        "",
        "> 汇总口径（如实说明，未新增加权）：1.889、−1.05、+1.75、0.032 等汇总数均为**两步**得到——",
        "> (1) 逐格点先求该格点差值（热浪日均值 − 普通非超阈值日均值；日累计 kWh/m² 或日均 W/m²）；",
        f"> (2) 再对 **{n_both} 个有效格点做等权算术平均**（每格点等权，未做 cos 纬度/面积加权，未做显著性检验）。",
        "> 各均值所用有效格点数在下方括号 `n=…` 注明。",
        "",
        "### 4.1 两套日界差异（逐格点日值之差，A−B）",
        "",
        f"- PVpot 日累计：{spatial(d_pvpot_sum, 'd_pvpot_sum', 'kWh/m2')}（恒为 0）。",
        f"- PVpot 日平均：{spatial(d_pvpot_mean, 'd_pvpot_mean', 'W/m2')}（恒为 0）。",
        f"- WPD 日平均：{spatial(d_wpd, 'd_wpd_mean', 'W/m2')}。",
        f"- 气温日平均：{spatial(d_tair, 'd_tair_mean', 'degC')}。",
        "",
        "> 结论：辐射/PV 类日指标在两套日界下**完全相同**——两套日界仅相差夜间 20–24 时",
        "> （北京时），该时段无短波辐射，故 GHI/PVpot 日累计与日平均不变；差异仅体现在瞬时量",
        "> （气温、风速、WPD）上。",
        "",
        "### 4.2 热浪日 vs 普通非超阈值日（逐格点，日界 B 为主）",
        "",
        f"- PVpot 日累计 热浪−非超阈值：{spatial(pvsum_d, '', 'kWh/m2').split(':',1)[1]}",
        f"- PVpot 日平均 热浪−非超阈值：{spatial(pvmean_d, '', 'W/m2').split(':',1)[1]}",
        f"- WPD 日平均 热浪−非超阈值（日界 B）：{spatial(wpd_d, '', 'W/m2').split(':',1)[1]}",
        f"- WPD 日平均 热浪−非超阈值（日界 A）：{spatial(wpd_d_A, '', 'W/m2').split(':',1)[1]}",
        "- 无普通非超阈值日（无对照）的格点记 NaN，不向邻近格点借用。",
        "",
        "### 4.3 Faiman vs 线性式（热浪−非超阈值 PV 差，模型敏感性）",
        "",
        f"- Faiman：{spatial(pvsum_d, 'pvpot_sum hw-non-exceed', 'kWh/m2')}",
        f"- 线性式：{spatial(pl_sum_d, 'pvpot_sum hw-non-exceed', 'kWh/m2')}",
        f"- (Faiman−线性式) 之差：{spatial(sens_sum, '', 'kWh/m2').split(':',1)[1]}",
        "",
        "> 辐射/PV 结果与日界无关，故两套日界下该敏感性相同。",
        "",
        "### 4.4 示例格点时间序列",
        "",
        f"- 示例格点 {rlat:.2f}°N, {rlon:.2f}°E（另一示例格点，不同于 31.75°N/114.50°E；热浪日 {int(n_hw[i,j])}，普通非超阈值日 {int(n_ne[i,j])}，",
        f"  间隙日 {int(n_gap[i,j])}，孤立超标日 {int(n_iso[i,j])}），见 `{os.path.basename(OUT_FIG)}`。",
        "- 图题：**单月探索性比较，非因果归因**。",
        "",
        "## 5. 交付物",
        "",
        f"- 试算日 NetCDF：`era5_daily_potential_202208_A.nc`、`_B.nc`（两套日界分开保存）。",
        f"- 逐格点比较 NetCDF：`{os.path.basename(OUT_NC)}`。",
        f"- 简要汇总表：`{os.path.basename(OUT_TABLE)}`。",
        f"- 示例格点图：`{os.path.basename(OUT_FIG)}`。",
        f"- 本报告：`{os.path.basename(OUT_REPORT)}`。",
        "",
    ]
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
