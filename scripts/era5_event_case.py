# -*- coding: utf-8 -*-
"""
Single-event case description at the example cell 114.5E / 31.75N.

Explains the EXISTING merged heatwave process 2022-08-02..2022-08-22. The event
start/end are read back from the existing event table (results_merge/
hw_2022_events_merged.csv) and asserted -- no event re-identification.

Reads existing daily boundary-B products + aligned CN05.1 labels (no download, no
recompute, no model change). Produces ONE faceted figure (each panel its own y-axis;
no dual-axis overlay) and a brief markdown note.

Panels:
  1. CN05.1 Tmax + calendar-day P90 threshold   [degC]
  2. GHI daily cumulative                        [kWh m-2]
  3. Faiman PV potential daily cumulative        [kWh m-2]
  4. 100 m wind speed daily mean                 [m s-1]
  5. Wind power density daily mean               [W m-2]
  6. Model temperature correction PVpot - GHI    [kWh m-2]  (same hourly interval,
     relative to 25 C reference; NOT a heatwave causal effect / "loss")

Heatwave days (label 3) and merged gap days (label 4) are shaded. Single-cell,
single-event descriptive only -- not representative of the whole region.
"""
import os
import sys
import csv

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "results_daily")
MERGE = os.path.join(ROOT, "results_merge")
OUT = os.path.join(ROOT, "results_case")

DAILY_B = os.path.join(DAILY, "era5_daily_potential_20220501_20220930_B.nc")
LABEL_NC = os.path.join(DAILY, "cn05_labels_20220501_20220930_aligned.nc")
EVENT_CSV = os.path.join(MERGE, "hw_2022_events_merged.csv")

EX_LON, EX_LAT = 114.5, 31.75
PAD_DAYS = 5

# colors (single series = title-named; annotations carry the legend)
C_DATA = "#1f6f9f"     # blue
C_P90 = "#d95f02"      # orange (dashed reference)
C_HW = "#e15759"       # heatwave shading
C_GAP = "#9e9e9e"      # gap shading
C_ZERO = "#8a8a8a"


def load_event():
    """Confirm the example-cell August event from the existing table (no re-identify)."""
    rows = []
    with open(EVENT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["lon"]) == EX_LON and float(r["lat"]) == EX_LAT:
                rows.append(r)
    ev = [r for r in rows if r["start"] == "2022-08-02" and r["end"] == "2022-08-22"]
    if len(ev) != 1:
        raise SystemExit(f"expected exactly 1 example-cell event 08-02..08-22, found {len(ev)}")
    return ev[0]


def shade_runs(ax, mask, color, alpha=0.28):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        ax.axvspan(r[0] - 0.5, r[-1] + 0.5, color=color, alpha=alpha, zorder=0, lw=0)


def fmt_date(d):
    return np.datetime_as_string(d, unit="D")[5:]


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = load_event()
    start = np.datetime64(ev["start"])
    end = np.datetime64(ev["end"])
    w0 = start - np.timedelta64(PAD_DAYS, "D")
    w1 = end + np.timedelta64(PAD_DAYS, "D")

    db = xr.open_dataset(DAILY_B)
    lb = xr.open_dataset(LABEL_NC)

    c = db.sel(latitude=EX_LAT, longitude=EX_LON, method="nearest")
    lc = lb.sel(latitude=EX_LAT, longitude=EX_LON, method="nearest")

    w = c.sel(date=slice(w0, w1))
    wl = lc.sel(date=slice(w0, w1))
    dates = w["date"].values
    x = np.arange(dates.size)

    tmax = wl["tmax"].values
    p90 = wl["threshold"].values
    lbl = wl["label"].values
    ghi = w["ghi_sum"].values
    pv = w["pvpot_sum"].values
    w100 = w["wind100_mean"].values
    wpd = w["wpd_mean"].values
    corr = pv - ghi                      # model temperature correction (vs 25 C ref)

    hw = lbl == 3
    gap = lbl == 4
    gap_day = dates[gap]

    # ---- descriptive numbers for the note ----
    hw_m = hw
    before_after = (dates < start) | (dates > end)      # ordinary context days
    def m(a, mask):
        return float(np.nanmean(a[mask]))
    num = {
        "n_seg": ev["n_segments"], "hw_days": ev["heatwave_days"],
        "gap_days": ev["gap_days"], "span": ev["span_days"],
        "excess": float(ev["mean_excess_C"]),
        "gap_date": fmt_date(gap_day[0]) if gap_day.size else "none",
        "tmax_peak": float(np.nanmax(tmax[hw_m])),
        "tmax_peak_date": fmt_date(dates[hw_m][np.argmax(tmax[hw_m])]),
        "p90_range": (float(np.nanmin(p90[hw_m])), float(np.nanmax(p90[hw_m]))),
        "ghi_hw_mean": m(ghi, hw_m), "ghi_ctx_mean": m(ghi, before_after),
        "pv_hw_mean": m(pv, hw_m), "pv_ctx_mean": m(pv, before_after),
        "ghi_peak_date": fmt_date(dates[np.argmax(ghi)]),
        "pv_peak_date": fmt_date(dates[np.argmax(pv)]),
        "corr_hw_mean": m(corr, hw_m), "corr_hw_min": float(np.nanmin(corr[hw_m])),
        "w100_range": (float(np.nanmin(w100[hw_m])), float(np.nanmax(w100[hw_m]))),
        "wpd_range": (float(np.nanmin(wpd[hw_m])), float(np.nanmax(wpd[hw_m]))),
    }

    # ---- figure ----
    panels = [
        ("CN05.1 Tmax", tmax, "°C", dict(p90=p90)),
        ("GHI daily cumulative", ghi, "kWh m$^{-2}$", {}),
        ("Faiman PV potential (daily cumulative)", pv, "kWh m$^{-2}$", {}),
        ("100 m wind speed (daily mean)", w100, "m s$^{-1}$", {}),
        ("Wind power density (daily mean)", wpd, "W m$^{-2}$", {}),
        ("Model temperature correction\n(PVpot $-$ GHI, vs 25 °C reference)",
         corr, "kWh m$^{-2}$", dict(zero=True)),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 13.2), sharex=True)
    for ax, (title, y, unit, opt) in zip(axes, panels):
        shade_runs(ax, hw, C_HW, alpha=0.28)
        shade_runs(ax, gap, C_GAP, alpha=0.45)
        ax.plot(x, y, color=C_DATA, lw=1.2, zorder=3)
        if "p90" in opt:
            ax.plot(x, opt["p90"], color=C_P90, lw=1.1, ls="--", zorder=3)
        if opt.get("zero"):
            ax.axhline(0, color=C_ZERO, ls=":", lw=1.0, zorder=2)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(True, ls=":", lw=0.5, alpha=0.4)
        ax.tick_params(axis="y", labelsize=8.5)

    # x ticks every 5 days
    tick_i = np.arange(0, dates.size, 5)
    axes[-1].set_xticks(tick_i)
    axes[-1].set_xticklabels([fmt_date(dates[i]) for i in tick_i], fontsize=8.5)
    axes[-1].set_xlabel("2022 (MM-DD)", fontsize=9)

    # panel-1 local legend: Tmax + P90 only (the blue line in other panels is NOT Tmax)
    axes[0].legend(
        handles=[plt.Line2D([0], [0], color=C_DATA, lw=1.2, label="Tmax (CN05.1)"),
                 plt.Line2D([0], [0], color=C_P90, lw=1.1, ls="--", label="P90 threshold")],
        loc="upper right", frameon=False, fontsize=8)

    # figure-wide shared legend: heatwave / gap shading applies to every panel
    fig.legend(
        handles=[mpatches.Patch(color=C_HW, alpha=0.28, label="heatwave day (label 3)"),
                 mpatches.Patch(color=C_GAP, alpha=0.45, label="gap day (label 4)")],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2,
        frameon=False, fontsize=8)

    fig.suptitle(
        f"Example cell {EX_LAT:.2f}°N, {EX_LON:.2f}°E — merged heatwave "
        f"process {fmt_date(start)}–{fmt_date(end)} (±{PAD_DAYS} d context)\n"
        f"day boundary B (provisional) = BJT previous-day 20:00 → same-day 20:00; "
        f"CN05.1 day boundary unconfirmed",
        fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    fig_png = os.path.join(OUT, "fig_hw_2022_case_114.5E_31.75N_20220802_20220822.png")
    fig.savefig(fig_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- brief note ----
    L = []
    L.append("# 示例格点热浪过程案例说明（单点、单事件描述，不代表整个区域）\n\n")
    L.append("## 示例格点（31.75°N、114.50°E）\n\n")
    L.append("- 术语：本文称该格点为**示例格点**，不称“代表格点”。\n")
    L.append("- 来源：坐标沿用自热浪合并试点（`scripts/hw_merge_pilot.py` 与 "
             "`scripts/hw_merge_plot.py` 中写死为“phase-1 representative cell”）。其最初选择依据 "
             "来自第一阶段诊断脚本 `scripts/eval_merge.py`：在 30–33°N、112–121°E 小区域内**自动筛选** "
             "2022 年热浪事件日最多（≥3 天连续超阈值的总日数最大）的格点，结果即 114.50°E、31.75°N"
             "（52 个热浪事件日，为该区域唯一最大值）。\n")
    L.append("- 定位：仅用于流程与事件过程展示，**未经区域代表性验证**，不代表整个区域。\n\n")
    L.append("## 事件（来自现有事件表 `hw_2022_events_merged.csv`，未重新识别）\n\n")
    L.append(f"- 格点：{EX_LON}°E, {EX_LAT}°N\n")
    L.append(f"- 过程：{fmt_date(start)} .. {fmt_date(end)} "
             f"（跨度 {num['span']} 天；{num['n_seg']} 段合并，"
             f"{num['hw_days']} 个热浪日 + {num['gap_days']} 个间隔日；"
             f"mean_excess = {num['excess']:.2f} °C）\n")
    L.append(f"- 间隔日（label 4）：{num['gap_date']}\n\n")
    L.append(f"## 图\n\n`{os.path.basename(fig_png)}` — 6 个分面（各自独立纵轴，无同轴叠图）："
             "Tmax+P90、GHI 日累计、Faiman PV 日累计、100m 风速日均、WPD 日均、"
             "模型温度修正（PVpot−GHI）。红色底纹=热浪日，灰色底纹=间隔日。\n\n")
    L.append("## 描述性观察（单点、单事件，不做显著性检验，不做因果归因）\n\n")
    L.append(f"1. **温度**：过程内 Tmax 普遍高于 P90，峰值 "
             f"{num['tmax_peak']:.1f} °C（{num['tmax_peak_date']}）；"
             f"P90 日历日阈值在过程内约 {num['p90_range'][0]:.1f}–{num['p90_range'][1]:.1f} °C。\n")
    L.append(f"2. **PV 偏高与辐照偏高相伴**：PV 日累计 ≈ GHI × pr（pr≈1），两者几乎同步。"
             f"过程内 GHI 日累计均值 {num['ghi_hw_mean']:.2f} kWh/m² "
             f"（前后普通日 {num['ghi_ctx_mean']:.2f}），"
             f"PV 日累计均值 {num['pv_hw_mean']:.2f} kWh/m² "
             f"（前后 {num['pv_ctx_mean']:.2f}）；"
             f"GHI 峰值日 {num['ghi_peak_date']} 与 PV 峰值日 {num['pv_peak_date']} "
             f"{'一致' if num['ghi_peak_date']==num['pv_peak_date'] else '不一致'}。\n")
    L.append(f"3. **风速与 WPD 一致变化**：WPD=0.5·ρ·V³ 是风速的单调函数，过程内两者同升降，"
             f"且 WPD 变化被三次方放大。过程内风速日均约 "
             f"{num['w100_range'][0]:.1f}–{num['w100_range'][1]:.1f} m/s，"
             f"WPD 日均约 {num['wpd_range'][0]:.0f}–{num['wpd_range'][1]:.0f} W/m²。\n")
    L.append(f"4. **模型温度修正**（PVpot−GHI，相对 25 °C 参考，同一小时区间累计）："
             f"过程内为负（热环境下 pr<1），日均约 {num['corr_hw_mean']:.2f} kWh/m² "
             f"（最负 {num['corr_hw_min']:.2f}）。这仅是模型内相对 25 °C 参考的温度修正项，"
             "不是热浪因果效应，也不直接称为热浪损失。\n\n")
    L.append("注：日界 B = 北京时间前日 20 时—当日 20 时（暂定方案）；"
             "CN05.1 日界尚未确认；原数据与全部已有结果未改动。\n")

    note_md = os.path.join(OUT, "note_hw_2022_case_114.5E_31.75N_20220802_20220822.md")
    with open(note_md, "w", encoding="utf-8") as f:
        f.write("".join(L))

    print(f"wrote {fig_png}")
    print(f"wrote {note_md}")
    for k, v in num.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
