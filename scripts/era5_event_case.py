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
    L.append("# Example-cell heatwave-process case note (single point, single event; does not represent the whole region)\n\n")
    L.append("## Example cell (31.75°N, 114.50°E)\n\n")
    L.append("- Terminology: this note calls the cell an **example cell**, not a \"representative cell\".\n")
    L.append("- Origin: the coordinates are carried over from the heatwave-merging pilot "
             "(`scripts/hw_merge_pilot.py` and `scripts/hw_merge_plot.py`, where they are hard-coded as "
             "\"phase-1 representative cell\"). The original selection criterion comes from the phase-1 "
             "diagnostic script `scripts/eval_merge.py`: within the small 30–33°N / 112–121°E region it "
             "**automatically selects** the grid cell with the most 2022 heatwave-event days (the largest total "
             "days of ≥3-day consecutive exceedance), which is 114.50°E, 31.75°N "
             "(52 heatwave-event days, the unique regional maximum).\n")
    L.append("- Purpose: only for pipeline and event-process display, **not verified for regional "
             "representativeness**, and does not represent the whole region.\n\n")
    L.append("## Event (from the existing event table `hw_2022_events_merged.csv`; not re-identified)\n\n")
    L.append(f"- Cell: {EX_LON}°E, {EX_LAT}°N\n")
    L.append(f"- Process: {fmt_date(start)} .. {fmt_date(end)} "
             f"(span {num['span']} days; {num['n_seg']} segments merged, "
             f"{num['hw_days']} heatwave days + {num['gap_days']} gap days; "
             f"mean_excess = {num['excess']:.2f} °C)\n")
    L.append(f"- Gap day(s) (label 4): {num['gap_date']}\n\n")
    L.append(f"## Figure\n\n`{os.path.basename(fig_png)}` — 6 panels (each with an independent y-axis, "
             "no shared-axis overplot): "
             "Tmax+P90, GHI daily total, Faiman PV daily total, 100 m wind-speed daily mean, WPD daily mean, "
             "model temperature correction (PVpot−GHI). Red shading = heatwave day, grey shading = gap day.\n\n")
    L.append("## Descriptive observations (single point, single event; no significance test, no causal attribution)\n\n")
    L.append(f"1. **Temperature**: within the process Tmax is generally above P90, peaking at "
             f"{num['tmax_peak']:.1f} °C ({num['tmax_peak_date']}); "
             f"the P90 calendar-day threshold within the process is about {num['p90_range'][0]:.1f}–{num['p90_range'][1]:.1f} °C.\n")
    L.append(f"2. **High PV accompanies high irradiance**: PV daily total ≈ GHI × pr (pr≈1), and the two are "
             f"almost synchronous. Within the process, the mean GHI daily total is {num['ghi_hw_mean']:.2f} kWh/m² "
             f"(ordinary days before/after {num['ghi_ctx_mean']:.2f}), "
             f"and the mean PV daily total is {num['pv_hw_mean']:.2f} kWh/m² "
             f"(before/after {num['pv_ctx_mean']:.2f}); "
             f"the GHI peak day {num['ghi_peak_date']} and the PV peak day {num['pv_peak_date']} "
             f"{'coincide' if num['ghi_peak_date']==num['pv_peak_date'] else 'do not coincide'}.\n")
    L.append(f"3. **Wind speed and WPD change together**: WPD=0.5·ρ·V³ is a monotonic function of wind speed, so "
             f"within the process the two rise and fall together, and WPD changes are amplified by the cubic term. "
             f"Within the process, the daily-mean wind speed is about "
             f"{num['w100_range'][0]:.1f}–{num['w100_range'][1]:.1f} m/s, "
             f"and the daily-mean WPD is about {num['wpd_range'][0]:.0f}–{num['wpd_range'][1]:.0f} W/m².\n")
    L.append(f"4. **Model temperature correction** (PVpot−GHI, relative to the 25 °C reference, accumulated over "
             f"the same hourly interval): negative within the process (pr<1 under hot conditions), with a daily "
             f"mean of about {num['corr_hw_mean']:.2f} kWh/m² "
             f"(most negative {num['corr_hw_min']:.2f}). This is only the model's temperature-correction term "
             "relative to the 25 °C reference, not a heatwave causal effect, and is not directly called a "
             "heatwave loss.\n\n")
    L.append("Note: day boundary B = Beijing time previous-day 20:00 to same-day 20:00 (provisional scheme); "
             "the CN05.1 day boundary is not yet confirmed; the raw data and all existing results are unchanged.\n")

    note_md = os.path.join(OUT, "note_hw_2022_case_114.5E_31.75N_20220802_20220822.md")
    with open(note_md, "w", encoding="utf-8") as f:
        f.write("".join(L))

    print(f"wrote {fig_png}")
    print(f"wrote {note_md}")
    for k, v in num.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
