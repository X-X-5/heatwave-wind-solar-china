# -*- coding: utf-8 -*-
"""
Redraw the representative merged-cell time-series figure (phase-2 pilot).

Reads ONLY the saved pilot outputs (results_merge/hw_2022_daily_labels.nc and
results_merge/hw_2022_events_merged.csv). It does NOT recompute events and does
NOT touch the daily-label / event-catalog / summary files. It regenerates
results_merge/fig_hw_2022_merge_cell.png with an improved layout:

  * two-line title (line 1: data + year; line 2: lon/lat);
  * an independent narrow event-marker strip ABOVE the temperature panel that
    shares the date axis and shows each merged process' start/end lines + dates;
  * temperature y-axis margin sized to the Tmax/P90 range only (no headroom for
    the markers, which now live in their own strip);
  * a shared legend at the bottom (full width, never clipped);
  * the "grid-cell event, not a regional event" note moved to the figure caption.

Data/event definitions are unchanged. The heat segments stay pink, the 1-day
gap days stay yellow/hatched, and the date range stays May 1 - Sep 30.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_merge")

REP_LON = 114.50
REP_LAT = 31.75
YEAR = 2022
D0 = pd.Timestamp(f"{YEAR}-05-01")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
HEAT_FILL = "#e34948"     # pink heatwave segments
GAP_FILL = "#eda100"      # yellow/amber 1-day gap days

# ---------------------------------------------------------------- load (read-only)
ds = xr.open_dataset(os.path.join(OUT, "hw_2022_daily_labels.nc"))
lon = ds["lon"].values
lat = ds["lat"].values
jj = int(np.argmin(np.abs(lon - REP_LON)))
ii = int(np.argmin(np.abs(lat - REP_LAT)))
tmax = ds["tmax"].values[:, ii, jj].astype(float)
thr = ds["threshold"].values[:, ii, jj].astype(float)
label = ds["label"].values[:, ii, jj].astype(np.int8)
ds.close()
lon_c = float(lon[jj])
lat_c = float(lat[ii])

df = pd.read_csv(os.path.join(OUT, "hw_2022_events_merged.csv"))
tol = 1e-3
rep = df[(np.abs(df["lon"] - lon_c) < tol) & (np.abs(df["lat"] - lat_c) < tol)]
merged_spans = []
for _, r in rep[rep["n_segments"] > 1].iterrows():
    s = (pd.Timestamp(r["start"]) - D0).days
    e = (pd.Timestamp(r["end"]) - D0).days
    merged_spans.append((int(s), int(e)))

x = np.arange(len(tmax))
hw = (label == 3)   # heatwave day -> contiguous segments
gap = (label == 4)  # 1-day gap day

segs = []
i, n = 0, len(hw)
while i < n:
    if hw[i]:
        j = i
        while j < n and hw[j]:
            j += 1
        segs.append((i, j - 1))
        i = j
    else:
        i += 1

# ---------------------------------------------------------------- figure
fig, (ax_top, ax) = plt.subplots(
    2, 1, figsize=(11, 5.6), sharex=True,
    gridspec_kw={"height_ratios": [0.7, 3.0], "hspace": 0.06})

# --- top strip: merged-process start/end markers (shares the date axis) ---
ax_top.set_ylim(-0.3, 1.55)
ax_top.axis("off")
for (s, e) in merged_spans:
    y = 0.45
    ax_top.plot([s, e], [y, y], color=INK, lw=1.6, solid_capstyle="butt")
    ax_top.plot([s, s], [y - 0.22, y + 0.22], color=INK, lw=1.3)
    ax_top.plot([e, e], [y - 0.22, y + 0.22], color=INK, lw=1.3)
    s_lab = (D0 + pd.Timedelta(days=s)).strftime("%m-%d")
    e_lab = (D0 + pd.Timedelta(days=e)).strftime("%m-%d")
    ax_top.text(s, y + 0.42, s_lab, ha="center", va="bottom", fontsize=7.5,
                color=INK)
    ax_top.text(e, y + 0.42, e_lab, ha="center", va="bottom", fontsize=7.5,
                color=INK)
    ax_top.text((s + e) / 2.0, y + 0.85, "merged process", ha="center",
                va="bottom", fontsize=7, color=MUTED)

# --- bottom panel: temperature ---
lo = float(np.nanmin(np.minimum(tmax, thr)))
hi = float(np.nanmax(np.maximum(tmax, thr)))
span = (hi - lo) if hi > lo else 1.0
ax.set_ylim(lo - 0.08 * span, hi + 0.08 * span)   # margin from Tmax/P90 range only

for (s, e) in segs:
    ax.axvspan(s - 0.5, e + 0.5, color=HEAT_FILL, alpha=0.16, lw=0)
for d in np.where(gap)[0]:
    ax.axvspan(d - 0.5, d + 0.5, color=GAP_FILL, alpha=0.45, lw=0, hatch="////")

ax.plot(x, tmax, color=BLUE, lw=1.8, label="Tmax", zorder=5)
ax.plot(x, thr, color=ORANGE, lw=1.8, ls="--", label="calendar-day P90",
        zorder=4)

ax.set_xticks(x[::14])
ax.set_xticklabels([(D0 + pd.Timedelta(days=int(d))).strftime("%m-%d")
                    for d in x[::14]], fontsize=8)
ax.tick_params(labelsize=8, colors=MUTED)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.set_ylabel("temperature (°C)", fontsize=9, color=INK)
ax.set_xlim(-0.5, len(x) - 0.5)

# shared legend at the bottom (full width, never clipped)
handles = [
    plt.Line2D([], [], color=BLUE, lw=1.8, label="Tmax"),
    plt.Line2D([], [], color=ORANGE, lw=1.8, ls="--", label="P90 threshold"),
    Patch(facecolor=HEAT_FILL, alpha=0.16, label="heatwave segment (≥3 d)"),
    Patch(facecolor=GAP_FILL, alpha=0.45, hatch="////", label="1-day gap day"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.05),
           ncol=4, frameon=False, fontsize=8, borderaxespad=0)

# two-line title + figure caption
fig.suptitle(f"CN05.1 Tmax vs calendar-day P90 — {YEAR}\n"
             f"{lon_c:.2f}°E, {lat_c:.2f}°N", fontsize=10.5, color=INK)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.22, hspace=0.06)
fig.text(0.5, 0.01, "grid-cell event, not a regional event",
         ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")

out_fig = os.path.join(OUT, "fig_hw_2022_merge_cell.png")
fig.savefig(out_fig, dpi=150, facecolor="white")
plt.close(fig)
print(f"wrote {out_fig}")
print(f"  representative cell  lon={lon_c:.2f}E  lat={lat_c:.2f}N")
print(f"  heat segments: {len(segs)}, merged processes: {len(merged_spans)}")
for (s, e) in merged_spans:
    print(f"    {(D0 + pd.Timedelta(days=s)).strftime('%m-%d')} ~ "
          f"{(D0 + pd.Timedelta(days=e)).strftime('%m-%d')}")
print("DONE redraw (layout only; data & events unchanged).")
