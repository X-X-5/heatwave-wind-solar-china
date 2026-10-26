# -*- coding: utf-8 -*-
"""
Phase 1 maps (fixed): heatwave feature maps (climatology 1961-2022) and 2022 maps.

Reads results_fixed/hw_annual_1961-2022.nc and results_fixed/valid_mask.nc;
writes PNGs to results_fixed/. Ocean / no-data cells are masked (neutral gray);
no external boundary file is used (the valid-data region itself outlines China).

Colour: single-hue sequential (light -> dark red) for heat magnitude; darker =
more / stronger heatwave. No rainbow, no diverging colour here (all four
metrics are unsigned magnitudes).

Colorbar fix (phase-1): the 2022 maps no longer reuse the climatology vmax.
Each 2022 panel's upper bound is rounded UP so it covers the actual 2022
maximum (nothing clipped; the actual max and the chosen vmax are printed for
every panel). Climatology keeps its original (unsaturated) bounds.
"""
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_fixed")

HEAT_CMAP = "Reds"          # single-hue sequential, light -> dark
OCEAN = "#dcdcdc"           # neutral gray for masked/no-data cells
INK = "#0b0b0b"

ds = xr.open_dataset(os.path.join(RES, "hw_annual_1961-2022.nc"))
vm = xr.open_dataset(os.path.join(RES, "valid_mask.nc"))
lat = ds["lat"].values
lon = ds["lon"].values
valid = vm["valid_mask"].values.astype(bool)
vm.close()

FILL_I = -9999

metrics = [
    ("hw_count", "Heatwave count", "events/yr"),
    ("hw_days", "Heatwave days", "day/yr"),
    ("hw_max_dur", "Max duration", "day"),
    ("hw_intensity", "Mean exceedance intensity", "degC"),
]
# climatology keeps the original (unsaturated) bounds
CLIM_VMAX = {"hw_count": 4, "hw_days": 20, "hw_max_dur": 10, "hw_intensity": 3}
# rounding step used to raise the 2022 upper bound just above the actual max
STEP = {"hw_count": 2, "hw_days": 10, "hw_max_dur": 5, "hw_intensity": 0.5}


def field(name, year_slice=None):
    a = ds[name].values.astype(float)
    a = np.where(a == FILL_I, np.nan, a)
    if year_slice is None:
        a = np.nanmean(a, axis=0)          # climatology over 1961-2022
    else:
        a = a[year_slice]
    a = np.where(valid, a, np.nan)         # mask ocean
    return np.ma.masked_invalid(a)


def plot_panel(ax, arr, title, unit, vmin, vmax):
    cm = plt.get_cmap(HEAT_CMAP).copy()
    cm.set_bad(OCEAN)
    im = ax.pcolormesh(lon, lat, arr, cmap=cm, vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.set_facecolor(OCEAN)
    ax.set_title(title, fontsize=11, color=INK, pad=6)
    ax.set_xlim(lon[0], lon[-1])
    ax.set_ylim(lat[0], lat[-1])
    ax.set_xticks(np.arange(70, 141, 10))
    ax.set_yticks(np.arange(15, 56, 10))
    ax.tick_params(labelsize=8, colors="#52514e")
    ax.set_aspect(1 / np.cos(np.deg2rad(35)))
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(unit, fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8, colors="#52514e")


def make_figure(suptitle, year_slice, outpath, caption=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (name, title, unit) in zip(axes.ravel(), metrics):
        arr = field(name, year_slice)
        if year_slice is None:
            vmax = CLIM_VMAX[name]
        else:
            m = float(arr.max())
            vmax = (np.floor(m / STEP[name]) + 1) * STEP[name]   # round up, cover
            print(f"  2022 {name:<12}: actual max={m:6.2f}  colorbar vmax={vmax:g}")
        plot_panel(ax, arr, title, unit, 0, vmax)
    fig.suptitle(suptitle, fontsize=13, color=INK)
    if caption:
        fig.text(0.5, 0.02, caption, ha="center", va="bottom",
                 fontsize=8, color="#52514e")
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", outpath)


BASE_CAP = ("Shading: CN05.1 Tmax strict consecutive events (≥3 days with Tmax above the "
            "calendar-day P90; gap days NOT merged). "
            "Grey = outside the analysis mask or no valid data. "
            "The valid-data outline shows data coverage only, not an independent national border.")
DUR_CAP = ("Climatological Max duration = multi-year mean over 1961–2022 of each grid cell's "
           "annual maximum duration (NOT the all-period maximum).")

make_figure("CN05.1 Tmax heatwave climatology 1961-2022 (May-Sep)", None,
            os.path.join(RES, "fig_hw_climatology.png"),
            caption=BASE_CAP + DUR_CAP)

iy2022 = int(np.where(ds["year"].values == 2022)[0][0])
make_figure("CN05.1 Tmax heatwaves in 2022 (May-Sep)", iy2022,
            os.path.join(RES, "fig_hw_2022.png"),
            caption=BASE_CAP)

ds.close()
print("DONE maps")
