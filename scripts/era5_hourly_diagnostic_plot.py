# -*- coding: utf-8 -*-
"""
Diagnostic time-series plot for ONE example ERA5 grid cell.

Reads a Faiman hourly NetCDF given by --input and plots, for the example
cell (31.75 N, 114.50 E -- the same cell used in the heatwave pilot figure):

    air temperature (degC), GHI (W m-2), 100 m wind speed (m s-1),
    WPD (W m-2), PV potential (W m-2)

Five single-axis panels sharing the UTC time axis. Annotates UTC, units, and
"potential, not actual generation". Read-only; never modifies the NetCDF.
"""
import os
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_hourly")

REP_LAT = 31.75
REP_LON = 114.50

PANELS = [
    ("tair", "Air temperature", "°C"),
    ("ghi", "GHI (horizontal)", "W m⁻²"),
    ("wind100", "Wind speed @ 100 m", "m s⁻¹"),
    ("wpd", "Wind power density", "W m⁻²"),
    ("pvpot", "PV potential", "W m⁻²"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to the Faiman hourly NetCDF (explicit, no glob fallback)")
    args = ap.parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"input NetCDF not found: {args.input}")
    ds = xr.open_dataset(args.input)
    cell = ds.sel(latitude=REP_LAT, longitude=REP_LON, method="nearest")
    lat = float(cell["latitude"])
    lon = float(cell["longitude"])
    time = cell["valid_time"].values

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9.2, 11.2), sharex=True)
    for ax, (var, label, unit) in zip(axes, PANELS):
        y = cell[var].values
        ax.plot(time, y, lw=1.1, color="#1f6f9f")
        ax.set_ylabel(f"{unit}", fontsize=9)
        ax.set_title(f"{label}", fontsize=9.5, loc="left")
        ax.grid(True, ls=":", lw=0.5, alpha=0.45)
        ax.tick_params(labelsize=8)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("UTC time (month-day)", fontsize=9)

    t0 = np.datetime_as_string(time[0], unit="m").replace("T", " ")
    t1 = np.datetime_as_string(time[-1], unit="m").replace("T", " ")
    fig.suptitle(
        f"Hourly ERA5 potential — example cell {lat:.2f}°N, {lon:.2f}°E  "
        f"(UTC)\n{t0} … {t1}  |  potential, not actual generation",
        fontsize=11.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    png = os.path.join(OUT, f"diagnostic_rep_cell_{lon:.2f}E_{lat:.2f}N.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")
    print(f"example cell used: lat={lat:.2f} (nearest to {REP_LAT}), "
          f"lon={lon:.2f} (nearest to {REP_LON})")


if __name__ == "__main__":
    main()
