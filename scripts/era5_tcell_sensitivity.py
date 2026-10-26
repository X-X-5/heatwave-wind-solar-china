# -*- coding: utf-8 -*-
"""
Model SENSITIVITY comparison (not a validation against measurements):

    Faiman (2008)  vs  linear cell-temperature model (Jerez et al. 2015 /
    Chenni et al. 2007).

Reads the existing Faiman trial NetCDF (results_hourly/era5_hourly_potential_*.nc),
which already holds tair / ghi / wind10 / tcell / pvpot. Keeps tair, ghi, wind10,
gamma, and the time pairing IDENTICAL; changes ONLY the temperature model:

    tcell_lin = 4.3 + 0.943*tair + 0.028*ghi - 1.528*wind10   [degC]
    pr_lin    = 1 + gamma*(tcell_lin - 25)                     [1]
    pvpot_lin = pr_lin * ghi                                    [W m-2]

Writes a SEPARATE NetCDF (does not overwrite the Faiman file), then reports a
limited comparison (daytime dT; August cumulative relative difference; per-cell
distribution; representative-cell curves). No winner is chosen.
"""
import os
import sys
import json
import argparse

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_hourly")

C1, C2, C3, C4 = 4.3, 0.943, 0.028, -1.528
GAMMA = -0.005
T_STC = 25.0
HOUR_S = 3600.0            # each ERA5 step is 1 h

REP_LAT, REP_LON = 31.75, 114.50
PLOT_DAYS = slice("2022-08-08", "2022-08-12")   # "a few consecutive days"

PLOT_PNG = os.path.join(OUT, "sensitivity_tcell_rep_cell_114.50E_31.75N.png")


def pct(a, p):
    return float(np.nanpercentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faim", required=True,
                    help="path to the Faiman hourly NetCDF (explicit, no glob fallback)")
    ap.add_argument("--out", default=None,
                    help="output path override (default: derived from --faim basename)")
    ap.add_argument("--no-diag", action="store_true",
                    help="skip the diagnostic comparison + plot (NetCDF only)")
    args = ap.parse_args()

    if not os.path.exists(args.faim):
        raise SystemExit(f"Faiman input not found: {args.faim}")

    base = os.path.basename(args.faim)
    lin_name = base.replace("era5_hourly_potential_",
                            "era5_hourly_potential_tcell_linear_", 1)
    lin_nc = args.out or os.path.join(OUT, lin_name)

    ds = xr.open_dataset(args.faim)

    tair = ds["tair"]
    ghi = ds["ghi"]
    w10 = ds["wind10"]

    # --- linear variant (same tair/ghi/wind10/gamma, only the temp model) ----
    tcell_lin = C1 + C2 * tair + C3 * ghi + C4 * w10
    pr_lin = 1.0 + GAMMA * (tcell_lin - T_STC)
    pvpot_lin = pr_lin * ghi

    lin = xr.Dataset(
        {
            "tair": tair,
            "ghi": ghi,
            "wind10": w10,
            "tcell_lin": tcell_lin,
            "performance_ratio_lin": pr_lin,
            "pvpot_lin": pvpot_lin,
        }
    )
    for name, units, long in [
        ("tair", "degree_Celsius", "2 m air temperature (identical to Faiman file)"),
        ("ghi", "W m-2", "ssrd/3600 hour-mean (identical to Faiman file)"),
        ("wind10", "m s-1", "10 m wind speed (identical to Faiman file)"),
        ("tcell_lin", "degree_Celsius", "Linear cell temp 4.3+0.943*Tair+0.028*GHI-1.528*V10 (Jerez 2015 / Chenni 2007)"),
        ("performance_ratio_lin", "1", "1 + gamma*(tcell_lin - 25)"),
        ("pvpot_lin", "W m-2", "pr_lin * ghi (simplified potential index, night = 0)"),
    ]:
        lin[name].attrs = {"units": units, "long_name": long}
    lin.attrs = {
        "title": "ERA5 hourly PV potential - LINEAR cell-temperature sensitivity variant",
        "note": "SENSITIVITY variant; identical tair/ghi/wind10/gamma/time to the Faiman trial. NOT a validation.",
        "cell_temp_model": "T_cell = 4.3 + 0.943*Tair + 0.028*GHI - 1.528*V10 (Jerez et al. 2015, after Chenni et al. 2007)",
        "C1_degC": C1, "C2": C2, "C3_degC_m2_W": C3, "C4_degC_s_m": C4,
        "gamma_per_degC": GAMMA, "T_STC_degC": T_STC,
        "reference_file": base,
        "config_signature": json.dumps({
            "scheme": "linear cell-temperature (Jerez 2015 / Chenni 2007)",
            "C1_degC": C1, "C2": C2, "C3_degC_m2_W": C3, "C4_degC_s_m": C4,
            "GAMMA_per_degC": GAMMA, "T_STC_degC": T_STC,
            "formulas": ("tcell_lin=C1+C2*tair+C3*ghi+C4*w10; "
                         "pr_lin=1+GAMMA*(tcell_lin-T_STC); pvpot_lin=pr_lin*ghi"),
        }, sort_keys=True),
    }
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for v in lin.data_vars}
    write_guarded(lin, lin_nc, enc)

    if args.no_diag:
        print("--no-diag: skipping diagnostic comparison + plot")
        ds.close()
        return

    # --- 1) daytime (GHI>0) cell-temperature difference ----------------------
    day = ghi > 0
    dT = (tcell_lin - ds["tcell"]).where(day).values
    dT = dT[np.isfinite(dT)]
    print("=" * 70)
    print("1) daytime (GHI>0)  dT = Tcell_lin - Tcell_faiman  [degC]")
    print(f"   n={dT.size}  mean={dT.mean():+.3f}  "
          f"P5={pct(dT,5):+.3f}  P50={pct(dT,50):+.3f}  P95={pct(dT,95):+.3f}")

    # --- 2) target-month (August) cumulative relative difference -------------
    aug = ds.sel(valid_time=slice("2022-08-01", "2022-08-31"))
    aug_lin = lin.sel(valid_time=slice("2022-08-01", "2022-08-31"))
    nh = aug.sizes["valid_time"]
    # per-cell cumulative over August, energy-equivalent: sum * 1 h  -> W h m-2
    E_f = aug["pvpot"].sum("valid_time") * 1.0
    E_l = aug_lin["pvpot_lin"].sum("valid_time") * 1.0
    Ef_mean = float(E_f.mean())
    El_mean = float(E_l.mean())
    rel_mean = (El_mean - Ef_mean) / Ef_mean * 100.0
    print("-" * 70)
    print(f"2) August 2022 cumulative (target month, {nh} h), domain-mean over 481 cells:")
    print(f"   E_mean (Faiman) = {Ef_mean/1000:.4f} kWh m-2   "
          f"E_mean (linear) = {El_mean/1000:.4f} kWh m-2")
    print(f"   relative diff (denominator = Faiman) = {rel_mean:+.3f} %")

    # --- 3) per-cell monthly cumulative relative-difference distribution -----
    rel_cell = ((E_l - E_f) / E_f * 100.0).values
    rel_cell = rel_cell[np.isfinite(rel_cell)]
    print("-" * 70)
    print("3) per-gridpoint August cumulative relative diff [%] (denominator = Faiman):")
    print(f"   n_cells={rel_cell.size}  min={rel_cell.min():+.3f}  "
          f"P5={pct(rel_cell,5):+.3f}  P25={pct(rel_cell,25):+.3f}  "
          f"P50={pct(rel_cell,50):+.3f}  P75={pct(rel_cell,75):+.3f}  "
          f"P95={pct(rel_cell,95):+.3f}  max={rel_cell.max():+.3f}")

    # linear-variant value ranges (for the report table)
    print("-" * 70)
    print("4) linear-variant ranges (whole window):")
    for v in ["tcell_lin", "pvpot_lin"]:
        a = lin[v].values
        fin = np.isfinite(a)
        print(f"   {v:20s} nan={int(a.size-fin.sum()):5d}  "
              f"min={float(np.nanmin(a)):.4g}  max={float(np.nanmax(a)):.4g}")

    # --- 5) representative-cell curves ---------------------------------------
    cell_f = aug.sel(latitude=REP_LAT, longitude=REP_LON, method="nearest")
    cell_l = aug_lin.sel(latitude=REP_LAT, longitude=REP_LON, method="nearest")
    lat = float(cell_f["latitude"]); lon = float(cell_f["longitude"])
    t = cell_f["valid_time"].sel(valid_time=PLOT_DAYS).values

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
    tc_f = cell_f["tcell"].sel(valid_time=PLOT_DAYS).values
    tc_l = cell_l["tcell_lin"].sel(valid_time=PLOT_DAYS).values
    pp_f = cell_f["pvpot"].sel(valid_time=PLOT_DAYS).values
    pp_l = cell_l["pvpot_lin"].sel(valid_time=PLOT_DAYS).values

    axes[0].plot(t, tc_f, lw=1.3, color="#1f6f9f", label="Faiman (2008)")
    axes[0].plot(t, tc_l, lw=1.3, ls="--", color="#d97706", label="linear (Jerez/Chenni)")
    axes[0].set_ylabel("T$_{cell}$ (°C)")
    axes[0].set_title("Cell temperature — two models", loc="left", fontsize=9.5)
    axes[0].grid(True, ls=":", lw=0.5, alpha=0.45)

    axes[1].plot(t, pp_f, lw=1.3, color="#1f6f9f", label="Faiman (2008)")
    axes[1].plot(t, pp_l, lw=1.3, ls="--", color="#d97706", label="linear (Jerez/Chenni)")
    axes[1].set_ylabel("PV potential (W m⁻²)")
    axes[1].set_title("PV potential — two models", loc="left", fontsize=9.5)
    axes[1].grid(True, ls=":", lw=0.5, alpha=0.45)

    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=24))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_xlabel("UTC (month-day)")
    fig.suptitle(
        f"Model sensitivity (not validation) — {lat:.2f}°N, {lon:.2f}°E, "
        f"{PLOT_DAYS.start}…{PLOT_DAYS.stop}\n"
        "same tair/GHI/V10/γ, different cell-temperature model", fontsize=10.5, y=0.99)
    fig.legend(
        handles=[plt.Line2D([0], [0], color="#1f6f9f", lw=1.3, label="Faiman (2008)"),
                 plt.Line2D([0], [0], color="#d97706", lw=1.3, ls="--",
                            label="linear (Jerez/Chenni)")],
        loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False,
        fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    fig.savefig(PLOT_PNG, dpi=150)
    print(f"wrote {PLOT_PNG}  (rep cell lat={lat:.2f} lon={lon:.2f})")


if __name__ == "__main__":
    main()
