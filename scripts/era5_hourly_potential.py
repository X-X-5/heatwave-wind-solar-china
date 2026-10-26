# -*- coding: utf-8 -*-
"""
Hourly wind/solar potential pilot from the unpacked ERA5 blocks.

Reads data_era5/unpacked_<tag>/{instant,accum}.nc (already integrity-checked),
keeps the UTC `valid_time` axis (full warm season: 2022-04-30 00:00 .. 2022-10-01 23:00, 3720 h),
and computes the FIRST-ROUND potential fields:

    wind100 = sqrt(u100^2 + v100^2)         [m s-1]   instantaneous @ 100 m
    wind10  = sqrt(u10^2  + v10^2)          [m s-1]   instantaneous @ 10 m
    tair    = t2m - 273.15                  [degC]    instantaneous
    ghi     = ssrd / 3600                   [W m-2]   hour-END interval mean
    wpd     = 0.5 * rho * wind100^3         [W m-2]   rho = 1.225 kg m-3
    tcell   = tair + ghi / (U0 + U1*wind10) [degC]    Faiman (2008)
    pr      = 1 + gamma * (tcell - 25)      [1]       gamma = -0.005 degC^-1
    pvpot   = pr * ghi                      [W m-2]   simplified potential index

WPD is a RESOURCE density, not electrical power: no wind capacity factor is
computed this round (no WPD/WPD_rated, no ad-hoc cubic power curve).

pvpot is a SIMPLIFIED potential index (horizontal-plane GHI x temperature-derated
performance ratio), NOT actual PV generation and NOT a system capacity factor.

Missing values are kept as NaN (none present in this download); no spatial
interpolation, no gap filling, no clipping of anomalies. Night GHI = 0 => pvpot = 0
by construction (pr * 0 = 0), so no explicit masking is applied.

Time correspondence approximation: t2m / u-v winds are instantaneous samples at the
`valid_time` stamp; ssrd is accumulated over the hour ENDING at that stamp, so
ghi = ssrd/3600 is the interval mean of [t-1h, t]. All fields are therefore paired
at the same `valid_time` label (end-of-hour), which is the standard ECMWF convention.

Write a single independent NetCDF into results_hourly/. Never overwrites the
downloaded files.
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_era5")
OUT = os.path.join(ROOT, "results_hourly")

# --- verified parameters / sources -------------------------------------------------
RHO = 1.225                # kg m-3, standard air density (15 C, 101325 Pa, sea level)
                           #   IEC 61400-12-1 / ISO 2533
FAIMAN_U0 = 25.0           # W m-2 K-1      Faiman (2008), doi:10.1002/pip.813
FAIMAN_U1 = 6.84           # W m-2 K-1 (m/s)-1   (open-rack, pvlib / IEC 61853 default)
GAMMA = -0.005             # degC^-1        typical c-Si power temp coefficient
T_STC = 25.0               # degC           STC cell temperature

# full warm-season target for the pre-concat validation (never silently drop steps)
TARGET_START = np.datetime64("2022-04-30T00")
TARGET_END_EXCL = np.datetime64("2022-10-02T00")
TARGET_HOURS = 3720


def pick(ds, candidates):
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    return None


def fmt_t(t):
    return str(np.datetime_as_string(t, unit="m")).replace("T", " ")


def validate_time_axis(t, ctx):
    t = np.asarray(t).astype("datetime64[h]")
    if len(t) != len(np.unique(t)):
        raise RuntimeError(f"{ctx}: duplicate timestamps "
                           f"({len(t)} values, {len(np.unique(t))} unique)")
    if len(t) > 1:
        d = np.diff(t).astype("timedelta64[h]").astype(np.int64)
        if not np.all(d == 1):
            raise RuntimeError(f"{ctx}: time axis not sorted/contiguous hourly")
    return t


def load_block(tag):
    """Load one block's needed variables, merged onto the instant grid/time axis."""
    d = os.path.join(DATA, f"unpacked_{tag}")
    inst = xr.open_dataset(os.path.join(d, "data_stream-oper_stepType-instant.nc"))
    acc = xr.open_dataset(os.path.join(d, "data_stream-oper_stepType-accum.nc"))
    tname = pick(inst, ["valid_time", "time"])
    latname = pick(inst, ["latitude", "lat"])
    lonname = pick(inst, ["longitude", "lon"])
    keep = {tname, latname, lonname}

    ds = inst[["t2m", "u100", "v100", "u10", "v10"]]
    ds["ssrd"] = acc["ssrd"]
    # drop auxiliary coords (expver, number, ...) so blocks concatenate cleanly
    for c in list(ds.coords):
        if c not in keep:
            ds = ds.reset_coords(c, drop=True)
    return ds, tname, latname, lonname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output path override (default: auto-derived from span)")
    args = ap.parse_args()

    tags = [os.path.basename(p)[len("unpacked_"):]
            for p in sorted(glob.glob(os.path.join(DATA, "unpacked_*")))]
    if not tags:
        raise SystemExit("no unpacked_* blocks found under data_era5")
    print("blocks:", tags)

    blocks = [load_block(t) for t in tags]
    tname = blocks[0][1]
    latname = blocks[0][2]
    lonname = blocks[0][3]

    # per-block time sanity (sorted + unique) BEFORE any concatenation
    for tag, b in zip(tags, blocks):
        validate_time_axis(b[0][tname].values, f"block {tag}")

    ds = xr.concat([b[0] for b in blocks], dim=tname).sortby(tname)
    n = ds.sizes[tname]

    # full warm-season coverage (unique, sorted, contiguous, exactly 3720 h)
    union = np.sort(ds[tname].values.astype("datetime64[h]"))
    target = np.arange(TARGET_START, TARGET_END_EXCL, dtype="datetime64[h]")
    if union.size != TARGET_HOURS or len(np.unique(union)) != TARGET_HOURS:
        raise RuntimeError(f"concatenated union has {union.size} steps "
                           f"({len(np.unique(union))} unique), expected {TARGET_HOURS}")
    if not np.array_equal(union, target):
        raise RuntimeError("concatenated union does not equal the full warm-season "
                           f"target {TARGET_START}..{TARGET_END_EXCL}")

    print(f"concatenated: {n} time steps, "
          f"{fmt_t(ds[tname].values[0])} .. {fmt_t(ds[tname].values[-1])} "
          f"(validated unique/sorted/contiguous, {TARGET_HOURS} h)")

    # --- compute ----------------------------------------------------------------
    tair = ds["t2m"] - 273.15                                    # degC
    wind10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)            # m s-1 @ 10 m
    wind100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)         # m s-1 @ 100 m
    ghi = ds["ssrd"] / 3600.0                                    # W m-2 (hour mean)
    wpd = 0.5 * RHO * wind100 ** 3                               # W m-2 (resource)
    tcell = tair + ghi / (FAIMAN_U0 + FAIMAN_U1 * wind10)        # degC (Faiman 2008)
    pr = 1.0 + GAMMA * (tcell - T_STC)                           # 1
    pvpot = pr * ghi                                             # W m-2 (night = 0)

    out = xr.Dataset(
        {
            "tair": tair,
            "ghi": ghi,
            "wind100": wind100,
            "wind10": wind10,
            "wpd": wpd,
            "tcell": tcell,
            "performance_ratio": pr,
            "pvpot": pvpot,
        }
    )

    # --- metadata ---------------------------------------------------------------
    for name, attrs in {
        "tair": ("degree_Celsius", "2 m air temperature (t2m - 273.15)", "instantaneous"),
        "ghi": ("W m-2", "Surface downward shortwave, ssrd/3600 (interval mean of the hour ending at valid_time)", "accumulated -> mean"),
        "wind100": ("m s-1", "Wind speed at 100 m, sqrt(u100^2+v100^2)", "instantaneous"),
        "wind10": ("m s-1", "Wind speed at 10 m, sqrt(u10^2+v10^2)", "instantaneous"),
        "wpd": ("W m-2", "Wind power density 0.5*rho*wind100^3 (RESOURCE, not electrical power)", "derived"),
        "tcell": ("degree_Celsius", "PV cell temperature, Faiman (2008): T_a + GHI/(U0+U1*wind10)", "derived"),
        "performance_ratio": ("1", "Temperature-derated PR = 1 + gamma*(tcell - 25)", "derived"),
        "pvpot": ("W m-2", "Simplified PV potential index = PR*GHI (potential, NOT generation / NOT system CF)", "derived"),
    }.items():
        out[name].attrs = {"units": attrs[0], "long_name": attrs[1],
                           "sampling": attrs[2]}
        if name in ("wind100", "wind10"):
            out[name].attrs["height_m"] = 100 if name == "wind100" else 10
        if name == "wpd":
            out[name].attrs["air_density_kg_m3"] = RHO
            out[name].attrs["air_density_source"] = "IEC 61400-12-1 / ISO 2533 (15 C, 101325 Pa)"
        if name == "tcell":
            out[name].attrs["model"] = "Faiman (2008), doi:10.1002/pip.813"
            out[name].attrs["U0_W_m2_K"] = FAIMAN_U0
            out[name].attrs["U1_W_m2_K_per_ms"] = FAIMAN_U1
            out[name].attrs["wind_height_m"] = 10
        if name in ("performance_ratio", "pvpot"):
            out[name].attrs["gamma_per_degC"] = GAMMA
            out[name].attrs["T_STC_degC"] = T_STC

    out.attrs = {
        "title": "ERA5 hourly wind/solar potential pilot (first round)",
        "source": "data_era5/unpacked_*/*.nc (reanalysis-era5-single-levels, 0.25 deg)",
        "region": "30-33 N, 112-121 E",
        "time_axis": f"UTC valid_time, {fmt_t(ds[tname].values[0])} .. {fmt_t(ds[tname].values[-1])} ({n} h)",
        "time_correspondence": (
            "t2m/u-v winds are instantaneous at valid_time; ssrd is accumulated over "
            "the hour ENDING at valid_time, so all fields are paired at the same "
            "end-of-hour label (ECMWF convention)."
        ),
        "wind_capacity_factor": "NOT computed (WPD is a resource density, not power)",
        "pv_potential_note": "pvpot = PR*GHI is a SIMPLIFIED potential index, not actual generation and not a system capacity factor",
        "missing_handling": "NaN preserved; no interpolation, no gap filling, no clipping",
        "air_density_kg_m3": RHO,
        "config_signature": json.dumps({
            "scheme": "Faiman (2008)",
            "RHO_kg_m3": RHO,
            "FAIMAN_U0_W_m2_K": FAIMAN_U0,
            "FAIMAN_U1_W_m2_K_per_ms": FAIMAN_U1,
            "GAMMA_per_degC": GAMMA,
            "T_STC_degC": T_STC,
            "formulas": ("tair=t2m-273.15; wind=sqrt(u^2+v^2); ghi=ssrd/3600; "
                         "wpd=0.5*RHO*wind100^3; tcell=tair+ghi/(U0+U1*wind10); "
                         "pr=1+GAMMA*(tcell-T_STC); pvpot=pr*ghi"),
        }, sort_keys=True),
    }

    # --- report to stdout -------------------------------------------------------
    print("=" * 72)
    print(f"output variable ranges (whole domain, {n} h):")
    for v in out.data_vars:
        a = out[v].values
        finite = np.isfinite(a)
        nan = int(a.size - finite.sum())
        lo = float(np.nanmin(a)) if finite.any() else float("nan")
        hi = float(np.nanmax(a)) if finite.any() else float("nan")
        print(f"  {v:18s} nan={nan:5d}/{a.size:7d}  min={lo:12.5g}  max={hi:12.5g} "
              f"[{out[v].attrs.get('units','')}]")
    ghi_neg = int((out["ghi"].values < 0).sum())
    print(f"  ghi negative grid-cell-hours: {ghi_neg} (should be 0; ssrd is "
          f"downward => non-negative)")
    print("=" * 72)

    # --- write ------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for v in out.data_vars}
    t0 = np.datetime_as_string(ds[tname].values[0].astype("datetime64[D]"),
                               unit="D").replace("-", "")
    t1 = np.datetime_as_string(ds[tname].values[-1].astype("datetime64[D]"),
                               unit="D").replace("-", "")
    out_nc = args.out or os.path.join(OUT, f"era5_hourly_potential_{t0}_{t1}.nc")
    write_guarded(out, out_nc, enc)


if __name__ == "__main__":
    main()
