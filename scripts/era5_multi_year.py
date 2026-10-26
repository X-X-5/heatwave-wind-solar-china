# -*- coding: utf-8 -*-
"""
Multi-year (2011-2022) per-year ERA5 wind/solar potential + CN05.1 label pipeline.

Processes ONE year at a time (memory control: never loads 12 years of hourly
data at once). For a given --year Y it runs, in order:
  hourly  -> Faiman hourly potential (+ linear cell-temp sensitivity variant)
  daily   -> daily aggregation under candidate boundaries A and B
  label   -> CN05.1 daily heatwave classification (merged def, gap<2), 153 d
  align   -> exact-coordinate label join onto the daily ERA5 grid

All formulas/parameters are copied VERBATIM from the verified 2022 pilot scripts
(no model change):
  RHO=1.225, Faiman U0=25.0/U1=6.84, gamma=-0.005, T_STC=25.0
  tair=t2m-273.15; wind=sqrt(u^2+v^2); ghi=ssrd/3600; wpd=0.5*RHO*wind100^3;
  tcell=tair+ghi/(U0+U1*wind10); pr=1+gamma*(tcell-25); pvpot=pr*ghi
  linear: tcell_lin=4.3+0.943*tair+0.028*ghi-1.528*wind10; pvpot_lin=pr_lin*ghi
  CN05.1: calendar-day P90 (1981-2010) reused unchanged; merged def gap<2.

Reuses the unchanged libraries via import:
  era5_daily_window.WINDOWS / point_mask / accum_mask
  hw_lib.slot_index   hw_merge_lib (find_qualified_segments / merge / classify)
  _nc_guard.write_guarded

2022 inputs are read from data_era5/ (verified pilot, reused); 2011-2021 inputs
from data_era5_multi/. Outputs go to results_multi/ (new directory; the 2022
pilot files in results_hourly//results_daily//results_merge/ are never touched).

CN05.1 day boundary remains UNCONFIRMED; boundaries A/B both kept.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded                    # noqa: E402
import hw_lib as hw                                    # noqa: E402
import hw_merge_lib as m                               # noqa: E402
from era5_daily_window import WINDOWS, point_mask, accum_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRS = [os.path.join(ROOT, "data_era5"),          # 2022 verified pilot
             os.path.join(ROOT, "data_era5_multi")]    # 2011-2021 new downloads
OUT = os.path.join(ROOT, "results_multi")
CN05_NC = os.path.join(ROOT, "data", "CN05.1_Tmax_1961-2022_daily_025x025.nc")
THR_NC = os.path.join(ROOT, "results_fixed", "hw_threshold_1981-2010.nc")

INST_MEMBER = "data_stream-oper_stepType-instant.nc"
ACC_MEMBER = "data_stream-oper_stepType-accum.nc"
META = {"number", "expver"}

# --- verified model parameters (verbatim) ------------------------------------
RHO = 1.225
FAIMAN_U0 = 25.0
FAIMAN_U1 = 6.84
GAMMA = -0.005
T_STC = 25.0
C1, C2, C3, C4 = 4.3, 0.943, 0.028, -1.528
HOUR_LEN_H = 1.0
KWH = 1000.0

REGION = dict(LAT0=30.0, LAT1=33.0, LON0=112.0, LON1=121.0)


def pick(ds, candidates):
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    return None


def fmt_t(t):
    return str(np.datetime_as_string(t, unit="m")).replace("T", " ")


# --------------------------------------------------------------------------- #
# hourly stage
# --------------------------------------------------------------------------- #
def target_span(year):
    return (np.datetime64(f"{year}-04-30T00"),
            np.datetime64(f"{year}-10-02T00"))          # [start, end) = 3720 h


def load_block_merged(d):
    inst = xr.open_dataset(os.path.join(d, INST_MEMBER))
    acc = xr.open_dataset(os.path.join(d, ACC_MEMBER))
    tname = pick(inst, ["valid_time", "time"])
    latname = pick(inst, ["latitude", "lat"])
    lonname = pick(inst, ["longitude", "lon"])
    keep = {tname, latname, lonname}
    ds = inst[["t2m", "u100", "v100", "u10", "v10"]]
    ds["ssrd"] = acc["ssrd"]
    for c in list(ds.coords):
        if c not in keep:
            ds = ds.reset_coords(c, drop=True)
    return ds, tname, latname, lonname


def collect_blocks(year):
    t0, t1 = target_span(year)
    blocks = []          # (tag, ds)
    tname = None
    for base in DATA_DIRS:
        for d in sorted(glob.glob(os.path.join(base, "unpacked_*"))):
            tag = os.path.basename(d)[len("unpacked_"):]
            inst = os.path.join(d, INST_MEMBER)
            acc = os.path.join(d, ACC_MEMBER)
            if not (os.path.exists(inst) and os.path.exists(acc)):
                continue
            ds, tn, _, _ = load_block_merged(d)
            tname = tname or tn
            t = np.asarray(ds[tn].values, dtype="datetime64[h]")
            if t.size == 0 or t[-1] < t0 or t[0] >= t1:
                ds.close()
                continue
            blocks.append((tag, ds))
    if not blocks:
        raise SystemExit(f"no ERA5 unpacked blocks found for year {year}")
    return blocks, tname


def build_hourly_faiman(year):
    t0, t1 = target_span(year)
    blocks, tname = collect_blocks(year)
    ds = xr.concat([b[1] for b in blocks], dim=tname).sortby(tname)
    union = np.sort(ds[tname].values.astype("datetime64[h]"))
    target = np.arange(t0, t1, dtype="datetime64[h]")
    if union.size != 3720 or len(np.unique(union)) != 3720:
        raise RuntimeError(f"{year}: concatenated union has {union.size} steps "
                           f"({len(np.unique(union))} unique), expected 3720")
    if not np.array_equal(union, target):
        raise RuntimeError(f"{year}: concatenated union != target "
                           f"{t0}..{t1} (contiguous 3720 h)")

    tair = ds["t2m"] - 273.15
    wind10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    wind100 = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2)
    ghi = ds["ssrd"] / 3600.0
    wpd = 0.5 * RHO * wind100 ** 3
    tcell = tair + ghi / (FAIMAN_U0 + FAIMAN_U1 * wind10)
    pr = 1.0 + GAMMA * (tcell - T_STC)
    pvpot = pr * ghi

    out = xr.Dataset({
        "tair": tair, "ghi": ghi, "wind100": wind100, "wind10": wind10,
        "wpd": wpd, "tcell": tcell, "performance_ratio": pr, "pvpot": pvpot,
    })
    for name, (units, long, sampling) in {
        "tair": ("degree_Celsius", "2 m air temperature (t2m - 273.15)", "instantaneous"),
        "ghi": ("W m-2", "Surface downward shortwave, ssrd/3600 (interval mean of the hour ending at valid_time)", "accumulated -> mean"),
        "wind100": ("m s-1", "Wind speed at 100 m, sqrt(u100^2+v100^2)", "instantaneous"),
        "wind10": ("m s-1", "Wind speed at 10 m, sqrt(u10^2+v10^2)", "instantaneous"),
        "wpd": ("W m-2", "Wind power density 0.5*rho*wind100^3 (RESOURCE, not electrical power)", "derived"),
        "tcell": ("degree_Celsius", "PV cell temperature, Faiman (2008): T_a + GHI/(U0+U1*wind10)", "derived"),
        "performance_ratio": ("1", "Temperature-derated PR = 1 + gamma*(tcell - 25)", "derived"),
        "pvpot": ("W m-2", "Simplified PV potential index = PR*GHI (potential, NOT generation / NOT system CF)", "derived"),
    }.items():
        out[name].attrs = {"units": units, "long_name": long, "sampling": sampling}
    out["wind100"].attrs["height_m"] = 100
    out["wind10"].attrs["height_m"] = 10
    out["wpd"].attrs["air_density_kg_m3"] = RHO
    out["tcell"].attrs["model"] = "Faiman (2008), doi:10.1002/pip.813"
    out["tcell"].attrs["U0_W_m2_K"] = FAIMAN_U0
    out["tcell"].attrs["U1_W_m2_K_per_ms"] = FAIMAN_U1
    out["performance_ratio"].attrs["gamma_per_degC"] = GAMMA
    out["performance_ratio"].attrs["T_STC_degC"] = T_STC
    out.attrs = {
        "title": f"ERA5 hourly wind/solar potential (Faiman), warm season {year}",
        "source": "ERA5 reanalysis-era5-single-levels, 0.25 deg, unpacked_* blocks",
        "region": "30-33 N, 112-121 E",
        "year": year,
        "config_signature": json.dumps({
            "year": year, "scheme": "Faiman (2008)",
            "RHO_kg_m3": RHO, "FAIMAN_U0_W_m2_K": FAIMAN_U0,
            "FAIMAN_U1_W_m2_K_per_ms": FAIMAN_U1, "GAMMA_per_degC": GAMMA,
            "T_STC_degC": T_STC,
        }, sort_keys=True),
    }
    return out


def build_linear(faim):
    tair = faim["tair"]
    ghi = faim["ghi"]
    w10 = faim["wind10"]
    tcell_lin = C1 + C2 * tair + C3 * ghi + C4 * w10
    pr_lin = 1.0 + GAMMA * (tcell_lin - T_STC)
    pvpot_lin = pr_lin * ghi
    lin = xr.Dataset({
        "tair": tair, "ghi": ghi, "wind10": w10,
        "tcell_lin": tcell_lin, "performance_ratio_lin": pr_lin, "pvpot_lin": pvpot_lin,
    })
    lin["tcell_lin"].attrs = {"units": "degree_Celsius",
                              "long_name": "Linear cell temp 4.3+0.943*Tair+0.028*GHI-1.528*V10 (Jerez 2015 / Chenni 2007)"}
    lin["performance_ratio_lin"].attrs = {"units": "1", "long_name": "1 + gamma*(tcell_lin - 25)"}
    lin["pvpot_lin"].attrs = {"units": "W m-2", "long_name": "pr_lin * ghi"}
    lin.attrs = {
        "title": "ERA5 hourly PV potential - LINEAR cell-temperature sensitivity variant",
        "cell_temp_model": "T_cell = 4.3 + 0.943*Tair + 0.028*GHI - 1.528*V10",
        "config_signature": json.dumps({
            "year": faim.attrs.get("year"), "scheme": "linear cell-temperature",
            "C1_degC": C1, "C2": C2, "C3_degC_m2_W": C3, "C4_degC_s_m": C4,
            "GAMMA_per_degC": GAMMA, "T_STC_degC": T_STC,
        }, sort_keys=True),
    }
    return lin


def stage_hourly(year):
    faim = build_hourly_faiman(year)
    enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
           for v in faim.data_vars}
    faim_nc = os.path.join(OUT, f"era5_hourly_potential_{year}0430_{year}1001.nc")
    write_guarded(faim, faim_nc, enc)

    lin = build_linear(faim)
    lenc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
            for v in lin.data_vars}
    lin_nc = os.path.join(OUT, f"era5_hourly_potential_tcell_linear_{year}0430_{year}1001.nc")
    write_guarded(lin, lin_nc, lenc)
    return faim_nc, lin_nc


# --------------------------------------------------------------------------- #
# daily stage
# --------------------------------------------------------------------------- #
def stage_daily(year, faim_nc, lin_nc):
    faim = xr.open_dataset(faim_nc).sortby("latitude")   # ascending 30..33
    lin = xr.open_dataset(lin_nc).sortby("latitude")
    time = faim["valid_time"].values.astype("datetime64[h]")
    lat = faim["latitude"].values
    lon = faim["longitude"].values
    nlat, nlon = lat.size, lon.size

    tair = faim["tair"].values
    wind100 = faim["wind100"].values
    wpd = faim["wpd"].values
    ghi = faim["ghi"].values
    pvpot = faim["pvpot"].values
    pvpot_lin = lin["pvpot_lin"].values

    start = f"{year}-05-01"
    end = f"{year}-10-01"
    dates = np.arange(start, end, dtype="datetime64[D]")
    nday = dates.size
    assert nday == 153

    results = {}
    for key, (label, wfn) in WINDOWS.items():
        s0, _ = wfn(dates[0])
        _, e_last = wfn(dates[-1])
        tair_mean = np.full((nday, nlat, nlon), np.nan)
        tair_max = np.full((nday, nlat, nlon), np.nan)
        wind100_mean = np.full((nday, nlat, nlon), np.nan)
        wpd_mean = np.full((nday, nlat, nlon), np.nan)
        ghi_sum = np.full((nday, nlat, nlon), np.nan)
        pvpot_mean = np.full((nday, nlat, nlon), np.nan)
        pvpot_sum = np.full((nday, nlat, nlon), np.nan)
        pvpot_lin_mean = np.full((nday, nlat, nlon), np.nan)
        pvpot_lin_sum = np.full((nday, nlat, nlon), np.nan)
        valid_hours = np.zeros((nday, nlat, nlon), dtype=np.int16)

        for i, D in enumerate(dates):
            s, e = wfn(D)
            pm = point_mask(time, s, e)
            am = accum_mask(time, s, e)
            assert int(pm.sum()) == 24 and int(am.sum()) == 24, (D, pm.sum(), am.sum())
            p_tair = tair[pm]; p_wind = wind100[pm]; p_wpd = wpd[pm]
            a_ghi = ghi[am]; a_pv = pvpot[am]; a_pvl = pvpot_lin[am]
            pv = np.isfinite(p_tair) & np.isfinite(p_wind) & np.isfinite(p_wpd)
            av = np.isfinite(a_ghi) & np.isfinite(a_pv) & np.isfinite(a_pvl)
            vh = np.minimum(pv.sum(0), av.sum(0))
            valid_hours[i] = vh
            complete = vh == 24
            with np.errstate(invalid="ignore"):
                tair_mean[i] = np.where(complete, np.nanmean(p_tair, 0), np.nan)
                tair_max[i] = np.where(complete, np.nanmax(p_tair, 0), np.nan)
                wind100_mean[i] = np.where(complete, np.nanmean(p_wind, 0), np.nan)
                wpd_mean[i] = np.where(complete, np.nanmean(p_wpd, 0), np.nan)
                ghi_sum[i] = np.where(complete, np.nansum(a_ghi, 0) * HOUR_LEN_H / KWH, np.nan)
                pvpot_mean[i] = np.where(complete, np.nanmean(a_pv, 0), np.nan)
                pvpot_sum[i] = np.where(complete, np.nansum(a_pv, 0) * HOUR_LEN_H / KWH, np.nan)
                pvpot_lin_mean[i] = np.where(complete, np.nanmean(a_pvl, 0), np.nan)
                pvpot_lin_sum[i] = np.where(complete, np.nansum(a_pvl, 0) * HOUR_LEN_H / KWH, np.nan)

        ds = xr.Dataset({
            "tair_mean": (("date", "latitude", "longitude"), tair_mean),
            "tair_max": (("date", "latitude", "longitude"), tair_max),
            "wind100_mean": (("date", "latitude", "longitude"), wind100_mean),
            "wpd_mean": (("date", "latitude", "longitude"), wpd_mean),
            "ghi_sum": (("date", "latitude", "longitude"), ghi_sum),
            "pvpot_mean": (("date", "latitude", "longitude"), pvpot_mean),
            "pvpot_sum": (("date", "latitude", "longitude"), pvpot_sum),
            "pvpot_lin_mean": (("date", "latitude", "longitude"), pvpot_lin_mean),
            "pvpot_lin_sum": (("date", "latitude", "longitude"), pvpot_lin_sum),
            "valid_hours": (("date", "latitude", "longitude"), valid_hours),
        }, coords={"date": dates.astype("datetime64[D]"),
                   "latitude": lat, "longitude": lon})
        ds.attrs = {
            "title": f"ERA5 daily wind/solar potential {year} warm season, boundary {key}",
            "day_boundary": label, "year": year,
            "cn05.1_day_boundary": "UNCONFIRMED (station convention is 20-20 BJT, not yet verified for CN05.1 itself)",
            "config_signature": json.dumps({
                "year": year, "boundary_key": key,
                "start": start, "end": end,
                "valid_hours_requirement": 24,
            }, sort_keys=True),
        }
        enc = {v: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
               for v in ds.data_vars if v != "valid_hours"}
        enc["valid_hours"] = {"zlib": True, "complevel": 4, "dtype": "int16"}
        out_nc = os.path.join(OUT, f"era5_daily_potential_{year}0501_{year}0930_{key}.nc")
        write_guarded(ds, out_nc, enc)
        results[key] = out_nc
        print(f"[daily {key}] {year}: valid_hours==24 in "
              f"{int((valid_hours == 24).sum())}/{valid_hours.size} gridpoint-days")
    return results["A"], results["B"]


# --------------------------------------------------------------------------- #
# label stage (CN05.1 merged def, gap<2) + align
# --------------------------------------------------------------------------- #
def stage_label(year):
    nc = Dataset(CN05_NC, "r")
    var = nc.variables["tmax"]
    lat = nc.variables["lat"][:]
    lon = nc.variables["lon"][:]
    t = nc.variables["time"][:]
    origin = pd.Timestamp("1961-01-01 00:00:00")
    dates = origin + pd.to_timedelta(t, unit="h")
    year_a = dates.year.values.astype(np.int64)
    month_a = dates.month.values.astype(np.int64)

    LAT0, LAT1 = REGION["LAT0"], REGION["LAT1"]
    LON0, LON1 = REGION["LON0"], REGION["LON1"]
    i0 = int(np.argmin(np.abs(lat - LAT0)))
    i1 = int(np.argmin(np.abs(lat - LAT1))) + 1
    j0 = int(np.argmin(np.abs(lon - LON0)))
    j1 = int(np.argmin(np.abs(lon - LON1))) + 1
    lat_r = lat[i0:i1]
    lon_r = lon[j0:j1]

    season = (month_a >= 5) & (month_a <= 9)
    seas_idx = np.where(season & (year_a == year))[0]
    assert len(seas_idx) == 153, f"expected 153 season days, got {len(seas_idx)}"
    g0, g1 = int(seas_idx[0]), int(seas_idx[-1]) + 1
    d0 = pd.Timestamp(f"{year}-05-01")
    d_season = [d0 + pd.Timedelta(days=d) for d in range(153)]
    slot_season = np.array([(dt.month, dt.day) for dt in d_season])

    blk = var[g0:g1, i0:i1, j0:j1]
    tmax = np.ma.filled(blk.astype(np.float32), np.float32(np.nan))
    nc.close()

    thr_ds = xr.open_dataset(THR_NC)
    thr366 = thr_ds["threshold"].values.astype(np.float64)
    thr_ds.close()
    thr_region = thr366[:, i0:i1, j0:j1]

    slots = np.array([int(hw.slot_index(np.array([mo]), np.array([dy]))[0])
                      for (mo, dy) in slot_season], dtype=np.int64)

    nlat, nlon = len(lat_r), len(lon_r)
    labels_all = np.full((153, nlat, nlon), m.LABEL_MISSING, dtype=np.int8)
    thr_all = np.full((153, nlat, nlon), np.nan, dtype=np.float32)
    for ii in range(nlat):
        for jj in range(nlon):
            col = tmax[:, ii, jj]
            if not np.isfinite(col).any():
                continue
            th = thr_region[slots, ii, jj]
            finite = np.isfinite(col) & np.isfinite(th)
            if not finite.any():
                continue
            hot = finite & (col > th)
            cool = finite & (col <= th)
            missing = ~finite
            segs = m.find_qualified_segments(hot)
            procs1 = m.merge_qualified_segments(segs, cool, missing, max_gap=1)
            labels = m.classify_days(hot, cool, missing, segs, procs1)
            labels_all[:, ii, jj] = labels
            thr_all[:, ii, jj] = th.astype(np.float32)

    ds = xr.Dataset({
        "label": (("date", "lat", "lon"), labels_all),
        "tmax": (("date", "lat", "lon"), tmax),
        "threshold": (("date", "lat", "lon"), thr_all),
    }, coords={"date": pd.to_datetime(d_season), "lat": lat_r, "lon": lon_r})
    ds["label"].attrs = {"long_name": "daily heatwave classification (merged definition, gap<2)",
                         "codes": "0=missing;1=cool;2=isolated exceedance;3=heatwave;4=gap day"}
    ds["tmax"].attrs = {"units": "degC", "note": "NaN = missing"}
    ds["threshold"].attrs = {"units": "degC", "long_name": "calendar-day P90 threshold"}
    ds.attrs = {
        "title": f"CN05.1 Tmax {year} daily heatwave classification (merged, gap<2)",
        "baseline_period": "1981-2010", "percentile": 90,
        "min_run_days": 3, "season": "May-Sep",
        "merge_rule": "adjacent >=3d segments merge iff complete gap < 2 days",
        "config_signature": json.dumps({"year": year, "merge_rule": "gap<2",
                                        "baseline": "1981-2010", "percentile": 90},
                                       sort_keys=True),
    }
    enc = {"label": {"zlib": True, "complevel": 4, "dtype": "int8"},
           "tmax": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
           "threshold": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
    raw_nc = os.path.join(OUT, f"hw_{year}_daily_labels.nc")
    write_guarded(ds, raw_nc, enc)
    return raw_nc


def stage_align(year, raw_nc, daily_a_nc):
    lab = xr.open_dataset(raw_nc)
    ref = xr.open_dataset(daily_a_nc)
    lab_rn = lab.rename({"lat": "latitude", "lon": "longitude"})

    coord_ok = True
    for lc in ("date", "latitude", "longitude"):
        if not np.array_equal(np.sort(lab_rn[lc].values), np.sort(ref[lc].values)):
            coord_ok = False
    if not coord_ok:
        raise SystemExit(f"{year}: label coordinate set does not match daily grid")

    aligned = lab_rn.reindex(date=ref["date"].values,
                             latitude=ref["latitude"].values,
                             longitude=ref["longitude"].values)
    val_ok = True
    for v in ("label", "tmax", "threshold"):
        a = lab[v].values
        b = aligned[v].values
        as_ = np.sort(a, axis=None)
        bs_ = np.sort(b, axis=None)
        nan_a = np.isnan(as_); nan_b = np.isnan(bs_)
        if int(nan_a.sum()) != int(nan_b.sum()):
            val_ok = False
            continue
        val_ok &= bool(np.array_equal(as_[~nan_a], bs_[~nan_b]))
    codes = sorted(int(c) for c in np.unique(aligned["label"].values))
    codes_ok = set(codes) <= {0, 1, 2, 3, 4}
    if not (val_ok and codes_ok):
        raise SystemExit(f"{year}: label align verification failed "
                         f"(values_ok={val_ok}, codes={codes})")

    aligned.attrs = {
        "title": "CN05.1 heatwave daily labels aligned to the ERA5 daily grid",
        "join_method": "exact coordinate match + reorder (no interpolation)",
        "label_codes": "0=missing;1=ordinary;2=isolated exceedance;3=heatwave;4=gap day",
        "label_0_meaning": "missing/undecidable (preserved, NOT replaced by ordinary day)",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "year": year,
        "config_signature": json.dumps({
            "year": year, "join_method": "coordinate match + reorder",
            "cn05.1_day_boundary": "UNCONFIRMED",
        }, sort_keys=True),
    }
    enc = {"label": {"zlib": True, "complevel": 4},
           "tmax": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
           "threshold": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
    aligned_nc = os.path.join(OUT, f"cn05_labels_{year}0501_{year}0930_aligned.nc")
    write_guarded(aligned, aligned_nc, enc)
    print(f"[align {year}] label codes present = {codes}")
    return aligned_nc


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--stage", default="all",
                    choices=["all", "hourly", "daily", "label", "align"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    year = args.year

    faim_nc = os.path.join(OUT, f"era5_hourly_potential_{year}0430_{year}1001.nc")
    lin_nc = os.path.join(OUT, f"era5_hourly_potential_tcell_linear_{year}0430_{year}1001.nc")
    daily_a = os.path.join(OUT, f"era5_daily_potential_{year}0501_{year}0930_A.nc")
    raw_nc = os.path.join(OUT, f"hw_{year}_daily_labels.nc")

    if args.stage in ("all", "hourly"):
        print(f"[{year}] hourly potential (Faiman + linear)")
        stage_hourly(year)
    if args.stage in ("all", "daily"):
        if not (os.path.exists(faim_nc) and os.path.exists(lin_nc)):
            raise SystemExit(f"{year}: hourly files missing; run --stage hourly first")
        print(f"[{year}] daily aggregation (A/B)")
        daily_a, _ = stage_daily(year, faim_nc, lin_nc)
    if args.stage in ("all", "label"):
        print(f"[{year}] CN05.1 heatwave labels (merged gap<2)")
        raw_nc = stage_label(year)
    if args.stage in ("all", "align"):
        if not os.path.exists(daily_a):
            raise SystemExit(f"{year}: daily A missing; run --stage daily first")
        if not os.path.exists(raw_nc):
            raw_nc = stage_label(year)
        print(f"[{year}] align labels to daily grid")
        stage_align(year, raw_nc, daily_a)

    print(f"[{year}] DONE (stage={args.stage})")


if __name__ == "__main__":
    main()
