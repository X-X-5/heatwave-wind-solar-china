# -*- coding: utf-8 -*-
"""
Verify the unpacked ERA5 blocks (structural + data sanity), NO energy calc,
NO daily aggregation.

Checks per block (data_era5/unpacked_<tag>/{instant,accum}.nc):
  * variables + units + dimensions (incl. number/expver metadata vars)
  * lat/lon ranges and counts
  * time axis: count, first/last, uniqueness, hourly contiguity
  * instant vs accum time axes identical (so 792 h is 792 h, not 1584)
  * missing (NaN) counts and min/max ranges per variable (anomaly screen)

Read-only. Prints a report; never writes data.
"""
import os
import glob

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_era5")

# plausible ranges for a sanity screen (Aug, 30-33 N / 112-121 E, ERA5 units)
RANGES = {
    "t2m": (200.0, 340.0),                         # K
    "u100": (-60.0, 60.0), "v100": (-60.0, 60.0),  # m/s
    "u10": (-60.0, 60.0), "v10": (-60.0, 60.0),    # m/s
    "ssrd": (-1.0, 6.0e6),                         # J/m2 (hour accumulation)
}
META = {"number", "expver"}

# full-coverage target: [2022-04-30 00:00, 2022-10-02 00:00) UTC = 155 d = 3720 h
TARGET_START = np.datetime64("2022-04-30T00")
TARGET_END_EXCL = np.datetime64("2022-10-02T00")


def fmt_ranges(times):
    """Compact a sorted datetime64 array into 'start..end' ranges."""
    t = np.asarray(times).astype("datetime64[h]")
    if len(t) == 0:
        return "(none)"
    H = np.timedelta64(1, "h")
    ranges = []
    start = prev = t[0]
    for x in t[1:]:
        if (x - prev) == H:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ", ".join(str(s) if s == e else f"{s}..{e}" for s, e in ranges)


def pick(ds, candidates):
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    return None


def check_time(ds, tag, step, tname):
    t = np.asarray(ds[tname].values, dtype="datetime64[ns]")
    n = len(t)
    first, last = t[0], t[-1]
    uniq = len(np.unique(t))
    diffs = np.diff(t) / np.timedelta64(1, "h")
    contiguous = bool(np.all(diffs == 1.0)) if n > 1 else True
    print(f"  [{tag} {step}] time: n={n}  {first} .. {last}  "
          f"unique={uniq}  hourly_contiguous={contiguous}")
    return t, n, uniq, contiguous


def main():
    dirs = sorted(glob.glob(os.path.join(DATA, "unpacked_*")))
    if not dirs:
        print("no unpacked_* directories found")
        return
    all_times = []
    var_times = {}      # var -> list of per-block time arrays (coverage)
    var_nan = {}        # var -> total NaN cell count across blocks
    for d in dirs:
        tag = os.path.basename(d)[len("unpacked_"):]
        inst = os.path.join(d, "data_stream-oper_stepType-instant.nc")
        acc = os.path.join(d, "data_stream-oper_stepType-accum.nc")
        print("=" * 70)
        print(f"block {tag}")
        di = xr.open_dataset(inst)
        da = xr.open_dataset(acc)
        tname = pick(di, ["valid_time", "time"])
        latname = pick(di, ["latitude", "lat"])
        lonname = pick(di, ["longitude", "lon"])
        print(f"  coord names: time={tname} lat={latname} lon={lonname}")
        print(f"  instant data_vars: {[v for v in di.data_vars if v not in META]}")
        print(f"  accum   data_vars: {[v for v in da.data_vars if v not in META]}")
        # metadata vars (expver/number can cause duplicate streams if >1)
        for v in META:
            for label, ds in [("instant", di), ("accum", da)]:
                if v in ds.variables:
                    print(f"  {label}.{v}: shape={ds[v].shape} "
                          f"values={np.unique(ds[v].values).tolist()}")
        for name, ds in [("instant", di), ("accum", da)]:
            for v in ds.data_vars:
                if v in META:
                    continue
                var = ds[v]
                print(f"    {v}: dims={var.dims} units={var.attrs.get('units')!r} "
                      f"long_name={var.attrs.get('long_name', '')!r}")
        # lat/lon
        la = di[latname].values
        lo = di[lonname].values
        print(f"  lat: {len(la)} pts, {la[0]:.2f} .. {la[-1]:.2f} "
              f"(step {np.diff(la)[0]:.2f})")
        print(f"  lon: {len(lo)} pts, {lo[0]:.2f} .. {lo[-1]:.2f} "
              f"(step {np.diff(lo)[0]:.2f})")
        # time axes
        ti, ni, ui, ci = check_time(di, tag, "instant", tname)
        ta, na, ua, ca = check_time(da, tag, "accum", tname)
        print(f"  instant vs accum time identical: "
              f"{bool(np.array_equal(ti, ta))}")
        all_times.append(ti)
        for v in di.data_vars:
            if v not in META:
                var_times.setdefault(v, []).append(ti)
        for v in da.data_vars:
            if v not in META:
                var_times.setdefault(v, []).append(ta)
        # missing + ranges
        for name, ds in [("instant", di), ("accum", da)]:
            for v in ds.data_vars:
                if v in META:
                    continue
                arr = ds[v].values.astype("float64")
                finite = np.isfinite(arr)
                n_nan = int(arr.size - finite.sum())
                var_nan[v] = var_nan.get(v, 0) + n_nan
                lo_v = float(np.nanmin(arr)) if finite.any() else float("nan")
                hi_v = float(np.nanmax(arr)) if finite.any() else float("nan")
                ok = RANGES.get(v)
                flag = ""
                if ok is not None and finite.any():
                    if lo_v < ok[0] or hi_v > ok[1]:
                        flag = f"  <-- OUT OF PLAUSIBLE RANGE {ok}"
                print(f"    {v}: nan={n_nan}/{arr.size}  min={lo_v:.4g}  "
                      f"max={hi_v:.4g}{flag}")
        di.close(); da.close()

    # cross-block continuity
    if all_times:
        cat = np.sort(np.concatenate(all_times))
        print("=" * 70)
        print(f"all blocks: {len(cat)} time steps, {len(np.unique(cat))} unique, "
              f"{cat[0]} .. {cat[-1]}")
        diffs = np.diff(cat) / np.timedelta64(1, "h")
        if len(diffs):
            bad = np.where(diffs != 1.0)[0]
            print(f"  non-1-hour transitions: {len(bad)}"
                  + (f" -> indices {bad.tolist()[:10]}" if len(bad) else ""))
            print(f"  union contiguous 1-hourly: {bool(np.all(diffs == 1.0))}")

    # full target coverage per variable (6 variables x 3720 h, no dup, no gap)
    if var_times:
        target = np.arange(TARGET_START, TARGET_END_EXCL, dtype="datetime64[h]")
        print("=" * 70)
        print("FULL TARGET COVERAGE "
              f"({len(target)} h, {target[0]} .. {target[-1]} UTC)")
        all_ok = True
        for v in sorted(var_times):
            cat = np.sort(np.concatenate(var_times[v]))
            uniq_ns, counts = np.unique(cat, return_counts=True)
            uniq = uniq_ns.astype("datetime64[h]")
            dup_vals = uniq[counts > 1]
            present = uniq[np.isin(uniq, target)]
            missing = target[~np.isin(target, uniq)]
            extra = uniq[~np.isin(uniq, target)]
            d = np.diff(uniq).astype("timedelta64[h]").astype(np.int64)
            contiguous = bool(np.all(d == 1)) if len(uniq) > 1 else (len(uniq) == 1)
            ok = (len(present) == len(target) and len(missing) == 0
                  and len(extra) == 0 and len(dup_vals) == 0 and contiguous)
            all_ok = all_ok and ok
            print(f"  {v:5s}: present={len(present)}/{len(target)}  "
                  f"dup={len(dup_vals)}  missing={len(missing)}  "
                  f"extra={len(extra)}  contiguous={contiguous}  "
                  f"nan_cells={var_nan.get(v, 0)}")
            if len(missing):
                print(f"        missing ranges: {fmt_ranges(missing)}")
            if len(dup_vals):
                print(f"        duplicated timestamps: {len(dup_vals)} -> "
                      f"{dup_vals[:10].tolist()}"
                      f"{' ...' if len(dup_vals) > 10 else ''}")
            if len(extra):
                print(f"        extra (outside target): {extra[:10].tolist()}"
                      f"{' ...' if len(extra) > 10 else ''}")
        print(f"  COVERAGE: {'OK' if all_ok else 'GAPS DETECTED'} "
              f"({'every variable 3720 unique contiguous hours, no dup/missing/extra'
                  if all_ok else 'see per-variable lines above'})")


if __name__ == "__main__":
    main()
