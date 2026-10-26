# -*- coding: utf-8 -*-
"""
Unpack + per-year verify the multi-year ERA5 warm-season downloads (2011-2021).

The 2022 pilot blocks already live (verified) in data_era5/unpacked_* and are
reused in place; the new years are downloaded as ZIPs named era5_<tag>.nc into
data_era5_multi/. This script:

  1. UNPACK  (idempotent): each era5_*.nc ZIP in data_era5_multi/ is
     integrity-tested (zipfile.testzip) and extracted into
     data_era5_multi/unpacked_<tag>/ (internal CDS member names preserved).
     Existing members are never overwritten; original ZIPs are kept.

  2. VERIFY  (per year): for the requested year(s), the union of unpacked block
     time axes must equal EXACTLY [YYYY-04-30 00:00, YYYY-10-02 00:00) UTC
     (155 d = 3720 unique contiguous hours), with the 6 expected variables
     (t2m, u100, v100, u10, v10, ssrd), 13 x 37 grid, 0 NaN. A year is reported
     COMPLETE only when every check passes; otherwise the missing/extra/duplicate
     detail is printed (no guessing, no silent partial).

Credentials are never read/printed here. Read-only w.r.t. every input; only
writes the unpacked_*/ members.
"""
import argparse
import glob
import os
import sys
import zipfile

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PILOT = os.path.join(ROOT, "data_era5")          # 2022 (verified, reused)
DATA = os.path.join(ROOT, "data_era5_multi")          # 2011-2021 new downloads

INST_MEMBER = "data_stream-oper_stepType-instant.nc"
ACC_MEMBER = "data_stream-oper_stepType-accum.nc"
META = {"number", "expver"}
EXPECTED_INSTANT = {"t2m", "u100", "v100", "u10", "v10"}
EXPECTED_ACCUM = {"ssrd"}
N_LAT, N_LON = 13, 37

RANGES = {
    "t2m": (200.0, 340.0),
    "u100": (-60.0, 60.0), "v100": (-60.0, 60.0),
    "u10": (-60.0, 60.0), "v10": (-60.0, 60.0),
    "ssrd": (-1.0, 6.0e6),
}


def pick(ds, candidates):
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    return None


def unpack(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "era5_*.nc")))
    n_done = 0
    for fp in files:
        base = os.path.basename(fp)
        tag = base[len("era5_"):-len(".nc")]
        with open(fp, "rb") as f:
            magic = f.read(4)
        if magic != b"PK\x03\x04":
            print(f"[{base}] NOT a zip (magic={magic!r}); skipping")
            continue
        try:
            with zipfile.ZipFile(fp) as z:
                bad = z.testzip()
                if bad is not None:
                    print(f"[{base}] ZIP CORRUPT at member {bad!r}; skipping")
                    continue
                members = z.namelist()
                outdir = os.path.join(data_dir, f"unpacked_{tag}")
                os.makedirs(outdir, exist_ok=True)
                for name in members:
                    dest = os.path.join(outdir, os.path.basename(name))
                    if os.path.exists(dest):
                        continue
                    with z.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                print(f"[{base}] unpacked {len(members)} member(s) -> unpacked_{tag}/")
                n_done += 1
        except zipfile.BadZipFile as e:
            print(f"[{base}] NOT a valid zip: {e}")
    return n_done


def target(year):
    return np.arange(f"{year}-04-30", f"{year}-10-02", dtype="datetime64[h]")


def collect_unpacked(data_dirs):
    """Return list of (tag, base, inst_path, acc_path) for unpacked dirs."""
    out = []
    for base in data_dirs:
        for d in sorted(glob.glob(os.path.join(base, "unpacked_*"))):
            tag = os.path.basename(d)[len("unpacked_"):]
            inst = os.path.join(d, INST_MEMBER)
            acc = os.path.join(d, ACC_MEMBER)
            if os.path.exists(inst) and os.path.exists(acc):
                out.append((tag, base, inst, acc))
    return out


def verify_year(year, data_dirs):
    tgt = target(year)
    t0, t1 = tgt[0], tgt[-1]
    blocks = []
    for tag, base, inst, acc in collect_unpacked(data_dirs):
        di = xr.open_dataset(inst)
        tn = pick(di, ["valid_time", "time"])
        t = np.asarray(di[tn].values, dtype="datetime64[h]")
        di.close()
        if t.size == 0 or t[-1] < t0 or t[0] > t1:
            continue
        blocks.append((tag, inst, acc))

    if not blocks:
        print(f"[{year}] NO unpacked blocks overlap the target -> INCOMPLETE")
        return False

    all_t = []
    inst_vars = set(); accum_vars = set()
    lat_ok = lon_ok = True
    nan_total = 0
    range_bad = []
    for tag, inst, acc in blocks:
        di = xr.open_dataset(inst); da = xr.open_dataset(acc)
        tn = pick(di, ["valid_time", "time"])
        latn = pick(di, ["latitude", "lat"]); lonn = pick(di, ["longitude", "lon"])
        t = np.asarray(di[tn].values, dtype="datetime64[h]")
        ta = np.asarray(da[tn].values, dtype="datetime64[h]")
        all_t.append(t)
        if not np.array_equal(t, ta):
            print(f"[{year}] {tag}: instant vs accum time MISMATCH")
        inst_vars |= {v for v in di.data_vars if v not in META}
        accum_vars |= {v for v in da.data_vars if v not in META}
        la = di[latn].values; lo = di[lonn].values
        if la.size != N_LAT or lo.size != N_LON:
            lat_ok = False
        for name, ds in [("instant", di), ("accum", da)]:
            for v in ds.data_vars:
                if v in META:
                    continue
                arr = ds[v].values.astype("float64")
                nan_total += int(arr.size - np.isfinite(arr).sum())
                rng = RANGES.get(v)
                fin = np.isfinite(arr)
                if rng is not None and fin.any() and (
                        arr[fin].min() < rng[0] or arr[fin].max() > rng[1]):
                    range_bad.append((tag, v))
        di.close(); da.close()

    union = np.sort(np.concatenate(all_t))
    uniq = np.unique(union)
    dup = union.size - uniq.size
    present = uniq[np.isin(uniq, tgt)]
    missing = tgt[~np.isin(tgt, uniq)]
    extra = uniq[~np.isin(uniq, tgt)]
    d = np.diff(uniq).astype("timedelta64[h]").astype(np.int64)
    contiguous = bool(np.all(d == 1)) if len(uniq) > 1 else (len(uniq) == 1)

    inst_ok = inst_vars == EXPECTED_INSTANT
    acc_ok = accum_vars == EXPECTED_ACCUM

    ok = (len(present) == len(tgt) and len(missing) == 0 and len(extra) == 0
          and dup == 0 and contiguous and inst_ok and acc_ok
          and lat_ok and lon_ok and nan_total == 0 and len(range_bad) == 0)

    print(f"[{year}] instant_vars={sorted(inst_vars)} accum_vars={sorted(accum_vars)}")
    print(f"[{year}] grid_ok={lat_ok and lon_ok}  NaN_cells={nan_total}  "
          f"out_of_range={len(range_bad)}")
    print(f"[{year}] present={len(present)}/{len(tgt)}  dup={dup}  "
          f"missing={len(missing)}  extra={len(extra)}  contiguous={contiguous}")
    if len(missing):
        print(f"[{year}]   missing: {missing[:6].tolist()}"
              f"{' ...' if len(missing) > 6 else ''}")
    if len(extra):
        print(f"[{year}]   extra: {extra[:6].tolist()}"
              f"{' ...' if len(extra) > 6 else ''}")
    if range_bad:
        print(f"[{year}]   out-of-range blocks/vars: {range_bad[:8]}")
    print(f"[{year}] => {'COMPLETE' if ok else 'INCOMPLETE / FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default=None,
                    help="comma list of years to verify (default: 2011-2021)")
    ap.add_argument("--no-unpack", action="store_true", help="skip the unpack step")
    args = ap.parse_args()

    if not args.no_unpack:
        print("UNPACK data_era5_multi/ ...")
        n = unpack(DATA)
        print(f"unpacked {n} package(s)")

    years = ([int(x) for x in args.years.split(",") if x.strip()]
             if args.years else list(range(2011, 2022)))
    data_dirs = [DATA_PILOT, DATA]
    results = {}
    for y in years:
        results[y] = verify_year(y, data_dirs)

    print("=" * 72)
    complete = [y for y, ok in results.items() if ok]
    print(f"COMPLETE years: {complete if complete else '(none)'}")
    print(f"INCOMPLETE     : {[y for y, ok in results.items() if not ok]}")
    sys.exit(0 if len(complete) == len(years) else 1)


if __name__ == "__main__":
    main()
