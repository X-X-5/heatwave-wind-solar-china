# -*- coding: utf-8 -*-
"""
ERA5 hourly single-levels download for the 2022 pilot-season extension
(May-Sep 2022 + aggregation buffers). DRY-RUN by default.

Region / grid / dataset / variables are UNCHANGED from the already-completed
August pilot (scripts/era5_download_2022.py):
    region  30-33 N, 112-121 E, 0.25 deg regular grid (13 x 37 = 481 points)
    dataset reanalysis-era5-single-levels, product_type reanalysis
    variables (6): 2m_temperature (t2m), 100m_u/v_component_of_wind (u100/v100),
                   10m_u/v_component_of_wind (u10/v10),
                   surface_solar_radiation_downwards (ssrd)
    time     24 hourly steps per day (00:00..23:00 UTC)

Target coverage:
    2022-05-01 .. 2022-09-30 (core), plus 2022-04-30 and 2022-10-01 as the
    daily-aggregation buffers. So the full needed span is
    [2022-04-30 00:00, 2022-10-02 00:00) UTC = 155 days = 3720 hours.

The script READS the actual time coverage of the existing unpacked blocks
(data_era5/unpacked_*/), not just file existence, and downloads ONLY the missing
date-hours. Already-verified data (07-31, all of August, 09-01) is reused and
never re-requested.

Resumable / safe behaviour (--download):
    * skip a block if its raw package era5_<tag>.nc already exists and passes a
      ZIP integrity test (no overwrite of a correct file);
    * retrieve to a temp file era5_<tag>.download, then atomically rename to the
      final name only on success;
    * the CDS returns a ZIP (instant + accum stepTypes) even when named .nc --
      the raw ZIP is kept verbatim as the traceability package and later
      extracted by scripts/era5_unpack_check.py;
    * a failed block is recorded (tag + error) and the process exits non-zero so
      an orchestrated rerun notices it.

Credentials: never read or printed here. Download mode imports `cdsapi`, which
reads the user's ~/.cdsapirc (Personal Access Token) internally.

Usage:
    python era5_download_2022_may_sep.py               # dry-run (coverage + plan)
    python era5_download_2022_may_sep.py --download     # retrieve missing blocks
    python era5_download_2022_may_sep.py --download --only 202205   # retry one month
"""
import argparse
import glob
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np

DATASET = "reanalysis-era5-single-levels"
PRODUCT_TYPE = "reanalysis"

VARIABLES = [
    "2m_temperature",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

# [North, West, South, East]
AREA = [33, 112, 30, 121]

TIMES = [f"{h:02d}:00" for h in range(24)]       # all 24 hours per day

N_LAT = 13                                       # 30.0 .. 33.0 at 0.25 deg
N_LON = 37                                       # 112.0 .. 121.0 at 0.25 deg
N_GRID = N_LAT * N_LON
BYTES_PER_VAL = 4                                # float32

TARGET_START = "2022-04-30"                      # 00:00 UTC (leading buffer)
TARGET_END_EXCL = "2022-10-02"                   # exclusive -> through 10-01 23:00

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_era5")

META = {"number", "expver"}
INST_MEMBER = "data_stream-oper_stepType-instant.nc"
ACC_MEMBER = "data_stream-oper_stepType-accum.nc"


# --------------------------------------------------------------------------- #
# coverage scan (actual time axis + variables, not file existence)
# --------------------------------------------------------------------------- #
def _scan_existing():
    import xarray as xr
    blocks = []            # (tag, t0, t1, n_hours, instant_vars, accum_vars)
    covered = set()        # set of numpy datetime64[h]
    for d in sorted(glob.glob(os.path.join(DATA, "unpacked_*"))):
        tag = os.path.basename(d)[len("unpacked_"):]
        inst = os.path.join(d, INST_MEMBER)
        acc = os.path.join(d, ACC_MEMBER)
        if not (os.path.exists(inst) and os.path.exists(acc)):
            blocks.append((tag, None, None, 0, [], []))
            continue
        di = xr.open_dataset(inst)
        da = xr.open_dataset(acc)
        tn = "valid_time" if "valid_time" in di.variables else "time"
        t = np.asarray(di[tn].values, dtype="datetime64[h]")
        iv = [v for v in di.data_vars if v not in META]
        av = [v for v in da.data_vars if v not in META]
        blocks.append((tag, t[0], t[-1], int(t.size), iv, av))
        covered.update(t.tolist())
        di.close(); da.close()
    return blocks, covered


def _derive_chunks(missing_hours):
    """Group missing hours into (year, month, [days]) chunks. Only full-day
    gaps become requests; partial-day gaps are reported as anomalies."""
    day_count = defaultdict(int)
    for h in missing_hours:
        day_count[np.datetime64(h).astype("datetime64[D]")] += 1

    month_days = defaultdict(list)
    partial = []
    for d in sorted(day_count):
        c = day_count[d]
        dt = d.astype(object)                      # datetime.date
        if c == 24:
            month_days[(dt.year, dt.month)].append(dt.day)
        else:
            partial.append((str(d), c))

    chunks = []
    for (y, m) in sorted(month_days):
        days = sorted(month_days[(y, m)])
        chunks.append((str(y), f"{m:02d}", [f"{dd:02d}" for dd in days]))
    return chunks, partial


def build_request(chunk):
    year, month, days = chunk
    return {
        "product_type": PRODUCT_TYPE,
        "variable": VARIABLES,
        "year": year,
        "month": month,
        "day": days,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
    }


def _zip_ok(path):
    """True if path is a non-empty, intact download package (ZIP for this
    mixed-stepType request; a plain NetCDF is also accepted defensively)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"PK\x03\x04":
        return True                                # not a zip -> assume plain nc
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) >= 1
    except zipfile.BadZipFile:
        return False


# --------------------------------------------------------------------------- #
# dry-run report
# --------------------------------------------------------------------------- #
def dry_run():
    blocks, covered = _scan_existing()
    target = set(np.arange(TARGET_START, TARGET_END_EXCL, dtype="datetime64[h]").tolist())
    missing = sorted(target - covered)
    chunks, partial = _derive_chunks(missing)

    n_missing_hours = len(missing)
    n_missing_days = n_missing_hours // 24
    raw_bytes = N_GRID * n_missing_hours * len(VARIABLES) * BYTES_PER_VAL

    # observed compressed size from the already-downloaded full August block
    aug_pkg = os.path.join(DATA, "era5_202208_01_31.nc")
    per_day_comp = os.path.getsize(aug_pkg) / 31.0 if os.path.exists(aug_pkg) else None

    print("=" * 74)
    print("ERA5 May-Sep 2022 extension - DRY-RUN (no request, no credential read)")
    print("=" * 74)
    print(f"dataset      : {DATASET} / {PRODUCT_TYPE}")
    print(f"variables    : {len(VARIABLES)}")
    for v in VARIABLES:
        print(f"               - {v}")
    print(f"area         : {AREA}  (30-33 N, 112-121 E)")
    print(f"grid         : {N_LAT} x {N_LON} = {N_GRID} points (0.25 deg)")
    print(f"target span  : {TARGET_START} 00:00 .. 2022-10-01 23:00 UTC "
          f"(155 days = 3720 h)")
    print("-" * 74)

    print("EXISTING coverage (read from unpacked_*/ time axes, not file existence):")
    for tag, t0, t1, n, iv, av in blocks:
        if t0 is None:
            print(f"  [{tag}] unpacked members missing -> treated as NOT covered")
            continue
        print(f"  [{tag}] {n:4d} h  {t0} .. {t1}   instant={len(iv)}v accum={len(av)}v")
    n_covered = len(covered)
    print(f"  covered hours: {n_covered} / 3720  "
          f"({n_covered // 24} days); all 6 variables present in covered blocks")
    print("-" * 74)

    print(f"MISSING: {n_missing_hours} h = {n_missing_days} days "
          f"({len(chunks)} request(s))")
    for y, m, days in chunks:
        print(f"   {y}-{m}  days {days[0]}..{days[-1]}  ({len(days)} d)  "
              f"-> era5_{y}{m}_{days[0]}_{days[-1]}.nc")
    if partial:
        print(f"   WARNING partial-day gaps (unexpected, will NOT be requested as-is): {partial}")
    print("-" * 74)
    print(f"size estimate (uncompressed float32): "
          f"{N_GRID} x {n_missing_hours} h x {len(VARIABLES)} vars x "
          f"{BYTES_PER_VAL} B = {raw_bytes:,} B ~ {raw_bytes/1e6:.1f} MB")
    if per_day_comp:
        print(f"  rough compressed (observed ~{per_day_comp/1e3:.0f} KB/day "
              f"from the August block): ~{per_day_comp * n_missing_days / 1e6:.1f} MB")
    print(f"save location: {DATA}")
    print("-" * 74)

    cmd = (f'& "{sys.executable}" -I '
           f'"{os.path.join(ROOT, "scripts", os.path.basename(__file__))}" --download')
    print("PowerShell download command:")
    print(f"  {cmd}")
    print("  (after: run  python scripts/era5_unpack_check.py  to extract the ZIPs)")
    print()
    print("NOT downloading now (dry-run). Re-run with --download to retrieve.")


# --------------------------------------------------------------------------- #
# actual download (resumable, temp-then-rename, non-zero exit on failure)
# --------------------------------------------------------------------------- #
def download(chunks):
    try:
        import cdsapi          # lazy: DRY-RUN never imports or reads credentials
    except ImportError:
        raise SystemExit("cdsapi is not installed; install it and configure "
                         "~/.cdsapirc, then re-run --download.")
    client = cdsapi.Client()   # reads ~/.cdsapirc internally; nothing printed
    os.makedirs(DATA, exist_ok=True)

    failed = []
    for year, month, days in chunks:
        tag = f"{year}{month}_{days[0]}_{days[-1]}"
        final = os.path.join(DATA, f"era5_{tag}.nc")
        if _zip_ok(final):
            print(f"SKIP {year}-{month} days {days[0]}..{days[-1]}: "
                  f"already downloaded (intact, no overwrite)")
            continue

        tmp = os.path.join(DATA, f"era5_{tag}.download")
        req = build_request((year, month, days))
        print(f"requesting {year}-{month} days {days[0]}..{days[-1]} ({len(days)} d)")
        try:
            client.retrieve(DATASET, req, tmp)
            if not _zip_ok(tmp):
                raise RuntimeError("downloaded result is empty or a corrupt ZIP")
            os.replace(tmp, final)                 # atomic once complete
            print(f"  wrote {final} (raw ZIP; run era5_unpack_check.py next)")
        except Exception as e:
            failed.append((tag, f"{type(e).__name__}: {e}"))
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"FAILED {tag}: {type(e).__name__}: {e}")

    if failed:
        print("\nFAILED blocks:", file=sys.stderr)
        for tag, err in failed:
            print(f"  {tag}: {err}", file=sys.stderr)
        print(f"{len(failed)} block(s) failed -> exiting non-zero", file=sys.stderr)
        sys.exit(1)
    print("DONE. Raw packages are in", DATA,
          "; run scripts/era5_unpack_check.py then scripts/era5_verify.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true",
                    help="actually retrieve the missing blocks (default: dry-run)")
    ap.add_argument("--only", metavar="YYYYMM",
                    help="download only the missing chunk matching this YYYYMM prefix")
    args = ap.parse_args()

    blocks, covered = _scan_existing()
    target = set(np.arange(TARGET_START, TARGET_END_EXCL, dtype="datetime64[h]").tolist())
    missing = sorted(target - covered)
    chunks, partial = _derive_chunks(missing)

    if not args.download:
        dry_run()
        return

    if args.only:
        sel = [c for c in chunks if f"{c[0]}{c[1]}" == args.only]
        if not sel:
            avail = [f"{y}{m}" for y, m, _ in chunks]
            raise SystemExit(f"--only {args.only} matched no missing chunk; "
                             f"missing months: {avail or 'none'}")
        chunks = sel
    if not chunks:
        print("Nothing to download: target span already fully covered.")
        return
    download(chunks)


if __name__ == "__main__":
    main()
