# -*- coding: utf-8 -*-
"""
ERA5 hourly single-levels download for the multi-year warm-season extension
(2011-2021; 2022 is already downloaded and verified in data_era5/ and is REUSED,
never re-requested). DRY-RUN by default.

Region / grid / dataset / variables are IDENTICAL to the verified 2022 pilot
(scripts/era5_download_2022_may_sep.py):
    region  30-33 N, 112-121 E, 0.25 deg regular grid (13 x 37 = 481 points)
    dataset reanalysis-era5-single-levels, product_type reanalysis
    variables (6): 2m_temperature (t2m), 100m_u/v_component_of_wind (u100/v100),
                   10m_u/v_component_of_wind (u10/v10),
                   surface_solar_radiation_downwards (ssrd)
    time     24 hourly steps per day (00:00..23:00 UTC)

Per-year target coverage (same 155-day / 3720-h window as the 2022 pilot):
    [YYYY-04-30 00:00, YYYY-10-02 00:00) UTC = Apr 30 + May..Sep + Oct 1.

Each year is requested in SEVEN resumable parts so a single failure never
invalidates the other parts and a retry is small. The 5-month core is split into
MONTHLY parts because a single 153-day request exceeds the CDS cost limit
("cost limits exceeded / request too large") -- monthly (<=31 d, 6 vars) matches
the verified 2022 pilot block sizes:
    * {y}04_30_30   April 30   (1 d, leading daily-aggregation buffer)
    * {y}05_01_31   May        (31 d)
    * {y}06_01_30   June       (30 d)
    * {y}07_01_31   July       (31 d)
    * {y}08_01_31   August     (31 d)
    * {y}09_01_30   September  (30 d)
    * {y}10_01_01   October 1  (1 d, trailing symmetric margin)

Existing files are verified by CONTENT (the union of unpacked_*/ time axes), not
by mere existence; a raw package is reused only if it passes a ZIP integrity test.

Credentials: never read or printed here. Download mode imports `cdsapi`, which
reads the user's ~/.cdsapirc internally.
"""
import argparse
import calendar
import glob
import os
import sys
import zipfile

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

AREA = [33, 112, 30, 121]          # [North, West, South, East]
TIMES = [f"{h:02d}:00" for h in range(24)]

YEARS = list(range(2011, 2022))    # 2011..2021 (2022 already done, reused)

N_LAT, N_LON = 13, 37
N_GRID = N_LAT * N_LON
BYTES_PER_VAL = 4                  # float32

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2022 pilot blocks (verified) live here and are reused; new years go to the
# multi-year directory so the old pilot files are never touched.
DATA_PILOT = os.path.join(ROOT, "data_era5")
DATA = os.path.join(ROOT, "data_era5_multi")

INST_MEMBER = "data_stream-oper_stepType-instant.nc"
ACC_MEMBER = "data_stream-oper_stepType-accum.nc"

# full single-year coverage target
def target_hours(year):
    return set(np.arange(f"{year}-04-30", f"{year}-10-02",
                         dtype="datetime64[h]").tolist())


def parts(year):
    """Seven resumable request parts per year: (tag, months, days, expected_days).

    The core (May..Sep) is split monthly because a single 153-day request exceeds
    the CDS cost limit; monthly (<=31 d) matches the verified 2022 pilot blocks.
    Day lists come from calendar.monthrange so June/September never submit day 31
    (no reliance on the server silently ignoring invalid days).
    """
    m = {mm: [f"{d:02d}" for d in range(1, calendar.monthrange(year, mm)[1] + 1)]
         for mm in (5, 6, 7, 8, 9)}
    return [
        (f"{year}04_30_30", ["04"], ["30"], 1),
        (f"{year}05_01_31", ["05"], m[5], len(m[5])),
        (f"{year}06_01_30", ["06"], m[6], len(m[6])),
        (f"{year}07_01_31", ["07"], m[7], len(m[7])),
        (f"{year}08_01_31", ["08"], m[8], len(m[8])),
        (f"{year}09_01_30", ["09"], m[9], len(m[9])),
        (f"{year}10_01_01", ["10"], ["01"], 1),
    ]


# --------------------------------------------------------------------------- #
# coverage scan (actual time axis, not file existence)
# --------------------------------------------------------------------------- #
def _scan_existing(data_dirs):
    import xarray as xr
    covered = set()
    rows = []   # (tag, t0, t1, n_hours, n_instant_vars, n_accum_vars)
    for base in data_dirs:
        for d in sorted(glob.glob(os.path.join(base, "unpacked_*"))):
            tag = os.path.basename(d)[len("unpacked_"):]
            inst = os.path.join(d, INST_MEMBER)
            acc = os.path.join(d, ACC_MEMBER)
            if not (os.path.exists(inst) and os.path.exists(acc)):
                rows.append((tag, None, None, 0, 0, 0))
                continue
            di = xr.open_dataset(inst)
            da = xr.open_dataset(acc)
            tn = "valid_time" if "valid_time" in di.variables else "time"
            t = np.asarray(di[tn].values, dtype="datetime64[h]")
            iv = [v for v in di.data_vars if v not in {"number", "expver"}]
            av = [v for v in da.data_vars if v not in {"number", "expver"}]
            rows.append((tag, t[0], t[-1], int(t.size), len(iv), len(av)))
            covered.update(t.tolist())
            di.close(); da.close()
    return covered, rows


def build_request(year, months, days):
    return {
        "product_type": PRODUCT_TYPE,
        "variable": VARIABLES,
        "year": year,
        "month": months,
        "day": days,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
    }


def _zip_ok(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"PK\x03\x04":
        return True                      # not a zip -> assume plain nc
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) >= 1
    except zipfile.BadZipFile:
        return False


def _expected_part_hours(year, months):
    """Return the exact UTC hours a part's (year, month list) is expected to cover."""
    hours = set()
    for m in months:
        y, mi = int(year), int(m)
        if mi == 12:
            nxt = (y + 1, 1)
        else:
            nxt = (y, mi + 1)
        start = np.datetime64(f"{y:04d}-{mi:02d}-01")
        end = np.datetime64(f"{nxt[0]:04d}-{nxt[1]:02d}-01")
        hours.update(np.arange(start, end, dtype="datetime64[h]").tolist())
    return hours


# --------------------------------------------------------------------------- #
# dry-run report
# --------------------------------------------------------------------------- #
def dry_run(covered, rows):
    # observed per-day compressed size from the verified 2022 pilot blocks
    pilot_days = 155
    pilot_bytes = sum(os.path.getsize(p)
                      for p in glob.glob(os.path.join(DATA_PILOT, "era5_*.nc")))
    per_day_comp = pilot_bytes / pilot_days if pilot_bytes else None

    n_req = 0
    n_missing_days = 0
    n_missing_hours = 0

    print("=" * 76)
    print("ERA5 multi-year (2011-2021) download - DRY-RUN (no request, no credential read)")
    print("=" * 76)
    print(f"dataset      : {DATASET} / {PRODUCT_TYPE}")
    print(f"variables    : {len(VARIABLES)}")
    for v in VARIABLES:
        print(f"               - {v}")
    print(f"area         : {AREA}  (30-33 N, 112-121 E)")
    print(f"grid         : {N_LAT} x {N_LON} = {N_GRID} points (0.25 deg)")
    print(f"years        : {YEARS[0]}..{YEARS[-1]} ({len(YEARS)} y); 2022 reused from data_era5/")
    print(f"per-year span: YYYY-04-30 00:00 .. YYYY-10-01 23:00 UTC (155 d = 3720 h)")
    print("-" * 76)

    print("EXISTING coverage (read from unpacked_*/ time axes):")
    for tag, t0, t1, n, ni, na in sorted(rows):
        if t0 is None:
            print(f"  [{tag}] unpacked members missing -> NOT covered")
        else:
            print(f"  [{tag}] {n:5d} h  {t0} .. {t1}  instant={ni}v accum={na}v")
    print(f"  total covered hours: {len(covered)}")
    print("-" * 76)

    print("MISSING parts (per year):")
    for y in YEARS:
        tgt = target_hours(y)
        miss = sorted(tgt - covered)
        part_lines = []
        for tag, months, days, nd in parts(y):
            exp = _expected_part_hours(str(y), months)
            have = exp <= covered
            raw_ok = _zip_ok(os.path.join(DATA, f"era5_{tag}.nc"))
            if have or raw_ok:
                part_lines.append(f"     {tag}: OK (content={'covered' if have else 'raw-package'} )")
            else:
                part_lines.append(f"     {tag}: MISSING ({nd} d)")
                n_req += 1
                n_missing_days += nd
                n_missing_hours += nd * 24
        print(f"  {y}: {len(miss)} h missing / {3720 - len(miss)} h covered")
        for pl in part_lines:
            print(pl)
    print("-" * 76)
    print(f"requests     : {n_req}  (7 per missing year-part)")
    print(f"missing data : {n_missing_days} days = {n_missing_hours} h")
    raw = N_GRID * n_missing_hours * len(VARIABLES) * BYTES_PER_VAL
    print(f"size estimate: uncompressed {raw:,} B ~ {raw/1e6:.1f} MB")
    if per_day_comp:
        print(f"               compressed ~{per_day_comp/1e3:.0f} KB/day "
              f"-> ~{per_day_comp * n_missing_days / 1e6:.1f} MB total")
    print(f"save location: {DATA}")
    print("-" * 76)
    print("NOT downloading now (dry-run). Re-run with --download to retrieve.")
    print(f"  download cmd: & \"{sys.executable}\" -I -u "
          f"\"{os.path.join(ROOT, 'scripts', os.path.basename(__file__))}\" --download")


# --------------------------------------------------------------------------- #
# actual download (resumable, temp-then-rename, non-zero exit on failure)
# --------------------------------------------------------------------------- #
def download(covered, selected_years):
    try:
        import cdsapi            # lazy: DRY-RUN never imports/reads credentials
    except ImportError:
        raise SystemExit("cdsapi is not installed; install it and configure "
                         "~/.cdsapirc, then re-run --download.")
    client = cdsapi.Client()     # reads ~/.cdsapirc internally; nothing printed
    os.makedirs(DATA, exist_ok=True)

    failed = []
    for y in selected_years:
        for tag, months, days, nd in parts(y):
            exp = _expected_part_hours(str(y), months)
            final = os.path.join(DATA, f"era5_{tag}.nc")
            if exp <= covered or _zip_ok(final):
                print(f"SKIP {tag}: already covered or intact package (no overwrite)")
                continue
            tmp = os.path.join(DATA, f"era5_{tag}.download")
            req = build_request(str(y), months, days)
            print(f"requesting {tag}  (year {y}, months {months}, days {len(days)})")
            try:
                client.retrieve(DATASET, req, tmp)
                if not _zip_ok(tmp):
                    raise RuntimeError("downloaded result is empty or a corrupt ZIP")
                os.replace(tmp, final)         # atomic once complete
                print(f"  wrote {final} (raw ZIP; unpack + verify next)")
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
          "; run the multi-year unpack/verify step next")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true",
                    help="actually retrieve missing parts (default: dry-run)")
    ap.add_argument("--years", metavar="Y1,Y2,...",
                    help="restrict to these years (e.g. 2011,2012)")
    ap.add_argument("--only", metavar="TAG",
                    help="download only the part matching this tag (e.g. 20110509_core)")
    args = ap.parse_args()

    covered, rows = _scan_existing([DATA_PILOT, DATA])

    if not args.download:
        dry_run(covered, rows)
        return

    sel = YEARS
    if args.years:
        sel = [int(x) for x in args.years.split(",") if x.strip()]
    if args.only:
        # filter to the single matching part
        for y in sel:
            for tag, months, days, nd in parts(y):
                if tag == args.only:
                    exp = _expected_part_hours(str(y), months)
                    if exp <= covered or _zip_ok(
                            os.path.join(DATA, f"era5_{tag}.nc")):
                        print(f"SKIP {tag}: already covered or intact")
                        return
                    import cdsapi
                    client = cdsapi.Client()
                    os.makedirs(DATA, exist_ok=True)
                    tmp = os.path.join(DATA, f"era5_{tag}.download")
                    final = os.path.join(DATA, f"era5_{tag}.nc")
                    print(f"requesting {tag} (year {y}, months {months})")
                    client.retrieve(DATASET, build_request(str(y), months, days), tmp)
                    if not _zip_ok(tmp):
                        raise RuntimeError("downloaded result is empty or corrupt ZIP")
                    os.replace(tmp, final)
                    print(f"  wrote {final}")
                    return
        raise SystemExit(f"--only {args.only} matched no missing part")

    download(covered, sel)


if __name__ == "__main__":
    main()
