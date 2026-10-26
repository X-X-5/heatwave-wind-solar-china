# -*- coding: utf-8 -*-
"""
One-month ERA5 reanalysis download script (DRY-RUN by default).

Target: 2022-08 (the month containing the pilot representative cell's merged
process 08-02 ~ 08-22), small region 30-33 N / 112-121 E, hourly, single-levels.

Variables (6) — wind/solar potential inputs, all on the same 0.25-degree
single-levels grid:
    * 2m_temperature                    (t2m,  K,  instantaneous)
    * 100m_u_component_of_wind          (u100, m/s, instantaneous)  turbine hub
    * 100m_v_component_of_wind          (v100, m/s, instantaneous)
    * 10m_u_component_of_wind           (u10,  m/s, instantaneous)  PV cell-temp
    * 10m_v_component_of_wind           (v10,  m/s, instantaneous)   cooling term
    * surface_solar_radiation_downwards (ssrd, J/m2, accumulated)   PV GHI
       -> u10/v10 are included because the in-house PV cell-temperature model
          needs the 10-m wind speed (wind10 = sqrt(u10^2 + v10^2)).

ssrd accumulation (VERIFIED against CDS/ECMWF docs, 2026-10):
    ERA5 *reanalysis* hourly values from CDS are accumulated over the HOUR
    ENDING at the validity time (e.g. 12:00 = energy 11:00-12:00 UTC), so
    GHI [W m-2] = ssrd [J m-2] / 3600, per hour, NO differencing needed.
    (ERA5-Land differs: it accumulates since 00 UTC and needs differencing.)

Observation-day window (Beijing = UTC+8):
    CN05.1 daily Tmax for date D may be a 00-24 or a 20-20 Beijing-time window:
      * 00-24 Beijing for D  =  [D-1 16:00 UTC, D 16:00 UTC)
      * 20-20 Beijing for D  =  [D-1 12:00 UTC, D 12:00 UTC)
    Both windows only reach BACK into the previous UTC day and end by D 16:00
    UTC. So an August target day needs only ONE LEADING day (07-31); it does NOT
    need 09-01 UTC data (e.g. 08-31 00-24 Beijing = 08-30 16:00 .. 08-31 16:00
    UTC). The trailing 09-01 block is a symmetric margin kept for the original
    33-day plan, NOT a correctness requirement for August daily aggregation.

Credentials: this script NEVER reads or prints any CDS credential. The real
download relies on `cdsapi` which reads the user's ~/.cdsapirc (new CDS uses a
Personal Access Token via https://cds.climate.copernicus.eu/api). DRY-RUN does
not import cdsapi and does not touch any credential.

Usage:
    python era5_download_2022.py            # dry-run: print request + size estimate
    python era5_download_2022.py --download # actually retrieve (after review)
    python era5_download_2022.py --download --only 202209  # retry one block alone
"""
import argparse
import os

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

# [North, West, South, East] = 30-33 N, 112-121 E
AREA = [33, 112, 30, 121]

TIMES = [f"{h:02d}:00" for h in range(24)]          # all 24 hours

# (year, month, [days]) chunks: 1-day buffer before + target month + 1-day margin.
# NOTE: only the LEADING day (07-31) is needed to aggregate August's Beijing-time
# daily windows (each ends by D 16:00 UTC). The trailing 09-01 is a symmetric
# margin for the original 33-day plan, NOT required by the time windows.
CHUNKS = [
    ("2022", "07", ["31"]),                          # buffer: 1 day before (needed)
    ("2022", "08", [f"{d:02d}" for d in range(1, 32)]),  # target month (31 d)
    ("2022", "09", ["01"]),                          # margin: 1 day after (optional)
]

# Grid / size bookkeeping (ERA5 single-levels is on a 0.25-deg regular grid).
N_LAT = 13    # 30.0 .. 33.0 inclusive at 0.25 deg
N_LON = 37    # 112.0 .. 121.0 inclusive at 0.25 deg
BYTES_PER_VAL = 4          # float32
HOURS_PER_DAY = 24

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data_era5")          # NEW independent directory


def total_days():
    return sum(len(days) for _, _, days in CHUNKS)


def estimate_size():
    n_days = total_days()
    n_steps = n_days * HOURS_PER_DAY
    n_grid = N_LAT * N_LON
    raw = n_grid * n_steps * len(VARIABLES) * BYTES_PER_VAL
    return n_days, n_steps, n_grid, raw


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


def dry_run():
    n_days, n_steps, n_grid, raw = estimate_size()
    print("=" * 72)
    print("ERA5 one-month download - DRY-RUN (no request sent, no credential read)")
    print("=" * 72)
    print(f"dataset      : {DATASET}")
    print(f"product_type : {PRODUCT_TYPE}")
    print(f"variables    : {len(VARIABLES)}")
    for v in VARIABLES:
        print(f"               - {v}")
    print(f"area [N,W,S,E]: {AREA}  (30-33 N, 112-121 E)")
    print(f"grid         : {N_LAT} lat x {N_LON} lon = {n_grid} points "
          f"(0.25-deg regular grid)")
    print(f"time         : {HOURS_PER_DAY} hourly steps {TIMES[0]}..{TIMES[-1]} "
          f"({HOURS_PER_DAY} / day)")
    print(f"chunks       :")
    for year, month, days in CHUNKS:
        print(f"               {year}-{month}  days {days[0]}..{days[-1]} "
              f"({len(days)} d)")
    print(f"total        : {n_days} days x {HOURS_PER_DAY} h = {n_steps} time steps")
    print("-" * 72)
    print("EXPECTED REQUEST SIZE (raw float32, uncompressed NetCDF):")
    print(f"  {n_grid} pts x {n_steps} steps x {len(VARIABLES)} vars x "
          f"{BYTES_PER_VAL} B")
    print(f"  = {raw:,} bytes ~ {raw/1e6:.1f} MB")
    print(f"  NetCDF4-compressed / GRIB would be ~2-5 MB; well under any CDS limit.")
    print("-" * 72)
    print("ssrd handling (verified): hourly ERA5 reanalysis = accumulation over")
    print("  the hour ENDING at the timestamp -> GHI = ssrd / 3600, NO differencing.")
    print("observation-day window (Beijing = UTC+8):")
    print("  00-24 Beijing for D = [D-1 16:00, D 16:00) UTC;")
    print("  20-20 Beijing for D = [D-1 12:00, D 12:00) UTC.")
    print("  -> only ONE LEADING day (07-31) is needed for August daily windows;")
    print("     the trailing 09-01 block is a symmetric margin, NOT required.")
    print()
    print("To actually download (after review):")
    print("  1) accept the dataset licence at the CDS page,")
    print("  2) have ~/.cdsapirc (Personal Access Token) configured,")
    print("  3) re-run with --download.")
    print("NOT downloading now (this round is dry-run only).")


def download_chunks(chunks):
    try:
        import cdsapi   # imported lazily so DRY-RUN never touches it
    except ImportError:
        raise SystemExit("cdsapi is not installed. Install it, configure "
                         "~/.cdsapirc, then re-run --download.")
    os.makedirs(OUT_DIR, exist_ok=True)
    client = cdsapi.Client()   # reads ~/.cdsapirc internally; nothing printed
    for year, month, days in chunks:
        tag = f"{year}{month}_{days[0]}_{days[-1]}"
        # Keep the same on-disk convention as the already-verified blocks: the
        # raw CDS result (a ZIP for this mixed-stepType request) is saved as
        # era5_<tag>.nc, then era5_unpack_check.py extracts it into
        # unpacked_<tag>/. Never overwrite an existing block (the 09-01 retry must
        # not clobber the already-checked 07-31 / August files).
        final = os.path.join(OUT_DIR, f"era5_{tag}.nc")
        unpacked = os.path.join(OUT_DIR, f"unpacked_{tag}")
        if os.path.exists(final) or os.path.isdir(unpacked):
            print(f"SKIP {year}-{month} days {days[0]}..{days[-1]}: "
                  f"already downloaded (no overwrite)")
            continue
        tmp = os.path.join(OUT_DIR, f"era5_{tag}.download")
        req = build_request((year, month, days))
        print(f"requesting {year}-{month} days {days[0]}..{days[-1]} "
              f"({len(days)} d)")
        try:
            client.retrieve(DATASET, req, tmp)
        except Exception as e:
            # Record the request status/error verbatim; do NOT retry in a loop and
            # do NOT assert a root cause beyond what the error reports.
            print(f"FAILED {year}-{month} days {days[0]}..{days[-1]}: "
                  f"{type(e).__name__}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        os.rename(tmp, final)
        print(f"  wrote {final} (raw; unpack with era5_unpack_check.py)")
    print("DONE. Raw files are in", OUT_DIR, "; run era5_unpack_check.py next")


def download():
    download_chunks(CHUNKS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true",
                    help="actually retrieve (default: dry-run only)")
    ap.add_argument("--only", metavar="YYYYMM",
                    help="download only the chunk matching this YYYYMM prefix "
                         "(e.g. 202209 to retry the 09-01 block alone)")
    args = ap.parse_args()
    if args.download:
        chunks = CHUNKS
        if args.only:
            chunks = [c for c in CHUNKS if f"{c[0]}{c[1]}" == args.only]
            if not chunks:
                raise SystemExit(f"--only {args.only} matched no chunk; available: "
                                 f"{[f'{y}{m}' for y, m, _ in CHUNKS]}")
        download_chunks(chunks)
    else:
        dry_run()
