# -*- coding: utf-8 -*-
"""
FINAL CHECK for the 2011-2022 warm-season expansion.

Verifies, ON DISK (not trusting any state file), that every year 2011-2022 has its
complete chain of products, that the per-year descriptive analyses exist for every
year, and that the multi-year correlation + aggregation summaries exist.

This is a presence + light-shape consistency check ONLY — it does NOT recompute
anything, does NOT add metrics, does NOT change definitions. It returns exit code
0 iff EVERYTHING is present and consistent; otherwise exit code 1 (and the caller
saves progress and does NOT shut down).

Expected per year (all in results_multi/ unless under monthly/ or joint/):
  * hourly Faiman  + linear            (time == 3720 h)
  * daily A + B                        (date == 153 d, 13 x 37)
  * raw + aligned labels
  * monthly_compare  (NetCDF + summary CSV + sensitivity CSV + report)
  * joint_low        (NetCDF + 2 summaries + report)

Expected multi-year (results_multi/):
  * wind_solar_corr_multi_2011_2022.nc + per_year nc + summaries + report
  * multi_year_summary.csv + REPORT_multi_year_summary.md
"""
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_multi")
MONTHLY_DIR = os.path.join(RESULTS, "monthly")
JOINT_DIR = os.path.join(RESULTS, "joint")

EXPECTED_YEARS = list(range(2011, 2023))


def check_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def shape_ok(path, dim, expect):
    """Light shape check (returns '' if ok, else a message)."""
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        return f"open failed: {type(e).__name__}"
    try:
        if dim not in ds.dims:
            return f"missing dim '{dim}'"
        n = ds.dims[dim]
        if n != expect:
            return f"dim '{dim}'={n} (expected {expect})"
        return ""
    finally:
        ds.close()


def main():
    failures = []
    ok_count = 0

    print("=" * 70)
    print("FINAL CHECK — per-year product chain (2011-2022)")
    print("=" * 70)
    for y in EXPECTED_YEARS:
        s = f"{y}0501_{y}0930"
        h = f"{y}0430_{y}1001"
        files = {
            "hourly_faiman":  (os.path.join(RESULTS, f"era5_hourly_potential_{h}.nc"), "valid_time", 3720),
            "hourly_linear":  (os.path.join(RESULTS, f"era5_hourly_potential_tcell_linear_{h}.nc"), "valid_time", 3720),
            "daily_A":        (os.path.join(RESULTS, f"era5_daily_potential_{s}_A.nc"), "date", 153),
            "daily_B":        (os.path.join(RESULTS, f"era5_daily_potential_{s}_B.nc"), "date", 153),
            "raw_labels":     (os.path.join(RESULTS, f"hw_{y}_daily_labels.nc"), None, None),
            "aligned_labels": (os.path.join(RESULTS, f"cn05_labels_{s}_aligned.nc"), "date", 153),
            "monthly_nc":     (os.path.join(MONTHLY_DIR, f"era5_monthly_hw_vs_ordinary_{s}.nc"), None, None),
            "monthly_summary":(os.path.join(MONTHLY_DIR, f"monthly_summary_{s}.csv"), None, None),
            "monthly_sens":   (os.path.join(MONTHLY_DIR, f"sensitivity_{s}.csv"), None, None),
            "monthly_report": (os.path.join(MONTHLY_DIR, f"REPORT_monthly_{s}.md"), None, None),
            "joint_nc":       (os.path.join(JOINT_DIR, f"joint_low_{s}.nc"), None, None),
            "joint_summary":  (os.path.join(JOINT_DIR, f"monthly_summary_joint_low_B_{y}.csv"), None, None),
            "joint_ab_sens":  (os.path.join(JOINT_DIR, f"ab_sensitivity_joint_low_{y}.csv"), None, None),
            "joint_report":   (os.path.join(JOINT_DIR, f"REPORT_joint_low_{y}.md"), None, None),
        }
        year_ok = True
        for name, (path, dim, expect) in files.items():
            if not check_file(path):
                failures.append(f"{y}: {name} MISSING ({path})")
                year_ok = False
                continue
            if dim is not None:
                msg = shape_ok(path, dim, expect)
                if msg:
                    failures.append(f"{y}: {name} shape problem: {msg}")
                    year_ok = False
        if year_ok:
            ok_count += 1
            print(f"  {y}: OK (14 products present + shapes consistent)")
        else:
            print(f"  {y}: INCOMPLETE")

    print("=" * 70)
    print("FINAL CHECK — multi-year summaries")
    print("=" * 70)
    multi = {
        "corr_pooled_nc":   os.path.join(RESULTS, "wind_solar_corr_multi_2011_2022.nc"),
        "corr_per_year_nc": os.path.join(RESULTS, "wind_solar_corr_multi_per_year_2011_2022.nc"),
        "corr_summary_csv": os.path.join(RESULTS, "monthly_summary_multi_B_2011_2022.csv"),
        "corr_per_year_csv": os.path.join(RESULTS, "per_year_monthly_summary_multi_B_2011_2022.csv"),
        "corr_report":      os.path.join(RESULTS, "REPORT_wind_solar_corr_multi_2011_2022.md"),
        "aggregate_csv":    os.path.join(RESULTS, "multi_year_summary.csv"),
        "aggregate_report": os.path.join(RESULTS, "REPORT_multi_year_summary.md"),
    }
    for name, path in multi.items():
        if not check_file(path):
            failures.append(f"multi-year: {name} MISSING ({path})")
            print(f"  {name}: MISSING")
        else:
            print(f"  {name}: OK")

    # ---- summary ----
    print("=" * 70)
    print(f"years complete: {ok_count}/{len(EXPECTED_YEARS)}")
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS — all 12 years + all analyses present and consistent")
    sys.exit(0)


if __name__ == "__main__":
    main()
