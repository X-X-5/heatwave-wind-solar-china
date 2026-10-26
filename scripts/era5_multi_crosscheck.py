# -*- coding: utf-8 -*-
"""
Cross-check the multi-year pipeline's 2022 outputs against the ORIGINAL 2022
pilot results (coords, NaN locations, labels, values). Read-only; writes nothing.

For each (original, multi) file pair:
  * dims / sizes must match
  * coordinate arrays must match EXACTLY (in order)
  * per variable: NaN mask compared element-wise (single-sided NaN = FAIL), and
    max abs diff over cells finite on BOTH sides (integer vars compared exactly).
The 2022 pilot files (results_hourly/, results_daily/, results_merge/) are never
touched or rewritten.
"""
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [
    ("hourly Faiman",
     os.path.join(ROOT, "results_hourly", "era5_hourly_potential_20220430_20221001.nc"),
     os.path.join(ROOT, "results_multi", "era5_hourly_potential_20220430_20221001.nc")),
    ("hourly linear",
     os.path.join(ROOT, "results_hourly", "era5_hourly_potential_tcell_linear_20220430_20221001.nc"),
     os.path.join(ROOT, "results_multi", "era5_hourly_potential_tcell_linear_20220430_20221001.nc")),
    ("daily A",
     os.path.join(ROOT, "results_daily", "era5_daily_potential_20220501_20220930_A.nc"),
     os.path.join(ROOT, "results_multi", "era5_daily_potential_20220501_20220930_A.nc")),
    ("daily B",
     os.path.join(ROOT, "results_daily", "era5_daily_potential_20220501_20220930_B.nc"),
     os.path.join(ROOT, "results_multi", "era5_daily_potential_20220501_20220930_B.nc")),
    ("raw labels",
     os.path.join(ROOT, "results_merge", "hw_2022_daily_labels.nc"),
     os.path.join(ROOT, "results_multi", "hw_2022_daily_labels.nc")),
    ("aligned labels",
     os.path.join(ROOT, "results_daily", "cn05_labels_20220501_20220930_aligned.nc"),
     os.path.join(ROOT, "results_multi", "cn05_labels_20220501_20220930_aligned.nc")),
]


def check_pair(name, orig_path, multi_path):
    o = xr.open_dataset(orig_path)
    m = xr.open_dataset(multi_path)
    ok = True
    rows = []

    if dict(o.sizes) != dict(m.sizes):
        ok = False
        rows.append(("dims", f"orig={dict(o.sizes)} multi={dict(m.sizes)}", "FAIL"))
    else:
        rows.append(("dims", str(dict(o.sizes)), "ok"))

    for c in o.coords:
        if c not in m.coords:
            ok = False
            rows.append((f"coord {c}", "missing in multi", "FAIL"))
            continue
        a = np.asarray(o[c].values)
        b = np.asarray(m[c].values)
        if a.dtype.kind == "M" or b.dtype.kind == "M":
            a = a.astype("datetime64[h]"); b = b.astype("datetime64[h]")
        same = a.shape == b.shape and np.array_equal(a, b)
        if not same:
            ok = False
        rows.append((f"coord {c} ({a.shape})", "exact" if same else "MISMATCH",
                     "ok" if same else "FAIL"))

    for v in o.data_vars:
        if v not in m.data_vars:
            ok = False
            rows.append((f"var {v}", "missing in multi", "FAIL"))
            continue
        av = o[v].values.astype(np.float64)
        bv = m[v].values.astype(np.float64)
        if av.shape != bv.shape:
            ok = False
            rows.append((f"var {v}", f"shape {av.shape} vs {bv.shape}", "FAIL"))
            continue
        if np.issubdtype(o[v].dtype, np.integer):
            same = bool(np.array_equal(o[v].values, m[v].values))
            if not same:
                ok = False
            rows.append((f"var {v} (int)", "exact" if same else "MISMATCH",
                         "ok" if same else "FAIL"))
            continue
        na = ~np.isfinite(av); nb = ~np.isfinite(bv)
        single_side = int((na & ~nb).sum()) + int((~na & nb).sum())
        both_finite = (~na) & (~nb)
        if both_finite.any():
            maxdiff = float(np.max(np.abs(av[both_finite] - bv[both_finite])))
        else:
            maxdiff = 0.0
        if single_side != 0 or maxdiff != 0.0:
            ok = False
        rows.append((f"var {v}", f"single_side_NaN={single_side} maxdiff={maxdiff:.3g}",
                     "ok" if (single_side == 0 and maxdiff == 0.0) else "FAIL"))

    o.close(); m.close()
    return ok, rows


def main():
    all_ok = True
    for name, op, mp in PAIRS:
        if not os.path.exists(op):
            print(f"[{name}] MISSING original {op}"); all_ok = False; continue
        if not os.path.exists(mp):
            print(f"[{name}] MISSING multi   {mp}"); all_ok = False; continue
        ok, rows = check_pair(name, op, mp)
        all_ok &= ok
        print(f"[{name}] {'PASS' if ok else 'FAIL'}")
        for what, detail, verdict in rows:
            if verdict != "ok":
                print(f"    {verdict}: {what}: {detail}")
    print("=" * 72)
    print("OVERALL:", "PASS — bit-identical (coords, NaN, labels, values)"
          if all_ok else "FAIL — investigate differences before proceeding")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
