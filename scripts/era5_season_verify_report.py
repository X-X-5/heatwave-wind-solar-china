# -*- coding: utf-8 -*-
"""
Warm-season (May-Sep 2022) verification + CN05.1 label join + report.

Read-only on inputs. Writes ONLY:
  * results_daily/cn05_labels_20220501_20220930_aligned.nc  (label join, saved)
  * results_daily/REPORT_season_20220501_20220930.md          (report)

Cross-checks the new season products against the August pilot (numeric, not byte
hash): dimensions, coordinates, NaN LOCATIONS (single-sided NaN = inconsistent),
valid-hours, and per-variable max abs difference over finite cells. Any mismatch
stops the run (no aligned labels, no report).

Label join is an exact coordinate match + reorder (no interpolation); after aligning,
the original label VALUES are verified unchanged (NaN-preserving multiset equality +
code set). Label 0 remains "missing" (not replaced); CN05.1 day boundary stays
UNCONFIRMED.
"""
import os
import sys
import json
import calendar

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nc_guard import write_guarded, inspect_summary  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOURLY = os.path.join(ROOT, "results_hourly")
DAILY = os.path.join(ROOT, "results_daily")
MERGE = os.path.join(ROOT, "results_merge")

SEASON_FAIM = os.path.join(HOURLY, "era5_hourly_potential_20220430_20221001.nc")
SEASON_LIN = os.path.join(HOURLY, "era5_hourly_potential_tcell_linear_20220430_20221001.nc")
AUG_FAIM = os.path.join(HOURLY, "era5_hourly_potential_20220731_20220901.nc")
AUG_LIN = os.path.join(HOURLY, "era5_hourly_potential_tcell_linear_20220731_20220901.nc")
SEASON_DAILY_A = os.path.join(DAILY, "era5_daily_potential_20220501_20220930_A.nc")
SEASON_DAILY_B = os.path.join(DAILY, "era5_daily_potential_20220501_20220930_B.nc")
AUG_DAILY_A = os.path.join(DAILY, "era5_daily_potential_202208_A.nc")
AUG_DAILY_B = os.path.join(DAILY, "era5_daily_potential_202208_B.nc")
LABEL_NC = os.path.join(MERGE, "hw_2022_daily_labels.nc")
ALIGNED_NC = os.path.join(DAILY, "cn05_labels_20220501_20220930_aligned.nc")
REPORT_MD = os.path.join(DAILY, "REPORT_season_20220501_20220930.md")

LABEL_LEGEND = {
    0: "missing/undecidable",
    1: "ordinary non-exceedance",
    2: "isolated exceedance",
    3: "heatwave",
    4: "merged gap day",
}


def compare_overlap(new_ds, old_ds, coord_names, ctx):
    """Compare two co-aligned datasets. Returns (rows, issues)."""
    rows = []
    issues = []
    if dict(new_ds.sizes) != dict(old_ds.sizes):
        issues.append(f"{ctx}: dims differ existing={dict(old_ds.sizes)} "
                      f"season={dict(new_ds.sizes)}")
    for c in coord_names:
        if c not in old_ds.coords or c not in new_ds.coords:
            issues.append(f"{ctx}: coord {c} missing on one side")
        elif not np.array_equal(old_ds[c].values, new_ds[c].values):
            issues.append(f"{ctx}: coord {c} differs")
    for v in old_ds.data_vars:
        if v not in new_ds.data_vars:
            issues.append(f"{ctx}: variable {v} missing from season file")
            continue
        a = new_ds[v].values
        b = old_ds[v].values
        if a.shape != b.shape:
            issues.append(f"{ctx}: variable {v} shape {a.shape} vs {b.shape}")
            continue
        nan_a = ~np.isfinite(a)
        nan_b = ~np.isfinite(b)
        single_nan = int((nan_a ^ nan_b).sum())
        both_nan = int((nan_a & nan_b).sum())
        finite = np.isfinite(a) & np.isfinite(b)
        maxd = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
        rows.append((v, maxd, single_nan, both_nan))
        if single_nan != 0:
            issues.append(f"{ctx}: {v} single-sided NaN = {single_nan} "
                          f"(inconsistent missing data)")
        if maxd > 0.0:
            issues.append(f"{ctx}: {v} max abs diff = {maxd:.6g} (not bit-identical)")
        if v == "valid_hours" and not np.array_equal(a, b):
            issues.append(f"{ctx}: valid_hours differs (not identical)")
    return rows, issues


def print_table(rows):
    print("| variable           | max abs diff | single-NaN | both-NaN |")
    print("|--------------------|--------------|------------|----------|")
    for v, maxd, sn, bn in rows:
        print(f"| {v:18s} | {maxd:>12.6g} | {sn:>10d} | {bn:>8d} |")


def main():
    issues = []
    print("=" * 72)
    print("WARM-SEASON VERIFICATION + CN05.1 LABEL JOIN")
    print("=" * 72)

    # ---- 1. hourly cross-check (August overlap) ----
    hourly_blocks = {}
    for name, newp, oldp in [
        ("Faiman hourly", SEASON_FAIM, AUG_FAIM),
        ("linear hourly", SEASON_LIN, AUG_LIN),
    ]:
        new = xr.open_dataset(newp).sel(valid_time=slice("2022-07-31", "2022-09-01"))
        old = xr.open_dataset(oldp)
        rows, iss = compare_overlap(new, old,
                                    ["valid_time", "latitude", "longitude"], name)
        issues += iss
        hourly_blocks[name] = rows
        print(f"\n{name} — August overlap ({new.sizes.get('valid_time')} h):")
        print_table(rows)
        new.close()
        old.close()

    # ---- 2. daily cross-check (August overlap) ----
    daily_blocks = {}
    for key, newp, oldp in [("A", SEASON_DAILY_A, AUG_DAILY_A),
                            ("B", SEASON_DAILY_B, AUG_DAILY_B)]:
        new = xr.open_dataset(newp).sel(date=slice("2022-08-01", "2022-08-31"))
        old = xr.open_dataset(oldp)
        rows, iss = compare_overlap(new, old, ["date", "latitude", "longitude"],
                                    f"daily {key}")
        issues += iss
        daily_blocks[key] = rows
        print(f"\ndaily {key} — August overlap ({new.sizes.get('date')} d):")
        print_table(rows)
        new.close()
        old.close()

    # ---- STOP on any mismatch, BEFORE writing anything ----
    if issues:
        print("\n" + "=" * 72)
        print("MISMATCH FOUND — stopping (no aligned labels, no report written).")
        for i in issues:
            print("  -", i)
        print("=" * 72)
        raise SystemExit(1)

    # ---- 3. label join (exact coordinate match + reorder, no interpolation) ----
    print("\n" + "=" * 72)
    print("CN05.1 LABEL JOIN")
    lab = xr.open_dataset(LABEL_NC)
    ref = xr.open_dataset(SEASON_DAILY_A)
    lab_rn = lab.rename({"lat": "latitude", "lon": "longitude"})

    coord_ok = True
    for lc in ("date", "latitude", "longitude"):
        if not np.array_equal(np.sort(lab_rn[lc].values), np.sort(ref[lc].values)):
            coord_ok = False
            issues.append(f"label join: coord {lc} set mismatch")
    print(f"  coordinate sets match daily grid: {coord_ok}")

    aligned = lab_rn.reindex(date=ref["date"].values,
                             latitude=ref["latitude"].values,
                             longitude=ref["longitude"].values)

    val_ok = True
    for v in ("label", "tmax", "threshold"):
        a = lab[v].values
        b = aligned[v].values
        as_ = np.sort(a, axis=None)
        bs_ = np.sort(b, axis=None)
        nan_a = np.isnan(as_)
        nan_b = np.isnan(bs_)
        if int(nan_a.sum()) != int(nan_b.sum()):
            val_ok = False
            print(f"  {v}: NaN count changed {int(nan_a.sum())} -> {int(nan_b.sum())}")
            continue
        same = bool(np.array_equal(as_[~nan_a], bs_[~nan_b]))
        val_ok &= same
        print(f"  {v}: finite values unchanged after align = {same} "
              f"(nan={int(nan_a.sum())})")
    codes = sorted(int(c) for c in np.unique(aligned["label"].values))
    codes_ok = set(codes) <= set(LABEL_LEGEND)
    print(f"  label codes present = {codes}  (subset of 0..4: {codes_ok})")
    val_ok &= codes_ok

    if not (coord_ok and val_ok):
        print("STOP: label join verification failed (see above).")
        raise SystemExit(1)

    aligned.attrs = {
        "title": "CN05.1 heatwave daily labels aligned to the ERA5 daily grid",
        "source": os.path.basename(LABEL_NC),
        "join_method": "exact coordinate match + reorder (no interpolation, no resampling)",
        "label_codes": "0=missing/undecidable, 1=ordinary non-exceedance, 2=isolated exceedance, 3=heatwave, 4=merged gap day",
        "label_0_meaning": "missing/undecidable (preserved as-is, NOT replaced by ordinary day)",
        "cn05.1_day_boundary": "UNCONFIRMED",
        "config_signature": json.dumps({
            "join_method": "coordinate match + reorder (no interpolation)",
            "label_0_meaning": "missing/undecidable",
            "cn05.1_day_boundary": "UNCONFIRMED",
        }, sort_keys=True),
    }
    enc = {"label": {"zlib": True, "complevel": 4},
           "tmax": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
           "threshold": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
    write_guarded(aligned, ALIGNED_NC, enc)

    # re-open the SAVED file and re-verify label values == original
    saved = xr.open_dataset(ALIGNED_NC)
    s_ok = bool(np.array_equal(np.sort(saved["label"].values, axis=None),
                               np.sort(lab["label"].values, axis=None)))
    print(f"  saved file label values == original: {s_ok}")
    saved.close()
    if not s_ok:
        raise SystemExit(1)

    # ---- 4. coverage + per-month counts ----
    print("\n" + "=" * 72)
    print("COVERAGE")
    hcov = {}
    for name, p in [("Faiman hourly", SEASON_FAIM), ("linear hourly", SEASON_LIN)]:
        ds = xr.open_dataset(p)
        t = ds["valid_time"].values
        hcov[name] = (int(t.size), str(t[0])[:16], str(t[-1])[:16],
                      {v: int((~np.isfinite(ds[v].values)).sum()) for v in ds.data_vars})
        print(f"  {name}: n={t.size}  {t[0]} .. {t[-1]}")
        ds.close()

    dcov = {}
    for key, p in [("A", SEASON_DAILY_A), ("B", SEASON_DAILY_B)]:
        ds = xr.open_dataset(p)
        vh = ds["valid_hours"].values
        total = int(vh.size)
        complete = int((vh == 24).sum())
        dcov[key] = (int(ds.sizes["date"]), total, complete)
        print(f"  daily {key}: {ds.sizes['date']} days, valid_hours==24: "
              f"{complete}/{total} gridpoint-days")
        ds.close()

    months = [f"2022-{m:02d}" for m in range(5, 10)]
    mcounts = {}
    for mo in months:
        y, m = int(mo[:4]), int(mo[5:7])
        last = calendar.monthrange(y, m)[1]
        sub = aligned["label"].sel(date=slice(f"{mo}-01", f"{y}-{m:02d}-{last:02d}"))
        mcounts[mo] = {c: int((sub == c).sum()) for c in range(5)}
    print("\nper-month label sample counts (gridpoint-days):")
    for mo in months:
        cnt = mcounts[mo]
        tot = sum(cnt.values())
        print(f"  {mo}: total={tot}  " +
              "  ".join(f"c{c}={cnt[c]}" for c in range(5)))

    # ---- 5. write report ----
    lines = []
    lines.append("# ERA5 warm-season (May–Sep 2022) potential — verification & report\n\n")
    lines.append("## 1. Coverage\n\n### Hourly\n")
    for name in ("Faiman hourly", "linear hourly"):
        n, t0, t1, nan = hcov[name]
        lines.append(f"- **{name}**: {n} h, {t0} .. {t1}, per-variable NaN = {nan}\n")
    lines.append("\n### Daily (153 days)\n")
    for key in ("A", "B"):
        nd, total, complete = dcov[key]
        lines.append(f"- **boundary {key}**: {nd} days, valid_hours==24 in "
                     f"{complete}/{total} gridpoint-days\n")

    lines.append("\n## 2. August overlap cross-check (vs pilot, numeric)\n\n")
    lines.append("Single-sided NaN = cells missing on one side only (inconsistent). "
                 "Max abs diff over finite cells; 0 = bit-identical.\n")
    for name, rows in list(hourly_blocks.items()) + list(daily_blocks.items()):
        lines.append(f"\n### {name}\n\n")
        lines.append("| variable | max abs diff | single-NaN | both-NaN |\n")
        lines.append("|---|---:|---:|---:|\n")
        for v, maxd, sn, bn in rows:
            lines.append(f"| {v} | {maxd:.6g} | {sn} | {bn} |\n")

    lines.append("\n## 3. CN05.1 label join\n\n")
    lines.append(f"- coordinate sets match daily grid: {coord_ok}\n")
    lines.append(f"- label values unchanged (NaN-preserving): {val_ok}\n")
    lines.append(f"- label codes present: {codes}\n")
    lines.append("- label 0 = missing/undecidable, preserved (NOT replaced by ordinary day)\n")
    lines.append("- CN05.1 day boundary: UNCONFIRMED\n")
    lines.append(f"- saved to `{os.path.basename(ALIGNED_NC)}`\n")

    lines.append("\n## 4. Per-month label sample counts (gridpoint-days)\n\n")
    lines.append("| month | total | 0 missing | 1 non-exceed | 2 isolated | 3 heatwave | 4 gap |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for mo in months:
        c = mcounts[mo]
        lines.append(f"| {mo} | {sum(c.values())} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |\n")

    lines.append("\n## 5. File list\n\n")
    lines.append("| file | dims | span | vars | NaN cells |\n")
    lines.append("|---|---|---|---|---|\n")
    for p in [SEASON_FAIM, SEASON_LIN, SEASON_DAILY_A, SEASON_DAILY_B, ALIGNED_NC]:
        s = inspect_summary(p)
        if "error" in s:
            lines.append(f"| {os.path.basename(p)} | ERROR: {s['error']} | | | |\n")
        else:
            lines.append(f"| {os.path.basename(p)} | {s['dims']} | {s['span']} "
                         f"| {len(s['vars'])} vars | {s['nan_cells']} |\n")
    lines.append("\nUntouched: August pilot hourly/daily files, CN05.1 label source, "
                 "and all data_era5/ inputs.\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\nwrote {REPORT_MD}")
    print("\nALL CHECKS PASSED (max abs diff = 0, single-side NaN = 0, "
          "label values unchanged).")


if __name__ == "__main__":
    main()
