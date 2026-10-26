# -*- coding: utf-8 -*-
"""
Multi-year SUMMARY (no new metrics) of the per-year descriptive analyses.

Reads the ALREADY-COMPUTED per-year summaries (no recomputation, no new metric,
no new definition):

  * resource comparison  `results_multi/monthly/monthly_summary_YYYY0501_YYYY0930.csv`
    (metrics: wpd, pvpot  -> per-month spatial-median diff heatwave-ordinary)
  * joint wind+solar low  `results_multi/joint/monthly_summary_joint_low_B_YYYY.csv`
    (metric: joint_low   -> per-month spatial-median joint-low proportion diff, pp)

For each calendar month and each metric it lists every complete year's headline
spatial median (diff_median) and then the MULTI-YEAR MEDIAN of those yearly
values (median-of-medians). This is a pure cross-year summary of the SAME metric
— it does NOT pool days, does NOT recompute per-year thresholds/means, and does
NOT add any new indicator.

It also reports which years are complete (present in BOTH summaries) and which
are missing, honestly.

No significance testing, no causal/conclusion statements.
"""
import os
import sys
import glob
import re
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results_multi")
MONTHLY_DIR = os.path.join(RESULTS, "monthly")
JOINT_DIR = os.path.join(RESULTS, "joint")

EXPECTED_YEARS = list(range(2011, 2023))
MONTHS = [f"{m:02d}" for m in range(5, 10)]
MONTH_LABEL = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}


def read_monthly_summary(year):
    """Return list of dict rows from the monthly_compare summary CSV, or None."""
    path = os.path.join(MONTHLY_DIR, f"monthly_summary_{year}0501_{year}0930.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_joint_summary(year):
    """Return list of dict rows from the joint_low summary CSV, or None."""
    path = os.path.join(JOINT_DIR, f"monthly_summary_joint_low_B_{year}.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, nd=4):
    try:
        x = float(v)
        if x != x:                      # NaN
            return "—"
        return f"{x:.{nd}g}"
    except (TypeError, ValueError):
        return "—"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=RESULTS,
                    help="results_multi directory (default: auto)")
    ap.add_argument("--out", default=None, help="output dir (default: --results)")
    args = ap.parse_args()
    results = args.results
    monthly_dir = os.path.join(results, "monthly")
    joint_dir = os.path.join(results, "joint")
    out = args.out or results
    os.makedirs(out, exist_ok=True)

    # discover complete years = both per-year summaries present
    years = []
    for y in EXPECTED_YEARS:
        if read_monthly_summary(y) is not None and read_joint_summary(y) is not None:
            years.append(y)
    missing = [y for y in EXPECTED_YEARS if y not in years]
    print(f"complete years (both summaries present): {years}")
    if missing:
        print(f"MISSING (excluded, reported honestly): {missing}")

    # gather: key=(month, metric) -> list of (year, diff_median)
    # metric: "wpd", "pvpot" (monthly_compare), "joint_low" (joint, in pp)
    by_key = {}
    for y in years:
        mrows = read_monthly_summary(y)
        for r in mrows:
            mo = r["month"][5:7]           # "YYYY-MM" -> "MM"
            metric = r["metric"]           # wpd / pvpot
            by_key.setdefault((mo, metric), []).append((y, r["diff_median"]))
        jrows = read_joint_summary(y)
        for r in jrows:
            mo = r["month"][5:7]
            by_key.setdefault((mo, "joint_low"), []).append((y, r["diff_median"]))

    # ---- combined long-form CSV ----
    csv_path = os.path.join(out, "multi_year_summary.csv")
    fieldnames = ["month", "metric", "year", "diff_median",
                  "multi_year_median", "n_years"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (mo, metric) in sorted(by_key):
            vals = [(y, float(v)) for y, v in by_key[(mo, metric)]
                    if v not in ("", "—") and float(v) == float(v)]
            n = len(vals)
            med = None
            if n:
                med = float(sorted(v[1] for v in vals)[n // 2]) if n % 2 else \
                      (float(sorted(v[1] for v in vals)[n // 2 - 1])
                       + float(sorted(v[1] for v in vals)[n // 2])) / 2.0
            for y, v in vals:
                w.writerow({
                    "month": mo, "metric": metric, "year": y,
                    "diff_median": f"{v:.6g}",
                    "multi_year_median": f"{med:.6g}" if med is not None else "",
                    "n_years": n,
                })
    print(f"wrote {csv_path}")

    # ---- markdown report ----
    L = []
    L.append("# Multi-year (2011–2022) warm-season per-month per-year descriptive summary (aggregation of existing indicators only)\n\n")
    L.append("This report **adds no new indicators and does not recompute any per-year "
             "thresholds/means**; it only aggregates the already-computed per-year spatial "
             "medians into a \"multi-year median (median of medians)\".\n\n")
    L.append("## Coverage\n\n")
    L.append(f"- Complete years included (both the resource-comparison and joint-low per-year "
             f"summaries exist): {years} ({len(years)} years).\n")
    if missing:
        L.append(f"- **Missing / excluded years: {missing}** (reported honestly, not summarized as complete years).\n")
    else:
        L.append("- All 2011–2022 years (12 years) are fully included.\n")
    L.append("\nIndicator meanings (identical to the per-year scripts, unchanged):\n")
    L.append("- `wpd` / `pvpot`: per-year `era5_monthly_compare.py` **heatwave − ordinary** difference (boundary B), monthly spatial grid-cell median.\n")
    L.append("- `joint_low`: per-year `era5_joint_low_analysis.py` **simultaneous-low proportion difference** (heatwave − ordinary, boundary B, percentage points), monthly spatial grid-cell median.\n")
    L.append("- \"Multi-year median\" = median across years of the per-year spatial grid-cell medians; "
             "**not** pooling all years' daily data first and then computing.\n\n")

    metric_label = {"wpd": "WPD diff (W m-2)", "pvpot": "PV-pot diff (kWh m-2)",
                    "joint_low": "joint-low diff (pp)"}
    for (mo, metric) in sorted(by_key):
        m_int = int(mo)
        vals = [(y, float(v)) for y, v in by_key[(mo, metric)]
                if v not in ("", "—") and float(v) == float(v)]
        n = len(vals)
        if n:
            sv = sorted(v[1] for v in vals)
            med = (sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0)
        else:
            med = None
        L.append(f"### {MONTH_LABEL[m_int]} — {metric_label[metric]}\n\n")
        if n == 0:
            L.append("No complete-year data.\n\n")
            continue
        L.append("| year | diff median | multi-year median |\n")
        L.append("|---|---|---|\n")
        for y, v in vals:
            L.append(f"| {y} | {v:.6g} | {fnum(med, 6)} |\n")
        L.append(f"\n> Multi-year median = {fnum(med, 6)} (n={n} years).\n\n")

    L.append("## Notes and boundaries\n\n")
    L.append("- Per-year spatial grid-cell medians are not mutually independent (same region, "
             "monthly time correlation); the multi-year median is shown only as a cross-year "
             "aggregation, **not** a confidence interval, **not** a significance test, **not** a "
             "long-term climate conclusion.\n")
    L.append("- Per-year \"low\" thresholds are **per-year** per-month relative P25, computed "
             "independently for each year; the multi-year median only aggregates these per-year "
             "results and does not change the per-year definition.\n")
    L.append("- Full per-year explanations of the resource comparison and joint-low analyses are in "
             "each year's `REPORT_monthly_*.md` / `REPORT_joint_low_*.md`; the multi-year correlation "
             "summary is in `REPORT_wind_solar_corr_multi_*.md`.\n")

    report_path = os.path.join(out, "REPORT_multi_year_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"wrote {report_path}")
    print("DONE multi-year summary (aggregation only, no new metrics)")


if __name__ == "__main__":
    main()
