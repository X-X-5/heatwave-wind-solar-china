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
    L.append("# 多年（2011–2022）暖季逐月逐年描述性指标汇总（仅汇总既有指标）\n\n")
    L.append("本报告**不新增任何指标、不重新计算任何逐年阈值/均值**，只把逐年已算好的"
             "空间中位数汇总成“多年中位数（中位数之中位数）”。\n\n")
    L.append("## 覆盖\n\n")
    L.append(f"- 完整纳入年份（资源比较与联合偏低两份逐年汇总均存在）：{years}（共 {len(years)} 年）。\n")
    if missing:
        L.append(f"- **缺失、未纳入年份：{missing}**（如实报告，不作为完整年份汇总）。\n")
    else:
        L.append("- 全部 2011–2022 年（12 年）均已完整纳入。\n")
    L.append("\n指标含义（与逐年脚本一致，未改动）：\n")
    L.append("- `wpd` / `pvpot`：逐年 `era5_monthly_compare.py` 的**热浪日−普通日**差值（日界 B）逐月空间格点中位数。\n")
    L.append("- `joint_low`：逐年 `era5_joint_low_analysis.py` 的**同时偏低比例差**（热浪−普通，日界 B，百分点）逐月空间格点中位数。\n")
    L.append("- “多年中位数”= 逐年空间格点中位数跨年份的中位数；"
             "**不是**把各年逐日数据直接混池再算。\n\n")

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
            L.append("无完整年份数据。\n\n")
            continue
        L.append("| 年 | diff 中位数 | 多年中位数 |\n")
        L.append("|---|---|---|\n")
        for y, v in vals:
            L.append(f"| {y} | {v:.6g} | {fnum(med, 6)} |\n")
        L.append(f"\n> 多年中位数 = {fnum(med, 6)}（n={n} 年）。\n\n")

    L.append("## 说明与边界\n\n")
    L.append("- 逐年空间格点中位数彼此不独立（同区域、逐月时间相关），多年中位数"
             "仅作跨年汇总展示，**不是**置信区间，**不是**显著性检验，**不是**长期气候结论。\n")
    L.append("- 逐年“偏低”阈值是**逐年**逐月相对 P25，各年独立计算；多年中位数"
             "只是把这些逐年结果汇总，不改变逐年定义。\n")
    L.append("- 逐年资源比较与联合偏低的完整说明见各年 `REPORT_monthly_*.md` / "
             "`REPORT_joint_low_*.md`；相关性多年汇总见 `REPORT_wind_solar_corr_multi_*.md`。\n")

    report_path = os.path.join(out, "REPORT_multi_year_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"wrote {report_path}")
    print("DONE multi-year summary (aggregation only, no new metrics)")


if __name__ == "__main__":
    main()
