# Phase-2 report: merged event definition (gap < 2) and 2022 small-region pilot

This report implements the event definition (consecutive ≥3 days + merge at gap < 2 days) and produces the 2022
small-region pilot results. **This round downloaded no ERA5, ran no energy computation, did not rerun QM/QDM or the old 17-model
project, performed no git operations, and did not overwrite any original data or existing results**; new results were written to the separate directory
`results_merge/`, with `results_fixed/` kept for verification.

---

## 1. Event definition (implemented)

- **Threshold (unchanged)**: per grid cell, per calendar day P90, baseline period 1981–2010, 5-day centered window, minimum
  sample 100, Tmax strictly greater than the threshold.
- **Qualified heat segment**: consecutive ≥ 3 days of hot days.
- **Merge**: adjacent qualified heat segments merge when the **gap < 2 days** (by complete gap-day count = only 1 day) and
  the gap day is a **valid cool day** (both Tmax and threshold valid, `Tmax ≤ threshold`); **chained merge** is transitive; missing days
  **never** merge; an isolated exceedance day **blocks** merging and is not absorbed.
- **Per-day classification** (0 missing/undecidable, 1 cool day, 2 isolated exceedance day, 3 heatwave day, 4 gap day),
  with missing/undecidable **not encoded as ordinary non-heatwave days**.

The full description is in `docs/heatwave_method_v2.md`; the implementation is in `scripts/hw_merge_lib.py`.

## 2. Test results

**Synthetic unit tests** (`scripts/hw_merge_test.py`, **19/19 pass**), covering the required cases:

| Case | Result |
|---|---|
| hot3—cool1—hot3 | ✅ merged (1 event, 2 segments, 6 heatwave days, 1 gap day, span 7) |
| hot3—cool2—hot3 | ✅ **not** merged (2 events) |
| hot2—cool1—hot2 | ✅ no event formed (both segments <3 days, isolated = code 2) |
| hot3—missing1—hot3 | ✅ **not** merged (missing never merges, missing day code 0) |
| hot3—cool1—hot3—cool1—hot3 | ✅ chained merge (1 event, 3 segments, 2 gaps, span 11) |
| qualified segment with an isolated exceedance day in between | ✅ **not** merged; isolated hot day = code 2 (neither gap day nor heatwave day) |
| missing breaks a consecutive run | ✅ preserved (`HHMH` → no event) |
| intensity over heatwave days only | ✅ gap days not counted in the intensity mean |
| missing day never encoded as an ordinary day | ✅ code 0 |

**The existing strict pipeline** (`scripts/hw_pilot.py`, season boundary + missing QC) **still passes 15/15**
(`hw_lib.py` unmodified).

## 3. Differences from the old gap≤2 pilot

2022 small region (30–33°N, 112–121°E, **476 valid grid cells**), three-definition comparison (
`results_merge/hw_2022_region_summary.txt`):

| Definition | Events/processes | Heatwave days | Total span | Gap days | Longest |
|---|---|---|---|---|---|
| strict (no merge) | 2,639 | 17,423 | — | — | 22 d |
| **gap<2 (new formal)** | **2,161** | 17,423 | 17,901 d | **478** | 26 d |
| gap≤2 (old diagnostic) | 1,993 | 17,423 | 18,237 d | 814 | 26 d |

> **Definition note**: the "events/processes, heatwave days, total span, gap days" in the table above are all
> **grid-cell-day totals** (summed grid cell by grid cell over the 476 valid grid cells, in units of
> "grid-cell-days / grid-cell-events"),
> **not** regional averages, **not** area-weighted, and **do not represent** a continuous region-level heatwave event;
> "longest" is the maximum process span over grid cells (26 d occurs at a single grid cell).

- **Cross-check passed**: this script's gap≤2 diagnostic results (1,993 / 18,237 / 814 / 26) are **item-by-item
  identical** to the old `eval_merge.py` report, proving the new implementation is from the same source as the old
  diagnostic and the difference comes only from tightening
  the merge rule.
- **gap≤2 → gap<2 change**: processes 1,993 → 2,161 (**+168**, fewer merges); gap days
  814 → 478 (**−336**); total span 18,237 → 17,901 (**−336**); longest process 26 d **unchanged**
  (the longest process itself is a 1-day gap, which both rules merge).
- **Heatwave days unchanged** (17,423): merging only regroups and does not change "which days are heatwave days".
- Grid cells with a merge: **292 / 476** (under gap<2).

## 4. Representative grid-cell figure (single grid-cell events, not regional independent events)

Representative grid cell **114.50°E, 31.75°N** (the same grid cell as in the old report, for comparability). **All
numbers in this section are counts for this single grid cell (grid-cell-days / grid-cell-events), not the §3 regional
grid-cell-day totals.**

- Strict definition: 10 heat segments / 52 heatwave days / longest 16 d (consistent with the old report).
- **After gap<2 merge: 8 events, 2 gap days**, with two merged processes:
  - `08-02 ~ 08-22`: 2 segments merged, heatwave days 20, gap 1, span 21 d (consistent with the old report's
    "span 21d heatwave days 20d gap 1d");
  - `09-05 ~ 09-12`: 2 segments merged, heatwave days 7, gap 1, span 8 d.
- Under the old gap≤2 this grid cell had 7 events (one more merge of a process through a 2-day gap); under gap<2 it has 8
  — **the direct manifestation of the tightened rule**.

Figure `results_merge/fig_hw_2022_merge_cell.png` (replotted by `scripts/hw_merge_plot.py`
from the saved per-day labels and event CSV, **does not change data/event definition**):
- two-line total title: first line "CN05.1 Tmax vs calendar-day P90 — 2022", second line the longitude/latitude;
- an event-marker narrow strip **above** the temperature panel, independent and sharing the date axis, marking the
  start/end lines and dates of the two merged processes;
- the temperature y-axis leaves padding per the Tmax/P90 range, no longer stretched for process markers;
- legend placed outside the curve area (right side); "grid-cell event, not a regional event" moved to the figure caption;
- consecutive heat segments (pink background), 1-day gap days (yellow hatched background), and the May–September date
  range unchanged.

**Figure rendering quality awaits manual visual inspection** (this environment cannot display PNGs inline).

## 5. Output files (separate directory `results_merge/`)

| File | Content |
|---|---|
| `hw_2022_daily_labels.nc` | 2022 per-day classification codes (0–4) + tmax + threshold (date×lat×lon) |
| `hw_2022_events_merged.csv` | per-grid-cell merged event catalog (2,161 records): lon/lat/start/end/n_segments/heatwave_days/gap_days/span_days/mean_excess_C |
| `hw_2022_region_summary.txt` | regional three-definition summary |
| `fig_hw_2022_merge_cell.png` | representative grid-cell time series |

Label statistics check: code 3 (heatwave day) = 17,423, code 4 (gap day) = 478, consistent with the summary.

## 6. Next step: ERA5 data plan (not downloaded yet)

- **Reusability**: this project and the old project both have **no hourly** ERA5; the old project only has **monthly**-scale ERA5/CMIP6
  with wind/solar formulas and a China boundary shapefile (see `docs/era5_data_plan.md` for details).
- **Data needs**: hourly `t2m`, `u100/v100`, `ssrd`, and also `u10/v10` (needed for the PV module temperature
  cooling term `wind10`).
- **Key processing (verified against the official CDS definition)**: ERA5 reanalysis hourly `ssrd` is an **hour-ending 1-hour
  accumulation**, directly `÷3600` to W m⁻², **no differencing needed** (only ERA5-Land needs differencing); wind/solar potential
  is **hourly first, then daily**; the combination uses unified capacity-factor/installed-capacity assumptions and does
  **not directly add** WPD and PVpot.
- **Prepared artifacts already produced**:
  - download script `scripts/era5_download_2022.py` (default **dry run**, only prints the request and
    ~9 MB expected size; `--download` actually requests; **does not read/print credentials**);
  - potential calculation scheme `docs/potential_calc_scheme.md` (the WPD vs wind CF vs PVpot vs PV
    CF distinction and the list of pending assumptions).
- **To resolve**: the CN05.1 observational day definition (00–24 h vs 20–20 h Beijing time) must be confirmed before
  ERA5 daily values can be aligned. Beijing time = UTC+8: 00–24 h Beijing time is `[D−1 16:00 UTC, D 16:00 UTC)`,
  20–20 h Beijing time is `[D−1 12:00 UTC, D 12:00 UTC)`, both extend **backward by only one UTC day** and
  end no later than `D 16:00 UTC`, so **only 1 day of pre-buffer is needed (2022-07-31)**; the 09-01 in the script's request window
  2022-07-31 ~ 2022-09-01 is only a **symmetric margin**, not required for August per-day aggregation.
  Also distinguish instantaneous variables (`t2m`/wind by timestamp) from hourly-accumulated `ssrd` (assigned by the
  accumulation interval's **ending time**,
  `÷3600` to GHI); the same time filtering must not be mixed for the two.
- **This round downloaded no** ERA5 and **ran no** energy computation; download awaits manual verification of the scheme.

The full data plan is in `docs/era5_data_plan.md`; the computation method is in `docs/potential_calc_scheme.md`.
