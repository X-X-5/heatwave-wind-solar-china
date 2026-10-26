# Phase-1 fix report + segment-merge evaluation

> **Historical document, superseded.** This document is a phase-1 historical report; its Part 2, "alternative definition B
> (merging ≤2 cool gap days, i.e. gap ≤ 2)", is **diagnostic evaluation only** and **not** the current formal method.
> The current formal event definition is in [`docs/heatwave_method_v2.md`](heatwave_method_v2.md):
> first identify consecutive ≥3-day qualified heat segments, then merge adjacent events separated by a gap `< 2` days
> (i.e. at most 1 complete gap day). The Part-1 fix conclusions are already implemented in the current code.

This report continues the verification conclusions of `docs/phase1_verification_report.md` and has two parts:
**Part 1** implements fixes for confirmed issues; **Part 2** is a read-only evaluation of the alternative
"heat segment merge" definition (**does not change** the formal definition). This round downloaded no ERA5, performed
no energy computation, performed no git operations, modified no original NetCDF, and did not overwrite the original
results under `results/` (fix results were written to the separate directory
`results_fixed/`).

---

## Part 1: Fix results

### Fix 1 — Annual validity gate (decidable-day count = 0 → missing)

Added `detect_cell_full` (with gating) to `scripts/hw_lib.py`, and `scripts/hw_full.py`
now uses that function: when a grid cell-year's **decidable-day count = 0** (the number of days within May–September
when both Tmax and the threshold are valid is
0), `hw_count / hw_days / hw_max_dur` output **−9999** and `hw_intensity` outputs
**NaN**, no longer outputting 0. The gate is implemented inside the library function, so
**rerunning reproduces the fixed results**.

Consistency verification (`scripts/verify_fix.py`, read-only comparison of `results/` and `results_fixed/`):

| Indicator | Changed grid cell-years | All fall in "undecidable" grid cell-years | Original value |
|---|---|---|---|
| hw_count | 2,790 | yes (0 out of range) | 0 → −9999 |
| hw_days | 2,790 | yes | 0 → −9999 |
| hw_max_dur | 2,790 | yes | 0 → −9999 |
| hw_intensity | 0 | — (already NaN) | unchanged |

**Conclusion**: the fix acts **exactly** and only on the 2,790 undecidable grid cell-years previously misreported as 0; the
four indicators of the remaining
944,971 complete + 343 incomplete grid cell-years are **bit-identical** to the original results.

### Fix 2 — Save diagnostic variables, flag "data incomplete"

`results_fixed/hw_annual_1961-2022.nc` adds three new per-year variables:

| Variable | Description | Values |
|---|---|---|
| `decidable_days` | number of days in that year's May–September when both Tmax and threshold are valid | 0–153 |
| `max_missing_run` | longest run of consecutive missing Tmax days in that year's May–September | 0–153 |
| `data_status` | 0=sea/no data, 1=complete, 2=incomplete, 3=undecidable | 0–3 |

Full statistics: complete 944,971; incomplete **343** (all "lacking 1 day", i.e. `decidable_days=152`,
corresponding to sporadic missing data in 2016–2018); undecidable 2,790 (Taiwan Strait 45 grid cells × 62 years). These 343
"lacking 1 day" grid cell-years are **flagged incomplete** (`data_status=2`), **not filled and not connected across missing data**
(a missing day still breaks the consecutive run; events do not merge across missing data).

### Fix 3 — Write fixed results to a separate directory

`results_fixed/` (coexisting with `results/`, original results untouched):
`hw_annual_1961-2022.nc` (with new variables), `hw_threshold_1981-2010.nc` (threshold unchanged,
rewritten for self-consistency), `valid_mask.nc`, `hw_2022_events.csv`,
`fig_hw_climatology.png`, `fig_hw_2022.png`.

### Fix 4 — 2022 figure color scales

The 2022 figures no longer reuse the climatology color-scale upper bound; they are now set to
**round up to cover the actual maximum** (no truncation, no
`extend` needed). The measured values printed by `hw_plot.py`:

| Indicator | 2022 actual max | New color-scale upper bound (old value) | Saturation fraction (old→new) |
|---|---|---|---|
| hw_count | 12 | **14** (4) | 18.6% → 0% |
| hw_days | 64 | **70** (20) | 23.5% → 0% |
| hw_max_dur | 26 | **30** (10) | 10.7% → 0% |
| hw_intensity | 4.36 ℃ | **4.5** (3) | 1.2% → 0% |

The climatology figure still uses the original upper bound (already 0% saturation). Note: after the fix the climatology
figure has 45 Taiwan Strait grid cells changed from
"0 heatwave days" to "no data (gray)", which is the correct representation. Figure rendering quality **awaits manual
visual inspection**
(this environment cannot display PNGs inline).

### Fix 5 — Summary numbers and aggregation convention

The four numbers are all **unweighted arithmetic means over valid grid cells** (each 0.25°×0.25° grid cell weighted equally),
**not area-weighted**, **not national/regional area-weighted**, using no boundary or area weights. Before/after the fix:

| Value | Before fix | After fix | Reason for change |
|---|---|---|---|
| Climatology heatwave days | 8.3433 | **8.3679** | 45 undecidable grid cells changed from "counted as 0" to "discarded" |
| 2022 heatwave days | 27.6173 | **27.6988** | same as above |
| 2022/climatology ratio | 3.310 | **3.310** (unchanged) | numerator and denominator changed from the same source |
| Mean exceedance intensity | 1.4890 ℃ | **1.4890 ℃** (unchanged) | intensity already discards no-heatwave years / no-data grid cells |

After rounding: **8.3→8.4 days, 27.6→27.7 days, 3.3× unchanged, 1.5 ℃ unchanged**.

Intensity convention (already written into the method document): first average over **years with heatwave days** within
each grid cell (`nanmean`
discards no-heatwave years, not counted as 0), then take the arithmetic mean over grid cells; so the intensity
denominator (15,247 grid cells with heatwave years)
is smaller than the days denominator. All conclusions involve no statistical testing, and the word "significant" has
been **deleted** and replaced with "markedly above normal".

### Fix 6 — Full-run in-memory note (note change only, no rerun optimization)

The "contiguous -> view" comment in
`hw_full.py` was wrong and has been corrected: the `baseline_idx` of `Xb = data[baseline_idx]`
comes from an int64 index **array** from `np.where`, which is **fancy indexing → produces a
2.02 GB copy** (`np.shares_memory` measured as False). The actual memory layout: `data` 4.18 GB
+ `Xb` copy 2.02 GB + threshold 0.135 GB + per-slot transients ≈ peak 6.4 GB (no overflow; about
12.9 GB available). **No memory-optimization rerun was done** (there is already headroom; low priority; the copy can
later be eliminated with the contiguous slice
`data[7305:18262]`).

---

## Part 2: Heat segment merge evaluation (read-only, does not change the formal definition)

- **Formal definition A**: heatwave = **consecutive ≥ 3 days** with Tmax strictly greater than the threshold.
- **Alternative definition B** (diagnostic only): first identify ≥3-day heat segments as in A, then **merge** adjacent heat
  segments separated by **≤ 2 complete non-exceedance
  days ("cool days")** into one "process". Rules:
  - **A missing day cannot be a mergeable gap** (the two heat segments stay separate);
  - gap cool days are **not counted in the number of heatwave days**;
  - process span = first hot day to last hot day (including cool gap days); number of heatwave days = exceedance days only;
    number of gap days = cool days inside the process;
  - merging **can be chained transitively** (consecutive multiple ≤2-day cool gaps compose one long process).

### Verification cases (`scripts/eval_merge.py`, all pass)

| Case | A (strict) | B (merged) |
|---|---|---|
| hot4—cool2—hot3 | 2 events (4d, 3d) | 1 process, span 9d, heatwave days 7d, gap 2d |
| hot2—cool2—hot2 | 0 (both segments < 3 days) | 0 |
| hot3—missing1—hot3 | 2 events | 2 processes (missing cannot merge) |
| hot3—cool3—hot3 | 2 events | 2 processes (gap 3 > 2) |
| hot3—cool2—hot3—cool2—hot3 | 3 events | 1 process, span 13d, heatwave days 9d, gap 4d (chained) |

### Small-region measured test (30–33°N, 112–121°E, 481 grid cells / 476 valid grid cell-years)

| Year | Definition | Events/processes | Number of heatwave days | Total span | Longest event/process | Gap days |
|---|---|---|---|---|---|---|
| 2000 (ordinary) | A | 1,281 | 4,402 | — | 7 d | — |
| 2000 (ordinary) | B | 1,281 | 4,402 | 4,402 d | 7 d | 0 |
| 2022 (extreme) | A | 2,639 | 17,423 | — | 22 d | — |
| 2022 (extreme) | B | **1,993** | 17,423 | **18,237 d** | **26 d** | 814 |

- **Ordinary year (2000)**: in this region no two ≥3-day heat segments are separated by ≤2 cool days, so **A and B are exactly
  identical** (merging has no effect).
- **Extreme year (2022)**: merging reduces the number of processes by 646 (2,639→1,993, about −24%), the longest process
  lengthens from
  22 d to **26 d**, and the total span exceeds the heatwave-day count by 814 cool gap days; **the number of heatwave days is unchanged**
  (merging only regroups and does not change which days are hot days).

**Representative grid cell** (114.50°E, 31.75°N, 2022): A = 10 events / 52 heatwave days / longest 16 d →
B = 7 processes / longest span 21 d; examples: `06-24~07-02 span 9d heatwave days 7d gap 2d`,
`08-02~08-22 span 21d heatwave days 20d gap 1d`.

### Segment-merge conclusion

1. Merging **does not change the number of heatwave days**, only the "event" count and span convention (treating hot
   processes separated by 1–2 cool days as one longer process).
2. The difference is **notable in the extreme year and zero in the ordinary year**: 2022 event count −24%, longest
   process +4 d; 2000 shows no difference.
3. Missing data by rule **never participates in merging**, so missing data cannot cause false connections.

---

## Definitions to be confirmed (checklist)

1. **Event definition**: keep A (strict consecutive ≥3 days)? or adopt B (merge processes with ≤2 cool gap days)?
   If B is adopted, confirm whether "gap days" are counted in the process total duration (this report defaults to
   **not counting them in the number of heatwave days, but counting them in
   the process span**).
2. **Season boundary**: keep "identify after truncation" (events must fall entirely within May–September)? or change
   to "identify year-round
   then assign by season" (which would count continuous runs crossing April–May/September–October)?
3. **Validity gate**: confirm the "decidable-day count = 0 → missing" rule; should the 343 "lacking 1 day"
   grid cell-years also be promoted to "undecidable" (this report only flags them incomplete, with indicators
   computed as usual)?
4. **Aggregation convention**: confirm "unweighted arithmetic mean over valid grid cells, not area-weighted" (already
   adopted in this report);
   is boundary + area weighting further needed (which requires first obtaining a verified China boundary)?
5. **Intensity convention**: confirm "average only over years with heatwave days" (discarding no-heatwave years),
   which differs from the days convention
   (no-heatwave years recorded as 0).
