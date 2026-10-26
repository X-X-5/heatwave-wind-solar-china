# Phase 1 Targeted Verification Report

This round is a **read-only verification**: it only inspects the code and
already-generated results, reading the raw NetCDF and the official products under
`results/` for cross-comparison. It **did not re-run** the heatwave calculation, **did
not overwrite** any official results, did not modify the raw data, did not modify the
old project, and did not perform git operations.

Verification scripts: `scripts/verify_calendar.py`, `scripts/verify_completeness.py`,
`scripts/verify_aggregation_figures.py` (all newly added, read-only).

---

## 1. Annual Data Completeness

For 15,292 valid grid cells × 62 years = **948,104 grid cell-years**, the following
are tallied cell by cell, year by year for May–September (153 days): number of valid
Tmax days, number of valid threshold days, decidable-day count (both valid
simultaneously), and maximum consecutive missing run.

**Distribution (decidable-day count)**:

| Decidable-day count | Grid cell-years |
|---|---|
| 153 (complete) | 944,971 |
| 152 (1 day missing) | 343 |
| 0 (completely undecidable) | **2,790** |
| Between 1–151 | **0** (no intermediate state exists) |

**Key finding**: a total of **2,790 grid cell-years** have "decidable-day count = 0"
(in that year the May–September Tmax is entirely missing **and** the thresholds are
entirely invalid), but they are output as `hw_count=0 / hw_days=0 / hw_max_dur=0`,
making them indistinguishable in the output from years that "truly have no heatwave".

These 2,790 = **45 grid cells × 62 years**. These 45 grid cells are located at
**approximately 22.25–25.5°N, 119.5–121.75°E (Taiwan Strait / coastal Taiwan)**, whose
summer (May–September) thresholds are entirely undefined (the 1981–2010 baseline
period has no valid summer data in this area), so the entire summer is undecidable
every year.

The remainder: the 343 "1 day missing" grid cell-years (corresponding to a small number
of temporal gaps in 2016–2018) have a maximum consecutive missing run of only 1 day,
and do not constitute "severe missingness"; apart from these 2,790, no grid cell-year
has a consecutive missing run > 5 days.

**Conclusion**: the expectation "fully missing / threshold-invalid years are not
output as 0" is not met. Affected are **2,790 grid cell-years (0.29% of 948,104)**,
belonging to the 45 grid cells in the Taiwan Strait / coastal Taiwan.

**Suggested validity rule (minimal change)**: when a grid cell-year's **decidable-day
count = 0** (or that cell's summer thresholds are entirely undefined), set its
`hw_count/hw_days/hw_max_dur` to NaN (undefined) instead of 0. Since the data are
bimodally distributed (0 / 152 / 153), this rule covers all affected grid cell-years.
An alternative stricter rule: require a decidable-day count ≥ 153 (or ≥ 152 tolerating
1 missing day), which also marks the 343 "1 day missing" grid cell-years as
to-be-rechecked (negligible impact; a 1-day gap cannot hide a ≥3-day event unless the
missing day happens to be the middle of a 3-day sequence, which is an inherent
limitation of the missing-data handling rule rather than an error).

---

## 2. Thresholds and Event Boundaries

**(A) Leap/non-leap year 5-day window** — evidence (`verify_calendar.py`):

- The `(month, day)→slot` mapping is **independent** of whether the year is a leap
  year: 2000-02-28 and 2001-02-28 both → slot 58; 2000-02-29 → slot 59 (appears only
  in leap years); 2000-03-01 and 2001-03-01 both → slot 60.
- Number of days per slot in the 1981–2010 baseline period: slot 59 (29 February)
  **7 days**, all other slots 30 days each.
- Sample sizes of the merged 5-day windows (cross-year wrap-around by mod 366):
  - 28 February (slot 58): window {56,57,58,59,60} → 127 days;
  - 29 February (slot 59): window {57,58,59,60,61} → 127 days;
  - 1 March (slot 60): window {58,59,60,61,62} → 127 days;
  - 1 January (slot 0): window {0,1,2,364,365} → 150 days (including 30/31 December
    of the previous year);
  - 31 December (slot 365): window {0,1,363,364,365} → 150 days (including
    1/2 January of the following year).
- All ≥ the minimum sample size of 100, so thresholds are definable; cross-year window
  handling is correct.

**(B) Seasonal-boundary semantics** — the current implementation is **"identify after
truncation"**: only the 153 days of May–September of each year are extracted, and
consecutive runs are identified within that window; events are **truncated** at the
boundaries and are not connected across the April/October boundaries. Hand-made example
(fixed threshold 30℃, hot day 31℃):

| Example | Identify after truncation (current) | Identify over the whole year then tally (reference) |
|---|---|---|
| 29,30 Apr + 1,2 May hot | only 1–2 May remain (2 days) → **no event** | 4-day event |
| 29,30 Sep + 1,2 Oct hot | only 29–30 Sep remain (2 days) → **no event** | 4-day event |
| 1,2,3 May hot | 1 event, 3 days | 1 event, 3 days |

**Conclusion**: this is a definitional choice, already stated in
`docs/heatwave_method.md` as "events are truncated at the boundaries". If it were
changed to "identify over the whole year then tally", 4-day events spanning April–May
and September–October would be counted, and the event count / number of days /
duration would increase slightly — if needed, this is a definition change rather than
a correction.

---

## 3. Tracing the Aggregation Convention

The four numbers are all **unweighted arithmetic means over valid grid cells** (each
0.25°×0.25° grid cell equally weighted), **not** area-weighted, **not**
nationwide/regional area-weighted results, and no boundary or area weights were used.

| Value | Exact computation | Denominator |
|---|---|---|
| **8.3 days** | For each grid cell take `nanmean` over the 62 years (mean of per-year heatwave-day counts), then take the arithmetic mean over valid grid cells | all **15,292** valid grid cells (including cells with 0 heatwave days) |
| **27.6 days** | 2022 field only, arithmetic mean over valid grid cells | all **15,292** valid grid cells |
| **3.3×** | 27.6173 / 8.3433 = 3.310 | ratio of the two means over the same fixed set of valid grid cells |
| **1.5 ℃** | For each grid cell take `nanmean` over the 62 years (**non-heatwave years dropped as NaN, not counted as 0**), then take the arithmetic mean over valid grid cells | the **15,247** grid cells with at least 1 year having a heatwave day |

A **fixed valid region** is used (`valid_mask`, i.e. grid cells with any valid Tmax
over 1961–2022, 15,292 in total).

Two points must be made clear:
1. The means of the heatwave-day count / event count include "non-heatwave years" as
   0; whereas the intensity mean **drops** non-heatwave years (`nanmean`), so the
   intensity denominator (15,247) is smaller than the day-count denominator (15,292),
   and the two conventions differ.
2. The above 45 undecidable grid cells (Taiwan Strait) are counted as 0 in the
   day-count mean (lowering the mean by about 0.29%× their weight), and are dropped in
   the intensity mean.

**Wording correction**: in the first-round report, the word "significant" in "2022 is a
significantly high year over the whole record" is inappropriate — this project
performed no statistical test. It should be changed to wording without statistical
meaning such as "clearly above normal / unusually above normal".

---

## 4. Memory and Figures

**(A) Memory** (actual layout in `hw_full.py`):

- `data` full array (22645,163,283) float32 = **4.18 GB** (resident memory);
- `Xb = data[baseline_idx]`: `baseline_idx` is an int64 index **array** from
  `np.where`, which is **fancy indexing → producing a 2.02 GB copy** (not a view; the
  code comment "contiguous -> view" is wrong, `np.shares_memory` measured False);
- Threshold `thr2d` (366,46129) float64 ≈ 0.135 GB (later converted to float32
  0.067 GB);
- Transient per-slot quantile samples ≈ 0.055 GB;
- Peak ≈ **6.4 GB** (available memory at runtime about 12.9 GB, no overflow).
- Event identification is a per-grid-cell loop using only small temporary arrays.

**(B) Figure colour-scale saturation**: the plotting uses **the same colour-scale
upper limit for both the climatology and 2022 figures** (`hw_plot.py` fixed vmax:
count=4, days=20, max_dur=10, intensity=3).

| Metric | vmax | Climatology over-limit share | 2022 over-limit share | 2022 actual maximum |
|---|---|---|---|---|
| hw_count | 4 | 0.00% | **18.56%** | 12 |
| hw_days | 20 | 0.00% | **23.47%** | 64 |
| hw_max_dur | 10 | 0.00% | **10.73%** | 26 |
| hw_intensity | 3 | 0.00% | **1.24%** | 4.36 |

The climatology figures are not saturated; **the 2022 figures are clearly saturated**
(about 1/5–1/4 of valid grid cells are clipped at the top).

**One corrected premise**: the first-round report's "0–78 days, 0–34 days" are extremes
over the **whole 62 years**, not specific to 2022. Measured: the record maximum number
of heatwave days is **78 days (2020, Hainan 109.25°E/18.5°N)** and the longest
duration is **34 days (1995, Tibetan Plateau 91.25°E/32°N)**; **the 2022 maxima are 64
days and 26 days**. Even with 2022's 64/26, they far exceed vmax=20/10, so the
saturation problem holds.

**Visual check of figures**: this environment cannot embed/display PNGs (returns
"Unsupported Image"), so the figure rendering quality **awaits manual inspection**;
the above gives only the numerical-level saturation shares.

---

## 5. Issue Summary and Suggested Minimal-Change Scope

| # | Issue | Severity | Minimal change |
|---|---|---|---|
| 1 | 2,790 undecidable grid cell-years (45 Taiwan Strait grid cells) are output as 0/0/0 | needs fixing | add a validity gate before output: decidable-day count = 0 (or summer thresholds entirely undefined) → set NaN; leave other years unchanged |
| 2 | The seasonal boundary is "identify after truncation", so events spanning April–May / September–October are not counted | definitional choice, not an error | no change needed; if the convention is to be changed, identify over the whole year then attribute by season (a definition change) |
| 3 | Aggregation convention: unweighted grid-cell mean, intensity drops non-heatwave years, day count includes 0 | must be made explicit in docs/report | keep the "unweighted, non-area-weighted" wording, note that the two metrics have different denominators; remove the word "significant" |
| 4 | 2022 figure colour-scale saturation (count 18.6%, days 23.5%, maxdur 10.7% clipped) | figures need re-plotting | set a separate vmax for the 2022 figures (or use a quantile upper limit); not re-plotted this round |
| 5 | `data[baseline_idx]` fancy indexing produces a redundant 2.02 GB copy | low (no memory overflow) | use contiguous slicing `data[7305:18262]` to avoid the copy |

None of the above **involves**: modifying the raw NetCDF, modifying the old project,
git operations, or re-running official results.
