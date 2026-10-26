# Heatwave identification method description (v2: merged event definition)

This document is the **current formal method description**, replacing `docs/heatwave_method.md` (the v1 strict definition). The v1 threshold, season window, and missing-data quality control are **all unchanged**; the only change in this round is the **event grouping rule**, changed from "strict consecutive ≥3 days" to "first identify ≥3-day qualified heat segments, then merge adjacent qualified heat segments separated by a gap < 2 days into one event". The threshold file `results_fixed/hw_threshold_1981-2010.nc` is unchanged and is still reused.

---

## 1. Data (same as v1, unchanged)

- Data source: `data/CN05.1_Tmax_1961-2022_daily_025x025.nc` (CN05.1, Wu & Gao, 2013;
  daily maximum temperature Tmax, 0.25°×0.25°, 1961-01-01 to 2022-12-31, unit ℃).
- Missing: `_FillValue = -9999.0`; no `scale_factor`/`add_offset`.
- Grid: lon 69.75–140.25°E, lat 14.75–55.25°N, resolution 0.25°.

## 2. Threshold (same as v1, unchanged)

- **Baseline period**: fixed 1981–2010 (30 years).
- **Threshold**: the Tmax **90th percentile** for each grid cell and each calendar day ((month, day),
  366-day leap-year coordinate system, leap day = February 29).
  - Threshold sample: a **5-day sliding window** centered on that calendar day (the day ±2 days), wrapped correctly
    across years on the 366-day cycle, pooling all baseline-year daily values that fall in the window.
  - Percentile algorithm: `numpy` linear interpolation (Hyndman–Fan Type 7).
  - **Minimum sample size**: a threshold is defined only if the pooled sample has ≥ **100** valid values; otherwise
    the threshold is NaN.
- **Hot day**: Tmax **strictly greater than** (`>`) the day's threshold, with both Tmax and threshold valid
  ("decidable").

## 3. Event definition (v2 formal definition)

### 3.1 Qualified heat segment

- **Qualified heat segment** = the longest consecutive run of **consecutive ≥ 3 days** of hot days.
- An isolated hot day shorter than 3 days is **not** a heatwave day and is **not absorbed** into any event
  (see §4 classification).

### 3.2 Merge rule (the core of this change)

- Two adjacent qualified heat segments are **merged into one
  event** when the **gap** between them satisfies **all** of the following:
  1. **Gap < 2 days** (interpreted as the "complete gap-day count"): i.e. **only a 1-day gap is allowed**; a
     **2-day gap is not merged**. (**Explicit: it is `gap < 2`, not `gap ≤ 2`.**)
  2. The gap day **must have valid Tmax and a valid threshold** ("decidable"), and must be a **non-exceedance day**
     (i.e. a "cool day" with `Tmax ≤ threshold`).
- **Missing/undecidable days never merge**: if the gap contains missing data or an invalid threshold, the two heat
  segments stay separate (**no merge across missing data**).
- **An isolated exceedance day blocks merging**: if the gap contains an isolated hot day that does not reach 3
  consecutive days, that hot day is neither a gap day nor a heatwave day (classified as "isolated exceedance day",
  code 2), and makes that gap non-mergeable.
- **Chained merge**: multiple qualified heat segments can be **transitively chained into one event** through
  consecutive 1-day cool gaps (e.g. hot3—cool1—hot3—cool1—hot3 merged into one event). This convention is recorded
  as a formal rule.

### 3.3 Season and boundary (same as v1, unchanged)

- Analysis season **May–September** (inclusive of boundaries). Events are identified only within the season window;
  **boundary truncation**, no cross-boundary linkage with April/October.
- Implementation: each year only the 153 days of May–September are taken to form the sequence; segment identification
  and merging are done within this sequence.

## 4. Per-day classification labels (new in v2)

Each grid cell, each season day outputs one classification code (missing/undecidable is **never** encoded as an
ordinary non-heatwave day):

| Code | Name | Meaning |
|---|---|---|
| 0 | missing/undecidable | Tmax or threshold invalid, cannot determine (**not** an ordinary day) |
| 1 | cool day | valid, `Tmax ≤ threshold`, non-merge gap day |
| 2 | isolated exceedance day | valid, `Tmax > threshold`, but **not belonging** to any ≥3-day consecutive run |
| 3 | heatwave day | valid, `Tmax > threshold`, belonging to some ≥3-day qualified heat segment |
| 4 | gap day | valid, `Tmax ≤ threshold`, a 1-day cool gap inside a merged event |

"Exceedance day" = code 2 or 3; "heatwave day" = only code 3; "gap day" = only code 4.

## 5. Event catalog fields

For each grid cell, each (merged) event records:

| Field | Meaning |
|---|---|
| `lon`, `lat` | grid cell longitude/latitude |
| `start`, `end` | dates from the event's **first hot day** to its **last hot day** (including gap days) |
| `n_segments` | number of qualified heat segments merged into this event |
| `heatwave_days` | number of heatwave days (only hot days in qualified heat segments) |
| `gap_days` | number of gap days (the 1-day cool gaps inside the event) |
| `span_days` | process span = `end − start + 1` (including gap days) |
| `mean_excess_C` | mean exceedance intensity = mean of `(Tmax − threshold)` over **heatwave days only** (**excluding** gap days and **excluding** isolated exceedance days) |

## 6. Missing data and quality control (same as v1, unchanged)

- Missing data is **not filled** and **not interpolated**.
- Missing days are **not counted as cool days** and **not counted as hot days**; in segment identification they
  **break the consecutive run**; in merging they are **never usable merge gaps**.
- Missing values are excluded from the threshold sample and from intensity averaging.
- **Annual validity gate**: when a grid cell-year's decidable-day count = 0, that year's indicators are output as
  missing, never as 0.

## 7. Observational day definition: verification conclusion (2026-10; CN05.1 itself still not finally confirmed)

**Local data**: `data/CN05.1_Tmax_*.nc` contains only `tmax` (`long_name`="CN05.1 daily,
maximum temperature") and `time` (daily 00:00, `hours since 1961-01-01`, standard calendar),
**no timezone/observation-window attributes**; it was converted from GrADS `.ctl` binary via CDO, and no
documentation accompanies the local data. Therefore **the local data itself contains no day-boundary information**.

**Source-chain verification (secondhand retrieval, not direct reading of the original)**:
- CN05.1 is interpolated from **daily observations** of 2400+ national stations (Wu & Gao 2013,
  *Chinese Journal of Geophysics* (in Chinese) 56(4):1102–1111, DOI 10.6038/cjg20130406; official NZC data page).
- Its underlying station data, the China surface climate daily-value dataset V3.0
  (`SURF_CLI_CHN_MUL_DAY`; title in Chinese), together with the meteorological industry standard
  **QX/T 22—2004**, state clearly: **day boundary and time system = Beijing time, with the day boundary at 20:00**
  (except sunshine, which uses true solar time); i.e. the daily-value statistics window =
  **previous day 20:00 to the current day 20:00 (Beijing time)**, with the daily value **assigned
  to the ending day** (date field D = the ending day of that 24 h window).
- **However**: CN05.1's own official README / the paper's "Data and methods" **do not explicitly state**
  this day boundary, and this environment cannot read the original text directly to verify. It can therefore only be
  judged as "**very likely inherits the stations' 20:00 day boundary**", **not confirmable**; it cannot
  be ruled out that CN05.1 separately uses calendar day 00–24 h (Beijing time).

**To confirm with the data provider (one sentence)**:
> For the CN05.1 daily Tmax value labeled with date D, is the day boundary "Beijing time 20:00 (previous day
> 20:00—current day 20:00, value assigned to the ending day D)", or "calendar day 00:00—24:00 (Beijing time)"?

**Before confirmation**: do not claim that ERA5 daily values are already aligned with CN05.1, and do **not**
reverse-infer the day-boundary definition from "which alignment gives a higher correlation".

## 8. Current phase scope

- The current phase does **only heatwave identification** (threshold + event grouping + per-day classification), and
  has **not produced any wind/solar or energy conclusions**. Wind/solar potential is a later phase requiring hourly
  ERA5 data (not yet downloaded; see the data plan).
- Threshold/intensity units are ℃; intensity is the exceedance over the threshold, not an absolute temperature.
- Only Tmax is used: we **cannot** claim to identify warm-night or humid-heat (wet-bulb) events.

## 9. File correspondence

- Algorithm core: `scripts/hw_merge_lib.py` (merged definition, reuses the slots/season of `hw_lib.py`)
- Synthetic unit tests: `scripts/hw_merge_test.py` (19 assertions)
- Small-region pilot: `scripts/hw_merge_pilot.py`
- Plotting (reads saved output and replots, does not change data): `scripts/hw_merge_plot.py`
- Output (separate directory, keeping `results_fixed/` for verification):
  - `results_merge/hw_2022_daily_labels.nc` (2022 per-day classification + tmax/threshold)
  - `results_merge/hw_2022_events_merged.csv` (per-grid-cell merged event catalog)
  - `results_merge/hw_2022_region_summary.txt` (regional summary, strict/gap<2/gap≤2 three definitions)
  - `results_merge/fig_hw_2022_merge_cell.png` (representative grid-cell time series)
