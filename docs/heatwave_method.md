# Heatwave identification method description (research definition, not ETCCDI WSDI)

> **Updated.** This document is v1 (the strict "consecutive ≥ 3 days" definition). The current formal definition is v2 (merged
> event definition, merging gaps `< 2` days, i.e. only a 1-day gap), see
> [`docs/heatwave_method_v2.md`](heatwave_method_v2.md). The threshold, season window, and missing-data
> quality control are all unchanged; only the event grouping rule changed.

This document describes the heatwave identification definition, implementation conventions, and all boundary handling adopted in the first phase of this project. This definition is a **research heatwave definition**, explicitly **not named ETCCDI WSDI**.

## 1. Data

- Data source: `data/CN05.1_Tmax_1961-2022_daily_025x025.nc`
  (CN05.1, Wu & Gao, 2013, *Chinese J. Geophys.* 56(4):1102–1111;
  daily maximum temperature Tmax, 0.25°×0.25°, 1961-01-01 to 2022-12-31).
- Variable: `tmax` (float32), unit **℃** (the file lacks a `units` attribute; verified as degrees Celsius from the value range
  [-40.06, 47.03] and the official CN05.1 documentation).
- Missing: `_FillValue = missing_value = -9999.0`; no `scale_factor`/`add_offset`
  (raw values stored directly).
- Grid: lon 69.75–140.25°E (283 points, increasing), lat 14.75–55.25°N (163 points, increasing),
  resolution 0.25°. Time `hours since 1961-01-01`, `calendar=standard`
  (proleptic Gregorian), daily step 24 h, no duplicate/missing dates, includes 15 leap days.
- Valid coverage: 66.9% of all records are missing (mainly foreign/sea grid cells);
  15,292 valid land grid cells (14,168 fully valid throughout + 1,124 partially missing),
  and no time slice in which "all days for all grid cells are missing".

## 2. Heatwave definition

- **Baseline period**: fixed 1981–2010 (30 years).
- **Threshold**: for each grid cell and each calendar day (by (month, day), 366 days total, leap day = February 29),
  compute the Tmax **90th** percentile threshold.
  - Threshold sample: a **5-day sliding window** centered on that calendar day (the day ±2 days),
    wrapped correctly across years on the 366-day cycle (e.g. the January 1 window includes December 30/31 of the
    previous year and January 1/2/3 of the current year), pooling all baseline-year daily values that fall in the
    window into one sample (about 150).
  - Percentile algorithm: `numpy` linear interpolation (Hyndman–Fan Type 7, the default of `np.nanpercentile`).
  - **Minimum sample size**: a threshold is defined only if the pooled sample has ≥ **100** valid values; otherwise the
    threshold is NaN (that grid cell for that calendar day does not participate in determination).
- **Hot day**: Tmax **strictly greater than** (`>`, excluding equal) the day's threshold.
- **Heatwave event**: **consecutive ≥ 3 days** of hot days.
- **Analysis season**: **May–September** (inclusive of boundaries). Events are identified only within the season
  window; the window boundary truncates events (no cross-boundary linkage with April/October).

## 3. Missing-data handling

- Missing values are **not filled** and **not spatially interpolated**.
- Missing days are **not counted as cool days** and **not counted as hot days**; in event identification, a missing
  day **breaks the consecutive run**, so events **never connect across missing data**.
- Missing values are excluded from the threshold sample and from intensity averaging.
- **Annual validity gate (phase-1 fix)**: if a grid cell-year's "decidable-day count" (the number of days within
  May–September when both Tmax and the threshold are valid) = 0, then all heatwave indicators for that year are output
  as **missing** (`hw_count/hw_days/
  hw_max_dur` = −9999, `hw_intensity` = NaN), **not output as 0**, so they are never conflated with years that truly
  have no heatwave. Years with decidable-day count > 0 are computed by the original algorithm; among them, grid
  cell-years that "lack 1 day" are flagged **data incomplete** (`data_status=2`), **not filled and not connected
  across missing data**.

## 4. Output indicators (grid cell × year, 1961–2022)

| Indicator | Definition | Unit |
|---|---|---|
| `hw_count` number of heatwave events | number of heatwave events in a year (each consecutive ≥3 days) | events |
| `hw_days` total heatwave days | sum of all days belonging to heatwave events in a year | days |
| `hw_max_dur` longest duration | number of days of the longest heatwave in a year (0 if none) | days |
| `hw_intensity` mean exceedance intensity | mean of (Tmax − threshold) over all heatwave days in a year (NaN if no heatwave days) | ℃ |

Note: `hw_intensity` averages only over "heatwave days" (days belonging to a ≥3-day event), excluding isolated hot days.

**Newly added diagnostic variables** (phase-1 fix, output on the same grid as the annual indicators):

| Variable | Definition | Unit |
|---|---|---|
| `decidable_days` | number of days in that year's May–September when both Tmax and threshold are valid (0–153) | days |
| `max_missing_run` | longest run of consecutive missing Tmax days in that year's May–September | days |
| `data_status` | 0=no data (sea), 1=complete, 2=incomplete (has missing days), 3=undecidable (decidable-day count=0) | — |

## 5. Scope and limitations

- Only **Tmax** is used: we **cannot** claim to identify warm nights, day-night compound heatwaves, or humid-heat
  (wet-bulb) events.
- No national average and no national heatwave coverage fraction (no verified China boundary and intersecting area
  weights).
- Threshold/intensity units are ℃; intensity is the exceedance over the threshold, not an absolute temperature.

**Aggregation convention (explicit)**: regional averages appearing in this document (e.g. 8.3 days, 27.6 days, 1.5 ℃)
are always **unweighted arithmetic means over valid grid cells** (each 0.25°×0.25° grid cell weighted equally),
**not area-weighted**, **not
national/regional area-weighted results**; no boundary or area weights are used. Intensity aggregation convention:
first average over **years with heatwave days** within each grid cell (`nanmean` discards no-heatwave years, not
counted as 0), then take the arithmetic mean over grid cells; whereas the mean of heatwave days/count records
no-heatwave years as 0, so the intensity denominator (grid cells with heatwave years) is smaller than the days
denominator (all valid grid cells). All conclusions involve no statistical testing, and the word "significant" is
**not used**.
- No inference about wind-solar complementarity is made — temperature alone cannot yield energy conclusions.

## 6. File correspondence

- Algorithm core: `scripts/hw_lib.py`
- Algorithm verification: `scripts/hw_pilot.py` (15 assertions all pass + small-region measured test)
- Full computation: `scripts/hw_full.py` (includes the validity gate; outputs to `results_fixed/`)
- Plotting: `scripts/hw_plot.py` (2022 plots set the color-scale upper bound by the actual range)
- Fix verification: `scripts/verify_fix.py` (consistency + aggregation-convention recomputation)
- Segment-merge evaluation (diagnostic only, does not change the formal definition): `scripts/eval_merge.py`
- Data check report: `results/data_check_report.txt`
- Results (after fix, separate directory): `results_fixed/hw_annual_1961-2022.nc`
  (year×lat×lon: four indicators + `decidable_days`/`max_missing_run`/`data_status`)
  `results_fixed/hw_threshold_1981-2010.nc` (366 calendar days×lat×lon thresholds)
  `results_fixed/valid_mask.nc`, `results_fixed/hw_2022_events.csv`
  `results_fixed/fig_hw_climatology.png`, `results_fixed/fig_hw_2022.png`
- Original (pre-validity-gate) results are still kept in `results/` for comparison.
