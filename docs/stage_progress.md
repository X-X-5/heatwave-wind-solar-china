# Stage Progress Note (verification date: 2026-10-26)

This document is a **concise stage progress note**, kept separate from the running
`results_multi/STATE.json`, `PROGRESS.md`, and the various logs, for status recording
in the public repository. It states only **completed and verified** content.

## Completed (verified)

### Heatwave Identification (CN05.1)

- Thresholds: baseline period 1981–2010, per-grid-cell, per-calendar-day Tmax P90
  (5-day sliding window, numpy linear interpolation Type 7, minimum sample ≥100);
  output `results_fixed/hw_threshold_1981-2010.nc`.
- 2022 event identification (merging definition, see `docs/heatwave_method_v2.md`) is
  completed and verified; output `results_merge/hw_2022_events_merged.csv`,
  `hw_2022_daily_labels.nc`, and the regional summary `hw_2022_region_summary.txt`.
- Annual heatwave-day counts for 1961–2022 are generated:
  `results_fixed/hw_annual_1961-2022.nc`.

### Energy Analysis (ERA5, 2022 Exploratory Pilot, Fully Verified)

- Download + completeness verification: 2022 warm-season hourly data (6 variables,
  3720 hours, 13×37, 0 NaN).
- Hourly potential: `results_hourly/` (Faiman primary scheme + linear-formula
  sensitivity).
- Daily aggregation: `results_daily/` (day-boundary A/B two definitions).
- Analysis: `results_monthly/` (heatwave vs ordinary month comparison),
  `results_complementarity/` (Pearson/Spearman correlation), `results_joint/`
  (simultaneous wind/solar low), `results_case/` (example grid-cell processes).

### Multi-Year Extension (2011–2022)

- Years with completed "download + completeness verification + basic processing"
  (8 years in total):
  **2011, 2012, 2013, 2014, 2015, 2016, 2017, 2022 (reused from the pilot)**.
- Each year passes: unpacking, `verify` (3720 hours / 6 variables / 13×37 / 0 NaN /
  value range), hourly potential, daily aggregation (A/B), labels, alignment.

## In Progress (Not Complete)

- **2018–2021**: download/processing in progress (as of 2026-10-26, May–August 2018 is
  downloaded; September–October and 2019–2021 are pending). The orchestration process
  and watchdog are still running; progress is in the local `results_multi/STATE.json`
  and `orchestrator.log`.
- **Multi-year summary analysis**: the merged correlation
  (`era5_wind_solar_correlation_multi.py`) and cross-year summary
  (`era5_multi_year_summary.py`) must wait until the 12 years of per-year analysis are
  complete and **have not yet started**.

## Next Phase (Planned, Design Not Finalized)

The next phase extends the analysis beyond the 2022 small-region pilot. The
following items are **planned but not yet started**, and their design details
are **still to be finalized, not yet determined**:

- Study by the **major power-grid regions of China** (the power-grid region
  boundaries and the **regional event definition** are still to be finalized).
- Compare heatwave-period wind/solar resource against a **fixed 30-year
  climatology** (the baseline reference period is still to be finalized).
- Prioritize the relationship between the **wind-resource anomaly** and
  **heatwave intensity and duration**.

## Honest Reporting Conventions

- Before the full 12-year analysis for 2011–2022 is complete, **do not** produce full
  12-year summaries or conclusions, and do not present the 8 years of "download +
  verification + basic processing" as a complete analysis.
- The currently citable scientific conclusions are limited to the **2022 single-year
  exploratory pilot** (descriptive, non-causal), with boundaries in `README.md` §8.
- The next-phase design items (30-year baseline, power-grid region boundaries,
  regional event definition) are **not yet finalized** and must not be written as
  determined or completed.

## Public Scope

- Published: method documents (`docs/`), code (`scripts/`, excluding the orchestration
  driver with embedded local-machine paths), README, a few verified result figures
  (`figures/`), and this progress note.
- Not published: CN05.1 raw data, ERA5 raw downloads, the underlying hourly / daily /
  per-grid-point intermediate data files, and the `results_multi/` run status and logs.
