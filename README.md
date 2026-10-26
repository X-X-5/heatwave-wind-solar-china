# Wind resource, PV potential, and their temporal complementarity during heatwaves (pilot)

This project studies how regional **wind energy resource**, **PV (photovoltaic)
potential**, and their **temporal complementarity** change during heatwaves
(high-temperature episodes). The core question is: compared with ordinary
non-exceedance days, how do the distributions of wind power density and PV
potential shift on heatwave days, and how do the day-to-day temporal
correlation between wind and solar (Pearson/Spearman) and the "simultaneously
low" state change?

The project is framed as a **descriptive comparison at the resource level**:
no causal attribution, no significance testing, and no treating resource
correlation as directly equivalent to joint-generation stability. The current
output is a **completed exploratory pilot** (one small region, one year
verified; multi-year extension in progress).

---

## 1. Data and processing pipeline

| Data | Variable | Use |
| --- | --- | --- |
| CN05.1 (Wu & Gao, 2013) daily maximum temperature, 0.25°, 1961–2022 | `Tmax` | Heatwave thresholds and event identification |
| ERA5 reanalysis single levels (CDS), hourly, 0.25° | `t2m`, `u100/v100`, `u10/v10`, `ssrd` | Wind/solar potential calculation |

**Pilot region**: 30–33°N, 112–121°E, regular 0.25° grid (13 × 37 = 481 grid
cells).

**Processing order**: hourly ERA5 fields are first converted to wind/solar
potential, then daily-aggregated (under two day boundaries), and finally
aligned with the CN05.1 daily heatwave labels (exact coordinate matching +
reordering; no interpolation, no filling).

## 2. Completed period (checked 2026-10-26)

- **Heatwave identification (CN05.1)**: the 1981–2010 threshold baseline is
  complete; 2022 event identification is verified; annual heatwave-day counts
  for 1961–2022 have been produced.
- **Energy analysis (ERA5)**: the 2022 warm season (May–September) pilot is
  **fully verified** (download → hourly potential → daily aggregation →
  monthly comparison / correlation / joint-low analysis).
- **Multi-year extension (2011–2022)**: 2011, 2012, 2013, 2014, 2015, 2016,
  2017 and 2022 (reused pilot) — **8 years** in total — have completed
  "download + completeness check + basic processing"; 2018–2021
  download/processing is **in progress**; the multi-year summary analysis
  (pooled correlation, cross-year summary) is **not yet complete** and is
  marked "extension in progress"; a full 12-year summary or conclusion has not
  been produced.

See [`docs/stage_progress.md`](docs/stage_progress.md).

## 3. Heatwave identification method

- **Baseline**: fixed 1981–2010 (30 years).
- **Threshold**: per grid cell and per calendar day, the **P90** of Tmax
  (samples from the 5-day sliding window centred on that calendar day, numpy
  linear-interpolation Type 7; a threshold is defined only when the pooled
  sample has ≥ 100 valid values).
- **Hot day**: Tmax **strictly greater than** that day's threshold, with both
  values valid ("decidable").
- **Qualified heat segment**: a run of **≥ 3 consecutive** hot days.
- **Merging**: two qualified heat segments are merged when the complete gap
  between them is **< 2 days** (i.e. exactly 1 day) (merging is transitive /
  chained); missing/undecidable days are never bridged, and an isolated
  exceedance day blocks merging.
- **Gap days labelled separately**: the 1-day cool gap inside an event is
  labelled 4; the "process span" `span = end − start + 1` (including gap days)
  and the "number of exceedance heat days" `heatwave_days` (qualified-segment
  hot days only) are **counted separately**.
- **Daily labels**: `0` missing/undecidable, `1` cool day, `2` isolated
  exceedance day, `3` heatwave day, `4` gap day.
- **Grid-cell events**: identification is currently per grid cell; **grid-cell
  event counts are not regional event counts**.

Method document: `docs/heatwave_method_v2.md`.

## 4. Energy indicators

- **Hourly first, then daily aggregation**: each grid cell/day must have 24
  valid hours, otherwise NaN.
- **Wind power density**: `WPD = 0.5·ρ·V100³`, `ρ = 1.225 kg m⁻³` (standard air
  density), `V100 = √(u100² + v100²)`.
- **Simplified PV potential**: `PVpot = PR · GHI`, `PR = 1 + γ·(T_cell − 25)`,
  `γ = −0.005 ℃⁻¹`; `GHI = ssrd / 3600` (ERA5 hourly `ssrd` is a 1-hour
  accumulation ending at the timestamp). `PVpot` is a simplified
  horizontal-plane indicator with no tilt/orientation/system losses; it is
  **not** grid-delivered energy or a system capacity factor.
- **Cell temperature**: primary scheme Faiman (`T_cell = T_air + GHI/(U0 +
  U1·wind10)`, `U0=25.0`, `U1=6.84`); the linear empirical form (Jerez 2015 /
  Chenni 2007: `4.3 + 0.943·T_air + 0.028·GHI − 1.528·wind10`) is used as a
  **sensitivity check**.
- **Group comparison**: heatwave days (label 3) vs ordinary non-exceedance days
  (label 1); labels 0/2/4 are excluded.
- **Correlation**: per grid cell and per month, Pearson **r** and Spearman
  **ρ** are computed on paired daily PV and WPD samples; the main presentation
  requires **≥10 days** per group, sensitivity **≥5 days**; group differences
  `delta_r`/`delta_ρ` = heatwave − ordinary are reported.
- **Two day-boundary definitions**: A = calendar 00–24 Beijing time, B = 20–20
  Beijing time (tentatively primary); PV/radiation-type daily accumulations are
  invariant to the day boundary (the two boundaries differ only in the
  nighttime 20–24 h, with no shortwave radiation).

Method documents: `docs/potential_calc_scheme.md`, `docs/era5_data_plan.md`.

## 5. Verified results — 2022 exploratory pilot (single year, single small region; descriptive, non-causal)

All results below are from the **completed 2022 exploratory pilot** — one year
(2022), one small region, descriptive; no significance testing or causal
attribution was performed.

- **Heatwave identification**: for the example grid cell 31.75°N/114.50°E, the
  2022 event 08-02…08-22 (2 merged segments, 20 heatwave days + 1 gap day,
  `mean_excess = 2.50 ℃`).
- **Monthly resource comparison (August, day boundary B, Faiman PV)**: median
  "heatwave − ordinary" daily-total PVpot **+1.938 kWh/m²** (equal-weight over
  grid cells); median "heatwave − ordinary" daily-mean WPD **−7.448 W/m²**
  (large spread across grid cells, direction not uniform).
- **Wind–solar temporal correlation (day boundary B, ≥10 days per group)**:
  August median `delta_r` **+0.566** (P25–P75 0.346–0.760, Spearman `delta_ρ`
  +0.640); June median `delta_r` **−0.390**. The direction is sensitive to the
  sample threshold and day boundary (see `results_complementarity/`).
- **Simultaneous wind–solar low (August, day boundary B)**: median "heatwave −
  ordinary" difference in simultaneous-low proportion **−6.35 pp** (negative
  for most grid cells); small samples, per-cell proportions unstable (see
  `results_joint/`).

**Key figures** (a few verified figures are included in [`figures/`](figures/);
the underlying data files are not distributed):

- `figures/fig_hw_climatology.png` — 1961–2022 climatological heatwave-day
  count.
- `figures/fig_hw_2022.png` — 2022 heatwave-day count.
- `figures/fig_hw_2022_merge_cell.png` — representative grid-cell event
  (114.50°E, 31.75°N, 2022) under the merged definition.
- `figures/fig_monthly_diff_distribution.png` — monthly heatwave − ordinary
  difference distributions (PVpot and WPD).
- `figures/fig_wind_solar_corr_monthly_B.png` — monthly wind–solar correlation
  r and delta_r distributions.
- `figures/fig_joint_low_diff_monthly_B.png` — monthly simultaneous-low
  proportion difference distributions.

## 6. Directory structure

```
scripts/                  # all analysis code (heatwave identification + ERA5 potential + comparison/correlation/joint-low)
docs/                     # method notes (heatwave method, potential scheme, ERA5 data plan)
figures/                  # a few verified result figures (PNG) included with the repository
data/                     # CN05.1 raw daily Tmax (not uploaded)
data_era5/                # ERA5 2022 pilot raw download (not uploaded)
data_era5_multi/          # ERA5 multi-year raw download (not uploaded)
results_fixed/            # CN05.1 heatwave thresholds / annual / events (derived, not uploaded)
results_daily/ hourly/ monthly/ complementarity/ joint/ case/
                          # per-stage analysis results and reports (derived, not uploaded)
results_multi/            # multi-year extension run state, logs, backup (not uploaded)
```

## 7. How to run

All scripts run from the **project root** as the working directory;
input/output paths are relative to the project root. The scripts below are all
tracked in the repository and can reproduce the work independently, without
the local orchestration driver.

### 7.1 Heatwave identification (CN05.1)

```bash
python scripts/hw_merge_test.py      # synthetic unit tests (no data; verify the merging definition)
python scripts/hw_merge_pilot.py     # 2022 event identification for the pilot region -> results_merge/
python scripts/hw_full.py            # whole-domain 1961-2022 heatwave identification -> results_fixed/
```

### 7.2 ERA5 wind/solar potential (single-year pipeline)

```bash
python scripts/era5_download_2022_may_sep.py --download   # download hourly data (requires ~/.cdsapirc)
python scripts/era5_unpack_check.py                       # unpack + completeness check
python scripts/era5_hourly_potential.py                   # hourly potential (reads data_era5/unpacked_*)
python scripts/era5_daily_aggregate.py \
    --faim results_hourly/<hourly_faiman>.nc \
    --lin  results_hourly/<hourly_linear>.nc \
    --start 2022-08-01 --end 2022-09-01 --tag 202208
```

### 7.3 Comparison and analysis

```bash
python scripts/era5_monthly_compare.py \
    --daily-a <daily_A.nc> --daily-b <daily_B.nc> --labels <labels.nc> --out results_monthly/
python scripts/era5_wind_solar_correlation.py \
    --daily-a <daily_A.nc> --daily-b <daily_B.nc> --labels <labels.nc> --out results_complementarity/
python scripts/era5_joint_low_analysis.py \
    --daily-a <daily_A.nc> --daily-b <daily_B.nc> --labels <labels.nc> --year 2022 --out results_joint/
```

### 7.4 Multi-year extension (per year, published scripts)

```bash
python scripts/era5_multi_year.py --year 2011 --stage all --out results_multi/
```

The multi-year **batch orchestration driver** (`era5_orchestrator.py`,
`watchdog_relaunch.py` and `launch_*.ps1`) embeds the absolute path of the
local Python interpreter and is therefore **kept locally and not uploaded**,
and is **not** a required entry point for public reproduction. To reproduce,
call `era5_multi_year.py` year by year as in 7.4, followed by the analysis
scripts above; the unified scientific configuration is documented in `docs/`
and in the parameter notes inside each script.

## 8. Limitations

- **CN05.1 observation window not specified in the source data**: the data
  carry no timezone/observation-window attribute, so the CN05.1 statistical
  window is not specified. Following guidance, the analysis adopts a
  **daily-scale** treatment, and the ERA5 daily aggregation window is recorded
  explicitly as a **working convention** (definition A = calendar 00–24 Beijing
  time, B = 20–20 Beijing time; see §4). We do **not** claim the CN05.1 true
  day boundary is confirmed, do **not** claim the ERA5 daily values are aligned
  with CN05.1, and do **not** infer the day boundary from "which alignment gives
  higher correlation". The A/B two-definition difference is reported separately.
- **Sensitivity to day boundary A/B and cell-temperature model**: WPD is
  sensitive to the choice of day boundary; the Faiman and linear monthly PV
  totals differ by < 1%, but the hourly temperature difference P5–P95 is
  roughly −2.9 ~ +3.2 ℃, so this cannot be generalized to "the heatwave
  response is insensitive to the model choice".
- **Resource correlation ≠ joint-generation stability**: a negative `delta_r`
  only means the correlation moves in a more negative direction, not directly
  an improvement in supply stability; no capacity-factor/installed-ratio
  joint-generation assessment was done.
- **Spatial/year/sample boundaries**: single region (30–33°N/112–121°E), 2022
  single year verified; small samples (per-cell per-month heatwave days are
  often single-digit), temporal autocorrelation and spatial dependence present,
  unequal group sample sizes; results are non-causal and not generalized to
  long-term or large-region patterns.

## 9. Next steps

The 2022 small-region analysis is a **completed exploratory pilot**. The next
phase shifts to a larger regional scope and a climatological reference frame;
the items marked "to be finalized" below are **not yet determined**.

1. Complete the 2018–2021 download and processing, then the 2011–2022
   multi-year summary analysis (pooled correlation, cross-year summary).
2. Extend the analysis to the **major power-grid regions of China**; the
   power-grid region boundaries and the definition of a **regional event** are
   still to be finalized.
3. Compare heatwave-period wind/solar resource against a **fixed 30-year
   climatology**; the reference climatology (baseline period) is still to be
   finalized.
4. Prioritize the relationship between the **wind-resource anomaly** and
   **heatwave intensity and duration**.
5. Record the CN05.1 observation window with the data provider and select the
   day-boundary working convention accordingly.
6. Publish additional figures and summaries once the multi-year analysis is
   complete.

## 10. Data and release licensing

- **CN05.1**: Wu & Gao (2013), "A gridded daily observation dataset over China
  region and comparison with the other datasets", *Chinese Journal of
  Geophysics* (in Chinese) 56(4):1102–1111, DOI 10.6038/cjg20130406. CN05.1
  daily maximum temperature is used to identify heatwaves. The raw data, any
  cropped or transformed copy, and the hourly / daily / per-grid-point
  intermediate fields are **not** distributed with this repository; users must
  obtain the data from the provider themselves and comply with the provider's
  terms of use.
- **ERA5**: Copernicus CDS reanalysis (`reanalysis-era5-single-levels`), used
  under its license; derived products must cite the source when published.
- **Publication scope**: this repository publishes the **code, method
  documents, and a few verified result figures and summary tables**; it does
  not distribute the underlying raw or intermediate data. This is a
  publication-scope choice and does **not** state that any third-party data
  license has been confirmed.
