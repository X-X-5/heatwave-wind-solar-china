# ERA5 Small-Area Pilot Data Plan (Not Yet Downloaded)

This plan prepares the **next phase** of the "heatwave-period wind/solar energy potential
over China" project: on top of the CN05.1 heatwave identification (the completed
v2 merging definition, see `docs/heatwave_method_v2.md`), it adds hourly wind/solar
potential calculation.
**This round does not download or run any energy calculation**; it only lists the data
requirements and the processing methods.

---

## 1. Review of Reusability of Existing Data (read-only; the old project was not modified)

| Location | Contents | Reusable? |
|---|---|---|
| This project `data/` | CN05.1 daily Tmax | Used (heatwave identification) |
| Old project `Github Project/data/raw/era5_*` | ERA5 **monthly** data (`moda`/`avgad`/`avgua`, monthly GRIB, monthly zip) | **No** — monthly scale only, no hourly 2 m temperature / 100 m wind / radiation |
| Old project `data/raw/cmip6_downloaded` | 17-model CMIP6 monthly data (tas/rsds/sfcWind) | **No** — monthly scale, not reanalysis hourly data |
| Old project `data/boundaries/china_qc_boundary.shp` | China boundary shapefile | Reusable (for later regional/boundary masking; not needed for the current pilot) |
| Old project `scripts/04_energy_metrics/` | Wind/solar potential formulas (PV cell temperature / performance ratio, wind power density) | Reusable as **formula reference** (see §4) |

**Conclusion**: Neither this project nor the old project has hourly-scale ERA5 data; the
hourly data needed for the wind/solar potential pilot must be newly downloaded (not
downloaded in this round).

## 2. ERA5 Variables Needed (hourly, small area 30–33°N, 112–121°E)

| Variable | Short name | Use | Notes |
|---|---|---|---|
| 2 m temperature | `t2m` | PV cell temperature + alignment/verification against CN05.1 daily Tmax | K, must be converted to ℃ |
| 100 m zonal/meridional wind | `u100`, `v100` | Hub-height wind speed `V100 = √(u²+v²)` | Use 100 m directly, no 10→100 m extrapolation needed |
| Surface downward solar radiation | `ssrd` | PV potential (GHI) | **Accumulated quantity**, see §3 |
| 10 m zonal/meridional wind | `u10`, `v10` | PV cell-temperature cooling term `wind10 = √(u10²+v10²)`; backup hub-height wind extrapolation | Needed by this project's PV cell-temperature model, **already included** |

- Time range: at least cover May–September 2022 (pilot); later extend to the
  baseline period 1981–2010 and the full study period.
- Space: prefer the ERA5 single-level 0.25° grid (consistent with CN05.1 0.25°,
  easiest to align); if finer resolution is needed, ERA5-Land 0.1° may be used
  (but the `ssrd` accumulation convention must be checked separately).

## 3. Radiation Accumulation and Unit Handling (critical; verified against the official CDS convention, 2026-10)

- The ERA5 single-level `surface_solar_radiation_downwards` (`ssrd`, parameter 169) is
  an **accumulated** quantity with units of **J m⁻²** (the total shortwave radiation
  reaching a horizontal plane within each step, including direct + diffuse, positive
  downward), **not** instantaneous W m⁻².
- **Current CDS convention (since 2021-12-08)**: CDS has stopped automatically
  converting accumulated quantities to rate values (W m⁻²) and now returns the
  accumulated J m⁻² directly per the catalogue definition. Therefore what you get is
  always J m⁻² and must be converted yourself.
- **The ERA5 reanalysis hourly product chosen for this plan (`reanalysis-era5-single-levels`)**:
  - The value at each time step is the **accumulation over the 1 hour before that
    time step** ("hour ending at the validity time"); for example, the value at
    12:00 UTC = the radiation energy for 11:00–12:00 UTC;
  - Therefore **directly dividing by `3600 s`** gives the hourly mean irradiance
    `GHI [W m⁻²] = ssrd / 3600`, **with no need to difference adjacent time steps**.
- **Two easily confused points ("÷3600" cannot be generalized unconditionally)**:
  - **ERA5-Land** hourly `ssrd` is **accumulated from 00 UTC** (increasing hour by
    hour); you must first difference adjacent steps `ssrd(t) − ssrd(t−1)` (no
    subtraction needed at 01 UTC) and then divide by 3600; the 00 UTC value is the
    full previous-day accumulation.
  - **Ensemble members / ensemble mean / ensemble spread** are **3-hour**
    accumulations and must be divided by 10800.
- Conclusion: This plan explicitly uses **ERA5 reanalysis hourly** (not ERA5-Land,
  not the ensemble), so dividing each time step independently by `3600` is correct;
  this conclusion holds only for "that product + hourly", and must be recorded
  explicitly in scripts/documents to avoid being copied blindly if ERA5-Land is used
  later.
- The old project's `rsds` (monthly-mean instantaneous W m⁻²) and this plan's `ssrd`
  (hourly accumulated J m⁻²) have **different conventions** and cannot be applied
  directly; this plan explicitly uses hourly `ssrd` and converts it to W m⁻².

### 3.1 Verified CDS Request Essentials (implemented in `scripts/era5_download_2022.py`)

| Item | Value |
|---|---|
| Dataset | `reanalysis-era5-single-levels` (ERA5 reanalysis single level, not ERA5-Land) |
| `product_type` | `reanalysis` |
| Grid | 0.25°×0.25° regular grid (consistent with CN05.1 0.25°, nearest-neighbour alignment is easiest) |
| Variables | `2m_temperature`, `100m_u/v_component_of_wind`, `10m_u/v_component_of_wind`, `surface_solar_radiation_downwards` (6 variables, see §2) |
| Time | Hourly `00:00`…`23:00` (`time` field) |
| Area | `area = [33, 112, 30, 121]` (order **North, West, South, East**, units °) |
| Download format | `data_format = "netcdf"` (also possible: `grib`; NetCDF chosen for easy xarray reading) |
| Credentials | The new CDS uses a Personal Access Token (`~/.cdsapirc`, `url: https://cds.climate.copernicus.eu/api`); the script **does not read or print** credentials |
| Licence | The Terms of Use must be accepted on the CDS dataset page before downloading |

### 3.2 Leading/Trailing Dates to Reserve for the Observation-Day Window (Beijing-time conversion corrected)

A CN05.1 daily Tmax value labelled with date D may correspond to either of two windows:
00–24 or 20–20 (Beijing time = UTC+8). With Beijing time = UTC+8, the UTC conversions
of the two candidate windows for date D are:

- 00–24 Beijing time = from the previous day `D−1 16:00 UTC` to the same day
  `D 16:00 UTC` (left-closed, right-open);
- 20–20 Beijing time = from the previous day `D−1 12:00 UTC` to the same day
  `D 12:00 UTC`.

Key conclusion: both windows **only reach back into the previous UTC day** and **end no
later than `D 16:00 UTC`**, so they **do not** reach into the UTC data of the following
day. For example, `2022-08-31` Beijing time 00–24 = UTC `08-30 16:00` to `08-31 16:00`,
which lies entirely within UTC 08-30/08-31, so **the UTC data of 09-01 is not needed**.
Therefore for each analysis day D only **1 leading day** must be reserved (starting from
`D−1 12:00 UTC`).

The request window for this plan's target month 2022-08 is **2022-07-31 ~ 2022-09-01
(33 days)**: `07-31` is a **necessary** leading buffer, while `09-01` is only a
**symmetric margin** kept to preserve the completeness of the original 33-day plan,
**not** a correctness requirement for the August daily aggregation.

### 3.3 Time Attribution of Instantaneous Variables vs Hourly Accumulated Radiation (must not reuse the same filtering)

- **Instantaneous variables** (`t2m`, `u10/v10`, `u100/v100`): the value is an
  **instantaneous value at the time step**, attributed directly by timestamp `t`
  (e.g. `t=12:00 UTC` is the instantaneous temperature/wind at 12:00).
- **Hourly accumulated radiation** (`ssrd`): the value is the accumulation over the
  **1 hour before** that time step ("hour ending at the validity time"), and must be
  attributed to a window by the **end time of the accumulation interval**; for example,
  the ssrd at `t=12:00 UTC` is the radiation for 11:00–12:00 UTC and **cannot** be
  treated as the instantaneous irradiance at 12:00, nor can the instantaneous-variable
  time filtering be copied. When converting to `GHI`, divide by `3600`, but **its
  attribution time step is still its end time**.
- **Not yet finalized**: the exact date attribution of the CN05.1 observation day
  (00–24 or 20–20 Beijing time). Until the CN05.1 observation-day definition is
  confirmed, **no formal daily aggregation is performed**; only the UTC hourly data are
  **staged** per the above conventions (instantaneous variables by timestamp, ssrd by
  end time), keeping the original UTC data without composing daily values.

### 3.4 ERA5 Daily Aggregation After Confirmation: Instantaneous Variables by Timestamp, ssrd by End Time (24 hours for each candidate window)

Let the daily-value window for CN05.1 date D be `[T_start_UTC, T_end_UTC)` (left-closed,
right-open, Beijing time = UTC+8). For **the same window**, the selected time steps of
the two kinds of quantities **differ by 1 hour**, and the same filtering must not be
used for both:

- **Instantaneous variables** (`t2m`, `u10/v10`, `u100/v100`; daily max or daily mean):
  take the 24 samples whose **time step falls within `[T_start, T_end)`**, i.e.
  `T_start, T_start+1h, …, T_end−1h`. The instantaneous sample at the boundary `T_end`
  **belongs to the next day** and must not be counted twice.
- **End-of-hour accumulated radiation** (`ssrd`; daily sum): take the 24 samples whose
  **accumulation interval `[t−1h, t]` lies entirely within `[T_start, T_end)`**, i.e.
  `t = T_start+1h, …, T_end`. The ssrd at `t=T_end` is the radiation for the last hour
  of the window `[T_end−1h, T_end)`; the ssrd at `t=T_start` belongs to the **previous
  hour** and **must not** be counted (otherwise out-of-range/duplicated).

| Candidate window | Beijing-time window | UTC window `[T_start, T_end)` | Instantaneous-variable time steps | ssrd time steps |
|---|---|---|---|---|
| A: calendar day 00–24 | `D 00:00–D+1 00:00` | `[D−1 16:00, D 16:00)` | D−1 16:00 … D 15:00 | D−1 17:00 … D 16:00 |
| B: 20:00–20:00 (station convention) | `D−1 20:00–D 20:00` | `[D−1 12:00, D 12:00)` | D−1 12:00 … D 11:00 | D−1 13:00 … D 12:00 |

Both candidates are exactly **24 hours / 24 samples, with no duplication or omission**.
Taking candidate B as an example: the instantaneous variables take `D−1 12:00…23:00`
(12 samples) + `D 00:00…11:00` (12 samples); ssrd takes `D−1 13:00…23:00` (11 samples)
+ `D 00:00…12:00` (13 samples).

- Daily Tmax alignment check: compare the **maximum** of the 24 instantaneous `t2m`
  samples in the candidate window with CN05.1 Tmax (the ERA5 hourly maximum is an
  approximation of the continuous maximum, not strictly equal).
- Target month 2022-08 coverage: candidate B only needs from `07-31 12:00 UTC` to
  `08-31 12:00 UTC`; candidate A ends at `08-31 16:00 UTC`; the existing 792 h
  (07-31 00:00–09-01 23:00) is **sufficient for both**, and 09-01 is still not
  necessary (only a symmetric margin).
- **No formal daily aggregation before the CN05.1 day boundary is confirmed**; only
  after confirmation, values are taken per the corresponding windows above.

### 3.5 Attribution and Calculation Order of Derived Quantities (supplementary constraints)

- **`ghi`/`pvpot` inherit ssrd's interval semantics and use the end-of-hour labelling
  rule for daily attribution**: `ghi = ssrd/3600` and `pvpot = PR·ghi` are both derived
  from the "end-of-hour accumulated" ssrd and represent the **interval-average
  quantity** of `[t−1h, t]`, **not** the instantaneous value at the time step. For daily
  aggregation they use the **same end-of-hour labelling rule** as ssrd (daily sum takes
  `t = T_start+1h…T_end`).
- **Uniform resampling of the whole hourly NetCDF is forbidden**: instantaneous
  variables (t2m, u/v wind, wpd) are attributed by time step, while ssrd/ghi/pvpot are
  attributed by end-of-hour interval, the two differing by 1 hour. Therefore
  `groupby('valid_time.date')` or `.resample('1D')` must **not** be applied to all
  variables at once — that would either cut the instantaneous variables at the wrong day
  boundary by UTC calendar day, or shift ssrd/ghi/pvpot by 1 hour by treating them as
  instantaneous values. Each variable must be **filtered separately by variable and by
  label** and then aggregated.
- **Sampling approximation of the WPD daily mean (stated separately)**: `wpd =
  0.5·ρ·V100³` is composed from the **instantaneous** 100 m wind speed (time-step
  values), so `wpd` is an instantaneous quantity; its "daily mean" = the arithmetic mean
  of the 24 instantaneous `wpd` samples in the window (time steps `T_start…T_end−1h`),
  which is a **1 h discrete-sampling approximation** of the continuous daily mean, not
  a strict integral.
- **Hourly non-linear quantities first, then daily aggregation (hard ordering
  constraint)**: `V100³` (the cube), `Tcell` (containing `GHI/(U0+U1·V)` and the linear
  wind-speed term), `PR`, and `pvpot` are all non-linear quantities and must be
  **computed hourly first**, then daily-averaged/accumulated over the hourly values;
  they must **not** be recomputed from daily-mean wind speed / temperature / irradiance
  (`mean(V³) ≠ mean(V)³`, `mean(PVpot) ≠ PR(mean·)·mean(GHI)`).
- **Cross-day-boundary composition test**: `scripts/era5_daily_aggregation_test.py`
  verifies, for the two candidate windows, that the radiation and PV accumulation has
  **no duplication or omission** and that **the sum of daily accumulations = the
  corresponding hourly accumulation**; read-only, no daily products are generated.

## 4. Wind/Solar Potential: Compute Hourly First, Then Aggregate to the Daily Scale

> The more rigorous formulas, parameters, sources, and the "list of assumptions to
> confirm before entering calculation" are in `docs/potential_calc_scheme.md`; this
> section keeps only the essentials.

The old project's formulas are retained as a starting point (reusable as reference),
but computed **at the hourly scale**:

- **Wind speed** (hourly): `V100 = √(u100² + v100²)` (compute the hourly composition
  first; do **not** first take daily means of u and v separately and then compose — that
  is systematically too low). If extrapolating from 10 m wind, use the old project's
  empirical roughness exponent `α = 0.37 − 0.0881·ln(v10)`,
  `V100 = v10·(100/10)^α`.
- **Wind power density** (hourly): `WPD = 0.5 · ρ · V100³`, `ρ = 1.225 kg m⁻³`
  (consistent with the old project). `V³` may be computed hourly first and then
  daily-averaged (the cube of wind must not be replaced by the cube of the daily-mean
  wind speed).
- **PV potential** (hourly):
  - Cell temperature `T_cell = 4.3 + 0.943·T_a + 0.028·GHI − 1.528·v10` (℃, old-project
    coefficients, requires t2m and 10 m wind).
  - Performance ratio `PR = 1 + γ·(T_cell − 25)`, `γ = −0.005 ℃⁻¹`, STC temperature
    25 ℃.
  - `PVpot = PR · GHI` (W m⁻², i.e. the hourly PV potential).
  - Compute `PVpot` hourly first, then take the daily mean (or the mean over sunlit
    hours).
- **Daily-scale aggregation**: for each grid cell-day, aggregate the hourly `WPD` and
  `PVpot` into daily values (daily mean or mean over sunlit hours; the convention must
  be fixed together with the consistency assumptions below), obtaining a per-day
  wind/solar potential series that can be aligned with "whether that day is a heatwave
  day".

## 5. Alignment with the CN05.1 Observation Day

- **Time alignment**: ERA5 is in UTC hours; CN05.1 daily data are calendar/observation
  days. For the day-boundary verification conclusion see `docs/heatwave_method_v2.md`
  §7 (station data use a Beijing-time 20:00 day boundary with values attributed to the
  ending day, but CN05.1 itself does not state this explicitly and awaits confirmation
  with the data provider); after confirmation, take the hours per the windows in §3.4
  of this file.
  - For example, daily Tmax alignment: take the maximum of 24 hours of `t2m` per the
    confirmed window and compare with CN05.1 Tmax (the ERA5 hourly maximum is an
    approximation of the continuous maximum).
- **Spatial alignment**: in the pilot area, directly use the ERA5 0.25° grid cells that
  are **nearest neighbours** of the CN05.1 0.25° grid cells (or do conservative
  resampling); **no interpolation or gap-filling is applied to CN05.1**.

## 6. Wind/Solar Combination: Unify Capacity-Factor / Installed-Capacity Assumptions, Do Not Add Directly

> The unified convention for the joint output (`CF_mix = a·CF_wind + (1−a)·CF_pv`,
> with the installed-capacity ratio `a` declared explicitly, never `WPD + PVpot`) is in
> `docs/potential_calc_scheme.md` §0/§4.

- `WPD` (W m⁻²) and `PVpot` (W m⁻²) have different physical meanings; although both
  have the dimensions W m⁻², they cannot be added directly, nor do they represent
  grid-deliverable electricity.
- Combination methods (to be selected; one must be fixed before entering calculation):
  1. **Unified capacity factor**: `CF_wind = P(V100)/P_rated` (NREL 5-MW power curve,
     see `docs/potential_calc_scheme.md` §2), `CF_pv = PVpot / G_ref`
     (`G_ref = 1000 W m⁻²`), then weight and combine per the assumed wind/solar
     installed-capacity ratio `CF_mix = a·CF_wind + (1−a)·CF_pv`;
  2. or convert to grid-deliverable electricity per a chosen **installed-capacity
     assumption** (e.g. wind and solar installed capacity 50% each, or specified MW
     values) and then sum.
- Whichever is chosen, the installed-capacity ratio / rated-parameter assumptions must
  be **declared explicitly**; the raw `WPD` and `PVpot` are **not** added directly.

## 7. Boundaries of This Round

- **Do not download** ERA5 and **do not run** energy calculations.
- No gap-filling of missing values and no additional interpolation.
- Once data are in place, new results are written to a new independent directory (a new
  sibling directory of `results_merge/`), without overwriting `results/`,
  `results_fixed/`, `results_merge/`, or the original data.
