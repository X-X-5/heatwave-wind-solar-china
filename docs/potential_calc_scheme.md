# Wind/Solar Potential Calculation Method (Scheme, Not Yet Run)

This document specifies the candidate formulas, parameters, and sources for the
**wind/solar potential**, for verification before entering calculation. **This round
does not run any energy calculation**; this document only distinguishes clearly between
the "resource-quantity metric" and "capacity factor" conventions, and gives the unified
assumptions for the combined output, to avoid adding quantities of different
dimensions/meanings directly.

---

## 0. Four Concepts to Distinguish First (the most critical distinction)

| Name | Definition | Dimensions/units | Directly addable? |
|---|---|---|---|
| **WPD** (wind power density) | Wind-energy flux per unit rotor swept area `0.5·ρ·V³` | W m⁻² (resource quantity) | No |
| **Wind capacity factor CF_wind** | `P(V100)/P_rated`, integrated hourly using a **sourced wind-turbine power curve** | dimensionless 0–1 | **Weightable** once brought to the same convention as the PV CF |
| **PV potential PVpot** | Output power per unit module area after accounting for the temperature loss `PR·GHI` | W m⁻² (still contains the GHI factor) | No (depends on irradiance, not grid-deliverable electricity) |
| **PV capacity factor CF_pv** | `PVpot / G_ref` (`G_ref = 1000 W m⁻²` STC irradiance) | dimensionless 0–1 | **Weightable** once brought to the same convention as the wind CF |

> Conclusion: **WPD is a "wind resource", and PVpot is a "solar resource including the
> performance ratio"**; neither is electricity, and their dimensions/bases differ, so
> they are **never added directly**. Only when both are normalized into **dimensionless
> capacity factors** and then multiplied by an **explicitly declared installed-capacity
> ratio** can a joint wind/solar output be made.

---

## 1. Wind Speed and Wind Power Density (hourly)

- Hourly hub-height wind speed (prefer directly using 100 m):
  `V100 = √(u100² + v100²)`
- Alternative (10 m extrapolated to 100 m, empirical power law, old-project
  coefficients):
  `α = 0.37 − 0.0881 · ln(v10)` (`v10 = √(u10²+v10²)`, m s⁻¹)
  `V100 = v10 · (100/10)^α`
  - Power-law source: empirical wind-profile exponent (Hellmann/power law, of the
    Justus & Mikhail 1976 type); the formula above is the old project's linearization
    of the empirical roughness exponent, **the coefficient source is to be verified**,
    for reference only.
- Hourly wind power density:
  `WPD = 0.5 · ρ · V100³`, `ρ = 1.225 kg m⁻³` (standard air density at 15 ℃ sea
  level, IEC 61400-12-1).

**Calculation-order constraint**: compute `V100` **hourly** first (compose u/v first,
then take the magnitude), then compute `V100³` hourly, and only then take daily/monthly
means. **Do not** first take daily means of u and v separately and then compose
(systematically too low), and **do not** replace the mean of the hourly cubes with the
cube of the daily-mean wind speed.

---

## 2. Wind Capacity Factor (use the turbine power curve; drop the WPD/WPD_rated ratio)

`CF_wind = Σ_h P(V_h) / (P_rated · N_h)`

Hour by hour, map the hub-height wind speed to power using the **turbine power curve**
`P(V)`, then average the power and divide by the rated power to get the capacity
factor. **No longer use** the `WPD / WPD_rated` "cubic–capped" ratio approximation.

- **Default turbine: NREL 5-MW reference turbine** (Jonkman et al. 2009,
  NREL/TP-500-38060) — three-blade upwind, variable-pitch variable-speed (variable
  speed + pitching to feather) control.
  - Parameters: **rated 5 MW, rated wind speed 11.4 m/s, cut-in 3 m/s, cut-out 25 m/s,
    rotor diameter 126 m, hub 90 m**; power curve in report §9 (FAST-generated,
    publicly citable).
  - **Hub-height difference**: ERA5 provides 100 m wind while this turbine's hub is
    90 m, so it must first be pushed down by the power law `V90 = V100 · (90/100)^α`
    (α in §1) before entering the curve, and this assumption must be **recorded
    explicitly**.
  - **Cut-in/rated/cut-out rules**: `V < 3` or `V ≥ 25` → `P = 0`; `3 ≤ V < 11.4` →
    interpolate from the curve table (variable-speed optimal-control region, not a pure
    cubic); `11.4 ≤ V < 25` → `P = 5 MW`.
- **Replacement turbine**: if a domestic turbine is used instead (Goldwind/Envision,
  etc.), its power curve must have a **citable public source** and likewise declare the
  cut-in/rated/cut-out wind speeds and hub height; otherwise the NREL 5-MW remains in
  use.
- **WPD is kept as an independent resource metric** (§1); it does not enter the
  capacity factor and is **not added directly to PVpot**.

---

## 3. PV Potential and PV Capacity Factor (hourly)

- Hourly irradiance: `GHI = ssrd / 3600` (W m⁻², see `docs/era5_data_plan.md` §3, ERA5
  reanalysis hourly is an "end-of-hour 1-hour accumulation", divide directly by 3600).
- Module operating temperature (old project's **empirical formula**, multiple linear
  regression, **not** the standard Faiman model):
  `T_cell = 4.3 + 0.943 · T_a + 0.028 · GHI − 1.528 · wind10` (℃)
  - `T_a = t2m − 273.15` (℃), `wind10 = √(u10²+v10²)` (m s⁻¹).
  - **Nature**: this formula contains an intercept 4.3 and a `T_a` term with slope ≠ 1
    (a linear regression on some dataset); the `0.028` term is of the order of the Ross
    coefficient and the `wind10` term is a Faiman-type wind-cooling term; it must **not
    be written as "standard Faiman"**, and its coefficient source (values inside the
    old project's `04_energy_metrics`) must be verified against a specific reference
    before formal use.
  - **Source-verification conclusion (2026-10, corrected)**: the coefficients
    4.3 / 0.943 / 0.028 / −1.528 appear in the old project's
    `scripts/04_energy_metrics/compute_energy_metrics_pilot_v2.py` L125–128 and
    `audit_energy_metrics_production.py` L80–83 as **bare constants with no stated
    origin**, but **no citation in the repository does not mean no source in the
    literature**: this coefficient set appears in **Jerez et al. (2015, Nat. Commun.
    6:10014)**, which cites the linear module-temperature model of **Chenni et al.
    (2007)** (with other literature tracing it to TamizhMani et al. 2003 / Tang 2003).
    The formula's inputs are 2 m temperature, horizontal-plane GHI (`RSDS`), and
    ≈10 m wind speed, matching ERA5 `t2m`/`ssrd→GHI`/`V10` one-to-one; but **the
    regional applicability of the empirical coefficients remains an assumption**. The
    primary scheme **tentatively uses Faiman (trial)** and keeps the linear formula as
    a sensitivity comparison (see `results_hourly/REPORT.md` §5 sensitivity comparison
    for the difference between the two models), **no further switching of formulas, no
    model extension**; Faiman (2008) needs POA and wind at a calibration height,
    requiring additional transposition/wind-profile assumptions.
  - **Applicability conditions (must be explicit)**: `GHI` is **horizontal-plane**
    irradiance (not tilted-plane POA); wind speed is taken at **10 m** height `wind10`;
    applies to the daytime operating temperature of **free-mounted / open-rack c-Si
    modules**.
  - **Citable standard alternatives** (choose one and declare it when entering formal
    calculation):
    - **Faiman (2008, doi:10.1002/pip.813)**: `T_cell = T_a + GHI / (U0 + U1·v)`,
      `U0 = 25.0 W m⁻² K⁻¹`, `U1 = 6.84 W m⁻² K⁻¹ (m/s)⁻¹` (open-rack default,
      adopted by pvlib / IEC 61853);
    - **Ross**: `T_cell = T_a + (NOCT − 20)/800 · GHI`, with coefficient
      `k = (NOCT−20)/800` ≈ 0.02–0.056 depending on mounting (free-mounted ≈ 0.021).
- Performance ratio (temperature loss, STC reference):
  `PR = 1 + γ · (T_cell − 25)`, `γ = −0.005 ℃⁻¹` (typical c-Si power temperature
  coefficient ≈ −0.5 %/℃; commonly used in PVsyst, Evans 1981 type), STC temperature
  25 ℃.
- PV potential:
  `PVpot = PR · GHI` (W m⁻², the "usable irradiance power density" including the
  performance-ratio loss).
- PV capacity factor (**simplified**):
  `CF_pv = PVpot / G_ref`, `G_ref = 1000 W m⁻²` (STC irradiance).
  Equivalently: `CF_pv = mean(PVpot) / 1000` (for daily-mean/monthly-mean PVpot).
- **Night/dark handling**: first clip ssrd with `max(0, ssrd)` (physically no negative
  values); when `GHI ≤ 0`, `PVpot = 0` and `PR` has no physical meaning, and the PV
  daily/monthly metrics are computed **only over the sunlit time steps with `GHI > 0`**.
- **Nature of the metric (must be stated clearly)**: `PVpot = PR·GHI` is a **simplified
  potential metric** (horizontal plane, no tilt/orientation, no
  inverter/wiring/soiling/degradation or other system losses), **not** a system
  capacity factor; `CF_pv` is only a "performance-ratio-weighted" simplified capacity
  factor. To call it a system CF, POA irradiance, inverter efficiency, DC/AC ratio, and
  the loss chain must be introduced.

**Calculation-order constraint**: compute `GHI`, `T_cell`, `PR`, `PVpot` hourly first,
then take daily/monthly means; do not back out the daily `T_cell` from daily-mean `GHI`
(non-linear).

---

## 4. Joint Wind/Solar Output: Unified Capacity Factor + Installed-Capacity Ratio, Never Added Directly

Given the wind/solar installed-capacity ratio `a` (the wind share of installed
capacity, `a ∈ [0,1]`, **must be declared explicitly**, e.g. 0.5/0.5 or a scenario set
0.3/0.5/0.7):

`CF_mix = a · CF_wind + (1 − a) · CF_pv` (dimensionless 0–1)

- This is a "weighted capacity factor" that can be aligned day by day with heatwave
  days as a proxy for the "joint wind/solar output level".
- If **electricity** is needed: `E_mix = CF_mix · P_total_installed · T`, with
  `P_total_installed` the combined wind/solar installed capacity (MW, assumed value).
- **Forbidden**: `WPD + PVpot`, `CF_wind + CF_pv` (unweighted), or any direct addition
  without a declared basis.

---

## 5. Daily Alignment Convention

- All hourly quantities are computed first and then aggregated to daily values (daily
  mean or mean over sunlit hours; the convention must be unified and recorded in the
  calculation script).
- Daily values are aligned to CN05.1 days per the observation-day windows in
  `docs/heatwave_method_v2.md` §7 / `docs/era5_data_plan.md` §5, for the "heatwave day
  vs non-heatwave day" wind/solar potential comparison.

---

## 6. Parameter Summary Table

| Symbol | Value | Source/basis |
|---|---|---|
| `ρ` | 1.225 kg m⁻³ | IEC 61400-12-1 standard air density |
| `α` | `0.37 − 0.0881·ln(v10)` | Old-project empirical formula (power law, source to be verified) |
| Hub height | 100 m (direct `u100/v100`) | ERA5 100 m wind field |
| `G_ref` | 1000 W m⁻² | STC (IEC 60904) |
| `γ` | −0.005 ℃⁻¹ | Typical power temperature coefficient of crystalline-silicon modules (PVsyst/Evans 1981) |
| `T_cell` coefficients | 4.3 / 0.943 / 0.028 / −1.528 | Old-project empirical formula (linear regression, **not Faiman**; source to be verified) |
| Faiman `U0, U1` | 25.0 W m⁻²K⁻¹, 6.84 W m⁻²K⁻¹(m/s)⁻¹ | Faiman 2008 (open-rack default, pvlib/IEC 61853) |
| Turbine power curve | NREL 5-MW (cut-in 3 / rated 11.4 / cut-out 25 m/s, rated 5 MW, hub 90 m) | Jonkman et al. 2009, NREL/TP-500-38060 |
| `a` | To be decided (suggested scenarios 0.3/0.5/0.7) | Wind/solar installed-capacity ratio assumption |

---

## 7. List of Assumptions That Must Be Confirmed Before Entering Calculation

1. The turbine power curve defaults to **NREL 5-MW** (cut-in 3 / rated 11.4 / cut-out
   25 m/s); a replacement turbine must have a public source and likewise declare its
   parameters and hub height (ERA5 100 m → this turbine's 90 m requires power-law
   push-down).
2. The PV module temperature **primary scheme tentatively uses Faiman (trial)**, and
   the linear formula (Jerez et al. 2015 / Chenni et al. 2007, i.e. the old-project
   empirical formula 4.3/0.943/0.028/−1.528) is kept as a sensitivity comparison, **no
   further switching of formulas, no model extension**; see `results_hourly/REPORT.md`
   §5 for the sensitivity comparison. Label it "the linear formula is an empirical
   regression, not standard Faiman".
3. Whether the wind/solar installed-capacity ratio `a` is a **single value** or a
   **scenario matrix**.
4. Daily aggregation convention: **daily mean** or **mean over sunlit hours** (wind and
   solar may differ).
5. If any of the above is unconfirmed, output only the "resource quantities" (WPD,
   PVpot, CF separately) and **do not** produce the joint `CF_mix`.

> This round (Block 2) **does not run** any of the above calculations; this document
> only fixes the conventions and the pending items, to be implemented after the ERA5
> data are downloaded and pass manual verification.
