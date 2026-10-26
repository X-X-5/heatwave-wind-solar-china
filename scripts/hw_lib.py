# -*- coding: utf-8 -*-
"""
Core heatwave algorithm library (research definition, NOT ETCCDI WSDI).

Definition implemented here (documented in docs/heatwave_method.md):

  * Fixed baseline period 1981-2010.
  * Per grid cell and per calendar day (366-day leap-year frame keyed by
    (month, day)), the Tmax 90th-percentile threshold is estimated from the
    pooled sample of the 5-day window centred on that calendar day
    (day-2 .. day+2), collected across all baseline years (cross-year wrap).
  * A heatwave = at least MIN_RUN (default 3) *consecutive* days with
    Tmax *strictly greater than* the calendar-day threshold.
  * Analysis season = May-September (inclusive). Events are identified only
    within this window; the window boundary truncates runs (no connection to
    April/October).
  * Missing values are excluded from the threshold sample and from intensity
    means. A missing day is never a heatwave day and always breaks a run
    (events are never connected across a gap).

Units: Tmax in degrees Celsius (CN05.1); thresholds and exceedance intensity
in degrees Celsius.
"""
from collections import defaultdict

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    HAS_NUMBA = False

# --------------------------------------------------------------------------
# Constants (single source of truth)
# --------------------------------------------------------------------------
BASELINE = (1981, 2010)
SEASON = (5, 9)          # May .. September inclusive
MIN_RUN = 3              # minimum consecutive days for a heatwave
Q = 90.0                 # percentile level (percent)
WINDOW_HALF = 2          # +/- 2 days -> 5-day centred window
MIN_SAMPLES = 100        # minimum valid values in pooled window to define threshold

# Cumulative day-offset at start of each month in a 366-day (leap) year, 0-based.
CUM_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                    dtype=np.int64)


def slot_index(month, day):
    """Map (month, day) -> slot 0..365 in a leap-year reference frame.

    Feb 29 -> slot 59. Non-leap years simply never produce slot 59, so the
    (month, day) -> slot mapping is identical every year (leap-agnostic).
    """
    month = np.asarray(month, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    return CUM_LEAP[month - 1] + day - 1


def season_mask(month, day, season=SEASON):
    """Boolean mask: day lies within the analysis season (month-day based)."""
    m = np.asarray(month)
    return (m >= season[0]) & (m <= season[1])


# --------------------------------------------------------------------------
# Threshold estimation
# --------------------------------------------------------------------------
def compute_thresholds(X, slot, q=Q, min_samples=MIN_SAMPLES):
    """Per-cell, per-calendar-day 90th-percentile thresholds.

    Parameters
    ----------
    X    : (n_days, n_cells) float array; NaN = missing.
    slot : (n_days,) int array of calendar-day slot indices 0..365.
    q    : percentile level (percent).
    min_samples : minimum valid values in the 5-day pooled window required to
                  define a threshold; otherwise the threshold is NaN.

    Returns
    -------
    thr : (366, n_cells) float64 array; NaN where undefined.
    """
    n_days, n_cells = X.shape
    groups = defaultdict(list)
    for d in range(n_days):
        groups[int(slot[d])].append(d)

    thr = np.full((366, n_cells), np.nan, dtype=np.float64)
    for s in range(366):
        idxs = []
        for k in range(s - WINDOW_HALF, s + WINDOW_HALF + 1):
            idxs.extend(groups.get(k % 366, []))
        if not idxs:
            continue
        samp = X[np.asarray(idxs, dtype=np.int64)]          # (n_samp, n_cells)
        with np.errstate(all="ignore"):
            pct = np.nanpercentile(samp, q, axis=0)          # (n_cells,)
        cnt = np.sum(~np.isnan(samp), axis=0)
        ok = cnt >= min_samples
        thr[s, ok] = pct[ok]
    return thr


# --------------------------------------------------------------------------
# Event detection (run-length logic)
# --------------------------------------------------------------------------
def _events_2d_py(hot, excess, min_run):
    """Identify runs of >= min_run consecutive hot days.

    hot    : (n_year, n_day) bool  -- True = heatwave day.
    excess : (n_year, n_day) float -- exceedance (Tmax - threshold); only
              summed where hot is True (so NaN on non-hot days is harmless).

    Returns (n_events, total_days, max_dur, sum_excess, n_hw_days), each length
    n_year.
    """
    ny, nd = hot.shape
    ne = np.zeros(ny, dtype=np.int32)
    td = np.zeros(ny, dtype=np.int32)
    md = np.zeros(ny, dtype=np.int32)
    se = np.zeros(ny, dtype=np.float64)
    hd = np.zeros(ny, dtype=np.int32)
    for y in range(ny):
        run = 0
        s = 0.0
        for d in range(nd):
            if hot[y, d]:
                run += 1
                s += excess[y, d]
            else:
                if run >= min_run:
                    ne[y] += 1
                    td[y] += run
                    hd[y] += run
                    se[y] += s
                    if run > md[y]:
                        md[y] = run
                run = 0
                s = 0.0
        # flush trailing run at end of the season
        if run >= min_run:
            ne[y] += 1
            td[y] += run
            hd[y] += run
            se[y] += s
            if run > md[y]:
                md[y] = run
    return ne, td, md, se, hd


if HAS_NUMBA:
    _events_2d = njit(_events_2d_py)
else:
    _events_2d = _events_2d_py


def detect_cell_metrics(tmax_1d, thr_366, season_idx, slot_season,
                        n_years, days_per_year, min_run=MIN_RUN):
    """Full per-cell detection: returns per-year metrics for one grid cell.

    Parameters
    ----------
    tmax_1d     : (n_total,) float, full daily Tmax series (NaN = missing).
    thr_366     : (366,) float, calendar-day thresholds.
    season_idx  : (n_season,) int, indices into tmax_1d selecting the
                  season days, ordered by year (n_season = n_years*days_per_year).
    slot_season : (n_season,) int, calendar-day slot of each season day.
    n_years, days_per_year : shape of the (year, season-day) array.

    Returns
    -------
    n_events, total_days, max_dur : (n_years,) int32
    intensity : (n_years,) float64, mean (Tmax - threshold) over heatwave days
                in each year (NaN where no heatwave day occurred).
    """
    tmax_s = np.asarray(tmax_1d)[season_idx]                 # (n_season,)
    thr_s = np.asarray(thr_366)[slot_season]                 # (n_season,)
    finite = np.isfinite(tmax_s) & np.isfinite(thr_s)
    hot = finite & (tmax_s > thr_s)                          # strict ">"
    excess = tmax_s - thr_s

    H = hot.reshape(n_years, days_per_year)
    E = excess.reshape(n_years, days_per_year)
    ne, td, md, se, hd = _events_2d(H, E, min_run)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = se / hd
    intensity = np.where(hd > 0, intensity, np.nan)
    return ne, td, md, intensity


# --------------------------------------------------------------------------
# Gated detection + completeness diagnostics (phase-1 fix)
# --------------------------------------------------------------------------
UNDEF_I = -9999          # fill value marking an "undecidable" cell-year


def _max_run_2d_py(mask):
    """mask: (n_year, n_day) bool (True = missing). -> longest run per year."""
    ny, nd = mask.shape
    out = np.zeros(ny, dtype=np.int32)
    for y in range(ny):
        run = 0
        best = 0
        for d in range(nd):
            if mask[y, d]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        out[y] = best
    return out


if HAS_NUMBA:
    _max_run_2d = njit(_max_run_2d_py)
else:
    _max_run_2d = _max_run_2d_py


def detect_cell_full(tmax_1d, thr_366, season_idx, slot_season,
                     n_years, days_per_year, min_run=MIN_RUN):
    """Gated per-cell detection: metrics + completeness diagnostics.

    Like detect_cell_metrics, but additionally:
      * returns `decidable` (per-year count of days where both Tmax and the
        threshold are finite) and `max_missing` (per-year longest run of
        consecutive missing-Tmax days within the season);
      * applies the validity gate: a cell-year with **zero** decidable days has
        its count/days/max_dur set to UNDEF_I and intensity to NaN (never 0),
        so an undecidable year is never silently reported as "no heatwave".

    Returns (n_events, total_days, max_dur, intensity, decidable, max_missing).
    """
    tmax_s = np.asarray(tmax_1d)[season_idx]
    thr_s = np.asarray(thr_366)[slot_season]
    finite = np.isfinite(tmax_s) & np.isfinite(thr_s)
    hot = finite & (tmax_s > thr_s)
    excess = tmax_s - thr_s

    H = hot.reshape(n_years, days_per_year)
    E = excess.reshape(n_years, days_per_year)
    ne, td, md, se, hd = _events_2d(H, E, min_run)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = se / hd
    intensity = np.where(hd > 0, intensity, np.nan)

    dec = finite.reshape(n_years, days_per_year).sum(axis=1).astype(np.int32)
    tmiss = (~np.isfinite(tmax_s)).reshape(n_years, days_per_year)
    maxmiss = _max_run_2d(tmiss)

    undef = dec == 0
    ne = np.where(undef, UNDEF_I, ne).astype(np.int32)
    td = np.where(undef, UNDEF_I, td).astype(np.int32)
    md = np.where(undef, UNDEF_I, md).astype(np.int32)
    intensity = np.where(undef, np.nan, intensity)
    return ne, td, md, intensity, dec, maxmiss
