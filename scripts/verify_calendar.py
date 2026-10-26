# -*- coding: utf-8 -*-
"""
Verification (read-only): calendar-day 5-day window and season-boundary
semantics. Produces evidence only; does not modify any result.

Point 2 of the phase-1 audit:
  (A) real leap / non-leap calendar behaviour of the 5-day window around
      Feb 28 / Feb 29 / Mar 1 and cross-year (Jan 1 / Dec 31);
  (B) whether the season boundary is "truncate-then-detect" vs
      "full-year-detect-then-count", with artificial Apr-May and Sep-Oct
      examples.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_lib as hw

print("=" * 72)
print("CHECK 2A -- (month, day) -> slot mapping in leap vs non-leap years")
print("=" * 72)

def slot_of(date):
    return int(hw.slot_index(np.array([date.month]), np.array([date.day]))[0])

# reference days
for (y, m, d) in [(2000, 2, 28), (2000, 2, 29), (2000, 3, 1),
                  (2001, 2, 28), (2001, 3, 1),
                  (2000, 1, 1), (2000, 12, 31), (2001, 12, 31)]:
    print(f"  {y:4d}-{m:02d}-{d:02d} -> slot {slot_of(pd.Timestamp(y, m, d))}")

print()
print("  slot index reference: Jan 1=0 ... Jan 31=30, Feb 1=31 ... Feb 28=58,")
print("  Feb 29=59 (leap only), Mar 1=60 ... Dec 31=365")
print()

# window membership for the key slots
print("5-day centred window (mod 366) for the key calendar days:")
for s, label in [(58, "Feb 28"), (59, "Feb 29"), (60, "Mar 1"),
                 (0, "Jan 1"), (365, "Dec 31")]:
    win = sorted({(s + k) % 366 for k in range(-2, 3)})
    print(f"  {label} (slot {s}): window slots = {win}")

# how many baseline days (1981-2010) actually fall in each window slot
print()
print("  Baseline 1981-2010 has 30 years, of which leap years =",
      [y for y in range(1981, 2011) if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0])

# build baseline date list and count days per slot
base_dates = pd.date_range("1981-01-01", "2010-12-31", freq="D")
bm = base_dates.month.values.astype(np.int64)
bd = base_dates.day.values.astype(np.int64)
bslot = hw.slot_index(bm, bd)
cnt = np.bincount(bslot, minlength=366)
print()
print("  days per slot in baseline (1981-2010):")
print("    slot 59 (Feb 29)   :", cnt[59], "occurrences")
print("    slot 58 (Feb 28)   :", cnt[58], "occurrences")
print("    slot 60 (Mar 1)    :", cnt[60], "occurrences")
print("    slot 0  (Jan 1)    :", cnt[0], "occurrences")
print("    any other slot     :", np.unique(cnt)[np.unique(cnt) != 0][0], "occurrences (all 30 except Feb 29)")

for s, label in [(58, "Feb 28"), (59, "Feb 29"), (60, "Mar 1"),
                 (0, "Jan 1"), (365, "Dec 31")]:
    win = sorted({(s + k) % 366 for k in range(-2, 3)})
    total = int(cnt[win].sum())
    print(f"    {label} window pool size = {total} baseline days "
          f"(= {win} -> {[int(cnt[w]) for w in win]})")

print()
print("=" * 72)
print("CHECK 2B -- season boundary: truncate-then-detect (current) vs full-year")
print("=" * 72)
print("Current implementation extracts ONLY May-Sep days (153 per year) and")
print("detects runs inside that window -> the boundary TRUNCATES a run. It does")
print("NOT detect runs across the full year and then attribute them to the season.")
print()

# helper to run the current detector on a synthetic full-year series
def current_run(tmax_arr, year_val):
    """Simulate the current pipeline for one year: only May-Sep days are fed."""
    dates = pd.date_range(f"{year_val}-01-01", f"{year_val}-12-31", freq="D")
    month = dates.month.values.astype(np.int64)
    day = dates.day.values.astype(np.int64)
    season = hw.season_mask(month, day)
    season_idx = np.where(season)[0]
    slot = hw.slot_index(month, day)[season_idx]
    thr = np.full(366, 30.0)   # fixed threshold 30 C
    tmax = np.asarray(tmax_arr, dtype=float)
    ne, td, md, it = hw.detect_cell_metrics(tmax, thr, season_idx, slot, 1, 153)
    return int(ne[0]), int(td[0]), int(md[0])


def full_year_run(tmax_arr):
    """Reference: detect runs over the FULL year (Apr/Oct included)."""
    hot = np.asarray(tmax_arr, dtype=float) > 30.0
    ne, td, md, se, hd = hw._events_2d(hot.reshape(1, -1), (np.asarray(tmax_arr, float) - 30.0).reshape(1, -1), 3)
    return int(ne[0]), int(td[0]), int(md[0])


def build(t0, hot_dates):
    """Return a full 365/366-day Tmax series, 29 C everywhere, 31 C on hot_dates."""
    n = 365 if t0.year % 4 != 0 else 366
    arr = np.full(n, 29.0)
    d0 = pd.Timestamp(t0)
    for dd in hot_dates:
        arr[(dd - d0).days] = 31.0
    return arr


# Example 1: hot Apr 29,30 + May 1,2 (4 consecutive hot days straddling boundary)
ex1 = build(pd.Timestamp(2010, 1, 1),
            [pd.Timestamp(2010, 4, 29), pd.Timestamp(2010, 4, 30),
             pd.Timestamp(2010, 5, 1), pd.Timestamp(2010, 5, 2)])
c = current_run(ex1, 2010)
f = full_year_run(ex1)
print("Example 1: hot Apr 29,30 + May 1,2 (2010)")
print(f"   current (truncate-then-detect): count={c[0]} days={c[1]} maxdur={c[2]}")
print(f"   full-year (reference)         : count={f[0]} days={f[1]} maxdur={f[2]}")
print("   -> current: Apr days excluded, only May 1-2 remain = 2-day run -> no event.")
print()

# Example 2: hot Sep 29,30 + Oct 1,2
ex2 = build(pd.Timestamp(2010, 1, 1),
            [pd.Timestamp(2010, 9, 29), pd.Timestamp(2010, 9, 30),
             pd.Timestamp(2010, 10, 1), pd.Timestamp(2010, 10, 2)])
c = current_run(ex2, 2010)
f = full_year_run(ex2)
print("Example 2: hot Sep 29,30 + Oct 1,2 (2010)")
print(f"   current (truncate-then-detect): count={c[0]} days={c[1]} maxdur={c[2]}")
print(f"   full-year (reference)         : count={f[0]} days={f[1]} maxdur={f[2]}")
print("   -> current: Oct days excluded, only Sep 29-30 remain = 2-day run -> no event.")
print()

# Example 3: May 1,2,3 hot -> current should give 1 event (boundary start works)
ex3 = build(pd.Timestamp(2010, 1, 1),
            [pd.Timestamp(2010, 5, 1), pd.Timestamp(2010, 5, 2), pd.Timestamp(2010, 5, 3)])
c = current_run(ex3, 2010)
print(f"Example 3: hot May 1,2,3 (2010) -> current: count={c[0]} days={c[1]} maxdur={c[2]}")
print()

print("Interpretation of the difference:")
print("  - 'truncate-then-detect' (used here): an event must be fully inside")
print("    May-Sep; a run touching Apr or Oct is cut at the boundary and must")
print("    still reach 3 days inside the season to count.")
print("  - 'full-year-detect-then-count' would first find e.g. a 4-day Apr29-May2")
print("    run and then decide attribution; it would count events the current")
print("    scheme does not. This is a definitional choice, not an error, but it")
print("    changes the count/days/duration for boundary-touching events.")
