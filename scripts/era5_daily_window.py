# -*- coding: utf-8 -*-
"""
Pure time-window helpers for the ERA5 hourly -> daily aggregation.

These are exactly the rules validated by scripts/era5_daily_aggregation_test.py:

  - instantaneous variables (t2m / u-v wind / wpd): POINT samples whose timestamp
    falls in [T_start, T_end);
  - end-of-hour accumulations (ssrd, and derived ghi / pvpot): INTERVAL [t-1h, t]
    fully inside [T_start, T_end), i.e. timestamps in (T_start, T_end].

Candidate CN05.1 day windows (date label D, Beijing = UTC+8):
  A: calendar day  00-24 BJT            -> [D-1 16:00, D 16:00) UTC
  B: station 20-20 BJT (provisional primary) -> [D-1 12:00, D 12:00) UTC
     (CN05.1 day boundary still UNCONFIRMED)
"""
import numpy as np

H = np.timedelta64(1, "h")


def window_calendar_day(D):
    d = np.datetime64(D).astype("datetime64[h]")
    start = d - np.timedelta64(8, "h")           # D 00:00 BJT -> D-1 16:00 UTC
    return start, start + np.timedelta64(24, "h")


def window_2020_day(D):
    d = np.datetime64(D).astype("datetime64[h]")
    start = d - np.timedelta64(12, "h")          # D-1 20:00 BJT = D-1 12:00 UTC
    return start, start + np.timedelta64(24, "h")


def point_mask(time, start, end):                # instantaneous: timestamp in [start, end)
    return (time >= start) & (time < end)


def accum_mask(time, start, end):                # ssrd/ghi/pvpot: [t-1h, t] fully in [start, end)
    return (time >= start + H) & (time <= end)


WINDOWS = {
    "A": ("calendar 00-24 BJT", window_calendar_day),
    "B": ("station 20-20 BJT (provisional primary; CN05.1 boundary UNCONFIRMED)",
          window_2020_day),
}
