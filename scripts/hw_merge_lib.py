# -*- coding: utf-8 -*-
"""
Merged heatwave event library (phase 2, research definition).

This module implements the UPDATED formal event definition adopted from the
advisor's feedback. It supersedes the strict ">=3 consecutive days" grouping in
`hw_lib.py` while reusing its threshold/slot/season machinery (thresholds are
unchanged -- only the event *grouping* changes).

Definition implemented here (see docs/heatwave_method_v2.md):

  * Threshold: per grid cell and per calendar day (366-day leap-year frame),
    90th percentile of Tmax over baseline 1981-2010 from the 5-day centred
    window (day-2..day+2, cross-year wrap). IDENTICAL to phase 1 -- reused
    from results_fixed/hw_threshold_1981-2010.nc.

  * Hot day: Tmax STRICTLY greater than the calendar-day threshold, with both
    Tmax and threshold finite ("decidable").

  * Qualified heat segment: a maximal run of >= MIN_RUN (3) consecutive hot
    days.

  * Merged event (process): adjacent qualified segments are merged into ONE
    process when the gap between them is
        (a) shorter than 2 days -- i.e. exactly ONE complete intervening day
            (gap < 2, NOT gap <= 2; a 2-day gap does NOT merge), and
        (b) made only of valid cool days (finite Tmax, finite threshold,
            Tmax <= threshold).
    Missing/undecidable days never merge; an isolated hot day (part of a
    <3-day run) sits in the gap and therefore also blocks merging (it is
    neither a gap day nor absorbed -- see classification below). Merging is
    transitive: a chain of 1-day cool gaps links segments into one process.

  * Per-day classification (codes):
        0  missing/undecidable   Tmax or threshold invalid (NEVER a normal day)
        1  cool day              valid, Tmax <= threshold, not a merge gap day
        2  isolated exceedance   valid, Tmax > threshold, NOT in a >=3-day run
        3  heatwave day          valid, Tmax > threshold, in a >=3-day run
        4  gap day               valid, Tmax <= threshold, 1-day gap inside a
                                 merged process
    "Exceedance day" (超阈值日) = code 2 or 3; "heatwave day" (热浪日) = code 3
    only; "gap day" (间隔日) = code 4 only.

Units: Tmax, thresholds and exceedance intensity in degrees Celsius.
"""
from __future__ import annotations

import numpy as np

# Per-day classification codes (single source of truth).
LABEL_MISSING = 0          # Tmax or threshold invalid -> cannot decide
LABEL_COOL = 1             # valid non-exceedance day, not a merge gap
LABEL_EXCEED_ISOLATED = 2  # valid exceedance day not in a >=3-day run
LABEL_HEATWAVE = 3         # valid exceedance day in a >=3-day run
LABEL_GAP = 4              # valid cool day serving as a 1-day merge gap

LABEL_NAMES = {
    LABEL_MISSING: "missing/undecidable",
    LABEL_COOL: "cool (non-heatwave) day",
    LABEL_EXCEED_ISOLATED: "isolated exceedance day",
    LABEL_HEATWAVE: "heatwave day",
    LABEL_GAP: "gap day (1-day cool gap in merged process)",
}

MIN_RUN = 3
MAX_GAP = 1                # merge only when the complete gap is < 2 days (== 1)


def find_qualified_segments(hot, min_run=MIN_RUN):
    """Maximal runs of consecutive hot days with length >= min_run.

    hot : 1-D bool array (True = decidable hot day).

    Returns a list of (start, end) inclusive index tuples. Runs shorter than
    min_run (isolated exceedance days) are deliberately omitted -- they are not
    heatwave days and are never absorbed into a process.
    """
    segs = []
    n = len(hot)
    i = 0
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            if j - i >= min_run:
                segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def _gap_mergeable(prev_e, s, cool, missing, max_gap=MAX_GAP):
    """True if the gap between a segment ending at prev_e and one starting at s
    is a valid, short-enough cool gap."""
    gap = list(range(prev_e + 1, s))
    # Segments are maximal runs, so a gap of 0 can only arise from adjacent
    # runs (impossible here); still guard len(gap) <= max_gap explicitly.
    if len(gap) == 0:
        return True
    if len(gap) > max_gap:
        return False
    return all(cool[d] and not missing[d] for d in gap)


def merge_qualified_segments(segs, cool, missing, max_gap=MAX_GAP):
    """Merge adjacent qualified segments into processes (transitive chaining).

    Parameters
    ----------
    segs    : list of (start, end) from find_qualified_segments.
    cool    : 1-D bool, True = decidable cool day (finite & Tmax <= threshold).
    missing : 1-D bool, True = undecidable day (Tmax or threshold invalid).

    Returns
    -------
    list of processes. Each process is a dict:
        start, end   : inclusive day indices of the FIRST and LAST hot day.
        segments     : list of (s, e) qualified segments merged into it.
        gap_days     : list of day indices of the 1-day cool gaps.
        n_hot        : number of heatwave days.
        n_gap        : number of gap days.
        span         : end - start + 1 (includes gap days).
    """
    if not segs:
        return []
    processes = []
    cur_segs = [segs[0]]
    cur_gaps = []
    for (s, e) in segs[1:]:
        prev_e = cur_segs[-1][1]
        gap = list(range(prev_e + 1, s))
        if _gap_mergeable(prev_e, s, cool, missing, max_gap):
            cur_segs.append((s, e))
            cur_gaps.extend(gap)
        else:
            processes.append(_make_process(cur_segs, cur_gaps))
            cur_segs = [(s, e)]
            cur_gaps = []
    processes.append(_make_process(cur_segs, cur_gaps))
    return processes


def _make_process(segments, gap_days):
    start = segments[0][0]
    end = segments[-1][1]
    n_hot = sum(e - s + 1 for s, e in segments)
    return {
        "start": start,
        "end": end,
        "segments": list(segments),
        "n_segments": len(segments),
        "gap_days": list(gap_days),
        "n_hot": n_hot,
        "n_gap": len(gap_days),
        "span": end - start + 1,
    }


def classify_days(hot, cool, missing, segs, processes):
    """Assign a per-day classification code (LABEL_*).

    hot, cool, missing : 1-D bool arrays (cool & missing are disjoint; hot is a
                         subset of ~missing).
    segs               : qualified segments (from find_qualified_segments).
    processes          : merged processes (from merge_qualified_segments).

    Returns
    -------
    labels : int8 array, same length as `hot`.
    """
    n = len(hot)
    labels = np.full(n, LABEL_MISSING, dtype=np.int8)
    hw = np.zeros(n, dtype=bool)
    for (s, e) in segs:
        hw[s:e + 1] = True
    gap = np.zeros(n, dtype=bool)
    for p in processes:
        for d in p["gap_days"]:
            gap[d] = True

    for d in range(n):
        if missing[d]:
            labels[d] = LABEL_MISSING
        elif hot[d]:
            labels[d] = LABEL_HEATWAVE if hw[d] else LABEL_EXCEED_ISOLATED
        else:  # cool
            labels[d] = LABEL_GAP if gap[d] else LABEL_COOL
    return labels


def run_cell(hot, cool, missing, excess):
    """Full per-cell detection: segments, processes, labels, event records.

    excess : 1-D float, (Tmax - threshold); meaningful only on hot days.

    Returns (segs, processes, labels, events) where `events` is a list of dicts
    with the merged-event catalog fields (start, end, n_hot, n_gap, span,
    mean_excess). mean_excess is over HEATWAVE days only (qualified segment
    days), never gap days or isolated exceedance days.
    """
    segs = find_qualified_segments(hot)
    processes = merge_qualified_segments(segs, cool, missing)
    labels = classify_days(hot, cool, missing, segs, processes)

    events = []
    for p in processes:
        seg_idx = []
        for (s, e) in p["segments"]:
            seg_idx.append(np.arange(s, e + 1))
        idx = np.concatenate(seg_idx)
        mean_excess = float(np.nanmean(excess[idx]))
        events.append({
            "start": p["start"],
            "end": p["end"],
            "n_segments": len(p["segments"]),
            "n_hot": p["n_hot"],
            "n_gap": p["n_gap"],
            "span": p["span"],
            "mean_excess": mean_excess,
        })
    return segs, processes, labels, events
