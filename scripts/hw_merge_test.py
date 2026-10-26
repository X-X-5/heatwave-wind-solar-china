# -*- coding: utf-8 -*-
"""
Synthetic unit tests for the merged heatwave event definition (phase 2).

Covers the confirmed cases plus the retained season-boundary / missing QC
invariants:

    T1  hot3 - cool1 - hot3            -> merge (1 process, gap day)
    T2  hot3 - cool2 - hot3            -> NO merge (2 processes)
    T3  hot2 - cool1 - hot2            -> not an event (both isolated)
    T4  hot3 - missing1 - hot3         -> NO merge (missing never merges)
    T5  hot3-c1-hot3-c1-hot3           -> chained merge (1 process, 2 gaps)
    T6  hot3 - cool - hot(isolated) - cool - hot3 -> NO merge; isolated day is
                                                     code 2 (exceedance, not gap)
    T7  missing breaks a run + never a normal day (code 0)
    T8  intensity over heatwave days ONLY (gap day excess ignored)
    T9  min-run boundary (2-day run is not a heatwave day)
    T10 undecidable day is code 0, never code 1/2/3/4

No data files needed. Run with the project's Python environment.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw_merge_lib as m

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}   {detail}")


def state(s):
    """'H' hot, 'C' cool, 'M' missing -> (hot, cool, missing)."""
    hot = np.array([c == "H" for c in s], dtype=bool)
    cool = np.array([c == "C" for c in s], dtype=bool)
    missing = np.array([c == "M" for c in s], dtype=bool)
    return hot, cool, missing


def run(s, excess=None):
    hot, cool, missing = state(s)
    if excess is None:
        excess = np.where(hot, 1.0, np.nan)
    else:
        excess = np.asarray(excess, dtype=float)
    return m.run_cell(hot, cool, missing, excess)


print("=" * 70)
print("SYNTHETIC TESTS -- merged heatwave definition (gap < 2 days)")
print("=" * 70)

# T1: hot3 - cool1 - hot3 -> merge
_, procs, labels, events = run("HHHCHHH")
check("T1 hot3-cool1-hot3 merges (1 process, 2 segments)",
      len(procs) == 1 and procs[0]["n_segments"] == 2,
      f"procs={[(p['n_segments'], p['n_hot'], p['n_gap'], p['span']) for p in procs]}")
check("T1 event catalog: 6 hot / 1 gap / span 7",
      events[0]["n_hot"] == 6 and events[0]["n_gap"] == 1 and events[0]["span"] == 7,
      f"ev={events[0]}")
check("T1 labels = [3,3,3,4,3,3,3]",
      list(labels) == [3, 3, 3, 4, 3, 3, 3], f"labels={list(labels)}")

# T2: hot3 - cool2 - hot3 -> no merge
_, procs, labels, events = run("HHHCCHHH")
check("T2 hot3-cool2-hot3 does NOT merge (2 processes)",
      len(procs) == 2 and all(p["n_segments"] == 1 for p in procs),
      f"procs={[(p['n_segments'], p['n_gap']) for p in procs]}")
check("T2 labels = [3,3,3,1,1,3,3,3]",
      list(labels) == [3, 3, 3, 1, 1, 3, 3, 3], f"labels={list(labels)}")

# T3: hot2 - cool1 - hot2 -> not an event
_, procs, labels, events = run("HHCHH")
check("T3 hot2-cool1-hot2 -> no event (0 processes)", len(procs) == 0,
      f"procs={procs}")
check("T3 labels = [2,2,1,2,2] (isolated exceedance)",
      list(labels) == [2, 2, 1, 2, 2], f"labels={list(labels)}")

# T4: hot3 - missing1 - hot3 -> no merge
_, procs, labels, events = run("HHHMHHH")
check("T4 hot3-missing1-hot3 does NOT merge (2 processes)", len(procs) == 2,
      f"procs={len(procs)}")
check("T4 labels = [3,3,3,0,3,3,3] (missing is code 0)",
      list(labels) == [3, 3, 3, 0, 3, 3, 3], f"labels={list(labels)}")

# T5: chained merge via 1-day gaps
_, procs, labels, events = run("HHHCHHHCHHH")
check("T5 chained merge -> 1 process, 3 segments, 2 gaps",
      len(procs) == 1 and procs[0]["n_segments"] == 3 and procs[0]["n_gap"] == 2,
      f"procs={[(p['n_segments'], p['n_hot'], p['n_gap'], p['span']) for p in procs]}")
check("T5 event catalog: 9 hot / 2 gap / span 11",
      events[0]["n_hot"] == 9 and events[0]["n_gap"] == 2 and events[0]["span"] == 11,
      f"ev={events[0]}")
check("T5 labels = [3,3,3,4,3,3,3,4,3,3,3]",
      list(labels) == [3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3], f"labels={list(labels)}")

# T6: qualified segments with an isolated exceedance day between them
_, procs, labels, events = run("HHHCHCHHH")
check("T6 isolated exceedance day blocks merge (2 processes)", len(procs) == 2,
      f"procs={[(p['n_segments'], p['n_gap']) for p in procs]}")
check("T6 labels = [3,3,3,1,2,1,3,3,3] (isolated hot = code 2, not gap)",
      list(labels) == [3, 3, 3, 1, 2, 1, 3, 3, 3], f"labels={list(labels)}")

# T7: missing breaks a run (4-day run split into two <3 runs -> no event)
_, procs, labels, events = run("HHMH")
check("T7 missing breaks run -> no event (0 processes)", len(procs) == 0,
      f"procs={procs}")
check("T7 labels = [2,2,0,2]",
      list(labels) == [2, 2, 0, 2], f"labels={list(labels)}")

# T8: intensity over heatwave days only; gap day excess ignored
hot, cool, missing = state("HHHCHHH")
excess = np.array([2.0, 4.0, 6.0, np.nan, 8.0, 10.0, 12.0])
_, procs, labels, events = m.run_cell(hot, cool, missing, excess)
check("T8 intensity = mean over 6 heatwave days = 7.0 (gap ignored)",
      len(events) == 1 and np.isclose(events[0]["mean_excess"], 7.0),
      f"ev={events[0] if events else None}")

# T9: min-run boundary (2-day run is not a heatwave day)
_, procs, labels, events = run("HH")
check("T9 2-day run -> 0 processes, labels [2,2]", len(procs) == 0
      and list(labels) == [2, 2], f"procs={procs} labels={list(labels)}")

# T10: undecidable day is never a normal day
_, procs, labels, events = run("CM")
check("T10 missing day -> code 0 (never 1/2/3/4)", list(labels) == [1, 0],
      f"labels={list(labels)}")

print()
print("=" * 70)
print(f"SUMMARY  PASS={len(PASS)}  FAIL={len(FAIL)}")
for f in FAIL:
    print("  FAILED ->", f)
print("=" * 70)
sys.exit(1 if FAIL else 0)
