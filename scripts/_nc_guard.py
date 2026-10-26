# -*- coding: utf-8 -*-
"""
Guarded NetCDF write. Reuse an existing output ONLY when it matches the current
configuration exactly (time coverage, variable set, model params/config). Otherwise
refuse (stop) rather than overwrite or blindly skip.

Every produced dataset carries a "config_signature" global attribute (JSON) holding
the model parameters + configuration, so the guard can compare deterministically.
"""
import os

import numpy as np
import xarray as xr

_TIME_NAMES = ("valid_time", "time", "date", "month")


def _span(ds):
    """Return (first, last, n) of the leading time coord, or None."""
    for tn in _TIME_NAMES:
        if tn in ds.coords and ds.sizes.get(tn, 0) > 0:
            v = ds[tn].values
            return (str(np.datetime_as_string(v[0], unit="m")),
                    str(np.datetime_as_string(v[-1], unit="m")),
                    int(ds.sizes[tn]))
    return None


def inspect_summary(path):
    """Read-only summary of an existing NetCDF (for reporting)."""
    try:
        ds = xr.open_dataset(path)
        s = {
            "path": path,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "vars": sorted(ds.data_vars),
            "span": _span(ds),
            "config_signature": ds.attrs.get("config_signature"),
            "nan_cells": {v: int((~np.isfinite(ds[v].values)).sum())
                          for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.floating)},
        }
        ds.close()
        return s
    except Exception as e:  # noqa: BLE001
        return {"path": path, "error": str(e)}


def write_guarded(ds, path, enc):
    """Write ds -> path; reuse only on an exact (span, vars, config) match.

    Returns "written" or "reused". Raises SystemExit(1) on a config mismatch so the
    caller stops and reports instead of overwriting or silently skipping.
    """
    if not os.path.exists(path):
        ds.to_netcdf(path, encoding=enc)
        print(f"wrote {path}")
        return "written"

    old = xr.open_dataset(path)
    old_vars = set(old.data_vars)
    new_vars = set(ds.data_vars)
    old_span = _span(old)
    new_span = _span(ds)
    old_cfg = old.attrs.get("config_signature")
    new_cfg = ds.attrs.get("config_signature")

    problems = []
    if old_vars != new_vars:
        problems.append(f"variables: existing={sorted(old_vars)} "
                        f"expected={sorted(new_vars)}")
    if old_span != new_span:
        problems.append(f"time coverage: existing={old_span} expected={new_span}")
    if old_cfg != new_cfg:
        problems.append("model params/config (config_signature) differ")
        problems.append(f"   existing: {old_cfg}")
        problems.append(f"   expected: {new_cfg}")

    if problems:
        old.close()
        print(f"STOP: {path} already exists but does NOT match the current "
              f"configuration; refusing to overwrite or reuse.")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    old.close()
    print(f"REUSE (existing file matches time coverage, variables, config): {path}")
    return "reused"
