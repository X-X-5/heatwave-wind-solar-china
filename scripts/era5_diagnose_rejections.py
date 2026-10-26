# -*- coding: utf-8 -*-
"""Diagnose the server-side rejection reason for the 2013 ERA5 jobs.

Read-only: lists recent CDS jobs and dumps full metadata (status, request params,
and any error/reason field) for the 2013 monthly jobs. Never prints credentials.
"""
import json
import sys

RIDS = {
    "201304_30_30": "490eaf13-9512-4e13-8652-f293c9dc633f",
    "201305_01_31": "63bf99a2-91fa-40d7-a775-3ff40f13475d",
    "201306_01_30": "8d835ab9-1cfa-4c16-aa0a-acd31ab9eb9d",
    "201307_01_31": "6d90aafe-7c87-4c7f-b444-6d444fc5b034",
    "201308_01_31": "ec08ade2-ab1d-435e-b09b-d7f1523475ca",
    "201309_01_30": "361f688d-2563-4b89-ad31-a263035ef75c",
    "201310_01_01": "0d9a5fe4-d067-4d55-b848-d97cfd35efa9",
}


def _safe(obj):
    """Return a json-safe repr of an object's public attributes."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # unknown object: dump public attrs
    d = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        d[k] = _safe(v)
    return d


def main():
    import cdsapi
    c = cdsapi.Client(retry_max=2, sleep_max=10)   # reads ~/.cdsapirc internally
    dc = c.client

    print("=" * 78)
    print("PART A: recent jobs list (raw metadata, fields incl. any error/reason)")
    print("=" * 78)
    try:
        jobs = dc.get_jobs(sortby="-created", limit=50)
        jd = jobs._json_dict
        print("raw top-level keys:", sorted(jd.keys()))
        jlist = jd.get("jobs", [])
        print("num jobs listed:", len(jlist))
        for j in jlist:
            jid = j.get("jobID")
            created = j.get("created") or j.get("created_at") or j.get("submitted")
            status = j.get("status")
            # only print interesting/2013-ish rows compactly
            print("-" * 78)
            print("job:", jid, "status:", status, "created:", created)
            # print every key EXCEPT anything credential-like
            for k in sorted(j.keys()):
                if k in ("jobID", "status", "created"):
                    continue
                v = j[k]
                s = json.dumps(v, ensure_ascii=False, default=str)
                if len(s) > 400:
                    s = s[:400] + "...(truncated)"
                print("   %s = %s" % (k, s))
    except Exception as e:
        print("get_jobs FAILED:", type(e).__name__, e)

    print()
    print("=" * 78)
    print("PART B: per-request-ID get_remote (status + full attrs)")
    print("=" * 78)
    for tag, rid in RIDS.items():
        print("-" * 78)
        print("TAG", tag, "RID", rid)
        try:
            remote = dc.get_remote(rid)
        except Exception as e:
            print("   get_remote FAILED:", type(e).__name__, e)
            continue
        print("   status:", getattr(remote, "status", "<none>"))
        # dump everything public
        d = _safe(remote)
        print("   attrs:", json.dumps(d, ensure_ascii=False, default=str)[:1200])
        # try common reason/error accessors
        for attr in ("reason", "error", "message", "info", "logs", "response",
                     "request", "_json_dict", "json"):
            try:
                v = getattr(remote, attr)
            except Exception:
                continue
            if v is None:
                continue
            print("   .%s = %s" % (attr, json.dumps(_safe(v), ensure_ascii=False,
                                                    default=str)[:1200]))


if __name__ == "__main__":
    main()
