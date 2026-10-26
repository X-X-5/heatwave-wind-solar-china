# -*- coding: utf-8 -*-
"""Recover already-submitted CDS requests by their request IDs.

Used for warm-season monthly requests whose status-query HTTP calls failed with a
transient SSLError (not the deterministic "request too large" error). Those requests
were ACCEPTED server-side, so the jobs may still be live and their results fetchable
without re-submitting. (A request that failed at SUBMISSION has no request ID and
cannot be recovered here — re-submit it instead.)

Behaviour (per user instruction):
  - Query status by the EXISTING request ID (never re-submit first).
  - "successful"            -> download the original result to --target (temp
                               file, ZIP check, then atomic replace).
  - "accepted" / "running"  -> report queued and exit 1 (keep waiting).
  - anything else / not found -> report unrecoverable and exit 5 (record reason,
                               then the caller may re-submit).

Exit codes: 0=recovered, 1=still queued, 2=status-query/transient error,
3=download error, 4=bad ZIP, 5=unrecoverable (failed/rejected/dismissed/deleted).

ZIP integrity is zipfile.is_zipfile (valid header/EOCD) + ZipFile.testzip
(CRC-check every member) — a bare PK header is NOT sufficient, and an existing
target is NOT accepted on existence alone.

With multiple --request-id/--target pairs (or --poll on a single pair), the script
polls in a SINGLE process: re-check each pending ID every --interval seconds up to
--max-wait, downloading each as soon as it becomes "successful".

The client is cdsapi.Client() == LegacyClient, which wraps
ecmwf.datastores.Client; get_remote(request_id) does
GET {url}/retrieve/v1/jobs/{request_id}?log=True&request=True and the Remote object
exposes .status / .download(). Credentials are read from ~/.cdsapirc by the library
and are never printed here.
"""
import argparse
import os
import sys
import time
import zipfile

import cdsapi


def _zip_ok(path):
    """ZIP integrity = zipfile.is_zipfile (valid header/EOCD) + ZipFile.testzip
    (CRC-check every member). A bare PK magic header is NOT sufficient."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and len(z.namelist()) >= 1
    except zipfile.BadZipFile:
        return False


def _query_once(request_id, target, client):
    try:
        remote = client.get_remote(request_id)
    except Exception as e:  # job-not-found HTTPError, transient SSL, etc.
        print("QUERY-FAILED %s: %s: %s" % (request_id, type(e).__name__, e))
        return 2

    try:
        status = remote.status
    except Exception as e:
        print("STATUS-FAILED %s: %s: %s" % (request_id, type(e).__name__, e))
        return 2

    print("STATUS %s = %s" % (request_id, status))

    if status == "successful":
        if os.path.exists(target):
            if _zip_ok(target):
                print("SKIP %s: already present and ZIP-verified (is_zipfile+testzip)" % target)
                return 0
            # Do NOT overwrite / delete the existing file; do NOT mark done.
            print("EXISTS-BUT-INVALID %s: present but fails is_zipfile/testzip; "
                  "leaving in place (no overwrite, not done)" % target)
            return 4
        tmp = target + ".recover"
        try:
            remote.download(tmp)
        except Exception as e:
            print("DOWNLOAD-FAILED %s: %s: %s" % (request_id, type(e).__name__, e))
            return 3
        if not _zip_ok(tmp):
            print("BAD-ZIP %s (fails is_zipfile/testzip); leaving temp file for inspection" % tmp)
            return 4
        os.replace(tmp, target)
        print("WROTE %s" % target)
        return 0

    if status in ("accepted", "running"):
        print("QUEUED %s = %s (not downloading, not re-submitting)" % (request_id, status))
        return 1

    # failed / rejected / dismissed / deleted / unknown
    reason = ""
    try:
        reply = remote.json
        meta = reply.get("metadata", {})
        reason = str(meta.get("error") or meta.get("statusInfo") or reply.get("status", ""))
    except Exception:
        pass
    print("UNRECOVERABLE %s = %s reason=%r" % (request_id, status, reason))
    return 5


def _poll_many(pairs, interval, max_wait):
    # Bounded retries (not the library default 500x120s); the poll loop re-invokes.
    c = cdsapi.Client(retry_max=20, sleep_max=60)  # LegacyClient; reads ~/.cdsapirc internally
    client = c.client                                   # ecmwf.datastores.client.Client

    pending = {rid: tgt for rid, tgt in pairs}
    outcomes = {}
    deadline = time.time() + max_wait
    while pending and time.time() < deadline:
        for rid, tgt in list(pending.items()):
            if os.path.exists(tgt) and _zip_ok(tgt):
                print("RECOVERED %s -> %s (already on disk, ZIP-verified)" % (rid, tgt))
                outcomes[rid] = "recovered"
                del pending[rid]
                continue
            rc = _query_once(rid, tgt, client)
            if rc == 0:
                outcomes[rid] = "recovered"
                del pending[rid]
            elif rc in (4, 5):
                outcomes[rid] = "unrecoverable(rc=%d)" % rc
                del pending[rid]
            # rc 1 (queued) / 2 (query error) / 3 (download error) -> keep pending
        if pending:
            time.sleep(interval)

    for rid in list(pending):
        outcomes[rid] = "still-pending(timeout)"

    print("=" * 60)
    all_ok = True
    for rid, tgt in pairs:
        print("%-10s %-22s %s" % (outcomes.get(rid, "?"), rid, tgt))
        if outcomes.get(rid) != "recovered":
            all_ok = False
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--request-id", action="append", required=True,
                    help="existing CDS request UUID (repeat for several)")
    ap.add_argument("--target", action="append", required=True,
                    help="output .nc path (raw ZIP), one per --request-id")
    ap.add_argument("--poll", action="store_true",
                    help="loop while queued / transient error (single process)")
    ap.add_argument("--interval", type=int, default=180, help="poll interval seconds")
    ap.add_argument("--max-wait", type=int, default=14400, help="max total wait seconds")
    args = ap.parse_args()

    if len(args.request_id) != len(args.target):
        raise SystemExit("number of --request-id and --target must match")

    pairs = list(zip(args.request_id, args.target))

    if len(pairs) == 1 and not args.poll:
        c = cdsapi.Client(retry_max=20, sleep_max=60)
        return _query_once(pairs[0][0], pairs[0][1], c.client)

    return _poll_many(pairs, args.interval, args.max_wait)


if __name__ == "__main__":
    sys.exit(main())
