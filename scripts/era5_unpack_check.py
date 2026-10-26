# -*- coding: utf-8 -*-
"""
Unpack + integrity-check downloaded ERA5 blocks.

The new CDS returns a ZIP (instant + accum stepTypes) even when the target file
is named .nc. This script:
  * confirms each era5_*.nc is really a zip (magic bytes PK\x03\x04),
  * runs zipfile.testzip() as a CRC/integrity check,
  * extracts each member into its own per-date-block directory
    data_era5/unpacked_<tag>/  (internal CDS names preserved),
  * NEVER overwrites an already-extracted file, and never deletes the original
    downloaded .nc (zip) file.

Idempotent: safe to re-run after each block finishes downloading.
"""
import os
import glob
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_era5")


def main():
    files = sorted(glob.glob(os.path.join(DATA, "era5_*.nc")))
    if not files:
        print("no era5_*.nc files found in", DATA)
        return
    for fp in files:
        base = os.path.basename(fp)
        tag = base[len("era5_"):-len(".nc")]          # e.g. 202208_01_31
        with open(fp, "rb") as f:
            magic = f.read(4)
        if magic != b"PK\x03\x04":
            print(f"[{base}] NOT a zip (magic={magic!r}); skipping")
            continue
        try:
            with zipfile.ZipFile(fp) as z:
                bad = z.testzip()
                if bad is not None:
                    print(f"[{base}] ZIP CORRUPT at member {bad!r}; skipping")
                    continue
                members = z.namelist()
                outdir = os.path.join(DATA, f"unpacked_{tag}")
                os.makedirs(outdir, exist_ok=True)
                for name in members:
                    dest = os.path.join(outdir, os.path.basename(name))
                    if os.path.exists(dest):
                        print(f"[{base}] {dest} already exists (skip, no overwrite)")
                    else:
                        with z.open(name) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                        print(f"[{base}] extracted -> {dest}")
                print(f"[{base}] OK: {len(members)} member(s), integrity test "
                      f"passed, original kept")
        except zipfile.BadZipFile as e:
            print(f"[{base}] NOT a valid zip: {e}")


if __name__ == "__main__":
    main()
