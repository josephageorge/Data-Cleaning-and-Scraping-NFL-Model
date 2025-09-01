#!/usr/bin/env python3
"""
Split QBERT_data_1950_2024_QB1_after_first_start_min50.csv into one CSV per QB1.

Usage (same folder):
  python split_qb1_files.py --outdir "./QB1_splits"

Or specify paths explicitly:
  python split_qb1_files.py \
    --qbert "QBERT_data_1950_2024_QB1_after_first_start_min50.csv" \
    --outdir "./QB1_splits"

Notes:
- Expects columns: frid, Displayname
- Writes one CSV per QB1: {FRID}_{Displayname}.csv (sanitized)
"""

import argparse
import os
import re
import sys
import pandas as pd

DEFAULT_QBERT = "QBERT_data_1950_2024_QB1_after_first_start_min50.csv"

def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qbert",
        default=DEFAULT_QBERT,
        help=f"Path to filtered QBERT CSV (default: {DEFAULT_QBERT})"
    )
    ap.add_argument("--outdir", required=True, help="Output directory for per-QB CSVs")
    args = ap.parse_args()

    try:
        qbert = pd.read_csv(args.qbert)
    except Exception as e:
        print(f"Failed to read QBERT CSV at '{args.qbert}': {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"frid", "Displayname"}
    if not required_cols.issubset(qbert.columns):
        missing = required_cols - set(qbert.columns)
        print(f"QBERT file missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    total_written = 0
    for frid, df_qb in qbert.groupby("frid"):
        displayname = (
            df_qb["Displayname"].mode().iat[0]
            if not df_qb["Displayname"].empty
            else frid
        )
        base = sanitize_filename(f"{frid}_{displayname}.csv")
        out_path = os.path.join(args.outdir, base)
        df_qb.to_csv(out_path, index=False)
        total_written += 1

    print(f"Done. Wrote {total_written} per-QB CSVs to: {args.outdir}")

if __name__ == "__main__":
    main()
