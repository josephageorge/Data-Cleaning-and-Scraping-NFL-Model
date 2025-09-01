#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def to_monthly_like_original(infile, outfile=None, date_col=None):
    df = pd.read_csv(infile)
    original_cols = df.columns.tolist()

    if date_col is None:
        if "datetime" in df.columns:
            date_col = "datetime"
        else:
            candidates = [c for c in df.columns if "date" in c.lower()]
            if not candidates:
                raise ValueError("No datetime column found. Pass --date-col.")
            date_col = candidates[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"Could not parse any dates in '{date_col}'.")

    df["Year"] = df[date_col].dt.year
    df["Month_num"] = df[date_col].dt.month

    num_cols = df.select_dtypes(include="number").columns.tolist()
    non_num_cols = [c for c in original_cols if c not in num_cols]
    agg_map = {c: "mean" for c in num_cols}
    for c in non_num_cols:
        if c != date_col:
            agg_map[c] = "first"

    # group & sort
    g = (df.groupby(["Year", "Month_num"], as_index=False)
            .agg(agg_map)
            .sort_values(["Year", "Month_num"]))

    # set datetime to first of month
    g[date_col] = pd.to_datetime(dict(year=g["Year"], month=g["Month_num"], day=1))

    # drop helpers, restore column order
    g = g.drop(columns=["Year", "Month_num"], errors="ignore")
    ordered = [c for c in original_cols if c in g.columns]
    rest = [c for c in g.columns if c not in ordered]
    g = g[ordered + rest]

    # output
    if outfile is None:
        p = Path(infile)
        outfile = str(p.with_name(f"{p.stem}_monthly_like_original.csv"))
    g.to_csv(outfile, index=False)
    return outfile

def main():
    ap = argparse.ArgumentParser(description="Monthly-average a daily CSV while preserving original column order.")
    ap.add_argument("infile")
    ap.add_argument("-o", "--outfile", default=None)
    ap.add_argument("--date-col", default=None)
    args = ap.parse_args()
    out = to_monthly_like_original(args.infile, args.outfile, args.date_col)
    print(out)

if __name__ == "__main__":
    main()

