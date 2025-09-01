import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DIVISION_MAP = {
    "BUF":"AFC East","MIA":"AFC East","NE":"AFC East","NYJ":"AFC East",
    "BAL":"AFC North","CIN":"AFC North","CLE":"AFC North","PIT":"AFC North",
    "HOU":"AFC South","IND":"AFC South","JAX":"AFC South","TEN":"AFC South",
    "DEN":"AFC West","KC":"AFC West","LV":"AFC West","LAC":"AFC West",
    "DAL":"NFC East","NYG":"NFC East","PHI":"NFC East","WAS":"NFC East",
    "CHI":"NFC North","DET":"NFC North","GB":"NFC North","MIN":"NFC North",
    "ATL":"NFC South","CAR":"NFC South","NO":"NFC South","TB":"NFC South",
    "ARI":"NFC West","LAR":"NFC West","SF":"NFC West","SEA":"NFC West",
}

PLAYOFF_NAMES_BY_COUNT = {
    4: ["Wild Card", "Divisional", "Conference", "Super Bowl"],
    3: ["Divisional", "Conference", "Super Bowl"],
    2: ["Conference", "Super Bowl"],
    1: ["Super Bowl"],
}


def dedupe_names_by_frid(wide_df: pd.DataFrame, frid_to_name: dict) -> pd.DataFrame:
    rename_map = {}
    seen = {}
    for col in wide_df.columns:
        if col == "SeasonWeek":
            continue
        name = frid_to_name.get(col, col)
        if pd.isna(name) or str(name).strip() == "":
            name = col  # fallback to FRID
        base = str(name)
        seen[base] = seen.get(base, 0) + 1
        rename_map[col] = base if seen[base] == 1 else f"{base} [{col}]"
    return wide_df.rename(columns=rename_map)


def add_seasonweek_with_playoff_labels(df: pd.DataFrame, include_playoffs: bool) -> pd.DataFrame:
    out = df.copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)

    out["SeasonWeek"] = out["season"].astype(str) + ", Week " + out["week"].astype(str)

    if not include_playoffs:
        # Simple chronological sort key for regular season
        out["SeasonWeekIndex"] = out["season"] * 100 + out["week"]
        return out

    gd = pd.to_datetime(out["game_date"].astype(str), format="%d%b%Y", errors="coerce")
    gd_fallback = pd.to_datetime(out["game_date"].astype(str), errors="coerce")
    out["gd_parsed"] = gd.fillna(gd_fallback)

    playoffs = out[out["playoffs"] == 1].copy()
    if not playoffs.empty:
        pw = (
            playoffs.groupby(["season", "week"], as_index=False)["gd_parsed"]
            .min()
            .rename(columns={"gd_parsed": "first_date"})
        )
        pw["rank"] = pw.sort_values(["season", "first_date"]) \
                       .groupby("season").cumcount() + 1

        label_map = {}
        for s, g in pw.groupby("season"):
            n = len(g)
            names = PLAYOFF_NAMES_BY_COUNT.get(n, None)
            if names is None:
                names = [f"Playoffs #{k}" for k in range(1, n + 1)]
            for (_, row), nm in zip(g.sort_values("first_date").iterrows(), names):
                label_map[(int(row["season"]), int(row["week"]))] = nm

        mask_po = out["playoffs"] == 1
        out.loc[mask_po, "SeasonWeek"] = out.loc[mask_po].apply(
            lambda r: f"{int(r['season'])}, {label_map.get((int(r['season']), int(r['week'])), 'Playoffs')}",
            axis=1
        )


        out = out.merge(
            pw[["season", "week", "rank"]],
            on=["season", "week"],
            how="left"
        )
        out["SeasonWeekIndex"] = np.where(
            out["playoffs"] == 0,
            out["season"] * 100 + out["week"],
            out["season"] * 100 + 100 + out["rank"].fillna(99).astype(int)  # playoffs after reg season
        )
    else:
        out["SeasonWeekIndex"] = out["season"] * 100 + out["week"]

    return out.drop(columns=["gd_parsed"])


def pivot_metric_wide_player(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    df = df.copy()
    for division, g in df.groupby("Division", dropna=True):
        g = g.sort_values(["SeasonWeekIndex", "frid"])
        ordered_sw = (
            g[["SeasonWeek", "SeasonWeekIndex"]]
            .drop_duplicates()
            .sort_values("SeasonWeekIndex")["SeasonWeek"]
            .tolist()
        )
        wide_frid = (
            g.pivot_table(index="SeasonWeek", columns="frid", values=metric, aggfunc="first")
             .reindex(ordered_sw)
             .reset_index()
        )
        frid_to_name = (
            g.groupby("frid")["player"]
             .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else s.iloc[0])
             .to_dict()
        )
        wide_player = dedupe_names_by_frid(wide_frid, frid_to_name)
        out_path = outdir / "wide" / f"QB1_{division.replace(' ', '_')}__{metric}__WIDE_PLAYER.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wide_player.to_csv(out_path, index=False)
        print(f"[write] {out_path}")

    ordered_sw_all = (
        df[["SeasonWeek", "SeasonWeekIndex"]]
        .drop_duplicates()
        .sort_values("SeasonWeekIndex")["SeasonWeek"]
        .tolist()
    )
    wide_all_frid = (
        df.pivot_table(index="SeasonWeek", columns="frid", values=metric, aggfunc="first")
         .reindex(ordered_sw_all)
         .reset_index()
    )
    frid_to_name_all = (
        df.groupby("frid")["player"]
          .agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else s.iloc[0])
          .to_dict()
    )
    wide_all_player = dedupe_names_by_frid(wide_all_frid, frid_to_name_all)
    out_path_all = outdir / "wide" / f"QB1_ALL_DIVISIONS__{metric}__WIDE_PLAYER.csv"
    out_path_all.parent.mkdir(parents=True, exist_ok=True)
    wide_all_player.to_csv(out_path_all, index=False)
    print(f"[write] {out_path_all}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True, help="Path to NFL QB depth chart CSV")
    ap.add_argument("--qbert", required=True, help="Path to QBERT data CSV")
    ap.add_argument("--outdir", default="qb1_outputs", help="Output directory root")
    ap.add_argument("--metric", default="qb_prior", help="Metric to pivot (e.g., qb_prior, qbert, adj_qbert, war)")
    ap.add_argument("--include_playoffs", action="store_true", help="Include playoff games and label them")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    depth = pd.read_csv(args.depth)
    qbert = pd.read_csv(args.qbert)

    need_depth = {"Team", "Order", "FRID"}
    need_qbert = {"frid", "assumed_starter", "season", "week", "playoffs", "game_date", args.metric}
    miss_d = need_depth - set(depth.columns)
    miss_q = need_qbert - set(qbert.columns)
    if miss_d:
        raise ValueError(f"Depth chart missing columns: {sorted(miss_d)}")
    if miss_q:
        raise ValueError(f"QBERT missing columns needed for '{args.metric}': {sorted(miss_q)}")

    qb1 = depth[depth["Order"] == "QB1"][["FRID", "Team"]].dropna()

    starts = qbert[qbert["assumed_starter"] == 1].copy()

    merged = starts.merge(qb1, left_on="frid", right_on="FRID", how="inner")
    merged["Division"] = merged["Team"].map(DIVISION_MAP)

    if "player" not in merged.columns:
        merged["player"] = merged["frid"]

    merged = add_seasonweek_with_playoff_labels(merged, include_playoffs=args.include_playoffs)

    merged = merged.sort_values(["Division", "Team", "frid", "SeasonWeekIndex"])

    aug_dir = outdir / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "player","frid","game_date","season","playoffs","week",
        "sb_fran_id","sb_def_id","assumed_starter","est_plays",
        "qbert","adj_qbert","qb_prior","qb_posterior","war",
        "Team","Division","SeasonWeek","SeasonWeekIndex"
    ]
    cols = [c for c in cols if c in merged.columns]
    merged[cols].to_csv(aug_dir / "QB1_starts_all_divisions_augmented.csv", index=False)
    for div, g in merged.groupby("Division", dropna=True):
        g[cols].to_csv(aug_dir / f"QB1_{div.replace(' ', '_')}_augmented.csv", index=False)

    pivot_metric_wide_player(merged, args.metric, outdir)

    print("[done]")


if __name__ == "__main__":
    main()
