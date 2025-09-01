#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_seed_lock.py — optimized, with historical tiebreakers

Implements your blurb:
1) Pre-1972 ties-ignored win% formula.
2) Division tiebreakers (all eras): H2H → Division → Common (if applicable) → Conference → SOV → SOS → coin toss.
3) 1970–1977 WC tiebreakers: H2H/mini-league → intra-conference → coin toss.
4) Post-1977 WC tiebreakers: H2H sweep → Conference → Common (if applicable) → SOV → SOS → coin toss.
5) Apply tiebreakers inside simulation; coin toss ⇒ ambiguous ⇒ not locked.

Dataset columns used:
  season,date,team,opp,conference,division,playoff,url,result,home_or_away,
  wins_before,losses_before,ties_before,wins_after,losses_after,ties_after
"""

from __future__ import annotations
import argparse
import itertools
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Optional

import numpy as np
import pandas as pd

# ------------------------- Logging helpers -------------------------

TRUE_STRS = {"1","true","t","yes","y"}

def log(msg: str, *, enable: bool = True):
    if enable:
        print(msg)
        sys.stdout.flush()

# ------------------------- Basic utilities -------------------------

def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(TRUE_STRS)

def add_wlt(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> Tuple[int,int,int]:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def season_games(season: int) -> int:
    if season < 1978:  # 1970–1977
        return 14
    if season < 2021:  # 1978–2020
        return 16
    return 17          # 2021–present

def pct_era(wlt: Tuple[int,int,int], season: int) -> float:
    """Win% with historical rule: pre-1972 ties ignored."""
    w, l, t = wlt
    if season < 1972:
        g = w + l
        return 0.0 if g == 0 else w / g
    g = w + l + t
    return 0.0 if g == 0 else (w + 0.5*t) / g

def within_one_game(a: Tuple[int,int,int], b: Tuple[int,int,int], season: int) -> bool:
    thr = 1.0 / season_games(season)
    return abs(pct_era(a, season) - pct_era(b, season)) <= thr + 1e-9

def wildcard_slots(year: int) -> int:
    # 1970–1977: 1; 1978–1989: 2; 1990–2001: 3; 2002–2019: 2; 2020– : 3
    if year == 1982:
        return 0
    if year < 1978: return 1
    if year < 1990: return 2
    if year < 2002: return 3
    if year < 2020: return 2
    return 3

def is_tournament_skip(season: int) -> bool:
    return season == 1982

# ------------------------- Meta helpers ----------------------------

def get_team_meta(df_season: pd.DataFrame) -> Dict[str, Dict[str,str]]:
    meta = {}
    for tm, g in df_season.groupby("team"):
        r = g.iloc[0]
        meta[tm] = {"conference": str(r["conference"]), "division": str(r["division"])}
    return meta

def order_overall_basic(teams: List[str], wlt_map: Dict[str,Tuple[int,int,int]], season: int) -> List[str]:
    return sorted(
        teams,
        key=lambda t: (round(pct_era(wlt_map.get(t,(0,0,0)), season), 6),
                       wlt_map.get(t,(0,0,0))[0]),
        reverse=True
    )

# -------------------- Build baseline “prior to date” ---------------

def dedupe_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per actual game; prefer home row when present."""
    if df.empty:
        return df
    if "home_or_away" in df.columns:
        df = df.assign(_home=(df["home_or_away"].astype(str).str.lower() == "home").astype(int))
        df = (df.sort_values(["url","_home"])
                .drop_duplicates(subset=["url"], keep="last")
                .drop(columns=["_home"]))
    else:
        df = df.drop_duplicates(subset=["url"], keep="first")
    return df.reset_index(drop=True)

def build_baseline_totals(df_season: pd.DataFrame, final_date: pd.Timestamp) -> Dict[str,Tuple[int,int,int]]:
    """
    Overall W/L/T as of morning of final_date:
      - Teams playing on final_date: *_before on that date
      - Teams not playing: *_after from last game
    """
    on_date = df_season[df_season["date"] == final_date]
    playing = set(on_date["team"].unique())
    last_rows = df_season.sort_values("date").groupby("team").tail(1)

    wlt: Dict[str, Tuple[int,int,int]] = {}
    for _, r in on_date.iterrows():
        t = r["team"]
        wlt[t] = (int(r["wins_before"]), int(r["losses_before"]), int(r["ties_before"]))
    for _, r in last_rows.iterrows():
        t = r["team"]
        if t in playing:
            continue
        wlt[t] = (int(r["wins_after"]), int(r["losses_after"]), int(r["ties_after"]))
    return wlt

def build_baseline_metrics(df_season: pd.DataFrame,
                           meta: Dict[str,Dict[str,str]],
                           final_date: pd.Timestamp):
    """
    Build “prior to kickoff” metrics from rows with date < final_date:
    - h2h[(A,B)] from A's perspective
    - conf_wlt[team], div_wlt[team]
    - beat_list[team] (opponents beaten)
    - opps_list[team] (all opponents faced)
    """
    prior = dedupe_games(df_season[df_season["date"] < final_date])
    h2h   : Dict[Tuple[str,str],Tuple[int,int,int]] = defaultdict(lambda:(0,0,0))
    conf_w: Dict[str,Tuple[int,int,int]] = defaultdict(lambda:(0,0,0))
    div_w : Dict[str,Tuple[int,int,int]] = defaultdict(lambda:(0,0,0))
    beat  : Dict[str,List[str]] = defaultdict(list)
    opps  : Dict[str,List[str]] = defaultdict(list)

    for _, row in prior.iterrows():
        t, o = row["team"], row["opp"]
        res = str(row["result"]).upper()[:1]
        same_conf = meta.get(t,{}).get("conference","") == meta.get(o,{}).get("conference","")
        same_div  = meta.get(t,{}).get("division","")   == meta.get(o,{}).get("division","")

        # H2H
        if res == "W":
            h2h[(t,o)] = add_wlt(h2h[(t,o)], (1,0,0))
            h2h[(o,t)] = add_wlt(h2h[(o,t)], (0,1,0))
            beat[t].append(o)
        elif res == "L":
            h2h[(t,o)] = add_wlt(h2h[(t,o)], (0,1,0))
            h2h[(o,t)] = add_wlt(h2h[(o,t)], (1,0,0))
            beat[o].append(t)
        elif res == "T":
            h2h[(t,o)] = add_wlt(h2h[(t,o)], (0,0,1))
            h2h[(o,t)] = add_wlt(h2h[(o,t)], (0,0,1))
        # Conference / Division
        if same_conf:
            if res == "W":
                conf_w[t] = add_wlt(conf_w[t], (1,0,0)); conf_w[o] = add_wlt(conf_w[o], (0,1,0))
            elif res == "L":
                conf_w[t] = add_wlt(conf_w[t], (0,1,0)); conf_w[o] = add_wlt(conf_w[o], (1,0,0))
            else:
                conf_w[t] = add_wlt(conf_w[t], (0,0,1)); conf_w[o] = add_wlt(conf_w[o], (0,0,1))
        if same_div:
            if res == "W":
                div_w[t] = add_wlt(div_w[t], (1,0,0)); div_w[o] = add_wlt(div_w[o], (0,1,0))
            elif res == "L":
                div_w[t] = add_wlt(div_w[t], (0,1,0)); div_w[o] = add_wlt(div_w[o], (1,0,0))
            else:
                div_w[t] = add_wlt(div_w[t], (0,0,1)); div_w[o] = add_wlt(div_w[o], (0,0,1))

        # SOS list (all opponents, both ways)
        opps[t].append(o)
        opps[o].append(t)

    return h2h, conf_w, div_w, beat, opps

# ------------------- Apply flips to date’s games -------------------

def apply_flips_to_totals(wlt_base, games_today, flips):
    wlt = {k: tuple(v) for k,v in wlt_base.items()}
    res = games_today["result_first"].tolist()
    T   = games_today["team"].tolist()
    O   = games_today["opp"].tolist()
    for i, bit in enumerate(flips):
        r = res[i]
        if r in ("W","L") and bit == 1:
            r = "W" if r == "L" else "L"
        t, o = T[i], O[i]
        if r == "W":
            wlt[t] = add_wlt(wlt.get(t,(0,0,0)), (1,0,0))
            wlt[o] = add_wlt(wlt.get(o,(0,0,0)), (0,1,0))
        elif r == "L":
            wlt[t] = add_wlt(wlt.get(t,(0,0,0)), (0,1,0))
            wlt[o] = add_wlt(wlt.get(o,(0,0,0)), (1,0,0))
        else:
            wlt[t] = add_wlt(wlt.get(t,(0,0,0)), (0,0,1))
            wlt[o] = add_wlt(wlt.get(o,(0,0,0)), (0,0,1))
    return wlt

def apply_flips_to_metrics(base_h2h, base_conf, base_div, base_beat, base_opps,
                           games_today, flips, meta):
    h2h  = {k: tuple(v) for k,v in base_h2h.items()}
    conf = {k: tuple(v) for k,v in base_conf.items()}
    div  = {k: tuple(v) for k,v in base_div.items()}
    beat = {k: list(v) for k,v in base_beat.items()}
    opps = {k: list(v) for k,v in base_opps.items()}

    res = games_today["result_first"].tolist()
    T   = games_today["team"].tolist()
    O   = games_today["opp"].tolist()

    for i, bit in enumerate(flips):
        r = res[i]
        if r in ("W","L") and bit == 1:
            r = "W" if r == "L" else "L"
        t, o = T[i], O[i]
        # H2H
        if r == "W":
            h2h[(t,o)] = add_wlt(h2h.get((t,o),(0,0,0)), (1,0,0))
            h2h[(o,t)] = add_wlt(h2h.get((o,t),(0,0,0)), (0,1,0))
            beat.setdefault(t, []).append(o)
        elif r == "L":
            h2h[(t,o)] = add_wlt(h2h.get((t,o),(0,0,0)), (0,1,0))
            h2h[(o,t)] = add_wlt(h2h.get((o,t),(0,0,0)), (1,0,0))
            beat.setdefault(o, []).append(t)
        else:
            h2h[(t,o)] = add_wlt(h2h.get((t,o),(0,0,0)), (0,0,1))
            h2h[(o,t)] = add_wlt(h2h.get((o,t),(0,0,0)), (0,0,1))
        # conference/division
        if meta.get(t,{}).get("conference","") == meta.get(o,{}).get("conference",""):
            if r == "W":
                conf[t] = add_wlt(conf.get(t,(0,0,0)), (1,0,0))
                conf[o] = add_wlt(conf.get(o,(0,0,0)), (0,1,0))
            elif r == "L":
                conf[t] = add_wlt(conf.get(t,(0,0,0)), (0,1,0))
                conf[o] = add_wlt(conf.get(o,(0,0,0)), (1,0,0))
            else:
                conf[t] = add_wlt(conf.get(t,(0,0,0)), (0,0,1))
                conf[o] = add_wlt(conf.get(o,(0,0,0)), (0,0,1))
        if meta.get(t,{}).get("division","") == meta.get(o,{}).get("division",""):
            if r == "W":
                div[t] = add_wlt(div.get(t,(0,0,0)), (1,0,0))
                div[o] = add_wlt(div.get(o,(0,0,0)), (0,1,0))
            elif r == "L":
                div[t] = add_wlt(div.get(t,(0,0,0)), (0,1,0))
                div[o] = add_wlt(div.get(o,(0,0,0)), (1,0,0))
            else:
                div[t] = add_wlt(div.get(t,(0,0,0)), (0,0,1))
                div[o] = add_wlt(div.get(o,(0,0,0)), (0,0,1))
        # opp lists (for SOS)
        opps.setdefault(t, []).append(o)
        opps.setdefault(o, []).append(t)

    return h2h, conf, div, beat, opps

# -------------------- Strength metrics (SOV/SOS) -------------------

def strength_of_victory(team: str, beat_map: Dict[str,List[str]],
                        wlt_map: Dict[str,Tuple[int,int,int]], season: int) -> float:
    opps = beat_map.get(team, [])
    if not opps:
        return 0.0
    vals = [pct_era(wlt_map.get(o,(0,0,0)), season) for o in opps]
    return float(np.mean(vals)) if vals else 0.0

def strength_of_schedule(team: str, opps_map: Dict[str,List[str]],
                         wlt_map: Dict[str,Tuple[int,int,int]], season: int) -> float:
    opps = opps_map.get(team, [])
    if not opps:
        return 0.0
    vals = [pct_era(wlt_map.get(o,(0,0,0)), season) for o in opps]
    return float(np.mean(vals)) if vals else 0.0

def common_games_record(t: str, u: str,
                        h2h: Dict[Tuple[str,str],Tuple[int,int,int]],
                        opps_map: Dict[str,List[str]],
                        season: int) -> Tuple[int,int,int]:
    """Aggregate WLT vs common opponents (exclude each other)."""
    common = set(opps_map.get(t, [])) & set(opps_map.get(u, []))
    common.discard(t); common.discard(u)
    wlt = (0,0,0)
    for o in common:
        wlt = add_wlt(wlt, h2h.get((t,o),(0,0,0)))
    return wlt

# ----------------- Division tiebreakers (all eras) -----------------

def break_division_tie(candidates: List[str],
                       season: int,
                       meta: Dict[str,Dict[str,str]],
                       wlt: Dict[str,Tuple[int,int,int]],
                       h2h: Dict[Tuple[str,str],Tuple[int,int,int]],
                       div_wlt: Dict[str,Tuple[int,int,int]],
                       conf_wlt: Dict[str,Tuple[int,int,int]],
                       beat_map: Dict[str,List[str]],
                       opps_map: Dict[str,List[str]]) -> Tuple[List[str], Set[str]]:
    """
    Return (ordered list, ambiguous_set). If ambiguous_set non-empty, coin toss would be needed.
    Order is determined using:
    1) H2H among tied, 2) Division record, 3) Common opponents (if applicable),
    4) Conference record, 5) SOV, 6) SOS, 7) Coin toss -> ambiguous.
    """
    if len(candidates) <= 1:
        return candidates[:], set()

    # 1) H2H among tied — mini-league win%
    def mini_h2h_rank(group):
        wins = {t: 0 for t in group}
        games = {t: 0 for t in group}
        for t in group:
            for u in group:
                if t == u: continue
                w,l,ties = h2h.get((t,u),(0,0,0))
                wins[t]  += w + 0.5*ties if season >= 1972 else w  # ties-ignored pre-1972
                games[t] += (w + l + (0 if season < 1972 else ties))
        # compute pct with era rule
        keys = {t: (wins[t] / games[t] if games[t] else -1.0) for t in group}
        best = max(keys.values())
        top  = [t for t in group if abs(keys[t] - best) < 1e-12]
        if len(top) == 1:
            # winner first, then the rest unchanged
            return top + [t for t in group if t not in top], set()
        return group, set()  # unresolved, move to next step

    order, amb = mini_h2h_rank(candidates)
    if order != candidates:
        return order, amb

    # Helper: rank by a metric key, return if unique best exists
    def rank_by_key(group, key_fn, reverse=True):
        keys = {t: key_fn(t) for t in group}
        best = max(keys.values()) if reverse else min(keys.values())
        top  = [t for t in group if abs(keys[t] - best) < 1e-12]
        if len(top) == 1:
            return top + [t for t in group if t not in top], set()
        return group, set()

    # 2) Division record
    order, amb = rank_by_key(
        candidates,
        key_fn=lambda t: pct_era(div_wlt.get(t,(0,0,0)), season),
        reverse=True
    )
    if order != candidates: return order, amb

    # 3) Common opponents (only if applicable: each tied team must have ≥4 common-games)
    def common_applicable(group):
        # For each pair, count total games vs common opponents; require min >= 4
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                t,u = group[i], group[j]
                w,l,ti = common_games_record(t,u,h2h,opps_map,season)
                if (w+l+ (0 if season < 1972 else ti)) < 4:
                    return False
        return True

    if common_applicable(candidates):
        def common_pct(t):
            # Sum over all other tied teams' common sets to approximate a multi-team common record
            WLT = (0,0,0)
            for u in candidates:
                if u == t: continue
                WLT = add_wlt(WLT, common_games_record(t,u,h2h,opps_map,season))
            return pct_era(WLT, season)
        order, amb = rank_by_key(candidates, key_fn=common_pct, reverse=True)
        if order != candidates: return order, amb

    # 4) Conference record
    order, amb = rank_by_key(candidates, key_fn=lambda t: pct_era(conf_wlt.get(t,(0,0,0)), season), reverse=True)
    if order != candidates: return order, amb

    # 5) Strength of victory
    order, amb = rank_by_key(candidates, key_fn=lambda t: strength_of_victory(t, beat_map, wlt, season), reverse=True)
    if order != candidates: return order, amb

    # 6) Strength of schedule
    order, amb = rank_by_key(candidates, key_fn=lambda t: strength_of_schedule(t, opps_map, wlt, season), reverse=True)
    if order != candidates: return order, amb

    # 7) Coin toss — ambiguous
    return order, set(candidates)

# ------------- Wild-card tiebreakers (1970–1977 only) --------------

def wc_pick_70s(cands: List[str], season: int,
                wlt, h2h, conf_wlt) -> Tuple[Optional[str], Set[str]]:
    """Return (winner or None if coin toss), ambiguous_set."""
    if len(cands) == 1:
        return cands[0], set()
    # Mini-league H2H wins
    wins = {t: 0.0 for t in cands}
    games = {t: 0.0 for t in cands}
    for t in cands:
        for u in cands:
            if t == u: continue
            w,l,ti = h2h.get((t,u),(0,0,0))
            if season < 1972:
                wins[t]  += w
                games[t] += w + l
            else:
                wins[t]  += w + 0.5*ti
                games[t] += w + l + ti
    h2h_pct = {t: (wins[t]/games[t] if games[t] else -1.0) for t in cands}
    best = max(h2h_pct.values())
    top  = [t for t in cands if abs(h2h_pct[t] - best) < 1e-12]
    if len(top) == 1:
        return top[0], set()
    # Intra-conference %
    confpct = {t: pct_era(conf_wlt.get(t,(0,0,0)), season) for t in cands}
    bestc = max(confpct.values())
    topc  = [t for t in cands if abs(confpct[t] - bestc) < 1e-12]
    if len(topc) == 1:
        return topc[0], set()
    # Coin toss
    return None, set(cands)

# --------- Wild-card tiebreakers (1978+; multiple WC slots) --------

def break_wc_post77(cands: List[str], season: int,
                    wlt, h2h, conf_wlt, beat_map, opps_map) -> Tuple[List[str], Set[str]]:
    """
    Return ordered list of candidates using:
    H2H sweep → Conference → Common (if applicable) → SOV → SOS → coin toss.
    If coin toss needed, all remaining tied teams are marked ambiguous.
    """
    remaining = cands[:]
    ordered: List[str] = []
    ambiguous: Set[str] = set()

    def common_applicable(group):
        # Require each pair have at least 4 common games
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                t,u = group[i], group[j]
                w,l,ti = common_games_record(t,u,h2h,opps_map,season)
                if (w+l+ (0 if season < 1972 else ti)) < 4:
                    return False
        return True

    while remaining:
        group = remaining[:]

        # H2H sweep (if everyone played everyone) — rank by mini-league H2H%
        played_all = True
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                t,u = group[i], group[j]
                g = sum(h2h.get((t,u),(0,0,0))) + sum(h2h.get((u,t),(0,0,0)))
                if season < 1972:
                    # ties ignored; still require at least one game to use H2H
                    if g == 0:
                        played_all = False
                else:
                    if g == 0:
                        played_all = False
        if played_all:
            wins = {t: 0.0 for t in group}
            games = {t: 0.0 for t in group}
            for t in group:
                for u in group:
                    if t == u: continue
                    w,l,ti = h2h.get((t,u),(0,0,0))
                    if season < 1972:
                        wins[t] += w
                        games[t]+= w+l
                    else:
                        wins[t] += w + 0.5*ti
                        games[t]+= w+l+ti
            vals = {t: (wins[t]/games[t] if games[t] else -1.0) for t in group}
            best = max(vals.values())
            top  = [t for t in group if abs(vals[t]-best) < 1e-12]
            if len(top) == 1:
                winner = top[0]
                ordered.append(winner)
                remaining.remove(winner)
                continue  # pick next slot

        # Conference record
        conf_vals = {t: pct_era(conf_wlt.get(t,(0,0,0)), season) for t in group}
        bestc = max(conf_vals.values())
        topc  = [t for t in group if abs(conf_vals[t]-bestc) < 1e-12]
        if len(topc) == 1:
            winner = topc[0]; ordered.append(winner); remaining.remove(winner); continue

        # Common opponents (if applicable)
        if common_applicable(group):
            def common_pct(t):
                WLT = (0,0,0)
                for u in group:
                    if u == t: continue
                    WLT = add_wlt(WLT, common_games_record(t,u,h2h,opps_map,season))
                return pct_era(WLT, season)
            vals = {t: common_pct(t) for t in group}
            bestv = max(vals.values())
            topv  = [t for t in group if abs(vals[t]-bestv) < 1e-12]
            if len(topv) == 1:
                winner = topv[0]; ordered.append(winner); remaining.remove(winner); continue

        # Strength of victory
        sov_vals = {t: strength_of_victory(t, beat_map, wlt, season) for t in group}
        bestsov = max(sov_vals.values()); topsov = [t for t in group if abs(sov_vals[t]-bestsov) < 1e-12]
        if len(topsov) == 1:
            winner = topsov[0]; ordered.append(winner); remaining.remove(winner); continue

        # Strength of schedule
        sos_vals = {t: strength_of_schedule(t, opps_map, wlt, season) for t in group}
        bestsos = max(sos_vals.values()); topsos = [t for t in group if abs(sos_vals[t]-bestsos) < 1e-12]
        if len(topsos) == 1:
            winner = topsos[0]; ordered.append(winner); remaining.remove(winner); continue

        # Coin toss — all in this group ambiguous; pick deterministic placeholder
        ambiguous.update(group)
        group_sorted = sorted(group)  # stable placeholder
        winner = group_sorted[0]
        ordered.append(winner)
        remaining.remove(winner)

    return ordered, ambiguous

# ----------------- Seeding snapshot (any era) ----------------------

def seed_snapshot(meta, wlt, season,
                  h2h=None, conf_wlt=None, div_wlt=None,
                  beat_map=None, opps_map=None) -> Tuple[Dict[str,int], Set[str]]:
    """
    Returns (seeds, ambiguous_set).
    """
    seeds: Dict[str,int] = {}
    ambiguous: Set[str] = set()
    if is_tournament_skip(season):
        return {}, set()

    slots_wc = wildcard_slots(season)

    for conf in ["AFC","NFC"]:
        conf_teams = [t for t,m in meta.items() if m["conference"] == conf]
        if not conf_teams:
            continue

        # group divisions
        div_groups: Dict[str,List[str]] = defaultdict(list)
        for t in conf_teams:
            div_groups[meta[t]["division"]].append(t)

        # pick division champs using full tiebreakers
        champs: List[str] = []
        champ_amb: Set[str] = set()
        for div, group in div_groups.items():
            # Find best overall% group top
            group_sorted = order_overall_basic(group, wlt, season)
            top_pct = pct_era(wlt.get(group_sorted[0],(0,0,0)), season)
            tied = [t for t in group if abs(pct_era(wlt.get(t,(0,0,0)), season) - top_pct) < 1e-12]
            if len(tied) == 1:
                champs.append(tied[0])
            else:
                # break division tie
                order, amb = break_division_tie(tied, season, meta, wlt, h2h or {}, div_wlt or {}, conf_wlt or {}, beat_map or {}, opps_map or {})
                champs.append(order[0])  # deterministic pick, but…
                champ_amb |= amb

        champ_order = order_overall_basic(champs, wlt, season)

        if season <= 1977:
            # Single WC with special rules
            remain = [t for t in conf_teams if t not in champs]
            if remain:
                remain_sorted = order_overall_basic(remain, wlt, season)
                best_pct = pct_era(wlt.get(remain_sorted[0],(0,0,0)), season)
                tied = [t for t in remain if abs(pct_era(wlt.get(t,(0,0,0)), season) - best_pct) < 1e-12]
                if len(tied) == 1:
                    wc_team = tied[0]
                else:
                    wc_team, amb = wc_pick_70s(tied, season, wlt, h2h or {}, conf_wlt or {})
                    if wc_team is None:
                        ambiguous |= amb  # coin toss needed
                        # deterministic placeholder to continue seeding:
                        wc_team = sorted(tied)[0]
                # assign seeds
                i = 1
                for t in champ_order:
                    seeds[t] = i; i += 1
                seeds[wc_team] = i
        else:
            # Post-1977: multiple WCs with post-77 tiebreakers
            remain = [t for t in conf_teams if t not in champs]
            if remain:
                wc_rank, amb = break_wc_post77(remain, season, wlt, h2h or {}, conf_wlt or {}, beat_map or {}, opps_map or {})
                amb = amb  # coin toss exposure, mark as ambiguous
                # trim to WC slots
                wc_rank = wc_rank[:slots_wc]
                # assign seeds
                i = 1
                for t in champ_order:
                    seeds[t] = i; i += 1
                for t in wc_rank:
                    seeds[t] = i; i += 1
                ambiguous |= amb

        ambiguous |= champ_amb

    return seeds, ambiguous

# ---------------------------- Main --------------------------------

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", dest="outfile", default="Seed_Locked_Teams.csv")
    parser.add_argument("--verbose", action="store_true", help="Extra debug output")
    parser.add_argument("--rand_probe", type=int, default=2000,
                        help="Randomized probe iterations when remaining games > 18 (default: 2000)")
    args = parser.parse_args()

    infile = Path("Playoff_Flag_Dataset.csv")
    log(f"Looking for input file: {infile.resolve()}")
    if not infile.exists():
        raise FileNotFoundError(f"Could not find {infile.resolve()}")

    usecols = [
        "season","date","team","opp","conference","division","playoff",
        "wins_before","losses_before","ties_before",
        "wins_after","losses_after","ties_after",
        "url","result","home_or_away"
    ]
    log("Loading CSV...")
    df = pd.read_csv(infile, usecols=usecols)
    log(f"Loaded {len(df):,} rows.")

    # Regular season only
    df["playoff"] = to_bool_series(df["playoff"])
    pre = len(df)
    df = df[~df["playoff"]].copy()
    log(f"Dropped playoff rows: {pre - len(df):,}. Regular-season rows: {len(df):,}")

    # Dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before_drop = len(df)
    df = df.dropna(subset=["date"])
    if len(df) != before_drop:
        log(f"Dropped {before_drop - len(df)} rows with invalid dates.")

    # Result letter
    df["result_first"] = df["result"].astype(str).str.upper().str[0].fillna("")

    # Validate game pairs
    url_counts = df.groupby("url")["team"].nunique()
    good_urls = url_counts[url_counts == 2].index
    before_pair = len(df)
    df = df[df["url"].isin(good_urls)].copy()
    if len(df) != before_pair:
        log(f"Dropped {before_pair - len(df)} rows with malformed games (non-2-team urls).")

    seasons = sorted(df["season"].unique().tolist())
    log(f"Seasons to process: {len(seasons)} -> {seasons[:5]}{'...' if len(seasons) > 5 else ''}")

    out_rows = []
    total_locked = 0

    for season in seasons:
        season = int(season)
        if is_tournament_skip(season):
            log(f"\nSeason {season}: special tournament year — skipping.")
            continue

        g_season = df[df["season"] == season].copy()
        meta = get_team_meta(g_season)

        # Precompute by date slices for the season
        by_date = {d: h for d, h in g_season.groupby("date")}
        final_dates = g_season.groupby("team")["date"].max()
        teams = final_dates.index.tolist()

        log(f"\nSeason {season}: {len(teams)} teams. Unique dates: {len(by_date)}.")

        # Convenience lists
        conf_teams_map = {
            "AFC": [t for t,m in meta.items() if m["conference"] == "AFC"],
            "NFC": [t for t,m in meta.items() if m["conference"] == "NFC"],
        }

        for idx, (team, final_date) in enumerate(final_dates.items(), 1):
            target_conf = meta[team]["conference"]
            log(f"[{season}] {idx}/{len(teams)}: {team} final date {final_date.date()}", enable=True)

            # Baselines
            base_wlt = build_baseline_totals(g_season, final_date)
            base_h2h, base_conf, base_div, base_beat, base_opps = build_baseline_metrics(g_season, meta, final_date)

            # Games on date (dedup)
            games_today = dedupe_games(by_date.get(final_date, pd.DataFrame()).reset_index(drop=True))

            # Only simulate games that touch the target conference
            if not games_today.empty:
                def row_touches_conf(row):
                    t, o = row["team"], row["opp"]
                    return (meta.get(t,{}).get("conference","") == target_conf) or \
                           (meta.get(o,{}).get("conference","") == target_conf)
                games_today = games_today[games_today.apply(row_touches_conf, axis=1)].reset_index(drop=True)

            # Relevance pruning
            if not games_today.empty:
                conf_teams = conf_teams_map[target_conf]
                # Div buckets
                by_div = defaultdict(list)
                for t2 in conf_teams:
                    by_div[meta[t2]["division"]].append(t2)

                def conf_rank(wlt_map):
                    return sorted(conf_teams,
                                  key=lambda t2: (round(pct_era(wlt_map.get(t2,(0,0,0)), season),6),
                                                  wlt_map.get(t2,(0,0,0))[0]),
                                  reverse=True)
                sorted_conf = conf_rank(base_wlt)
                slots_wc = wildcard_slots(season)
                num_div = len([d for d in by_div if by_div[d]])
                playoff_span = set(sorted_conf[:max(0, num_div + slots_wc + 2)])

                target_w = base_wlt.get(team,(0,0,0))
                relevant: Set[str] = set()

                for t2 in conf_teams:
                    if within_one_game(base_wlt.get(t2,(0,0,0)), target_w, season):
                        relevant.add(t2)
                relevant |= playoff_span

                my_div = meta[team]["division"]
                if by_div[my_div]:
                    leader = max(by_div[my_div],
                                 key=lambda x: (pct_era(base_wlt.get(x,(0,0,0)), season),
                                                base_wlt.get(x,(0,0,0))[0]))
                    for t2 in by_div[my_div]:
                        if within_one_game(base_wlt.get(t2,(0,0,0)), base_wlt.get(leader,(0,0,0)), season):
                            relevant.add(t2)

                def game_is_relevant(row):
                    t2, o2 = row["team"], row["opp"]
                    return (t2 in relevant) or (o2 in relevant)

                before = len(games_today)
                games_today = games_today[games_today.apply(game_is_relevant, axis=1)].reset_index(drop=True)
                if args.verbose and before != len(games_today):
                    log(f"  Pruned {before - len(games_today)} games as irrelevant.", enable=True)

            n = len(games_today)
            if n == 0:
                seeds, ambiguous = seed_snapshot(meta, base_wlt, season,
                                                 base_h2h, base_conf, base_div, base_beat, base_opps)
                if team in seeds and (team not in ambiguous):
                    out_rows.append({
                        "season": season,
                        "team": team,
                        "conference": meta[team]["conference"],
                        "division": meta[team]["division"],
                        "final_date": final_date.date().isoformat(),
                        "seed": seeds[team]
                    })
                    total_locked += 1
                    log(f"  -> Locked (no relevant games). Seed {seeds[team]}")
                else:
                    if team in ambiguous:
                        log("  -> Not locked (coin toss needed in path).")
                    else:
                        log("  -> Not locked (not in field or seed ambiguous).")
                continue

            log(f"  Simulating date {final_date.date()}: {n} relevant game(s) → up to 2^{n} outcomes")

            # Randomized probe if large
            seed_variance = False
            ambiguous_seen = False
            seed_seen: Set[int] = set()
            HARD_ENUM_CAP = 18

            if n > HARD_ENUM_CAP and args.rand_probe > 0:
                for _ in range(args.rand_probe):
                    flips = tuple(random.getrandbits(1) for _ in range(n))
                    wlt_state = apply_flips_to_totals(base_wlt, games_today, flips)
                    h2h_s, conf_s, div_s, beat_s, opps_s = apply_flips_to_metrics(
                        base_h2h, base_conf, base_div, base_beat, base_opps, games_today, flips, meta
                    )
                    seeds, ambiguous = seed_snapshot(meta, wlt_state, season,
                                                     h2h_s, conf_s, div_s, beat_s, opps_s)
                    if team in ambiguous:
                        ambiguous_seen = True
                        break
                    seed_seen.add(seeds.get(team, -1))
                    if len(seed_seen) > 1:
                        seed_variance = True
                        break

                if args.verbose:
                    log(f"  Probe: seeds seen={sorted(seed_seen)} ambiguous={ambiguous_seen}", enable=True)

            if ambiguous_seen:
                log("  -> Not locked (coin toss possible at boundary).")
                continue
            if seed_variance:
                log("  -> Not locked (seed varies across scenarios — probe).")
                continue

            if n > HARD_ENUM_CAP:
                if len(seed_seen) == 1 and next(iter(seed_seen)) != -1:
                    seed_val = next(iter(seed_seen))
                    out_rows.append({
                        "season": season,
                        "team": team,
                        "conference": meta[team]["conference"],
                        "division": meta[team]["division"],
                        "final_date": final_date.date().isoformat(),
                        "seed": seed_val
                    })
                    total_locked += 1
                    log(f"  -> Locked at seed {seed_val} (probe).")
                else:
                    log("  -> Not locked (probe inconclusive).")
                continue

            # Full enumeration
            seed_seen.clear()
            ambiguous_seen = False
            early_broke = False
            for flips in itertools.product((0,1), repeat=n):
                wlt_state = apply_flips_to_totals(base_wlt, games_today, flips)
                h2h_s, conf_s, div_s, beat_s, opps_s = apply_flips_to_metrics(
                    base_h2h, base_conf, base_div, base_beat, base_opps, games_today, flips, meta
                )
                seeds, ambiguous = seed_snapshot(meta, wlt_state, season,
                                                 h2h_s, conf_s, div_s, beat_s, opps_s)

                if team in ambiguous:
                    ambiguous_seen = True
                    if args.verbose:
                        log("    Early-exit: coin-toss contingency → NOT locked.", enable=True)
                    early_broke = True
                    break

                seed_seen.add(seeds.get(team, -1))
                if len(seed_seen) > 1:
                    if args.verbose:
                        log("    Early-exit: seed varies → NOT locked.", enable=True)
                    early_broke = True
                    break

            if not early_broke and len(seed_seen) == 1 and next(iter(seed_seen)) != -1:
                seed_val = next(iter(seed_seen))
                out_rows.append({
                    "season": season,
                    "team": team,
                    "conference": meta[team]["conference"],
                    "division": meta[team]["division"],
                    "final_date": final_date.date().isoformat(),
                    "seed": seed_val
                })
                total_locked += 1
                log(f"  -> Locked at seed {seed_val}")
            elif not early_broke:
                if ambiguous_seen:
                    log("  -> Not locked (coin toss needed).")
                else:
                    log("  -> Not locked (team could miss or seed can vary).")

    out = pd.DataFrame(out_rows, columns=["season","team","conference","division","final_date","seed"])
    out = out.sort_values(["season","conference","seed","team"])
    out.to_csv(args.outfile, index=False)

    elapsed = time.time() - t0
    log(f"\nWrote {len(out)} locked seeds -> {args.outfile}")
    log(f"Total locked seeds: {total_locked}")
    log(f"Done in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()
