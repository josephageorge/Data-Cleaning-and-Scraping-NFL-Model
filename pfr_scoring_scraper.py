# scrape_pfr_scoring_1998plus_by_date_fast.py
# Scrapes the Pro-Football-Reference "Scoring" table for all box scores
# listed in your local CSV, filtering to NFL seasons >= 1998 using date_text.
#
# NFL season year logic:
#   - Aug–Dec => season = game year
#   - Jan–Jul => season = game year - 1
#
# Output is resumable: skips already-scraped game_ids if output exists.
# Optimized with ThreadPoolExecutor + retry/backoff + global rate limit.

import os
import csv
import sys
import time
import random
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

# ------------------ CONFIG ------------------
SHEET_PATH   = "Grouped_by_Game_URL__Away_then_Home_.csv"
OUTPUT_PATH  = "pfr_scoring_1998_plus.csv"

SCRAPERAPI_KEY = "d23e6ab187f839bf309c8e6f53754027"

WORKERS = 20       # concurrent threads
MAX_RPS = 12       # max global requests per second
TIMEOUT = 40       # per-request timeout (sec)
# ---------------------------------------------

SCRAPERAPI_ENDPOINT = "https://api.scraperapi.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# --- global rate limiter ---
_last_req_time = 0.0
_rate_lock = Lock()
def rate_limit():
    global _last_req_time
    if MAX_RPS <= 0:
        return
    with _rate_lock:
        now = time.time()
        min_interval = 1.0 / MAX_RPS
        wait = _last_req_time + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last_req_time = time.time()

# --- HTTP helpers ---
def build_scraperapi_url(target_url: str) -> str:
    params = {"api_key": SCRAPERAPI_KEY, "url": target_url}
    qs = "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
    return f"{SCRAPERAPI_ENDPOINT}?{qs}"

def fetch_html(url: str, session: requests.Session, max_retries: int=6) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            rate_limit()
            resp = session.get(build_scraperapi_url(url), headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200 and resp.text:
                return resp.text
            if resp.status_code in (403, 409, 425, 429, 500, 502, 503, 504):
                sleep = 0.6 * (2 ** (attempt - 1)) + random.uniform(0, 0.9)
                time.sleep(min(sleep, 15.0))
                continue
            resp.raise_for_status()
        except Exception as e:
            last_err = e
            time.sleep(min(0.6 * attempt + random.random() * 0.5, 8.0))
            continue
    raise RuntimeError(f"Failed after {max_retries} attempts: {url} :: {last_err}")

# --- parsing helpers ---
def clean_text(node) -> str:
    if node is None:
        return ""
    txt = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node)
    return " ".join(txt.split())

def find_scoring_table(soup: BeautifulSoup):
    container = soup.find(id="all_scoring")
    if not container:
        return soup.find("table", id="scoring")
    table = container.find("table", id="scoring")
    if table:
        return table
    for c in container.find_all(string=lambda t: isinstance(t, Comment)):
        csoup = BeautifulSoup(c, "lxml")
        table = csoup.find("table", id="scoring")
        if table:
            return table
    return None

def parse_scoring_rows(html: str, game_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "lxml")
    table = find_scoring_table(soup)
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []

    gid = game_url.split("/")[-1].replace(".htm", "")
    gdate = gid[:8]
    home_code = gid[8:]

    rows: List[Dict] = []
    last_quarter = None
    for tr in tbody.find_all("tr"):
        th_q   = tr.find("th", {"data-stat": "quarter"})
        t_time = tr.find("td", {"data-stat": "time"})
        t_team = tr.find("td", {"data-stat": "team"})
        t_desc = tr.find("td", {"data-stat": "description"})
        t_vis  = tr.find("td", {"data-stat": "vis_team_score"})
        t_home = tr.find("td", {"data-stat": "home_team_score"})

        quarter = clean_text(th_q)
        if quarter == "":
            quarter = last_quarter
        else:
            last_quarter = quarter

        rows.append({
            "game_id": gid,
            "game_date": f"{gdate[0:4]}-{gdate[4:6]}-{gdate[6:8]}",
            "home_code": home_code,
            "game_url": game_url,
            "quarter": quarter,
            "time": clean_text(t_time),
            "team_text": clean_text(t_team),
            "description": clean_text(t_desc),
            "vis_team_score": clean_text(t_vis),
            "home_team_score": clean_text(t_home),
        })
    return rows

# --- output helpers ---
def read_done_game_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    try:
        done = set()
        for chunk in pd.read_csv(output_path, usecols=["game_id"], chunksize=50000):
            done.update(chunk["game_id"].dropna().astype(str))
        return done
    except Exception:
        return set()

def ensure_header(output_path: str, fieldnames: List[str]):
    need_header = True
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        need_header = False
    if need_header:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def append_rows(output_path: str, rows: List[Dict], fieldnames: List[str], lock: Lock):
    if not rows:
        return
    with lock:
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

# --- worker ---
def worker(url: str) -> Tuple[str, List[Dict], Optional[str]]:
    session = requests.Session()
    try:
        html = fetch_html(url, session)
        rows = parse_scoring_rows(html, url)
        gid = rows[0]["game_id"] if rows else url.split("/")[-1].replace(".htm", "")
        return gid, rows, None
    except Exception as e:
        return "", [], f"{url} :: {e}"

def main():
    try:
        df = pd.read_csv(SHEET_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Could not find: {SHEET_PATH}")
        sys.exit(1)

    if "url" not in df.columns or "date_text" not in df.columns:
        print("[ERROR] CSV must have 'url' and 'date_text' columns.")
        sys.exit(1)

    # Parse season year from date_text
    df["date"] = pd.to_datetime(df["date_text"])
    df["season_year"] = df["date"].apply(lambda d: d.year if d.month >= 8 else d.year - 1)
    df = df[df["season_year"] >= 1998]

    urls = df["url"].drop_duplicates().tolist()
    print(f"[INFO] Found {len(urls)} unique URLs for NFL seasons 1998+")

    # Resume support
    done_ids = read_done_game_ids(OUTPUT_PATH)
    to_process = []
    for u in urls:
        gid = u.split("/")[-1].replace(".htm", "")
        if gid not in done_ids:
            to_process.append(u)

    print(f"[INFO] Already scraped: {len(done_ids)} | To scrape: {len(to_process)}")

    if not to_process:
        print("[OK] Nothing to do.")
        return

    fieldnames = [
        "game_id","game_date","home_code","game_url",
        "quarter","time","team_text","description",
        "vis_team_score","home_team_score",
    ]
    ensure_header(OUTPUT_PATH, fieldnames)

    processed = 0
    errors = 0
    write_lock = Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(worker, u) for u in to_process]
        for i, fut in enumerate(as_completed(futures), 1):
            gid, rows, err = fut.result()
            if err:
                errors += 1
                print(f"[WARN] {err}")
            else:
                append_rows(OUTPUT_PATH, rows, fieldnames, write_lock)
                processed += 1
            if i % 50 == 0 or i == len(to_process):
                print(f"[INFO] Progress: {i}/{len(to_process)} | Done: {processed} | Errors: {errors}")

    print(f"[DONE] Written {processed} games | Errors: {errors} | Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
