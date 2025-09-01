"""
PFR scraper — TWO ROWS PER URL (Away then Home) with rate limiting + 429 backoff

Outputs EXACT columns (in this order):
url,team,home_or_away,date_text,quarter_1_score,quarter_2_score,quarter_3_score,quarter_4_score,
overtime_score,new_final_score,season,date,neutral,playoff,opponent_score,margin,result,
roof,surface,attendance,stadium

Usage example:
  pip install pandas requests beautifulsoup4 lxml
  python scraper.py \
    --in "NFL _missing_ URLs - Sheet1.csv" \
    --out scraped_games.csv \
    --workers 6 \
    --rps 1 \
    --render 0
"""

import argparse
import sys
import time
import re
import random
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ====== REQUIRED: your ScraperAPI key ======
SCRAPERAPI_KEY = "d23e6ab187f839bf309c8e6f53754027"

BASE_SCRAPERAPI = "http://api.scraperapi.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; pfr-scraper/1.0)"}

COLS = [
    "url","team","home_or_away","date_text","quarter_1_score","quarter_2_score",
    "quarter_3_score","quarter_4_score","overtime_score","new_final_score","season","date",
    "neutral","playoff","opponent_score","margin","result","roof","surface","attendance","stadium"
]

# ====== HTTP session (connection reuse) ======
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)

# ====== Global rate limiter (token bucket lite) ======
_last_request_time = 0.0
_rate_lock = Lock()

def _respect_rps(rps: float):
    """Ensure we don't exceed requests-per-second across all threads."""
    global _last_request_time
    if rps <= 0:
        return
    min_interval = 1.0 / rps
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_time = time.time()

def scraperapi_get(url: str, inflight: Semaphore, rps: float, timeout: int, max_retries: int, render: bool) -> Optional[str]:
    """
    Fetch URL through ScraperAPI with:
      - global RPS limiter
      - inflight concurrency cap
      - 429-aware exponential backoff (honors Retry-After when present)
    Returns HTML text or None on failure.
    """
    params = {"api_key": SCRAPERAPI_KEY, "url": url}
    if render:
        params["render"] = "true"

    backoff = 1.25  # base for exponential backoff
    for attempt in range(max_retries + 1):
        try:
            _respect_rps(rps)
            inflight.acquire()
            try:
                resp = SESSION.get(BASE_SCRAPERAPI, params=params, headers=HEADERS, timeout=timeout)
            finally:
                inflight.release()

            # Handle 2xx
            if 200 <= resp.status_code < 300 and resp.text:
                return resp.text

            # Rate limited
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        sleep_s = max(1.0, float(ra))
                    except:
                        sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
                else:
                    sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
                print(f"[429] Backing off {sleep_s:.2f}s (attempt {attempt+1}/{max_retries}) for {url}", file=sys.stderr)
                time.sleep(sleep_s)
                continue

            # Other transient server errors: 5xx -> backoff
            if 500 <= resp.status_code < 600:
                sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
                print(f"[{resp.status_code}] Retrying in {sleep_s:.2f}s (attempt {attempt+1}/{max_retries}) for {url}", file=sys.stderr)
                time.sleep(sleep_s)
                continue

            # Non-retriable (4xx other than 429)
            print(f"[{resp.status_code}] {resp.text[:200]} ... for {url}", file=sys.stderr)
            return None

        except requests.RequestException as e:
            sleep_s = (backoff ** attempt) + random.uniform(0, 0.5)
            print(f"[EXC] {e}. Retrying in {sleep_s:.2f}s (attempt {attempt+1}/{max_retries}) for {url}", file=sys.stderr)
            time.sleep(sleep_s)

    print(f"[FAIL] Exhausted retries for {url}", file=sys.stderr)
    return None

# ====== Helpers to extract tables (normal DOM or comment-wrapped) ======
def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")

def get_table_html_by_id(html: str, table_id: str) -> Optional[str]:
    """Return HTML for a table with given id, from DOM or within comments."""
    soup = _soup(html)
    node = soup.select_one(f"table#{table_id}")
    if node is not None:
        return str(node)
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if f'id="{table_id}"' in c or f"id='{table_id}'" in c:
            try:
                inner = BeautifulSoup(c, "lxml").select_one(f"table#{table_id}")
                if inner is not None:
                    return str(inner)
            except Exception:
                pass
    return None

def get_linescore_table_node(html: str) -> Optional[BeautifulSoup]:
    """Return a BeautifulSoup node for table.linescore (DOM or comments)."""
    soup = _soup(html)
    node = soup.select_one("table.linescore")
    if node is not None:
        return node
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if 'class="linescore' in c or "class='linescore" in c:
            try:
                inner = BeautifulSoup(c, "lxml").select_one("table.linescore")
                if inner is not None:
                    return inner
            except Exception:
                pass
    return None

# ====== Parsing blocks ======
def parse_game_info_table(html: str) -> Dict[str, str]:
    """Parse stadium, attendance, roof, surface, neutral, playoff, date_text."""
    t_html = get_table_html_by_id(html, "game_info")
    if not t_html:
        return {
            "stadium": "", "attendance": "", "roof": "", "surface": "",
            "neutral": "FALSE", "playoff": "", "date_text": ""
        }
    try:
        df = pd.read_html(t_html)[0]
    except Exception:
        return {
            "stadium": "", "attendance": "", "roof": "", "surface": "",
            "neutral": "FALSE", "playoff": "", "date_text": ""
        }

    info: Dict[str, str] = {}
    for _, row in df.iterrows():
        if len(row) < 2:
            continue
        key = str(row[0]).strip().lower()
        val = str(row[1]).strip()
        info[key] = val

    neutral_value = (info.get("neutral") or info.get("neutral field") or "").strip().lower()
    playoff_terms = ("wild card", "divisional", "conference", "super bowl")

    return {
        "stadium": info.get("stadium") or info.get("venue") or "",
        "attendance": re.sub(r"[^\d]", "", info.get("attendance", "")) if info.get("attendance") else "",
        "roof": info.get("roof", ""),
        "surface": info.get("surface", ""),
        "date_text": info.get("date", ""),
        "neutral": "TRUE" if neutral_value in ("yes", "true", "1") else "FALSE",
        "playoff": "TRUE" if any(t in str(info.get("week","")).lower() for t in playoff_terms) else "FALSE",
    }

def _parse_ordinal_date(txt: str) -> Optional[datetime]:
    """Parse 'October 10th, 1920' / 'Oct 10, 1920' / 'Sun Oct 10, 1920'."""
    if not txt:
        return None
    t = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', txt.strip(), flags=re.IGNORECASE)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%A %b %d, %Y", "%a %b %d, %Y"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    return None

def parse_scorebox_meta_and_title(html: str) -> Dict[str, str]:
    """Pull date/season/date_text and (fallback) stadium from scorebox_meta/title."""
    out = {"date": "", "season": "", "date_text": "", "stadium": ""}
    soup = _soup(html)

    meta = soup.select_one(".scorebox_meta")
    if meta:
        meta_texts = [getattr(ch, "get_text", lambda **_: str(ch))(" ", strip=True) for ch in meta.children]
        meta_texts = [t for t in meta_texts if t]
        date_line = next((t for t in meta_texts if re.search(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\b', t)), "")
        if date_line:
            dt = _parse_ordinal_date(date_line)
            if dt:
                out["date"] = dt.date().isoformat()
                out["season"] = str(dt.year)
                out["date_text"] = date_line

        for div in meta.select("div"):
            txt = div.get_text(" ", strip=True).lower()
            if txt.startswith("stadium"):
                a = div.find("a")
                out["stadium"] = a.get_text(strip=True) if a else div.get_text(" ", strip=True).split(":", 1)[-1].strip()
                break

    if not out["date"]:
        title = soup.title.get_text(strip=True) if soup.title else ""
        m = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}', title)
        if not m:
            h1 = soup.find("h1")
            if h1:
                t = h1.get_text(" ", strip=True)
                m = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}', t)
        if m:
            dt = _parse_ordinal_date(m.group(0))
            if dt:
                out["date"] = dt.date().isoformat()
                out["season"] = str(dt.year)
                out["date_text"] = m.group(0)
    return out

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    for nm in names:
        if nm in df.columns:
            return nm
    return None

def _extract_team_names_from_linescore(html: str) -> List[str]:
    """
    Extract team names directly from the second <td> in each linescore row,
    ignoring the first logo column. Returns [away_name, home_name].
    """
    node = get_linescore_table_node(html)
    if node is None:
        return []
    teams: List[str] = []
    tbody = node.find("tbody")
    if not tbody:
        return []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        # logo td is index 0, team link is index 1
        if len(tds) >= 2:
            teams.append(tds[1].get_text(strip=True))
    # Expect two rows: away then home
    return teams[:2]

def parse_linescore_both_rows(html: str) -> List[Dict[str, str]]:
    """
    Return TWO dicts (away then home) with:
      team, home_or_away, q1..q4, new_final_score, opponent_score
    Ignores overtime entirely (overtime_score = "").
    Team names are pulled from the <td><a>Team</a></td> cells to avoid NaN/logo columns.
    """
    rows: List[Dict[str, str]] = []

    node = get_linescore_table_node(html)
    if node is None:
        return rows

    # Use pandas for the numeric columns (quarters/totals)
    try:
        df = pd.read_html(str(node))[0]
    except Exception:
        return rows

    df = _normalize_headers(df)
    col_q1 = _find_col(df, "1", "1st", "q1")
    col_q2 = _find_col(df, "2", "2nd", "q2")
    col_q3 = _find_col(df, "3", "3rd", "q3")
    col_q4 = _find_col(df, "4", "4th", "q4")
    col_tot = _find_col(df, "final", "t", "total", "pts")

    if len(df.index) < 2:
        return rows

    away_row = df.iloc[0]
    home_row = df.iloc[1]

    # Get team names directly from HTML second <td> for each row
    team_names = _extract_team_names_from_linescore(html)
    away_name = team_names[0] if len(team_names) > 0 else ""
    home_name = team_names[1] if len(team_names) > 1 else ""

    def cell(row, colname: Optional[str]) -> str:
        if not colname or colname not in row.index:
            return ""
        v = row[colname]
        s = "" if pd.isna(v) else str(v).strip()
        if re.fullmatch(r"-?\d+", s):
            return s
        try:
            return str(int(float(s)))
        except Exception:
            return ""

    def build_row(team_name: str, row, opp_row, hoa: str) -> Dict[str, str]:
        return {
            "team": team_name,
            "home_or_away": hoa,
            "quarter_1_score": cell(row, col_q1),
            "quarter_2_score": cell(row, col_q2),
            "quarter_3_score": cell(row, col_q3),
            "quarter_4_score": cell(row, col_q4),
            "overtime_score": "",
            "new_final_score": cell(row, col_tot),
            "opponent_score": cell(opp_row, col_tot),
        }

    rows.append(build_row(away_name, away_row, home_row, "away"))
    rows.append(build_row(home_name, home_row, away_row, "home"))
    return rows

def compute_margin_result(team_pts: str, opp_pts: str) -> Tuple[str, str]:
    try:
        t, o = int(team_pts), int(opp_pts)
        margin = t - o
        return (str(margin), ("W" if margin > 0 else "L" if margin < 0 else "T"))
    except Exception:
        return ("", "")

def determine_playoff_if_missing(html: str, existing_playoff: str) -> str:
    if existing_playoff in ("TRUE", "FALSE"):
        return existing_playoff
    soup = _soup(html)
    h1 = soup.find("h1")
    txt = (h1.get_text(" ", strip=True).lower() if h1 else "")
    playoff_terms = ("wild card", "divisional", "conference", "super bowl", "playoffs")
    return "TRUE" if any(t in txt for t in playoff_terms) else "FALSE"

# ====== Worker ======
def process_url(url: str, inflight: Semaphore, rps: float, timeout: int, max_retries: int, render: bool) -> List[Dict[str, Any]]:
    html = scraperapi_get(url, inflight=inflight, rps=rps, timeout=timeout, max_retries=max_retries, render=render)
    if not html:
        return []

    # Common game-level fields
    game_info = parse_game_info_table(html)
    meta = parse_scorebox_meta_and_title(html)

    date_text = game_info.get("date_text") or meta.get("date_text") or ""
    date_iso = meta.get("date", "")
    season = meta.get("season", "")
    playoff = determine_playoff_if_missing(html, game_info.get("playoff", ""))
    neutral = game_info.get("neutral", "FALSE")
    stadium = game_info.get("stadium", "") or meta.get("stadium", "")
    attendance = game_info.get("attendance", "")
    roof = game_info.get("roof", "")
    surface = game_info.get("surface", "")

    team_rows = parse_linescore_both_rows(html)
    out_rows: List[Dict[str, Any]] = []

    for tr in team_rows:
        margin, result = compute_margin_result(tr.get("new_final_score", ""), tr.get("opponent_score", ""))
        rec = {
            "url": url,
            "team": tr.get("team", ""),
            "home_or_away": tr.get("home_or_away", ""),
            "date_text": date_text,
            "quarter_1_score": tr.get("quarter_1_score", ""),
            "quarter_2_score": tr.get("quarter_2_score", ""),
            "quarter_3_score": tr.get("quarter_3_score", ""),
            "quarter_4_score": tr.get("quarter_4_score", ""),
            "overtime_score": "",  # ignored
            "new_final_score": tr.get("new_final_score", ""),
            "season": season,
            "date": date_iso,
            "neutral": neutral,
            "playoff": playoff,
            "opponent_score": tr.get("opponent_score", ""),
            "margin": margin,
            "result": result,
            "roof": roof,
            "surface": surface,
            "attendance": attendance,
            "stadium": stadium,
        }
        out_rows.append(rec)

    return out_rows

# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV with a 'url' column")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path")
    ap.add_argument("--workers", type=int, default=6, help="Number of parsing threads")
    ap.add_argument("--max-inflight", type=int, default=12, help="Max simultaneous HTTP requests")
    ap.add_argument("--rps", type=float, default=1.0, help="Global requests per second to ScraperAPI (float)")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--max-retries", type=int, default=6, help="Max retries per URL (429/5xx backoff)")
    ap.add_argument("--render", type=int, default=0, help="1 to use ScraperAPI render=true, else 0")
    args = ap.parse_args()

    try:
        df_in = pd.read_csv(args.in_csv)
    except Exception as e:
        print(f"ERROR reading input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "url" not in df_in.columns:
        print("ERROR: Input CSV must contain a 'url' column.", file=sys.stderr)
        sys.exit(1)

    urls: List[str] = [str(u).strip() for u in df_in["url"].dropna().tolist() if str(u).strip()]
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    urls = deduped

    inflight = Semaphore(args.max_inflight)
    results: List[Dict[str, Any]] = []

    print(
        f"Scraping {len(urls)} URLs with workers={args.workers}, max_inflight={args.max_inflight}, rps={args.rps}, render={bool(args.render)}",
        file=sys.stderr,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_url = {
            ex.submit(process_url, u, inflight, args.rps, args.timeout, args.max_retries, bool(args.render)): u
            for u in urls
        }
        completed = 0
        for fut in as_completed(future_to_url):
            u = future_to_url[fut]
            try:
                recs = fut.result() or []
                results.extend(recs)
            except Exception as e:
                print(f"!! Error processing {u}: {e}", file=sys.stderr)
            completed += 1
            if completed % 10 == 0 or completed == len(urls):
                print(f"  {completed}/{len(urls)} done", file=sys.stderr)

    if not results:
        pd.DataFrame(columns=COLS).to_csv(args.out_csv, index=False)
        print("No rows scraped.", file=sys.stderr)
        return

    df_out = pd.DataFrame(results).reindex(columns=COLS)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df_out)} rows to {args.out_csv}", file=sys.stderr)

if __name__ == "__main__":
    main()
