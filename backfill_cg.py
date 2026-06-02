"""
Historical backfill for Color Game (RG3M).
Fetches all available history pages and inserts into cg.db.
Usage: python backfill_cg.py
"""
import os, sys, json, time, ssl, sqlite3, csv, urllib.request, urllib.error
sys.stdout.reconfigure(encoding='utf-8')

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

def _load_env():
    env = {}
    path = os.path.join(_BASE, '.env')
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env       = _load_env()
BASE_URL   = _env.get('BASE_URL',   'https://m.fxpro1.net')
GAME2_NAME = _env.get('GAME2_NAME', 'RG3M')
GAME2_URL  = _env.get('GAME2_URL',  f'{BASE_URL}/openHistory?gameName={GAME2_NAME}')
API_URL    = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
DB_PATH    = os.path.join(_DATA_DIR, _env.get('DB_FILE2', 'cg.db'))
CSV_PATH   = os.path.join(_DATA_DIR, 'cg.csv')
CHROME     = _env.get('CHROME', '')
PAGE_SIZE  = 100


# ── Color helpers ──────────────────────────────────────────────────────────────

def number_to_color(n):
    return ('red' if n % 2 == 0 else 'green'), (1 if n in (0, 5) else 0)


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn

def ensure_table():
    with get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS rounds (
                id         INTEGER PRIMARY KEY,
                number     INTEGER NOT NULL,
                color      TEXT NOT NULL,
                is_purple  INTEGER NOT NULL DEFAULT 0,
                result     TEXT DEFAULT "Not Predicted",
                pred_color TEXT DEFAULT "",
                confidence REAL DEFAULT 0,
                bet        INTEGER DEFAULT 0
            )
        ''')

def existing_ids():
    with get_conn() as conn:
        rows = conn.execute('SELECT id FROM rounds').fetchall()
    return {r[0] for r in rows}

def save_batch(records):
    """records: list of (id, number, color, is_purple)"""
    with get_conn() as conn:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds (id, number, color, is_purple) VALUES (?,?,?,?)',
            records
        )

def sync_csv():
    try:
        with get_conn() as conn:
            rows = conn.execute(
                'SELECT id,number,color,is_purple,result,pred_color,confidence,bet FROM rounds ORDER BY id'
            ).fetchall()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','number','color','is_purple','result','pred_color','confidence','bet'])
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
        print(f"  CSV synced: {len(rows)} rows → cg.csv")
    except Exception as e:
        print(f"  [CSV] {e}")


# ── Parse helpers ──────────────────────────────────────────────────────────────

def parse_value(value_str):
    """Extract single digit 0-9 from API value field."""
    s = str(value_str).strip()
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None

def parse_records(data):
    records = []
    for item in data.get('data', {}).get('records', []):
        issue = item.get('issue')
        val   = parse_value(item.get('value', ''))
        if issue is not None and val is not None:
            records.append((int(issue), val))
    return records

def get_total_pages(data):
    """Use 'pages' key directly if available, else calculate from 'total'."""
    d = data.get('data', {})
    pages = d.get('pages')
    if pages:
        return int(pages)
    total = d.get('total', 0)
    return (int(total) + PAGE_SIZE - 1) // PAGE_SIZE if total else None


# ── Fetch ──────────────────────────────────────────────────────────────────────

_HARD_LIMIT_ERRORS = {400, 403, 404, 410}

def _is_connection_reset(exc):
    """True for WinError 10054 or similar forcible-close errors."""
    return (isinstance(exc, OSError) and
            getattr(exc, 'winerror', None) in (10054, 10053, 10061))

def fetch_page(headers, page_num, retries=3):
    url = f"{API_URL}?subServiceCode={GAME2_NAME}&size={PAGE_SIZE}&current={page_num}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in _HARD_LIMIT_ERRORS:
                return 'SKIP_400'   # API history limit — no retry
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  [ERR] page {page_num}: HTTP {e.code}")
        except Exception as e:
            if _is_connection_reset(e):
                return 'SKIP_400'   # Server forcibly closed — same as 400
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  [ERR] page {page_num}: {e}")
    return None


# ── Auth via Playwright ────────────────────────────────────────────────────────

def capture_auth_headers():
    from playwright.sync_api import sync_playwright
    captured = {}
    print(f"[1] Opening browser to capture auth headers for {GAME2_NAME}...")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, executable_path=CHROME or None)
        ctx = browser.new_context(
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            viewport={'width': 390, 'height': 844},
            ignore_https_errors=True,
        )
        page = ctx.new_page()

        def on_request(request):
            if 'issue-result/page' in request.url:
                captured.update(dict(request.headers))
                print("  [AUTH] Headers captured.")

        page.on('request', on_request)

        try:
            page.goto(GAME2_URL, wait_until='load', timeout=30000)
        except Exception:
            pass

        deadline = time.time() + 90
        while not captured and time.time() < deadline:
            time.sleep(1)

        if not captured:
            input("  Headers not captured. Log in manually then press Enter: ")
            time.sleep(3)

        browser.close()
    return captured


# ── Main backfill ──────────────────────────────────────────────────────────────

def main():
    ensure_table()
    known = existing_ids()
    print(f"  DB currently has {len(known)} records in cg.db.")

    headers = capture_auth_headers()
    if not headers:
        print("Could not capture auth headers. Exiting.")
        return

    print(f"\n[2] Probing API for total available history ({GAME2_NAME})...")
    data = fetch_page(headers, 1)
    if not data:
        print("Failed to fetch page 1. Check GAME2_NAME in .env and auth headers.")
        return
    if data.get('code') == 401:
        print("Auth error (401). Re-run the script to re-authenticate.")
        return

    # Show raw response snippet to help debug value format if needed
    sample = (data.get('data') or {}).get('records', [])
    if sample:
        print(f"  Sample record: {sample[0]}")

    pages = get_total_pages(data)
    if not pages:
        print("Could not determine total pages. API response:")
        print(json.dumps(data)[:600])
        return

    d_info = data.get('data', {})
    total_records = d_info.get('total', pages * PAGE_SIZE)
    print(f"  Total pages  : {pages}")
    print(f"  Total records: ~{total_records}")

    # Save page 1 records immediately (they came free with the probe)
    p1_records = parse_records(data)
    batch = []
    for issue, number in p1_records:
        if issue not in known:
            color, is_purple = number_to_color(number)
            batch.append((issue, number, color, is_purple))
            known.add(issue)

    inserted = 0
    errors   = 0

    # Fetch remaining pages oldest-first (highest page = oldest data).
    # 400 errors on high page numbers just mean the API doesn't hold data
    # that far back — skip silently and continue to lower page numbers.
    page_range   = list(range(pages, 1, -1))   # pages → 2, skip 1 (already fetched)
    skipped_400  = 0
    consec_skip  = 0   # consecutive unavailable pages (suppress noise)
    for idx, page_num in enumerate(page_range):
        data = fetch_page(headers, page_num)

        if data == 'SKIP_400':
            skipped_400 += 1
            consec_skip += 1
            if consec_skip == 1:
                print(f"  Page {page_num}: API history limit reached — skipping unavailable pages silently…")
            continue   # no data here, keep going to lower page numbers

        if consec_skip > 1:
            print(f"  …skipped {consec_skip} unavailable pages, resuming at page {page_num}")
        consec_skip = 0

        if not data:
            errors += 1
            print(f"  Skipping page {page_num} (network error).")
            continue

        if data.get('code') == 401:
            print(f"\n  Auth expired at page {page_num}. Re-run to resume (existing records are saved).")
            break

        records = parse_records(data)
        for issue, number in records:
            if issue not in known:
                color, is_purple = number_to_color(number)
                batch.append((issue, number, color, is_purple))
                known.add(issue)

        # Flush batch every 500 records
        if len(batch) >= 500:
            save_batch(batch)
            inserted += len(batch)
            batch = []

        # Progress every 10 pages
        if (idx + 1) % 10 == 0 or page_num == 2:
            pct_done = (idx + 1) / len(page_range) * 100
            print(f"  Page {page_num:>4}/{pages}  [{pct_done:4.0f}%]  saved: {inserted + len(batch)}  (skipped {skipped_400} unavailable)")

        time.sleep(0.12)

    # Flush remaining
    if batch:
        save_batch(batch)
        inserted += len(batch)

    print(f"\nDone. Inserted {inserted} new records.")
    if skipped_400:
        print(f"  ({skipped_400} pages skipped — API history limit reached for old data)")
    if errors:
        print(f"  ({errors} pages failed with network errors)")
    with get_conn() as conn:
        total = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
    print(f"cg.db now has {total} total records.")

    sync_csv()


if __name__ == '__main__':
    main()
