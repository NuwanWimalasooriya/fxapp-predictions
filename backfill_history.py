"""
Historical backfill script for DS3M.
Fetches all available history pages from the API and inserts into the DB.
Run: python backfill_history.py
"""
import os, sys, json, time, ssl, sqlite3, urllib.request
sys.stdout.reconfigure(encoding='utf-8')

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# ── Reuse config from collector ────────────────────────────────────────────────
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

_env      = _load_env()
BASE_URL  = _env.get('BASE_URL',  'https://m.fxpro1.net')
GAME_NAME = _env.get('GAME_NAME', 'DS3M')
GAME_URL  = f'{BASE_URL}/openHistory?gameName={GAME_NAME}'
API_URL   = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
DB_PATH   = os.path.join(_DATA_DIR, _env.get('DB_FILE', 'ds3m.db'))
CHROME    = _env.get('CHROME', '')
PAGE_SIZE = 100

# ── DB helpers ─────────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def ensure_table():
    with get_conn() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS rounds (
            id      INTEGER PRIMARY KEY,
            disc1   TEXT, disc2 TEXT, disc3 TEXT, disc4 TEXT,
            pattern TEXT, flag TEXT, oe TEXT,
            result  TEXT DEFAULT "Not Predicted",
            pred_oe TEXT DEFAULT "", confidence REAL, bet INTEGER DEFAULT 0
        )''')

def existing_ids():
    with get_conn() as conn:
        rows = conn.execute('SELECT id FROM rounds').fetchall()
    return {r[0] for r in rows}

def save_records(records):
    with get_conn() as conn:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds (id,disc1,disc2,disc3,disc4,pattern,flag,oe) VALUES (?,?,?,?,?,?,?,?)',
            records
        )

# ── Parse helpers ──────────────────────────────────────────────────────────────
def parse_value(s):
    if not s: return None
    parts = [c for c in str(s) if c in '01']
    if len(parts) != 4: return None
    return ['R' if c == '1' else 'W' for c in parts]

def parse_records(data):
    records = []
    for item in data.get('data', {}).get('records', []):
        issue = item.get('issue')
        discs = parse_value(item.get('value', ''))
        if issue and discs:
            records.append((int(issue), discs))
    return records

def total_pages(data):
    d = data.get('data', {})
    total = d.get('total', 0)
    return (int(total) + PAGE_SIZE - 1) // PAGE_SIZE if total else None

# ── Fetch ──────────────────────────────────────────────────────────────────────
def fetch_page(headers, page_num):
    url = f"{API_URL}?subServiceCode={GAME_NAME}&size={PAGE_SIZE}&current={page_num}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [ERR] page {page_num}: {e}")
        return None

# ── Auth via Playwright ────────────────────────────────────────────────────────
def capture_auth_headers():
    from playwright.sync_api import sync_playwright
    captured = {}
    print("[1] Opening browser to capture auth headers...")
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
            page.goto(GAME_URL, wait_until='load', timeout=30000)
        except Exception:
            pass

        deadline = time.time() + 60
        while not captured and time.time() < deadline:
            time.sleep(1)

        browser.close()
    return captured

# ── Main backfill ──────────────────────────────────────────────────────────────
def main():
    ensure_table()
    known = existing_ids()
    print(f"DB currently has {len(known)} records.")

    headers = capture_auth_headers()
    if not headers:
        print("Could not capture auth headers. Exiting.")
        return

    # Probe page 1 to find total pages available
    print("\n[2] Probing API to find total available history...")
    data = fetch_page(headers, 1)
    if not data:
        print("Failed to fetch page 1.")
        return

    pages = total_pages(data)
    print(f"  Total pages available: {pages}  (~{pages*PAGE_SIZE if pages else '?'} rounds)")

    # Fetch all pages oldest-first (last page = oldest data)
    inserted = 0
    errors   = 0
    for page_num in range(pages, 0, -1):
        data = fetch_page(headers, page_num)
        if not data:
            errors += 1
            time.sleep(2)
            continue

        records = parse_records(data)
        new = [(r[0], r[1][0], r[1][1], r[1][2], r[1][3],
                ''.join(r[1]),
                'WWWW' if ''.join(r[1])=='WWWW' else ('RRRR' if ''.join(r[1])=='RRRR' else ''),
                'ODD' if ''.join(r[1]).count('R')%2 else 'EVEN')
               for r in records if r[0] not in known]

        if new:
            save_records(new)
            for r in new:
                known.add(r[0])
            inserted += len(new)

        if page_num % 50 == 0 or page_num == pages:
            print(f"  Page {page_num}/{pages} — inserted so far: {inserted}")

        time.sleep(0.2)  # polite rate limit

    print(f"\nDone. Inserted {inserted} new records. Errors: {errors}")
    print(f"DB now has {len(known)} total records.")

if __name__ == '__main__':
    main()
