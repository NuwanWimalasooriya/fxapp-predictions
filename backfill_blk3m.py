"""
BLK3M Historical Backfill
Fetches up to 80 pages (8 000 rounds) of history from the API,
inserts missing records into blk3m.db, then runs predict_blk3m.py.

Usage: python backfill_blk3m.py
"""
import os, sys, json, time, ssl, sqlite3, subprocess, urllib.request, csv
sys.stdout.reconfigure(encoding='utf-8')

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

MAX_PAGES = 9999   # fetch all available pages from the API
PAGE_SIZE = 100


# ── Config ─────────────────────────────────────────────────────────────────────

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
BASE_URL  = _env.get('BASE_URL_BLK3M', _env.get('BASE_URL', 'https://m.fxpro1.net'))
GAME_NAME = _env.get('GAME_NAME_BLK3M', 'BLK3M')
GAME_URL  = f'{BASE_URL}/openHistory?gameName={GAME_NAME}'
API_URL   = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
DB_PATH   = os.path.join(_DATA_DIR, _env.get('DB_FILE_BLK3M',   'blk3m.db'))
CSV_PATH  = os.path.join(_DATA_DIR, _env.get('DATA_FILE_BLK3M', 'blk3m.csv'))
CHROME    = _env.get('CHROME', '')

print(f"Game  : {GAME_NAME}")
print(f"DB    : {DB_PATH}")
print(f"Max   : {MAX_PAGES} pages ({MAX_PAGES * PAGE_SIZE} rounds)")


# ── Classify ───────────────────────────────────────────────────────────────────

_BIG_HEX = set('89abcdef')
_ODD_HEX = set('13579bdf')
_HEX_NUM = {**{str(i): i for i in range(10)},
            **{'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}}

def classify(hex_char):
    h  = str(hex_char).strip().lower()
    bs = 'BIG'  if h in _BIG_HEX else 'SMALL'
    oe = 'ODD'  if h in _ODD_HEX else 'EVEN'
    return bs, oe


# ── Parse value ────────────────────────────────────────────────────────────────

def parse_value(value_str):
    """Extract the last hex character (0-9, a-f) from the API value string."""
    s = str(value_str).strip().lower()
    for c in reversed(s):
        if c in '0123456789abcdef':
            return c, s
    return None, s


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn

def ensure_table():
    with get_conn() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS rounds (
            id          INTEGER PRIMARY KEY,
            value       TEXT    NOT NULL,
            total       INTEGER NOT NULL,
            big_small   TEXT    NOT NULL,
            odd_even    TEXT    NOT NULL,
            result      TEXT    DEFAULT "Not Predicted",
            pred_bs     TEXT    DEFAULT "",
            confidence  REAL    DEFAULT 0,
            bet         INTEGER DEFAULT 0,
            pred_oe     TEXT    DEFAULT "",
            conf_oe     REAL    DEFAULT 0,
            result_oe   TEXT    DEFAULT "Not Predicted",
            bet_oe      INTEGER DEFAULT 0
        )''')

def existing_ids():
    with get_conn() as conn:
        rows = conn.execute('SELECT id FROM rounds').fetchall()
    return {r[0] for r in rows}

def save_batch(records):
    with get_conn() as conn:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds(id,value,total,big_small,odd_even) VALUES(?,?,?,?,?)',
            records
        )

def sync_csv():
    try:
        with get_conn() as conn:
            rows = conn.execute(
                'SELECT id,value,total,big_small,odd_even,'
                'result,pred_bs,confidence,bet,'
                'pred_oe,conf_oe,result_oe,bet_oe FROM rounds ORDER BY id'
            ).fetchall()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','value','total','big_small','odd_even',
                        'result','pred_bs','confidence','bet',
                        'pred_oe','conf_oe','result_oe','bet_oe'])
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
        print(f"  CSV synced: {len(rows)} rows → {CSV_PATH}")
    except Exception as e:
        print(f"  [CSV] {e}")


# ── API helpers ────────────────────────────────────────────────────────────────

def fetch_page(headers, page_num):
    url = f"{API_URL}?subServiceCode={GAME_NAME}&size={PAGE_SIZE}&current={page_num}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [ERR] page {page_num}: {e}")
        return None

def total_pages(data):
    d     = data.get('data', {})
    total = d.get('total', 0)
    return (int(total) + PAGE_SIZE - 1) // PAGE_SIZE if total else None

def parse_records(data):
    records = []
    for item in data.get('data', {}).get('records', []):
        issue    = item.get('issue')
        raw      = str(item.get('value', ''))
        hex_char, _ = parse_value(raw)
        if issue is not None and hex_char is not None:
            records.append((int(issue), raw, hex_char))
    return records


# ── Browser auth ───────────────────────────────────────────────────────────────

def capture_auth_headers():
    from playwright.sync_api import sync_playwright

    def _load_credentials():
        import hashlib
        secret = _env.get('SECRET_KEY', 'ds3m-default-key')
        key    = hashlib.sha256(secret.encode()).digest()
        p = os.path.join(_DATA_DIR, 'credentials.json')
        if os.path.exists(p):
            try:
                with open(p) as f:
                    c = json.load(f)
                data = bytes.fromhex(c['username'])
                ks   = (key * (len(data) // len(key) + 1))[:len(data)]
                user = bytes(a ^ b for a, b in zip(data, ks)).decode()
                data = bytes.fromhex(c['password'])
                ks   = (key * (len(data) // len(key) + 1))[:len(data)]
                pwd  = bytes(a ^ b for a, b in zip(data, ks)).decode()
                return user, pwd
            except Exception:
                pass
        return _env.get('USERNAME', ''), _env.get('PASSWORD', '')

    user, pwd = _load_credentials()
    captured  = {}

    print("\n[1] Opening browser to capture auth headers...")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, executable_path=CHROME or None)
        ctx = browser.new_context(
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
                       'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            viewport={'width': 390, 'height': 844},
            ignore_https_errors=True,
        )
        page = ctx.new_page()

        def on_request(request):
            if 'issue-result/page' in request.url:
                captured.update(dict(request.headers))
                print("  [AUTH] Headers captured.")

        page.on('request', on_request)

        page.goto(BASE_URL, wait_until='load', timeout=30000)
        page.wait_for_timeout(2000)

        for sel in ['text=Login', 'a:has-text("Login")', '[href*="login"]']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(); page.wait_for_timeout(1500); break
            except Exception:
                pass

        for sel in ['input[type="text"]', 'input[name*="phone"]',
                    'input[name*="user"]', 'input[placeholder*="phone"]']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.fill(user); break
            except Exception:
                pass

        try:
            el = page.query_selector('input[type="password"]')
            if el and el.is_visible():
                el.fill(pwd)
        except Exception:
            pass

        for sel in ['button[type="submit"]', 'text=Login', 'text=Confirm']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(); break
            except Exception:
                pass

        page.wait_for_timeout(4000)
        page.goto(GAME_URL, wait_until='load', timeout=30000)
        page.wait_for_timeout(4000)

        deadline = time.time() + 30
        while not captured and time.time() < deadline:
            time.sleep(1)

        if not captured:
            input("  Still no headers — log in manually then press Enter: ")
            time.sleep(3)

        browser.close()
    return captured


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ensure_table()
    known = existing_ids()
    print(f"\nDB currently has {len(known)} records.")

    headers = capture_auth_headers()
    if not headers:
        print("Could not capture auth headers. Exiting.")
        return

    print(f"\n[2] Probing API for available BLK3M history...")
    data1 = fetch_page(headers, 1)
    if not data1:
        print("Failed to fetch page 1.")
        return
    if data1.get('code') == 401:
        print("401 Unauthorized — re-run and log in properly.")
        return

    api_pages = total_pages(data1)
    if not api_pages:
        print("Could not determine page count. Response:", json.dumps(data1)[:300])
        return

    pages_to_fetch = min(api_pages, MAX_PAGES)
    print(f"  API has {api_pages} pages total. Will fetch {pages_to_fetch} pages "
          f"({pages_to_fetch * PAGE_SIZE} rounds).")
    print(f"  Already have: {len(known)} records.\n")

    inserted = 0
    errors   = 0

    # Fetch oldest-first (highest page number = oldest data)
    for page_num in range(pages_to_fetch, 0, -1):
        data = fetch_page(headers, page_num)
        if not data:
            errors += 1
            time.sleep(2)
            continue
        if data.get('code') == 401:
            print("  Auth expired — stopping.")
            break

        records = parse_records(data)
        batch = []
        for iss, raw, hex_char in records:
            if iss not in known:
                bs, oe  = classify(hex_char)
                numeric = _HEX_NUM.get(hex_char, 0)
                batch.append((iss, raw, numeric, bs, oe))
                known.add(iss)

        if batch:
            save_batch(batch)
            inserted += len(batch)

        if page_num % 10 == 0 or page_num <= 5 or page_num == pages_to_fetch:
            print(f"  Page {page_num:3d}/{pages_to_fetch} — inserted so far: {inserted}")

        time.sleep(0.2)

    print(f"\nBackfill done. Inserted {inserted} new records. Errors: {errors}")
    print(f"DB now has {len(known)} total records.")

    if inserted > 0:
        sync_csv()
        print("\n[3] Running predict_blk3m.py to generate initial prediction...")
        subprocess.run([sys.executable, os.path.join(_BASE, 'predict_blk3m.py')])
        print("Done.")


if __name__ == '__main__':
    main()
