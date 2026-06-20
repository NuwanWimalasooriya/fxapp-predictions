"""
Color Game Collector — polls for new rounds every cycle, stores in rg3m.db, runs predictor.

Color rules:
  0,2,4,6,8  → red    (even)
  1,3,5,7,9  → green  (odd)
  0          → red + purple
  5          → green + purple

Usage: python collector_cg.py
Press Ctrl+C to stop.
"""
import os, sys, json, time, subprocess, ssl, sqlite3, csv
import urllib.request
sys.stdout.reconfigure(encoding='utf-8')
from license_manager import check_license

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE
from playwright.sync_api import sync_playwright

_BASE        = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_BASE, 'data')
PAGE_SIZE    = 100
FETCH_SECOND = 5


def _load_env():
    env = {}
    env_path = os.path.join(_BASE, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env = _load_env()


def _cipher_key():
    import hashlib
    secret = _env.get('SECRET_KEY', 'ds3m-default-key')
    return hashlib.sha256(secret.encode()).digest()

def _decrypt(hex_text):
    key = _cipher_key()
    data = bytes.fromhex(hex_text)
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, key_stream)).decode('utf-8')

def _load_credentials():
    creds_path = os.path.join(_DATA_DIR, 'credentials.json')
    if os.path.exists(creds_path):
        try:
            with open(creds_path) as f:
                c = json.load(f)
            return _decrypt(c['username']), _decrypt(c['password'])
        except Exception:
            pass
    return _env.get('USERNAME', ''), _env.get('PASSWORD', '')


USERNAME   = _load_credentials()[0]
PASSWORD   = _load_credentials()[1]
BASE_URL   = _env.get('BASE_URL',   'https://m.fxpro1.net')
GAME2_NAME = _env.get('GAME2_NAME', 'CGAME')
GAME2_URL  = _env.get('GAME2_URL',  f'{BASE_URL}/openHistory?gameName={GAME2_NAME}')
API_URL    = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
CHROME     = _env.get('CHROME', '')

DB_PATH      = os.path.join(_DATA_DIR, _env.get('DB_FILE2', 'rg3m.db'))
LOG_PATH     = os.path.join(_DATA_DIR, 'pred_log_cg.json')
CSV_PATH     = os.path.join(_DATA_DIR, 'cg.csv')
PREDICT_CG   = os.path.join(_BASE, 'predict_cg.py')


# ── Color helpers ──────────────────────────────────────────────────────────────

def number_to_color(n):
    """Return (color, is_purple) for digit n (0-9)."""
    color     = 'red' if n % 2 == 0 else 'green'
    is_purple = 1 if n in (0, 5) else 0
    return color, is_purple


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn

def ensure_table():
    conn = get_conn()
    try:
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
        conn.commit()
    finally:
        conn.close()

def last_db_id():
    conn = get_conn()
    try:
        row = conn.execute('SELECT MAX(id) FROM rounds').fetchone()
        return row[0] if row and row[0] else 0
    finally:
        conn.close()


# ── API helpers ────────────────────────────────────────────────────────────────

def parse_value(value_str):
    """Parse a single digit result (0-9) from the API value field."""
    s = str(value_str).strip()
    # Value may be a single digit or a short string — extract first digit
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return None

def fetch_page(headers, page_num):
    url = f"{API_URL}?subServiceCode={GAME2_NAME}&size={PAGE_SIZE}&current={page_num}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [ERR] {e}")
        return None

def parse_records(data):
    records = []
    for item in data.get('data', {}).get('records', []):
        issue = item.get('issue')
        val   = parse_value(item.get('value', ''))
        if issue is not None and val is not None:
            records.append((int(issue), val))
    return records


# ── Browser auth ───────────────────────────────────────────────────────────────

def capture_auth_headers():
    captured_headers = {}
    captured_data    = []

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
                captured_headers.update(dict(request.headers))
                print(f"  [AUTH] Headers captured.")

        def on_response(response):
            if 'issue-result/page' in response.url:
                try:
                    recs = parse_records(response.json())
                    if recs:
                        captured_data.extend(recs)
                        print(f"  [DATA] Page 1: {len(recs)} records")
                except:
                    pass

        page.on('request',  on_request)
        page.on('response', on_response)

        print("[1] Opening site...")
        page.goto(BASE_URL, wait_until='load', timeout=30000)
        page.wait_for_timeout(2000)

        for sel in ['text=Login', 'a:has-text("Login")', '[href*="login"]']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(); page.wait_for_timeout(1500); break
            except: pass

        for sel in ['input[type="text"]','input[name*="phone"]','input[name*="user"]',
                    'input[name*="account"]','input[placeholder*="phone"]']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.fill(USERNAME); break
            except: pass

        try:
            el = page.query_selector('input[type="password"]')
            if el and el.is_visible():
                el.fill(PASSWORD)
        except: pass

        for sel in ['button[type="submit"]','text=Login','text=Confirm','button:has-text("Log")']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(); break
            except: pass

        print("  If auto-login failed, complete it manually in the browser window.")
        page.wait_for_timeout(5000)

        print("[2] Navigating to Color Game history page...")
        page.goto(GAME2_URL, wait_until='load', timeout=30000)
        page.wait_for_timeout(4000)

        if not captured_headers:
            print("  Waiting 10 more seconds...")
            page.wait_for_timeout(10000)
        if not captured_headers:
            input("  Still no headers. Log in manually then press Enter: ")
            page.wait_for_timeout(3000)

        browser.close()

    return captured_headers, captured_data


# ── CSV backup ─────────────────────────────────────────────────────────────────

def sync_csv():
    try:
        conn = get_conn()
        try:
            rows = conn.execute(
                'SELECT id,number,color,is_purple,result,pred_color,confidence,bet FROM rounds ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','number','color','is_purple','result','pred_color','confidence','bet'])
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
    except Exception as e:
        print(f'  [CSV] {e}')


# ── Pred log ───────────────────────────────────────────────────────────────────

def _load_pred_log():
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH, encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return {}

def _win_loss(issue, color, pred_log):
    entry = pred_log.get(str(issue))
    if not entry or not entry.get('pred_color'):
        return 'Not Predicted'
    return 'WIN' if entry['pred_color'] == color else 'LOSS'


# ── Save records ───────────────────────────────────────────────────────────────

def save_records(new_items):
    pred_log = _load_pred_log()
    rows = []
    for issue, number in new_items:
        color, is_purple = number_to_color(number)
        entry      = pred_log.get(str(issue), {})
        pred_color = entry.get('pred_color', '')
        confidence = float(entry.get('confidence', 0) or 0)
        bet        = 1 if entry.get('bet', False) else 0
        result     = _win_loss(issue, color, pred_log)
        rows.append((issue, number, color, is_purple, result, pred_color, confidence, bet))

    conn = get_conn()
    try:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds(id,number,color,is_purple,result,pred_color,confidence,bet) '
            'VALUES(?,?,?,?,?,?,?,?)',
            rows
        )
        # Update predictions for rows already inserted (in case predictor ran earlier)
        for issue, number, color, is_purple, result, pred_color, confidence, bet in rows:
            if pred_color:
                conn.execute(
                    'UPDATE rounds SET pred_color=?, confidence=?, bet=?, result=? WHERE id=?',
                    (pred_color, confidence, bet, result, issue)
                )
        conn.commit()
    finally:
        conn.close()
    sync_csv()


def backfill_results():
    """Fix rows that have pred_color set but result still 'Not Predicted'."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, color, pred_color FROM rounds WHERE pred_color != '' AND result='Not Predicted'"
        ).fetchall()
        updates = [('WIN' if pc == c else 'LOSS', rid) for rid, c, pc in rows]
        if updates:
            conn.executemany('UPDATE rounds SET result=? WHERE id=?', updates)
            conn.commit()
            print(f"  Backfilled {len(updates)} result(s).")
    finally:
        conn.close()
    if updates:
        sync_csv()


# ── Fetch new records ──────────────────────────────────────────────────────────

def fetch_new_records(auth_headers):
    cutoff = last_db_id()
    new_records = []

    data1 = fetch_page(auth_headers, 1)
    if not data1:
        return []
    if data1.get('code') == 401:
        raise PermissionError("401 — auth expired")
    page1 = parse_records(data1)
    for iss, val in page1:
        if iss > cutoff:
            new_records.append((iss, val))
    if not new_records:
        return []

    p = 2
    while True:
        data = fetch_page(auth_headers, p)
        if not data:
            break
        code = data.get('code')
        if code == 401:
            raise PermissionError("401 — auth expired")
        if code != 200:
            break
        records = parse_records(data)
        if not records:
            break
        added = 0
        for iss, v in records:
            if iss > cutoff:
                new_records.append((iss, v))
                added += 1
        if added == 0:
            break
        pages_total = data.get('data', {}).get('pages', 1)
        if p >= pages_total:
            break
        p += 1
        time.sleep(0.3)

    new_records.sort(key=lambda x: x[0])
    seen, unique = set(), []
    for iss, val in new_records:
        if iss not in seen:
            seen.add(iss)
            unique.append((iss, val))
    return unique


# ── Prediction ─────────────────────────────────────────────────────────────────

def run_prediction():
    print(f"\n{'='*55}")
    print("  COLOR GAME PREDICTION")
    print(f"{'='*55}\n")
    subprocess.run([sys.executable, PREDICT_CG])
    backfill_results()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    check_license()
    print("=" * 55)
    print(f"  Color Game Collector  [{GAME2_NAME}]  (Ctrl+C to stop)")
    print("=" * 55)

    ensure_table()
    with get_conn() as conn:
        count = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
    print(f"  DB has {count} existing records.\n")

    auth_headers, seed_data = capture_auth_headers()
    if not auth_headers:
        print("Could not capture auth headers. Exiting.")
        return

    # Save any records captured during auth
    if seed_data:
        save_records(seed_data)

    # Determine cycle interval from game name (default 3 min)
    INTERVAL_SECS = 180

    cycle = 0
    while True:
        cycle += 1
        now = time.strftime('%H:%M:%S')
        print(f"\n[Cycle {cycle} @ {now}] Checking for new records...")

        try:
            new_items = fetch_new_records(auth_headers)

            if new_items:
                save_records(new_items)
                last = new_items[-1][0]
                print(f"  +{len(new_items)} new records saved. Latest: {last}")
                for issue, number in new_items:
                    color, is_purple = number_to_color(number)
                    purple_tag = '+purple' if is_purple else ''
                    print(f"  {issue}  {number}  [{color}{purple_tag}]")

                # Drain any additional records
                while True:
                    extra = fetch_new_records(auth_headers)
                    if not extra:
                        break
                    save_records(extra)
                    print(f"  +{len(extra)} additional records. Latest: {extra[-1][0]}")
                    for issue, number in extra:
                        color, is_purple = number_to_color(number)
                        print(f"  {issue}  {number}  [{color}{'+purple' if is_purple else ''}]")

                run_prediction()
            else:
                print("  No new records. Skipping prediction.")

        except PermissionError:
            print("  Auth expired — re-logging in...")
            auth_headers, _ = capture_auth_headers()
            if not auth_headers:
                print("  Re-auth failed. Retrying next cycle.")
            continue

        now_t  = time.localtime()
        epoch  = time.mktime(now_t)
        slot_start = (int(epoch) // INTERVAL_SECS) * INTERVAL_SECS
        next_fire  = slot_start + INTERVAL_SECS + FETCH_SECOND
        wait = int(next_fire - epoch)
        if wait < 2:
            wait += INTERVAL_SECS
        print(f"\n  Next update in {wait}s...")
        time.sleep(wait)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
