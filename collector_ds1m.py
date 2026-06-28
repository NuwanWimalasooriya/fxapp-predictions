"""
DS1M Live Collector — logs in once, then polls every 1 minute for new records and predicts.
Usage: python collector_ds1m.py
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

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')
PAGE_SIZE     = 100
FETCH_SECOND  = 5    # fetch at this second of each 1-minute slot
INTERVAL_SECS = 60   # 1-minute cycle

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

USERNAME, PASSWORD = _load_credentials()
BASE_URL  = _env.get('BASE_URL',  'https://m.fxpro1.net')
GAME_NAME = _env.get('GAME_NAME_DS1M', 'DS1M')
GAME_URL  = _env.get('GAME_URL',  f'{BASE_URL}/openHistory?gameName={GAME_NAME}')
API_URL   = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
CHROME    = _env.get('CHROME', '')

DB_PATH      = os.path.join(_DATA_DIR, _env.get('DB_FILE_DS1M',   'ds1m.db'))
CSV_PATH     = os.path.join(_DATA_DIR, _env.get('DATA_FILE_DS1M', '1min-discs.csv'))
LOG_PATH     = os.path.join(_DATA_DIR, 'pred_log_ds1m.json')
PREDICT_PY   = os.path.join(_BASE,     'predict_ds1m.py')

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
                disc1      TEXT NOT NULL,
                disc2      TEXT NOT NULL,
                disc3      TEXT NOT NULL,
                disc4      TEXT NOT NULL,
                pattern    TEXT NOT NULL,
                flag       TEXT DEFAULT "",
                oe         TEXT NOT NULL,
                result     TEXT DEFAULT "Not Predicted",
                pred_oe    TEXT DEFAULT "",
                confidence REAL DEFAULT 0
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
    s = str(value_str).strip()
    if len(s) != 4 or not all(c in '01' for c in s):
        return None
    return ['R' if c == '1' else 'W' for c in s]

def fetch_page(url, headers, page_num, size=PAGE_SIZE):
    req_url = f"{url}?subServiceCode={GAME_NAME}&size={size}&current={page_num}"
    req = urllib.request.Request(req_url, headers=headers)
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
        discs = parse_value(item.get('value', ''))
        if issue and discs:
            records.append((int(issue), discs))
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
                except Exception:
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
            except Exception:
                pass

        for sel in ['input[type="text"]', 'input[name*="phone"]', 'input[name*="user"]',
                    'input[name*="account"]', 'input[placeholder*="phone"]']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.fill(USERNAME); break
            except Exception:
                pass

        try:
            el = page.query_selector('input[type="password"]')
            if el and el.is_visible():
                el.fill(PASSWORD)
        except Exception:
            pass

        for sel in ['button[type="submit"]', 'text=Login', 'text=Confirm', 'button:has-text("Log")']:
            try:
                el = page.query_selector(sel)
                if el and el.is_visible():
                    el.click(); break
            except Exception:
                pass

        print("  If auto-login failed, complete it manually in the browser window.")
        page.wait_for_timeout(5000)

        print("[2] Navigating to history page...")
        page.goto(GAME_URL, wait_until='load', timeout=30000)
        page.wait_for_timeout(4000)

        if not captured_headers:
            print("  Waiting 10 more seconds...")
            page.wait_for_timeout(10000)

        if not captured_headers:
            input("  Still no headers. Log in manually then press Enter: ")
            page.wait_for_timeout(3000)

        browser.close()

    return captured_headers, captured_data

# ── CSV backup ────────────────────────────────────────────────────────────────

def sync_csv_backup():
    try:
        conn = get_conn()
        try:
            rows = conn.execute(
                'SELECT id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence '
                'FROM rounds ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id','disc1','disc2','disc3','disc4','pattern','flag','oe',
                             'result','pred_oe','confidence'])
            writer.writerows(rows)
        os.replace(tmp, CSV_PATH)
    except Exception as e:
        print(f'  [CSV backup] {e}')

# ── Fetch new records ──────────────────────────────────────────────────────────

def fetch_new_records(auth_headers):
    cutoff      = last_db_id()
    new_records = []

    data1 = fetch_page(API_URL, auth_headers, 1)
    if not data1:
        return []
    if data1.get('code') == 401:
        raise PermissionError("401 — auth expired")
    page1_records = parse_records(data1)

    for iss, discs in page1_records:
        if iss > cutoff:
            new_records.append((iss, discs))

    if not new_records:
        return []

    p = 2
    while True:
        data = fetch_page(API_URL, auth_headers, p)
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
        for iss, discs in records:
            if iss > cutoff:
                new_records.append((iss, discs))
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
    for iss, discs in new_records:
        if iss not in seen:
            seen.add(iss)
            unique.append((iss, discs))
    return unique

# ── Save records ───────────────────────────────────────────────────────────────

def _load_pred_log():
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH, encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return {}

def _win_loss(issue, pattern, pred_log):
    entry = pred_log.get(str(issue))
    if not entry or not entry.get('pred_oe'):
        return 'Not Predicted'
    actual_oe = 'ODD' if pattern.count('R') % 2 else 'EVEN'
    return 'WIN' if entry['pred_oe'] == actual_oe else 'LOSS'

def save_records(new_items):
    pred_log = _load_pred_log()
    rows = []
    for issue, discs in new_items:
        pattern    = ''.join(discs)
        flag       = pattern if pattern in ('WWWW', 'RRRR') else ''
        oe         = 'ODD' if pattern.count('R') % 2 else 'EVEN'
        result     = _win_loss(issue, pattern, pred_log)
        entry      = pred_log.get(str(issue), {})
        pred_oe    = entry.get('pred_oe', '')
        confidence = float(entry.get('confidence', 0) or 0)
        rows.append((issue, discs[0], discs[1], discs[2], discs[3],
                     pattern, flag, oe, result, pred_oe, confidence))
    conn = get_conn()
    try:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds'
            '(id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        conn.commit()
    finally:
        conn.close()
    sync_csv_backup()

def backfill_results():
    conn = get_conn()
    updates = []
    try:
        rows = conn.execute(
            "SELECT id, disc1||disc2||disc3||disc4, pred_oe "
            "FROM rounds WHERE pred_oe != '' AND result='Not Predicted'"
        ).fetchall()
        for rid, pattern, p_oe in rows:
            if not pattern or len(pattern) != 4:
                continue
            actual_oe = 'ODD' if pattern.count('R') % 2 else 'EVEN'
            updates.append(('WIN' if p_oe == actual_oe else 'LOSS', rid))
        if updates:
            conn.executemany('UPDATE rounds SET result=? WHERE id=?', updates)
            conn.commit()
            print(f"  Backfilled {len(updates)} result(s).")
    finally:
        conn.close()
    if updates:
        sync_csv_backup()

# ── Prediction ─────────────────────────────────────────────────────────────────

def run_prediction():
    print(f"\n{'='*55}")
    print("  DS1M PREDICTION")
    print(f"{'='*55}\n")
    subprocess.run([sys.executable, PREDICT_PY])
    backfill_results()


def _push_to_remote(first_id):
    try:
        from remote_push import push_game
        conn = get_conn()
        rows = conn.execute(
            'SELECT id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence,bet '
            'FROM rounds WHERE id >= ?', (first_id,)
        ).fetchall()
        conn.close()
        records = [
            {'id': r[0], 'disc1': r[1], 'disc2': r[2], 'disc3': r[3], 'disc4': r[4],
             'pattern': r[5], 'flag': r[6], 'oe': r[7], 'result': r[8],
             'pred_oe': r[9], 'confidence': r[10], 'bet': r[11]}
            for r in rows
        ]
        snap_path = os.path.join(_DATA_DIR, 'latest_prediction_ds1m.json')
        push_game('ds1m', records, snap_path, LOG_PATH)
    except Exception as e:
        print(f'  [PUSH] {e}')

# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    check_license()
    print("=" * 55)
    print(f"  DS1M Live Collector  (Ctrl+C to stop)")
    print("=" * 55)

    ensure_table()
    with get_conn() as conn:
        count = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
    print(f"  DB has {count} existing records.\n")

    auth_headers, _ = capture_auth_headers()
    if not auth_headers:
        print("Could not capture auth headers. Exiting.")
        return

    cycle = 0
    while True:
        cycle += 1
        now = time.strftime('%H:%M:%S')
        print(f"\n[Cycle {cycle} @ {now}] Checking for new records...")

        new_items = []
        try:
            new_items = fetch_new_records(auth_headers)

            if new_items:
                save_records(new_items)
                last = new_items[-1][0]
                print(f"  +{len(new_items)} new records saved. Latest issue: {last}")
                for issue, discs in new_items:
                    pattern = ''.join(discs)
                    oe      = 'ODD' if pattern.count('R') % 2 else 'EVEN'
                    if pattern == 'RRRR':
                        print(f"  *** RRRR  {issue} ***  [{oe}]")
                    elif pattern == 'WWWW':
                        print(f"  *** WWWW  {issue} ***  [{oe}]")
                    else:
                        print(f"  {issue}  {pattern}  [{oe}]")

                while True:
                    extra = fetch_new_records(auth_headers)
                    if not extra:
                        break
                    save_records(extra)
                    print(f"  +{len(extra)} additional records saved. Latest: {extra[-1][0]}")
                    for issue, discs in extra:
                        pattern = ''.join(discs)
                        oe      = 'ODD' if pattern.count('R') % 2 else 'EVEN'
                        print(f"  {issue}  {pattern}  [{oe}]")

                run_prediction()
                _push_to_remote(new_items[0][0])
            else:
                print("  No new records. Skipping prediction.")

        except PermissionError:
            print("  Auth expired — re-logging in...")
            auth_headers, _ = capture_auth_headers()
            if not auth_headers:
                print("  Re-auth failed. Retrying next cycle.")
            continue

        now_t = time.localtime()
        epoch = time.mktime(now_t)
        if new_items:
            # Align to next 1-minute slot
            slot_start = (int(epoch) // INTERVAL_SECS) * INTERVAL_SECS
            next_fire  = slot_start + INTERVAL_SECS + FETCH_SECOND
            wait = int(next_fire - epoch)
            if wait < 2:
                wait += INTERVAL_SECS
            print(f"\n  Next update in {wait}s (at :{FETCH_SECOND:02d} of next minute)...")
        else:
            next_fire = (int(epoch) // 60) * 60 + 60 + FETCH_SECOND
            wait = int(next_fire - epoch)
            if wait < 2:
                wait += 60
            print(f"\n  No records — retrying in {wait}s...")
        time.sleep(wait)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
