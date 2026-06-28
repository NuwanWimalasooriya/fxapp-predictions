"""
BLK3M Live Collector — 3-minute Blocks.
Result = last hex character of the hash value (0–f).
Big/Small : 0-7 = SMALL, 8-f = BIG
Odd/Even  : 0,2,4,6,8,a,c,e = EVEN  |  1,3,5,7,9,b,d,f = ODD

Usage: python collector_blk3m.py
Press Ctrl+C to stop.
"""
import os, sys, json, time, subprocess, ssl, sqlite3, csv
import urllib.request
sys.stdout.reconfigure(encoding='utf-8')
from license_manager import check_license

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE
from playwright.sync_api import sync_playwright

_BASE         = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR     = os.path.join(_BASE, 'data')
PAGE_SIZE     = 100
FETCH_SECOND  = 10
INTERVAL_SECS = 180   # 3-minute cycle


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

_env = _load_env()


def _cipher_key():
    import hashlib
    secret = _env.get('SECRET_KEY', 'ds3m-default-key')
    return hashlib.sha256(secret.encode()).digest()

def _decrypt(hex_text):
    key = _cipher_key()
    data = bytes.fromhex(hex_text)
    ks   = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, ks)).decode('utf-8')

def _load_credentials():
    p = os.path.join(_DATA_DIR, 'credentials.json')
    if os.path.exists(p):
        try:
            with open(p) as f:
                c = json.load(f)
            return _decrypt(c['username']), _decrypt(c['password'])
        except Exception:
            pass
    return _env.get('USERNAME', ''), _env.get('PASSWORD', '')


USERNAME  = _load_credentials()[0]
PASSWORD  = _load_credentials()[1]
BASE_URL  = _env.get('BASE_URL_BLK3M', _env.get('BASE_URL', 'https://m.fxpro1.net'))
GAME_NAME = _env.get('GAME_NAME_BLK3M', 'BLK3M')
GAME_URL  = f'{BASE_URL}/openHistory?gameName={GAME_NAME}'
API_URL   = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
CHROME    = _env.get('CHROME', '')

DB_PATH    = os.path.join(_DATA_DIR, _env.get('DB_FILE_BLK3M',   'blk3m.db'))
CSV_PATH   = os.path.join(_DATA_DIR, _env.get('DATA_FILE_BLK3M', 'blk3m.csv'))
LOG_PATH   = os.path.join(_DATA_DIR, 'pred_log_blk3m.json')
PREDICT_PY = os.path.join(_BASE, 'predict_blk3m.py')

# ── Result helpers ─────────────────────────────────────────────────────────────

_BIG_HEX  = set('89abcdef')
_ODD_HEX  = set('13579bdf')
_HEX_NUM  = {**{str(i): i for i in range(10)},
             **{'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}}

def classify(hex_char):
    """Classify a single hex character (0-f) into BIG/SMALL and ODD/EVEN."""
    h  = str(hex_char).strip().lower()
    bs = 'BIG'  if h in _BIG_HEX else 'SMALL'
    oe = 'ODD'  if h in _ODD_HEX else 'EVEN'
    return bs, oe

def parse_value(value_str):
    """
    Extract the last hex character from the API value string.
    The game result is the last nibble of a hash (0-9 or a-f).
    Returns the hex char (str) or None.
    """
    s = str(value_str).strip().lower()
    for c in reversed(s):
        if c in '0123456789abcdef':
            return c
    return None


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

def fetch_page(headers, page_num):
    url = f"{API_URL}?subServiceCode={GAME_NAME}&size={PAGE_SIZE}&current={page_num}"
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
        issue    = item.get('issue')
        raw      = str(item.get('value', ''))
        hex_char = parse_value(raw)
        if issue is not None and hex_char is not None:
            records.append((int(issue), raw, hex_char))
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

        print("[1] Opening BLK3M site...")
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

        print("[2] Navigating to BLK3M history page...")
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


# ── CSV backup ─────────────────────────────────────────────────────────────────

def sync_csv_backup():
    try:
        conn = get_conn()
        try:
            rows = conn.execute(
                'SELECT id,value,total,big_small,odd_even,'
                'result,pred_bs,confidence,bet,'
                'pred_oe,conf_oe,result_oe,bet_oe '
                'FROM rounds ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','value','total','big_small','odd_even',
                        'result','pred_bs','confidence','bet',
                        'pred_oe','conf_oe','result_oe','bet_oe'])
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
    except Exception as e:
        print(f'  [CSV backup] {e}')


# ── Fetch new records ──────────────────────────────────────────────────────────

def fetch_new_records(auth_headers):
    cutoff      = last_db_id()
    new_records = []

    data1 = fetch_page(auth_headers, 1)
    if not data1:
        return []
    if data1.get('code') == 401:
        raise PermissionError("401 — auth expired")
    for iss, val, total in parse_records(data1):
        if iss > cutoff:
            new_records.append((iss, val, total))

    if not new_records:
        return []

    p = 2
    while True:
        data = fetch_page(auth_headers, p)
        if not data:
            break
        if data.get('code') == 401:
            raise PermissionError("401 — auth expired")
        if data.get('code') != 200:
            break
        records = parse_records(data)
        if not records:
            break
        added = 0
        for iss, val, total in records:
            if iss > cutoff:
                new_records.append((iss, val, total)); added += 1
        if added == 0:
            break
        if p >= data.get('data', {}).get('pages', 1):
            break
        p += 1
        time.sleep(0.3)

    new_records.sort(key=lambda x: x[0])
    seen, unique = set(), []
    for iss, val, total in new_records:
        if iss not in seen:
            seen.add(iss); unique.append((iss, val, total))
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

def _win_loss(issue, bs, oe, pred_log):
    entry     = pred_log.get(str(issue), {})
    result    = 'Not Predicted'
    result_oe = 'Not Predicted'
    if entry.get('pred_bs'):
        result    = 'WIN' if entry['pred_bs'] == bs else 'LOSS'
    if entry.get('pred_oe'):
        result_oe = 'WIN' if entry['pred_oe'] == oe else 'LOSS'
    return result, result_oe

def save_records(new_items):
    pred_log = _load_pred_log()
    rows = []
    for iss, raw, hex_char in new_items:
        bs, oe            = classify(hex_char)
        numeric           = _HEX_NUM.get(hex_char, 0)
        result, result_oe = _win_loss(iss, bs, oe, pred_log)
        entry             = pred_log.get(str(iss), {})
        pred_bs           = entry.get('pred_bs', '')
        confidence        = float(entry.get('confidence', 0) or 0)
        bet               = int(entry.get('bet', 0) or 0)
        pred_oe_val       = entry.get('pred_oe', '')
        conf_oe           = float(entry.get('conf_oe', 0) or 0)
        bet_oe            = int(entry.get('bet_oe', 0) or 0)
        rows.append((iss, raw, numeric, bs, oe,
                     result, pred_bs, confidence, bet,
                     pred_oe_val, conf_oe, result_oe, bet_oe))
    conn = get_conn()
    try:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds'
            '(id,value,total,big_small,odd_even,result,pred_bs,confidence,bet,'
            'pred_oe,conf_oe,result_oe,bet_oe) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        # Fix any stale wrong classifications (preserves prediction columns)
        conn.executemany(
            'UPDATE rounds SET big_small=?, odd_even=?, total=? WHERE id=? '
            'AND (big_small != ? OR odd_even != ? OR total != ?)',
            [(bs, oe, num, iss, bs, oe, num)
             for iss, raw, num, bs, oe, *_ in rows]
        )
        conn.commit()
    finally:
        conn.close()
    sync_csv_backup()

def backfill_results():
    conn    = get_conn()
    updates = []
    try:
        rows = conn.execute(
            "SELECT id,big_small,odd_even,pred_bs,pred_oe FROM rounds "
            "WHERE (pred_bs != '' AND result='Not Predicted') "
            "   OR (pred_oe != '' AND result_oe='Not Predicted')"
        ).fetchall()
        for rid, bs, oe, pbs, poe in rows:
            result    = ('WIN' if pbs == bs else 'LOSS') if pbs else None
            result_oe = ('WIN' if poe == oe else 'LOSS') if poe else None
            if result and result_oe:
                updates.append((result, result_oe, rid))
            elif result:
                updates.append((result, 'Not Predicted', rid))
            elif result_oe:
                updates.append(('Not Predicted', result_oe, rid))
        if updates:
            conn.executemany('UPDATE rounds SET result=?, result_oe=? WHERE id=?', updates)
            conn.commit()
            print(f"  Backfilled {len(updates)} result(s).")
    finally:
        conn.close()
    if updates:
        sync_csv_backup()


# ── Prediction ─────────────────────────────────────────────────────────────────

def run_prediction():
    print(f"\n{'='*55}")
    print("  BLK3M PREDICTION")
    print(f"{'='*55}\n")
    subprocess.run([sys.executable, PREDICT_PY])
    backfill_results()


def _push_to_remote(_first_id=None):
    try:
        from remote_push import push_game
        conn = get_conn()
        rows = conn.execute(
            'SELECT id,value,total,big_small,odd_even,result,pred_bs,confidence,bet,'
            'pred_oe,conf_oe,result_oe,bet_oe FROM rounds ORDER BY id DESC LIMIT 20'
        ).fetchall()
        conn.close()
        records = [
            {'id': r[0], 'value': r[1], 'total': r[2], 'big_small': r[3], 'odd_even': r[4],
             'result': r[5], 'pred_bs': r[6], 'confidence': r[7], 'bet': r[8],
             'pred_oe': r[9], 'conf_oe': r[10], 'result_oe': r[11], 'bet_oe': r[12]}
            for r in rows
        ]
        push_game('blk3m', records, os.path.join(_DATA_DIR, 'latest_prediction_blk3m.json'), LOG_PATH)
    except Exception as e:
        print(f'  [PUSH] {e}')


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    check_license()
    print("=" * 55)
    print(f"  BLK3M Live Collector  (Ctrl+C to stop)")
    print("=" * 55)

    ensure_table()
    with get_conn() as conn:
        count = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
    print(f"  DB has {count} existing records.\n")

    auth_headers, seed = capture_auth_headers()
    if not auth_headers:
        print("Could not capture auth headers. Exiting.")
        return

    if seed:
        save_records(seed)
        print(f"  Seeded {len(seed)} records from initial load.")
        run_prediction()

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
                print(f"  +{len(new_items)} new record(s) saved. Latest: {last}")
                for iss, raw, hex_char in new_items:
                    bs, oe = classify(hex_char)
                    print(f"  {iss}  raw={raw}  hex={hex_char}  {bs}/{oe}")

                while True:
                    extra = fetch_new_records(auth_headers)
                    if not extra:
                        break
                    save_records(extra)
                    for iss, raw, hex_char in extra:
                        bs, oe = classify(hex_char)
                        print(f"  {iss}  hex={hex_char}  {bs}/{oe}  (catch-up)")

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

        epoch     = time.time()
        slot      = (int(epoch) // INTERVAL_SECS) * INTERVAL_SECS
        next_fire = slot + INTERVAL_SECS + FETCH_SECOND
        wait      = int(next_fire - epoch)
        if wait < 2:
            wait += INTERVAL_SECS
        print(f"\n  Next update in {wait}s...")
        time.sleep(wait)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
