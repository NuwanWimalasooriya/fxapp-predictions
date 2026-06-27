"""
FST1M Live Collector — 1-minute 4-Star Lottery.
Each round produces 4 numbers (0-9). Records total sum, BIG/SMALL, ODD/EVEN.
BIG = sum >= 18  |  SMALL = sum <= 17
ODD = sum is odd |  EVEN  = sum is even

Usage: python collector_fst1m.py
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
FETCH_SECOND  = 5
INTERVAL_SECS = 60


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


USERNAME   = _load_credentials()[0]
PASSWORD   = _load_credentials()[1]
BASE_URL   = _env.get('BASE_URL_FST1M', 'https://m.fxpro-usdt.vip')
GAME_NAME  = _env.get('GAME_NAME_FST1M', 'FST1M')
GAME_URL   = f'{BASE_URL}/openHistory?gameName={GAME_NAME}'
API_URL    = f'{BASE_URL}/api/rocket-api/game/issue-result/page'
CHROME     = _env.get('CHROME', '')

DB_PATH    = os.path.join(_DATA_DIR, _env.get('DB_FILE_FST1M',   'fst1m.db'))
CSV_PATH   = os.path.join(_DATA_DIR, _env.get('DATA_FILE_FST1M', 'fst1m.csv'))
LOG_PATH   = os.path.join(_DATA_DIR, 'pred_log_fst1m.json')
PREDICT_PY = os.path.join(_BASE, 'predict_fst1m.py')


# ── Result helpers ─────────────────────────────────────────────────────────────

def classify(total):
    """Return (big_small, odd_even) for a sum value."""
    bs = 'BIG' if total >= 18 else 'SMALL'
    oe = 'ODD'  if total % 2  else 'EVEN'
    return bs, oe


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
                id           INTEGER PRIMARY KEY,
                n1           INTEGER NOT NULL,
                n2           INTEGER NOT NULL,
                n3           INTEGER NOT NULL,
                n4           INTEGER NOT NULL,
                total        INTEGER NOT NULL,
                big_small    TEXT NOT NULL,
                odd_even     TEXT NOT NULL,
                result       TEXT    DEFAULT "Not Predicted",
                pred_bs      TEXT    DEFAULT "",
                confidence   REAL    DEFAULT 0,
                bet          INTEGER DEFAULT 0,
                pred_oe      TEXT    DEFAULT "",
                conf_oe      REAL    DEFAULT 0,
                result_oe    TEXT    DEFAULT "Not Predicted",
                bet_oe       INTEGER DEFAULT 0,
                pred_sum_val  INTEGER DEFAULT 0,
                pred_sum_zone TEXT   DEFAULT "",
                pred_hot_sums TEXT   DEFAULT "[]"
            )
        ''')
        # Safe migration for existing tables
        existing = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
        for col, defn in [
            ('pred_oe',       'TEXT    DEFAULT ""'),
            ('conf_oe',       'REAL    DEFAULT 0'),
            ('result_oe',     'TEXT    DEFAULT "Not Predicted"'),
            ('bet_oe',        'INTEGER DEFAULT 0'),
            ('pred_sum_val',  'INTEGER DEFAULT 0'),
            ('pred_sum_zone', 'TEXT    DEFAULT ""'),
            ('pred_hot_sums', 'TEXT    DEFAULT "[]"'),
        ]:
            if col not in existing:
                conn.execute(f'ALTER TABLE rounds ADD COLUMN {col} {defn}')
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
    """
    Extract 4 digits (0-9 each) from API value string.
    Handles formats: '3456', '3,4,5,6', '3 4 5 6'.
    Returns list of 4 ints or None.
    """
    s = str(value_str).strip()
    # Try comma or space separated
    parts = [p.strip() for p in s.replace(',', ' ').split() if p.strip().isdigit()]
    if len(parts) == 4:
        nums = [int(p) for p in parts]
        if all(0 <= n <= 9 for n in nums):
            return nums
    # Try concatenated 4-digit string
    digits = [c for c in s if c.isdigit()]
    if len(digits) == 4:
        nums = [int(d) for d in digits]
        if all(0 <= n <= 9 for n in nums):
            return nums
    return None

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
        issue = item.get('issue')
        nums  = parse_value(item.get('value', ''))
        if issue is not None and nums is not None:
            records.append((int(issue), nums))
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

        print("[1] Opening FST1M site...")
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

        print("[2] Navigating to FST1M history page...")
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
                'SELECT id,n1,n2,n3,n4,total,big_small,odd_even,'
                'result,pred_bs,confidence,bet,'
                'pred_oe,conf_oe,result_oe,bet_oe,pred_sum_val,pred_sum_zone,pred_hot_sums '
                'FROM rounds ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        tmp = CSV_PATH + '.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['id','n1','n2','n3','n4','total','big_small','odd_even',
                        'result','pred_bs','confidence','bet',
                        'pred_oe','conf_oe','result_oe','bet_oe','pred_sum_val','pred_sum_zone','pred_hot_sums'])
            w.writerows(rows)
        os.replace(tmp, CSV_PATH)
    except Exception as e:
        print(f'  [CSV backup] {e}')


# ── Fetch new records ──────────────────────────────────────────────────────────

def fetch_new_records(auth_headers):
    cutoff = last_db_id()
    new_records = []

    data1 = fetch_page(auth_headers, 1)
    if not data1:
        return []
    if data1.get('code') == 401:
        raise PermissionError("401 — auth expired")
    for iss, nums in parse_records(data1):
        if iss > cutoff:
            new_records.append((iss, nums))

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
        for iss, nums in records:
            if iss > cutoff:
                new_records.append((iss, nums)); added += 1
        if added == 0:
            break
        if p >= data.get('data', {}).get('pages', 1):
            break
        p += 1
        time.sleep(0.3)

    new_records.sort(key=lambda x: x[0])
    seen, unique = set(), []
    for iss, nums in new_records:
        if iss not in seen:
            seen.add(iss); unique.append((iss, nums))
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
    entry = pred_log.get(str(issue), {})
    result    = 'Not Predicted'
    result_oe = 'Not Predicted'
    if entry.get('pred_bs'):
        result    = 'WIN' if entry['pred_bs'] == bs else 'LOSS'
    if entry.get('pred_oe'):
        result_oe = 'WIN' if entry['pred_oe'] == oe else 'LOSS'
    return result, result_oe

def save_records(new_items):
    from collections import Counter
    pred_log = _load_pred_log()
    rows = []
    for issue, nums in new_items:
        total            = sum(nums)
        bs, oe           = classify(total)
        result, result_oe = _win_loss(issue, bs, oe, pred_log)
        entry            = pred_log.get(str(issue), {})
        pred_bs          = entry.get('pred_bs', '')
        confidence       = float(entry.get('confidence', 0) or 0)
        bet              = int(entry.get('bet', 0) or 0)
        pred_oe_val      = entry.get('pred_oe', '')
        conf_oe          = float(entry.get('conf_oe', 0) or 0)
        bet_oe           = int(entry.get('bet_oe', 0) or 0)
        pred_sum_val     = int(entry.get('pred_sum_val', 0) or 0)
        pred_sum_zone    = entry.get('pred_sum_zone', '')
        rows.append((issue, nums[0], nums[1], nums[2], nums[3],
                     total, bs, oe, result, pred_bs, confidence, bet,
                     pred_oe_val, conf_oe, result_oe, bet_oe,
                     pred_sum_val, pred_sum_zone, '[]'))
    conn = get_conn()
    try:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds'
            '(id,n1,n2,n3,n4,total,big_small,odd_even,result,pred_bs,confidence,bet,'
            'pred_oe,conf_oe,result_oe,bet_oe,pred_sum_val,pred_sum_zone,pred_hot_sums) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        conn.commit()
        # Compute pred_hot_sums: last 50 rounds including this round (inclusive).
        # Matches the Hot sums (last 50) panel which also uses ORDER BY id DESC LIMIT 50.
        for issue, _ in new_items:
            last50 = conn.execute(
                'SELECT total FROM rounds WHERE id <= ? ORDER BY id DESC LIMIT 50',
                (issue,)
            ).fetchall()
            counts = Counter(r[0] for r in last50)
            hot = [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])[:5]]
            conn.execute('UPDATE rounds SET pred_hot_sums=? WHERE id=?',
                         (json.dumps(hot), issue))
        conn.commit()
    finally:
        conn.close()
    sync_csv_backup()

def backfill_results():
    conn    = get_conn()
    updates = []
    try:
        rows = conn.execute(
            "SELECT id, big_small, odd_even, pred_bs, pred_oe FROM rounds "
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
            conn.executemany(
                'UPDATE rounds SET result=?, result_oe=? WHERE id=?', updates
            )
            conn.commit()
            print(f"  Backfilled {len(updates)} result(s).")
    finally:
        conn.close()
    if updates:
        sync_csv_backup()


# ── Prediction ─────────────────────────────────────────────────────────────────

def run_prediction():
    print(f"\n{'='*55}")
    print("  FST1M PREDICTION")
    print(f"{'='*55}\n")
    subprocess.run([sys.executable, PREDICT_PY])
    backfill_results()


def _push_to_remote(first_id):
    try:
        from remote_push import push_game
        conn = get_conn()
        rows = conn.execute(
            'SELECT id,n1,n2,n3,n4,total,big_small,odd_even,result,pred_bs,confidence,bet,'
            'pred_oe,conf_oe,result_oe,bet_oe,pred_sum_val,pred_sum_zone,pred_hot_sums '
            'FROM rounds WHERE id >= ?', (first_id,)
        ).fetchall()
        conn.close()
        records = [
            {'id': r[0], 'n1': r[1], 'n2': r[2], 'n3': r[3], 'n4': r[4],
             'total': r[5], 'big_small': r[6], 'odd_even': r[7],
             'result': r[8], 'pred_bs': r[9], 'confidence': r[10], 'bet': r[11],
             'pred_oe': r[12], 'conf_oe': r[13], 'result_oe': r[14], 'bet_oe': r[15],
             'pred_sum_val': r[16], 'pred_sum_zone': r[17], 'pred_hot_sums': r[18]}
            for r in rows
        ]
        snap_path = os.path.join(_DATA_DIR, 'latest_prediction_fst1m.json')
        push_game('fst1m', records, snap_path, LOG_PATH)
    except Exception as e:
        print(f'  [PUSH] {e}')


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    check_license()
    print("=" * 55)
    print(f"  FST1M Live Collector  (Ctrl+C to stop)")
    print("=" * 55)

    ensure_table()
    with get_conn() as conn:
        count = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
    print(f"  DB has {count} existing records.\n")

    auth_headers, seed = capture_auth_headers()
    if not auth_headers:
        print("Could not capture auth headers. Exiting.")
        return

    # Save any seeded records from initial page load
    if seed:
        save_records(seed)
        print(f"  Seeded {len(seed)} records from initial load.")
        run_prediction()

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
                print(f"  +{len(new_items)} new record(s) saved. Latest: {last}")
                for issue, nums in new_items:
                    total    = sum(nums)
                    bs, oe   = classify(total)
                    nums_str = ' '.join(str(n) for n in nums)
                    print(f"  {issue}  [{nums_str}]  sum={total}  {bs}/{oe}")

                while True:
                    extra = fetch_new_records(auth_headers)
                    if not extra:
                        break
                    save_records(extra)
                    for issue, nums in extra:
                        total  = sum(nums)
                        bs, oe = classify(total)
                        print(f"  {issue}  sum={total}  {bs}/{oe}  (catch-up)")

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
