"""
DS3M Web Dashboard — Flask backend.
Run: python server.py
"""
import os, json, sqlite3, subprocess, sys, threading, secrets, hashlib
from flask import (Flask, jsonify, send_from_directory, request,
                   session, redirect, url_for, render_template_string)
from werkzeug.security import check_password_hash

def _load_env():
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env      = _load_env()
_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')
SNAP_PATH   = os.path.join(_DATA_DIR, 'latest_prediction.json')
LOG_PATH    = os.path.join(_DATA_DIR, 'pred_log.json')
DB_PATH     = os.path.join(_DATA_DIR, _env.get('DB_FILE', 'ds3m.db'))
CONFIG_PATH = os.path.join(_DATA_DIR, 'config.json')
CREDS_PATH  = os.path.join(_DATA_DIR, 'credentials.json')
PORT        = int(_env.get('PORT', 5050))

app = Flask(__name__, static_folder=os.path.join(_BASE, 'static'))
app.secret_key = _env.get('SECRET_KEY') or secrets.token_hex(32)

# ── helpers ────────────────────────────────────────────────────────────────────

def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8-sig') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def _no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response

def _cipher_key():
    secret = _env.get('SECRET_KEY', 'ds3m-default-key')
    return hashlib.sha256(secret.encode()).digest()

def _decrypt(hex_text):
    key = _cipher_key()
    data = bytes.fromhex(hex_text)
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, key_stream)).decode('utf-8')

def _load_credentials():
    creds = _load_json(CREDS_PATH)
    if not creds or 'username' not in creds or 'password_hash' not in creds:
        return None
    try:
        return {
            'username':      _decrypt(creds['username']),
            'password_hash': creds['password_hash'],
        }
    except Exception:
        return None

_predict_lock = threading.Lock()

def _run_predict():
    predict_py = os.path.join(_BASE, 'predict.py')
    if _predict_lock.acquire(blocking=False):
        try:
            subprocess.run([sys.executable, predict_py], capture_output=True)
        finally:
            _predict_lock.release()

# ── auth guard ─────────────────────────────────────────────────────────────────

_PUBLIC_PATHS = {'/login', '/logout'}

@app.before_request
def _require_login():
    if request.path in _PUBLIC_PATHS:
        return
    if not session.get('logged_in'):
        return redirect(url_for('login'))

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DS3M — Login</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0;
          --muted:#94a3b8; --accent:#3b82f6; --red:#ef4444; --green:#22c55e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
         min-height:100vh; display:flex; align-items:center; justify-content:center; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:36px 40px; width:100%; max-width:380px; }
  h1 { font-size:20px; font-weight:700; margin-bottom:6px; letter-spacing:.02em; }
  .sub { font-size:12px; color:var(--muted); margin-bottom:28px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:.06em; }
  input { width:100%; background:#0f172a; border:1px solid var(--border); border-radius:6px;
          color:var(--text); font-size:14px; padding:10px 12px; outline:none; margin-bottom:16px; }
  input:focus { border-color:var(--accent); }
  button { width:100%; background:var(--accent); color:#fff; border:none; border-radius:6px;
           font-size:14px; font-weight:600; padding:11px; cursor:pointer; margin-top:4px; }
  button:hover { background:#2563eb; }
  .error { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3); border-radius:6px;
           color:var(--red); font-size:13px; padding:10px 14px; margin-bottom:18px; }
</style>
</head>
<body>
<div class="card">
  <h1>DS3M Predictor</h1>
  <div class="sub">Sign in to continue</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="/login">
    <label>Username</label>
    <input type="text" name="username" autocomplete="username" autofocus required>
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
  </form>
</div>
</body>
</html>"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        creds = _load_credentials()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if (creds and
                username == creds['username'] and
                check_password_hash(creds['password_hash'], password)):
            session['logged_in'] = True
            session.permanent = False
            return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template_string(_LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(os.path.join(_BASE, 'static'), 'index.html')

@app.route('/api/prediction')
def api_prediction():
    snap = _load_json(SNAP_PATH)
    if snap is None:
        return jsonify({'error': 'No prediction data yet. Run collector.py first.'}), 404
    return _no_cache(jsonify(snap))


@app.route('/api/history')
def api_history():
    """Return last 30 predicted rounds from the DB (pred_oe populated), plus pending from pred_log."""
    entries = []
    # Rows from DB that have predictions
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5)
            conn.execute('PRAGMA journal_mode=WAL')
            rows = conn.execute(
                "SELECT id, pred_oe, confidence, bet, "
                "disc1||disc2||disc3||disc4 AS actual, oe, result "
                "FROM rounds ORDER BY id DESC LIMIT 30"
            ).fetchall()
            conn.close()
            for r in rows:
                entries.append({
                    'round_id':   str(r[0]),
                    'pred_oe':    r[1],
                    'p_oe_odd':   None,
                    'confidence': r[2],
                    'bet_label':  r[3],
                    'actual':     r[4],
                    'oe':         r[5],
                    'result':     r[6],
                })
        except Exception:
            pass
    # Enrich DB entries with band_wr from pred_log (stored at prediction time)
    log = _load_json(LOG_PATH)
    if log:
        for e in entries:
            le = log.get(e['round_id'])
            if le:
                e['band_wr']      = le.get('band_wr')
                e['band_n']       = le.get('band_n')
                e['pat_even_lbl'] = le.get('pat_even_lbl')
                e['pat_even_pct'] = le.get('pat_even_pct')
                e['pat_odd_lbl']  = le.get('pat_odd_lbl')
                e['pat_odd_pct']  = le.get('pat_odd_pct')

        # Only enrich DB entries with band_wr/band_n — no pending/future rows added
    return _no_cache(jsonify(entries[:30]))

@app.route('/api/status')
def api_status():
    """Lightweight endpoint — returns mtime of latest_prediction.json so the UI can detect updates."""
    mtime = os.path.getmtime(SNAP_PATH) if os.path.exists(SNAP_PATH) else 0
    return _no_cache(jsonify({'mtime': mtime}))

@app.route('/api/data')
def api_data():
    VALID_COLS = {'id', 'pattern', 'oe', 'flag', 'result', 'pred_oe', 'confidence', 'bet'}
    page      = max(1, int(request.args.get('page', 1)))
    page_size = min(200, max(10, int(request.args.get('size', 50))))
    search    = request.args.get('search', '').strip()
    sort_col  = request.args.get('sort', 'id')
    sort_dir  = 'ASC' if request.args.get('dir', 'desc').lower() == 'asc' else 'DESC'
    if sort_col not in VALID_COLS:
        sort_col = 'id'

    where, params = '', []
    if search:
        where  = ('WHERE CAST(id AS TEXT) LIKE ? OR pattern LIKE ? OR oe LIKE ? '
                  'OR result LIKE ? OR pred_oe LIKE ? OR bet LIKE ?')
        params = [f'%{search}%'] * 6

    if not os.path.exists(DB_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        total  = conn.execute(f'SELECT COUNT(*) FROM rounds {where}', params).fetchone()[0]
        offset = (page - 1) * page_size
        rows   = conn.execute(
            f'SELECT id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence,bet '
            f'FROM rounds {where} ORDER BY {sort_col} {sort_dir} LIMIT ? OFFSET ?',
            params + [page_size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    pages = max(1, (total + page_size - 1) // page_size)
    return _no_cache(jsonify({
        'total':     total,
        'page':      page,
        'page_size': page_size,
        'pages':     pages,
        'rows': [
            {'id': r[0], 'disc1': r[1], 'disc2': r[2], 'disc3': r[3], 'disc4': r[4],
             'pattern': r[5], 'flag': r[6], 'oe': r[7], 'result': r[8],
             'pred_oe': r[9], 'confidence': r[10], 'bet': r[11]}
            for r in rows
        ]
    }))

@app.route('/api/stats')
def api_stats():
    """Win rate stats: overall last 100 predicted rounds, BET-only rounds, and total predicted count."""
    if not os.path.exists(DB_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')

        total_pred = conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE pred_oe != '' AND result IN ('WIN','LOSS')"
        ).fetchone()[0]

        last100 = conn.execute(
            "SELECT result FROM rounds WHERE pred_oe != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        w100 = sum(1 for r in last100 if r[0] == 'WIN')
        n100 = len(last100)

        last200 = conn.execute(
            "SELECT result FROM rounds WHERE pred_oe != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        w200 = sum(1 for r in last200 if r[0] == 'WIN')
        n200 = len(last200)

        bet_rows = conn.execute(
            "SELECT result FROM rounds WHERE bet=1 AND result IN ('WIN','LOSS')"
        ).fetchall()
        wb = sum(1 for r in bet_rows if r[0] == 'WIN')
        nb = len(bet_rows)

        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100': {'win': w100, 'n': n100, 'wr': round(w100 / n100, 4) if n100 else None},
        'last200': {'win': w200, 'n': n200, 'wr': round(w200 / n200, 4) if n200 else None},
        'bet_only': {'win': wb,  'n': nb,  'wr': round(wb  / nb,  4) if nb  else None},
    }))

@app.route('/api/config', methods=['GET'])
def api_config_get():
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {'training_enabled': True}
    return _no_cache(jsonify(cfg))

@app.route('/api/config', methods=['POST'])
def api_config_set():
    body = request.get_json(silent=True) or {}
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.update({k: v for k, v in body.items() if k in ('training_enabled', 'bankroll', 'kelly_fraction')})
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f)
    return _no_cache(jsonify(cfg))

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    t = threading.Thread(target=_run_predict, daemon=True)
    t.start()
    return jsonify({'status': 'running'})

def _ensure_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'training_enabled': True}, f)

if __name__ == '__main__':
    _ensure_config()
    if not _load_credentials():
        print("  ⚠  No credentials found. Run: python create_credentials.py")
    print(f"  Starting DS3M dashboard on http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
