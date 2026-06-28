"""
DS3M Web Dashboard — Flask backend.
Run: python server.py
"""
import os, json, sqlite3, subprocess, sys, threading, secrets, hashlib
from flask import (Flask, jsonify, send_from_directory, request,
                   session, redirect, url_for, render_template_string)
from werkzeug.security import check_password_hash
from license_manager import validate_key, _load_stored_key, _save_key

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
SNAP_PATH      = os.path.join(_DATA_DIR, 'latest_prediction.json')
LOG_PATH       = os.path.join(_DATA_DIR, 'pred_log.json')
DB_PATH        = os.path.join(_DATA_DIR, _env.get('DB_FILE', 'ds3m.db'))
CONFIG_PATH    = os.path.join(_DATA_DIR, 'config.json')
CREDS_PATH     = os.path.join(_DATA_DIR, 'credentials.json')
PORT           = int(_env.get('PORT', 5050))
# New model paths
SNAP_NEW_PATH  = os.path.join(_DATA_DIR, 'latest_prediction_new.json')
LOG_NEW_PATH   = os.path.join(_DATA_DIR, 'pred_log_new.json')
DB_NEW_PATH    = os.path.join(_DATA_DIR, 'ds3m_new.db')
# Color game paths
SNAP_CG_PATH   = os.path.join(_DATA_DIR, 'latest_prediction_cg.json')
LOG_CG_PATH    = os.path.join(_DATA_DIR, 'pred_log_cg.json')
DB_CG_PATH     = os.path.join(_DATA_DIR, _env.get('DB_FILE2', 'rg3m.db'))
# DS1M (1-minute) paths
SNAP_DS1M_PATH  = os.path.join(_DATA_DIR, 'latest_prediction_ds1m.json')
LOG_DS1M_PATH   = os.path.join(_DATA_DIR, 'pred_log_ds1m.json')
DB_DS1M_PATH    = os.path.join(_DATA_DIR, _env.get('DB_FILE_DS1M', 'ds1m.db'))
# FST1M (1-minute 4-Star Lottery) paths
SNAP_FST1M_PATH = os.path.join(_DATA_DIR, 'latest_prediction_fst1m.json')
LOG_FST1M_PATH  = os.path.join(_DATA_DIR, 'pred_log_fst1m.json')
DB_FST1M_PATH   = os.path.join(_DATA_DIR, _env.get('DB_FILE_FST1M', 'fst1m.db'))
# BLK3M (3-minute Blocks) paths
SNAP_BLK3M_PATH = os.path.join(_DATA_DIR, 'latest_prediction_blk3m.json')
LOG_BLK3M_PATH  = os.path.join(_DATA_DIR, 'pred_log_blk3m.json')
DB_BLK3M_PATH   = os.path.join(_DATA_DIR, _env.get('DB_FILE_BLK3M', 'blk3m.db'))

_BLK3M_BIG_HEX = set('89abcdef')
_BLK3M_ODD_HEX = set('13579bdf')

def _blk3m_classify(value_str):
    """Recompute big_small/odd_even from the stored hash — always authoritative."""
    s = str(value_str).strip().lower()
    for c in reversed(s):
        if c in '0123456789abcdef':
            bs = 'BIG'  if c in _BLK3M_BIG_HEX else 'SMALL'
            oe = 'ODD'  if c in _BLK3M_ODD_HEX else 'EVEN'
            return bs, oe
    return None, None
# Access log
ACCESS_LOG_DB_PATH = os.path.join(_DATA_DIR, 'access_log.db')

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

def _init_access_log_db():
    conn = sqlite3.connect(ACCESS_LOG_DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ip_access_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            code      TEXT NOT NULL,
            ip        TEXT NOT NULL,
            user_agent TEXT,
            timestamp TEXT NOT NULL,
            action    TEXT NOT NULL DEFAULT "activate"
        )
    ''')
    conn.commit()
    conn.close()

_init_access_log_db()

def _log_ip_access(code: str, ip: str, user_agent: str, action: str = 'activate'):
    import datetime as _dt
    try:
        conn = sqlite3.connect(ACCESS_LOG_DB_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute(
            'INSERT INTO ip_access_log (code, ip, user_agent, timestamp, action) VALUES (?,?,?,?,?)',
            (code.upper(), ip, user_agent or '', _dt.datetime.now().isoformat(timespec='seconds'), action)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

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

_predict_lock       = threading.Lock()
_predict_new_lock   = threading.Lock()
_predict_ds1m_lock  = threading.Lock()
_predict_fst1m_lock = threading.Lock()

def _run_predict():
    predict_py = os.path.join(_BASE, 'predict.py')
    if _predict_lock.acquire(blocking=False):
        try:
            subprocess.run([sys.executable, predict_py], capture_output=True)
        finally:
            _predict_lock.release()

def _run_predict_new():
    predict_new_py = os.path.join(_BASE, 'predict_new.py')
    if _predict_new_lock.acquire(blocking=False):
        try:
            subprocess.run([sys.executable, predict_new_py], capture_output=True)
        finally:
            _predict_new_lock.release()

def _run_predict_ds1m():
    predict_ds1m_py = os.path.join(_BASE, 'predict_ds1m.py')
    if _predict_ds1m_lock.acquire(blocking=False):
        try:
            subprocess.run([sys.executable, predict_ds1m_py], capture_output=True)
        finally:
            _predict_ds1m_lock.release()

def _run_predict_fst1m():
    predict_fst1m_py = os.path.join(_BASE, 'predict_fst1m.py')
    if _predict_fst1m_lock.acquire(blocking=False):
        try:
            subprocess.run([sys.executable, predict_fst1m_py], capture_output=True)
        finally:
            _predict_fst1m_lock.release()

# ── auth guard ─────────────────────────────────────────────────────────────────

_PUBLIC_PATHS = {'/login', '/logout', '/activate', '/setup'}

def _push_token_valid():
    auth  = request.headers.get('Authorization', '')
    token = auth[7:] if auth.startswith('Bearer ') else ''
    if not token:
        return False
    valid, _ = validate_key(token)
    return valid

@app.before_request
def _require_login():
    if request.path in _PUBLIC_PATHS:
        return
    if request.path.startswith('/api/push/'):
        return  # token-authenticated in the endpoint itself
    if request.path.startswith('/admin/') or request.path.startswith('/api/admin/'):
        if not session.get('is_admin'):
            return redirect(url_for('login'))
        return
    if not session.get('logged_in'):
        return redirect(url_for('activate'))

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
            session['is_admin']  = True
            session.permanent = False
            return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template_string(_LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('activate'))

_SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DS3M — Admin Setup</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0;
          --muted:#94a3b8; --accent:#3b82f6; --red:#ef4444; --green:#22c55e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
         min-height:100vh; display:flex; align-items:center; justify-content:center; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:36px 40px; width:100%; max-width:400px; }
  h1 { font-size:20px; font-weight:700; margin-bottom:6px; }
  .sub { font-size:12px; color:var(--muted); margin-bottom:28px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px;
          text-transform:uppercase; letter-spacing:.06em; }
  input { width:100%; background:#0f172a; border:1px solid var(--border); border-radius:6px;
          color:var(--text); font-size:14px; padding:10px 12px; outline:none; margin-bottom:16px; }
  input:focus { border-color:var(--accent); }
  button { width:100%; background:var(--accent); color:#fff; border:none; border-radius:6px;
           font-size:14px; font-weight:600; padding:11px; cursor:pointer; margin-top:4px; }
  button:hover { background:#2563eb; }
  .error { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3); border-radius:6px;
           color:var(--red); font-size:13px; padding:10px 14px; margin-bottom:18px; }
  .done  { background:rgba(34,197,94,.1); border:1px solid rgba(34,197,94,.3); border-radius:6px;
           color:var(--green); font-size:13px; padding:10px 14px; margin-bottom:18px; }
</style>
</head>
<body>
<div class="card">
  <h1>Admin Setup</h1>
  <div class="sub">Create the admin login credentials. This page is only available once.</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  {% if done %}<div class="done">Credentials saved. <a href="/login" style="color:inherit;font-weight:600">Sign in →</a></div>
  {% else %}
  <form method="post" action="/setup">
    <label>Username</label>
    <input type="text" name="username" autocomplete="off" autofocus required>
    <label>Password</label>
    <input type="password" name="password" required>
    <label>Confirm Password</label>
    <input type="password" name="confirm" required>
    <button type="submit">Create Admin Account</button>
  </form>
  {% endif %}
</div>
</body>
</html>"""

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if os.path.exists(CREDS_PATH):
        return redirect(url_for('login'))
    error = None
    done  = False
    if request.method == 'POST':
        from werkzeug.security import generate_password_hash
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')
        if not username:
            error = 'Username cannot be empty.'
        elif not password:
            error = 'Password cannot be empty.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            key = _cipher_key()
            def _encrypt(text):
                data = text.encode('utf-8')
                ks   = (key * (len(data) // len(key) + 1))[:len(data)]
                return bytes(a ^ b for a, b in zip(data, ks)).hex()
            import json as _json
            with open(CREDS_PATH, 'w') as f:
                _json.dump({
                    'username':      _encrypt(username),
                    'password_hash': generate_password_hash(password, method='pbkdf2:sha256:600000'),
                }, f, indent=2)
            done = True
    return render_template_string(_SETUP_HTML, error=error, done=done)

_ACTIVATION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DS3M — User Access</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0;
          --muted:#94a3b8; --accent:#3b82f6; --red:#ef4444; --green:#22c55e; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
         min-height:100vh; display:flex; align-items:center; justify-content:center; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:36px 40px; width:100%; max-width:420px; }
  h1 { font-size:20px; font-weight:700; margin-bottom:6px; letter-spacing:.02em; }
  .sub { font-size:12px; color:var(--muted); margin-bottom:28px; }
  label { display:block; font-size:12px; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:.06em; }
  input { width:100%; background:#0f172a; border:1px solid var(--border); border-radius:6px;
          color:var(--text); font-size:14px; padding:10px 12px; outline:none; margin-bottom:16px;
          font-family:monospace; letter-spacing:.06em; }
  input:focus { border-color:var(--accent); }
  input::placeholder { letter-spacing:normal; font-family:inherit; color:#475569; }
  button { width:100%; background:#0f4c3a; color:#4ade80; border:1px solid #166534;
           border-radius:6px; font-size:14px; font-weight:600; padding:11px; cursor:pointer; margin-top:4px; }
  button:hover { background:#166534; }
  .error { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3); border-radius:6px;
           color:var(--red); font-size:13px; padding:10px 14px; margin-bottom:18px; }
  .hint { font-size:11px; color:var(--muted); margin-top:18px; text-align:center; }
</style>
</head>
<body>
<div class="card">
  <h1>DS3M Predictor</h1>
  <div class="sub">Enter your activation code to access the dashboard</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="/activate">
    <label>Activation Code</label>
    <input type="text" name="code" placeholder="FXPRO-XXXXXXXXXX-XXXXXXXX"
           autocomplete="off" autofocus spellcheck="false" required>
    <button type="submit">Activate</button>
  </form>
  <div class="hint">Contact your administrator if you don't have an activation code.</div>
</div>
</body>
</html>"""

@app.route('/activate', methods=['GET', 'POST'])
def activate():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        valid, message = validate_key(code)
        if valid:
            _save_key(code)
            ip = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
            _log_ip_access(code, ip, request.user_agent.string, 'activate')
            session['logged_in']   = True
            session['is_admin']    = False
            session['license_key'] = code.upper()
            session.permanent = False
            return redirect(url_for('index'))
        error = message
    return render_template_string(_ACTIVATION_HTML, error=error)

# ── admin: IP access log ───────────────────────────────────────────────────────

_IP_LOG_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IP Access Log — Admin</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --border:#334155; --text:#e2e8f0;
          --muted:#94a3b8; --accent:#3b82f6; --red:#ef4444; --green:#22c55e;
          --yellow:#f59e0b; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
         min-height:100vh; padding:32px 24px; }
  h1 { font-size:18px; font-weight:700; margin-bottom:4px; }
  .sub { font-size:12px; color:var(--muted); margin-bottom:24px; }
  .toolbar { display:flex; gap:12px; align-items:center; margin-bottom:18px; flex-wrap:wrap; }
  .toolbar input { background:#1e293b; border:1px solid var(--border); border-radius:6px;
                   color:var(--text); font-size:13px; padding:7px 12px; outline:none; width:260px; }
  .toolbar input:focus { border-color:var(--accent); }
  .btn { background:var(--accent); color:#fff; border:none; border-radius:6px;
         font-size:12px; font-weight:600; padding:7px 14px; cursor:pointer; text-decoration:none;
         display:inline-block; }
  .btn-sm { padding:4px 10px; font-size:11px; }
  .btn:hover { background:#2563eb; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; font-size:11px; color:var(--muted); text-transform:uppercase;
       letter-spacing:.06em; padding:8px 12px; border-bottom:1px solid var(--border); }
  td { padding:9px 12px; border-bottom:1px solid #1e293b; vertical-align:top; }
  tr:hover td { background:#1e293b44; }
  .badge { display:inline-block; border-radius:4px; font-size:11px; font-weight:600;
           padding:2px 8px; }
  .ok   { background:rgba(34,197,94,.15); color:var(--green); }
  .warn { background:rgba(245,158,11,.15); color:var(--yellow); }
  .flag { background:rgba(239,68,68,.15); color:var(--red); }
  .code { font-family:monospace; font-size:12px; letter-spacing:.04em; }
  .ip   { font-family:monospace; font-size:12px; color:#93c5fd; }
  .ua   { font-size:11px; color:var(--muted); max-width:300px; overflow:hidden;
          text-overflow:ellipsis; white-space:nowrap; }
  .ts   { font-size:11px; color:var(--muted); white-space:nowrap; }
  details summary { cursor:pointer; list-style:none; }
  details summary::marker { display:none; }
  .toggle { font-size:11px; color:var(--accent); cursor:pointer; }
  .detail-row td { background:#131f35; }
  .detail-table { width:100%; border-collapse:collapse; font-size:12px; }
  .detail-table td { padding:5px 10px; border-bottom:1px solid #1e293b22; }
  .empty { color:var(--muted); font-size:13px; padding:40px; text-align:center; }
  .summary-bar { display:flex; gap:24px; margin-bottom:20px; flex-wrap:wrap; }
  .stat { background:var(--card); border:1px solid var(--border); border-radius:8px;
          padding:12px 20px; min-width:120px; }
  .stat-val { font-size:22px; font-weight:700; }
  .stat-lbl { font-size:11px; color:var(--muted); margin-top:2px; }
</style>
</head>
<body>
<h1>IP Access Log</h1>
<div class="sub">Activation events — detect shared activation codes</div>

<div class="summary-bar">
  <div class="stat"><div class="stat-val" id="s-total">—</div><div class="stat-lbl">Total Events</div></div>
  <div class="stat"><div class="stat-val" id="s-codes">—</div><div class="stat-lbl">Unique Codes</div></div>
  <div class="stat"><div class="stat-val" id="s-ips">—</div><div class="stat-lbl">Unique IPs</div></div>
  <div class="stat"><div class="stat-val" style="color:var(--red)" id="s-flag">—</div><div class="stat-lbl">Flagged Codes</div></div>
</div>

<div class="toolbar">
  <input type="text" id="filter" placeholder="Filter by code or IP…" oninput="applyFilter()">
  <label style="font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;">
    <input type="checkbox" id="only-flag" onchange="applyFilter()"> Show flagged only
  </label>
  <a href="/" class="btn btn-sm">← Dashboard</a>
</div>

<table id="log-table">
  <thead>
    <tr>
      <th>Activation Code</th>
      <th>Unique IPs</th>
      <th>Total Hits</th>
      <th>Last Seen</th>
      <th>Status</th>
      <th></th>
    </tr>
  </thead>
  <tbody id="log-body">
    <tr><td class="empty" colspan="6">Loading…</td></tr>
  </tbody>
</table>

<script>
let _data = [];

async function load() {
  const r = await fetch('/api/admin/ip-log');
  if (!r.ok) { document.getElementById('log-body').innerHTML = '<tr><td class="empty" colspan="6">Access denied or error.</td></tr>'; return; }
  const d = await r.json();
  _data = d.codes || [];
  document.getElementById('s-total').textContent = d.total_events ?? '—';
  document.getElementById('s-codes').textContent = d.total_codes ?? '—';
  document.getElementById('s-ips').textContent   = d.total_ips   ?? '—';
  document.getElementById('s-flag').textContent  = d.flagged     ?? '—';
  applyFilter();
}

function applyFilter() {
  const q    = document.getElementById('filter').value.toLowerCase();
  const flag = document.getElementById('only-flag').checked;
  const rows = _data.filter(c => {
    if (flag && c.unique_ips <= 1) return false;
    if (q && !c.code.toLowerCase().includes(q) && !c.ips.some(i => i.ip.includes(q))) return false;
    return true;
  });
  renderRows(rows);
}

function renderRows(rows) {
  const tb = document.getElementById('log-body');
  if (!rows.length) { tb.innerHTML = '<tr><td class="empty" colspan="6">No records match.</td></tr>'; return; }
  tb.innerHTML = rows.map((c, idx) => {
    const badge = c.unique_ips >= 3 ? '<span class="badge flag">SHARED</span>'
                : c.unique_ips === 2 ? '<span class="badge warn">SUSPICIOUS</span>'
                : '<span class="badge ok">OK</span>';
    const detail = c.ips.map(e =>
      `<tr><td class="ip">${e.ip}</td><td class="ua" title="${e.ua}">${e.ua||'—'}</td><td class="ts">${e.last}</td><td style="color:var(--muted)">${e.count} hit${e.count!==1?'s':''}</td></tr>`
    ).join('');
    return `
    <tr data-idx="${idx}">
      <td class="code">${c.code}</td>
      <td>${c.unique_ips}</td>
      <td>${c.total_hits}</td>
      <td class="ts">${c.last_seen}</td>
      <td>${badge}</td>
      <td><span class="toggle" onclick="toggleDetail(${idx})">▶ Details</span></td>
    </tr>
    <tr id="detail-${idx}" class="detail-row" style="display:none">
      <td colspan="6">
        <table class="detail-table">
          <tr><th style="padding:5px 10px;color:var(--muted);font-size:11px">IP</th>
              <th style="padding:5px 10px;color:var(--muted);font-size:11px">User Agent</th>
              <th style="padding:5px 10px;color:var(--muted);font-size:11px">Last Seen</th>
              <th style="padding:5px 10px;color:var(--muted);font-size:11px">Hits</th></tr>
          ${detail}
        </table>
      </td>
    </tr>`;
  }).join('');
}

function toggleDetail(idx) {
  const row = document.getElementById('detail-' + idx);
  const tog = row.previousElementSibling.querySelector('.toggle');
  if (row.style.display === 'none') {
    row.style.display = '';
    tog.textContent = '▼ Details';
  } else {
    row.style.display = 'none';
    tog.textContent = '▶ Details';
  }
}

load();
</script>
</body>
</html>"""

@app.route('/admin/ip-log')
def admin_ip_log():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    return render_template_string(_IP_LOG_HTML)

@app.route('/api/admin/ip-log')
def api_admin_ip_log():
    if not session.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403
    try:
        conn = sqlite3.connect(ACCESS_LOG_DB_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        rows = conn.execute(
            'SELECT code, ip, user_agent, timestamp FROM ip_access_log ORDER BY timestamp DESC'
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    # Group by code
    from collections import defaultdict
    code_map = defaultdict(list)
    for code, ip, ua, ts in rows:
        code_map[code].append({'ip': ip, 'ua': ua, 'ts': ts})

    codes = []
    all_ips = set()
    flagged = 0
    for code, entries in sorted(code_map.items()):
        ip_map = defaultdict(lambda: {'count': 0, 'last': ''})
        for e in entries:
            ip_map[e['ip']]['count'] += 1
            if e['ts'] > ip_map[e['ip']]['last']:
                ip_map[e['ip']]['last'] = e['ts']
                ip_map[e['ip']]['ua']   = e['ua']
        ips_detail = sorted(
            [{'ip': ip, 'count': v['count'], 'last': v['last'], 'ua': v['ua']}
             for ip, v in ip_map.items()],
            key=lambda x: x['last'], reverse=True
        )
        all_ips.update(ip_map.keys())
        unique = len(ip_map)
        if unique >= 2:
            flagged += 1
        codes.append({
            'code':       code,
            'unique_ips': unique,
            'total_hits': len(entries),
            'last_seen':  entries[0]['ts'] if entries else '',
            'ips':        ips_detail,
        })

    codes.sort(key=lambda x: (-x['unique_ips'], x['last_seen']), reverse=False)
    codes.sort(key=lambda x: x['unique_ips'], reverse=True)

    return _no_cache(jsonify({
        'total_events': len(rows),
        'total_codes':  len(codes),
        'total_ips':    len(all_ips),
        'flagged':      flagged,
        'codes':        codes,
    }))

# ── routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    resp = send_from_directory(os.path.join(_BASE, 'static'), 'index.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp

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
    mtime     = os.path.getmtime(SNAP_PATH)     if os.path.exists(SNAP_PATH)     else 0
    mtime_new = os.path.getmtime(SNAP_NEW_PATH) if os.path.exists(SNAP_NEW_PATH) else 0
    last_id_new = 0
    snap_new = _load_json(SNAP_NEW_PATH)
    if snap_new:
        last_id_new = snap_new.get('last_id', 0) or 0
    mtime_cg    = os.path.getmtime(SNAP_CG_PATH) if os.path.exists(SNAP_CG_PATH) else 0
    last_id_ds1m = 0
    if os.path.exists(DB_DS1M_PATH):
        try:
            _c = sqlite3.connect(DB_DS1M_PATH, timeout=2)
            _r = _c.execute("SELECT MAX(id) FROM rounds WHERE pred_oe != '' AND pred_oe IS NOT NULL").fetchone()
            _c.close()
            if _r and _r[0]:
                last_id_ds1m = int(_r[0])
        except Exception:
            pass
    if last_id_ds1m == 0:
        snap_ds1m = _load_json(SNAP_DS1M_PATH)
        if snap_ds1m:
            last_id_ds1m = snap_ds1m.get('last_id', 0) or 0
    last_id_fst1m = 0
    snap_fst1m = _load_json(SNAP_FST1M_PATH)
    if snap_fst1m:
        last_id_fst1m = snap_fst1m.get('last_id', 0) or 0
    last_id_blk3m = 0
    snap_blk3m = _load_json(SNAP_BLK3M_PATH)
    if snap_blk3m:
        last_id_blk3m = snap_blk3m.get('last_id', 0) or 0
    return _no_cache(jsonify({
        'mtime':          mtime,
        'mtime_new':      mtime_new,
        'last_id_new':    last_id_new,
        'mtime_cg':       mtime_cg,
        'last_id_ds1m':   last_id_ds1m,
        'last_id_fst1m':  last_id_fst1m,
        'last_id_blk3m':  last_id_blk3m,
    }))

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
    threading.Thread(target=_run_predict,       daemon=True).start()
    threading.Thread(target=_run_predict_new,   daemon=True).start()
    threading.Thread(target=_run_predict_ds1m,  daemon=True).start()
    threading.Thread(target=_run_predict_fst1m, daemon=True).start()
    return jsonify({'status': 'running'})

@app.route('/api/new/prediction')
def api_new_prediction():
    snap = _load_json(SNAP_NEW_PATH)
    if snap is None:
        return jsonify({'error': 'No new model data yet. Run predict_new.py first.'}), 404
    return _no_cache(jsonify(snap))

@app.route('/api/new/history')
def api_new_history():
    entries = []
    if os.path.exists(DB_NEW_PATH):
        try:
            conn = sqlite3.connect(DB_NEW_PATH, timeout=5)
            conn.execute('PRAGMA journal_mode=WAL')
            rows = conn.execute(
                "SELECT id, pred_oe, confidence, "
                "disc1||disc2||disc3||disc4 AS actual, oe, result "
                "FROM rounds WHERE pred_oe != '' ORDER BY id DESC LIMIT 30"
            ).fetchall()
            conn.close()
            for r in rows:
                entries.append({
                    'round_id':   str(r[0]),
                    'pred_oe':    r[1],
                    'confidence': r[2],
                    'actual':     r[3],
                    'oe':         r[4],
                    'result':     r[5],
                })
        except Exception:
            pass
    log = _load_json(LOG_NEW_PATH)
    if log:
        for e in entries:
            le = log.get(e['round_id'])
            if le:
                e['signals'] = le.get('signals', {})
    return _no_cache(jsonify(entries[:30]))

@app.route('/api/new/data')
def api_new_data():
    page      = max(1, int(request.args.get('page', 1)))
    page_size = min(200, max(10, int(request.args.get('size', 50))))
    search    = request.args.get('search', '').strip()
    if not os.path.exists(DB_NEW_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_NEW_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        base   = "pred_oe != ''"
        params = []
        if search:
            s_up = search.upper()
            if s_up == 'BET':
                base += " AND confidence >= 0.65"
            elif s_up == 'SKIP':
                base += " AND confidence >= 0.60 AND confidence < 0.65"
            elif s_up == 'AVOID':
                base += " AND confidence > 0 AND confidence < 0.60"
            else:
                base += (" AND (CAST(id AS TEXT) LIKE ? OR pred_oe LIKE ?"
                         " OR oe LIKE ? OR result LIKE ?)")
                params = [f'%{search}%'] * 4
        total  = conn.execute(f"SELECT COUNT(*) FROM rounds WHERE {base}", params).fetchone()[0]
        offset = (page - 1) * page_size
        rows   = conn.execute(
            f"SELECT id, pred_oe, confidence, disc1||disc2||disc3||disc4, oe, result "
            f"FROM rounds WHERE {base} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500
    pages   = max(1, (total + page_size - 1) // page_size)
    log     = _load_json(LOG_NEW_PATH) or {}
    entries = []
    for r in rows:
        e = {'round_id': str(r[0]), 'pred_oe': r[1], 'confidence': r[2],
             'actual': r[3], 'oe': r[4], 'result': r[5]}
        le = log.get(str(r[0]))
        if le:
            e['signals'] = le.get('signals', {})
        entries.append(e)
    return _no_cache(jsonify({'total': total, 'page': page, 'pages': pages, 'rows': entries}))

@app.route('/api/new/stats')
def api_new_stats():
    if not os.path.exists(DB_NEW_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_NEW_PATH, timeout=5)
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

        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100': {'win': w100, 'n': n100, 'wr': round(w100 / n100, 4) if n100 else None},
        'last200': {'win': w200, 'n': n200, 'wr': round(w200 / n200, 4) if n200 else None},
        'bet_only': {'win': 0, 'n': 0, 'wr': None},
    }))

# ── DS1M API ───────────────────────────────────────────────────────────────────

@app.route('/api/ds1m/prediction')
def api_ds1m_prediction():
    snap = _load_json(SNAP_DS1M_PATH)
    if snap is None:
        return jsonify({'error': 'No DS1M data yet. Run predict_ds1m.py first.'}), 404
    return _no_cache(jsonify(snap))

@app.route('/api/ds1m/data')
def api_ds1m_data():
    page      = max(1, int(request.args.get('page', 1)))
    page_size = min(200, max(10, int(request.args.get('size', 50))))
    search    = request.args.get('search', '').strip()
    if not os.path.exists(DB_DS1M_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_DS1M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        base   = "pred_oe != ''"
        params = []
        if search:
            s_up = search.upper()
            if s_up == 'BET':
                base += " AND confidence >= 0.65"
            elif s_up == 'SKIP':
                base += " AND confidence >= 0.60 AND confidence < 0.65"
            elif s_up == 'AVOID':
                base += " AND confidence > 0 AND confidence < 0.60"
            else:
                base += (" AND (CAST(id AS TEXT) LIKE ? OR pred_oe LIKE ?"
                         " OR oe LIKE ? OR result LIKE ?)")
                params = [f'%{search}%'] * 4
        total  = conn.execute(f"SELECT COUNT(*) FROM rounds WHERE {base}", params).fetchone()[0]
        offset = (page - 1) * page_size
        rows   = conn.execute(
            f"SELECT id, pred_oe, confidence, disc1||disc2||disc3||disc4, oe, result "
            f"FROM rounds WHERE {base} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500
    pages   = max(1, (total + page_size - 1) // page_size)
    log     = _load_json(LOG_DS1M_PATH) or {}
    entries = []
    for r in rows:
        e = {'round_id': str(r[0]), 'pred_oe': r[1], 'confidence': r[2],
             'actual': r[3], 'oe': r[4], 'result': r[5]}
        le = log.get(str(r[0]))
        if le:
            e['signals'] = le.get('signals', {})
        entries.append(e)
    return _no_cache(jsonify({'total': total, 'page': page, 'pages': pages, 'rows': entries}))

@app.route('/api/ds1m/stats')
def api_ds1m_stats():
    if not os.path.exists(DB_DS1M_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_DS1M_PATH, timeout=5)
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
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500
    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100': {'win': w100, 'n': n100, 'wr': round(w100 / n100, 4) if n100 else None},
        'last200': {'win': w200, 'n': n200, 'wr': round(w200 / n200, 4) if n200 else None},
    }))

# ── DS1M gap analysis ──────────────────────────────────────────────────────────

@app.route('/api/ds1m/gap-analysis')
def api_ds1m_gap_analysis():
    if not os.path.exists(DB_DS1M_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_DS1M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        rows = conn.execute(
            'SELECT id, disc1||disc2||disc3||disc4 FROM rounds ORDER BY id'
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    if not rows:
        return _no_cache(jsonify({}))

    total   = len(rows)
    last_id = rows[-1][0]

    def _analyze(target):
        import statistics as _s
        positions  = [i for i, (_, pat) in enumerate(rows) if pat == target]
        round_ids  = [rows[i][0] for i in positions]

        gaps = [positions[k+1] - positions[k] - 1 for k in range(len(positions) - 1)]
        last10_gaps = gaps[-10:]
        last10_ids  = round_ids[-11:]   # 11 ids → 10 gaps

        mean_gap = round(sum(last10_gaps) / len(last10_gaps), 1) if last10_gaps else None
        std_gap  = round(_s.stdev(last10_gaps), 1) if len(last10_gaps) >= 2 else 0.0
        min_gap  = min(last10_gaps) if last10_gaps else None
        max_gap  = max(last10_gaps) if last10_gaps else None

        current_gap  = (total - 1 - positions[-1]) if positions else None
        last_occ_id  = round_ids[-1] if round_ids else None

        p_round  = len(positions) / total if total > 0 else 0.0625
        p_next5  = round((1 - (1 - p_round) ** 5)  * 100, 1)
        p_next10 = round((1 - (1 - p_round) ** 10) * 100, 1)

        # Gap trend: compare first-half vs second-half of last10
        trend = None
        if len(last10_gaps) >= 4:
            mid  = len(last10_gaps) // 2
            avg1 = sum(last10_gaps[:mid]) / mid
            avg2 = sum(last10_gaps[mid:]) / (len(last10_gaps) - mid)
            if avg2 < avg1 * 0.85:
                trend = 'shortening'
            elif avg2 > avg1 * 1.15:
                trend = 'lengthening'
            else:
                trend = 'stable'

        overdue  = (current_gap is not None and mean_gap is not None and current_gap >= mean_gap)
        due_ratio = round(current_gap / mean_gap, 2) if (current_gap is not None and mean_gap) else None

        return {
            'total_occurrences': len(positions),
            'last10_gaps':       last10_gaps,
            'last10_ids':        last10_ids,
            'mean_gap':          mean_gap,
            'std_gap':           std_gap,
            'min_gap':           min_gap,
            'max_gap':           max_gap,
            'current_gap':       current_gap,
            'last_occ_id':       last_occ_id,
            'p_per_round_pct':   round(p_round * 100, 2),
            'p_next5':           p_next5,
            'p_next10':          p_next10,
            'overdue':           overdue,
            'due_ratio':         due_ratio,
            'trend':             trend,
        }

    # Combined WWWW + RRRR analysis (both treated as one bet)
    def _analyze_combined():
        import statistics as _s
        combined = sorted(
            [(i, pat) for i, (_, pat) in enumerate(rows) if pat in ('WWWW', 'RRRR')],
            key=lambda x: x[0]
        )
        positions = [x[0] for x in combined]
        labels    = [x[1] for x in combined]   # 'WWWW' or 'RRRR'
        round_ids = [rows[i][0] for i in positions]

        gaps = [positions[k+1] - positions[k] - 1 for k in range(len(positions) - 1)]
        gap_labels = [labels[k+1] for k in range(len(labels) - 1)]  # which pattern ended each gap
        last10_gaps   = gaps[-10:]
        last10_labels = gap_labels[-10:]
        last10_ids    = round_ids[-11:]

        mean_gap = round(sum(last10_gaps) / len(last10_gaps), 1) if last10_gaps else None
        std_gap  = round(_s.stdev(last10_gaps), 1) if len(last10_gaps) >= 2 else 0.0
        min_gap  = min(last10_gaps) if last10_gaps else None
        max_gap  = max(last10_gaps) if last10_gaps else None

        current_gap = (total - 1 - positions[-1]) if positions else None
        last_occ_id = round_ids[-1] if round_ids else None
        last_pat    = labels[-1] if labels else None

        p_round  = len(positions) / total if total > 0 else 0.125

        trend = None
        if len(last10_gaps) >= 4:
            mid  = len(last10_gaps) // 2
            avg1 = sum(last10_gaps[:mid]) / mid
            avg2 = sum(last10_gaps[mid:]) / (len(last10_gaps) - mid)
            if avg2 < avg1 * 0.85:   trend = 'shortening'
            elif avg2 > avg1 * 1.15: trend = 'lengthening'
            else:                    trend = 'stable'

        overdue   = (current_gap is not None and mean_gap is not None and current_gap >= mean_gap)
        due_ratio = round(current_gap / mean_gap, 2) if (current_gap is not None and mean_gap) else None

        # ── Pattern-based probability ─────────────────────────────────────────
        # Use the last 5 WWWW/RRRR occurrences as a fingerprint pattern,
        # then scan full history for the same pattern and measure how often
        # the NEXT event appeared within 5 / 10 rounds.
        LOOKBACK = 5
        last5_seq   = ['W' if l == 'WWWW' else 'R' for l in labels[-LOOKBACK:]]
        last5_gaps  = gaps[-(LOOKBACK - 1):] if len(gaps) >= LOOKBACK - 1 else gaps

        p_next5 = p_next10 = None
        pattern_matches = 0

        if len(labels) >= LOOKBACK + 1 and current_gap is not None:
            matched_next_gaps = []
            for i in range(len(labels) - LOOKBACK):
                window = ['W' if l == 'WWWW' else 'R' for l in labels[i:i + LOOKBACK]]
                if window == last5_seq:
                    next_gap_idx = i + LOOKBACK - 1   # gap between labels[i+4] and labels[i+5]
                    if next_gap_idx < len(gaps):
                        matched_next_gaps.append(gaps[next_gap_idx])

            pattern_matches = len(matched_next_gaps)

            if matched_next_gaps:
                # Keep only historical gaps >= current_gap (event hasn't appeared yet)
                still_open = [g for g in matched_next_gaps if g >= current_gap]
                if still_open:
                    p_next5  = round(sum(1 for g in still_open if g <= current_gap + 5)  / len(still_open) * 100, 1)
                    p_next10 = round(sum(1 for g in still_open if g <= current_gap + 10) / len(still_open) * 100, 1)
                else:
                    # All historical gaps were shorter — currently overdue beyond every match
                    # Use unconditional distribution from matched gaps
                    p_next5  = round(sum(1 for g in matched_next_gaps if g <= 5)  / len(matched_next_gaps) * 100, 1)
                    p_next10 = round(sum(1 for g in matched_next_gaps if g <= 10) / len(matched_next_gaps) * 100, 1)

        # Fallback to simple binomial if not enough pattern history
        if p_next5 is None:
            p_next5  = round((1 - (1 - p_round) ** 5)  * 100, 1)
            p_next10 = round((1 - (1 - p_round) ** 10) * 100, 1)

        return {
            'total_occurrences': len(positions),
            'last10_gaps':       last10_gaps,
            'last10_labels':     last10_labels,
            'last10_ids':        last10_ids,
            'mean_gap':          mean_gap,
            'std_gap':           std_gap,
            'min_gap':           min_gap,
            'max_gap':           max_gap,
            'current_gap':       current_gap,
            'last_occ_id':       last_occ_id,
            'last_pat':          last_pat,
            'p_per_round_pct':   round(p_round * 100, 2),
            'p_next5':           p_next5,
            'p_next10':          p_next10,
            'overdue':           overdue,
            'due_ratio':         due_ratio,
            'trend':             trend,
            'last5_pattern':     last5_seq,
            'last5_gaps':        last5_gaps,
            'pattern_matches':   pattern_matches,
        }

    return _no_cache(jsonify({
        'total_rounds': total,
        'last_id':      last_id,
        'WWWW':         _analyze('WWWW'),
        'RRRR':         _analyze('RRRR'),
        'COMBINED':     _analyze_combined(),
    }))


# ── Color Game API ─────────────────────────────────────────────────────────────

@app.route('/api/cg/prediction')
def api_cg_prediction():
    snap = _load_json(SNAP_CG_PATH)
    if snap is None:
        return jsonify({'error': 'No color game data yet. Run collector_cg.py first.'}), 404
    return _no_cache(jsonify(snap))

@app.route('/api/cg/data')
def api_cg_data():
    page      = max(1, int(request.args.get('page', 1)))
    page_size = min(200, max(10, int(request.args.get('size', 50))))
    search    = request.args.get('search', '').strip()
    sort_col  = request.args.get('sort', 'id')
    sort_dir  = 'ASC' if request.args.get('dir', 'desc') == 'asc' else 'DESC'
    VALID_COLS = {'id', 'number', 'color', 'is_purple', 'result', 'pred_color',
                  'pred_number', 'pred_purple', 'confidence', 'bet'}
    if sort_col not in VALID_COLS:
        sort_col = 'id'

    if not os.path.exists(DB_CG_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_CG_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        # Ensure new columns exist (safe migration)
        existing = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
        for col, defn in [('pred_number','INTEGER DEFAULT -1'),
                          ('pred_purple','INTEGER DEFAULT -1'),
                          ('number_result','TEXT DEFAULT ""')]:
            if col not in existing:
                conn.execute(f'ALTER TABLE rounds ADD COLUMN {col} {defn}')
        conn.commit()

        where, params = '', []
        if search:
            s = search.lower()
            if s == 'purple':
                where, params = 'WHERE is_purple=1', []
            elif s in ('red', 'green'):
                where, params = 'WHERE color=?', [s]
            elif s in ('win', 'loss'):
                where, params = 'WHERE result=?', [s.upper()]
            elif s in ('hit', 'miss'):
                where, params = 'WHERE number_result=?', [s.upper()]
            elif s == 'bet':
                where, params = 'WHERE bet=1', []
            else:
                where  = 'WHERE CAST(id AS TEXT) LIKE ? OR CAST(number AS TEXT) LIKE ? OR result LIKE ?'
                params = [f'%{search}%'] * 3

        total  = conn.execute(f'SELECT COUNT(*) FROM rounds {where}', params).fetchone()[0]
        offset = (page - 1) * page_size
        rows   = conn.execute(
            f'SELECT id,number,color,is_purple,result,pred_color,confidence,bet,'
            f'pred_number,pred_purple,number_result '
            f'FROM rounds {where} ORDER BY {sort_col} {sort_dir} LIMIT ? OFFSET ?',
            params + [page_size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    pages = max(1, (total + page_size - 1) // page_size)
    return _no_cache(jsonify({
        'total': total, 'page': page, 'pages': pages,
        'rows': [
            {'id': r[0], 'number': r[1], 'color': r[2], 'is_purple': r[3],
             'result': r[4], 'pred_color': r[5], 'confidence': r[6], 'bet': r[7],
             'pred_number': r[8], 'pred_purple': r[9], 'number_result': r[10]}
            for r in rows
        ]
    }))

@app.route('/api/cg/stats')
def api_cg_stats():
    if not os.path.exists(DB_CG_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_CG_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        total_pred = conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE pred_color != '' AND result IN ('WIN','LOSS')"
        ).fetchone()[0]
        last100 = conn.execute(
            "SELECT result FROM rounds WHERE pred_color != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        w100 = sum(1 for r in last100 if r[0] == 'WIN')
        last200 = conn.execute(
            "SELECT result FROM rounds WHERE pred_color != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        w200 = sum(1 for r in last200 if r[0] == 'WIN')
        bet_rows = conn.execute(
            "SELECT result FROM rounds WHERE bet=1 AND result IN ('WIN','LOSS')"
        ).fetchall()
        wb = sum(1 for r in bet_rows if r[0] == 'WIN')
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500
    n100, n200, nb = len(last100), len(last200), len(bet_rows)
    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100':  {'win': w100, 'n': n100, 'wr': round(w100/n100, 4) if n100 else None},
        'last200':  {'win': w200, 'n': n200, 'wr': round(w200/n200, 4) if n200 else None},
        'bet_only': {'win': wb,   'n': nb,   'wr': round(wb/nb,   4) if nb   else None},
    }))


# ── FST1M API ──────────────────────────────────────────────────────────────────

@app.route('/api/fst1m/prediction')
def api_fst1m_prediction():
    snap = _load_json(SNAP_FST1M_PATH)
    if snap is None:
        return jsonify({'error': 'No FST1M data yet. Run predict_fst1m.py first.'}), 404
    # Augment with hot sums: last 50 completed rounds (inclusive of latest)
    try:
        conn = sqlite3.connect(DB_FST1M_PATH, timeout=5)
        rows = conn.execute(
            'SELECT total FROM rounds ORDER BY id DESC LIMIT 50'
        ).fetchall()
        conn.close()
        from collections import Counter
        counts = Counter(r[0] for r in rows)
        hot = sorted(counts.items(), key=lambda x: -x[1])[:6]
        snap['hot_sums'] = [{'sum': s, 'count': c} for s, c in hot]
    except Exception:
        snap['hot_sums'] = []
    return _no_cache(jsonify(snap))

@app.route('/api/fst1m/data')
def api_fst1m_data():
    page      = max(1, int(request.args.get('page', 1)))
    page_size = min(200, max(10, int(request.args.get('size', 50))))
    search    = request.args.get('search', '').strip()
    if not os.path.exists(DB_FST1M_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_FST1M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        # Safe column migration
        existing = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
        for col, defn in [('pred_oe','TEXT DEFAULT ""'), ('conf_oe','REAL DEFAULT 0'),
                          ('result_oe','TEXT DEFAULT "Not Predicted"'), ('bet_oe','INTEGER DEFAULT 0'),
                          ('pred_sum_val','INTEGER DEFAULT 0'), ('pred_sum_zone','TEXT DEFAULT ""'),
                          ('pred_hot_sums','TEXT DEFAULT "[]"')]:
            if col not in existing:
                conn.execute(f'ALTER TABLE rounds ADD COLUMN {col} {defn}')
        conn.commit()

        base   = "1=1"
        params = []
        if search:
            s_up = search.upper()
            if s_up in ('BIG', 'SMALL'):
                base += " AND big_small=?"; params.append(s_up)
            elif s_up in ('ODD', 'EVEN'):
                base += " AND odd_even=?"; params.append(s_up)
            elif s_up in ('WIN', 'LOSS'):
                base += " AND result=?"; params.append(s_up)
            elif s_up == 'WIN-OE':
                base += " AND result_oe='WIN'"
            elif s_up == 'LOSS-OE':
                base += " AND result_oe='LOSS'"
            elif s_up == 'BET':
                base += " AND bet=1"
            elif s_up == 'BET-OE':
                base += " AND bet_oe=1"
            elif s_up == 'SKIP':
                base += " AND confidence >= 0.55 AND bet=0"
            elif s_up in ('LOW', 'MID', 'HIGH'):
                base += " AND pred_sum_zone=?"; params.append(s_up)
            else:
                base += " AND CAST(id AS TEXT) LIKE ?"; params.append(f'%{search}%')

        total  = conn.execute(f"SELECT COUNT(*) FROM rounds WHERE {base}", params).fetchone()[0]
        offset = (page - 1) * page_size
        rows   = conn.execute(
            f"SELECT id,n1,n2,n3,n4,total,big_small,odd_even,"
            f"result,pred_bs,confidence,bet,"
            f"pred_oe,conf_oe,result_oe,bet_oe,pred_sum_val,pred_sum_zone,pred_hot_sums "
            f"FROM rounds WHERE {base} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    pages   = max(1, (total + page_size - 1) // page_size)
    log     = _load_json(LOG_FST1M_PATH) or {}
    import json as _json
    entries = []
    for r in rows:
        db_hot_sums = []
        try:
            db_hot_sums = _json.loads(r[18] or "[]")
        except Exception:
            pass
        e = {
            'round_id':      str(r[0]),
            'n1': r[1], 'n2': r[2], 'n3': r[3], 'n4': r[4],
            'total':         r[5],
            'big_small':     r[6],
            'odd_even':      r[7],
            'result':        r[8],
            'pred_bs':       r[9],
            'confidence':    r[10],
            'bet':           r[11],
            'pred_oe':       r[12] or '',
            'conf_oe':       r[13] or 0,
            'result_oe':     r[14] or 'Not Predicted',
            'bet_oe':        r[15] or 0,
            'pred_sum_val':  r[16] or 0,
            'pred_sum_zone': r[17] or '',
            'pred_hot_sums': db_hot_sums,
        }
        le = log.get(str(r[0]))
        if le:
            sigs = le.get('signals', {})
            e['signals'] = sigs
            if not db_hot_sums and sigs.get('pred_hot_sums'):
                e['pred_hot_sums'] = sigs['pred_hot_sums']
        entries.append(e)
    return _no_cache(jsonify({'total': total, 'page': page, 'pages': pages, 'rows': entries}))

@app.route('/api/fst1m/stats')
def api_fst1m_stats():
    if not os.path.exists(DB_FST1M_PATH):
        return _no_cache(jsonify({}))
    import json as _json
    try:
        conn = sqlite3.connect(DB_FST1M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        # BIG/SMALL stats
        total_pred = conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS')"
        ).fetchone()[0]
        last100 = conn.execute(
            "SELECT result FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        w100 = sum(1 for r in last100 if r[0] == 'WIN'); n100 = len(last100)
        last200 = conn.execute(
            "SELECT result FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        w200 = sum(1 for r in last200 if r[0] == 'WIN'); n200 = len(last200)
        bet_rows = conn.execute(
            "SELECT result FROM rounds WHERE bet=1 AND result IN ('WIN','LOSS')"
        ).fetchall()
        wb = sum(1 for r in bet_rows if r[0] == 'WIN'); nb = len(bet_rows)
        # ODD/EVEN stats
        oe100 = conn.execute(
            "SELECT result_oe FROM rounds WHERE pred_oe != '' AND result_oe IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        oe_w100 = sum(1 for r in oe100 if r[0] == 'WIN'); oe_n100 = len(oe100)
        bet_oe_rows = conn.execute(
            "SELECT result_oe FROM rounds WHERE bet_oe=1 AND result_oe IN ('WIN','LOSS')"
        ).fetchall()
        oe_wb = sum(1 for r in bet_oe_rows if r[0] == 'WIN'); oe_nb = len(bet_oe_rows)

        # HOT SUM hit rate — actual total in pred_hot_sums list
        hot_rows = conn.execute(
            "SELECT id, total, pred_hot_sums FROM rounds "
            "WHERE total IS NOT NULL AND pred_hot_sums IS NOT NULL AND pred_hot_sums != '[]' "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        conn.close()

        # HD SUM hit rate — actual total in pred_sum_nums_filtered (from pred log)
        log_fst1m = _load_json(LOG_FST1M_PATH) or {}

        hot_hits, hd_hits = [], []
        for rid, total, hot_json in hot_rows:
            entry = log_fst1m.get(str(rid), {})
            sigs  = entry.get('signals', {})

            # HOT SUM — use pred_hot_sums_filtered from log (what is displayed to user)
            hot_filtered = sigs.get('pred_hot_sums_filtered')
            if hot_filtered:
                hot_list = hot_filtered
            else:
                try:
                    hot_list = _json.loads(hot_json or '[]')
                except Exception:
                    hot_list = []
            if hot_list:
                hot_hits.append(1 if (total in hot_list) else 0)

            # HD SUM
            hd_raw = sigs.get('pred_sum_nums_filtered') or sigs.get('pred_sum_nums') or []
            if hd_raw:
                hd_vals = [v for v, _ in hd_raw] if isinstance(hd_raw[0], (list, tuple)) else hd_raw
                hd_hits.append(1 if (total in hd_vals) else 0)

        def _wr(hits, n):
            w = hits[:n]
            if not w:
                return {'win': 0, 'n': 0, 'wr': None}
            s = sum(w)
            return {'win': s, 'n': len(w), 'wr': round(s / len(w), 4)}

    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100':         {'win': w100,    'n': n100,    'wr': round(w100/n100,       4) if n100    else None},
        'last200':         {'win': w200,    'n': n200,    'wr': round(w200/n200,       4) if n200    else None},
        'bet_only':        {'win': wb,      'n': nb,      'wr': round(wb/nb,           4) if nb      else None},
        'oe_last100':      {'win': oe_w100, 'n': oe_n100, 'wr': round(oe_w100/oe_n100,4) if oe_n100 else None},
        'oe_bet':          {'win': oe_wb,   'n': oe_nb,   'wr': round(oe_wb/oe_nb,    4) if oe_nb   else None},
        'hot_sum_last50':  _wr(hot_hits, 50),
        'hot_sum_last100': _wr(hot_hits, 100),
        'hot_sum_last200': _wr(hot_hits, 200),
        'hd_sum_last50':   _wr(hd_hits, 50),
        'hd_sum_last100':  _wr(hd_hits, 100),
        'hd_sum_last200':  _wr(hd_hits, 200),
    }))


# ── BLK3M API ──────────────────────────────────────────────────────────────────

@app.route('/api/blk3m/prediction')
def api_blk3m_prediction():
    snap = _load_json(SNAP_BLK3M_PATH)
    if snap is None:
        return _no_cache(jsonify({'error': 'No prediction yet'})), 404
    return _no_cache(jsonify(snap))


@app.route('/api/blk3m/data')
def api_blk3m_data():
    page   = max(1, int(request.args.get('page', 1)))
    size   = max(1, min(200, int(request.args.get('size', 50))))
    search = request.args.get('search', '').strip()
    if not os.path.exists(DB_BLK3M_PATH):
        return _no_cache(jsonify({'total': 0, 'page': 1, 'pages': 1, 'rows': []}))
    try:
        conn = sqlite3.connect(DB_BLK3M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        where  = ''
        params = []
        if search:
            where  = 'WHERE CAST(id AS TEXT) LIKE ?'
            params = [f'%{search}%']
        total  = conn.execute(f'SELECT COUNT(*) FROM rounds {where}', params).fetchone()[0]
        offset = (page - 1) * size
        rows   = conn.execute(
            f'SELECT id,value,total,big_small,odd_even,result,pred_bs,confidence,bet,'
            f'pred_oe,conf_oe,result_oe,bet_oe FROM rounds {where} ORDER BY id DESC LIMIT ? OFFSET ?',
            params + [size, offset]
        ).fetchall()
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    entries = []
    log = _load_json(LOG_BLK3M_PATH) or {}
    for r in rows:
        rid = r[0]
        entry = log.get(str(rid), {})
        sigs  = entry.get('signals', {})
        # Always recompute from the stored hash — guards against stale DB values
        bs, oe = _blk3m_classify(r[1])
        if bs is None:
            bs, oe = r[3], r[4]
        pred_bs = r[6]
        pred_oe = r[9]
        result    = ('WIN' if pred_bs == bs else 'LOSS') if pred_bs else r[5]
        result_oe = ('WIN' if pred_oe == oe else 'LOSS') if pred_oe else r[11]
        entries.append({
            'id': rid, 'value': r[1], 'total': r[2],
            'big_small': bs, 'odd_even': oe,
            'result': result, 'pred_bs': pred_bs, 'confidence': r[7], 'bet': r[8],
            'pred_oe': pred_oe, 'conf_oe': r[10], 'result_oe': result_oe, 'bet_oe': r[12],
            'signals': sigs,
        })
    pages = max(1, (total + size - 1) // size)
    return _no_cache(jsonify({'total': total, 'page': page, 'pages': pages, 'rows': entries}))


@app.route('/api/blk3m/stats')
def api_blk3m_stats():
    if not os.path.exists(DB_BLK3M_PATH):
        return _no_cache(jsonify({}))
    try:
        conn = sqlite3.connect(DB_BLK3M_PATH, timeout=5)
        conn.execute('PRAGMA journal_mode=WAL')
        total_pred = conn.execute(
            "SELECT COUNT(*) FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS')"
        ).fetchone()[0]
        last100 = conn.execute(
            "SELECT result FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        w100 = sum(1 for r in last100 if r[0] == 'WIN'); n100 = len(last100)
        last200 = conn.execute(
            "SELECT result FROM rounds WHERE pred_bs != '' AND result IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        w200 = sum(1 for r in last200 if r[0] == 'WIN'); n200 = len(last200)
        bet_rows = conn.execute(
            "SELECT result FROM rounds WHERE bet=1 AND result IN ('WIN','LOSS')"
        ).fetchall()
        wb = sum(1 for r in bet_rows if r[0] == 'WIN'); nb = len(bet_rows)
        oe100 = conn.execute(
            "SELECT result_oe FROM rounds WHERE pred_oe != '' AND result_oe IN ('WIN','LOSS') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
        oe_w100 = sum(1 for r in oe100 if r[0] == 'WIN'); oe_n100 = len(oe100)
        bet_oe = conn.execute(
            "SELECT result_oe FROM rounds WHERE bet_oe=1 AND result_oe IN ('WIN','LOSS')"
        ).fetchall()
        oe_wb = sum(1 for r in bet_oe if r[0] == 'WIN'); oe_nb = len(bet_oe)
        conn.close()
    except Exception as e:
        return _no_cache(jsonify({'error': str(e)})), 500

    def _wr(w, n):
        return {'win': w, 'n': n, 'wr': round(w / n, 4) if n else None}

    return _no_cache(jsonify({
        'total_predicted': total_pred,
        'last100':    _wr(w100,    n100),
        'last200':    _wr(w200,    n200),
        'bet_only':   _wr(wb,      nb),
        'oe_last100': _wr(oe_w100, oe_n100),
        'oe_bet':     _wr(oe_wb,   oe_nb),
    }))


# ── Push API (local collectors → PythonAnywhere) ──────────────────────────────

def _ensure_ds3m_table():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY,
            disc1 TEXT NOT NULL, disc2 TEXT NOT NULL,
            disc3 TEXT NOT NULL, disc4 TEXT NOT NULL,
            pattern TEXT NOT NULL, flag TEXT DEFAULT "",
            oe TEXT NOT NULL,
            result TEXT DEFAULT "Not Predicted",
            pred_oe TEXT DEFAULT "", confidence REAL DEFAULT 0, bet TEXT DEFAULT ""
        )
    ''')
    conn.commit()
    conn.close()

def _ensure_ds3m_new_table():
    conn = sqlite3.connect(DB_NEW_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY,
            disc1 TEXT NOT NULL, disc2 TEXT NOT NULL,
            disc3 TEXT NOT NULL, disc4 TEXT NOT NULL,
            pattern TEXT DEFAULT "", oe TEXT NOT NULL,
            result TEXT DEFAULT "Not Predicted",
            pred_oe TEXT DEFAULT "", confidence REAL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def _ensure_blk3m_table():
    conn = sqlite3.connect(DB_BLK3M_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY, value TEXT NOT NULL, total INTEGER NOT NULL,
            big_small TEXT NOT NULL, odd_even TEXT NOT NULL,
            result TEXT DEFAULT "Not Predicted", pred_bs TEXT DEFAULT "",
            confidence REAL DEFAULT 0, bet INTEGER DEFAULT 0,
            pred_oe TEXT DEFAULT "", conf_oe REAL DEFAULT 0,
            result_oe TEXT DEFAULT "Not Predicted", bet_oe INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def _ensure_cg_table():
    conn = sqlite3.connect(DB_CG_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY, number INTEGER NOT NULL,
            color TEXT NOT NULL, is_purple INTEGER NOT NULL DEFAULT 0,
            result TEXT DEFAULT "Not Predicted", pred_color TEXT DEFAULT "",
            confidence REAL DEFAULT 0, bet INTEGER DEFAULT 0,
            pred_number INTEGER DEFAULT -1, pred_purple INTEGER DEFAULT -1,
            number_result TEXT DEFAULT ""
        )
    ''')
    conn.commit()
    conn.close()

def _ensure_ds1m_table():
    conn = sqlite3.connect(DB_DS1M_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY,
            disc1 TEXT NOT NULL, disc2 TEXT NOT NULL,
            disc3 TEXT NOT NULL, disc4 TEXT NOT NULL,
            pattern TEXT DEFAULT "", flag TEXT DEFAULT "",
            oe TEXT NOT NULL,
            result TEXT DEFAULT "Not Predicted",
            pred_oe TEXT DEFAULT "", confidence REAL DEFAULT 0,
            bet INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def _ensure_fst1m_table():
    conn = sqlite3.connect(DB_FST1M_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY,
            n1 INTEGER NOT NULL, n2 INTEGER NOT NULL,
            n3 INTEGER NOT NULL, n4 INTEGER NOT NULL,
            total INTEGER NOT NULL, big_small TEXT NOT NULL, odd_even TEXT NOT NULL,
            result TEXT DEFAULT "Not Predicted", pred_bs TEXT DEFAULT "",
            confidence REAL DEFAULT 0, bet INTEGER DEFAULT 0,
            pred_oe TEXT DEFAULT "", conf_oe REAL DEFAULT 0,
            result_oe TEXT DEFAULT "Not Predicted", bet_oe INTEGER DEFAULT 0,
            pred_sum_val INTEGER DEFAULT 0, pred_sum_zone TEXT DEFAULT "",
            pred_hot_sums TEXT DEFAULT "[]"
        )
    ''')
    conn.commit()
    conn.close()

def _write_json_atomic(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp, path)

@app.route('/api/push/<game>', methods=['POST'])
def api_push(game):
    if not _push_token_valid():
        return jsonify({'error': 'Unauthorized'}), 401
    if game not in ('blk3m', 'cg', 'ds1m', 'fst1m', 'ds3m', 'new'):
        return jsonify({'error': f'Unknown game: {game}'}), 400

    body     = request.get_json(silent=True) or {}
    records  = body.get('records', [])
    snapshot = body.get('snapshot')
    log_upd  = body.get('pred_log', {})
    inserted = 0

    if game == 'blk3m':
        _ensure_blk3m_table()
        if records:
            conn = sqlite3.connect(DB_BLK3M_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,value,total,big_small,odd_even,result,pred_bs,confidence,bet,'
                    'pred_oe,conf_oe,result_oe,bet_oe) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('value',''), r.get('total',0),
                      r.get('big_small',''), r.get('odd_even',''),
                      r.get('result','Not Predicted'), r.get('pred_bs',''),
                      r.get('confidence',0), r.get('bet',0),
                      r.get('pred_oe',''), r.get('conf_oe',0),
                      r.get('result_oe','Not Predicted'), r.get('bet_oe',0))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_BLK3M_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_BLK3M_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_BLK3M_PATH, existing)

    elif game == 'cg':
        _ensure_cg_table()
        if records:
            conn = sqlite3.connect(DB_CG_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,number,color,is_purple,result,pred_color,confidence,bet) VALUES(?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('number',0), r.get('color',''),
                      r.get('is_purple',0), r.get('result','Not Predicted'),
                      r.get('pred_color',''), r.get('confidence',0), r.get('bet',0))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_CG_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_CG_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_CG_PATH, existing)

    elif game == 'ds1m':
        _ensure_ds1m_table()
        if records:
            conn = sqlite3.connect(DB_DS1M_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence,bet)'
                    ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('disc1',''), r.get('disc2',''),
                      r.get('disc3',''), r.get('disc4',''),
                      r.get('pattern',''), r.get('flag',''), r.get('oe',''),
                      r.get('result','Not Predicted'), r.get('pred_oe',''), r.get('confidence',0),
                      r.get('bet',0))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_DS1M_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_DS1M_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_DS1M_PATH, existing)

    elif game == 'fst1m':
        _ensure_fst1m_table()
        if records:
            conn = sqlite3.connect(DB_FST1M_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,n1,n2,n3,n4,total,big_small,odd_even,result,pred_bs,confidence,bet,'
                    'pred_oe,conf_oe,result_oe,bet_oe,pred_sum_val,pred_sum_zone,pred_hot_sums)'
                    ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('n1',0), r.get('n2',0), r.get('n3',0), r.get('n4',0),
                      r.get('total',0), r.get('big_small',''), r.get('odd_even',''),
                      r.get('result','Not Predicted'), r.get('pred_bs',''),
                      r.get('confidence',0), r.get('bet',0),
                      r.get('pred_oe',''), r.get('conf_oe',0),
                      r.get('result_oe','Not Predicted'), r.get('bet_oe',0),
                      r.get('pred_sum_val',0), r.get('pred_sum_zone',''),
                      r.get('pred_hot_sums','[]'))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_FST1M_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_FST1M_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_FST1M_PATH, existing)

    elif game == 'ds3m':
        _ensure_ds3m_table()
        if records:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,confidence,bet)'
                    ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('disc1',''), r.get('disc2',''),
                      r.get('disc3',''), r.get('disc4',''),
                      r.get('pattern',''), r.get('flag',''), r.get('oe',''),
                      r.get('result','Not Predicted'), r.get('pred_oe',''),
                      r.get('confidence',0), r.get('bet',''))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_PATH, existing)

    elif game == 'new':
        _ensure_ds3m_new_table()
        if records:
            conn = sqlite3.connect(DB_NEW_PATH, timeout=10)
            conn.execute('PRAGMA journal_mode=WAL')
            try:
                conn.executemany(
                    'INSERT OR REPLACE INTO rounds'
                    '(id,disc1,disc2,disc3,disc4,pattern,oe,result,pred_oe,confidence)'
                    ' VALUES(?,?,?,?,?,?,?,?,?,?)',
                    [(r['id'], r.get('disc1',''), r.get('disc2',''),
                      r.get('disc3',''), r.get('disc4',''),
                      r.get('pattern',''), r.get('oe',''),
                      r.get('result','Not Predicted'), r.get('pred_oe',''),
                      r.get('confidence',0))
                     for r in records]
                )
                inserted = conn.execute('SELECT changes()').fetchone()[0]
                conn.commit()
            finally:
                conn.close()
        if snapshot:
            _write_json_atomic(SNAP_NEW_PATH, snapshot)
        if log_upd:
            existing = _load_json(LOG_NEW_PATH) or {}
            existing.update(log_upd)
            _write_json_atomic(LOG_NEW_PATH, existing)

    return jsonify({'ok': True, 'inserted': inserted})


def _ensure_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'training_enabled': True}, f)


def _db_watcher():
    """Background thread: runs predict_new.py whenever DB is newer than its snapshot."""
    import time
    predict_new_py = os.path.join(_BASE, 'predict_new.py')
    while True:
        try:
            time.sleep(30)
            if not os.path.exists(DB_PATH):
                continue
            db_mtime   = os.path.getmtime(DB_PATH)
            snap_mtime = os.path.getmtime(SNAP_NEW_PATH) if os.path.exists(SNAP_NEW_PATH) else 0
            if db_mtime > snap_mtime:
                if _predict_new_lock.acquire(blocking=False):
                    try:
                        subprocess.run([sys.executable, predict_new_py], capture_output=True)
                    finally:
                        _predict_new_lock.release()
        except Exception:
            pass


if __name__ == '__main__':
    key = _load_stored_key()
    valid, msg = validate_key(key) if key else (False, 'No activation code found.')
    print(f'  [LICENSE] {msg}' if valid else f'  [LICENSE] {msg} — visit /activate in browser.')
    _ensure_config()
    if not _load_credentials():
        print("  [!]  No credentials found. Run: python create_credentials.py")
    threading.Thread(target=_db_watcher, daemon=True).start()
    print(f"  Starting DS3M dashboard on http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
