"""
remote_push — push new records, prediction snapshot, and pred_log entries
to the PythonAnywhere server after each collection cycle.

Configure in .env:
  REMOTE_URL=https://yourusername.pythonanywhere.com

The activation code stored in data/license.key is used as the bearer token.
"""
import os, json, ssl, urllib.request

_BASE     = os.path.dirname(os.path.abspath(__file__))
_KEY_FILE = os.path.join(_BASE, 'data', 'license.key')

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

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _load_license_key():
    if os.path.exists(_KEY_FILE):
        try:
            with open(_KEY_FILE, encoding='utf-8') as f:
                key = f.read().strip()
            if key:
                return key
        except Exception:
            pass
    return os.environ.get('LICENSE_KEY', '').strip()


def push_game(game, records, snap_path, log_path, log_limit=100):
    """
    Push records + snapshot + recent pred_log to the remote server.
    Silently skips if REMOTE_URL is not configured or no license key is found.
    """
    remote_url = _env.get('REMOTE_URL', '').rstrip('/')
    push_token = _load_license_key()
    if not remote_url or not push_token:
        return

    payload = {}

    if records:
        payload['records'] = records

    if snap_path and os.path.exists(snap_path):
        try:
            with open(snap_path, encoding='utf-8-sig') as f:
                payload['snapshot'] = json.load(f)
        except Exception:
            pass

    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, encoding='utf-8-sig') as f:
                full_log = json.load(f)
            if isinstance(full_log, dict):
                keys = sorted(full_log.keys(), key=lambda k: int(k) if k.isdigit() else 0)
                recent = keys[-log_limit:]
                payload['pred_log'] = {k: full_log[k] for k in recent}
        except Exception:
            pass

    if not payload:
        return

    data = json.dumps(payload, default=str).encode('utf-8')
    req = urllib.request.Request(
        f'{remote_url}/api/push/{game}',
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {push_token}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as r:
            resp = json.loads(r.read())
        inserted = resp.get('inserted', 0)
        if inserted:
            print(f'  [PUSH] {inserted} record(s) sent to remote ({game}).')
        else:
            print(f'  [PUSH] Snapshot/log sent to remote ({game}).')
    except Exception as e:
        print(f'  [PUSH] {e}')
