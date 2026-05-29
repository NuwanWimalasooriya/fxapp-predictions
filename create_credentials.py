"""
Setup dashboard login credentials for DS3M.
Reads USERNAME/PASSWORD from .env by default, or prompts to override.
Run: python create_credentials.py
"""
import os, json, getpass, sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')

from werkzeug.security import generate_password_hash

_BASE      = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE, 'data')
CREDS_PATH = os.path.join(_DATA_DIR, 'credentials.json')

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

def _cipher_key(env):
    secret = env.get('SECRET_KEY', 'ds3m-default-key')
    return hashlib.sha256(secret.encode()).digest()

def encrypt(text, key):
    data = text.encode('utf-8')
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, key_stream)).hex()

def decrypt(hex_text, key):
    data = bytes.fromhex(hex_text)
    key_stream = (key * (len(data) // len(key) + 1))[:len(data)]
    return bytes(a ^ b for a, b in zip(data, key_stream)).decode('utf-8')

os.makedirs(_DATA_DIR, exist_ok=True)

_env      = _load_env()
_key      = _cipher_key(_env)
_env_user = _env.get('USERNAME', '')
_env_pass = _env.get('PASSWORD', '')

print("DS3M — Set Login Credentials")
print("=" * 36)

if _env_user and _env_pass:
    print(f"Found credentials in .env  (username: {_env_user})")
    choice = input("Use these? [Y/n]: ").strip().lower()
    if choice in ('', 'y', 'yes'):
        username = _env_user
        password = _env_pass
    else:
        username = input("Username: ").strip()
        if not username:
            print("Username cannot be empty.")
            sys.exit(1)
        while True:
            password = getpass.getpass("Password: ")
            confirm  = getpass.getpass("Confirm password: ")
            if not password:
                print("Password cannot be empty.")
                continue
            if password != confirm:
                print("Passwords do not match. Try again.")
                continue
            break
else:
    username = input("Username: ").strip()
    if not username:
        print("Username cannot be empty.")
        sys.exit(1)
    while True:
        password = getpass.getpass("Password: ")
        confirm  = getpass.getpass("Confirm password: ")
        if not password:
            print("Password cannot be empty.")
            continue
        if password != confirm:
            print("Passwords do not match. Try again.")
            continue
        break

password_hash      = generate_password_hash(password, method='pbkdf2:sha256:600000')
username_encrypted = encrypt(username, _key)
password_encrypted = encrypt(password, _key)

with open(CREDS_PATH, 'w') as f:
    json.dump({
        'username':      username_encrypted,
        'password':      password_encrypted,
        'password_hash': password_hash,
    }, f, indent=2)

print(f"\nCredentials saved to: {CREDS_PATH}")
print("Restart server.py to activate.")
