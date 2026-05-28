"""
One-time setup: create or update login credentials for the DS3M dashboard.
Run: python create_credentials.py
"""
import os, json, getpass, sys
sys.stdout.reconfigure(encoding='utf-8')

from werkzeug.security import generate_password_hash

_BASE      = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_BASE, 'data')
CREDS_PATH = os.path.join(_DATA_DIR, 'credentials.json')

os.makedirs(_DATA_DIR, exist_ok=True)

print("DS3M — Set Login Credentials")
print("=" * 36)

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

password_hash = generate_password_hash(password, method='pbkdf2:sha256:600000')

with open(CREDS_PATH, 'w') as f:
    json.dump({'username': username, 'password_hash': password_hash}, f, indent=2)

print(f"\nCredentials saved to: {CREDS_PATH}")
print("Restart server.py to activate.")
