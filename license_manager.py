"""
Activation code system for FXPro Predictions.

Key formats (both supported):
  FXPRO-YYYYMMDD-XXXXXXXX    date-only key (valid until end of expiry day)
  FXPRO-YYYYMMDDHH-XXXXXXXX  hourly key    (valid until expiry hour, e.g. 2026062115 = 21 Jun 15:xx)

  XXXXXXXX = 8-char HMAC-SHA256 checksum (uppercase hex) of the date/datetime part

Usage:
  from license_manager import check_license
  check_license()   # call at app startup; exits with code 1 if invalid
"""
import hashlib
import hmac
import datetime
import json
import os
import sys

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')
_KEY_FILE = os.path.join(_DATA_DIR, 'license.key')

# Change this secret before distributing — keep it out of public repos
_SECRET = b'fxpro-pred-lic-v1-2026-$3cur3key!'


def _checksum(expiry_str: str) -> str:
    mac = hmac.new(_SECRET, expiry_str.encode(), hashlib.sha256)
    return mac.hexdigest()[:8].upper()


def generate_key(expiry) -> str:
    """
    Generate a valid activation key.
    expiry can be:
      datetime.date     → FXPRO-YYYYMMDD-XXXXXXXX  (valid until end of that day)
      datetime.datetime → FXPRO-YYYYMMDDHH-XXXXXXXX (valid until that hour)
    """
    if isinstance(expiry, datetime.datetime):
        expiry_str = expiry.strftime('%Y%m%d%H')
    else:
        expiry_str = expiry.strftime('%Y%m%d')
    return f'FXPRO-{expiry_str}-{_checksum(expiry_str)}'


def validate_key(key: str) -> tuple:
    """Return (valid: bool, message: str)."""
    try:
        parts = key.strip().upper().split('-')
        if len(parts) != 3 or parts[0] != 'FXPRO':
            return False, 'Invalid key format (expected FXPRO-YYYYMMDD-XXXXXXXX).'
        expiry_str, checksum = parts[1], parts[2]
        if not expiry_str.isdigit() or len(expiry_str) not in (8, 10):
            return False, 'Invalid key format (bad date part).'
        if _checksum(expiry_str) != checksum:
            return False, 'Activation code is invalid or tampered.'

        if len(expiry_str) == 10:
            # Hourly key: YYYYMMDDHH — compare against current datetime
            expiry_dt = datetime.datetime.strptime(expiry_str, '%Y%m%d%H')
            now = datetime.datetime.now()
            if now >= expiry_dt + datetime.timedelta(hours=1):
                return False, f'Activation code expired at {expiry_dt.strftime("%Y-%m-%d %H:00")}.'
            mins_left = int((expiry_dt + datetime.timedelta(hours=1) - now).total_seconds() / 60)
            return True, f'Licensed until {expiry_dt.strftime("%Y-%m-%d %H:00")} ({mins_left} min remaining).'
        else:
            # Date key: YYYYMMDD — compare against today
            expiry_date = datetime.datetime.strptime(expiry_str, '%Y%m%d').date()
            today = datetime.date.today()
            if today > expiry_date:
                return False, f'Activation code expired on {expiry_date}.'
            days_left = (expiry_date - today).days
            return True, f'Licensed until {expiry_date} ({days_left} day(s) remaining).'
    except ValueError:
        return False, 'Invalid key format (bad date).'
    except Exception as e:
        return False, f'Key validation error: {e}'


def _load_stored_key() -> str:
    """Load key from license.key file or .env / environment variable."""
    # 1. File
    if os.path.exists(_KEY_FILE):
        try:
            with open(_KEY_FILE, encoding='utf-8') as f:
                key = f.read().strip()
            if key:
                return key
        except Exception:
            pass
    # 2. Environment variable (also covers .env loaded by dotenv)
    return os.environ.get('LICENSE_KEY', '').strip()


def _save_key(key: str):
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_KEY_FILE, 'w', encoding='utf-8') as f:
        f.write(key.strip())


def check_license():
    """
    Validate the activation code.  Exits with code 1 if no valid key is found.
    Call this at the top of main() in every entry-point script.
    """
    print()
    key = _load_stored_key()

    if not key:
        print('  ┌─────────────────────────────────────────┐')
        print('  │        ACTIVATION REQUIRED              │')
        print('  └─────────────────────────────────────────┘')
        print('  No activation code found.')
        print('  Enter your activation code below.')
        print('  (It will be saved to data/license.key for future runs.)')
        print()
        key = input('  Activation code: ').strip()
        if not key:
            print('  No code entered. Exiting.')
            sys.exit(1)
        _save_key(key)

    valid, message = validate_key(key)
    if valid:
        print(f'  [LICENSE] {message}')
    else:
        print()
        print('  ┌─────────────────────────────────────────┐')
        print('  │        ACTIVATION FAILED                │')
        print('  └─────────────────────────────────────────┘')
        print(f'  {message}')
        print('  Please contact support for a valid activation code.')
        print()
        # Remove bad key so user is prompted again next run
        if os.path.exists(_KEY_FILE):
            os.remove(_KEY_FILE)
        sys.exit(1)
