"""
Admin tool — generate activation codes.
Keep this script private (do not distribute to end users).

Usage:
  python generate_key.py              # 30-day key (default)
  python generate_key.py 90           # 90-day key
  python generate_key.py 2026-12-31   # key expiring on a specific date
"""
import sys
import datetime
from license_manager import generate_key, validate_key


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '30'

    if '-' in arg:
        # Specific date
        try:
            expiry = datetime.date.fromisoformat(arg)
        except ValueError:
            print(f'Invalid date: {arg}  (use YYYY-MM-DD)')
            sys.exit(1)
    else:
        # Number of days
        try:
            days = int(arg)
        except ValueError:
            print(f'Invalid argument: {arg}  (use a number of days or YYYY-MM-DD)')
            sys.exit(1)
        expiry = datetime.date.today() + datetime.timedelta(days=days)

    key = generate_key(expiry)
    valid, msg = validate_key(key)

    print()
    print(f'  Activation Code : {key}')
    print(f'  Expiry Date     : {expiry}')
    print(f'  Status          : {msg}')
    print()


if __name__ == '__main__':
    main()
