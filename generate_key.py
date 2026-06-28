"""
Admin tool — generate activation codes.
Keep this script private (do not distribute to end users).

Usage:
  python generate_key.py              # 30-day key (default)
  python generate_key.py 90           # 90-day key
  python generate_key.py 2026-12-31   # key expiring on a specific date
  python generate_key.py 24h          # 24-hour key (expires in 24 hours from now)
  python generate_key.py 48h          # 48-hour key
  python generate_key.py 6h           # 6-hour key
"""
import sys
import datetime
from license_manager import generate_key, validate_key


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '30'

    if arg.lower().endswith('h') and arg[:-1].isdigit():
        # Hour-based key: e.g. 24h, 6h, 48h
        hours = int(arg[:-1])
        expiry = datetime.datetime.now() + datetime.timedelta(hours=hours)
        expiry = expiry.replace(minute=0, second=0, microsecond=0)  # round to the hour
        key    = generate_key(expiry)
        valid, msg = validate_key(key)
        print()
        print(f'  Activation Code : {key}')
        print(f'  Expires At      : {expiry.strftime("%Y-%m-%d %H:00")}  ({hours}h from now)')
        print(f'  Status          : {msg}')
        print()

    elif '-' in arg:
        # Specific date: YYYY-MM-DD
        try:
            expiry = datetime.date.fromisoformat(arg)
        except ValueError:
            print(f'Invalid date: {arg}  (use YYYY-MM-DD)')
            sys.exit(1)
        key = generate_key(expiry)
        valid, msg = validate_key(key)
        print()
        print(f'  Activation Code : {key}')
        print(f'  Expiry Date     : {expiry}')
        print(f'  Status          : {msg}')
        print()

    else:
        # Number of days
        try:
            days = int(arg)
        except ValueError:
            print(f'Invalid argument: {arg}  (use Nh for hours, a number of days, or YYYY-MM-DD)')
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
