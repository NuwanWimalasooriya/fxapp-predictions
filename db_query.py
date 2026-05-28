"""
Interactive SQLite query tool for ds3m.db
Usage: python db_query.py
Type any SQL query and press Enter. Type 'exit' to quit.
Built-in shortcuts:
  \last      — last 20 records
  \stats     — result distribution
  \bet       — bet rounds (WIN/LOSS only)
  \recent    — last 5 predicted rounds with results
  \count     — total row count
  \tables    — show all tables
"""
import sqlite3, os, sys
sys.stdout.reconfigure(encoding='utf-8')

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

_env    = _load_env()
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', _env.get('DB_FILE', 'ds3m.db'))

SHORTCUTS = {
    r'\last':   'SELECT id, disc1||disc2||disc3||disc4 AS pattern, oe, result FROM rounds ORDER BY id DESC LIMIT 20',
    r'\stats':  'SELECT result, COUNT(*) AS count FROM rounds GROUP BY result ORDER BY count DESC',
    r'\bet':    "SELECT id, disc1||disc2||disc3||disc4 AS pattern, oe, result FROM rounds WHERE result IN ('WIN','LOSS') ORDER BY id DESC LIMIT 30",
    r'\recent': "SELECT id, disc1||disc2||disc3||disc4 AS pattern, oe, result FROM rounds ORDER BY id DESC LIMIT 5",
    r'\count':  'SELECT COUNT(*) AS total_rows, MAX(id) AS last_id FROM rounds',
    r'\tables': "SELECT name FROM sqlite_master WHERE type='table'",
}

def print_rows(cursor):
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        print('  (no rows)')
        return
    widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    fmt = '  ' + '  '.join(f'{{:<{w}}}' for w in widths)
    sep = '  ' + '  '.join('-' * w for w in widths)
    print(fmt.format(*cols))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) if v is not None else '' for v in row]))
    print(f'\n  {len(rows)} row(s)')

def main():
    if not os.path.exists(DB_PATH):
        print(f'DB not found: {DB_PATH}')
        return

    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    print(f'Connected to {DB_PATH}')
    print('Type SQL or a shortcut (\\last, \\stats, \\bet, \\recent, \\count, \\tables). Type exit to quit.\n')

    while True:
        try:
            query = input('db> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ('exit', 'quit', '.quit'):
            break
        sql = SHORTCUTS.get(query, query)
        try:
            cur = conn.execute(sql)
            if cur.description:
                print_rows(cur)
            else:
                conn.commit()
                print(f'  OK — {cur.rowcount} row(s) affected')
        except sqlite3.Error as e:
            print(f'  Error: {e}')
        print()

    conn.close()
    print('Bye.')

if __name__ == '__main__':
    main()
