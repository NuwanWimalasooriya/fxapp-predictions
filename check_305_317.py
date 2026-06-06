import sqlite3, os, json

db = os.path.join('data', 'ds3m.db')
conn = sqlite3.connect(db)

rows = conn.execute(
    'SELECT id, oe, pred_oe, confidence, result FROM rounds WHERE id BETWEEN 2061000305 AND 2061000317'
).fetchall()
conn.close()

log_path = os.path.join('data', 'pred_log.json')
with open(log_path, encoding='utf-8-sig') as f:
    log = json.load(f)

print(f'{"Round":<14} {"Actual":<6} {"Pred":<6} {"Conf":<6} {"Result"}')
print('-' * 50)

# Count EVEN/ODD in the FULL 295-323 window to determine dominant direction
full_rows = conn = sqlite3.connect(db).execute(
    'SELECT id, oe FROM rounds WHERE id BETWEEN 2061000295 AND 2061000323'
).fetchall()
sqlite3.connect(db).close()
n_even = sum(1 for _, oe in full_rows if oe == 'EVEN')
n_odd  = len(full_rows) - n_even
dominant = 'EVEN' if n_even >= n_odd else 'ODD'
print(f'295-323 window: EVEN={n_even}, ODD={n_odd} => Dominant = {dominant}')
print()

# Re-open for the main query
conn = sqlite3.connect(db)
rows = conn.execute(
    'SELECT id, oe, pred_oe, confidence, result FROM rounds WHERE id BETWEEN 2061000305 AND 2061000317'
).fetchall()
conn.close()

for rid, actual_oe, pred_oe, conf, result in rows:
    flag = '<< actual=' + dominant + ' dom' if actual_oe == dominant else ''
    print(f'{rid}  {actual_oe:<6} {pred_oe:<6} {conf:<6} {result}  {flag}')
