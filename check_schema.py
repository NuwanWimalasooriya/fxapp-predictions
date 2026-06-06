import sqlite3, os, json

db = os.path.join('data', 'ds3m.db')
conn = sqlite3.connect(db)

# Check rounds around FROM_ID
print('=== DB rows 295-300 ===')
rows = conn.execute(
    'SELECT id, oe, pred_oe, confidence, result FROM rounds WHERE id BETWEEN 2061000295 AND 2061000300'
).fetchall()
for r in rows:
    print(r)

# Check last 5 rows for null oe
print()
print('=== Last 5 DB rows ===')
rows = conn.execute(
    'SELECT id, oe, pred_oe, confidence, result FROM rounds ORDER BY id DESC LIMIT 5'
).fetchall()
for r in rows:
    print(r)

conn.close()

# Check pred_log for round 295
log_path = os.path.join('data', 'pred_log.json')
with open(log_path, encoding='utf-8-sig') as f:
    log = json.load(f)

print()
print('=== pred_log for 2061000295 ===')
entry = log.get('2061000295', {})
for k, v in entry.items():
    print(f'  {k}: {v}')
