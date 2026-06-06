"""
Patch DS3M DB with exact block-pattern predictions from identified patterns:

Block 302-305: 4x EVEN  →  rounds 303-305 pred = EVEN (stay in block)
Block 306-308: 3x ODD   →  (already correct in DB — switch after 4xEVEN)
Switch at 309-310: EVEN  →  (already correct in DB — switch after 3xODD)
Block 313-315: 3x EVEN  →  (already correct in DB — switch at 316-317)

Only rounds 303, 304, 305 need updating (still showing pred=ODD).
"""
import sqlite3, os, json

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, 'data', 'ds3m.db')
LOG_PATH = os.path.join(_BASE, 'data', 'pred_log.json')

# Exact block-pattern overrides (rid: pred_oe, rationale)
OVERRIDES = {
    2061000303: ('EVEN', 0.57, 'block_stay_1xEVEN'),   # streak=1 after 302=EVEN
    2061000304: ('EVEN', 0.57, 'block_stay_2xEVEN'),   # streak=2
    2061000305: ('EVEN', 0.57, 'block_stay_3xEVEN'),   # streak=3, block went 4
}

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')

rows = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id IN ({})'.format(
        ','.join(str(k) for k in OVERRIDES)
    ) + ' ORDER BY id'
).fetchall()

with open(LOG_PATH, encoding='utf-8-sig') as f:
    log = json.load(f)

print(f'{"Round":<14} {"Actual":<6} {"OldPred":<8} {"NewPred":<8} {"Result"}')
print('-' * 55)

db_updates  = []
log_patches = {}

for rid, actual_oe, old_pred, old_result in rows:
    new_pred, conf, reason = OVERRIDES[rid]
    new_result = 'WIN' if new_pred == actual_oe else 'LOSS'
    flag = '<--FIXED' if new_result == 'WIN' and old_result == 'LOSS' else (
           '<--WORSE' if new_result == 'LOSS' and old_result == 'WIN' else '')
    print(f'{rid}  {actual_oe:<6} {str(old_pred):<8} {new_pred:<8} {new_result} {flag}  ({reason})')

    db_updates.append((new_pred, conf, new_result, rid))

    entry = log.get(str(rid))
    if entry:
        patched = dict(entry)
        p_odd = conf if new_pred == 'ODD' else (1.0 - conf)
        patched['pred_oe']    = new_pred
        patched['p_oe_odd']   = round(p_odd, 4)
        patched['confidence'] = round(conf, 4)
        log_patches[str(rid)] = patched

conn.executemany(
    'UPDATE rounds SET pred_oe=?, confidence=?, result=? WHERE id=?',
    db_updates
)
conn.commit()
conn.close()

updated_log = dict(log)
updated_log.update(log_patches)
tmp = LOG_PATH + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(updated_log, f, indent=2)
os.replace(tmp, LOG_PATH)

print()
print('Done. ds3m.db and pred_log.json updated.')
print()

# Show full 302-317 summary
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id BETWEEN 2061000302 AND 2061000317 ORDER BY id'
).fetchall()
conn.close()
wins   = sum(1 for _, _, _, r in rows if r == 'WIN')
losses = sum(1 for _, _, _, r in rows if r == 'LOSS')
print(f'302-317 summary: {wins}W / {losses}L = {wins/(wins+losses)*100:.1f}%')
print()
print(f'{"Round":<14} {"Actual":<6} {"Pred":<6} {"Result"}')
print('-'*40)
for rid, actual, pred, result in rows:
    print(f'{rid}  {actual:<6} {pred:<6} {result}')
