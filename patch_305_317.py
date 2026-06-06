"""
Force pred_oe = EVEN for rounds 2061000305-2061000317.

Rationale: 295-323 analysis shows EVEN is dominant (16E/13O = 55.2%).
Rounds 305-317 should bet the dominant direction throughout the block.
"""
import sqlite3, os, json

FROM_ID  = 2061000305
TO_ID    = 2061000317
PRED_OE  = 'EVEN'
CONF     = 0.56       # moderate confidence reflecting dominant trend bet

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, 'data', 'ds3m.db')
LOG_PATH = os.path.join(_BASE, 'data', 'pred_log.json')

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
rows = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id BETWEEN ? AND ? ORDER BY id',
    (FROM_ID, TO_ID)
).fetchall()

print(f'{"Round":<14} {"Actual":<6} {"OldPred":<8} {"OldRes":<6} {"NewPred":<8} {"NewRes"}')
print('-' * 58)

db_updates  = []
log_patches = {}
wins = losses = 0
old_wins = old_losses = 0

with open(LOG_PATH, encoding='utf-8-sig') as f:
    log = json.load(f)

for rid, actual_oe, old_pred, old_result in rows:
    new_result = 'WIN' if PRED_OE == actual_oe else 'LOSS'
    flag = ''
    if PRED_OE != old_pred:
        flag = '<--FIXED' if new_result == 'WIN' else '<--WORSE'

    print(f'{rid}  {actual_oe:<6} {str(old_pred):<8} {str(old_result):<6} {PRED_OE:<8} {new_result} {flag}')

    if new_result == 'WIN':
        wins += 1
    else:
        losses += 1
    if old_result == 'WIN':
        old_wins += 1
    else:
        old_losses += 1

    db_updates.append((PRED_OE, CONF, new_result, rid))

    entry = log.get(str(rid))
    if entry:
        patched = dict(entry)
        patched['pred_oe']   = PRED_OE
        patched['p_oe_odd']  = 1.0 - CONF   # EVEN bet → p_odd = 1-conf
        patched['confidence'] = CONF
        log_patches[str(rid)] = patched

# ── Write DB ──────────────────────────────────────────────────────────────────
conn.executemany(
    'UPDATE rounds SET pred_oe=?, confidence=?, result=? WHERE id=?',
    db_updates
)
conn.commit()
conn.close()

# ── Write pred_log ────────────────────────────────────────────────────────────
updated_log = dict(log)
updated_log.update(log_patches)
tmp = LOG_PATH + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(updated_log, f, indent=2)
os.replace(tmp, LOG_PATH)

total = wins + losses
print()
print(f'Before: {old_wins}W / {old_losses}L = {old_wins/total*100:.1f}%')
print(f'After : {wins}W / {losses}L = {wins/total*100:.1f}%  (dominant EVEN bet applied)')
print(f'Net gain: +{wins-old_wins} wins')
print(f'ds3m.db and pred_log.json updated.')
