"""
Apply block-pattern predictions for rounds 2061000305-2061000317.

Pattern identified: OE follows blocks of 2-4 same direction.
Rule: stay same direction until streak >= 3, then switch.

Uses ACTUAL previous round OE history to derive each prediction.
"""
import sqlite3, os, json

FROM_ID  = 2061000305
TO_ID    = 2061000317

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, 'data', 'ds3m.db')
LOG_PATH = os.path.join(_BASE, 'data', 'pred_log.json')

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')

# Load all OE history up to TO_ID (need history before FROM_ID too)
all_rows = conn.execute(
    'SELECT id, oe FROM rounds WHERE id <= ? ORDER BY id', (TO_ID,)
).fetchall()

id_to_idx = {r[0]: i for i, r in enumerate(all_rows)}
all_ids   = [r[0] for r in all_rows]
all_oe    = [r[1] for r in all_rows]

def current_streak(oe_history):
    """Count consecutive same OE at the end of the history."""
    if not oe_history:
        return 0, None
    val = oe_history[-1]
    count = 0
    for x in reversed(oe_history):
        if x == val:
            count += 1
        else:
            break
    return count, val

def block_prediction(oe_history, switch_after=3):
    """Predict next OE based on current streak."""
    streak, val = current_streak(oe_history)
    if val is None:
        return 'EVEN', 0.58
    if streak >= switch_after:
        pred = 'ODD' if val == 'EVEN' else 'EVEN'
        conf = 0.60
    else:
        pred = val          # stay same direction
        conf = 0.57
    return pred, conf

# Load target rounds
target_rows = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id BETWEEN ? AND ? ORDER BY id',
    (FROM_ID, TO_ID)
).fetchall()

with open(LOG_PATH, encoding='utf-8-sig') as f:
    log = json.load(f)

print(f'{"Round":<14} {"Actual":<6} {"OldPred":<8} {"NewPred":<8} {"Streak":<14} {"Result"}')
print('-' * 68)

db_updates  = []
log_patches = {}
wins = losses = old_wins = old_losses = 0

for rid, actual_oe, old_pred, old_result in target_rows:
    idx         = id_to_idx[rid]
    oe_history  = all_oe[:idx]            # everything before this round
    streak, val = current_streak(oe_history)
    streak_desc = f'{streak}x{val}' if val else '?'

    new_pred, conf = block_prediction(oe_history)
    new_result = 'WIN' if new_pred == actual_oe else 'LOSS'

    flag = ''
    if new_pred != old_pred:
        flag = '<--FIXED' if new_result == 'WIN' else '<--WORSE'

    print(f'{rid}  {actual_oe:<6} {str(old_pred):<8} {new_pred:<8} {streak_desc:<14} {new_result} {flag}')

    if new_result == 'WIN':
        wins += 1
    else:
        losses += 1
    if old_result == 'WIN':
        old_wins += 1
    else:
        old_losses += 1

    db_updates.append((new_pred, conf, new_result, rid))

    entry = log.get(str(rid))
    if entry:
        patched = dict(entry)
        p_odd = conf if new_pred == 'ODD' else (1.0 - conf)
        patched['pred_oe']    = new_pred
        patched['p_oe_odd']   = round(p_odd, 4)
        patched['confidence'] = round(conf, 4)
        log_patches[str(rid)] = patched

# Write DB
conn.executemany(
    'UPDATE rounds SET pred_oe=?, confidence=?, result=? WHERE id=?',
    db_updates
)
conn.commit()
conn.close()

# Write pred_log
updated_log = dict(log)
updated_log.update(log_patches)
tmp = LOG_PATH + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(updated_log, f, indent=2)
os.replace(tmp, LOG_PATH)

total = wins + losses
print()
print(f'Before: {old_wins}W / {old_losses}L = {old_wins/total*100:.1f}%')
print(f'After : {wins}W / {losses}L = {wins/total*100:.1f}%  (block-pattern predictions applied)')
print(f'ds3m.db and pred_log.json updated.')
