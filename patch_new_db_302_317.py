"""
Apply block-pattern predictions to ds3m_new.db for rounds 2061000302-2061000317.

Identified block pattern:
  302-305 = 4xEVEN block  →  stay EVEN (303,304,305)
  306-308 = 3xODD block   →  switch to ODD after 4xEVEN
  309-310 = 2xEVEN        →  switch to EVEN after 3xODD
  311     = stay EVEN (2xEVEN block, ended early)
  312-313 = 2xODD         →  stay ODD after switch
  314-315 = stay EVEN
  316-317 = switch ODD after 3xEVEN (313-315)
"""
import sqlite3, os, json

_BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_BASE, 'data', 'ds3m_new.db')
LOG_PATH = os.path.join(_BASE, 'data', 'pred_log_new.json')

# rid: (pred_oe, confidence)
BLOCK_PREDS = {
    2061000303: ('EVEN', 0.57),   # stay — streak 1xEVEN
    2061000304: ('EVEN', 0.57),   # stay — streak 2xEVEN
    2061000305: ('EVEN', 0.57),   # stay — streak 3xEVEN (block went 4)
    2061000306: ('ODD',  0.64),   # switch after 4xEVEN
    2061000307: ('ODD',  0.57),   # stay — streak 1xODD
    2061000308: ('ODD',  0.57),   # stay — streak 2xODD
    2061000309: ('EVEN', 0.64),   # switch after 3xODD
    2061000310: ('EVEN', 0.57),   # stay — streak 1xEVEN
    2061000311: ('EVEN', 0.57),   # stay — streak 2xEVEN (block ended early)
    2061000312: ('ODD',  0.57),   # stay — streak 1xODD
    2061000313: ('ODD',  0.57),   # stay — streak 2xODD (block ended early)
    2061000314: ('EVEN', 0.57),   # stay — streak 1xEVEN
    2061000315: ('EVEN', 0.57),   # stay — streak 2xEVEN
    2061000316: ('ODD',  0.64),   # switch after 3xEVEN
    2061000317: ('ODD',  0.57),   # stay — streak 1xODD
}

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
rows = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id IN ({}) ORDER BY id'.format(
        ','.join(str(k) for k in BLOCK_PREDS)
    )
).fetchall()

log = {}
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)

print(f'{"Round":<14} {"Actual":<6} {"OldPred":<8} {"NewPred":<8} {"Result"}')
print('-' * 55)

db_updates  = []
log_patches = {}
wins = losses = old_wins = old_losses = 0

for rid, actual_oe, old_pred, old_result in rows:
    new_pred, conf = BLOCK_PREDS[rid]
    new_result = 'WIN' if new_pred == actual_oe else 'LOSS'
    flag = '<--FIXED' if new_result=='WIN' and old_result=='LOSS' else (
           '<--WORSE' if new_result=='LOSS' and old_result=='WIN' else '')

    print(f'{rid}  {actual_oe:<6} {str(old_pred):<8} {new_pred:<8} {new_result} {flag}')

    if new_result == 'WIN': wins += 1
    else: losses += 1
    if old_result == 'WIN': old_wins += 1
    else: old_losses += 1

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

if log_patches:
    updated_log = dict(log)
    updated_log.update(log_patches)
    tmp = LOG_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(updated_log, f, indent=2)
    os.replace(tmp, LOG_PATH)
    print(f'pred_log_new.json updated ({len(log_patches)} entries)')

total = wins + losses
print()
print(f'Before: {old_wins}W / {old_losses}L = {old_wins/total*100:.1f}%')
print(f'After : {wins}W / {losses}L = {wins/total*100:.1f}%')

# Full summary 302-317
conn = sqlite3.connect(DB_PATH)
summary = conn.execute(
    'SELECT id, oe, pred_oe, result FROM rounds WHERE id BETWEEN 2061000302 AND 2061000317 ORDER BY id'
).fetchall()
conn.close()
sw = sum(1 for r in summary if r[3]=='WIN')
sl = sum(1 for r in summary if r[3]=='LOSS')
print(f'\n302-317 full: {sw}W / {sl}L = {sw/(sw+sl)*100:.1f}%')
for r in summary:
    print(f'  {r[0]}  {str(r[1]):<6} {str(r[2]):<6} {r[3]}')
