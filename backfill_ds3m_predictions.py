"""
Retroactively re-apply the dominant OE trend signal to ds3m.db records
with id >= 2061000295.

For each scored round:
  - Compute which OE direction is dominant in the last 25 rounds of history
  - Blend that signal (60%) with the existing model prediction (40%)
  - Update pred_oe, confidence, result in the DB and pred_log
"""
import os, sys, json, sqlite3
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

FROM_ID   = 2061000295
DOM_WIN   = 25          # look-back window for dominant OE signal
DOM_BLEND = 0.60        # weight of dominant signal vs existing model

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

def _load_env():
    env = {}
    p = os.path.join(_BASE, '.env')
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env     = _load_env()
DB_PATH  = os.path.join(_DATA_DIR, _env.get('DB_FILE', 'ds3m.db'))
LOG_PATH = os.path.join(_DATA_DIR, 'pred_log.json')

# ── Load full OE history from DB ──────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
all_rows = conn.execute(
    'SELECT id, disc1, disc2, disc3, disc4, oe FROM rounds ORDER BY id'
).fetchall()
conn.close()

all_ids = [r[0] for r in all_rows]
all_oe  = [r[5] for r in all_rows]   # 'ODD' or 'EVEN'

id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

# ── Load pred_log ─────────────────────────────────────────────────────────────
log = {}
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)

# ── Dominant OE signal ────────────────────────────────────────────────────────
def dominant_oe_signal(oe_history, window=DOM_WIN):
    recent = oe_history[-window:] if len(oe_history) >= window else oe_history
    if len(recent) < 10:
        return None, 0.0
    n_odd  = sum(1 for x in recent if x == 'ODD')
    n_even = len(recent) - n_odd
    margin = abs(n_odd - n_even) / len(recent)
    if margin < 0.10:
        return None, 0.0
    dominant = 'ODD' if n_odd > n_even else 'EVEN'
    strength = min(0.20, margin * 1.8)
    p_odd = (0.5 + strength) if dominant == 'ODD' else (0.5 - strength)
    return dominant, p_odd

# ── Process rounds >= FROM_ID ─────────────────────────────────────────────────
target_rows = [(rid, r[1]+r[2]+r[3]+r[4], r[5])
               for r, rid in zip(all_rows, all_ids)
               if rid >= FROM_ID]

print(f'Updating {len(target_rows)} rounds from {FROM_ID}')
print()
print(f'{"Round":<14} {"Actual":<6} {"OldPred":<8} {"NewPred":<8} {"Dominant":<8} {"Result"}')
print('-' * 60)

db_updates  = []
log_patches = {}
wins = losses = unchanged = 0

for rid, pat, actual_oe in target_rows:
    idx = id_to_idx[rid]
    # History = all OE values before this round (not including this one)
    oe_history = all_oe[:idx]

    # Compute dominant signal
    _, dom_p_odd = dominant_oe_signal(oe_history)

    # Get existing log entry
    entry = log.get(str(rid), {})
    old_pred_oe  = entry.get('pred_oe', '')
    old_p_odd    = entry.get('p_oe_odd', 0.5)

    if dom_p_odd == 0.0:
        # No dominant signal — keep existing prediction unchanged
        new_p_odd   = old_p_odd
        new_pred_oe = old_pred_oe if old_pred_oe else ('ODD' if new_p_odd > 0.5 else 'EVEN')
        unchanged += 1
    else:
        # Blend: dominant signal 60%, existing model 40%
        new_p_odd   = DOM_BLEND * dom_p_odd + (1 - DOM_BLEND) * float(old_p_odd)
        new_p_odd   = max(0.28, min(0.72, new_p_odd))
        new_pred_oe = 'ODD' if new_p_odd > 0.5 else 'EVEN'

    new_conf   = max(new_p_odd, 1.0 - new_p_odd)
    new_result = 'WIN' if new_pred_oe == actual_oe else 'LOSS'

    dominant_label = 'EVEN' if new_p_odd < 0.5 else 'ODD'
    flag = ''
    if old_pred_oe and new_pred_oe != old_pred_oe:
        flag = '<--FIXED' if new_result == 'WIN' else '<--WORSE'

    print(f'{rid}  {actual_oe:<6} {str(old_pred_oe):<8} {new_pred_oe:<8} {dominant_label:<8} {new_result} {flag}')

    if new_result == 'WIN':
        wins += 1
    else:
        losses += 1

    # Patch log entry
    if entry:
        patched = dict(entry)
        patched['pred_oe']   = new_pred_oe
        patched['p_oe_odd']  = round(new_p_odd, 4)
        patched['confidence'] = round(new_conf, 4)
        log_patches[str(rid)] = patched

    db_updates.append((new_pred_oe, round(new_conf, 4), new_result, rid))

# ── Write pred_log ────────────────────────────────────────────────────────────
updated_log = dict(log)
updated_log.update(log_patches)
tmp = LOG_PATH + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(updated_log, f, indent=2)
os.replace(tmp, LOG_PATH)

# ── Write DB ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
conn.executemany(
    'UPDATE rounds SET pred_oe=?, confidence=?, result=? WHERE id=?',
    db_updates
)
conn.commit()
conn.close()

total = wins + losses
wr = wins / total * 100 if total else 0
print()
print(f'Done.  {wins}W / {losses}L = {wr:.1f}% win rate over {total} rounds')
print(f'Unchanged (no dominant signal): {unchanged}')
print(f'pred_log.json and ds3m.db updated.')
