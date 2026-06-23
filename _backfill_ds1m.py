"""
Backfill predictions for rounds 2062401157-2062401171 in ds1m.db.
Applies the new 3 rules:
  1. Rule 1 / 1b: XGBoost blend only when it agrees with streak direction
  2. Rule 5 (streak_flip): restricted to non-structural predictions only

Run once after updating predict_ds1m.py.
"""
import sys, os, json, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

_BASE    = os.path.dirname(os.path.abspath(__file__))
_DATA    = os.path.join(_BASE, 'data')
DB_PATH  = os.path.join(_DATA, 'ds1m.db')
LOG_PATH = os.path.join(_DATA, 'pred_log_ds1m.json')

BACKFILL_START = 2062401157
BACKFILL_END   = 2062401171

# ── Step 1: Remove target entries from log ────────────────────────────────────
print(f"[1] Removing log entries {BACKFILL_START}–{BACKFILL_END}...")
with open(LOG_PATH, encoding='utf-8-sig') as f:
    log = json.load(f)

removed = [k for k in list(log.keys()) if BACKFILL_START <= int(k) <= BACKFILL_END]
for k in removed:
    del log[k]
print(f"    Removed {len(removed)} entries: {[int(k) for k in removed]}")

tmp = LOG_PATH + '.tmp'
with open(tmp, 'w') as f:
    json.dump(log, f, indent=2)
os.replace(tmp, LOG_PATH)

# ── Step 2: Load actual OE values from DB ─────────────────────────────────────
print("[2] Loading actual OE for backfill range from DB...")
conn = sqlite3.connect(DB_PATH)
db_rows = conn.execute(
    f'SELECT id, disc1||disc2||disc3||disc4 FROM rounds '
    f'WHERE id BETWEEN {BACKFILL_START} AND {BACKFILL_END} ORDER BY id'
).fetchall()
conn.close()

actual_pattern = {rid: pat for rid, pat in db_rows}
actual_oe_map  = {
    rid: ('ODD' if pat.count('R') % 2 else 'EVEN')
    for rid, pat in actual_pattern.items()
}
print(f"    Found {len(db_rows)} rounds in DB.")
for rid in sorted(actual_oe_map):
    print(f"      {rid}: {actual_oe_map[rid]}")

# ── Step 3: Import predict_ds1m (runs gap-fill for the 15 missing rounds) ────
print("\n[3] Importing predict_ds1m — gap-fill will run for missing entries...")
import predict_ds1m as m
print("    Import complete.")

# ── Step 4: Re-predict with correct per-round streak ─────────────────────────
print("\n[4] Re-predicting with per-round streak tracking...")

# Compute initial streak from entries strictly BEFORE BACKFILL_START
pre_entries = sorted(
    [(int(k), v) for k, v in m.log_new.items()
     if v.get('actual') and v.get('pred_oe') and int(k) < BACKFILL_START],
    key=lambda x: x[0], reverse=True
)
initial_streak = 0
for _, entry in pre_entries:
    act = entry['actual']
    act_oe = 'ODD' if act.count('R') % 2 else 'EVEN'
    if entry['pred_oe'] != act_oe:
        initial_streak += 1
    else:
        break
print(f"    Initial streak before round {BACKFILL_START}: {initial_streak}")

# Index lookup: round_id → position in m.df
id_to_idx = {int(row['id']): i for i, row in m.df.iterrows()}

current_streak = initial_streak
wins = losses = 0

for target_id in sorted(actual_oe_map.keys()):
    bi = id_to_idx.get(target_id)
    if bi is None:
        print(f"    SKIP {target_id}: not found in df")
        continue
    if bi < 1:
        print(f"    SKIP {target_id}: no prior context (bi={bi})")
        continue

    # Set module-level streak for this prediction
    m._cur_loss_streak = current_streak

    # Predict using data up to (but not including) this round
    pred_oe, p_odd, sinfo = m.predict_ds1m(m.oe_list, bi - 1)
    if pred_oe is None:
        print(f"    SKIP {target_id}: predict returned None")
        continue

    conf      = round(max(p_odd, 1.0 - p_odd), 4)
    actual    = actual_oe_map[target_id]
    result    = 'WIN' if pred_oe == actual else 'LOSS'
    rule_tag  = sinfo.get('rule', '?')

    print(f"    {target_id}: pred={pred_oe:<4} actual={actual:<4} {result:<4}  "
          f"streak={current_streak}  rule={rule_tag}")

    # Update running streak
    if result == 'WIN':
        current_streak = 0
        wins += 1
    else:
        current_streak += 1
        losses += 1

    # Overwrite log entry with per-round-streak prediction
    m.log_new[str(target_id)] = {
        'round_id':   str(target_id),
        'pred_oe':    pred_oe,
        'p_oe_odd':   round(p_odd, 4),
        'confidence': conf,
        'actual':     actual_pattern[target_id],
        'signals':    sinfo,
    }

print(f"\n    Result: {wins} WIN, {losses} LOSS over {wins+losses} rounds")

# ── Step 5: Save updated log ──────────────────────────────────────────────────
print("\n[5] Saving updated log...")
tmp = LOG_PATH + '.tmp'
with open(tmp, 'w') as f:
    json.dump(m.log_new, f, indent=2)
os.replace(tmp, LOG_PATH)
print("    Log saved.")

# ── Step 6: Update DB ─────────────────────────────────────────────────────────
print("\n[6] Updating DB...")
conn = sqlite3.connect(DB_PATH)
conn.execute('PRAGMA journal_mode=WAL')
db_updates = []
for target_id in sorted(actual_oe_map.keys()):
    entry = m.log_new.get(str(target_id))
    if not entry or not entry.get('pred_oe'):
        continue
    p       = float(entry.get('p_oe_odd', 0.5))
    cv      = round(max(p, 1.0 - p), 4)
    act_pat = entry.get('actual', '')
    act_oe  = 'ODD' if act_pat.count('R') % 2 else 'EVEN' if act_pat else ''
    res     = 'WIN' if entry['pred_oe'] == act_oe else 'LOSS'
    db_updates.append((entry['pred_oe'], cv, res, target_id))

conn.executemany(
    'UPDATE rounds SET pred_oe=?, confidence=?, result=? WHERE id=?',
    db_updates
)
conn.commit()
conn.close()
print(f"    Updated {len(db_updates)} rows in DB.")

print("\nDone.")
