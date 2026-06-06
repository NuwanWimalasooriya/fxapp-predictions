"""
Retroactively re-apply updated predict_cg.py rules to rg3m.db records
with id > 2061000281.

Simulates running the predictor after each round's result comes in,
computing the running loss streak dynamically so all corrections
(streak_flip guard, dominant_stay, etc.) fire correctly.
"""
import os, sys, json, sqlite3
sys.stdout.reconfigure(encoding='utf-8')

FROM_ID = 2061000281  # re-predict everything AFTER this round

# ── Import predictor (loads DB into module globals) ───────────────────────────
import predict_cg as pcg

# ── Find the initial loss streak just before FROM_ID ─────────────────────────
def _initial_streak(log, ids_before_cutoff):
    """Count consecutive losses ending at the last round <= FROM_ID."""
    streak = 0
    for rid in reversed(ids_before_cutoff):
        entry = log.get(str(rid))
        if not entry or not entry.get('actual') or not entry.get('pred_color'):
            break
        won = entry['pred_color'] == entry['actual']
        if not won:
            streak += 1
        else:
            break
    return streak

cutoff_ids = [rid for rid in pcg.ids if rid <= FROM_ID]
init_streak = _initial_streak(pcg.log, cutoff_ids)
print(f"  Starting loss streak at {FROM_ID}: {init_streak}")

# ── Find start index ──────────────────────────────────────────────────────────
try:
    start_idx = next(i for i, rid in enumerate(pcg.ids) if rid > FROM_ID)
except StopIteration:
    print(f"No rounds after {FROM_ID} in DB. Nothing to do.")
    sys.exit(0)

total = len(pcg.ids) - start_idx
print(f"  Re-predicting {total} rounds ({pcg.ids[start_idx]} → {pcg.ids[-1]})")

# ── Run retroactive predictions ───────────────────────────────────────────────
BET_THRESHOLD = 0.65

running_streak = init_streak
updated_log    = dict(pcg.log)  # copy; we'll patch then save at end
db_updates     = []             # (pred_color, confidence, bet, result, id)

wins = losses = 0

for i in range(start_idx, len(pcg.ids)):
    rid        = pcg.ids[i]
    actual_col = pcg.colors[i]
    actual_num = pcg.numbers[i]

    # ── Override global loss streak for this prediction ───────────────────────
    pcg._cur_loss_streak = running_streak

    # ── Predict color ─────────────────────────────────────────────────────────
    pred_color, color_conf, signals = pcg.predict_color(i)
    if pred_color is None:
        continue

    p_red    = color_conf if pred_color == 'red' else 1.0 - color_conf
    is_bet   = color_conf >= BET_THRESHOLD
    result   = 'WIN' if pred_color == actual_col else 'LOSS'

    # ── Predict number (conditioned on predicted color) ───────────────────────
    pred_num, num_conf, num_sinfo = pcg.predict_number(i, pred_color)

    # ── Predict purple ────────────────────────────────────────────────────────
    pred_purple, purple_conf, purple_sinfo = pcg.predict_purple(i)

    # ── Update running streak ─────────────────────────────────────────────────
    if pred_color == actual_col:
        running_streak = 0
        wins += 1
    else:
        running_streak += 1
        losses += 1

    # ── Build log entry ───────────────────────────────────────────────────────
    updated_log[str(rid)] = {
        'pred_color':    pred_color,
        'p_red':         round(p_red, 4),
        'confidence':    round(color_conf, 4),
        'bet':           is_bet,
        'pred_number':   pred_num,
        'number_conf':   round(num_conf, 4),
        'number_signals': num_sinfo,
        'pred_purple':   pred_purple,
        'purple_conf':   round(purple_conf, 4),
        'purple_signals': purple_sinfo,
        'signals':       signals,
        'actual':        actual_col,
        'actual_number': actual_num,
        'actual_purple': actual_num in pcg.PURPLE_NUMS,
        'timestamp':     pcg.log.get(str(rid), {}).get('timestamp', ''),
    }

    # ── Queue DB update ───────────────────────────────────────────────────────
    actual_purple_val = 1 if actual_num in pcg.PURPLE_NUMS else 0
    num_result = 'HIT' if pred_num == actual_num else 'MISS'
    db_updates.append((
        pred_color, round(color_conf, 4), 1 if is_bet else 0, result,
        pred_num, 1 if pred_purple else 0, num_result,
        rid
    ))

    if (i - start_idx + 1) % 20 == 0 or i == len(pcg.ids) - 1:
        pct = (i - start_idx + 1) / total * 100
        wr  = wins / (wins + losses) * 100 if (wins + losses) else 0
        print(f"  [{pct:5.1f}%] round {rid}  WR={wr:.1f}%  streak={running_streak}")

# ── Write pred_log ────────────────────────────────────────────────────────────
tmp_log = pcg.LOG_PATH + '.tmp'
with open(tmp_log, 'w', encoding='utf-8') as f:
    json.dump(updated_log, f, indent=2)
os.replace(tmp_log, pcg.LOG_PATH)
print(f"\n  pred_log_cg.json updated ({len(updated_log)} entries)")

# ── Write DB ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect(pcg.DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
conn.executemany(
    '''UPDATE rounds
       SET pred_color=?, confidence=?, bet=?, result=?,
           pred_number=?, pred_purple=?, number_result=?
       WHERE id=?''',
    db_updates
)
conn.commit()
conn.close()
print(f"  DB updated: {len(db_updates)} rows")

# ── Final summary ─────────────────────────────────────────────────────────────
total_scored = wins + losses
wr_final = wins / total_scored * 100 if total_scored else 0
print(f"\n  Done. {wins}W / {losses}L = {wr_final:.1f}% win rate over {total_scored} rounds")
print(f"  (from round {pcg.ids[start_idx]} → {pcg.ids[-1]})")
