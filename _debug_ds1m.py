"""Trace which rule fires at each round from 2062401158."""
import sqlite3, os, sys
sys.stdout.reconfigure(encoding='utf-8')

db = os.path.join('data', 'ds1m.db')
conn = sqlite3.connect(db)
rows = conn.execute(
    'SELECT id,disc1,disc2,disc3,disc4,pred_oe,confidence,result FROM rounds '
    'WHERE id >= 2062401140 ORDER BY id LIMIT 50'
).fetchall()
conn.close()

def oe(disc1,disc2,disc3,disc4):
    return 'ODD' if (disc1+disc2+disc3+disc4).count('R') % 2 else 'EVEN'

oe_list = [oe(*r[1:5]) for r in rows]
ids     = [r[0] for r in rows]

def build_runs(arr):
    if not arr: return []
    runs, cur, cnt = [], arr[0], 1
    for v in arr[1:]:
        if v == cur: cnt += 1
        else: runs.append((cur, cnt)); cur, cnt = v, 1
    runs.append((cur, cnt))
    return runs

loss_streak = 0

print(f'{"Round":<15} {"OE":<6} {"Pred":<6} {"Conf":<7} {"R":<5} {"Rule (simplified)"}')
print('-'*85)

for ti in range(10, len(rows)-1):
    arr     = oe_list[:ti+1]
    actual  = oe_list[ti+1] if ti+1 < len(rows) else '?'
    pred_db = rows[ti][5]
    conf_db = rows[ti][6]
    res_db  = rows[ti][7]

    runs     = build_runs(arr)
    cur_val  = runs[-1][0]; cur_run  = runs[-1][1]
    prev_val = runs[-2][0] if len(runs)>=2 else None
    prev_run = runs[-2][1] if len(runs)>=2 else 0

    # alt_run
    alt_run = 1
    for i in range(len(arr)-2, max(-1, len(arr)-32), -1):
        if arr[i] != arr[i+1]: alt_run += 1
        else: break
    alt_ok = alt_run >= 3 and cur_run == 1 and prev_run == 1

    # Determine rule (simplified, matching code logic)
    structural = False
    if cur_run >= 4:
        rule = f'Rule1 long_block cr={cur_run}'
        rule_pred = cur_val if cur_run <= 5 else ('ODD' if cur_val=='EVEN' else 'EVEN')
        structural = True
    elif cur_run == 3:
        rule = 'Rule1b stay cr=3'
        rule_pred = cur_val
        structural = True
    elif alt_ok:
        rule = f'Rule3 alt alt_run={alt_run}'
        rule_pred = 'ODD' if cur_val == 'EVEN' else 'EVEN'
        structural = True
    elif cur_run <= 2 and prev_run >= 3:
        rule = f'Rule6 trend_persist pv={prev_val} pr={prev_run}'
        rule_pred = prev_val
        structural = True
    else:
        rule = f'Maj20/XGB cr={cur_run} pr={prev_run}'
        rule_pred = '?'

    # Fixed Rule 5: streak_flip only applies to non-structural predictions
    flip = '+FLIP' if (loss_streak >= 5 and not structural) else ''
    if flip:
        rule_pred = ('ODD' if rule_pred=='EVEN' else 'EVEN') if rule_pred != '?' else '?'

    hit = (pred_db == actual) if actual != '?' else None
    marker = '' if hit else '<< LOSS'
    if ids[ti] >= 2062401158:
        print(f'{ids[ti]:<15} {cur_val:<6} {pred_db or "?":<6} {conf_db or 0:<7.4f} {res_db or "?":<5} {rule}{flip}  {marker}')

    if res_db == 'LOSS': loss_streak += 1
    elif res_db == 'WIN': loss_streak = 0
