import json, os

log_path = os.path.join('data', 'pred_log_cg.json')
with open(log_path, encoding='utf-8-sig') as f:
    log = json.load(f)

print('=== Issue 1: sym_complete + dominant_stay (rounds 432-436) ===')
print('Round      Actual  OldPred  OldResult  Rule                  cur_run  maj6  NewPred')
print('-' * 100)
for rid_int in range(2060900432, 2060900437):
    rid = str(rid_int)
    entry = log.get(rid)
    if not entry:
        continue
    s = entry.get('signals', {})
    pred = entry['pred_color']
    actual = entry.get('actual', '?')
    result = 'WIN' if pred == actual else 'LOSS'
    rule = s.get('rule', '?')
    cr = s.get('cur_run', 0)
    cv = s.get('cur_val', '?')
    maj6 = s.get('maj6')
    sym = s.get('sym_block_info', {}) or {}
    # Simulate new rule
    new_pred = pred
    if sym.get('phase') == 'complete' and maj6 == cv:
        new_pred = cv  # dominant_stay override
    new_result = 'WIN' if new_pred == actual else 'LOSS'
    flag = '<-- FIXED' if new_pred != pred and new_result == 'WIN' else ''
    print(f'{rid}  {actual:<7} {pred:<8} {result:<10} {rule:<22} {cr}  {str(maj6):<6}  {new_pred} {flag}')

print()
print('=== Issue 2: streak >= 3 + alternating — streak_flip guard (rounds 300-319) ===')
print('Round      Actual  OldPred  OldResult  Rule                       cr  alt  maj6  NewPred')
print('-' * 105)
for rid_int in range(2061000300, 2061000320):
    rid = str(rid_int)
    entry = log.get(rid)
    if not entry:
        continue
    s = entry.get('signals', {})
    pred = entry['pred_color']
    actual = entry.get('actual', '?')
    result = 'WIN' if pred == actual else 'LOSS'
    rule = s.get('rule', '?')
    cr = s.get('cur_run', 0)
    cv = s.get('cur_val', '?')
    alt = s.get('alt_run', 0)
    maj6 = s.get('maj6')
    had_flip = '+streak_flip' in rule
    # Simulate new streak_flip guard
    # Flip would NOT fire if: cr >= 3 OR alt >= 3 OR maj6 == cv
    flip_blocked = cr >= 3 or alt >= 3 or maj6 == cv
    if had_flip and flip_blocked:
        # Undo the flip to get new prediction
        # The flip inverted p_red; undoing it means going back to original direction
        # Since flip reverses pred, the new pred = opposite of old pred
        new_pred = 'red' if pred == 'green' else 'green'
    else:
        new_pred = pred
    new_result = 'WIN' if new_pred == actual else 'LOSS'
    flag = '<-- FIXED' if new_pred != pred and new_result == 'WIN' else ('<-- WORSE' if new_pred != pred and new_result == 'LOSS' else '')
    print(f'{rid}  {actual:<7} {pred:<8} {result:<10} {rule:<26} {cr}  {alt}  {str(maj6):<6}  {new_pred} {flag}')
