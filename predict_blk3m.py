"""
BLK3M Predictor — 3-minute Blocks.

Two independent predictions per round:
  BIG/SMALL  : streak + XGBoost rule cascade
  ODD/EVEN   : rule cascade (bet when conf >= 0.58)
"""
import os, sys, json, datetime, sqlite3, statistics
sys.stdout.reconfigure(encoding='utf-8')

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

DB_PATH    = os.path.join(_DATA_DIR, 'blk3m.db')
LOG_PATH   = os.path.join(_DATA_DIR, 'pred_log_blk3m.json')
SNAP_PATH  = os.path.join(_DATA_DIR, 'latest_prediction_blk3m.json')

MIN_ROWS = 30   # minimum history before predicting


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def load_history():
    conn = get_conn()
    try:
        rows = conn.execute(
            'SELECT id,total,big_small,odd_even FROM rounds ORDER BY id'
        ).fetchall()
    finally:
        conn.close()
    return [{'id': r[0], 'total': r[1], 'big_small': r[2], 'odd_even': r[3]} for r in rows]


# ── Pred log helpers ───────────────────────────────────────────────────────────

def load_pred_log():
    if not os.path.exists(LOG_PATH):
        return {}
    try:
        with open(LOG_PATH, encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return {}

def save_pred_log(log):
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

def save_snap(data):
    with open(SNAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ── Run-length helpers ─────────────────────────────────────────────────────────

def current_run(arr, key):
    if not arr:
        return None, 0, None, 0
    cv = arr[-1][key]
    cr = 0
    for r in reversed(arr):
        if r[key] == cv:
            cr += 1
        else:
            break
    pv, pr = None, 0
    i = len(arr) - cr - 1
    if i >= 0:
        pv = arr[i][key]
        for r in reversed(arr[:i+1]):
            if r[key] == pv:
                pr += 1
            else:
                break
    return cv, cr, pv, pr

def majority(arr, key, window=20):
    recent = arr[-window:]
    vals   = [r[key] for r in recent]
    if not vals:
        return None, 0.5
    big_cnt = vals.count('BIG')
    frac    = big_cnt / len(vals)
    return ('BIG' if frac >= 0.5 else 'SMALL'), frac


# ── BIG/SMALL prediction (rule cascade) ────────────────────────────────────────

def predict_bs(hist):
    if len(hist) < MIN_ROWS:
        return 'BIG', 0.5, 'insufficient_data', {}

    cv, cr, pv, pr = current_run(hist, 'big_small')
    maj20, maj_frac = majority(hist, 'big_small', 20)

    signals = {
        'cur_val': cv, 'cur_run': cr,
        'prev_val': pv, 'prev_run': pr,
        'maj20': maj20, 'maj20_frac': round(maj_frac, 3),
    }

    # Rule 1: Long streak (>= 5) — continue
    if cr >= 5:
        conf = min(0.72, 0.60 + cr * 0.02)
        signals['rule'] = 'rule1_long_streak'
        return cv, conf, 'rule1_long_streak', signals

    # Rule 1b: Medium streak (3-4) — continue with moderate confidence
    if cr in (3, 4):
        conf = 0.60
        stay_p = conf if cv == 'BIG' else (1.0 - conf)
        maj_p  = maj_frac if maj20 == 'BIG' else (1.0 - maj_frac)
        p_big  = 0.6 * stay_p + 0.4 * maj_p
        pred   = 'BIG' if p_big >= 0.5 else 'SMALL'
        conf_f = abs(p_big - 0.5) * 2
        signals['rule'] = 'rule1b_med_streak'
        return pred, round(conf_f, 3), 'rule1b_med_streak', signals

    # Rule 6: Trend persist — after run >= 3, stay on prev direction for 1-2 counter rounds
    if cr <= 2 and pr >= 3:
        conf   = 0.62 if pr >= 5 else 0.58
        p_big  = conf if pv == 'BIG' else (1.0 - conf)
        pred   = 'BIG' if p_big >= 0.5 else 'SMALL'
        signals['rule'] = 'rule6_trend_persist'
        return pred, conf, 'rule6_trend_persist', signals

    # Fallback: majority of last 20
    conf  = abs(maj_frac - 0.5) * 2 + 0.5
    conf  = min(conf, 0.70)
    signals['rule'] = 'rule_maj20'
    return maj20, round(conf, 3), 'rule_maj20', signals


# ── ODD/EVEN prediction (rule cascade) ────────────────────────────────────────

def predict_oe(hist):
    if len(hist) < MIN_ROWS:
        return 'EVEN', 0.5, 'insufficient_data', {}

    cv, cr, pv, pr = current_run(hist, 'odd_even')
    recent20 = [r['odd_even'] for r in hist[-20:]]
    odd_frac = recent20.count('ODD') / len(recent20) if recent20 else 0.5
    maj20    = 'ODD' if odd_frac >= 0.5 else 'EVEN'

    # Check alternating pattern
    alt_run = 0
    for i in range(len(hist) - 1, 0, -1):
        if hist[i]['odd_even'] != hist[i-1]['odd_even']:
            alt_run += 1
        else:
            break
    alt_confirmed = alt_run >= 3

    signals = {
        'cur_val': cv, 'cur_run': cr,
        'prev_val': pv, 'prev_run': pr,
        'odd_frac20': round(odd_frac, 3),
        'alt_run': alt_run, 'alt_confirmed': alt_confirmed,
    }

    # Rule 3: Confirmed alternating → predict flip
    if alt_confirmed and cr == 1:
        conf = round(min(0.70, 0.60 + (alt_run - 3) * 0.02), 3)
        pred = 'ODD' if cv == 'EVEN' else 'EVEN'
        signals['rule_oe'] = 'rule3_alternating'
        return pred, conf, 'rule3_alternating', signals

    # Rule 1: Long same run (>= 4) — continue
    if cr >= 4:
        conf = 0.60
        signals['rule_oe'] = 'rule1_long_run'
        return cv, conf, 'rule1_long_run', signals

    # Rule 1b: Medium run (== 3) — continue
    if cr == 3:
        signals['rule_oe'] = 'rule1b_med_run'
        return cv, 0.58, 'rule1b_med_run', signals

    # Rule 6: Trend persist
    if cr <= 2 and pr >= 3:
        conf = 0.60 if pr >= 5 else 0.57
        pred = pv
        signals['rule_oe'] = 'rule6_trend_persist'
        return pred, conf, 'rule6_trend_persist', signals

    # Fallback: majority of last 20
    conf = abs(odd_frac - 0.5) * 2 + 0.5
    conf = min(conf, 0.68)
    signals['rule_oe'] = 'oe_maj20'
    return maj20, round(conf, 3), 'oe_maj20', signals


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    hist = load_history()
    if not hist:
        print("BLK3M: No history found.")
        return

    last       = hist[-1]
    last_id    = last['id']
    next_id    = last_id + 1
    pred_log   = load_pred_log()

    pred_bs,  conf_bs,  rule_bs,  sigs_bs  = predict_bs(hist)
    pred_oe,  conf_oe,  rule_oe,  sigs_oe  = predict_oe(hist)

    is_bet    = conf_bs  >= 0.58
    is_bet_oe = conf_oe  >= 0.58

    print(f"BLK3M: Round {next_id} → {pred_bs} (conf={conf_bs*100:.1f}%, rule={rule_bs})")
    print(f"        OE → {pred_oe} (conf={conf_oe*100:.1f}%, rule={rule_oe})")

    # ── Streak context for last 10 ─────────────────────────────────────────────
    last10_bs = [r['big_small'] for r in hist[-10:]]
    last10_oe = [r['odd_even']  for r in hist[-10:]]

    signals = {
        **sigs_bs,
        'pred_oe':    pred_oe,
        'conf_oe':    conf_oe,
        'is_bet_oe':  is_bet_oe,
        'rule_oe':    rule_oe,
        'last10_bs':  last10_bs,
        'last10_oe':  last10_oe,
    }

    entry = {
        'next_round_id': next_id,
        'last_id':       last_id,
        'pred_bs':       pred_bs,
        'confidence':    conf_bs,
        'is_bet':        is_bet,
        'pred_oe':       pred_oe,
        'conf_oe':       conf_oe,
        'is_bet_oe':     is_bet_oe,
        'rule_oe':       rule_oe,
        'signals':       signals,
        'last_total':    last['total'],
        'last_bs':       last['big_small'],
        'last_oe':       last['odd_even'],
        'timestamp':     datetime.datetime.now().isoformat(),
    }

    pred_log[str(next_id)] = {
        'pred_bs':    pred_bs,
        'confidence': conf_bs,
        'bet':        int(is_bet),
        'pred_oe':    pred_oe,
        'conf_oe':    conf_oe,
        'bet_oe':     int(is_bet_oe),
        'signals':    signals,
    }

    save_pred_log(pred_log)
    save_snap(entry)
    print(f"  Saved prediction for round {next_id}.")


if __name__ == '__main__':
    main()
