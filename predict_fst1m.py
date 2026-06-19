"""
FST1M Predictor — 1-minute 4-Star Lottery.

Each round: 4 numbers (0-9), sum 0-36.
Three independent predictions:
  BIG/SMALL  : sum >= 18 vs sum <= 17  (XGBoost + rule cascade)
  ODD/EVEN   : sum parity               (rule cascade, bet when conf >= 0.58)
  Sum zone   : LOW(0-13) / MID(14-22) / HIGH(23-36)  (trend-based)

Rule cascade (BIG/SMALL):
  0.  Low-streak regime (max run <= 2 in last 20) → majority of last 20
  1.  Long streak (run >= 5)                       → continue strongly
  1b. Medium streak (run == 3 or 4)                → continue + XGBoost blend
  6.  Trend persist (run <= 2, prev >= 3)          → stay on previous trend
  Maj20. Oscillating (run <= 2, no prev streak)   → majority of last 20 + XGBoost
  XGB. Fallback                                    → pure XGBoost
"""
import os, sys, json, datetime, sqlite3, statistics, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

DB_PATH        = os.path.join(_DATA_DIR, 'fst1m.db')
LOG_PATH       = os.path.join(_DATA_DIR, 'pred_log_fst1m.json')
SNAP_PATH      = os.path.join(_DATA_DIR, 'latest_prediction_fst1m.json')
XGB_MODEL_PATH = os.path.join(_DATA_DIR, 'xgb_model_fst1m.pkl')
XGB_META_PATH  = os.path.join(_DATA_DIR, 'xgb_meta_fst1m.json')

MIN_TRAIN = 200


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def load_history():
    conn = get_conn()
    try:
        rows = conn.execute(
            'SELECT id,n1,n2,n3,n4,total,big_small,odd_even FROM rounds ORDER BY id'
        ).fetchall()
    finally:
        conn.close()
    return [{'id': r[0], 'n1': r[1], 'n2': r[2], 'n3': r[3], 'n4': r[4],
             'total': r[5], 'big_small': r[6], 'odd_even': r[7]} for r in rows]


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

def current_run(arr, key='big_small'):
    """Return (current_val, run_length, prev_val, prev_run_length)."""
    if not arr:
        return None, 0, None, 0
    cv = arr[-1][key]
    cr = 0
    for r in reversed(arr):
        if r[key] == cv:
            cr += 1
        else:
            break
    pv = arr[-(cr + 1)][key] if len(arr) > cr else None
    pr = 0
    if pv:
        for r in reversed(arr[:len(arr) - cr]):
            if r[key] == pv:
                pr += 1
            else:
                break
    return cv, cr, pv, pr

def detect_low_streak_regime(arr, key='big_small', window=20):
    """True when max run in last `window` rounds is <= 2 (oscillating regime)."""
    recent = arr[-window:] if len(arr) >= window else arr
    if len(recent) < 4:
        return False
    max_run = cur = 1
    for i in range(1, len(recent)):
        if recent[i][key] == recent[i-1][key]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run <= 2

def majority_binary(arr, key, val_pos, window=20):
    """Return (dominant_val, fraction_of_dominant) for any binary key."""
    recent = arr[-window:] if len(arr) >= window else arr
    if not recent:
        return None, 0.5
    pos_n = sum(1 for r in recent if r[key] == val_pos)
    frac  = pos_n / len(recent)
    if frac > 0.5:
        return val_pos, round(frac, 4)
    elif frac < 0.5:
        neg = next((r[key] for r in recent if r[key] != val_pos), None)
        return neg, round(1 - frac, 4)
    return None, 0.5

def majority(arr, key='big_small', window=20):
    return majority_binary(arr, key, 'BIG', window)

def loss_streak(arr, pred_log):
    """Count consecutive BIG/SMALL losses at end of history."""
    streak = 0
    for r in reversed(arr):
        entry = pred_log.get(str(r['id']))
        if not entry or not entry.get('pred_bs'):
            break
        if entry['pred_bs'] != r['big_small']:
            streak += 1
        else:
            break
    return streak


# ── XGBoost features ───────────────────────────────────────────────────────────

def make_features(arr, idx):
    """Build feature vector for XGBoost at position idx (predicting idx+1)."""
    if idx < 20:
        return None
    hist = arr[:idx + 1]

    cv, cr, pv, pr = current_run(hist)
    recent5  = hist[-5:]
    recent10 = hist[-10:]
    recent20 = hist[-20:]

    big_r5  = sum(1 for r in recent5  if r['big_small'] == 'BIG') / 5
    big_r10 = sum(1 for r in recent10 if r['big_small'] == 'BIG') / 10
    big_r20 = sum(1 for r in recent20 if r['big_small'] == 'BIG') / 20
    odd_r10 = sum(1 for r in recent10 if r['odd_even']  == 'ODD') / 10

    sums5  = [r['total'] for r in recent5]
    sums10 = [r['total'] for r in recent10]
    avg5   = sum(sums5)  / 5
    avg10  = sum(sums10) / 10
    last_s = hist[-1]['total']
    trend  = (avg5 - avg10) / 18.0

    max_r10 = cur_r = 1
    for i in range(1, len(recent10)):
        if recent10[i]['big_small'] == recent10[i-1]['big_small']:
            cur_r += 1; max_r10 = max(max_r10, cur_r)
        else:
            cur_r = 1

    digits_last5 = [d for r in recent5 for d in [r['n1'],r['n2'],r['n3'],r['n4']]]
    high_digit   = sum(1 for d in digits_last5 if d >= 7) / len(digits_last5)
    low_digit    = sum(1 for d in digits_last5 if d <= 2) / len(digits_last5)

    return [
        1 if cv == 'BIG' else 0,
        min(cr, 10) / 10,
        1 if pv == 'BIG' else 0,
        min(pr, 10) / 10,
        last_s / 36,
        avg5 / 36,
        avg10 / 36,
        trend,
        big_r5,
        big_r10,
        big_r20,
        odd_r10,
        max_r10 / 10,
        high_digit,
        low_digit,
    ]


# ── XGBoost train / load ───────────────────────────────────────────────────────

def _need_retrain(hist):
    n = len(hist)
    if n < MIN_TRAIN:
        return False
    if not os.path.exists(XGB_META_PATH):
        return True
    try:
        with open(XGB_META_PATH) as f:
            meta = json.load(f)
        return n - meta.get('trained_on', 0) >= 200
    except Exception:
        return True

def _train(hist):
    try:
        from xgboost import XGBClassifier
        import joblib
    except ImportError:
        print("  [WARN] xgboost/joblib not installed — skipping XGBoost training.")
        return None

    X, y = [], []
    for i in range(20, len(hist) - 1):
        feats = make_features(hist, i)
        if feats is None:
            continue
        label = 1 if hist[i + 1]['big_small'] == 'BIG' else 0
        X.append(feats); y.append(label)

    if len(X) < 100:
        return None

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=42, n_jobs=-1)
    model.fit(np.array(X, dtype=np.float32), y)
    joblib.dump(model, XGB_MODEL_PATH)
    with open(XGB_META_PATH, 'w') as f:
        json.dump({'trained_on': len(hist), 'features': 15,
                   'timestamp': datetime.datetime.now().isoformat()}, f)
    print(f"  [XGB] Trained on {len(X)} samples.")
    return model


xgb_model = None

def _init_xgb(hist):
    global xgb_model
    if _need_retrain(hist):
        print("  Training XGBoost (FST1M)...", flush=True)
        xgb_model = _train(hist)
    elif os.path.exists(XGB_MODEL_PATH):
        try:
            import joblib
            xgb_model = joblib.load(XGB_MODEL_PATH)
        except Exception:
            pass


# ── ODD/EVEN prediction (rule cascade) ────────────────────────────────────────

def predict_odd_even(hist):
    """
    Rule cascade for ODD/EVEN, mirroring the BIG/SMALL logic.
    Returns (pred_oe, conf_oe, p_odd, is_bet_oe, rule_oe).
    """
    if len(hist) < 5:
        return 'EVEN', 0.50, 0.50, False, 'oe_default'

    cv, cr, pv, pr = current_run(hist, key='odd_even')
    low_oe          = detect_low_streak_regime(hist, key='odd_even')
    maj_val, maj_frac = majority_binary(hist, 'odd_even', 'ODD')

    p_odd   = 0.5
    rule_oe = 'oe_default'

    if low_oe:
        if maj_val == 'ODD':
            p_odd = 0.50 + (maj_frac - 0.5) * 0.80
        elif maj_val == 'EVEN':
            p_odd = 0.50 - (maj_frac - 0.5) * 0.80
        rule_oe = 'oe_low_regime'

    elif cr >= 5:
        p_odd   = 0.70 if cv == 'ODD' else 0.30
        rule_oe = 'oe_long_streak'

    elif cr >= 3:
        p_odd   = 0.62 if cv == 'ODD' else 0.38
        rule_oe = 'oe_medium_streak'

    elif cr <= 2 and pr >= 3:
        conf_oe = 0.68 if pr >= 6 else (0.63 if pr >= 4 else 0.58)
        p_odd   = conf_oe if pv == 'ODD' else (1.0 - conf_oe)
        rule_oe = 'oe_trend_persist'

    elif maj_val is not None:
        p_odd   = 0.50 + (maj_frac - 0.5) * 0.70 if maj_val == 'ODD' else 0.50 - (maj_frac - 0.5) * 0.70
        rule_oe = 'oe_maj20'

    p_odd     = max(0.05, min(0.95, p_odd))
    pred_oe   = 'ODD' if p_odd > 0.5 else 'EVEN'
    conf_oe   = round(max(p_odd, 1 - p_odd), 4)
    is_bet_oe = conf_oe >= 0.58

    return pred_oe, conf_oe, round(p_odd, 4), is_bet_oe, rule_oe


# ── Sum prediction — specific numbers ─────────────────────────────────────────

_THEO_DIST = None  # theoretical P(sum=s) for 4 digits 0-9

def _get_theo_dist():
    """Precompute theoretical distribution of sum for 4 independent digits 0-9."""
    global _THEO_DIST
    if _THEO_DIST is None:
        counts = [0] * 37
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    for d in range(10):
                        counts[a + b + c + d] += 1
        total = 10000.0
        _THEO_DIST = {s: counts[s] / total for s in range(37)}
    return _THEO_DIST

def _per_position_dist(hist, window=20):
    """
    For each of the 4 digit positions, compute P(digit=d) from the last
    `window` rounds. Returns list of 4 lists (each length 10).
    """
    recent = hist[-window:] if len(hist) >= window else hist
    result = []
    for pos in ('n1', 'n2', 'n3', 'n4'):
        cnt   = [0] * 10
        for r in recent:
            cnt[r[pos]] += 1
        total = sum(cnt) or 1
        result.append([c / total for c in cnt])
    return result

def _convolve_sum_dist(pos_dists):
    """
    Convolve 4 per-position digit distributions into a sum distribution.
    Each pos_dist is a list of 10 probabilities (digits 0-9).
    Returns a list of 37 values (P(sum=s) for s=0..36).
    """
    dist = pos_dists[0][:]          # start with position-1 distribution
    for pd in pos_dists[1:]:
        new_dist = [0.0] * (len(dist) + 10)
        for i, p in enumerate(dist):
            if p == 0:
                continue
            for d, q in enumerate(pd):
                new_dist[i + d] += p * q
        dist = new_dist
    return dist[:37]

def hot_sums_from_history(hist, window=30, top_n=5):
    """Return top_n most frequent sum values from the last `window` rounds.
    Reads newest-first so tie-breaking matches the 'Hot sums (last 30)' panel."""
    from collections import Counter
    recent = hist[-window:] if len(hist) >= window else hist
    counts = Counter(r['total'] for r in reversed(recent))
    return [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n]]


def hot_digits_per_position(hist, window=20, top_n=3):
    """
    Return the top_n hottest digits for each of the 4 positions over the
    last `window` rounds.  Result: [[[digit, count], …], …] per position.
    """
    recent = hist[-window:] if len(hist) >= window else hist
    result = []
    for pos in ('n1', 'n2', 'n3', 'n4'):
        cnt = {}
        for r in recent:
            d = r[pos]
            cnt[d] = cnt.get(d, 0) + 1
        top = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result.append([[d, c] for d, c in top])
    return result


def filter_to_n(items, oe_filter, bs_filter, target_n=5):
    """
    Filter a list of sum values (or [(val, score), ...] tuples) down to target_n
    by keeping values that match the current ODD/EVEN and BIG/SMALL predictions.
    Always tries to fill up to target_n: starts with both filters, then relaxes
    BS filter, then falls back to unfiltered, filling slots incrementally.
    """
    def val(item):
        return item[0] if isinstance(item, (list, tuple)) else item

    def oe_ok(item):
        s = val(item)
        if oe_filter == 'ODD':  return s % 2 == 1
        if oe_filter == 'EVEN': return s % 2 == 0
        return True

    def bs_ok(item):
        s = val(item)
        if bs_filter == 'BIG':   return s > 18
        if bs_filter == 'SMALL': return s <= 18
        return True

    seen = []
    seen_vals = set()

    def add_unique(candidates):
        for x in candidates:
            v = val(x)
            if v not in seen_vals:
                seen_vals.add(v)
                seen.append(x)
            if len(seen) >= target_n:
                break

    # Tier 1: both OE + BS match
    add_unique(x for x in items if oe_ok(x) and bs_ok(x))
    # Tier 2: OE only (BS relaxed)
    if len(seen) < target_n:
        add_unique(x for x in items if oe_ok(x))
    # Tier 3: anything remaining (both filters relaxed)
    if len(seen) < target_n:
        add_unique(items)

    return seen[:target_n]


def filtered_hot_sums(hist, window=50, top_n=5, oe_filter=None, bs_filter=None):
    """
    Hot sums from last `window` rounds, narrowed to those matching the
    current ODD/EVEN and BIG/SMALL pattern predictions.
    Falls back gracefully (drops BS filter, then both) if too few candidates.
    """
    from collections import Counter
    recent = hist[-window:] if len(hist) >= window else hist
    counts = Counter(r['total'] for r in reversed(recent))
    pool = [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])[:max(top_n * 3, 15)]]
    return filter_to_n(pool, oe_filter, bs_filter, target_n=top_n)


def filtered_hot_digits_per_position(hist, window=50, top_n=3, oe_window=10):
    """
    Hot digits per position filtered by each position's recent odd/even trend.
    Returns list of dicts: {pos, oe_trend, odd_frac, hot: [[digit, count], ...]}
    """
    recent_win = hist[-window:] if len(hist) >= window else hist
    recent_oe  = hist[-oe_window:] if len(hist) >= oe_window else hist

    result = []
    for pos in ('n1', 'n2', 'n3', 'n4'):
        vals_oe = [r[pos] for r in recent_oe]
        odd_frac = sum(1 for d in vals_oe if d % 2 == 1) / len(vals_oe) if vals_oe else 0.5
        pos_oe = 'ODD' if odd_frac >= 0.60 else ('EVEN' if odd_frac <= 0.40 else 'MIXED')

        cnt = {}
        for r in recent_win:
            d = r[pos]
            cnt[d] = cnt.get(d, 0) + 1
        top_all = sorted(cnt.items(), key=lambda x: x[1], reverse=True)

        if pos_oe != 'MIXED':
            filtered = [(d, c) for d, c in top_all if (
                (pos_oe == 'ODD' and d % 2 == 1) or
                (pos_oe == 'EVEN' and d % 2 == 0)
            )]
            if len(filtered) < 2:
                filtered = top_all
        else:
            filtered = top_all

        result.append({
            'pos':      pos,
            'oe_trend': pos_oe,
            'odd_frac': round(odd_frac, 3),
            'hot':      [[d, c] for d, c in filtered[:top_n]],
        })
    return result

def predict_sum_numbers(hist, n=5):
    """
    Predict the N most likely sum values for the next round.
    Blends four components:
      1. Theoretical distribution (combinatorics of 4 digits 0-9)      — 25%
      2. Recent sum frequency (last 30 rounds hot/cold bias)             — 20%
      3. Rolling-average trend (Gaussian centred on predicted mean)      — 25%
      4. Per-position hot digit convolution (each disc's hot numbers)    — 30%
    Returns list of (sum_val, probability_pct) sorted by sum_val.
    """
    theo = _get_theo_dist()

    if len(hist) < 5:
        top = sorted(theo.items(), key=lambda x: x[1], reverse=True)[:n]
        return sorted([(s, round(p * 100, 1)) for s, p in top], key=lambda x: x[0])

    # 1 — Recent sum frequency (last 30 rounds)
    window = min(30, len(hist))
    recent = hist[-window:]
    freq   = {}
    for r in recent:
        s = r['total']
        freq[s] = freq.get(s, 0) + 1

    # 2 — Rolling average + trend centre for Gaussian
    sums   = [r['total'] for r in hist]
    r10    = sums[-10:] if len(sums) >= 10 else sums
    r5     = sums[-5:]
    avg10  = sum(r10) / len(r10)
    avg5   = sum(r5)  / len(r5)
    trend  = avg5 - avg10
    centre = max(0.0, min(36.0, avg10 + 0.5 * trend))
    sigma  = max(3.0, statistics.stdev(r10) if len(r10) >= 3 else 5.0)

    # 3 — Per-position hot digit convolution
    pos_dists  = _per_position_dist(hist, window=20)
    conv_dist  = _convolve_sum_dist(pos_dists)
    conv_total = sum(conv_dist) or 1.0
    p_conv     = {s: conv_dist[s] / conv_total for s in range(37)}

    # Score each sum 0-36
    scores = {}
    for s in range(37):
        p_theo  = theo[s]
        p_freq  = freq.get(s, 0) / window
        p_gauss = math.exp(-0.5 * ((s - centre) / sigma) ** 2)
        scores[s] = (0.25 * p_theo
                   + 0.20 * p_freq
                   + 0.25 * p_gauss
                   + 0.30 * p_conv.get(s, 0))

    total = sum(scores.values())
    if total > 0:
        scores = {s: v / total for s, v in scores.items()}

    top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return sorted([(s, round(p * 100, 1)) for s, p in top_n], key=lambda x: x[0])


def predict_sum(hist):
    """
    Thin wrapper kept for backward compat.
    Returns (pred_sum_val, pred_sum_zone, conf_sum).
    """
    if len(hist) < 5:
        return 18, 'MID', 0.50
    sums  = [r['total'] for r in hist]
    r10   = sums[-10:] if len(sums) >= 10 else sums
    r5    = sums[-5:]
    r3    = sums[-3:]
    avg10 = sum(r10) / len(r10)
    avg5  = sum(r5)  / len(r5)
    avg3  = sum(r3)  / len(r3)
    trend = avg5 - avg10
    pred_v = max(0.0, min(36.0, avg10 + 0.5 * trend + 0.2 * (avg3 - avg5)))
    pred_i = round(pred_v)
    zone   = 'LOW' if pred_i <= 13 else ('HIGH' if pred_i >= 23 else 'MID')
    std    = statistics.stdev(r10) if len(r10) >= 3 else 5.0
    conf   = max(0.40, min(0.62, 0.62 - (std - 3) / 30))
    return pred_i, zone, round(conf, 4)


# ── BIG/SMALL prediction (unchanged) ──────────────────────────────────────────

def predict_fst1m(hist):
    """
    Predict BIG/SMALL for next round.
    Returns (pred_bs, p_big, confidence, signal_info).
    signal_info now includes ODD/EVEN and sum prediction data.
    """
    if len(hist) < 10:
        return None, 0.5, 0.5, {}

    idx = len(hist) - 1
    cv, cr, pv, pr = current_run(hist)
    low_streak_bs   = detect_low_streak_regime(hist)
    maj_val, maj_frac = majority(hist, window=20)

    p_xgb = 0.5
    if xgb_model is not None:
        feats = make_features(hist, idx)
        if feats is not None:
            p_xgb = float(xgb_model.predict_proba(
                np.array(feats, dtype=np.float32).reshape(1, -1)
            )[0][1])

    p_big        = p_xgb
    applied_rule = 'xgb'

    if low_streak_bs:
        if maj_val is not None:
            p_maj = maj_frac if maj_val == 'BIG' else (1 - maj_frac)
            p_big = 0.60 * p_maj + 0.40 * p_xgb
        else:
            p_big = p_xgb
        applied_rule = 'rule0_low_regime_maj20'

    elif cr >= 5:
        stay_p = 0.72 if cv == 'BIG' else 0.28
        p_big  = 0.65 * stay_p + 0.35 * p_xgb
        applied_rule = 'rule1_long_streak'

    elif cr >= 3:
        stay_p = 0.62 if cv == 'BIG' else 0.38
        p_big  = 0.55 * stay_p + 0.45 * p_xgb
        applied_rule = 'rule1b_stay'

    elif cr <= 2 and pr >= 3:
        conf   = 0.70 if pr >= 6 else (0.65 if pr >= 4 else 0.60)
        stay_p = conf if pv == 'BIG' else (1.0 - conf)
        p_big  = 0.60 * stay_p + 0.40 * p_xgb
        applied_rule = 'rule6_trend_persist'

    elif cr <= 2:
        if maj_val is not None:
            p_maj = maj_frac if maj_val == 'BIG' else (1 - maj_frac)
            p_big = 0.60 * p_maj + 0.40 * p_xgb
        else:
            p_big = p_xgb
        applied_rule = 'rule_maj20'

    p_big = max(0.05, min(0.95, p_big))
    if applied_rule == 'xgb' and abs(p_big - 0.5) < 0.04:
        p_big = 0.48
    p_big = max(0.05, min(0.95, p_big))

    pred_bs    = 'BIG' if p_big > 0.5 else 'SMALL'
    confidence = round(max(p_big, 1 - p_big), 4)
    is_bet_bs  = confidence >= 0.60

    # ── ODD/EVEN (dedicated rule cascade) ─────────────────────────────────────
    pred_oe, conf_oe, p_odd, is_bet_oe, rule_oe = predict_odd_even(hist)

    # ── Sum prediction — specific numbers ──────────────────────────────────────
    pred_sum_val, pred_sum_zone, conf_sum = predict_sum(hist)
    pred_sum_nums = predict_sum_numbers(hist, n=6)      # [(val, pct), ...]
    hot_digits    = hot_digits_per_position(hist, window=50, top_n=3)
    pred_hot_sums = hot_sums_from_history(hist, window=50, top_n=6)
    # Filter each 6-item list down to 5 using OE + BS pattern predictions
    pred_sum_nums_filtered  = filter_to_n(pred_sum_nums, pred_oe, pred_bs, target_n=5)
    pred_hot_sums_filtered  = filter_to_n(pred_hot_sums, pred_oe, pred_bs, target_n=5)
    hot_digits_filtered     = filtered_hot_digits_per_position(hist, window=50, top_n=3)

    sums_last10 = [r['total'] for r in (hist[-10:] if len(hist) >= 10 else hist)]
    avg_sum     = sum(sums_last10) / len(sums_last10)

    signal_info = {
        # BIG/SMALL
        'cur_val':      cv,
        'cur_run':      cr,
        'prev_val':     pv,
        'prev_run':     pr,
        'rule':         applied_rule,
        'p_big':        round(p_big, 4),
        'p_xgb':        round(p_xgb, 4),
        'maj20':        maj_val,
        'maj20_frac':   round(maj_frac, 3),
        'low_streak':   low_streak_bs,
        'is_bet':       is_bet_bs,
        # ODD/EVEN
        'pred_oe':      pred_oe,
        'conf_oe':      conf_oe,
        'p_odd':        p_odd,
        'is_bet_oe':    is_bet_oe,
        'rule_oe':      rule_oe,
        # Sum numbers
        'pred_sum_val':  pred_sum_val,
        'pred_sum_zone': pred_sum_zone,
        'conf_sum':      conf_sum,
        'pred_sum_nums':  pred_sum_nums,
        'hot_digits':              hot_digits,
        'pred_hot_sums':           pred_hot_sums,
        'pred_hot_sums_filtered':  pred_hot_sums_filtered,
        'pred_sum_nums_filtered':  pred_sum_nums_filtered,
        'hot_digits_filtered':     hot_digits_filtered,
        'avg_sum_10':     round(avg_sum, 1),
        'last_sum':      hist[-1]['total'],
    }

    return pred_bs, p_big, confidence, signal_info


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    hist = load_history()
    if not hist:
        print("No history. Waiting for collector to populate DB.")
        save_snap({'error': 'no_history'})
        return

    _init_xgb(hist)
    pred_log = load_pred_log()

    next_id = hist[-1]['id'] + 1
    pred_bs, p_big, confidence, sinfo = predict_fst1m(hist)

    if pred_bs is None:
        print("Not enough history to predict.")
        save_snap({'error': 'insufficient_history'})
        return

    ls = loss_streak(hist, pred_log)

    snap = {
        'next_round_id': next_id,
        'last_id':       hist[-1]['id'],
        # BIG/SMALL
        'pred_bs':       pred_bs,
        'p_big':         round(p_big, 4),
        'confidence':    confidence,
        'is_bet':        sinfo['is_bet'],
        # ODD/EVEN
        'pred_oe':       sinfo['pred_oe'],
        'conf_oe':       sinfo['conf_oe'],
        'p_odd':         sinfo['p_odd'],
        'is_bet_oe':     sinfo['is_bet_oe'],
        'rule_oe':       sinfo['rule_oe'],
        # Sum numbers
        'pred_sum_val':           sinfo['pred_sum_val'],
        'pred_sum_zone':          sinfo['pred_sum_zone'],
        'conf_sum':               sinfo['conf_sum'],
        'pred_sum_nums':          sinfo['pred_sum_nums'],
        'hot_digits':             sinfo['hot_digits'],
        'pred_hot_sums_filtered':  sinfo['pred_hot_sums_filtered'],
        'pred_sum_nums_filtered':  sinfo['pred_sum_nums_filtered'],
        'hot_digits_filtered':     sinfo['hot_digits_filtered'],
        # Shared stats
        'avg_sum_10':    sinfo['avg_sum_10'],
        'last_sum':      sinfo['last_sum'],
        'signals':       sinfo,
        'loss_streak':   ls,
        'timestamp':     datetime.datetime.now().isoformat(),
    }
    save_snap(snap)

    pred_log[str(next_id)] = {
        'pred_bs':       pred_bs,
        'p_big':         round(p_big, 4),
        'confidence':    confidence,
        'is_bet':        sinfo['is_bet'],
        'bet':           1 if sinfo['is_bet'] else 0,
        'pred_oe':       sinfo['pred_oe'],
        'conf_oe':       sinfo['conf_oe'],
        'is_bet_oe':     sinfo['is_bet_oe'],
        'bet_oe':        1 if sinfo['is_bet_oe'] else 0,
        'pred_sum_val':  sinfo['pred_sum_val'],
        'pred_sum_zone': sinfo['pred_sum_zone'],
        'conf_sum':      sinfo['conf_sum'],
        'pred_sum_nums': sinfo['pred_sum_nums'],
        'signals':       sinfo,
        'timestamp':     datetime.datetime.now().isoformat(),
    }
    save_pred_log(pred_log)

    bet_bs  = 'BET' if sinfo['is_bet'] else 'SKIP'
    bet_oe  = 'BET' if sinfo['is_bet_oe'] else 'SKIP'
    nums_str = '  '.join(f"{v}({p}%)" for v, p in sinfo['pred_sum_nums'])
    print(f"  FST1M  Round {next_id}")
    print(f"  BIG/SMALL : {pred_bs}  (conf {confidence*100:.1f}%)  [{bet_bs}]  rule={sinfo['rule']}")
    print(f"  ODD/EVEN  : {sinfo['pred_oe']}  (conf {sinfo['conf_oe']*100:.1f}%)  [{bet_oe}]  rule={sinfo['rule_oe']}")
    print(f"  Sum bet   : {nums_str}")
    print(f"  Avg10={sinfo['avg_sum_10']}  last={sinfo['last_sum']}")
    if ls:
        print(f"  [!] Loss streak: {ls}")


if __name__ == '__main__':
    main()
