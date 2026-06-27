"""
Color Game Predictor

Predicts three things per round:
  1. COLOR   — RED  (0,2,4,6,8) or GREEN (1,3,5,7,9)
               Same run/tile/sequence patterns as the ODD/EVEN disc game.
  2. NUMBER  — specific digit 0-9, conditioned on predicted color.
  3. PURPLE  — whether the number will be 0 or 5 (the purple specials).

Purple numbers: 0 (red+purple), 5 (green+purple).
"""
import os, sys, json, sqlite3, datetime, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

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

_env      = _load_env()
DB_PATH   = os.path.join(_DATA_DIR, _env.get('DB_FILE2', 'rg3m.db'))
LOG_PATH  = os.path.join(_DATA_DIR, 'pred_log_cg.json')
SNAP_PATH      = os.path.join(_DATA_DIR, 'latest_prediction_cg.json')
XGB_MODEL_PATH = os.path.join(_DATA_DIR, 'xgb_model_cg.pkl')
XGB_META_PATH  = os.path.join(_DATA_DIR, 'xgb_meta_cg.json')
RETRAIN_EVERY  = 50

RED_NUMS    = [0, 2, 4, 6, 8]
GREEN_NUMS  = [1, 3, 5, 7, 9]
PURPLE_NUMS = [0, 5]
COLOR_NUMS  = {'red': RED_NUMS, 'green': GREEN_NUMS}
_FLIP       = {'red': 'green', 'green': 'red'}

# ── DB schema migration ────────────────────────────────────────────────────────

def _ensure_columns():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    existing = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
    for col, defn in [('pred_number',  'INTEGER DEFAULT -1'),
                      ('pred_purple',  'INTEGER DEFAULT -1'),
                      ('number_result','TEXT DEFAULT ""')]:
        if col not in existing:
            conn.execute(f'ALTER TABLE rounds ADD COLUMN {col} {defn}')
    conn.commit()
    conn.close()

_ensure_columns()

# ── Load data ──────────────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH, timeout=10)
conn.execute('PRAGMA journal_mode=WAL')
rows = conn.execute(
    'SELECT id, number, color, is_purple, result FROM rounds ORDER BY id'
).fetchall()
conn.close()

if not rows:
    print("No data in rg3m.db yet.")
    sys.exit(0)

ids     = [r[0] for r in rows]
numbers = [r[1] for r in rows]
colors  = [r[2] for r in rows]
purples = [r[3] for r in rows]
results = [r[4] for r in rows]
n_rows  = len(rows)
last_id = ids[-1]

log = {}
if os.path.exists(LOG_PATH):
    try:
        with open(LOG_PATH, encoding='utf-8-sig') as f:
            log = json.load(f)
    except Exception:
        pass

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_runs(arr):
    if not arr:
        return []
    runs, cur, cnt = [], arr[0], 1
    for v in arr[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = v, 1
    runs.append((cur, cnt))
    return runs

def current_run_info(arr):
    if not arr:
        return None, 0, None, 0
    runs = build_runs(arr)
    cv, cr = runs[-1]
    pv, pr = (runs[-2][0], runs[-2][1]) if len(runs) >= 2 else (None, 0)
    return cv, cr, pv, pr

def alt_run_length(arr):
    """Count how many trailing alternations exist."""
    alt = 0
    for k in range(len(arr) - 1, 0, -1):
        if arr[k] != arr[k - 1]:
            alt += 1
        else:
            break
    return alt

def majority_color(arr, window=6):
    """Return the majority color in the last `window` rounds, or None if tied."""
    recent = arr[-window:] if len(arr) >= window else arr
    r = recent.count('red')
    g = recent.count('green')
    if r == g:
        return None
    return 'red' if r > g else 'green'

# ── Mirror-tile detection (mirrors the ODD/EVEN disc game logic exactly) ───────

def detect_mirror_tile(arr, tile_sizes=(6, 8)):
    n = len(arr)
    for ts in tile_sizes:
        if n < ts * 2:
            continue
        tile_a = arr[n - ts:]
        tile_b = arr[n - ts*2: n - ts]
        if not all(_FLIP[tile_b[i]] == tile_a[i] for i in range(ts)):
            continue
        cycles, pos, ref = 1, n - ts*2, tile_b
        while pos >= ts:
            chunk = arr[pos - ts: pos]
            if all(_FLIP[ref[i]] == chunk[i] for i in range(ts)):
                cycles += 1; ref = chunk; pos -= ts
            else:
                break
        conf = round(0.70 + min(0.15, (cycles - 1) * 0.05), 3)
        return {'pred_val': _FLIP[tile_a[-1]], 'conf': conf,
                'tile_size': ts, 'cycles': cycles + 1,
                'next_tile': [_FLIP[v] for v in tile_a]}
    return None

def detect_mirror_tile_inprogress(arr, tile_sizes=(6, 8)):
    n = len(arr)
    for ts in tile_sizes:
        if n < ts * 2:
            continue
        for offset in range(1, ts):
            cur_start = n - offset
            if cur_start < ts:
                continue
            partial   = arr[cur_start:]
            prev_tile = arr[cur_start - ts: cur_start]
            if len(prev_tile) < ts:
                continue
            expected = [_FLIP[v] for v in prev_tile]
            if not all(expected[i] == partial[i] for i in range(len(partial))):
                continue
            pair_count, pe, pr = 1, cur_start - ts, prev_tile
            while pe >= ts:
                pt = arr[pe - ts: pe]
                if all(_FLIP[pr[i]] == pt[i] for i in range(ts)):
                    pair_count += 1; pr = pt; pe -= ts
                else:
                    break
            if pair_count < 1:
                continue
            next_pos = offset
            if next_pos >= ts:
                continue
            pred_val = expected[next_pos]
            conf = round((0.65 if pair_count == 1 else 0.68) + min(0.12, 0.04 * offset), 3)
            return {'pred_val': pred_val, 'conf': conf,
                    'tile_size': ts, 'cycles': pair_count + 1, 'tile_pos': offset}
    return None

def _loss_streak():
    streak = 0
    for r in reversed([r for r in results if r in ('WIN', 'LOSS')]):
        if r == 'LOSS': streak += 1
        else: break
    return streak

_cur_loss_streak = _loss_streak()


def _get_sticky_cg(log):
    """Return (pred_color, confidence) to reuse if last scored round was WIN, else (None, None)."""
    scored = sorted(
        [(int(k), v) for k, v in log.items() if v.get('actual') and v.get('pred_color')],
        key=lambda x: x[0], reverse=True
    )
    if not scored:
        return None, None
    _, last = scored[0]
    if last['pred_color'] == last['actual']:
        return last['pred_color'], last.get('confidence', 0.65)
    return None, None

# ── Alternating block pattern (GGGRRR / RRRGGG — exactly 3+ of each) ─────────

def detect_alternating_block_pattern(arr, min_block=3):
    """
    Detects A×n → B×m → A×n style alternating block patterns.

    Requires ALL THREE of the most recent runs to be >= min_block so that a
    single short interruption (e.g. G×1) or an asymmetric pair (R×5→G×3) does
    not trigger a false flip prediction.

    Specifically rejects:
      - G×1→R×3→G×3  (G×1 is below min_block — not a real block)
      - R×5→G×3→R×3  (R×5 is >1.5× G×3 — asymmetric, pattern not balanced)
    """
    if len(arr) < min_block * 3:
        return False, None, 0.5, {}

    runs = build_runs(arr)
    if len(runs) < 3:
        return False, None, 0.5, {}

    cur_val,  cur_run  = runs[-1]
    prev_val, prev_run = runs[-2]
    pp_val,   pp_run   = runs[-3]

    # All three blocks must reach the minimum size
    if cur_run < min_block or prev_run < min_block or pp_run < min_block:
        return False, None, 0.5, {}

    # Run 2 steps back must be the same colour as current (A→B→A structure)
    if pp_val != cur_val:
        return False, None, 0.5, {}

    # The block 2 steps back must not be much bigger than the middle block —
    # if it is, the pattern is asymmetric (dominant run + short detour) rather
    # than a true equal-sized alternation
    if pp_run > prev_run * 1.5:
        return False, None, 0.5, {}

    # Current block must not have grown much larger than prev (still valid alternation)
    if cur_run > prev_run * 1.5:
        return False, None, 0.5, {}

    # Confidence scales with block symmetry
    balance = min(cur_run, prev_run) / max(cur_run, prev_run)
    conf    = round(min(0.72, 0.63 + 0.09 * balance), 3)

    return True, prev_val, conf, {
        'prev_color': prev_val,
        'prev_run':   prev_run,
        'cur_color':  cur_val,
        'cur_run':    cur_run,
        'pp_run':     pp_run,
        'pattern':    f'{pp_val[0].upper()}x{pp_run}_{prev_val[0].upper()}x{prev_run}_{cur_val[0].upper()}x{cur_run}',
    }


# ── XGBoost features ──────────────────────────────────────────────────────────

def make_features(arr):
    """Feature vector for predicting the next color after arr."""
    n = len(arr)
    if n < 11:
        return None
    runs = build_runs(arr)
    cv, cr = runs[-1]
    pv, pr = (runs[-2][0], runs[-2][1]) if len(runs) >= 2 else (cv, 0)

    recent8  = arr[-8:]
    dom_r8   = recent8.count('red') / len(recent8) if recent8 else 0.5
    recent20 = arr[-20:] if n >= 20 else arr
    dom_r20  = recent20.count('red') / len(recent20) if recent20 else 0.5

    alt = alt_run_length(arr)

    bl = [(r[1], 1 if r[0] == 'red' else 0) for r in runs[-5:][::-1]]
    while len(bl) < 5:
        bl.append((0, 0))
    avg_bl = sum(r[1] for r in runs[-min(5, len(runs)):]) / min(5, len(runs))

    return [
        cr,
        1 if cv == 'red' else 0,
        pr,
        1 if pv == 'red' else 0,
        alt,
        dom_r8,
        dom_r20,
        bl[0][0], bl[0][1],
        bl[1][0], bl[1][1],
        bl[2][0], bl[2][1],
        bl[3][0], bl[3][1],
        bl[4][0], bl[4][1],
        avg_bl,
        1 if cr >= 4 else 0,
        1 if pr >= 4 else 0,
        1 if alt >= 3 else 0,
    ]


def train_xgb_color(colors_list, meta):
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = [], []
    for i in range(11, len(colors_list)):
        f = make_features(colors_list[:i])
        if f is None:
            continue
        X.append(f)
        y.append(1 if colors_list[i] == 'red' else 0)
    if len(X) < 200:
        return None
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42
    )
    clf.fit(X, y)
    n_eval = min(500, len(X))
    preds  = clf.predict(X[-n_eval:])
    acc    = sum(p == t for p, t in zip(preds, y[-n_eval:])) / n_eval
    meta.update({'n_samples': len(X), 'n_features': len(X[0]),
                 'accuracy': round(acc, 4),
                 'trained_at': datetime.datetime.now().isoformat()})
    return clf


# ── COLOR prediction ───────────────────────────────────────────────────────────

def predict_color(idx):
    if idx < 11:
        return None, 0.5, {}
    arr = colors[:idx]
    cv, cr, pv, pr = current_run_info(arr)
    recent8 = arr[-8:]
    r8, g8  = recent8.count('red'), recent8.count('green')
    alt     = alt_run_length(arr)
    maj6    = majority_color(arr, window=6)

    mirror_info    = detect_mirror_tile(arr)
    mirror_in_info = detect_mirror_tile_inprogress(arr) if mirror_info is None else None
    alt_blk_det, alt_blk_pred, alt_blk_conf, alt_blk_info = detect_alternating_block_pattern(arr)

    # XGBoost baseline probability (red)
    p_xgb = 0.50
    feats = make_features(arr)
    if feats is not None and _xgb_model is not None:
        p_xgb = float(_xgb_model.predict_proba([feats])[0][1])

    applied_rule = 'xgb'
    p_red = p_xgb

    # Rule 4: Mirror tile — highest priority structural pattern (56.5% historical)
    if mirror_info is not None:
        p_red = mirror_info['conf'] if mirror_info['pred_val'] == 'red' else 1.0 - mirror_info['conf']
        applied_rule = 'rule4_mirror_tile'

    # Rule 1: Long streak (4+) — XGBoost-dominant blend
    elif cr >= 4:
        block_p = 0.75 if cv == 'red' else 0.25
        p_red = 0.35 * block_p + 0.65 * p_xgb
        applied_rule = 'rule1_long_streak'

    # Rule 1_alt_block: Alternating block (A×n B×m A pattern) — blend with XGBoost
    elif alt_blk_det:
        alt_p = alt_blk_conf if alt_blk_pred == 'red' else 1.0 - alt_blk_conf
        p_red = 0.40 * alt_p + 0.60 * p_xgb
        applied_rule = f'rule1_alt_block_{alt_blk_info["pattern"]}'

    # Rule 1b: Medium streak (3) — checked before rule2 so cr==3 doesn't trigger trend persist
    elif cr == 3:
        streak_p = 0.62 if cv == 'red' else 0.38
        p_red = 0.30 * streak_p + 0.70 * p_xgb
        applied_rule = 'rule1_medium_streak'

    # Rule 2: After dominant run (3+), hold previous direction for 1-2 counter rounds
    elif pr >= 3 and cr <= 2:
        _runs_full = build_runs(arr)
        _ppv = _runs_full[-3][0] if len(_runs_full) >= 3 else None
        _ppr = _runs_full[-3][1] if len(_runs_full) >= 3 else 0
        if _ppv == cv and _ppr >= pr:
            pass  # current color was dominant 2 steps back — let XGBoost decide
        else:
            conf_t = 0.68 if pr >= 6 else (0.65 if pr >= 4 else 0.62)
            trend_p = conf_t if pv == 'red' else (1.0 - conf_t)
            p_red = 0.35 * trend_p + 0.65 * p_xgb
            applied_rule = 'rule2_trend_persist'

    # Rule 1a: Short streak (2) with no dominant previous run
    elif cr == 2:
        streak_p = 0.56 if cv == 'red' else 0.44
        p_red = 0.30 * streak_p + 0.70 * p_xgb
        applied_rule = 'rule1a_short_streak'

    # Rule 3: Confirmed alternating (3+ unbroken) — blend with XGBoost
    elif cr == 1 and pr == 1 and alt >= 3:
        conf_alt = round(0.65 + min(0.10, (alt - 3) * 0.025), 3)
        alt_p = (1.0 - conf_alt) if cv == 'red' else conf_alt
        p_red = 0.35 * alt_p + 0.65 * p_xgb
        applied_rule = 'rule3_alternating'

    # Rule 5: Mirror tile in progress (high-cycle only)
    elif mirror_in_info is not None and mirror_in_info.get('cycles', 0) >= 3:
        p_red = mirror_in_info['conf'] if mirror_in_info['pred_val'] == 'red' else 1.0 - mirror_in_info['conf']
        applied_rule = 'rule5_mirror_in'

    p_red = max(0.05, min(0.95, p_red))

    # Streak flip: only when XGBoost is indeterminate and loss streak is clear
    if _cur_loss_streak >= 5 and cr < 3 and alt < 3 and abs(p_xgb - 0.5) < 0.10 and maj6 is not None and maj6 != cv:
        p_red = 1.0 - p_red
        applied_rule += '+streak_flip'

    p_red = max(0.05, min(0.95, p_red))

    pred_color = 'red' if p_red > 0.5 else 'green'
    conf       = max(p_red, 1.0 - p_red)
    return pred_color, conf, {
        'cur_val': cv, 'cur_run': cr, 'prev_val': pv, 'prev_run': pr,
        'alt_run': alt, 'dom_r8': round(r8/len(recent8), 3) if recent8 else 0.5,
        'rule': applied_rule, 'loss_streak': _cur_loss_streak,
        'p_xgb': round(p_xgb, 4), 'maj6': maj6,
        'mirror_info':  mirror_info or mirror_in_info,
        'alt_blk_info': alt_blk_info if alt_blk_det else None,
    }

# ── NUMBER prediction ──────────────────────────────────────────────────────────

def _num_seq_signal(num_ctx, candidates, min_matches=2):
    n = len(num_ctx)
    for wlen in range(min(8, n - 1), 1, -1):
        pat = num_ctx[n - wlen:]
        after = {}
        for i in range(n - wlen - 1):
            if num_ctx[i: i + wlen] == pat:
                nxt = num_ctx[i + wlen]
                after[nxt] = after.get(nxt, 0) + 1
        total = sum(after.values())
        if total < min_matches:
            continue
        cand_after = {k: v for k, v in after.items() if k in candidates}
        if not cand_after:
            continue
        best = max(cand_after, key=cand_after.get)
        conf = cand_after[best] / total
        return best, conf
    return None, 0.0

def predict_number(idx, pred_color):
    candidates = COLOR_NUMS[pred_color]
    if idx < 5:
        base = round(1.0 / len(candidates), 3)
        ranked = [(n, base) for n in candidates]
        return candidates[0], base, {'ranked': ranked, 'probs': {str(n): base for n in candidates}}

    num_ctx   = numbers[:idx]
    color_ctx = [n for n in num_ctx if n in candidates]

    # Sequence signals
    seq_num,  seq_conf  = _num_seq_signal(num_ctx, candidates)
    wc_num,   wc_conf   = _num_seq_signal(color_ctx, candidates)

    # Hot signal: most frequent in last 10 rounds (empirically +0.9% above random)
    recent10 = num_ctx[-min(10, len(num_ctx)):]
    counts10 = {n: recent10.count(n) for n in candidates}
    hot_num  = max(candidates, key=lambda d: (counts10[d], -d))
    hot_cnt  = counts10[hot_num]

    # Run signal: if same number repeating, slight lean to continue
    num_runs = build_runs(num_ctx)
    run_num, run_conf = None, 0.0
    if num_runs:
        last_num, last_run = num_runs[-1]
        if last_num in candidates:
            if last_run >= 3:
                run_num, run_conf = last_num, 0.58
            elif last_run >= 2:
                run_num, run_conf = last_num, 0.52

    # Combine into scores — hot number gets a boost, no cold penalty
    scores = {n: 1.0 / len(candidates) for n in candidates}
    # Hot boost (proportional to how much hotter vs average)
    base10 = len(recent10) / len(candidates)
    for n in candidates:
        excess = max(0, counts10[n] - base10)
        scores[n] += 0.35 * (excess / max(len(recent10), 1))
    if seq_num is not None and seq_conf >= 0.30:
        for n in candidates:
            scores[n] += 0.30 * (seq_conf if n == seq_num else 0.0)
    if wc_num is not None and wc_conf >= 0.30:
        for n in candidates:
            scores[n] += 0.20 * (wc_conf if n == wc_num else 0.0)
    if run_num is not None:
        for n in candidates:
            scores[n] += 0.15 * (run_conf if n == run_num else 0.0)

    total_s = sum(scores.values()) or 1
    probs   = {n: round(scores[n] / total_s, 3) for n in candidates}
    ranked  = sorted(probs.items(), key=lambda x: -x[1])
    best_n, best_conf = ranked[0]

    return best_n, best_conf, {
        'hot_num': hot_num, 'hot_cnt': hot_cnt,
        'seq_num': seq_num, 'seq_conf': round(seq_conf, 3),
        'wc_num': wc_num, 'wc_conf': round(wc_conf, 3),
        'run_num': run_num, 'run_conf': round(run_conf, 3),
        'ranked': ranked,
        'probs': {str(n): p for n, p in probs.items()},
    }

# ── PURPLE prediction ─────────────────────────────────────────────────────────
# Purple = numbers 0 or 5. Base rate ~20%.
# Uses gap-since-last-purple, sequence, and frequency.

def predict_purple(idx):
    """
    Returns (pred_purple: bool, confidence: float, sinfo: dict).
    pred_purple=True means we predict the next number IS 0 or 5.
    """
    if idx < 10:
        return False, 0.20, {}

    purple_seq = purples[:idx]   # 1 if purple, 0 if not
    n = len(purple_seq)

    # Gap since last purple
    gap = 0
    for v in reversed(purple_seq):
        if v == 0:
            gap += 1
        else:
            break

    # Recent frequency in last 30
    recent30 = purple_seq[-30:]
    freq = sum(recent30) / len(recent30) if recent30 else 0.2

    # Sequence: does purple follow specific non-purple runs?
    # Look for pattern: if gap >= threshold, purple is "due"
    # Empirical: average gap between purples = ~5 rounds (2/10 base)
    # If gap is large (>8), slight uplift
    base_p = 0.20
    gap_boost = 0.0
    if gap >= 8:
        gap_boost = min(0.15, (gap - 7) * 0.03)
    elif gap == 0:   # current round is purple — unlikely to repeat immediately
        gap_boost = -0.08

    # Sequence signal on binary purple stream
    seq_p = None
    for wlen in range(min(6, n - 1), 1, -1):
        pat = purple_seq[n - wlen:]
        after_purple = after_not = 0
        for i in range(n - wlen - 1):
            if purple_seq[i: i + wlen] == pat:
                if purple_seq[i + wlen] == 1: after_purple += 1
                else:                          after_not    += 1
        total = after_purple + after_not
        if total < 3:
            continue
        ratio = after_purple / total
        if ratio >= 0.40 or ratio <= 0.15:
            seq_p = ratio
        break

    p_purple = base_p + gap_boost + (freq - 0.20) * 0.3
    if seq_p is not None:
        p_purple = 0.5 * p_purple + 0.5 * seq_p

    p_purple  = max(0.05, min(0.60, p_purple))
    pred_bool = p_purple >= 0.30   # predict purple if probability >= 30%

    return pred_bool, round(p_purple, 3), {
        'gap_since_last': gap,
        'recent30_freq':  round(freq, 3),
        'seq_p':          round(seq_p, 3) if seq_p is not None else None,
        'p_purple':       round(p_purple, 3),
    }

# ── XGBoost model load / train ────────────────────────────────────────────────

_xgb_model = None
_xgb_meta  = {}

if os.path.exists(XGB_META_PATH):
    try:
        with open(XGB_META_PATH) as f:
            _xgb_meta = json.load(f)
    except Exception:
        pass

_need_train = not os.path.exists(XGB_MODEL_PATH)
if not _need_train:
    _need_train = (n_rows - 11 - _xgb_meta.get('n_samples', 0)) >= RETRAIN_EVERY

if _need_train and n_rows >= 211:
    print('  Training XGBoost color model...', flush=True)
    _clf = train_xgb_color(colors, _xgb_meta)
    if _clf is not None:
        with open(XGB_MODEL_PATH, 'wb') as f:
            pickle.dump(_clf, f)
        with open(XGB_META_PATH, 'w') as f:
            json.dump(_xgb_meta, f, indent=2)
        _xgb_model = _clf
        print(f'  XGBoost trained: {_xgb_meta["n_samples"]} samples, acc={_xgb_meta["accuracy"]*100:.1f}%', flush=True)

if _xgb_model is None and os.path.exists(XGB_MODEL_PATH):
    try:
        with open(XGB_MODEL_PATH, 'rb') as f:
            _xgb_model = pickle.load(f)
        print(f'  XGBoost loaded: {_xgb_meta.get("n_samples","?")} samples, acc={_xgb_meta.get("accuracy",0)*100:.1f}%', flush=True)
    except Exception as e:
        print(f'  XGBoost load failed: {e}', flush=True)


# ── Gap-fill ───────────────────────────────────────────────────────────────────

# Fill ALL rows that don't have predictions yet (covers post-backfill scenario).
# For live operation this is fast (only new rounds need filling).
_missing = [
    i for i in range(1, n_rows)
    if str(ids[i]) not in log
    or not log[str(ids[i])].get('pred_color')
    or log[str(ids[i])].get('pred_number') is None
]

if _missing:
    total_miss = len(_missing)
    print(f'  Filling {total_miss} missing predictions…', flush=True)
    for done, _bi in enumerate(_missing, 1):
        if done % 500 == 0 or done == total_miss:
            print(f'    {done}/{total_miss}', flush=True)
        _rid = str(ids[_bi])
        _pc, _pp, _si   = predict_color(_bi)
        if _pc is None:
            continue
        _pn, _pnc, _pns = predict_number(_bi, _pc)
        _pur, _purc, _purs = predict_purple(_bi)
        if _rid not in log:
            log[_rid] = {}
        e = log[_rid]
        e['pred_color']    = _pc
        e['p_red']         = round(_pp if _pc == 'red' else 1.0 - _pp, 4)
        e['confidence']    = round(max(_pp, 1.0 - _pp), 4)
        e['bet']           = max(_pp, 1.0 - _pp) >= 0.60
        e['pred_number']   = _pn
        e['number_conf']   = _pnc
        e['number_signals']= _pns
        e['pred_purple']   = _pur
        e['purple_conf']   = _purc
        e['purple_signals']= _purs
        e['signals']       = _si
        if colors[_bi]:
            e['actual']        = colors[_bi]
            e['actual_number'] = numbers[_bi]
            e['actual_purple'] = bool(purples[_bi])

# ── Update actuals ─────────────────────────────────────────────────────────────

_log_changed = False
for i, rid_int in enumerate(ids):
    rid = str(rid_int)
    if rid not in log:
        continue
    e = log[rid]
    if e.get('pred_color') and not e.get('actual'):
        e['actual']        = colors[i]
        e['actual_number'] = numbers[i]
        e['actual_purple'] = bool(purples[i])
        _log_changed = True
    if e.get('pred_number') is None and e.get('pred_color'):
        _pn, _pnc, _pns = predict_number(i, e['pred_color'])
        e['pred_number']    = _pn
        e['number_conf']    = _pnc
        e['number_signals'] = _pns
        _log_changed = True
    if e.get('pred_purple') is None:
        _pur, _purc, _purs = predict_purple(i)
        e['pred_purple']    = _pur
        e['purple_conf']    = _purc
        e['purple_signals'] = _purs
        _log_changed = True

# ── Next prediction ────────────────────────────────────────────────────────────

_sticky_color, _sticky_color_conf = _get_sticky_cg(log)
if _sticky_color is not None:
    pred_color  = _sticky_color
    p_red       = _sticky_color_conf if _sticky_color == 'red' else (1.0 - _sticky_color_conf)
    color_conf  = _sticky_color_conf
    is_bet      = color_conf >= 0.60
    color_sinfo = {
        'cur_val': '', 'cur_run': 0, 'prev_val': '', 'prev_run': 0,
        'alt_run': 0, 'dom_r8': 0.5, 'rule': 'sticky_win',
        'loss_streak': _cur_loss_streak, 'p_xgb': 0.5,
        'maj6': None, 'mirror_info': None, 'alt_blk_info': None,
    }
else:
    pred_color, color_conf, color_sinfo = predict_color(n_rows)
    if pred_color is None:
        print("Not enough data to predict.")
        sys.exit(0)
    p_red      = color_conf if pred_color == 'red' else 1.0 - color_conf
    color_conf = max(p_red, 1.0 - p_red)
    is_bet     = color_conf >= 0.60

pred_number, number_conf, number_sinfo = predict_number(n_rows, pred_color)
pred_purple, purple_conf, purple_sinfo = predict_purple(n_rows)

# ── Cold numbers: 3 least-appeared in last 50 rounds ──────────────────────────
_COLD_WINDOW = 50
_cold_window_nums = numbers[max(0, n_rows - _COLD_WINDOW): n_rows]
_cold_freq   = {d: _cold_window_nums.count(d) for d in range(10)}
cold_numbers = sorted(range(10), key=lambda d: (_cold_freq[d], d))[:3]
cold_details = [{'number': d, 'count': _cold_freq[d],
                 'color': 'red' if d % 2 == 0 else 'green',
                 'is_purple': d in (0, 5)} for d in cold_numbers]

hot_numbers = sorted(range(10), key=lambda d: (-_cold_freq[d], d))[:5]
hot_details = [{'number': d, 'count': _cold_freq[d],
                'color': 'red' if d % 2 == 0 else 'green',
                'is_purple': d in (0, 5)} for d in hot_numbers]

# If pred_number is already 0 or 5, override pred_purple accordingly
if pred_number in PURPLE_NUMS:
    pred_purple  = True
    purple_conf  = max(purple_conf, number_conf)

next_id = last_id + 1
log[str(next_id)] = {
    'pred_color':     pred_color,
    'p_red':          round(p_red, 4),
    'confidence':     round(color_conf, 4),
    'bet':            is_bet,
    'pred_number':    pred_number,
    'number_conf':    number_conf,
    'number_signals': number_sinfo,
    'pred_purple':    pred_purple,
    'purple_conf':    purple_conf,
    'purple_signals': purple_sinfo,
    'signals':        color_sinfo,
    'actual':         None,
    'actual_number':  None,
    'actual_purple':  None,
    'timestamp':      datetime.datetime.now().isoformat(),
}
_log_changed = True

if _log_changed:
    _tmp = LOG_PATH + '.tmp'
    with open(_tmp, 'w') as f:
        json.dump(log, f, indent=2)
    os.replace(_tmp, LOG_PATH)

# ── Stats ──────────────────────────────────────────────────────────────────────

def _stats(n_last):
    scored = sorted(
        [e for e in log.values() if e.get('actual') and e.get('pred_color')],
        key=lambda e: e.get('timestamp', '')
    )[-n_last:]
    if not scored:
        return None, 0, None, None
    n = len(scored)
    cw = sum(1 for e in scored if e['pred_color'] == e['actual'])
    nh = sum(1 for e in scored
             if e.get('pred_number') is not None and e.get('actual_number') is not None
             and e['pred_number'] == e['actual_number'])
    ph = sum(1 for e in scored
             if e.get('pred_purple') is not None and e.get('actual_purple') is not None
             and bool(e['pred_purple']) == bool(e['actual_purple']))
    return round(cw/n,4), n, round(nh/n,4), round(ph/n,4)

wr100, n100, nr100, pr100 = _stats(100)
wr200, n200, nr200, pr200 = _stats(200)
bet_s = [e for e in log.values() if e.get('actual') and e.get('pred_color') and e.get('bet')]
wr_bet = round(sum(1 for e in bet_s if e['pred_color']==e['actual'])/len(bet_s),4) if bet_s else None
nr_bet = round(sum(1 for e in bet_s if e.get('pred_number')==e.get('actual_number'))/len(bet_s),4) if bet_s else None
pr_bet = round(sum(1 for e in bet_s if bool(e.get('pred_purple'))==bool(e.get('actual_purple')))/len(bet_s),4) if bet_s else None

# ── Recent history ─────────────────────────────────────────────────────────────

recent_history = []
for i in range(max(0, n_rows - 30), n_rows):
    rid   = str(ids[i])
    e     = log.get(rid, {})
    pn    = e.get('pred_number')
    pp    = e.get('pred_purple')
    recent_history.append({
        'id':          ids[i],
        'number':      numbers[i],
        'color':       colors[i],
        'is_purple':   purples[i],
        'pred_color':  e.get('pred_color', ''),
        'pred_number': pn if pn is not None else -1,
        'pred_purple': bool(pp) if pp is not None else False,
        'purple_conf': e.get('purple_conf', 0),
        'number_conf': e.get('number_conf', 0),
        'confidence':  e.get('confidence', 0),
        'bet':         e.get('bet', False),
        'result':      results[i],
        'number_hit':  (pn == numbers[i]) if pn is not None else None,
        'purple_hit':  (bool(pp) == bool(purples[i])) if pp is not None else None,
    })

# ── Update DB ──────────────────────────────────────────────────────────────────

def _update_db():
    conn2 = sqlite3.connect(DB_PATH, timeout=10)
    conn2.execute('PRAGMA journal_mode=WAL')
    try:
        upd = []
        for rid_str, e in log.items():
            pc = e.get('pred_color', '')
            cf = float(e.get('confidence', 0) or 0)
            bt = 1 if e.get('bet', False) else 0
            pn = e.get('pred_number'); pn = pn if pn is not None else -1
            pp = 1 if e.get('pred_purple') else 0
            if pc:
                upd.append((pc, cf, bt, pn, pp, int(rid_str)))
        if upd:
            conn2.executemany(
                'UPDATE rounds SET pred_color=?, confidence=?, bet=?, pred_number=?, pred_purple=? WHERE id=?',
                upd
            )

        all_pred = conn2.execute(
            'SELECT id, color, pred_color, number, pred_number, is_purple, pred_purple '
            "FROM rounds WHERE pred_color != '' AND color != ''"
        ).fetchall()
        res_upd = []
        for rid, c, pc, num, pn, ip, pp in all_pred:
            if pc not in ('red', 'green'):
                continue
            color_r  = 'WIN' if pc == c else 'LOSS'
            num_r    = 'HIT' if (pn is not None and pn >= 0 and pn == num) else 'MISS'
            res_upd.append((color_r, num_r, rid))
        if res_upd:
            conn2.executemany(
                'UPDATE rounds SET result=?, number_result=? WHERE id=?', res_upd
            )
        conn2.commit()
    finally:
        conn2.close()

_update_db()

# ── Snapshot ───────────────────────────────────────────────────────────────────

snapshot = {
    'next_id': next_id,
    'prediction': {
        'pred_color':     pred_color,
        'pred_number':    pred_number,
        'pred_purple':    pred_purple,
        'p_red':          round(p_red, 4),
        'color_conf':     round(color_conf, 4),
        'number_conf':    number_conf,
        'purple_conf':    purple_conf,
        'is_bet':         is_bet,
        'ranked_numbers': number_sinfo.get('ranked', []),
        'num_probs':      number_sinfo.get('probs', {}),
    },
    'signals':        color_sinfo,
    'number_signals': number_sinfo,
    'purple_signals': purple_sinfo,
    'stats': {
        'last100':  {'wr': wr100, 'n': n100, 'nr': nr100, 'pr': pr100},
        'last200':  {'wr': wr200, 'n': n200, 'nr': nr200, 'pr': pr200},
        'bet_only': {'wr': wr_bet, 'n': len(bet_s), 'nr': nr_bet, 'pr': pr_bet},
    },
    'recent':      recent_history,
    'cold_50':     cold_details,
    'hot_50':      hot_details,
    'cold_100':    sorted(
        [{'number': d,
          'count': numbers[max(0, n_rows-100): n_rows].count(d),
          'color': 'red' if d % 2 == 0 else 'green',
          'is_purple': d in (0, 5)}
         for d in range(10)],
        key=lambda x: (x['count'], x['number'])
    )[:3],
    'hot_100':     sorted(
        [{'number': d,
          'count': numbers[max(0, n_rows-100): n_rows].count(d),
          'color': 'red' if d % 2 == 0 else 'green',
          'is_purple': d in (0, 5)}
         for d in range(10)],
        key=lambda x: (-x['count'], x['number'])
    )[:5],
    'timestamp':   datetime.datetime.now().isoformat(),
}

_tmp = SNAP_PATH + '.tmp'
with open(_tmp, 'w') as f:
    json.dump(snapshot, f, indent=2)
os.replace(_tmp, SNAP_PATH)

# ── Console ────────────────────────────────────────────────────────────────────

_purple_tag = ' +PURPLE' if pred_purple else ''
print(f"\n  Color Game — Round {next_id}")
print(f"  Color      : {pred_color.upper()}  (conf={color_conf*100:.1f}%)")
print(f"  Number     : {pred_number}{_purple_tag}  (conf={number_conf*100:.1f}%)")
print(f"  Purple?    : {'YES' if pred_purple else 'no'}  (p={purple_conf*100:.1f}%)")
print(f"  Ranked     : {' > '.join(str(n) for n,_ in number_sinfo.get('ranked',[])[:5])}")
print(f"  Cold-50    : {[d['number'] for d in cold_details]}  counts={[d['count'] for d in cold_details]}")
print(f"  Cold-100   : {[d['number'] for d in snapshot['cold_100']]}  counts={[d['count'] for d in snapshot['cold_100']]}")
print(f"  Bet        : {'YES' if is_bet else 'no'}  rule={color_sinfo.get('rule','?')}")
if wr100:
    print(f"  Color WR   : {wr100*100:.1f}% (n={n100})   Number HR: {nr100*100:.1f}%   Purple acc: {pr100*100:.1f}%")
else:
    print("  Accuracy   : no scored data yet")
print()
