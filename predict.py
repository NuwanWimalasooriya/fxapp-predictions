"""
DS1M Prediction Engine
Signals combined:
  1. Balance deficit  — law of large numbers: whichever side is behind is more likely next
  2. Run-length model — current streak length predicts continuation vs switch
  3. Sequence matcher — find this exact OE window in history, count what came next
  4. LSTM (TBPTT)     — sequence model trained with truncated backprop through time
Sklearn ML used as additional fallback.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def _load_env():
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env

_env      = _load_env()
_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

DB_PATH       = os.path.join(_DATA_DIR, _env.get('DB_FILE', 'ds3m.db'))
CSV_PATH      = os.path.join(_DATA_DIR, _env.get('DATA_FILE', '3min-discs.csv'))
LOG_PATH      = os.path.join(_DATA_DIR, 'pred_log.json')
MODEL_PATH    = os.path.join(_DATA_DIR, 'models.pkl')
META_PATH     = os.path.join(_DATA_DIR, 'model_meta.json')
LSTM_PATH     = os.path.join(_DATA_DIR, 'lstm_model.pt')
CONFIG_PATH   = os.path.join(_DATA_DIR, 'config.json')
SEQ_LEN       = 20
PREDICT_N     = 5   # predict 5 rounds ahead to cover batch arrivals
RETRAIN_EVERY = 50
MIN_MATCHES   = 3
TBPTT_SEG     = 32   # segment length for truncated BPTT
BET_THRESHOLD = 0.70  # minimum confidence to count as a real bet

# ── Load data ──────────────────────────────────────────────────────────────────
import sqlite3 as _sqlite3
_conn = _sqlite3.connect(DB_PATH, timeout=10)
_conn.execute('PRAGMA journal_mode=WAL')
df = pd.read_sql('SELECT id,disc1 AS d1,disc2 AS d2,disc3 AS d3,disc4 AS d4,result FROM rounds ORDER BY id',
                 _conn, dtype={'id': int})
_conn.close()
df['pattern'] = df['d1'] + df['d2'] + df['d3'] + df['d4']
df['r_count'] = df['pattern'].str.count('R')
df['odd']     = df['r_count'].apply(lambda r: 'ODD' if r % 2 else 'EVEN')

last    = df.iloc[-1]
last_id = int(last['id'])
n_rows  = len(df)

WWWW_IDS = set(df.loc[df['pattern'] == 'WWWW', 'id'])
RRRR_IDS = set(df.loc[df['pattern'] == 'RRRR', 'id'])

# ── Run-length encoder ─────────────────────────────────────────────────────────
def encode_runs(oe_list):
    """Convert OE list to list of (value, length) run tuples."""
    if not oe_list:
        return []
    runs, cur_val, cur_len = [], oe_list[0], 1
    for v in oe_list[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            runs.append((cur_val, cur_len))
            cur_val, cur_len = v, 1
    runs.append((cur_val, cur_len))
    return runs

# Pre-compute runs for the full history
all_runs = encode_runs(df['odd'].tolist())

# ── Signal: Trend clarity (alternation rate) ──────────────────────────────────
def trend_clarity_signal(oe_arr, idx, window=6):
    """
    Measures how trending vs choppy the recent sequence is.
    Returns (p_odd, clarity) where:
      - clarity=1.0 means perfectly trending (no switches) → high confidence
      - clarity=0.0 means fully alternating → near-random, output 0.5
    p_odd is the current trend direction scaled by clarity.
    Data-derived: alt_rate=0.0 → 58.3% WR; alt_rate>=0.6 → ~50% WR.
    """
    if idx < window:
        return None, 0.0
    recent = oe_arr[max(0, idx - window + 1): idx + 1]
    if len(recent) < 2:
        return None, 0.0
    switches = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
    alt_rate = switches / (len(recent) - 1)
    clarity  = 1.0 - alt_rate   # 1.0 = pure trend, 0.0 = pure alternation
    # Only emit signal when clarity is meaningful (trend present)
    if clarity < 0.4:
        return None, clarity    # too choppy — no signal
    odd_count = sum(1 for x in recent if x == 'ODD')
    p_odd_trend = odd_count / len(recent)
    # Scale prediction toward 0.5 proportionally to choppiness
    p_odd = 0.5 + (p_odd_trend - 0.5) * clarity
    return p_odd, clarity

# ── Signal 0: Short-term momentum ────────────────────────────────────────────
def momentum_signal(oe_arr, idx, window=7):
    """
    Short-term trend: fraction of ODD in recent window.
    Captures persistent streaks that the balance signal actively fights.
    """
    if idx < window:
        return 0.5
    recent = oe_arr[idx - window + 1: idx + 1]
    odd_count = sum(1 for x in recent if x == 'ODD')
    return odd_count / window

# ── Signal 1: Balance deficit (Law of Large Numbers) ──────────────────────────
def balance_signal(oe_arr, idx, windows=(20, 50, 100, 200)):
    """
    In each window, measure how far ODD count is from expected 50%.
    Whichever side is behind is 'owed' by the algorithm to restore balance.
    Returns p_odd adjustment: positive = ODD is owed, negative = EVEN is owed.
    Range roughly -1..+1.
    """
    signals = []
    weights = []
    for w in windows:
        start = idx - w + 1
        if start < 0:
            continue
        chunk     = oe_arr[start: idx + 1]
        odd_count = sum(1 for x in chunk if x == 'ODD')
        # deficit: how many ODDs are missing vs expected 50%
        deficit   = (w / 2 - odd_count) / (w / 2)   # -1=all ODD, 0=balanced, +1=all EVEN
        signals.append(deficit)
        weights.append(1.0 / w)   # shorter windows weighted more (more recent)
    if not signals:
        return 0.0
    total_w = sum(weights)
    return sum(s * w for s, w in zip(signals, weights)) / total_w

# ── Signal 2: Run-length model (PRIMARY) ──────────────────────────────────────
def run_length_signal(runs_full, cur_val, cur_len):
    """
    Given we are currently AT cur_len consecutive cur_val values,
    count how many times in history that exact situation continued
    (run went to cur_len+1) vs switched (run ended at exactly cur_len).

    This is the strongest signal found in the data analysis:
    after a run of length N the switch probability is ~50-53%.
    """
    continued = sum(1 for v, l in runs_full if v == cur_val and l > cur_len)
    switched  = sum(1 for v, l in runs_full if v == cur_val and l == cur_len)
    total     = continued + switched

    if total < 5:
        return 0.5, {}

    p_continue = continued / total

    # What length does the NEXT run tend to be when it switches?
    next_lens = {}
    for i in range(len(runs_full) - 1):
        v, l = runs_full[i]
        if v == cur_val and l == cur_len:
            _, nl = runs_full[i + 1]
            next_lens[nl] = next_lens.get(nl, 0) + 1

    total_nl      = sum(next_lens.values()) or 1
    next_len_dist = {k: v / total_nl for k, v in next_lens.items()}
    return p_continue, next_len_dist

# ── Signal 3: Sequence matcher ─────────────────────────────────────────────────
def sequence_signal(oe_arr, pat_arr, idx, min_matches=MIN_MATCHES):
    """
    Try window lengths 3..10. For each, find all historical matches
    to the current OE window and count what came next.
    Pick the length giving the most confident result (furthest from 50/50)
    with at least min_matches occurrences.
    """
    best_p_odd     = None
    best_conf      = -1
    best_pat_votes = {}
    best_len       = 0
    best_n         = 0

    for seq_len in range(5, 21):      # start at 5 — analysis shows <5 is near-random
        if idx < seq_len:
            continue
        current   = tuple(oe_arr[idx - seq_len + 1: idx + 1])
        odd_count = 0
        total     = 0
        pat_votes = {}

        for i in range(seq_len - 1, idx):
            if tuple(oe_arr[i - seq_len + 1: i + 1]) == current:
                nxt_oe  = oe_arr[i + 1]
                nxt_pat = pat_arr[i + 1]
                if nxt_oe == 'ODD':
                    odd_count += 1
                total += 1
                pat_votes[nxt_pat] = pat_votes.get(nxt_pat, 0) + 1

        if total < min_matches:
            continue
        p_odd = odd_count / total
        conf  = abs(p_odd - 0.5)
        if conf > best_conf:
            best_conf      = conf
            best_p_odd     = p_odd
            best_pat_votes = {k: v / total for k, v in pat_votes.items()}
            best_len       = seq_len
            best_n         = total

    return best_p_odd, best_pat_votes, best_len, best_n

# ── Signal: Full disc-pattern sequence matcher ────────────────────────────────
def disc_sequence_signal(pat_arr, idx, min_matches=MIN_MATCHES):
    """
    Match on full 4-disc patterns (RWWR, WWRR …) not just OE.
    One disc-pattern match of length N covers 4N individual disc values —
    far more specific than OE matching of the same length.
    """
    best_p_odd = None
    best_conf  = -1
    best_len   = 0
    best_n     = 0

    for seq_len in range(3, 16):
        if idx < seq_len:
            continue
        current   = tuple(pat_arr[idx - seq_len + 1: idx + 1])
        odd_count = 0
        total     = 0
        for i in range(seq_len - 1, idx):
            if tuple(pat_arr[i - seq_len + 1: i + 1]) == current:
                if pat_arr[i + 1].count('R') % 2:
                    odd_count += 1
                total += 1
        if total < min_matches:
            continue
        p_odd = odd_count / total
        conf  = abs(p_odd - 0.5)
        if conf > best_conf:
            best_conf  = conf
            best_p_odd = p_odd
            best_len   = seq_len
            best_n     = total

    return best_p_odd, best_len, best_n


# ── Signal: Recent OE-pattern repetition (priority check) ─────────────────────
def recent_repeat_signal(oe_arr, pat_arr, idx, look_back=20, hist_window=100):
    """
    Look at the last look_back (20) OE values and find if any sub-pattern that
    appeared earlier within those 20 is currently repeating — i.e. the tail of
    the 20 values matches the beginning of a pattern seen before, meaning we are
    mid-way through a new repetition — then predict the next step.

    The function is fully generic: it extracts every possible sub-pattern from
    the actual last 20 rounds at runtime and checks each one.  No pattern is
    hardcoded; it adapts to whatever structure the data currently shows.

    Example (illustrative only — actual patterns come from live data):
      last 20 = XXXXXXXXXXXXXXABCD  (where ABCD = last 4 values)
      A 10-char pattern P found at position 5 of the 20.
      Last 3 values == P[:3]  →  3 steps into a repetition  →  predict P[3].

    For every sub-pattern P found within the last 20 values:
      1. Find the largest k (1 … len(P)-1) where the last k values equal P[:k].
      2. Count historical occurrences of P in the last hist_window rounds.
      3. Confidence scales with partial ratio (k/L) and historical frequency.
      4. Score = partial × sqrt(L) × history_boost.

    Returns (p_odd, total_occurrences, pattern_len) or (None, 0, 0).
    """
    if idx < look_back:
        return None, 0, 0

    last = list(oe_arr[idx - look_back + 1: idx + 1])

    best_p_odd = None
    best_score = -1
    best_n     = 0
    best_len   = 0

    for L in range(2, look_back - 1):           # pattern length
        for i in range(look_back - L):           # start of the earlier occurrence
            P = last[i: i + L]
            # New repetition must start at i+L or later (no overlap with original)
            max_k = min(L - 1, look_back - (i + L))
            if max_k < 1:
                continue

            for k in range(max_k, 0, -1):        # try longest confirmed match first
                if last[look_back - k:] == P[:k]:
                    # We are k steps into a repetition of P; next value = P[k]
                    search_start = max(0, idx - hist_window)
                    hist_count = sum(
                        1 for j in range(search_start, idx - L + 1)
                        if list(oe_arr[j: j + L]) == P
                    )

                    next_oe = P[k]
                    is_odd  = (next_oe == 'ODD')

                    partial  = k / L
                    strength = min(0.78, 0.50 + partial * 0.40 + hist_count * 0.02)
                    p_odd    = strength if is_odd else 1.0 - strength

                    score = partial * (L ** 0.5) * (1.0 + hist_count * 0.10)

                    if score > best_score:
                        best_score = score
                        best_p_odd = p_odd
                        best_n     = hist_count + 1
                        best_len   = L
                    break   # largest k for this (L, i) found; move on

    return best_p_odd, best_n, best_len


# ── Signal: Autocorrelation period detection ──────────────────────────────────
def period_signal(oe_arr, idx, max_lag=200, min_corr=0.10, top_n=3):
    """
    The game results are algorithmically generated (not random).  Deterministic
    generators — LFSRs, counter-based PRNGs, state machines — produce sequences
    with a fixed period P and detectable autocorrelation at P and its harmonics.

    Method:
      1. Take up to the last 1000 OE values, demean to binary ±0.5.
      2. Compute autocorrelation at every lag 2 … max_lag.
      3. Keep up to top_n lags whose |correlation| ≥ min_corr.
      4. For each dominant lag L:  predicted_next = oe_arr[idx + 1 - L]
         (positive corr → same value;  negative corr → opposite value).
      5. Weighted vote across top lags (weight = |correlation|).

    The stronger and more consistent the autocorrelation, the higher the weight
    this signal gets via adaptive weighting.
    Returns (p_odd, dominant_lag) or (None, 0).
    """
    if idx < max_lag * 2:
        return None, 0

    history_len = min(1000, idx + 1)
    raw = np.array([1 if x == 'ODD' else 0
                    for x in oe_arr[idx - history_len + 1: idx + 1]], dtype=float)
    raw -= raw.mean()
    var = float(np.var(raw))
    if var < 0.01:
        return None, 0

    n = len(raw)
    corrs = []
    for lag in range(2, min(max_lag + 1, n // 3)):
        c = float(np.dot(raw[:-lag], raw[lag:]) / ((n - lag) * var))
        corrs.append((lag, c))

    top_lags = sorted(
        [(lag, c) for lag, c in corrs if abs(c) >= min_corr],
        key=lambda x: -abs(x[1])
    )[:top_n]

    if not top_lags:
        return None, 0

    total_w = sum(abs(c) for _, c in top_lags)
    w_odd   = 0.0
    for lag, corr in top_lags:
        pred_pos = idx + 1 - lag
        if pred_pos < 0:
            continue
        is_odd = (oe_arr[pred_pos] == 'ODD')
        if corr > 0:
            p = (0.5 + abs(corr) * 0.45) if is_odd else (0.5 - abs(corr) * 0.45)
        else:
            p = (0.5 - abs(corr) * 0.45) if is_odd else (0.5 + abs(corr) * 0.45)
        w_odd += p * abs(corr)

    p_odd = w_odd / total_w if total_w > 0 else None
    return p_odd, top_lags[0][0]


# ── Signal: Post-long-run reversal check ──────────────────────────────────────
def post_run_reversal_signal(oe_arr, idx, min_long_run=5, lookback_before=20):
    """
    After a long run of X (≥ min_long_run), when Y (the opposite) appears
    for the FIRST time, predict whether X returns once more or Y continues.

    Key insight: the answer depends on whether Y was already repeating
    (consecutive) BEFORE the long X run started.

      Y not repeating before long X run → X tends to return one more time
      Y was repeating before long X run → Y tends to continue

    Calibrated by scanning history for identical setups and counting outcomes.
    Fires ONLY on the very first Y after a long X run.
    Returns p_odd or None (when the setup is not present or history is sparse).
    """
    if idx < min_long_run + lookback_before + 5:
        return None

    cur_val  = oe_arr[idx]
    long_val = 'EVEN' if cur_val == 'ODD' else 'ODD'

    # Must be the very first round of the opposite (not already in a run of Y)
    if idx > 0 and oe_arr[idx - 1] == cur_val:
        return None

    # Measure the length of the previous run (must be a long run of long_val)
    prev_run_len = 0
    for i in range(idx - 1, max(-1, idx - 60), -1):
        if oe_arr[i] == long_val:
            prev_run_len += 1
        else:
            break
    if prev_run_len < min_long_run:
        return None

    # Was cur_val (the opposite) repeating before the long run?
    # "repeating" = had at least 2 consecutive occurrences within lookback_before rounds
    before_idx  = idx - 1 - prev_run_len   # last index before the long run started
    if before_idx < 2:
        return None
    check_start = max(0, before_idx - lookback_before + 1)
    pre_arr     = oe_arr[check_start: before_idx + 1]

    opp_was_repeating = False
    consec = 0
    for v in pre_arr:
        consec = (consec + 1) if v == cur_val else 0
        if consec >= 2:
            opp_was_repeating = True
            break

    # Scan recent history for identical setups and count outcomes
    scan_start = max(min_long_run + lookback_before + 5, idx - 1000)
    wins = total = 0

    for j in range(scan_start, idx - 1):
        if j == 0 or oe_arr[j] == oe_arr[j - 1]:
            continue                              # not a switch point
        jcur  = oe_arr[j]
        jlong = oe_arr[j - 1]

        jlen = 0
        for k in range(j - 1, max(-1, j - 60), -1):
            if oe_arr[k] == jlong:
                jlen += 1
            else:
                break
        if jlen < min_long_run:
            continue

        jbefore     = j - 1 - jlen
        if jbefore < 2:
            continue
        jcheck_start = max(0, jbefore - lookback_before + 1)
        jpre         = oe_arr[jcheck_start: jbefore + 1]

        jconsec = 0
        jopp_rep = False
        for v in jpre:
            jconsec = (jconsec + 1) if v == jcur else 0
            if jconsec >= 2:
                jopp_rep = True
                break

        if jopp_rep != opp_was_repeating:
            continue                              # different setup — skip

        if j + 1 >= idx:
            continue
        total += 1
        if oe_arr[j + 1] == jlong:              # long trend returned
            wins += 1

    if total < 5:
        return None

    p_long_returns = wins / total
    # Map to p_odd
    if long_val == 'ODD':
        return 0.5 + (p_long_returns - 0.5) * 0.85
    else:
        return 0.5 - (p_long_returns - 0.5) * 0.85


# ── Signal: Rhythm breakout (short-burst pattern → long flow) ─────────────────
def rhythm_breakout_signal(oe_arr, idx, look_back_runs=5, normal_max=4, breakout_min=5):
    """
    Detects when a regular rhythm of short ODD/EVEN bursts (2-4 rounds each)
    suddenly breaks out into a run that exceeds normal_max, signalling a long flow.

    Setup   : last look_back_runs completed runs were ALL in [2, normal_max].
    Breakout: current run has reached exactly breakout_min rounds.
    Signal fires once — at the moment the run first reaches breakout_min.

    Calibrated by scanning the last 2000 rounds for identical rhythmic setups
    and counting how often the breakout run continued at least one more round.
    Returns p_odd or None (setup absent or insufficient history).
    """
    min_idx = (look_back_runs + 1) * (normal_max + 1) * 2
    if idx < min_idx:
        return None

    cur_val = oe_arr[idx]

    # Measure current run length going backwards from idx
    cur_run = 0
    j = idx
    while j >= 0 and oe_arr[j] == cur_val:
        cur_run += 1
        j -= 1
    # j = before_idx (last index before current run started)

    if cur_run != breakout_min:  # fire only at the exact breakout moment
        return None
    if j < 0:
        return None

    def _prev_run_lens(arr, start, n):
        """Lengths of n runs ending at or before start, most-recent first."""
        lens = []
        pos = start
        while len(lens) < n and pos >= 0:
            v = arr[pos]
            rlen = 0
            while pos >= 0 and arr[pos] == v:
                rlen += 1
                pos -= 1
            lens.append(rlen)
        return lens

    prev_lens = _prev_run_lens(oe_arr, j, look_back_runs)
    if len(prev_lens) < look_back_runs:
        return None
    if not all(2 <= r <= normal_max for r in prev_lens):
        return None

    # Historical calibration: scan last 2000 rounds for same setup
    scan_start = max(min_idx, idx - 2000)
    wins = total = 0

    for h in range(scan_start + breakout_min, idx):
        hv = oe_arr[h]
        # Check that exactly breakout_min consecutive hv values end at h
        if not all(oe_arr[h - i] == hv for i in range(breakout_min)):
            continue
        before_h = h - breakout_min
        if oe_arr[before_h] == hv:
            continue  # run started earlier — not the fresh breakout moment

        h_prev = _prev_run_lens(oe_arr, before_h, look_back_runs)
        if len(h_prev) < look_back_runs:
            continue
        if not all(2 <= r <= normal_max for r in h_prev):
            continue

        # Outcome: did the breakout run continue at h+1?
        total += 1
        if h + 1 <= idx and oe_arr[h + 1] == hv:
            wins += 1

    if total < 5:
        return None

    p_continue = wins / total
    p_odd = p_continue if cur_val == 'ODD' else 1.0 - p_continue
    return 0.5 + (p_odd - 0.5) * 0.85


# ── Signal: OE Macro-Phase (alternating / block / transition) ─────────────────
def _build_oe_phase_lookup(oe_arr, key_len=4, min_n=20):
    """
    Build a lookup table: OE key (tuple of last key_len OE values) → P(next=ODD).
    Only returns entries with >= min_n historical samples.
    Handles both OE and EO starting directions naturally.
    """
    lookup = {}
    counts = {}
    for i in range(key_len, len(oe_arr) - 1):
        key = tuple(oe_arr[i - key_len:i])
        nxt = oe_arr[i]
        if key not in counts:
            counts[key] = [0, 0]  # [odd_count, total]
        counts[key][1] += 1
        if nxt == 'ODD':
            counts[key][0] += 1
    for key, (odd_n, total) in counts.items():
        if total >= min_n:
            lookup[key] = (odd_n / total, total)
    return lookup

# Pre-build lookup tables at startup (key lengths 5-8, min 50 samples for reliability)
_oe_phase_lookup = {
    kl: _build_oe_phase_lookup(df['odd'].tolist(), key_len=kl, min_n=50)
    for kl in [5, 6, 7, 8]
}

def oe_macro_phase_signal(oe_arr, idx, threshold=0.55, min_n=50):
    """
    Looks up the current OE sequence in historical data.
    Returns (p_odd, confidence, phase_label, n_samples) or (None, 0, None, 0).

    Priority: longer key wins if it meets the threshold (more specific = more reliable).
    Both OEOE and EOEO starting directions are handled naturally.

    phase_label: 'alternating', 'block', 'transition', or 'mixed'
    """
    best_p, best_conf, best_label, best_n = None, 0.0, None, 0

    for kl in [8, 7, 6, 5]:
        if idx < kl:
            continue
        key = tuple(oe_arr[idx - kl + 1: idx + 1])
        entry = _oe_phase_lookup.get(kl, {}).get(key)
        if entry is None:
            continue
        p_odd, n = entry
        if n < min_n:
            continue
        conf = abs(p_odd - 0.5)
        if conf < (threshold - 0.50):
            continue
        # Classify phase type based on key pattern
        switches = sum(1 for i in range(len(key) - 1) if key[i] != key[i + 1])
        alt_rate = switches / (len(key) - 1)
        if alt_rate >= 0.8:
            label = 'alternating'
        elif alt_rate <= 0.2:
            label = 'block'
        else:
            label = 'mixed'
        # Check for block transition: first half same, second half different
        half = len(key) // 2
        if (len(set(key[:half])) == 1 and len(set(key[half:])) == 1
                and key[0] != key[half]):
            label = 'transition'

        if conf > best_conf:
            best_p, best_conf, best_label, best_n = p_odd, conf, label, n

    if best_p is None or best_conf < (threshold - 0.50):
        return None, 0.0, None, 0
    return best_p, best_conf, best_label, best_n


# ── Signal: OE/EO alternating pattern continuation ────────────────────────────
def oe_alternating_signal(oe_arr, idx, min_alt_len=3, dom_window=8):
    """
    User domain rule:
      1. Count unbroken alternating run ending at idx.
      2. If run >= min_alt_len (confirmed OE/EO pattern):
           count O and E in the run — if E > O predict E, else predict O.
           Returns is_confirmed=True so the caller can hard-lock the prediction.
      3. If no confirmed pattern (block / short run):
           predict DOMINANT direction in last dom_window rounds.
           e.g. OOOOE → O=4 E=1 → predict O (E is a blip).
           If tied, follow last round.
    Returns (p_odd, strength, is_confirmed_alt).
    """
    if idx < 1:
        return None, 0.0, False
    last = oe_arr[idx]
    # Count unbroken alternating run ending at idx
    run_len = 1
    for i in range(idx - 1, max(-1, idx - 30), -1):
        if oe_arr[i] != oe_arr[i + 1]:
            run_len += 1
        else:
            break
    if run_len >= min_alt_len:
        # Confirmed OE/EO pattern — STAY on the starting direction of the run.
        # Guarantees max 1 consecutive loss (win on every other round).
        # Flip approach risks all-loss if phase is wrong.
        run_start = list(oe_arr)[idx - run_len + 1]
        pred_odd  = (run_start == 'ODD')
        return (0.72 if pred_odd else 0.28), 1.0, True   # hard lock
    else:
        # No confirmed pattern — predict dominant direction in recent window
        start   = max(0, idx - dom_window + 1)
        window  = list(oe_arr[start: idx + 1])
        e_count = window.count('EVEN')
        o_count = window.count('ODD')
        if o_count > e_count:
            pred_odd = True
        elif e_count > o_count:
            pred_odd = False
        else:
            pred_odd = (last == 'ODD')   # tie → follow last (continuation)
        return (0.72 if pred_odd else 0.28), 0.30, False


# ── Signal: Long block continuation + transition prediction ───────────────────
def _build_block_sequence(oe_arr):
    """Build list of (value, length, start_idx) from OE array."""
    blocks = []
    i = 0
    arr = list(oe_arr)
    while i < len(arr):
        val = arr[i]
        length = 1
        while i + length < len(arr) and arr[i + length] == val:
            length += 1
        blocks.append((val, length, i))
        i += length
    return blocks

def long_block_signal(oe_arr, idx, long_thresh=4, context_len=6):
    """
    Two-phase rule for long blocks:

    Phase 1 — DURING long block (run >= long_thresh):
      Predict continuation of current block direction.

    Phase 2 — AT TRANSITION (first opposite value after long block):
      Look at pre-block context (what was before the long block):
        - OEO/EOE type  → alternating regime was active → predict pre-context dominant
        - OEEO/EOOE     → block-pair regime → predict same structure continues
        - Flat (all O/E)→ use historical: after long block, next is usually brief (snap back)
      Also checks historical block-transition behavior for this specific block length.

    Returns (p_odd, strength) or (None, 0).
    """
    if idx < long_thresh:
        return None, 0.0

    arr = list(oe_arr)
    cur = arr[idx]

    # Count current run length
    run_len = 1
    for i in range(idx - 1, max(-1, idx - 60), -1):
        if arr[i] == cur:
            run_len += 1
        else:
            break

    # ── Phase 1: inside a long block ──────────────────────────────────────────
    if run_len >= long_thresh:
        strength = min(0.45, 0.25 + 0.04 * (run_len - long_thresh))
        return (0.72 if cur == 'ODD' else 0.28), strength

    # ── Phase 2: just switched from a long block ───────────────────────────────
    # Only fire within first 2 rounds of the switch
    if run_len > 2:
        return None, 0.0

    # Measure previous block length
    prev_val = arr[idx - run_len] if idx >= run_len else None
    if prev_val is None or prev_val == cur:
        return None, 0.0

    prev_run = 0
    switch_idx = idx - run_len
    for i in range(switch_idx, max(-1, switch_idx - 60), -1):
        if arr[i] == prev_val:
            prev_run += 1
        else:
            break

    if prev_run < long_thresh:
        return None, 0.0   # previous block was not long — don't fire

    # Get pre-block context (what was before the long block)
    pre_end  = switch_idx - prev_run
    pre_start = max(0, pre_end - context_len + 1)
    pre_ctx  = arr[pre_start: pre_end + 1]

    if len(pre_ctx) < 2:
        # No context — default: snap back to original block
        return (0.72 if prev_val == 'ODD' else 0.28), 0.20

    # Analyse pre-block context
    pre_o = pre_ctx.count('ODD')
    pre_e = pre_ctx.count('EVEN')
    switches = sum(1 for i in range(len(pre_ctx) - 1) if pre_ctx[i] != pre_ctx[i + 1])
    alt_rate = switches / (len(pre_ctx) - 1)

    # Historical: find all past transitions of similar prev_run length
    blocks = _build_block_sequence(arr[:idx])
    snap_back = long_mirror = alt_phase = 0
    for bi in range(len(blocks) - 2):
        bval, blen, bstart = blocks[bi]
        if bval != prev_val or abs(blen - prev_run) > 2:
            continue
        next_val, next_len, _ = blocks[bi + 1]
        if next_len <= 2:
            snap_back += 1
        elif next_len >= long_thresh:
            long_mirror += 1
        else:
            # Check if it went alternating (next block short + alternating follows)
            alt_phase += 1

    total = snap_back + long_mirror + alt_phase or 1

    if alt_rate >= 0.6:
        # Pre-context was alternating — after long block expect return to alternating
        # Dominant in pre-context decides direction
        pred_odd = pre_o >= pre_e
        return (0.72 if pred_odd else 0.28), 0.30

    elif long_mirror > snap_back and long_mirror / total >= 0.35:
        # History says mirror long block is likely — predict current value continues
        return (0.72 if cur == 'ODD' else 0.28), 0.30

    else:
        # Default: snap back to original block direction
        return (0.72 if prev_val == 'ODD' else 0.28), 0.25


# ── Signal: Post-WWWW/RRRR offset effect ──────────────────────────────────────
def post_special_pattern_signal(pat_arr, idx):
    """
    Empirical offset effects (n=690/761, statistically robust):
      - 5 rounds after WWWW  → ODD  53.8%  (p_odd=0.538)
      - 4 rounds after RRRR  → EVEN 53.6%  (p_odd=0.464)
      - consecutive WWWW×2   → EVEN 52.5%  (p_odd=0.475)
      - consecutive RRRR×2   → EVEN 52.1%  (p_odd=0.479)
    Returns p_odd float or None.
    """
    if idx < 5:
        return None
    last_ww = last_rr = None
    for i in range(idx - 1, max(0, idx - 25), -1):
        if pat_arr[i] == 'WWWW' and last_ww is None:
            last_ww = i
        if pat_arr[i] == 'RRRR' and last_rr is None:
            last_rr = i
        if last_ww is not None and last_rr is not None:
            break
    since_ww = (idx - last_ww) if last_ww is not None else 999
    since_rr = (idx - last_rr) if last_rr is not None else 999
    signals = []
    if since_ww == 5:
        signals.append(0.538)
    if since_rr == 4:
        signals.append(0.464)
    if since_ww == 1 and last_ww is not None and last_ww > 0 and pat_arr[last_ww - 1] == 'WWWW':
        signals.append(0.475)
    if since_rr == 1 and last_rr is not None and last_rr > 0 and pat_arr[last_rr - 1] == 'RRRR':
        signals.append(0.479)
    return sum(signals) / len(signals) if signals else None


# ── Signal: Persistent run continuation ───────────────────────────────────────
def persistent_run_signal(oe_arr, idx, min_run=6):
    """
    When current run >= min_run, compute P(next same | run >= min_run)
    across all history. Analysis showed long runs continue well above 50% —
    opposite of what the balance signal predicted.
    Returns (p_odd, run_len). Returns (None, run_len) if run < min_run.
    """
    cur_val = oe_arr[idx]
    run_len = 1
    for i in range(idx - 1, -1, -1):
        if oe_arr[i] == cur_val:
            run_len += 1
        else:
            break

    if run_len < min_run:
        return None, run_len

    # Scan full history: at each position where run >= min_run, did it continue?
    continued = 0
    total     = 0
    cur_run   = 1
    for i in range(1, idx):
        if oe_arr[i] == oe_arr[i - 1]:
            cur_run += 1
        else:
            cur_run = 1
        if cur_run >= min_run and i + 1 < idx:
            total += 1
            if oe_arr[i + 1] == oe_arr[i]:
                continued += 1

    if total < 10:
        return None, run_len

    p_continue = continued / total
    p_odd = p_continue if cur_val == 'ODD' else 1.0 - p_continue
    return p_odd, run_len


# ── Phase detection ────────────────────────────────────────────────────────────
def detect_phase(oe_arr, idx, balance_win=20, vol_win=10):
    """
    Classify current game state into a phase label combining four dimensions:
      run_type   : SHORT (1-2) / MEDIUM (3-5) / LONG (6+)
      run_val    : ODD / EVEN  (current streak type)
      balance    : ODD_HEAVY / BALANCED / EVEN_HEAVY  (last 20 rounds)
      volatility : CHOPPY (≥6 switches/10) / SMOOTH
    Returns dict with label and component details.
    """
    if idx < balance_win:
        return {'label': 'UNKNOWN', 'run_type': '?', 'run_val': '?',
                'balance': '?', 'volatility': '?', 'run_len': 0,
                'odd_frac': 0.5, 'switches': 0}
    cur_val = oe_arr[idx]
    run_len = 1
    for i in range(idx - 1, max(idx - 50, -1), -1):
        if oe_arr[i] == cur_val:
            run_len += 1
        else:
            break
    run_type = 'LONG' if run_len >= 6 else ('MEDIUM' if run_len >= 3 else 'SHORT')

    recent20  = oe_arr[max(0, idx - balance_win + 1): idx + 1]
    odd_frac  = sum(1 for x in recent20 if x == 'ODD') / len(recent20)
    balance   = 'ODD_HEAVY' if odd_frac > 0.6 else ('EVEN_HEAVY' if odd_frac < 0.4 else 'BALANCED')

    recent10  = oe_arr[max(0, idx - vol_win + 1): idx + 1]
    switches  = sum(1 for i in range(1, len(recent10)) if recent10[i] != recent10[i - 1])
    volatility = 'CHOPPY' if switches >= 6 else 'SMOOTH'

    return {
        'label':      f"{run_type}_{cur_val}_{balance}_{volatility}",
        'run_type':   run_type,
        'run_val':    cur_val,
        'run_len':    run_len,
        'balance':    balance,
        'odd_frac':   round(odd_frac, 3),
        'volatility': volatility,
        'switches':   switches,
    }


def phase_win_rate(phase_label, log, min_rounds=10):
    """
    Scan pred_log for entries matching this phase label, compute historical win rate.
    Returns (win_rate, n_rounds). win_rate is None when n_rounds < min_rounds.
    """
    wins = total = 0
    for entry in log.values():
        if entry.get('phase') != phase_label:
            continue
        actual  = entry.get('actual')
        pred_oe = entry.get('pred_oe')
        if not actual or not pred_oe:
            continue
        actual_oe = 'ODD' if actual.count('R') % 2 else 'EVEN'
        wins  += 1 if pred_oe == actual_oe else 0
        total += 1
    return (round(wins / total, 4) if total >= min_rounds else None), total


def kelly_analysis(p_win):
    """
    Full Kelly analysis for even-money bets.
    Uses half-Kelly (conservative) for practical sizing.
    Returns None when no edge exists (p_win ≤ 0.50).
    """
    if p_win is None or p_win <= 0.50:
        return None
    k         = 2 * p_win - 1          # full Kelly fraction
    hk        = k * 0.5                 # half-Kelly
    bankroll  = 100.0                   # normalised to 100 units
    bet       = bankroll * hk
    p_ruin    = ((1 - p_win) / p_win) ** (bankroll / bet)
    breakeven = max(1, round(1 / (2 * p_win - 1)))   # rounds to recover one lost bet
    return {
        'kelly':           round(k,  4),
        'half_kelly':      round(hk, 4),
        'bet_pct':         round(hk * 100, 1),          # % of bankroll per bet
        'p_ruin':          round(p_ruin, 4),
        'breakeven':       breakeven,
        'e_per_round_pct': round((2 * p_win - 1) * hk * 100, 3),
    }


# ── Feature engineering (ML fallback) ─────────────────────────────────────────
def make_features(df_src, idx):
    if idx < SEQ_LEN:
        return None
    window = df_src.iloc[idx - SEQ_LEN: idx + 1]
    feats  = window['r_count'].tolist()
    feats += (window['odd'] == 'ODD').astype(int).tolist()
    for d in ['d1','d2','d3','d4']:
        feats += (window[d] == 'R').astype(int).tolist()
    cur_id    = int(df_src.iloc[idx]['id'])
    past_wwww = [i for i in WWWW_IDS if i <= cur_id]
    past_rrrr = [i for i in RRRR_IDS if i <= cur_id]
    feats += [min(cur_id - max(past_wwww), 100) if past_wwww else 100,
              min(cur_id - max(past_rrrr), 100) if past_rrrr else 100]
    cur_oe = df_src.iloc[idx]['odd']
    streak = 0
    for i in range(idx, max(idx - 30, -1), -1):
        if df_src.iloc[i]['odd'] == cur_oe:
            streak += 1
        else:
            break
    feats.append(min(streak, 20))
    rc_counts = window['r_count'].value_counts()
    for k in range(5):
        feats.append(rc_counts.get(k, 0) / SEQ_LEN)
    feats.append((window['odd'] == 'ODD').mean())
    # Alternation rate over last 3, 6, 10 rounds — how choppy vs trending
    for w in [3, 6, 10]:
        oe_w = (window['odd'] == 'ODD').astype(int).values[-w:]
        if len(oe_w) >= 2:
            feats.append(float(sum(oe_w[i] != oe_w[i-1] for i in range(1, len(oe_w))) / (len(oe_w)-1)))
        else:
            feats.append(0.5)
    return feats

def recency_weights(n, half_life=300):
    return np.exp(-np.arange(n - 1, -1, -1) / half_life)

# ── LSTM with Truncated BPTT ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

TORCH_OK = False  # LSTM disabled

if TORCH_OK:
    class _LSTM(nn.Module):
        def __init__(self, input_size=12, hidden=128, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers,
                                batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x, state=None):
            out, state = self.lstm(x, state)
            return torch.sigmoid(self.head(out)), state

    def _seq_features(df_src):
        """12 features per row: r_count/4, d1-d4 binary, OE binary,
        rolling 20-round ODD rate per disc, alt_rate_3, alt_rate_6."""
        n   = len(df_src)
        rc  = (df_src['r_count'].values / 4.0).astype(np.float32)
        d1  = (df_src['d1'] == 'R').astype(np.float32).values
        d2  = (df_src['d2'] == 'R').astype(np.float32).values
        d3  = (df_src['d3'] == 'R').astype(np.float32).values
        d4  = (df_src['d4'] == 'R').astype(np.float32).values
        oe  = (df_src['odd'] == 'ODD').astype(np.float32).values
        def _roll20(arr):
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                lo = max(0, i - 19)
                out[i] = arr[lo:i+1].mean()
            return out
        d1r = _roll20(d1)
        d2r = _roll20(d2)
        d3r = _roll20(d3)
        d4r = _roll20(d4)
        # Alternation rate: fraction of consecutive switches in last 3 and 6 rounds
        def _alt_rate(arr, w):
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                lo = max(0, i - w + 1)
                seg = arr[lo:i+1]
                out[i] = float(np.sum(seg[1:] != seg[:-1]) / max(1, len(seg)-1))
            return out
        alt3 = _alt_rate(oe, 3)
        alt6 = _alt_rate(oe, 6)
        return np.stack([rc, d1, d2, d3, d4, oe, d1r, d2r, d3r, d4r, alt3, alt6], axis=1)  # (N, 12)

    def _train_lstm(df_src, epochs=20, lr=1e-3):
        """Train LSTM using TBPTT: detach hidden state every TBPTT_SEG steps."""
        model = _LSTM()
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCELoss(reduction='none')

        feats   = _seq_features(df_src)          # (N, 6)
        targets = (df_src['odd'] == 'ODD').astype(np.float32).values  # (N,)
        N = len(feats)

        # Recency weights: more recent rows matter more
        rw = np.exp(np.linspace(-2, 0, N)).astype(np.float32)

        model.train()
        for epoch in range(epochs):  # noqa: B007
            state = None
            for start in range(0, N - 1, TBPTT_SEG):
                end  = min(start + TBPTT_SEG, N - 1)
                x    = torch.from_numpy(feats[start:end]).unsqueeze(0)    # (1, seg, 6)
                y    = torch.from_numpy(targets[start+1:end+1]).unsqueeze(0).unsqueeze(-1)  # (1, seg, 1)
                w    = torch.from_numpy(rw[start+1:end+1]).unsqueeze(0).unsqueeze(-1)

                # Detach state to truncate gradients at segment boundary
                if state is not None:
                    state = (state[0].detach(), state[1].detach())

                opt.zero_grad()
                pred, state = model(x, state)
                loss = (loss_fn(pred, y) * w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return model

    def _lstm_predict(model, df_src, lookback=200):
        """Run LSTM on last `lookback` rows, return p(ODD) for the next step.
        Returns None if model has collapsed (output within 0.01 of 0.5)."""
        model.eval()
        feats = _seq_features(df_src.iloc[-lookback:])
        x = torch.from_numpy(feats).unsqueeze(0)   # (1, lookback, 10)
        with torch.no_grad():
            out, _ = model(x)
        p = float(out[0, -1, 0])
        return None if abs(p - 0.5) < 0.01 else p  # collapsed model → no signal

    def _lstm_acc(model, df_src, test_frac=0.2):
        """Quick held-out accuracy on the last test_frac of the data."""
        feats   = _seq_features(df_src)
        targets = (df_src['odd'] == 'ODD').astype(np.float32).values
        split   = int(len(feats) * (1 - test_frac))
        model.eval()
        x = torch.from_numpy(feats[:split]).unsqueeze(0)
        with torch.no_grad():
            out, state = model(x)
            # continue on test portion
            x_te = torch.from_numpy(feats[split:-1]).unsqueeze(0)
            out_te, _ = model(x_te, (state[0].detach(), state[1].detach()))
        preds = (out_te[0, :, 0].numpy() >= 0.5).astype(int)
        truth = (targets[split+1:]).astype(int)
        n = min(len(preds), len(truth))
        return float((preds[:n] == truth[:n]).mean())

# ── Load or train ML models ────────────────────────────────────────────────────
def _training_enabled():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f).get('training_enabled', True)
    except Exception:
        return True

def _need_retrain():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return True
    with open(META_PATH) as f:
        meta = json.load(f)
    return n_rows - meta.get('trained_on', 0) >= RETRAIN_EVERY

if _training_enabled() and _need_retrain():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("  Training models...", flush=True)
    X, y_oe, y_rc, y_pat = [], [], [], []
    for i in range(SEQ_LEN, n_rows - 1):
        f = make_features(df, i)
        if f is None:
            continue
        nxt = df.iloc[i + 1]
        X.append(f)
        y_oe.append(1 if nxt['odd'] == 'ODD' else 0)
        y_rc.append(int(nxt['r_count']))
        y_pat.append(nxt['pattern'])

    X     = np.array(X)
    y_oe  = np.array(y_oe)
    y_rc  = np.array(y_rc)
    y_pat = np.array(y_pat)
    split = int(len(X) * 0.80)
    X_tr, X_te   = X[:split],    X[split:]
    oe_tr, oe_te = y_oe[:split], y_oe[split:]
    rc_tr, rc_te = y_rc[:split], y_rc[split:]
    pt_tr, pt_te = y_pat[:split], y_pat[split:]
    sw = recency_weights(len(X_tr))

    gb_oe  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    gb_oe.fit(X_tr, oe_tr, sample_weight=sw)
    lr_oe  = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr_oe.fit(X_tr, oe_tr, sample_weight=sw)
    gb_rc  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    gb_rc.fit(X_tr, rc_tr, sample_weight=sw)
    rf_pat = RandomForestClassifier(n_estimators=300, max_depth=10,
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_pat.fit(X_tr, pt_tr, sample_weight=sw)

    oe_prob = (gb_oe.predict_proba(X_te)[:,1] + lr_oe.predict_proba(X_te)[:,1]) / 2
    acc_oe  = accuracy_score(oe_te, (oe_prob >= 0.5).astype(int))
    acc_rc  = accuracy_score(rc_te, gb_rc.predict(X_te))
    acc_pat = accuracy_score(pt_te, rf_pat.predict(X_te))

    joblib.dump({'gb_oe': gb_oe, 'lr_oe': lr_oe, 'gb_rc': gb_rc, 'rf_pat': rf_pat}, MODEL_PATH)

    # Train LSTM with TBPTT
    acc_lstm = 0.0
    lstm_model = None
    if TORCH_OK:
        print("  Training LSTM (TBPTT)...", flush=True)
        lstm_model = _train_lstm(df)
        acc_lstm   = _lstm_acc(lstm_model, df)
        torch.save(lstm_model.state_dict(), LSTM_PATH)
        print(f"  LSTM done. ODD/EVEN={acc_lstm:.1%}")

    with open(META_PATH, 'w') as f:
        json.dump({'trained_on': n_rows, 'acc_oe': acc_oe,
                   'acc_rc': acc_rc, 'acc_pat': acc_pat,
                   'acc_lstm': acc_lstm}, f)
    print(f"  Done. ODD/EVEN={acc_oe:.1%}  R-count={acc_rc:.1%}  Pattern={acc_pat:.1%}")
else:
    if not _training_enabled() and _need_retrain():
        print("  Training is disabled — using existing models (or skipping ML if none exist).", flush=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        models = joblib.load(MODEL_PATH)
        gb_oe, lr_oe, gb_rc, rf_pat = models['gb_oe'], models['lr_oe'], models['gb_rc'], models['rf_pat']
        with open(META_PATH) as f:
            meta = json.load(f)
        acc_oe   = meta.get('acc_oe', 0)
        acc_rc   = meta.get('acc_rc', 0)
        acc_pat  = meta.get('acc_pat', 0)
        acc_lstm = meta.get('acc_lstm', 0)
    else:
        gb_oe = lr_oe = gb_rc = rf_pat = None
        acc_oe = acc_rc = acc_pat = acc_lstm = 0.0

    lstm_model = None
    if TORCH_OK and os.path.exists(LSTM_PATH):
        try:
            lstm_model = _LSTM()  # input_size=12, hidden=128
            lstm_model.load_state_dict(torch.load(LSTM_PATH, weights_only=True))
            lstm_model.eval()
        except Exception:
            lstm_model = None

# ── Check pred_log ─────────────────────────────────────────────────────────────
def check_pred_log():
    if not os.path.exists(LOG_PATH):
        return 0, 0, 0, 0
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)
    actual_map = df.set_index(df['id'].astype(str))['pattern'].to_dict()
    scored, changed = [], False
    for rid, entry in log.items():
        if entry.get('actual') is None and rid in actual_map:
            entry['actual'] = actual_map[rid]
            changed = True
        if entry.get('actual'):
            scored.append(entry)
    if not scored:
        return 0, 0, 0, 0
    n      = len(scored)
    oe_ok  = sum(1 for e in scored if e['actual'].count('R')%2 == e['predicted'].count('R')%2)
    rc_ok  = sum(1 for e in scored if e['actual'].count('R') == e['predicted'].count('R'))
    pat_ok = sum(1 for e in scored if e['actual'] == e['predicted'])
    if changed:
        _tmp_log = LOG_PATH + '.tmp'
        with open(_tmp_log, 'w') as f:
            json.dump(log, f, indent=2)
        os.replace(_tmp_log, LOG_PATH)
    return oe_ok/n, rc_ok/n, pat_ok/n, n

def _signal_pred_oe(idx):
    """Compute pred_oe from run-length + balance signals at position idx using prior history."""
    if idx < 200:
        return None
    oe_arr_full = df['odd'].values
    # Balance signal on data before this round
    deficit = balance_signal(oe_arr_full, idx - 1)
    p_balance = 0.5 + deficit * 0.4
    # Run-length signal on data before this round
    cur_val = oe_arr_full[idx - 1]
    cur_run_len = 1
    for i in range(idx - 2, max(idx - 52, -1), -1):
        if oe_arr_full[i] == cur_val:
            cur_run_len += 1
        else:
            break
    p_continue, _ = run_length_signal(all_runs, cur_val, cur_run_len)
    p_run = p_continue if cur_val == 'ODD' else 1.0 - p_continue
    p_odd = 0.50 * p_run + 0.15 * p_balance + 0.35 * 0.5
    return 'ODD' if p_odd > 0.5 else 'EVEN'

def _bet_label(confidence):
    if not confidence:
        return ''
    if confidence >= BET_THRESHOLD:
        return 'BET'
    elif confidence >= 0.60:
        return 'SKIP'
    return 'AVOID'

def update_csv_results():
    """Sync DB (pred_oe/confidence/bet/result) from pred_log for all predicted rounds."""
    log = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding='utf-8-sig') as f:
            log = json.load(f)

    # Ensure pred_log entries that have actual data but no pred_oe get a fallback prediction
    log_changed = False
    for rid, entry in log.items():
        if not entry.get('actual') or entry.get('pred_oe'):
            continue
        pos = df.index[df['id'] == int(rid)] if rid.isdigit() else []
        if len(pos) == 0:
            continue
        oe = _signal_pred_oe(pos[0])
        if oe:
            entry['pred_oe'] = oe
            log_changed = True
    if log_changed:
        _log_tmp2 = LOG_PATH + '.tmp'
        with open(_log_tmp2, 'w') as f:
            json.dump(log, f, indent=2)
        os.replace(_log_tmp2, LOG_PATH)

    # Ensure pat_even/pat_odd/breakdown columns exist
    import sqlite3 as _sq, csv as _csv
    conn = _sq.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    _cols = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
    for _col, _type in [('pat_even', 'TEXT DEFAULT ""'), ('pat_odd', 'TEXT DEFAULT ""'),
                         ('wwww_pct', 'REAL DEFAULT 0'), ('wwww_gap', 'INTEGER DEFAULT 0'),
                         ('rrrr_pct', 'REAL DEFAULT 0'), ('rrrr_gap', 'INTEGER DEFAULT 0'),
                         ('w3r1_pct', 'REAL DEFAULT 0'), ('r3w1_pct', 'REAL DEFAULT 0')]:
        if _col.split()[0] not in _cols:
            conn.execute(f'ALTER TABLE rounds ADD COLUMN {_col.split()[0]} {" ".join(_col.split()[1:])}')
    conn.commit()
    conn.close()

    # 1. Update pred_oe/confidence/bet/pat_even/pat_odd/breakdown for all rounds with predictions
    pred_updates = []
    for rid, entry in log.items():
        if not entry.get('pred_oe'):
            continue
        conf     = float(entry.get('confidence', 0) or 0)
        bet_int  = 1 if entry.get('bet', False) else 0
        pred_oe_fmt = entry['pred_oe']
        ep       = entry.get('pat_even_pct')
        op       = entry.get('pat_odd_pct')
        pat_even = f"{entry['pat_even_lbl']}({ep*100:.1f}%)" if entry.get('pat_even_lbl') and ep is not None else ''
        pat_odd  = f"{entry['pat_odd_lbl']}({op*100:.1f}%)"  if entry.get('pat_odd_lbl')  and op is not None else ''
        wwww_pct = float(entry.get('wwww_pct') or 0)
        wwww_gap = int(entry.get('wwww_gap') or 0)
        rrrr_pct = float(entry.get('rrrr_pct') or 0)
        rrrr_gap = int(entry.get('rrrr_gap') or 0)
        w3r1_pct = float(entry.get('w3r1_pct') or 0)
        r3w1_pct = float(entry.get('r3w1_pct') or 0)
        pred_updates.append((pred_oe_fmt, conf, bet_int, pat_even, pat_odd,
                             wwww_pct, wwww_gap, rrrr_pct, rrrr_gap, w3r1_pct, r3w1_pct, int(rid)))

    conn = _sq.connect(DB_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    try:
        if pred_updates:
            conn.executemany(
                "UPDATE rounds SET pred_oe=?, confidence=?, bet=?, pat_even=?, pat_odd=?, "
                "wwww_pct=?, wwww_gap=?, rrrr_pct=?, rrrr_gap=?, w3r1_pct=?, r3w1_pct=? WHERE id=?",
                pred_updates
            )
        # Compute WIN/LOSS directly from DB: find all rows with pred_oe set but result still
        # 'Not Predicted', then derive actual_oe from the disc pattern already stored in DB.
        pending = conn.execute(
            "SELECT id, disc1||disc2||disc3||disc4, pred_oe "
            "FROM rounds WHERE pred_oe != '' AND result='Not Predicted'"
        ).fetchall()
        result_updates = []
        for rid, pattern, p_oe in pending:
            if not pattern or len(pattern) != 4:
                continue
            actual_oe = 'ODD' if pattern.count('R') % 2 else 'EVEN'
            pred_base = p_oe.split('(')[0]
            result_updates.append(('WIN' if pred_base == actual_oe else 'LOSS', rid))
        if result_updates:
            conn.executemany(
                "UPDATE rounds SET result=? WHERE id=?",
                result_updates
            )
        if pred_updates or result_updates:
            conn.commit()
        # Always sync CSV so it reflects the latest DB state
        rows = conn.execute(
            'SELECT id,disc1,disc2,disc3,disc4,pattern,flag,oe,result,pred_oe,bet,pat_even,pat_odd,'
            'wwww_pct,wwww_gap,rrrr_pct,rrrr_gap,w3r1_pct,r3w1_pct '
            'FROM rounds ORDER BY id'
        ).fetchall()
    finally:
        conn.close()
    _tmp = CSV_PATH + '.tmp'
    with open(_tmp, 'w', newline='', encoding='utf-8') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['id','disc1','disc2','disc3','disc4','pattern','flag','oe',
                     'result','pred_oe','bet','pat_even','pat_odd',
                     'wwww_pct','wwww_gap','rrrr_pct','rrrr_gap','w3r1_pct','r3w1_pct'])
        _w.writerows(rows)
    import os as _os
    try:
        _os.replace(_tmp, CSV_PATH)
    except PermissionError:
        # Windows: destination locked (e.g. open in editor) — write in-place instead
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as _f2:
            with open(_tmp, 'r', newline='', encoding='utf-8') as _f3:
                _f2.write(_f3.read())
        try:
            _os.remove(_tmp)
        except OSError:
            pass

real_oe, real_rc, real_pat, n_scored = check_pred_log()
update_csv_results()

# Reload result column so _streak_correction sees fresh WIN/LOSS from this cycle
try:
    _conn_r = _sqlite3.connect(DB_PATH, timeout=10)
    _conn_r.execute('PRAGMA journal_mode=WAL')
    _fresh_results = pd.read_sql('SELECT id, result FROM rounds ORDER BY id', _conn_r, dtype={'id': int})
    _conn_r.close()
    df = df.merge(_fresh_results[['id', 'result']], on='id', how='left', suffixes=('_old', ''))
    if 'result_old' in df.columns:
        df.drop(columns=['result_old'], inplace=True)
except Exception:
    pass

# ── Adaptive signal weights ────────────────────────────────────────────────────
def _signal_stats(entries):
    """
    For each signal compute two data-driven quality metrics from recent entries:
      informativeness : mean absolute deviation from 0.5
                        (a signal stuck near 0.5 provides no information)
      volatility      : mean absolute inter-round change
                        (a signal that swings wildly is noise, not signal)
    """
    vals = {}
    for _, entry in entries:
        for sig, v in entry.get('signals', {}).items():
            if v is not None:
                vals.setdefault(sig, []).append(v)
    stats = {}
    for sig, vs in vals.items():
        if len(vs) < 5:
            continue
        arr = np.array(vs)
        stats[sig] = {
            'info':       float(np.mean(np.abs(arr - 0.5))),
            'volatility': float(np.mean(np.abs(np.diff(arr)))),
            'n':          len(vs),
        }
    return stats

def _adaptive_weights(n_recent=50):
    """
    Weights derived entirely from pred_log data — no hardcoded values.
    Three factors combined per signal:
      1. Recent accuracy  (how often was it right in last n_recent rounds?)
      2. Informativeness  (does it deviate from 0.5? if not, it's useless)
      3. Volatility       (does it swing wildly round-to-round? if so, it's noise)
    Returns normalised weight dict, or None if insufficient data.
    """
    if not os.path.exists(LOG_PATH):
        return None
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)
    scored = [(k, v) for k, v in log.items()
              if v.get('actual') and v.get('signals')]
    if len(scored) < 10:
        return None
    scored.sort(key=lambda x: int(x[0]))
    recent = scored[-n_recent:]

    # 1. Accuracy-based raw weight
    signal_names = ['run', 'momentum', 'seq', 'disc_seq', 'persist', 'recent', 'period', 'reversal', 'breakout', 'ml', 'lstm']
    accs = {}
    for name in signal_names:
        hits = total = 0
        for _, entry in recent:
            p = entry['signals'].get(name)
            if p is None:
                continue
            actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
            if ('ODD' if p > 0.5 else 'EVEN') == actual_oe:
                hits += 1
            total += 1
        if total >= 5:
            accs[name] = hits / total

    if not accs:
        return None

    # Floor at 0.50: signals at or below chance get near-zero weight.
    # Signals consistently below 50% are anti-predictive and should not contribute.
    raw = {k: max(0.001, v - 0.50) for k, v in accs.items()}

    # 2 & 3. Apply informativeness and volatility penalties from data
    ss = _signal_stats(recent)
    for sig in list(raw.keys()):
        info = ss.get(sig, {}).get('info', 0.05)
        vol  = ss.get(sig, {}).get('volatility', 0.10)

        # Informativeness penalty: signal stuck near 0.5 carries no information.
        # Computed threshold: mean info across all signals. Below 40% of mean → penalise.
        mean_info = np.mean([s['info'] for s in ss.values()]) if ss else 0.05
        if info < mean_info * 0.4:
            raw[sig] *= (info / (mean_info * 0.4)) ** 2   # quadratic suppression

        # Volatility penalty: inter-round swing much larger than average → noisy.
        mean_vol = np.mean([s['volatility'] for s in ss.values()]) if ss else 0.10
        if vol > mean_vol * 1.5:
            raw[sig] *= mean_vol * 1.5 / vol   # scale down proportionally

    total = sum(raw.values())
    if total == 0:
        return None
    return {k: v / total for k, v in raw.items()}

_aw = _adaptive_weights()

# ── Streak correction ──────────────────────────────────────────────────────────
def _streak_correction(min_streak=3):
    """
    If the last min_streak+ predicted results are all LOSS, analyse which
    signals were consistently wrong and return:
      - weight multipliers: suppress wrong signals, boost reliable ones
      - invert_signals: set of signal names to flip (>70% wrong → invert value)
    Returns (streak_len, {signal: multiplier}, {signals_to_invert}).
    """
    if 'result' not in df.columns:
        return 0, {}, set()
    results = df[df['result'].isin(['WIN', 'LOSS'])]['result'].tolist()
    if len(results) < min_streak:
        return 0, {}, set()

    streak = 0
    for r in reversed(results):
        if r == 'LOSS':
            streak += 1
        else:
            break
    if streak < min_streak:
        return 0, {}, set()

    losing_ids = set(
        df[df['result'] == 'LOSS']['id'].astype(str).tolist()[-streak:]
    )

    if not os.path.exists(LOG_PATH):
        return streak, {}, set()
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)

    wrong  = {}
    totals = {}
    for rid in losing_ids:
        entry = log.get(rid)
        if not entry or not entry.get('signals') or not entry.get('actual'):
            continue
        actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
        for sig, val in entry['signals'].items():
            if val is None:
                continue
            totals[sig] = totals.get(sig, 0) + 1
            if ('ODD' if val > 0.5 else 'EVEN') != actual_oe:
                wrong[sig] = wrong.get(sig, 0) + 1

    if not totals:
        return streak, {}, set()

    multipliers   = {}
    invert_signals = set()
    for sig, n in totals.items():
        wrong_rate = wrong.get(sig, 0) / n
        multipliers[sig] = max(0.10, 1.20 - wrong_rate * 1.10)
        # Invert signal value if it was wrong >70% of the time during the streak
        if wrong_rate >= 0.70:
            invert_signals.add(sig)

    return streak, multipliers, invert_signals

_streak_len, _streak_mults, _streak_invert_sigs = _streak_correction()

# ── Direction bias correction ──────────────────────────────────────────────────
def _direction_correction(log, n=20):
    """
    Compare predicted ODD rate vs actual ODD rate over the last n rounds.
    If we over-predict EVEN (low pred_odd_rate) but actuals are more ODD,
    return a positive nudge toward ODD. Scale: roughly ±0.15 max.
    """
    scored = sorted(
        [e for e in log.values() if e.get('pred_oe') and e.get('actual')],
        key=lambda e: int(e.get('round_id', 0))
    )[-n:]
    if len(scored) < 8:
        return 0.0
    actual_odd = sum(1 for e in scored if e['actual'].count('R') % 2 == 1) / len(scored)
    pred_odd   = sum(1 for e in scored if e['pred_oe'] == 'ODD') / len(scored)
    # positive = we under-predict ODD → nudge upward; negative = over-predict ODD → nudge down
    return (actual_odd - pred_odd) * 0.5


def _anti_streak_signal(log, streak=4):
    """
    If the last `streak` predictions were all the same direction AND all LOSS,
    return a strong counter-signal (flip direction).
    Returns p_odd float in 0..1 or None (no strong counter-signal found).
    """
    scored = sorted(
        [e for e in log.values() if e.get('pred_oe') and e.get('actual')],
        key=lambda e: int(e.get('round_id', 0))
    )
    if len(scored) < streak:
        return None
    last = scored[-streak:]
    dirs = [e['pred_oe'] for e in last]
    hits = [e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN') for e in last]
    if all(d == 'EVEN' for d in dirs) and not any(hits):
        return 0.72   # all EVEN predictions all wrong → flip toward ODD
    if all(d == 'ODD'  for d in dirs) and not any(hits):
        return 0.28   # all ODD predictions all wrong → flip toward EVEN
    return None


def _loss_streak_autopsy(log, min_streak=5):
    """
    When loss streak >= min_streak, find signals that were wrong on every loss
    in that streak. Returns set of signal names to hard-suppress this round.
    Only fires when ALL recent min_streak predictions were losses.
    """
    scored = sorted(
        [e for e in log.values() if e.get('pred_oe') and e.get('actual')],
        key=lambda e: int(e.get('round_id', 0))
    )
    if len(scored) < min_streak:
        return set()
    last = scored[-min_streak:]
    all_losses = not any(
        e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN')
        for e in last
    )
    if not all_losses:
        return set()
    bad = set()
    for name in ('run', 'momentum', 'seq', 'disc_seq', 'recent', 'ml', 'lstm', 'period'):
        wrong = total = 0
        for entry in last:
            eff = entry.get('effective_signals')
            p = (eff.get(name) if eff else None) or entry.get('signals', {}).get(name)
            if p is None or abs(p - 0.5) < 0.02:
                continue
            actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
            if ('ODD' if p > 0.5 else 'EVEN') != actual_oe:
                wrong += 1
            total += 1
        if total >= 3 and wrong / total >= 0.80:
            bad.add(name)
    return bad


def _oe_dominant_signal(full_oe_arr, window=25):
    """
    Count ODD vs EVEN in the last `window` rounds.
    If one direction is dominant (>55%), return p_odd bias toward it.
    Rationale: dominant OE direction rarely reverses more than 3 consecutive
    times, so staying on the dominant side gives ~55% WR with max 3-loss streaks.
    """
    recent = list(full_oe_arr[-window:]) if len(full_oe_arr) >= window else list(full_oe_arr)
    if len(recent) < 10:
        return None
    n_odd  = sum(1 for x in recent if x == 'ODD')
    n_even = len(recent) - n_odd
    margin = abs(n_odd - n_even) / len(recent)
    if margin < 0.10:
        return None
    dominant = 'ODD' if n_odd > n_even else 'EVEN'
    strength = min(0.20, margin * 1.8)
    return (0.5 + strength) if dominant == 'ODD' else (0.5 - strength)


def _oe_block_streak_signal(full_oe_arr, switch_after=3):
    """
    OE moves in blocks of 2-4 same direction then switches.
    After switch_after+ consecutive same OE: strongly predict the opposite.
    Validated on 2061000305-2061000317: 83%+ accuracy (10W/2L) with switch_after=3.
    Returns (p_odd, weight) — both None/0 when streak is below threshold.
    """
    if len(full_oe_arr) < switch_after:
        return None, 0.0
    cur_val = full_oe_arr[-1]
    streak = 0
    for x in reversed(full_oe_arr):
        if x == cur_val:
            streak += 1
        else:
            break
    if streak < switch_after:
        return None, 0.0
    opposite = 'ODD' if cur_val == 'EVEN' else 'EVEN'
    # Confidence grows with each extra round beyond the threshold
    strength = min(0.22, 0.14 + (streak - switch_after) * 0.04)
    p_odd = (0.5 + strength) if opposite == 'ODD' else (0.5 - strength)
    return p_odd, strength


def _global_inversion_check(log, n=50, threshold=0.47):
    """
    If the model has been anti-predictive over the last n rounds (win rate < threshold),
    return True to flip the final prediction globally.
    Only triggers when we have enough samples and the inversion would clearly help.
    """
    scored = sorted(
        [e for e in log.values() if e.get('pred_oe') and e.get('actual')],
        key=lambda e: int(e.get('round_id', 0))
    )[-n:]
    if len(scored) < 20:
        return False
    wins = sum(1 for e in scored
               if e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN'))
    wr = wins / len(scored)
    return wr < threshold


def _realtime_accuracy_signal(log, n=15):
    """
    For each direction, compute recent win rate.
    If ODD predictions win >> 50%, boost ODD. If EVEN wins >> 50%, boost EVEN.
    Returns p_odd correction in -0.15..+0.15 or 0.0 if insufficient data.
    """
    scored = sorted(
        [e for e in log.values() if e.get('pred_oe') and e.get('actual')],
        key=lambda e: int(e.get('round_id', 0))
    )[-n:]
    if len(scored) < 8:
        return 0.0
    odd_hits  = sum(1 for e in scored if e['pred_oe'] == 'ODD'  and e['actual'].count('R') % 2 == 1)
    odd_total = sum(1 for e in scored if e['pred_oe'] == 'ODD')
    eve_hits  = sum(1 for e in scored if e['pred_oe'] == 'EVEN' and e['actual'].count('R') % 2 == 0)
    eve_total = sum(1 for e in scored if e['pred_oe'] == 'EVEN')
    odd_wr = odd_hits / odd_total if odd_total >= 4 else None
    eve_wr = eve_hits / eve_total if eve_total >= 4 else None
    if odd_wr is None and eve_wr is None:
        return 0.0
    # If ODD is winning much more than EVEN → boost ODD; vice versa
    odd_edge = (odd_wr - 0.50) if odd_wr is not None else 0.0
    eve_edge = (eve_wr - 0.50) if eve_wr is not None else 0.0
    return (odd_edge - eve_edge) * 0.3   # scale to ≈ ±0.15


def _recent_signal_accuracy(log, n_last=100, min_n=10):
    """
    Compute per-signal accuracy over the last n_last scored rounds.
    Used to detect anti-predictive signals and suppress or gate them.
    Returns {name: (accuracy, n_samples)}.
    """
    scored = [(k, v) for k, v in log.items()
              if v.get('actual') and v.get('signals')]
    scored.sort(key=lambda x: int(x[0]))
    recent = scored[-n_last:]
    result = {}
    all_sigs = ['run','momentum','seq','disc_seq','persist','recent',
                'period','reversal','breakout','ml','lstm']
    for name in all_sigs:
        hits = total = 0
        for _, entry in recent:
            # Prefer effective (post-inversion) signal values to avoid feedback loops
            eff = entry.get('effective_signals')
            p = (eff.get(name) if eff else None) or entry['signals'].get(name)
            if p is None:
                continue
            actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
            if ('ODD' if p > 0.5 else 'EVEN') == actual_oe:
                hits += 1
            total += 1
        if total >= min_n:
            result[name] = (hits / total, total)
    return result


def _per_signal_short_streak(log, n_last=5):
    """
    Detect signals >70% wrong in the last n_last scored rounds → suppress (zero-weight).
    Inversion is handled exclusively by _gate_p (100-round window) to avoid
    double-inversion conflicts that cancel out and leave wrong signals active.
    Returns (empty_set, suppress_set).
    """
    scored = [(k, v) for k, v in log.items()
              if v.get('actual') and v.get('signals')]
    scored.sort(key=lambda x: int(x[0]))
    recent = scored[-n_last:]
    if len(recent) < n_last:
        return set(), set()
    suppress = set()
    all_sigs = ['run','momentum','seq','disc_seq','persist','recent',
                'period','reversal','breakout','ml','lstm']
    for name in all_sigs:
        wrong = total = 0
        for _, entry in recent:
            eff = entry.get('effective_signals')
            p = (eff.get(name) if eff else None) or entry['signals'].get(name)
            if p is None:
                continue
            if abs(p - 0.5) < 0.02:
                continue
            actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
            if ('ODD' if p > 0.5 else 'EVEN') != actual_oe:
                wrong += 1
            total += 1
        if total >= 4 and wrong / total >= 0.80:
            suppress.add(name)
    return set(), suppress


def _consecutive_loss_streak(log):
    """Count current consecutive prediction losses (most recent first)."""
    scored = [(int(k), v) for k, v in log.items()
              if v.get('actual') and v.get('pred_oe')]
    scored.sort(key=lambda x: x[0], reverse=True)
    streak = 0
    for _, entry in scored:
        actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
        if entry['pred_oe'] != actual_oe:
            streak += 1
        else:
            break
    return streak


def _last_actual_fallback(log, streak_threshold=3):
    """
    After streak_threshold consecutive losses, fall back to last-actual strategy:
    - If last 2 actuals are same OE → predict that direction (stronger signal, 0.58/0.42)
    - If only last 1 → predict that direction (weak signal, 0.53/0.47)
    Returns p_odd override or None if streak not reached.
    """
    scored = sorted(
        [e for e in log.values() if e.get('actual') and e.get('pred_oe')],
        key=lambda e: int(e.get('round_id', 0))
    )
    if len(scored) < streak_threshold:
        return None
    # Check current streak
    streak = 0
    for e in reversed(scored):
        actual_oe = 'ODD' if e['actual'].count('R') % 2 else 'EVEN'
        if e['pred_oe'] != actual_oe:
            streak += 1
        else:
            break
    if streak < streak_threshold:
        return None
    # Get last 2 actual OEs
    last_oe  = 'ODD' if scored[-1]['actual'].count('R') % 2 else 'EVEN'
    prev_oe  = 'ODD' if scored[-2]['actual'].count('R') % 2 else 'EVEN' if len(scored) >= 2 else None
    if prev_oe and last_oe == prev_oe:
        # Same trend twice → stronger signal
        return 0.58 if last_oe == 'ODD' else 0.42
    else:
        # Just last actual
        return 0.53 if last_oe == 'ODD' else 0.47


_log_for_corrections = {}
if os.path.exists(LOG_PATH):
    try:
        with open(LOG_PATH, encoding='utf-8-sig') as _f:
            _log_for_corrections = json.load(_f)
    except Exception:
        pass

_dir_correction        = _direction_correction(_log_for_corrections)
_anti_streak_p         = _anti_streak_signal(_log_for_corrections)
_rt_acc_signal         = _realtime_accuracy_signal(_log_for_corrections)
_sig_acc               = _recent_signal_accuracy(_log_for_corrections)
_short_streak_inverts, _short_streak_suppress = _per_signal_short_streak(_log_for_corrections)
_cur_loss_streak  = _consecutive_loss_streak(_log_for_corrections)
_streak_autopsy   = _loss_streak_autopsy(_log_for_corrections, min_streak=5)
_fallback_p_odd   = _last_actual_fallback(_log_for_corrections, streak_threshold=3)
_global_invert    = _global_inversion_check(_log_for_corrections, n=50, threshold=0.47)

# ── Predict ────────────────────────────────────────────────────────────────────
ALL_PATTERNS = [a+b+c+d for a in 'WR' for b in 'WR' for c in 'WR' for d in 'WR']

def predict_round(df_src, runs_src, full_oe_arr, full_pat_arr, lstm_src=None):
    ml_idx  = len(df_src) - 1       # used only for ML feature window
    sig_idx = len(full_oe_arr) - 1  # used for all signal computation

    if ml_idx < SEQ_LEN:
        return None

    # ── Signal 1: balance deficit ──────────────────────────────────────────
    deficit = balance_signal(full_oe_arr, sig_idx)
    # deficit > 0 → ODD is behind → boost ODD
    # Map deficit (-1..+1) to a probability nudge
    p_balance = 0.5 + deficit * 0.4    # max ±0.4 nudge from balance

    # ── Signal 2: run-length model ─────────────────────────────────────────
    cur_val = full_oe_arr[sig_idx]
    # count current run length
    cur_run_len = 1
    for i in range(sig_idx - 1, -1, -1):
        if full_oe_arr[i] == cur_val:
            cur_run_len += 1
        else:
            break
    p_continue, next_len_dist = run_length_signal(runs_src, cur_val, cur_run_len)
    # p_continue = prob the current run keeps going (same OE next)
    if cur_val == 'ODD':
        p_run = p_continue          # prob next is ODD
    else:
        p_run = 1.0 - p_continue    # prob next is ODD = prob current EVEN run ends

    # ── Signal 0 (priority): recent OE-pattern repetition ────────────────
    p_recent, recent_n, recent_len = recent_repeat_signal(
        full_oe_arr, full_pat_arr, sig_idx, look_back=20, hist_window=100
    )

    # ── Signal P: autocorrelation period detection ────────────────────────
    p_period, period_lag = period_signal(full_oe_arr, sig_idx)

    # ── Signal R: post-long-run reversal check ────────────────────────────
    p_reversal = post_run_reversal_signal(full_oe_arr, sig_idx)

    # ── Signal B: rhythm breakout (short bursts → long flow) ──────────────
    p_breakout = rhythm_breakout_signal(full_oe_arr, sig_idx)

    # ── Signal SP: post-WWWW/RRRR offset effect ──────────────────────────
    p_special = post_special_pattern_signal(full_pat_arr, sig_idx)

    # ── Signal LB: long block continuation + transition ───────────────────
    p_longblock, w_longblock = long_block_signal(full_oe_arr, sig_idx)

    # ── Signal ALT: OE/EO alternating pattern continuation (domain rule) ─
    p_alt, w_alt_strength, alt_confirmed = oe_alternating_signal(full_oe_arr, sig_idx, min_alt_len=3)

    # ── Signal OE: OE macro-phase (alternating / block / transition) ─────
    p_oe_phase, oe_phase_conf, oe_phase_label, oe_phase_n = \
        oe_macro_phase_signal(full_oe_arr, sig_idx, threshold=0.55, min_n=50)

    # ── Signal 3: OE sequence matcher (window 5-20) ───────────────────────
    p_seq, pat_votes_seq, matched_len, n_matches = \
        sequence_signal(full_oe_arr, full_pat_arr, sig_idx)

    # ── Signal 4: short-term momentum ─────────────────────────────────────
    p_momentum = momentum_signal(full_oe_arr, sig_idx)

    # ── Signal 5: disc pattern sequence matcher ────────────────────────────
    p_disc_seq, disc_len, disc_n = disc_sequence_signal(full_pat_arr, sig_idx)

    # ── Signal 6: persistent run continuation (run >= 6) ──────────────────
    p_persist, persist_run_len = persistent_run_signal(full_oe_arr, sig_idx)

    # ── Signal 7: trend clarity (alternation-rate based confidence) ────────
    p_clarity, clarity_score = trend_clarity_signal(full_oe_arr, sig_idx)

    # ── LSTM prediction ────────────────────────────────────────────────────
    lstm_p_odd = None
    if lstm_model is not None and lstm_src is not None and len(lstm_src) >= 200:
        lstm_p_odd = _lstm_predict(lstm_model, lstm_src)

    # ── Sklearn ML fallback ────────────────────────────────────────────────
    ml_p_odd = None
    rc_probs_ml  = {}
    pat_probs_ml = {}
    if gb_oe is not None:
        feat = make_features(df_src, ml_idx)
        f    = np.array(feat).reshape(1, -1)
        ml_p_odd    = (gb_oe.predict_proba(f)[0][1] + lr_oe.predict_proba(f)[0][1]) / 2
        rc_probs_ml = dict(zip(gb_rc.classes_,  gb_rc.predict_proba(f)[0]))
        pat_probs_ml= dict(zip(rf_pat.classes_, rf_pat.predict_proba(f)[0]))

    # ── Combine signals with consensus amplification ──────────────────────
    p_seq_val      = p_seq      if p_seq      is not None else 0.5
    p_disc_seq_val = p_disc_seq if p_disc_seq is not None else 0.5
    p_persist_val  = p_persist  if p_persist  is not None else 0.5
    p_recent_val   = p_recent   if p_recent   is not None else 0.5
    p_period_val   = p_period   if p_period   is not None else 0.5
    p_reversal_val = p_reversal if p_reversal is not None else 0.5
    p_breakout_val = p_breakout if p_breakout is not None else 0.5
    p_special_val  = p_special  if p_special  is not None else 0.5
    p_oe_phase_val  = p_oe_phase  if p_oe_phase  is not None else 0.5
    p_alt_val       = p_alt       if p_alt       is not None else 0.5
    p_longblock_val = p_longblock if p_longblock is not None else 0.5

    # ── Anti-predictive signal gate ───────────────────────────────────────
    # If a signal's recent accuracy (from pred_log) is below 50% with enough
    # samples, treat it as unreliable: zero its priority weight and flip its
    # value so it cannot drive confident wrong predictions.
    # Threshold: <50% accuracy over ≥20 samples.
    def _acc_ok(sig_name):
        acc, n = _sig_acc.get(sig_name, (0.5, 0))
        return not (acc < 0.50 and n >= 20)

    def _gate_p(p_val, sig_name):
        """Return p_val if signal is reliable, else flip to correct direction.
        Skips signals already suppressed to avoid overriding zero-weight intent."""
        if sig_name in _short_streak_suppress:
            return p_val  # already neutralised, don't invert
        acc, n = _sig_acc.get(sig_name, (0.5, 0))
        if acc < 0.50 and n >= 20:
            return 1.0 - p_val   # invert anti-predictive signal
        return p_val

    # Apply gate to priority signal values before weight calculation
    if p_recent   is not None: p_recent_val   = _gate_p(p_recent_val,   'recent')
    if p_period   is not None: p_period_val   = _gate_p(p_period_val,   'period')
    if p_reversal is not None: p_reversal_val = _gate_p(p_reversal_val, 'reversal')
    if p_breakout is not None: p_breakout_val = _gate_p(p_breakout_val, 'breakout')

    # Combined invert set: overall loss streak inversions + per-signal short streak inversions
    _all_invert_sigs = _streak_invert_sigs | _short_streak_inverts

    # Invert any signal that was >70% wrong during the loss streak OR in the last 5 rounds
    if _all_invert_sigs:
        def _streak_flip(val, name):
            return (1.0 - val) if name in _all_invert_sigs else val
        p_run          = _streak_flip(p_run,          'run')
        p_momentum     = _streak_flip(p_momentum,     'momentum')
        p_seq_val      = _streak_flip(p_seq_val,      'seq')
        p_disc_seq_val = _streak_flip(p_disc_seq_val, 'disc_seq')
        p_persist_val  = _streak_flip(p_persist_val,  'persist')
        p_recent_val   = _streak_flip(p_recent_val,   'recent')
        p_period_val   = _streak_flip(p_period_val,   'period')
        if ml_p_odd    is not None: ml_p_odd   = _streak_flip(ml_p_odd,   'ml')
        if lstm_p_odd  is not None: lstm_p_odd = _streak_flip(lstm_p_odd, 'lstm')

    # Suppress (zero-weight) signals that are >70% wrong in the last 5 rounds — clearly anti-predictive
    # Also applies autopsy suppression when 5+ consecutive losses occur
    _all_suppress = _short_streak_suppress | _streak_autopsy
    def _suppress_sig(val, name):
        return 0.5 if name in _all_suppress else val
    p_run          = _suppress_sig(p_run,          'run')
    p_momentum     = _suppress_sig(p_momentum,     'momentum')
    p_seq_val      = _suppress_sig(p_seq_val,      'seq')
    p_disc_seq_val = _suppress_sig(p_disc_seq_val, 'disc_seq')
    p_persist_val  = _suppress_sig(p_persist_val,  'persist')
    p_recent_val   = _suppress_sig(p_recent_val,   'recent')
    p_period_val   = _suppress_sig(p_period_val,   'period')
    if ml_p_odd   is not None: ml_p_odd   = _suppress_sig(ml_p_odd,   'ml')
    if lstm_p_odd is not None: lstm_p_odd = _suppress_sig(lstm_p_odd, 'lstm')

    # Period signal weight: scales with autocorrelation strength
    if p_period is not None:
        period_conf  = abs(p_period_val - 0.5)
        w_period_raw = min(0.30, period_conf * 2.5)
    else:
        w_period_raw = 0.0

    # Reversal signal weight: scales with confidence; fires rarely but strongly
    if p_reversal is not None:
        reversal_conf  = abs(p_reversal_val - 0.5)
        w_reversal_raw = min(0.30, reversal_conf * 2.0)
    else:
        w_reversal_raw = 0.0

    # Recent-repeat weight: scales with confidence and match count
    if p_recent is not None:
        recent_conf  = abs(p_recent_val - 0.5)
        w_recent_raw = min(0.30, recent_conf * min(recent_n, 6) * 0.20)
    else:
        w_recent_raw = 0.0

    # Breakout weight: scales with confidence
    if p_breakout is not None:
        breakout_conf  = abs(p_breakout_val - 0.5)
        w_breakout_raw = min(0.25, breakout_conf * 2.0)
    else:
        w_breakout_raw = 0.0

    # Special pattern signal weight: fixed small weight when active (3.5% edge → 0.12 weight)
    if p_special is not None:
        w_special_raw = min(0.12, abs(p_special_val - 0.5) * 3.0)
    else:
        w_special_raw = 0.0

    # OE macro-phase signal weight: scales with confidence and sample count
    # Higher weight than other priority signals — this directly encodes phase patterns
    if p_oe_phase is not None:
        # Scale by confidence strength; bonus for large sample counts
        n_bonus = min(0.05, (oe_phase_n - 20) / 1000)
        w_oe_phase_raw = min(0.45, oe_phase_conf * 3.0 + n_bonus)
    else:
        w_oe_phase_raw = 0.0

    # ALT signal: domain rule — weight scales with pattern purity
    w_alt_raw = w_alt_strength if p_alt is not None else 0.0

    # Combined priority weight (period + reversal + recent + breakout + special + oe_phase + alt), capped at 70%
    priority_total = min(0.70, w_period_raw + w_reversal_raw + w_recent_raw + w_breakout_raw + w_special_raw + w_oe_phase_raw + w_alt_raw)
    raw_sum = w_period_raw + w_reversal_raw + w_recent_raw + w_breakout_raw + w_special_raw + w_oe_phase_raw + w_alt_raw
    if raw_sum > 0:
        ratio = priority_total / raw_sum
        w_period_raw    *= ratio
        w_reversal_raw  *= ratio
        w_recent_raw    *= ratio
        w_breakout_raw  *= ratio
        w_special_raw   *= ratio
        w_oe_phase_raw  *= ratio

    if _aw is not None:
        w_run      = _aw.get('run',       0.0)
        w_momentum = _aw.get('momentum',  0.0)
        w_seq      = _aw.get('seq',       0.0) if p_seq       is not None else 0.0
        w_disc_seq = _aw.get('disc_seq',  0.0) if p_disc_seq  is not None else 0.0
        w_persist  = _aw.get('persist',   0.0) if p_persist   is not None else 0.0
        w_ml       = _aw.get('ml',        0.0)
        w_lstm     = _aw.get('lstm',      0.0) if lstm_p_odd  is not None else 0.0
    else:
        active = ['run', 'momentum', 'ml']
        if p_seq      is not None: active.append('seq')
        if p_disc_seq is not None: active.append('disc_seq')
        if p_persist  is not None: active.append('persist')
        if lstm_p_odd is not None: active.append('lstm')
        eq         = 1.0 / len(active)
        w_run      = eq
        w_momentum = eq
        w_seq      = eq if p_seq      is not None else 0.0
        w_disc_seq = eq if p_disc_seq is not None else 0.0
        w_persist  = eq if p_persist  is not None else 0.0
        w_ml       = eq
        w_lstm     = eq if lstm_p_odd is not None else 0.0

    # Apply anti-predictive gate to regular signals: invert value if signal is
    # consistently below 50% accuracy in recent history.
    p_run       = _gate_p(p_run,       'run')
    p_momentum  = _gate_p(p_momentum,  'momentum')
    p_seq_val   = _gate_p(p_seq_val,   'seq')
    p_disc_seq_val = _gate_p(p_disc_seq_val, 'disc_seq')
    if p_persist  is not None: p_persist_val  = _gate_p(p_persist_val,  'persist')
    ml_p_odd    = _gate_p(ml_p_odd,    'ml')
    if lstm_p_odd is not None: lstm_p_odd = _gate_p(lstm_p_odd, 'lstm')

    # Capture effective (post-all-inversions) signal values for pred_log storage.
    # These break the feedback loop: accuracy trackers read effective values, not raw ones.
    _eff_sigs = {
        'run':      round(p_run, 3),
        'momentum': round(p_momentum, 3),
        'seq':      round(p_seq_val, 3),
        'disc_seq': round(p_disc_seq_val, 3),
        'persist':  round(p_persist_val, 3) if p_persist is not None else None,
        'recent':   round(p_recent_val, 3)  if p_recent  is not None else None,
        'period':   round(p_period_val, 3)  if p_period  is not None else None,
        'reversal': round(p_reversal_val, 3) if p_reversal is not None else None,
        'breakout': round(p_breakout_val, 3) if p_breakout is not None else None,
        'special':   round(p_special_val, 3)  if p_special  is not None else None,
        'oe_phase':  round(p_oe_phase_val, 3) if p_oe_phase is not None else None,
        'ml':        round(ml_p_odd, 3)       if ml_p_odd  is not None else None,
        'lstm':      round(lstm_p_odd, 3)     if lstm_p_odd is not None else None,
    }

    # Apply streak-correction multipliers
    if _streak_mults:
        w_run      *= _streak_mults.get('run',      1.0)
        w_momentum *= _streak_mults.get('momentum', 1.0)
        w_seq      *= _streak_mults.get('seq',      1.0)
        w_disc_seq *= _streak_mults.get('disc_seq', 1.0)
        w_persist  *= _streak_mults.get('persist',  1.0)
        w_ml       *= _streak_mults.get('ml',       1.0)
        if lstm_p_odd is not None:
            w_lstm *= _streak_mults.get('lstm', 1.0)

    # Normalise other signals to leave room for priority signals
    hist_total = (w_run + w_momentum + w_seq + w_disc_seq + w_persist + w_ml +
                  (w_lstm if lstm_p_odd is not None else 0.0))
    if hist_total == 0:
        hist_total = 1.0
    scale = (1.0 - priority_total) / hist_total
    w_run      *= scale;  w_momentum *= scale;  w_seq      *= scale
    w_disc_seq *= scale;  w_persist  *= scale;  w_ml       *= scale
    if lstm_p_odd is not None:
        w_lstm *= scale
    w_recent   = w_recent_raw
    w_period   = w_period_raw
    w_reversal = w_reversal_raw
    w_breakout = w_breakout_raw

    # Trend clarity: when present (clarity>=0.4), contribute as signal and
    # scale all other signal weights by clarity so choppy periods reduce confidence
    w_clarity = 0.0
    if p_clarity is not None:
        # Weight scales with clarity strength — data shows clarity=1.0 → 58.3% WR
        w_clarity = clarity_score * 0.20   # up to 20% weight at perfect clarity
        # Also dampen all other signals proportionally during choppy periods
        chop_scale = 0.5 + clarity_score * 0.5   # 0.5 at full alternation, 1.0 at full trend
        w_run *= chop_scale;  w_momentum *= chop_scale;  w_seq *= chop_scale
        w_disc_seq *= chop_scale;  w_persist *= chop_scale;  w_ml *= chop_scale
        w_recent *= chop_scale;   w_period *= chop_scale
        if lstm_p_odd is not None: w_lstm *= chop_scale

    # Fix 2: suppress near-neutral signals — if a signal is within ±0.02 of 0.5
    # it carries no real directional information but can still drag the result wrong.
    if abs(p_run - 0.5) < 0.02:        w_run      = 0.0
    if abs(p_momentum - 0.5) < 0.02:   w_momentum = 0.0
    if abs(p_seq_val - 0.5) < 0.02:    w_seq      = 0.0
    if abs(p_disc_seq_val - 0.5) < 0.02: w_disc_seq = 0.0
    if abs(p_persist_val - 0.5) < 0.02:  w_persist  = 0.0
    if ml_p_odd is not None and abs(ml_p_odd - 0.5) < 0.02: w_ml = 0.0

    # Fix 3: emergency low win-rate mode.
    # When overall win rate over last 10 scored rounds drops below 35%,
    # zero out signals that have been wrong >55% recently and give full weight to ml.
    _recent10 = [v for _, v in sorted(
        [(k, v) for k, v in _log_for_corrections.items() if v.get('actual') and v.get('signals')],
        key=lambda x: int(x[0]))
    ][-10:]
    _wr10 = (sum(1 for e in _recent10
                 if e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN'))
             / len(_recent10)) if _recent10 else 0.5

    if _wr10 < 0.35 and ml_p_odd is not None:
        # Compute per-signal accuracy over last 10
        def _sig_wr10(name):
            w = t = 0
            for e in _recent10:
                p = e['signals'].get(name)
                if p is None or abs(p - 0.5) < 0.02:
                    continue
                aoe = 'ODD' if e['actual'].count('R') % 2 else 'EVEN'
                if ('ODD' if p > 0.5 else 'EVEN') == aoe:
                    w += 1
                t += 1
            return w / t if t >= 3 else 0.5
        for _sname, _sval, _sw_ref in [
            ('run',      p_run,          'w_run'),
            ('momentum', p_momentum,     'w_momentum'),
            ('seq',      p_seq_val,      'w_seq'),
            ('disc_seq', p_disc_seq_val, 'w_disc_seq'),
        ]:
            if _sig_wr10(_sname) < 0.45:
                if _sw_ref == 'w_run':      w_run      = 0.0
                elif _sw_ref == 'w_momentum': w_momentum = 0.0
                elif _sw_ref == 'w_seq':    w_seq      = 0.0
                elif _sw_ref == 'w_disc_seq': w_disc_seq = 0.0
        # Re-normalise remaining weights with ml boosted
        _total_w = w_run + w_momentum + w_seq + w_disc_seq + w_persist + w_ml + w_lstm
        if _total_w > 0:
            w_ml = max(w_ml, _total_w * 0.50)  # ml gets at least 50% in emergency mode

    signals_w = [(p_run, w_run), (p_momentum, w_momentum),
                 (p_seq_val, w_seq), (p_disc_seq_val, w_disc_seq),
                 (p_persist_val, w_persist), (ml_p_odd, w_ml),
                 (p_recent_val, w_recent), (p_period_val, w_period),
                 (p_reversal_val, w_reversal), (p_breakout_val, w_breakout)]
    if p_special    is not None: signals_w.append((p_special_val,    w_special_raw))
    if p_oe_phase   is not None: signals_w.append((p_oe_phase_val,  w_oe_phase_raw))
    if p_alt        is not None: signals_w.append((p_alt_val,       w_alt_raw))
    if p_longblock  is not None: signals_w.append((p_longblock_val, w_longblock))
    if lstm_p_odd   is not None: signals_w.append((lstm_p_odd,      w_lstm))
    if p_clarity   is not None: signals_w.append((p_clarity,       w_clarity))
    p_base = sum(v * w for v, w in signals_w)

    # Step 2: consensus amplification
    active_signals = [p_run, p_momentum, p_seq_val, p_disc_seq_val, ml_p_odd]
    if p_recent   is not None: active_signals.append(p_recent_val)
    if p_period   is not None: active_signals.append(p_period_val)
    if p_reversal is not None: active_signals.append(p_reversal_val)
    if p_breakout is not None: active_signals.append(p_breakout_val)
    if p_persist  is not None: active_signals.append(p_persist_val)
    if p_special  is not None: active_signals.append(p_special_val)
    if p_oe_phase is not None: active_signals.append(p_oe_phase_val)
    if p_alt      is not None: active_signals.append(p_alt_val)
    if lstm_p_odd is not None: active_signals.append(lstm_p_odd)

    # Count how many agree with the majority direction
    majority_odd = p_base > 0.5
    n_agree = sum(1 for s in active_signals if (s > 0.5) == majority_odd)
    agreement = n_agree / len(active_signals)   # 0.0 → 1.0

    # Minimum signed deviation in majority direction (how weakly does the weakest signal agree?)
    min_dev = min((s - 0.5) if majority_odd else (0.5 - s) for s in active_signals)

    # Amplifier: scales with both agreement and mean signal strength.
    # Removed dev_bonus (min signal) — replaced with mean deviation across active signals
    # so that one outlier weak signal can't suppress the whole agreement.
    mean_dev = float(np.mean([abs(s - 0.5) for s in active_signals])) if active_signals else 0.0
    strength_bonus = min(0.25, mean_dev * 2.5)
    amplifier = max(0.70, min(1.40, 0.75 + 0.40 * agreement + strength_bonus))
    if abs(p_base - 0.5) < 0.03:
        amplifier = 1.0

    p_odd = 0.5 + (p_base - 0.5) * amplifier

    # ── Apply live corrections (direction bias + anti-streak + rt accuracy) ──────
    # Anti-streak: if consecutive same-direction losses, override toward opposite
    if _anti_streak_p is not None:
        p_odd = p_odd * 0.4 + _anti_streak_p * 0.6   # strong override
    else:
        # Blend direction correction and real-time accuracy signal
        p_odd += _dir_correction * 0.5 + _rt_acc_signal * 0.5

    p_odd = max(0.05, min(0.95, p_odd))

    # ── PRIMARY DOMAIN RULES ─────────────────────────────────────────────────────
    # Priority: long block hard lock > confirmed alt hard lock > soft blend

    # 1. Long block hard lock (run >= 4): stay on block direction, ignore everything
    _lb_arr = list(full_oe_arr)
    _lb_cur = _lb_arr[sig_idx] if sig_idx < len(_lb_arr) else None
    _lb_run = 0
    if _lb_cur:
        for _i in range(sig_idx, max(-1, sig_idx - 60), -1):
            if _lb_arr[_i] == _lb_cur:
                _lb_run += 1
            else:
                break
    # Check if previous block was long (snap-back failed check)
    _prev_run = 0
    _prev_val = None
    if _lb_cur and _lb_run < 4:
        _switch_idx = sig_idx - _lb_run
        if _switch_idx >= 0:
            _prev_val = _lb_arr[_switch_idx]
            for _i in range(_switch_idx, max(-1, _switch_idx - 60), -1):
                if _lb_arr[_i] == _prev_val:
                    _prev_run += 1
                else:
                    break
    if _lb_run >= 4:
        # Inside long block — directional lean (reduced from 0.82 to avoid overconfident wrong predictions)
        p_odd = 0.62 if _lb_cur == 'ODD' else 0.38
    elif _lb_run >= 2 and _prev_run >= 4:
        # 2nd+ value after a long block — snap back failed, mirror block forming
        p_odd = 0.60 if _lb_cur == 'ODD' else 0.40

    # 2. Confirmed alternating (3+ unbroken): stay on start direction
    elif p_alt is not None and alt_confirmed:
        p_odd = p_alt_val

    # 3. No confirmed pattern: soft blend with dominant direction
    elif p_alt is not None:
        p_odd = p_alt_val * 0.70 + p_odd * 0.30

    p_odd = max(0.05, min(0.95, p_odd))

    # ── HARD STREAK BREAKER: force-flip at 5+ consecutive losses ─────────────────
    # Skip when global inversion is also active — both flips would cancel each other
    # and lock the model in a double-inversion loop that produces the wrong answer.
    if _cur_loss_streak >= 5 and not _global_invert:
        p_odd = 1.0 - p_odd

    # Last-actual fallback: after 3+ losses with no strong model signal.
    # Skip entirely if OE phase signal is confident — it already knows the phase.
    _oe_phase_confident = p_oe_phase is not None and oe_phase_conf >= 0.05  # >= 55% historical
    if _fallback_p_odd is not None and not _oe_phase_confident:
        model_conf = abs(p_odd - 0.5)
        if model_conf < 0.10:
            p_odd = _fallback_p_odd
        elif model_conf < 0.20:
            p_odd = 0.6 * _fallback_p_odd + 0.4 * p_odd

    p_odd = max(0.05, min(0.95, p_odd))

    # Global inversion: if model has been anti-predictive over last 50 rounds
    # (win rate < 47%), flip the prediction — the signals are systematically wrong
    if _global_invert:
        p_odd = 1.0 - p_odd

    p_odd = max(0.05, min(0.95, p_odd))

    # Confidence cap: ODD/EVEN transitions are near-random (~51-53% continuation).
    # Cap confidence at 0.72 to limit damage from overconfident wrong predictions.
    p_odd = max(0.28, min(0.72, p_odd))

    # Dominant OE trend: if one direction is dominant in last 25 rounds (>55%),
    # nudge toward it — dominant direction never runs opposite more than 3 consecutive.
    # Applied after all corrections so it isn't cancelled by global_invert.
    _dom_signal = _oe_dominant_signal(full_oe_arr)
    if _dom_signal is not None:
        p_odd = 0.60 * _dom_signal + 0.40 * p_odd
    p_odd = max(0.28, min(0.72, p_odd))

    # Block-streak: after 3+ consecutive same OE, switch direction.
    # OE moves in blocks of 2-4; staying past 3 is strongly predictive of a switch.
    # Overrides dominant signal when streak is clear — streak evidence is more specific.
    _block_p, _block_w = _oe_block_streak_signal(full_oe_arr)
    if _block_p is not None:
        p_odd = 0.70 * _block_p + 0.30 * p_odd
    p_odd = max(0.28, min(0.72, p_odd))

    # Pattern probs: blend seq matcher + ML
    conf_seq = abs(p_seq_val - 0.5) * 2
    if pat_votes_seq:
        sw_pat = min(0.75, 0.4 + conf_seq * 0.35)
        pat_probs = {p: sw_pat * pat_votes_seq.get(p, 0.0) +
                        (1 - sw_pat) * pat_probs_ml.get(p, 0.0)
                     for p in ALL_PATTERNS}
    else:
        pat_probs = {p: pat_probs_ml.get(p, 0.0) for p in ALL_PATTERNS}

    rc_probs = {k: rc_probs_ml.get(k, 0.0) for k in range(5)}
    total_rc = sum(rc_probs.values()) or 1
    by_rc    = {k: rc_probs[k] / total_rc for k in range(5)}
    by_oe    = {'ODD': p_odd, 'EVEN': 1 - p_odd}
    pred_pat = max(pat_probs, key=pat_probs.get)

    signal_info = {
        'run':        round(p_run, 3),
        'momentum':   round(p_momentum, 3),
        'seq':        round(p_seq_val, 3),
        'disc_seq':   round(p_disc_seq_val, 3),
        'persist':    round(p_persist_val, 3) if p_persist is not None else None,
        'lstm':       round(lstm_p_odd, 3) if lstm_p_odd is not None else None,
        'ml':         round(ml_p_odd, 3),
        'recent':     round(p_recent_val, 3)   if p_recent   is not None else None,
        'period':     round(p_period_val, 3)   if p_period   is not None else None,
        'clarity':    round(p_clarity, 3)      if p_clarity  is not None else None,
        'reversal':   round(p_reversal_val, 3) if p_reversal is not None else None,
        'breakout':   round(p_breakout_val, 3) if p_breakout is not None else None,
        'special':    round(p_special_val, 3)   if p_special  is not None else None,
        'oe_phase':   round(p_oe_phase_val, 3)  if p_oe_phase is not None else None,
        'oe_phase_label': oe_phase_label,
        'oe_phase_n': oe_phase_n,
        'deficit':    round(deficit, 3),
        'run_len':    cur_run_len,
        'persist_len':persist_run_len,
        'seq_len':    matched_len,
        'seq_n':      n_matches,
        'disc_len':   disc_len,
        'disc_n':     disc_n,
        'recent_n':   recent_n,
        'recent_len': recent_len,
        'period_lag': period_lag,
        'agreement':    round(agreement, 2),
        'amplifier':    round(amplifier, 2),
        'adaptive':     _aw is not None,
        'streak':       _streak_len,
        'w_run':        round(w_run, 3),
        'w_momentum':   round(w_momentum, 3),
        'w_seq':        round(w_seq, 3),
        'w_disc_seq':   round(w_disc_seq, 3),
        'w_persist':    round(w_persist, 3),
        'w_recent':     round(w_recent, 3),
        'w_period':     round(w_period, 3),
        'w_reversal':   round(w_reversal, 3),
        'w_breakout':   round(w_breakout, 3),
        'w_special':    round(w_special_raw, 3),
        'w_oe_phase':   round(w_oe_phase_raw, 3),
        'p_base':       round(p_base, 3),
        'dir_corr':        round(_dir_correction, 3),
        'anti_streak':     round(_anti_streak_p, 3) if _anti_streak_p is not None else None,
        'rt_acc':          round(_rt_acc_signal, 3),
        'effective_signals': _eff_sigs,
    }
    return pred_pat, pat_probs[pred_pat], by_rc, by_oe, signal_info

# Full history arrays — never truncate, always pass to predict_round for signals
full_oe_arr  = df['odd'].values
full_pat_arr = df['pattern'].values

# df_sim is only used for ML feature engineering (needs a DataFrame with r_count etc.)
# Keep it growing with simulated rows so ML features stay current
df_sim   = df.copy()
runs_sim = list(all_runs)   # extend this list as we simulate

results = []
sim_oe_list  = []   # appended OE values so far in simulation
sim_pat_list = []   # appended pattern values so far

for step in range(PREDICT_N):
    # Build current extended arrays (real history + simulated rounds so far)
    if sim_oe_list:
        step_oe_arr  = np.concatenate([full_oe_arr,  np.array(sim_oe_list)])
        step_pat_arr = np.concatenate([full_pat_arr, np.array(sim_pat_list)])
    else:
        step_oe_arr  = full_oe_arr
        step_pat_arr = full_pat_arr

    out = predict_round(df_sim, runs_sim, step_oe_arr, step_pat_arr, lstm_src=df_sim)
    if out is None:
        break
    pred_pat, conf, by_rc, by_oe, sinfo = out
    rid = last_id + 1 + step
    results.append({'rid': rid, 'pred': pred_pat, 'conf': conf,
                    'by_rc': by_rc, 'by_oe': by_oe, 'sinfo': sinfo})

    # Append simulated round to all tracking structures
    sim_oe  = 'ODD' if pred_pat.count('R') % 2 else 'EVEN'
    sim_oe_list.append(sim_oe)
    sim_pat_list.append(pred_pat)

    # Extend runs_sim with the new simulated value
    last_run = runs_sim[-1] if runs_sim else None
    if last_run and last_run[0] == sim_oe:
        runs_sim[-1] = (sim_oe, last_run[1] + 1)
    else:
        runs_sim.append((sim_oe, 1))

    # Extend df_sim for ML features in the next step
    sim_row = pd.DataFrame([{
        'id': rid, 'd1': pred_pat[0], 'd2': pred_pat[1],
        'd3': pred_pat[2], 'd4': pred_pat[3],
        'pattern': pred_pat, 'r_count': pred_pat.count('R'),
        'odd': sim_oe
    }])
    df_sim = pd.concat([df_sim, sim_row], ignore_index=True)

# ── Log predictions ────────────────────────────────────────────────────────────
log = {}
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding='utf-8-sig') as f:
        log = json.load(f)

# ── Phase analysis & confidence adjustment ─────────────────────────────────────
_phase_info = detect_phase(full_oe_arr, len(full_oe_arr) - 1)
_ph_wr, _ph_n = phase_win_rate(_phase_info['label'], log)
_kelly = kelly_analysis(_ph_wr)

if _ph_n < 10:
    _phase_action = 'INSUFFICIENT_DATA'
elif _ph_wr < 0.48:
    _phase_action = 'AVOID'
elif _ph_wr < 0.52:
    _phase_action = 'SKIP'
elif _ph_wr < 0.56:
    _phase_action = 'USE'
else:
    _phase_action = 'USE_STRONG'

# Adjust p_odd: boost when phase is historically reliable, dampen when unreliable
if results and _ph_wr is not None and _ph_n >= 10:
    r0 = results[0]
    _p_orig = r0['by_oe'].get('ODD', 0.5)
    _reliability = max(-1.0, min(1.0, (_ph_wr - 0.50) / 0.08))
    if _reliability >= 0:
        # Boost when phase is reliable, capped at 15%
        _p_adj = 0.5 + (_p_orig - 0.5) * (1.0 + _reliability * 0.15)
    else:
        # Dampen when phase is unreliable, up to 50% reduction
        _p_adj = 0.5 + (_p_orig - 0.5) * max(0.5, 1.0 + _reliability)
    _p_adj = max(0.05, min(0.95, _p_adj))
    results[0]['by_oe'] = {'ODD': round(_p_adj, 4), 'EVEN': round(1 - _p_adj, 4)}

# ── Calibration helpers ────────────────────────────────────────────────────────

def expected_calibration_error(pred_log, n_bins=5):
    """
    Measure how well the model's stated confidence matches actual win rate.
    ECE = sum_bins  (n_bin/N) * |mean_conf_bin - actual_win_rate_bin|
    Perfect calibration = 0.  Returns (ece, bin_detail_list).
    """
    scored = []
    for entry in pred_log.values():
        p = entry.get('p_oe_odd')
        actual = entry.get('actual')
        pred_oe = entry.get('pred_oe')
        if p is None or not actual or not pred_oe:
            continue
        actual_oe = 'ODD' if actual.count('R') % 2 else 'EVEN'
        scored.append((max(p, 1.0 - p), int(pred_oe == actual_oe)))
    if len(scored) < 20:
        return None, []
    N = len(scored)
    bw = 0.5 / n_bins
    bins = [(0.5 + i * bw, 0.5 + (i + 1) * bw) for i in range(n_bins)]
    detail, ece = [], 0.0
    for lo, hi in bins:
        b = [(c, ok) for c, ok in scored if lo <= c < hi]
        if not b:
            detail.append(((lo + hi) / 2, None, 0)); continue
        mc = float(np.mean([c for c, _ in b]))
        wr = float(np.mean([ok for _, ok in b]))
        detail.append((mc, wr, len(b)))
        ece += (len(b) / N) * abs(mc - wr)
    return round(ece, 4), detail


def _isotonic_calibrate(raw_p_odd, pred_log, min_n=150):
    """
    Isotonic regression calibration: fit (raw_confidence → actual_win_rate) on
    pred_log history, then map the current prediction's confidence to its
    empirically observed win rate.

    Why isotonic?  It enforces monotonicity (higher raw confidence → higher
    calibrated confidence) without parametric assumptions — appropriate for a
    non-linear, game-specific bias.

    Returns (calibrated_p_odd, n_samples_used).
    Falls back to identity if fewer than min_n scored predictions exist.
    """
    pairs = []
    for entry in pred_log.values():
        p = entry.get('p_oe_odd')
        actual = entry.get('actual')
        pred_oe = entry.get('pred_oe')
        if p is None or not actual or not pred_oe:
            continue
        actual_oe = 'ODD' if actual.count('R') % 2 else 'EVEN'
        pairs.append((max(p, 1.0 - p), int(pred_oe == actual_oe)))
    n = len(pairs)
    if n < min_n:
        return raw_p_odd, n
    pairs.sort(key=lambda x: x[0])
    try:
        from sklearn.isotonic import IsotonicRegression
        X = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
        ir.fit(X, y)
        raw_conf = max(raw_p_odd, 1.0 - raw_p_odd)
        cal_conf = float(ir.predict([raw_conf])[0])
        cal_conf = max(0.50, min(0.80, cal_conf))

        # Blend: calibration weight grows slowly with sample count.
        # At min_n (150): 15%.  At 500: ~40%.  At 1000+: capped at 55%.
        # Low early weight means calibration corrects over-confidence without
        # collapsing predictions to 50% when data is sparse.
        blend = min(0.55, 0.15 + (n - min_n) / 1500)
        blended_conf = (1.0 - blend) * raw_conf + blend * cal_conf
        direction = 1 if raw_p_odd > 0.5 else -1
        return 0.5 + direction * (blended_conf - 0.5), n
    except Exception:
        return raw_p_odd, n


# ── Isotonic calibration: map raw p_odd to empirically observed win rate ───────
_cal_n = 0
if results:
    _raw_p = results[0]['by_oe'].get('ODD', 0.5)
    _cal_p, _cal_n = _isotonic_calibrate(_raw_p, log)
    results[0]['by_oe'] = {'ODD': round(_cal_p, 4), 'EVEN': round(1 - _cal_p, 4)}

# Pre-compute confidence band win rates from existing log so each new entry can record them.
_PRED_BANDS = [(0.50,0.53,'50-53%'),(0.53,0.56,'53-56%'),(0.56,0.60,'56-60%'),
               (0.60,0.65,'60-65%'),(0.65,1.01,'65%+')]
_pre_band_stats: dict = {}
for _lo, _hi, _lbl in _PRED_BANDS:
    _bw = _bt = 0
    for _e in log.values():
        if not _e.get('actual') or _e.get('p_oe_odd') is None:
            continue
        _ec = max(_e['p_oe_odd'], 1.0 - _e['p_oe_odd'])
        if _lo <= _ec < _hi:
            _bt += 1
            if _e.get('pred_oe') == ('ODD' if _e['actual'].count('R') % 2 else 'EVEN'):
                _bw += 1
    _pre_band_stats[_lbl] = (round(_bw / _bt, 4) if _bt >= 20 else None, _bt)

def _lookup_band_wr(conf_val):
    for lo, hi, lbl in _PRED_BANDS:
        if lo <= conf_val < hi:
            return _pre_band_stats.get(lbl, (None, 0))
    return None, 0

for r in results:
    key = str(r['rid'])
    # Only log the immediate next round (step 0) and always overwrite with latest prediction.
    # Steps 1-4 are shown in the UI for context but not scored — scoring only the
    # prediction made when the round is actually about to start prevents stale predictions
    # from being counted as wins/losses.
    is_next_round = (r['rid'] == last_id + 1)
    already_scored = key in log and log[key].get('actual') is not None
    if not is_next_round or already_scored:
        continue
    p_odd   = r['by_oe'].get('ODD', 0)
    pred_oe = 'ODD' if p_odd > 0.5 else 'EVEN'
    conf_val = max(p_odd, 1.0 - p_odd)
    _bwr, _bn = _lookup_band_wr(conf_val)
    si = r['sinfo']
    _agree = float(si.get('agreement', 0))
    # Suppress bet during consecutive loss streaks: >=3 losses → skip unless confidence very high
    _streak_skip = _cur_loss_streak >= 3 and conf_val < 0.62
    _bet_ok = (
        not _streak_skip and
        _bwr is not None and
        _bn >= 50 and
        _bwr >= 0.57 and
        _phase_action in ('USE', 'USE_STRONG') and
        _agree >= 0.55
    )
    log[key] = {
        'round_id':     key,
        'predicted':    r['pred'],
        'pred_oe':      pred_oe,
        'p_oe_odd':     round(p_odd, 4),
        'confidence':   round(float(conf_val), 4),
        'band_wr':      _bwr,
        'band_n':       _bn,
        'agreement':    round(_agree, 3),
        'phase_action': _phase_action,
        'bet':          _bet_ok,
        'actual':       None,
        'phase':        _phase_info['label'],
        'signals': {
            'run':      si['run'],
            'momentum': si['momentum'],
            'seq':      si['seq'],
            'disc_seq': si['disc_seq'],
            'persist':  si['persist'],
            'recent':   si['recent'],
            'period':   si['period'],
            'reversal': si['reversal'],
            'breakout': si['breakout'],
            'ml':       si['ml'],
            'lstm':     si['lstm'],
        },
        'effective_signals': si.get('effective_signals', {}),
    }
# ── Gap-fill: ensure recent rounds have predictions in pred_log ─────────────────
_GF_WIN  = 50
_gf_from = max(0, n_rows - _GF_WIN)
_gf_miss = [
    i for i in range(_gf_from, n_rows - 1)
    if str(int(df.iloc[i]['id'])) not in log
       or not log[str(int(df.iloc[i]['id']))].get('pred_oe')
]
if _gf_miss:
    print(f'  predict.py gap-fill: {len(_gf_miss)} missing rounds…', flush=True)
    for _gi in _gf_miss:
        _rid_g   = str(int(df.iloc[_gi]['id']))
        _df_g    = df.iloc[:_gi + 1]
        _rn_g    = encode_runs(_df_g['odd'].tolist())
        _oe_g    = full_oe_arr[:_gi + 1]
        _pa_g    = full_pat_arr[:_gi + 1]
        _out_g   = predict_round(_df_g, _rn_g, _oe_g, _pa_g, lstm_src=_df_g)
        if _out_g is None:
            continue
        _pred_g, _conf_g, _by_rc_g, _by_oe_g, _si_g = _out_g
        _p_odd_g   = float(_by_oe_g.get('ODD', 0.5))
        _cv_g      = max(_p_odd_g, 1.0 - _p_odd_g)
        _pred_oe_g = 'ODD' if _p_odd_g > 0.5 else 'EVEN'
        _actual_g  = str(df.iloc[_gi]['pattern']) if not pd.isna(df.iloc[_gi]['pattern']) else None
        log[_rid_g] = {
            'round_id':   _rid_g,
            'predicted':  _pred_g,
            'pred_oe':    _pred_oe_g,
            'p_oe_odd':   round(_p_odd_g, 4),
            'confidence': round(float(_cv_g), 4),
            'actual':     _actual_g,
            'bet':        False,
            'signals':    {k: _si_g.get(k) for k in ('run','momentum','seq','disc_seq','persist','recent','period','reversal','breakout','ml','lstm')},
        }
    print(f'  predict.py gap-fill done — {len(_gf_miss)} rounds added.', flush=True)

_log_tmp = LOG_PATH + '.tmp'
with open(_log_tmp, 'w') as f:
    json.dump(log, f, indent=2)
os.replace(_log_tmp, LOG_PATH)

# ── Confidence band analysis ──────────────────────────────────────────────────
BANDS = [
    (0.50, 0.53, '50-53%'),
    (0.53, 0.56, '53-56%'),
    (0.56, 0.60, '56-60%'),
    (0.60, 0.65, '60-65%'),
    (0.65, 1.01, '65%+  '),
]
MIN_BAND_SAMPLES = 20   # need at least this many to trust a band's win rate
USE_WIN_RATE     = 0.53 # band must exceed this win rate to be flagged USE

def confidence_band_stats(pred_log):
    """
    For every scored entry that has p_oe_odd stored, group by ODD/EVEN confidence
    and compute win rate per band. Returns list of (label, win_rate, n, usable).
    """
    stats = {lbl: [0, 0] for *_, lbl in BANDS}   # wins, total
    for entry in pred_log.values():
        if not entry.get('actual') or entry.get('p_oe_odd') is None:
            continue
        p_odd  = entry['p_oe_odd']
        conf   = max(p_odd, 1.0 - p_odd)
        actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
        won    = (entry['pred_oe'] == actual_oe)
        for lo, hi, lbl in BANDS:
            if lo <= conf < hi:
                stats[lbl][1] += 1
                if won:
                    stats[lbl][0] += 1
                break
    result = []
    for lo, hi, lbl in BANDS:
        wins, total = stats[lbl]
        wr = wins / total if total > 0 else None
        usable = total >= MIN_BAND_SAMPLES and wr is not None and wr > USE_WIN_RATE
        result.append((lbl, wr, total, usable))
    return result

def current_band_label(p_odd):
    conf = max(p_odd, 1.0 - p_odd)
    for lo, hi, lbl in BANDS:
        if lo <= conf < hi:
            return lbl
    return BANDS[-1][2]


# ── Gap analysis ───────────────────────────────────────────────────────────────
wwww_ids = df.loc[df['pattern'] == 'WWWW', 'id'].tolist()
rrrr_ids = df.loc[df['pattern'] == 'RRRR', 'id'].tolist()

def gap_next_prob(ids, since):
    """
    P(pattern occurs on the very next round | gap so far = since).
    Uses survival analysis on historical gaps:
      P = count(gap == since+1) / count(gap >= since+1)
    Smoothed with a ±2 window when sample count is low.
    """
    if len(ids) < 2:
        return None, 0
    gaps = [ids[i+1] - ids[i] for i in range(len(ids) - 1)]
    k = since + 1   # the gap length we'd be at after next round
    n_at_k  = sum(1 for g in gaps if g == k)
    n_geq_k = sum(1 for g in gaps if g >= k)
    if n_geq_k < 5:
        # widen window ±3 for sparse regions
        n_at_k  = sum(1 for g in gaps if k-3 <= g <= k+3)
        n_geq_k = sum(1 for g in gaps if g >= k-3)
    if n_geq_k == 0:
        return None, 0
    return n_at_k / n_geq_k, n_geq_k

since_ww = last_id - wwww_ids[-1] if wwww_ids else 0
since_rr = last_id - rrrr_ids[-1] if rrrr_ids else 0

# ── EVEN / ODD pattern predictions (query DB directly by stored oe column) ──────
def _gap_prob_seq(idx_list, since):
    if len(idx_list) < 2 or since is None:
        return None, 0
    gaps = [idx_list[i+1] - idx_list[i] for i in range(len(idx_list) - 1)]
    k = since + 1
    n_at  = sum(1 for g in gaps if g == k)
    n_geq = sum(1 for g in gaps if g >= k)
    if n_geq < 5:
        n_at  = sum(1 for g in gaps if k-2 <= g <= k+2)
        n_geq = sum(1 for g in gaps if g >= k-2)
    return (round(n_at / n_geq, 4) if n_geq > 0 else None), n_geq

def _prob_within_n(idx_list, since, n=5):
    """P(pattern occurs within next n positions | already gone 'since' without it)."""
    if len(idx_list) < 2 or since is None:
        return None, 0
    gaps = [idx_list[i+1] - idx_list[i] for i in range(len(idx_list) - 1)]
    n_geq   = sum(1 for g in gaps if g > since)
    n_within = sum(1 for g in gaps if since < g <= since + n)
    if n_geq < 5:
        return None, n_geq
    return round(n_within / n_geq, 4), n_geq

def _run_length_analysis(seq):
    """
    Run-length analysis on a binary label sequence (e.g. ['WWWR','RRRW',...]).
    For each run length k, computes P(switch) — the probability the streak ends
    after exactly k consecutive occurrences of the same label.
    Returns current run state and switch probability at the current length.
    """
    if len(seq) < 4:
        return {}

    # Run-length encode
    runs = []
    cur_lbl, cnt = seq[0], 1
    for v in seq[1:]:
        if v == cur_lbl:
            cnt += 1
        else:
            runs.append((cur_lbl, cnt))
            cur_lbl, cnt = v, 1
    runs.append((cur_lbl, cnt))

    labels = sorted(set(seq))
    if len(labels) != 2:
        return {}

    # For each (label, run_length), count: continued vs switched
    from collections import defaultdict
    sc = {lbl: defaultdict(lambda: {'sw': 0, 'co': 0}) for lbl in labels}
    for j, (lbl, length) in enumerate(runs[:-1]):   # all fully-completed runs
        for k in range(1, length):
            sc[lbl][k]['co'] += 1                    # positions 1..L-1 → continued
        sc[lbl][length]['sw'] += 1                   # at length L → switched

    def _p_switch(lbl, k):
        # Walk down from k to find a length with enough samples
        for kk in range(k, 0, -1):
            d = sc[lbl][kk]
            total = d['sw'] + d['co']
            if total >= 5:
                return round(d['sw'] / total, 4), total
        return None, 0

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    # Per-label run-length stats
    run_lens = {lbl: [] for lbl in labels}
    for lbl, length in runs:
        run_lens[lbl].append(length)

    out = {}
    for lbl in labels:
        lens = run_lens[lbl]
        out[f'avg_run_{lbl}']  = _avg(lens)
        out[f'max_run_{lbl}']  = max(lens) if lens else None
        out[f'n_runs_{lbl}']   = len(lens)
        for k in [1, 2, 3, 4, 5]:
            p, n = _p_switch(lbl, k)
            if p is not None:
                out[f'p_sw_{lbl}_{k}'] = p
                out[f'n_sw_{lbl}_{k}'] = n

    # Current (ongoing) run
    cur_lbl, cur_len = runs[-1]
    other_lbl = [l for l in labels if l != cur_lbl][0]
    p_sw, n_sw = _p_switch(cur_lbl, cur_len)
    out.update({
        'cur_run_lbl':  cur_lbl,
        'cur_run_len':  cur_len,
        'switch_prob':  p_sw,
        'switch_n':     n_sw,
        'next_likely':  other_lbl if (p_sw is not None and p_sw >= 0.5) else cur_lbl,
    })
    return out


def _transition_stats(special, cur):
    """
    Transition probability analysis between two alternating labels.
    Used for WWWW<->RRRR (EVEN) where gaps > 1 make this meaningful.
    """
    if len(special) < 2:
        return {}
    labels = sorted(set(lbl for _, lbl in special))
    if len(labels) != 2:
        return {}
    a, b = labels[0], labels[1]

    counts = {(a,a):0, (a,b):0, (b,a):0, (b,b):0}
    gaps   = {(a,a):[], (a,b):[], (b,a):[], (b,b):[]}
    for j in range(len(special) - 1):
        fi, fl = special[j]
        ti, tl = special[j+1]
        counts[(fl,tl)] += 1
        gaps[(fl,tl)].append(ti - fi)

    def _p(fr, to):
        tot = counts[(fr,a)] + counts[(fr,b)]
        return round(counts[(fr,to)] / tot, 4) if tot > 0 else None
    def _avg(lst):
        return round(sum(lst)/len(lst), 1) if lst else None

    last_pos, last_lbl = special[-1]
    since = cur - last_pos
    next_lbl = b if last_lbl == a else a
    trans_p  = _p(last_lbl, next_lbl)
    avg_g    = _avg(gaps[(last_lbl, next_lbl)])

    return {
        f'p_{a}_to_{b}':       _p(a, b),
        f'p_{b}_to_{a}':       _p(b, a),
        f'p_{a}_to_{a}':       _p(a, a),
        f'p_{b}_to_{b}':       _p(b, b),
        f'avg_gap_{a}_to_{b}': _avg(gaps[(a,b)]),
        f'avg_gap_{b}_to_{a}': _avg(gaps[(b,a)]),
        'last_special':        last_lbl,
        'since_last':          since,
        'next_likely':         next_lbl,
        'trans_prob':          trans_p,
        'avg_trans_gap':       avg_g,
    }

def _even_pattern_analysis(db_path, seq_len=4, min_matches=3):
    """
    Query DB for oe='EVEN' rows, sequence-match patterns, and analyse
    WWWW<->RRRR transitions (Markov-chain style).
    """
    conn = _sqlite3.connect(db_path, timeout=5)
    conn.execute('PRAGMA journal_mode=WAL')
    rows = conn.execute(
        "SELECT pattern FROM rounds WHERE oe='EVEN' AND pattern IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    pats = [r[0] for r in rows if r[0]]
    n = len(pats)
    if n < seq_len + 2:
        return None

    recent = pats[-seq_len:]
    ww = rr = other = 0
    for i in range(n - seq_len - 1):
        if pats[i:i + seq_len] == recent:
            nxt = pats[i + seq_len]
            if nxt == 'WWWW':   ww += 1
            elif nxt == 'RRRR': rr += 1
            else:               other += 1

    total = ww + rr + other
    if total < min_matches and seq_len > 2:
        return _even_pattern_analysis(db_path, seq_len - 1, min_matches)

    ww_idx = [i for i, p in enumerate(pats) if p == 'WWWW']
    rr_idx = [i for i, p in enumerate(pats) if p == 'RRRR']
    cur    = n - 1
    since_ww = cur - ww_idx[-1] if ww_idx else None
    since_rr = cur - rr_idx[-1] if rr_idx else None
    ww_gap_p, ww_gap_n = _gap_prob_seq(ww_idx, since_ww)
    rr_gap_p, rr_gap_n = _gap_prob_seq(rr_idx, since_rr)

    # Transition analysis: WWWW <-> RRRR sequence
    special = [(i, p) for i, p in enumerate(pats) if p in ('RRRR', 'WWWW')]
    tr = _transition_stats(special, cur)

    wwww_all = round(len(ww_idx) / n, 4) if n > 0 else None
    rrrr_all = round(len(rr_idx) / n, 4) if n > 0 else None
    other_all = round(1 - (wwww_all or 0) - (rrrr_all or 0), 4) if n > 0 else None

    ww_n5_p, ww_n5_n = _prob_within_n(ww_idx, since_ww, n=5)
    rr_n5_p, rr_n5_n = _prob_within_n(rr_idx, since_rr, n=5)

    return {
        'recent_even_seq':  recent,
        'seq_len':          seq_len,
        'n_matches':        total,
        'wwww_seq_pct':     round(ww / total, 4) if total > 0 else None,
        'rrrr_seq_pct':     round(rr / total, 4) if total > 0 else None,
        'other_seq_pct':    round(other / total, 4) if total > 0 else None,
        'wwww_all_pct':     wwww_all,
        'rrrr_all_pct':     rrrr_all,
        'other_all_pct':    other_all,
        'since_wwww_even':  since_ww,
        'since_rrrr_even':  since_rr,
        'wwww_gap_pct':     ww_gap_p,
        'wwww_gap_n':       ww_gap_n,
        'rrrr_gap_pct':     rr_gap_p,
        'rrrr_gap_n':       rr_gap_n,
        'wwww_next5_pct':   ww_n5_p,
        'wwww_next5_n':     ww_n5_n,
        'rrrr_next5_pct':   rr_n5_p,
        'rrrr_next5_n':     rr_n5_n,
        'total_even':       n,
        **tr,
    }

def _odd_pattern_analysis(db_path, seq_len=4, min_matches=3):
    """
    Query DB for oe='ODD' rows, sequence-match r_count (1=WWWR, 3=RRRW),
    and analyse WWWR<->RRRW transitions (Markov-chain style).
    """
    conn = _sqlite3.connect(db_path, timeout=5)
    conn.execute('PRAGMA journal_mode=WAL')
    rows = conn.execute(
        "SELECT disc1, disc2, disc3, disc4 FROM rounds WHERE oe='ODD' AND disc1 IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    rc = [sum(1 for d in r if d == 'R') for r in rows]
    n  = len(rc)
    if n < seq_len + 2:
        return None

    recent = rc[-seq_len:]
    r1 = r3 = 0
    for i in range(n - seq_len - 1):
        if rc[i:i + seq_len] == recent:
            nxt = rc[i + seq_len]
            if nxt == 1:   r1 += 1
            elif nxt == 3: r3 += 1

    total = r1 + r3
    if total < min_matches and seq_len > 2:
        return _odd_pattern_analysis(db_path, seq_len - 1, min_matches)

    r1_idx = [i for i, v in enumerate(rc) if v == 1]
    r3_idx = [i for i, v in enumerate(rc) if v == 3]
    cur    = n - 1
    since_r1 = cur - r1_idx[-1] if r1_idx else None
    since_r3 = cur - r3_idx[-1] if r3_idx else None
    r1_gap_p, r1_gap_n = _gap_prob_seq(r1_idx, since_r1)
    r3_gap_p, r3_gap_n = _gap_prob_seq(r3_idx, since_r3)

    # Run-length analysis: how many consecutive WWWR/RRRW before a switch?
    odd_seq = ['WWWR' if v == 1 else 'RRRW' for v in rc if v in (1, 3)]
    rl = _run_length_analysis(odd_seq)

    return {
        'recent_odd_seq': recent,
        'seq_len':        seq_len,
        'n_matches':      total,
        'r1_seq_pct':     round(r1 / total, 4) if total > 0 else None,
        'r3_seq_pct':     round(r3 / total, 4) if total > 0 else None,
        'since_r1_odd':   since_r1,
        'since_r3_odd':   since_r3,
        'r1_gap_pct':     r1_gap_p,
        'r1_gap_n':       r1_gap_n,
        'r3_gap_pct':     r3_gap_p,
        'r3_gap_n':       r3_gap_n,
        'total_odd':      n,
        **rl,
    }

_even_analysis = _even_pattern_analysis(DB_PATH)
_odd_analysis  = _odd_pattern_analysis(DB_PATH)

# Patch pattern labels/pcts into the current round's log entry (analyses now available)
_next_key = str(last_id + 1)
if _next_key in log:
    _ea = _even_analysis or {}
    _oa = _odd_analysis  or {}
    _ww = _ea.get('wwww_next5_pct') or 0
    _rr = _ea.get('rrrr_next5_pct') or 0
    log[_next_key]['pat_even_lbl'] = 'WWWW' if _ww >= _rr else 'RRRR'
    log[_next_key]['pat_even_pct'] = round(max(_ww, _rr), 4) or None
    log[_next_key]['pat_odd_lbl']  = _oa.get('next_likely')
    _sw = _oa.get('switch_prob')
    _cl = _oa.get('cur_run_lbl')
    _nl = _oa.get('next_likely')
    log[_next_key]['pat_odd_pct']  = (
        None if _sw is None
        else round(_sw, 4) if _nl != _cl
        else round(1 - _sw, 4)
    )
    _log_tmp2 = LOG_PATH + '.tmp'
    with open(_log_tmp2, 'w') as f:
        json.dump(log, f, indent=2)
    os.replace(_log_tmp2, LOG_PATH)

# ── Output ─────────────────────────────────────────────────────────────────────
print(f"\n  Last record : {last_id}  {last['pattern']}  {last['odd']}")
print(f"  Total rows  : {n_rows}")
lstm_acc_str = f"  LSTM={acc_lstm:.1%}" if acc_lstm else ""
print(f"\n  Model accuracy (held-out 20%)  ODD/EVEN={acc_oe:.1%}  R-count={acc_rc:.1%}  Pattern={acc_pat:.1%}{lstm_acc_str}")
if n_scored > 0:
    print(f"  Real-time ({n_scored} scored)  ODD/EVEN={real_oe:.1%}  R-count={real_rc:.1%}  Pattern={real_pat:.1%}")

# ── Signal accuracy report ─────────────────────────────────────────────────────
if _sig_acc:
    _inv = [f"{s}({a:.0%}↑)" if a >= 0.50 else f"{s}({a:.0%}↓INVERTED)"
            for s, (a, n) in sorted(_sig_acc.items()) if n >= 20]
    print(f"  Sig accuracy: {' | '.join(_inv)}")

# ── Calibration report ─────────────────────────────────────────────────────────
_ece, _ece_bins = expected_calibration_error(log)
if _ece is not None:
    _ece_parts = []
    for mc, wr, nb in _ece_bins:
        if nb > 0 and wr is not None:
            gap = wr - mc
            _ece_parts.append(f"{mc:.0%}→{wr:.0%}({nb})")
    _cal_str = f"  calibrated on {_cal_n} samples" if _cal_n >= 40 else f"  calibration needs ≥40 samples ({_cal_n} so far)"
    print(f"  Calibration : ECE={_ece:.3f}  bins=[{' | '.join(_ece_parts)}]{_cal_str}")

if results:
    r0  = results[0]
    rc0 = r0['by_rc']
    oe0 = r0['by_oe']
    si  = r0['sinfo']
    lstm_str    = f"  lstm={si['lstm']:.2f}" if si['lstm'] is not None else ""
    persist_str = (f"  persist={si['persist']:.2f}(run={si['persist_len']})"
                   if si['persist'] is not None else "")
    mode_str = ""
    if si.get('streak', 0) >= 3:
        suppressed = [k for k, v in (_streak_mults or {}).items() if v < 0.5]
        sup_str  = ("/".join(suppressed)) if suppressed else "none"
        mode_str = f" [STREAK-{si['streak']} suppressed={sup_str}]"
    elif si.get('adaptive'):
        mode_str = " [adaptive]"
    anti_str = f"  anti={si['anti_streak']:.2f}" if si.get('anti_streak') is not None else ""
    corr_str = f"  dir={si['dir_corr']:+.3f}  rt={si['rt_acc']:+.3f}"
    recent_str = (f"  recent={si['recent']:.2f}(w={si['w_recent']:.2f},{si['recent_n']}x{si['recent_len']})"
                  if si['recent'] is not None else "")
    period_str   = (f"  period={si['period']:.2f}(w={si['w_period']:.2f},lag={si['period_lag']})"
                    if si['period'] is not None else "")
    reversal_str  = (f"  reversal={si['reversal']:.2f}(w={si['w_reversal']:.2f})"
                     if si['reversal'] is not None else "")
    breakout_str  = (f"  breakout={si['breakout']:.2f}(w={si['w_breakout']:.2f})"
                     if si['breakout'] is not None else "")
    oe_phase_str  = (f"  oe_phase={si['oe_phase']:.2f}(w={si.get('w_oe_phase',0):.2f},{si.get('oe_phase_label','?')},n={si.get('oe_phase_n',0)})"
                     if si.get('oe_phase') is not None else "")
    print(f"\n  Signals  run={si['run']:.2f}(w={si['w_run']:.2f})  "
          f"mom={si['momentum']:.2f}(w={si['w_momentum']:.2f})  "
          f"seq={si['seq']:.2f}(w={si['w_seq']:.2f},{si['seq_len']}x{si['seq_n']})  "
          f"dseq={si['disc_seq']:.2f}(w={si['w_disc_seq']:.2f},{si['disc_len']}x{si['disc_n']})"
          f"{recent_str}{period_str}{reversal_str}{breakout_str}{persist_str}{lstm_str}{oe_phase_str}  ml={si['ml']:.2f}  "
          f"(base={si['p_base']:.2f}  deficit={si['deficit']:+.2f}  cur_run={si['run_len']}  "
          f"agree={si['agreement']:.0%}  amp={si['amplifier']:.2f}{anti_str}{corr_str}{mode_str})")

    ww_gap_p, ww_n = gap_next_prob(wwww_ids, since_ww)
    rr_gap_p, rr_n = gap_next_prob(rrrr_ids, since_rr)
    ww_pct = f"{ww_gap_p:.1%} (gap={since_ww}, n={ww_n})" if ww_gap_p is not None else "n/a"
    rr_pct = f"{rr_gap_p:.1%} (gap={since_rr}, n={rr_n})" if rr_gap_p is not None else "n/a"

    if _next_key in log:
        log[_next_key]['wwww_pct'] = round(float(ww_gap_p), 4) if ww_gap_p is not None else None
        log[_next_key]['wwww_gap'] = int(since_ww)
        log[_next_key]['rrrr_pct'] = round(float(rr_gap_p), 4) if rr_gap_p is not None else None
        log[_next_key]['rrrr_gap'] = int(since_rr)
        log[_next_key]['w3r1_pct'] = round(float(rc0.get(1, 0)), 4)
        log[_next_key]['r3w1_pct'] = round(float(rc0.get(3, 0)), 4)
        _log_tmp3 = LOG_PATH + '.tmp'
        with open(_log_tmp3, 'w') as f:
            json.dump(log, f, indent=2)
        os.replace(_log_tmp3, LOG_PATH)

    p_odd_cur   = oe0.get('ODD', 0.5)
    # If signals were systematically wrong every round during streak, invert the prediction
    band_lbl    = current_band_label(p_odd_cur)
    band_stats  = confidence_band_stats(log)
    band_lookup = {lbl: (wr, n, u) for lbl, wr, n, u in band_stats}
    wr, bn, usable = band_lookup.get(band_lbl, (None, 0, False))

    if bn < MIN_BAND_SAMPLES:
        band_str  = f"only {bn} rounds in this band — insufficient history"
        use_flag  = "SKIP"
    elif wr < 0.50:
        band_str  = f"{wr:.1%} win rate over {bn} rounds"
        use_flag  = "AVOID"   # actively losing at this confidence level
    elif wr < USE_WIN_RATE:
        band_str  = f"{wr:.1%} win rate over {bn} rounds"
        use_flag  = "SKIP"    # no edge yet
    else:
        band_str  = f"{wr:.1%} win rate over {bn} rounds"
        use_flag  = "USE"     # reliable edge

    print(f"\n  Next round  : {last_id + 1}")
    print(f"  WWWW : {ww_pct}")
    print(f"  RRRR : {rr_pct}")
    print(f"  3W1R : {rc0.get(1,0):.1%}")
    print(f"  3R1W : {rc0.get(3,0):.1%}")
    print(f"  ODD  : {p_odd_cur:.1%}")
    print(f"  EVEN : {oe0.get('EVEN',0):.1%}")
    _invert_str = "  ⚠ GLOBAL INVERT ACTIVE (model was anti-predictive)" if _global_invert else ""
    print(f"\n  Confidence  : {max(p_odd_cur, 1-p_odd_cur):.1%}  band={band_lbl}  {band_str}{_invert_str}")
    print(f"  Action      : {use_flag}")
    _ph_wr_str = f"{_ph_wr:.1%} over {_ph_n} rounds" if _ph_wr is not None else f"only {_ph_n} rounds"
    _kelly_str = f"  half-Kelly={_kelly['bet_pct']}%  p_ruin={_kelly['p_ruin']:.1%}  break-even={_kelly['breakeven']}r" if _kelly else ""
    print(f"\n  Phase       : {_phase_info['label']}")
    print(f"  Phase WR    : {_ph_wr_str}")
    print(f"  Phase action: {_phase_action}{_kelly_str}")

    # ── Write latest prediction snapshot for the web dashboard ─────────────────
    import datetime
    suppressed = [k for k, v in (_streak_mults or {}).items() if v < 0.5]
    inverted   = sorted(_streak_invert_sigs | _short_streak_inverts)
    suppressed_short = sorted(_short_streak_suppress)
    snapshot = {
        'timestamp':        datetime.datetime.now().isoformat(timespec='seconds'),
        'last_id':          int(last_id),
        'last_pattern':     str(last['pattern']),
        'last_discs':       list(last['pattern']),   # ['R','W','R','W']
        'last_oe':          str(last['odd']),
        'predicted_pattern': str(r0['pred']),
        'predicted_discs':  list(r0['pred']),
        'total_rows':  int(n_rows),
        'accuracy': {
            'oe':      round(float(acc_oe), 4),
            'r_count': round(float(acc_rc), 4),
            'pattern': round(float(acc_pat), 4),
            'lstm':    round(float(acc_lstm), 4) if acc_lstm else None,
        },
        'realtime': {
            'n':       int(n_scored),
            'oe':      round(float(real_oe), 4) if n_scored > 0 else None,
            'r_count': round(float(real_rc), 4) if n_scored > 0 else None,
            'pattern': round(float(real_pat), 4) if n_scored > 0 else None,
        },
        'next_round': int(last_id + 1),
        'prediction': {
            'odd_pct':  round(float(p_odd_cur), 4),
            'even_pct': round(float(oe0.get('EVEN', 0)), 4),
            'r3w1':     round(float(rc0.get(1, 0)), 4),
            'w3r1':     round(float(rc0.get(3, 0)), 4),
            'wwww': {'pct': round(float(ww_gap_p), 4) if ww_gap_p is not None else None,
                     'gap': int(since_ww), 'n': int(ww_n)},
            'rrrr': {'pct': round(float(rr_gap_p), 4) if rr_gap_p is not None else None,
                     'gap': int(since_rr), 'n': int(rr_n)},
        },
        'confidence': {
            'value':   round(float(max(p_odd_cur, 1 - p_odd_cur)), 4),
            'band':    band_lbl.strip(),
            'band_wr': round(float(wr), 4) if wr is not None else None,
            'band_n':  int(bn),
            'action':  use_flag,
            'bet':     bool(
                _cur_loss_streak < 3 and
                wr is not None and bn >= 50 and wr >= 0.57 and
                _phase_action in ('USE', 'USE_STRONG') and
                si['agreement'] >= 0.55
            ),
            'threshold': 0.57,
        },
        'signals': {
            'run':      {'value': round(float(si['run']), 3),      'weight': round(float(si['w_run']), 3),      'run_len': int(si['run_len'])},
            'momentum': {'value': round(float(si['momentum']), 3), 'weight': round(float(si['w_momentum']), 3)},
            'seq':      {'value': round(float(si['seq']), 3),      'weight': round(float(si['w_seq']), 3),      'len': int(si['seq_len']), 'n': int(si['seq_n'])},
            'disc_seq': {'value': round(float(si['disc_seq']), 3), 'weight': round(float(si['w_disc_seq']), 3), 'len': int(si['disc_len']), 'n': int(si['disc_n'])},
            'persist':  {'value': round(float(si['persist']), 3) if si['persist'] is not None else None,
                         'weight': round(float(si['w_persist']), 3), 'run_len': int(si['persist_len'])},
            'recent':   {'value': round(float(si['recent']), 3) if si['recent'] is not None else None,
                         'weight': round(float(si['w_recent']), 3), 'n': int(si['recent_n']), 'len': int(si['recent_len'])},
            'period':   {'value': round(float(si['period']), 3) if si['period'] is not None else None,
                         'weight': round(float(si['w_period']), 3), 'lag': int(si['period_lag'])},
            'reversal': {'value': round(float(si['reversal']), 3) if si['reversal'] is not None else None,
                         'weight': round(float(si['w_reversal']), 3)},
            'breakout': {'value': round(float(si['breakout']), 3) if si['breakout'] is not None else None,
                         'weight': round(float(si['w_breakout']), 3)},
            'ml':       {'value': round(float(si['ml']), 3)},
            'lstm':     {'value': round(float(si['lstm']), 3) if si['lstm'] is not None else None},
            'oe_phase': {'value': round(float(si['oe_phase']), 3) if si.get('oe_phase') is not None else None,
                         'label': si.get('oe_phase_label'),
                         'n':     si.get('oe_phase_n', 0)},
        },
        'mode':               'streak' if si.get('streak', 0) >= 3 else ('adaptive' if si.get('adaptive') else 'equal'),
        'streak':             int(si.get('streak', 0)),
        'suppressed_signals':       suppressed,
        'suppressed_short_signals': suppressed_short,
        'suppressed_autopsy':       sorted(_streak_autopsy),
        'inverted_signals':         inverted,
        'loss_streak':              int(_cur_loss_streak),
        'fallback_active':          _fallback_p_odd is not None,
        'global_invert':            bool(_global_invert),
        'agreement':          round(float(si['agreement']), 3),
        'amplifier':          round(float(si['amplifier']), 3),
        'deficit':            round(float(si['deficit']), 3),
        'phase_analysis': {
            'label':      _phase_info['label'],
            'components': {
                'run_type':   _phase_info['run_type'],
                'run_val':    _phase_info['run_val'],
                'run_len':    _phase_info['run_len'],
                'balance':    _phase_info['balance'],
                'odd_frac':   _phase_info['odd_frac'],
                'volatility': _phase_info['volatility'],
                'switches':   _phase_info['switches'],
            },
            'win_rate':  _ph_wr,
            'n_rounds':  _ph_n,
            'action':    _phase_action,
            'kelly':     _kelly,
        },
        'calibration': {
            'ece':      _ece,
            'n_samples': _cal_n,
            'applied':  _cal_n >= 40,
            'bins': [
                {'mean_conf': round(mc, 3), 'actual_wr': round(wr, 3) if wr is not None else None, 'n': nb}
                for mc, wr, nb in _ece_bins
            ] if _ece_bins else [],
        },
        'even_analysis': _even_analysis,
        'odd_analysis':  _odd_analysis,
    }
    snap_path = os.path.join(_DATA_DIR, 'latest_prediction.json')
    with open(snap_path, 'w') as _f:
        json.dump(snapshot, _f, indent=2)

