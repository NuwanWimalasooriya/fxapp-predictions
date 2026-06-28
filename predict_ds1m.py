"""
DS1M Prediction Engine — XGBoost + Domain Rules

Reads and writes to ds1m.db / pred_log_ds1m.json.

Domain rules (6):
  1. Long block (run >= 4) → lock current direction
  2. Just switched from long block (cur_run >= 2, prev_run >= 4) → lock new direction
  3. Confirmed alternating (3+ unbroken OE/EO) → stay on start direction
  4. No confirmed pattern → dominant direction in last 8 rounds blended with XGBoost
  5. Hard streak breaker (5+ consecutive losses → flip)
  6. After very long block, analyse pre-block context to choose mirror/return direction

XGBoost features: cur_run, cur_val, prev_len, prev_val, mirror_ratio,
  b1..b5 len/val (last 5 blocks), avg_block_len, alt_rate_8, dom_o_8,
  is_long_block, is_after_long, is_alt_confirmed, alt_start_odd.
"""
import sys, os, json, datetime
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

_BASE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE, 'data')

DB_SRC_PATH    = os.path.join(_DATA_DIR, 'ds1m.db')
DB_NEW_PATH    = os.path.join(_DATA_DIR, 'ds1m.db')
CSV_NEW_PATH   = os.path.join(_DATA_DIR, '1min-discs.csv')
LOG_NEW_PATH   = os.path.join(_DATA_DIR, 'pred_log_ds1m.json')
SNAP_NEW_PATH  = os.path.join(_DATA_DIR, 'latest_prediction_ds1m.json')
XGB_MODEL_PATH = os.path.join(_DATA_DIR, 'xgb_model_ds1m.pkl')
XGB_META_PATH  = os.path.join(_DATA_DIR, 'xgb_meta_ds1m.json')

RETRAIN_EVERY = 50

# ── Load historical data ────────────────────────────────────────────────────────
import sqlite3 as _sqlite3
_conn = _sqlite3.connect(DB_SRC_PATH, timeout=10)
_conn.execute('PRAGMA journal_mode=WAL')
df = pd.read_sql(
    'SELECT id,disc1,disc2,disc3,disc4 FROM rounds ORDER BY id',
    _conn, dtype={'id': int}
)
_conn.close()

df['pattern'] = df['disc1'] + df['disc2'] + df['disc3'] + df['disc4']
df['r_count'] = df['pattern'].str.count('R')
df['odd']     = df['r_count'].apply(lambda r: 'ODD' if r % 2 else 'EVEN')

last_id = int(df.iloc[-1]['id'])
n_rows  = len(df)
oe_list = df['odd'].tolist()


# ── Block sequence builder ──────────────────────────────────────────────────────
def _build_blocks(oe_arr):
    blocks, i = [], 0
    while i < len(oe_arr):
        val, length = oe_arr[i], 1
        while i + length < len(oe_arr) and oe_arr[i + length] == val:
            length += 1
        blocks.append((val, length, i))
        i += length
    return blocks


# ── Mirror-tile detection ────────────────────────────────────────────────────────
_FLIP = {'ODD': 'EVEN', 'EVEN': 'ODD'}

def detect_mirror_tile(oe_arr, idx, tile_sizes=(6, 8, 12)):
    """
    Detect round-level mirror/complement tile patterns.
    e.g. OOEEOE → EEOOEO → OOEEOE...  (tile size 6, values flip each cycle)

    Works at round level (not block level) so tile-boundary block merges don't hide the pattern.
    Requires 2+ complete tile cycles to confirm.

    Returns a dict or None.
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)

    for ts in tile_sizes:
        if n < ts * 2:
            continue

        tile_a = arr[n - ts : n]           # most recent tile
        tile_b = arr[n - ts*2 : n - ts]   # tile before it

        # Is tile_a the complement of tile_b?
        if not all(_FLIP[tile_b[i]] == tile_a[i] for i in range(ts)):
            continue

        # Count how many consecutive complementary tile-pairs exist going back
        cycles = 1
        pos    = n - ts * 2
        ref    = tile_b  # tile_b is the "base" reference for this pair
        while pos >= ts:
            chunk = arr[pos - ts : pos]
            # Each step further back should be the complement of ref
            if all(_FLIP[ref[i]] == chunk[i] for i in range(ts)):
                cycles += 1
                ref = chunk
                pos -= ts
            else:
                break

        if cycles < 1:   # we already confirmed 2 tiles (1 pair), that's enough
            continue

        # Next tile = complement of tile_a
        next_tile = [_FLIP[v] for v in tile_a]

        # Position within next tile: idx+1 is offset 0 of next tile
        # (we're at the end of tile_a, so next round is next_tile[0])
        next_pos  = 0
        pred_val  = next_tile[next_pos]

        # Confidence based on how many pairs confirmed + tile size
        conf = round(0.70 + min(0.15, (cycles - 1) * 0.05), 3)

        pname = f'mirror_t{ts}'
        return {
            'pred_val':    pred_val,
            'conf':        conf,
            'cycles':      cycles + 1,   # total tiles seen
            'tile_size':   ts,
            'pattern_name': pname,
            'next_tile':   next_tile,
            'tile_pos':    0,
            'is_transition': True,
        }

    return None


def detect_mirror_tile_inprogress(oe_arr, idx, tile_sizes=(6, 8, 12)):
    """
    When we're INSIDE a tile (not at the boundary), predict based on the established
    mirror tile pattern by finding our position within the expected current tile.
    Returns a dict or None.
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)

    for ts in tile_sizes:
        if n < ts * 2:
            continue

        # Try to find the start of the current tile
        for tile_start_offset in range(1, ts):
            # current tile starts at n - tile_start_offset
            cur_tile_start = n - tile_start_offset
            if cur_tile_start < ts:
                continue

            partial       = arr[cur_tile_start : n]           # partial current tile
            prev_tile_end = cur_tile_start
            prev_tile     = arr[prev_tile_end - ts : prev_tile_end]

            if len(prev_tile) < ts:
                continue

            # Expected current tile = complement of prev tile
            expected_tile = [_FLIP[v] for v in prev_tile]

            # Does partial match expected_tile[:len(partial)]?
            if not all(expected_tile[i] == partial[i] for i in range(len(partial))):
                continue

            # Check how many prior pairs exist (need at least 1 = 2 tiles total)
            pair_count = 1
            pp_end = prev_tile_end - ts
            pp_ref = prev_tile
            while pp_end >= ts:
                pp_tile = arr[pp_end - ts : pp_end]
                if all(_FLIP[pp_ref[i]] == pp_tile[i] for i in range(ts)):
                    pair_count += 1
                    pp_ref = pp_tile
                    pp_end -= ts
                else:
                    break

            if pair_count < 1:
                continue

            # We are at position tile_start_offset - 1 within the current tile
            tile_pos  = tile_start_offset - 1
            next_pos  = tile_pos + 1

            if next_pos < ts:
                pred_val = expected_tile[next_pos]
                pname    = f'mirror_t{ts}_in'
                # Lower confidence for first pair (pair_count==1), builds with more pairs
                base_conf = 0.65 if pair_count == 1 else 0.68
                conf      = round(base_conf + min(0.12, 0.04 * tile_pos), 3)
                return {
                    'pred_val':    pred_val,
                    'conf':        conf,
                    'cycles':      pair_count + 1,
                    'tile_size':   ts,
                    'pattern_name': pname,
                    'next_tile':   expected_tile,
                    'tile_pos':    tile_pos,
                    'is_transition': False,
                }

    return None


# ── Known-sequence tile detection ───────────────────────────────────────────────
# Tiles where the mirror-pair rule doesn't apply but round-by-round sequence is known.
# EEOEOOEO / OOEOEEOE: alternating-base with 2 blocks extended by 1 ("trend-shift" tiles).
_KNOWN_TILES_OE = [
    ['EVEN','EVEN','ODD','EVEN','ODD','ODD','EVEN','ODD'],   # EEOEOOEO
    ['ODD','ODD','EVEN','ODD','EVEN','EVEN','ODD','EVEN'],   # OOEOEEOE (complement)
]

def detect_known_tile(oe_arr, idx):
    """
    Detect progress inside a known 8-round tile sequence (EEOEOOEO / OOEOEEOE).
    Requires: ≥4 rounds confirmed at a natural block boundary.
    Confidence scales with how many rounds are confirmed (0.63–0.68).
    Returns dict or None.
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)

    for tile_seq in _KNOWN_TILES_OE:
        ts = len(tile_seq)
        # Try each possible start position (2..ts-2 rounds already confirmed)
        for matched in range(4, ts):
            if n < matched + 1:
                break
            start = n - matched
            partial = arr[start : n]
            if partial != tile_seq[:matched]:
                continue
            # Require a boundary: round before start must differ from tile_seq[0]
            if start > 0 and arr[start - 1] == tile_seq[0]:
                continue
            next_pos = matched
            if next_pos >= ts:
                break
            pred_val = tile_seq[next_pos]
            conf     = round(0.63 + min(0.05, (matched - 4) * 0.025), 3)
            pname    = 'known_t8_' + ('EEOE' if tile_seq[0]=='EVEN' else 'OOEO')
            return {
                'pred_val':    pred_val,
                'conf':        conf,
                'tile_size':   ts,
                'pattern_name': pname,
                'tile_pos':    matched,
                'tile_seq':    tile_seq,
            }
    return None


# ── Symmetric tile precursor detection ─────────────────────────────────────────
def detect_sym_tile_inprogress(oe_arr, idx):
    """
    Detect if we're currently 1 or 2 rounds into the SECOND HALF of a 3+3 symmetric tile.
    Pattern: XXX (3 same) then YY... (switch just happened, cur_run 1-2).
    The 3 rounds immediately before the switch must all be the same value.
    Returns dict predicting continuation of current direction, or None.
    """
    arr     = list(oe_arr[:idx + 1])
    n       = len(arr)
    cur_val = arr[idx]

    # Count current run
    cur_run = 0
    for i in range(idx, max(-1, idx - 10), -1):
        if arr[i] == cur_val: cur_run += 1
        else: break

    if cur_run > 2 or n < cur_run + 3:
        return None

    prev_val = _FLIP[cur_val]
    # The 3 rounds immediately before the current run
    prev3_start = n - cur_run - 3
    if prev3_start < 0:
        return None
    prev3 = arr[prev3_start : n - cur_run]

    if prev3 != [prev_val, prev_val, prev_val]:
        return None

    # Confirmed: XXX + (Y or YY) — we're in the second half of EEEOOO / OOOEEE
    conf = 0.65 if cur_run == 1 else 0.68
    return {
        'pred_val':    cur_val,
        'conf':        conf,
        'pattern_name': 'sym_in',
        'cur_run':     cur_run,
    }


def detect_sym_tile(oe_arr, idx, tile_size=6):
    """
    Detect N+N symmetric tile (EEEOOO / OOOEEE) just completed at idx.
    After such a tile, the next tile is the semi-symmetric OOEEOE/EEOOEO ~70% of time.
    Predicts the first round of that expected tile.

    Based on empirical data (12 confirmed cases):
    - EEEOOO (ends ODD)  → OOEEOE starts OO (ODD same)  — 75% accuracy → conf 0.70
    - OOOEEE (ends EVEN) → OOEEOE starts OO (ODD flip)  — 63% accuracy → conf 0.63
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)
    if n < tile_size:
        return None
    half  = tile_size // 2
    tile  = arr[n - tile_size : n]
    first = tile[:half]
    second = tile[half:]
    # Both halves must be homogeneous and different from each other
    if len(set(first)) != 1 or len(set(second)) != 1 or first[0] == second[0]:
        return None
    first_val  = first[0]
    second_val = second[0]
    # EEEOOO: first=EVEN, second=ODD → predict ODD (continue), conf 0.70
    # OOOEEE: first=ODD,  second=EVEN → predict ODD (flip),    conf 0.63
    if first_val == 'EVEN':   # EEEOOO
        pred_val = second_val   # ODD
        conf     = 0.70
    else:                      # OOOEEE
        pred_val = _FLIP[second_val]  # ODD (flip of EVEN)
        conf     = 0.63
    return {
        'pred_val':     pred_val,
        'conf':         conf,
        'tile_size':    tile_size,
        'pattern_name': f'sym_t{tile_size}',
        'first_val':    first_val,
        'second_val':   second_val,
    }


# ── Post-symmetric tile in-progress detection ───────────────────────────────────
# After EEEOOO or OOOEEE (symmetric tile), the next tile is OOEEOE ~63-75% of the time.
# detect_sym_tile handles the first round at the boundary.
# This handles rounds 2-6 WITHIN that OOEEOE tile.
_POST_SYM_TILE = ['ODD', 'ODD', 'EVEN', 'EVEN', 'ODD', 'EVEN']  # OOEEOE

def detect_post_sym_tile_inprogress(oe_arr, idx, tile_size=6):
    """
    Detect if we're 1–(tile_size-1) rounds into the OOEEOE tile that follows a
    symmetric tile (EEEOOO / OOOEEE).

    Returns dict with pred_val and conf, or None.
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)
    if n < tile_size + 1:
        return None

    for pos in range(1, tile_size):   # pos = rounds already seen in OOEEOE tile
        if n < tile_size + pos:
            continue

        sym_tile = arr[n - tile_size - pos : n - pos]
        partial  = arr[n - pos : n]

        if len(sym_tile) != tile_size or len(partial) != pos:
            continue

        # Symmetric tile check: two homogeneous halves that differ
        half = tile_size // 2
        fh, sh = sym_tile[:half], sym_tile[half:]
        if len(set(fh)) != 1 or len(set(sh)) != 1 or fh[0] == sh[0]:
            continue

        # Check partial matches OOEEOE[:pos]
        if partial != _POST_SYM_TILE[:pos]:
            continue

        if pos >= tile_size:
            continue

        pred_val = _POST_SYM_TILE[pos]
        conf     = round(0.65 + min(0.10, (pos - 1) * 0.025), 3)

        return {
            'pred_val':     pred_val,
            'conf':         conf,
            'tile_size':    tile_size,
            'pattern_name': f'post_sym_t{tile_size}',
            'tile_pos':     pos,
        }

    return None


# ── Uniform-block alternating pattern ──────────────────────────────────────────
def detect_uniform_block(all_blocks, min_len=2, max_len=6):
    """
    Detect a repeating uniform-length alternating block pattern, e.g. EEE|OOO|EEE|OOO.

    Requires the last 2 COMPLETE blocks to have the same length L and alternate in value.
    The current (partial) block must be the expected continuation (opposite of prev).

    Returns a dict or None:
      phase='in_progress'  → inside the expected block → predict CONTINUE (same as cur_val)
      phase='at_boundary'  → cur_run just hit L        → predict SWITCH  (opposite of cur_val)
    """
    if len(all_blocks) < 3:
        return None

    cur_val, cur_run = all_blocks[-1][0], all_blocks[-1][1]
    p_val,   p_len   = all_blocks[-2][0], all_blocks[-2][1]
    pp_val,  pp_len  = all_blocks[-3][0], all_blocks[-3][1]

    # Last 2 complete blocks must share the same length and alternate
    if p_len != pp_len or p_val == pp_val:
        return None

    block_len = p_len
    if not (min_len <= block_len <= max_len):
        return None

    # Current block must be the expected next value (opposite of prev)
    if cur_val != _FLIP[p_val]:
        return None

    if cur_run < block_len:
        conf = round(0.67 + min(0.08, (cur_run - 1) * 0.03), 3)
        return {
            'pred_val':     cur_val,           # continue current block
            'conf':         conf,
            'block_len':    block_len,
            'pattern_name': f'uniform_blk_L{block_len}',
            'phase':        'in_progress',
        }
    elif cur_run == block_len:
        return {
            'pred_val':     _FLIP[cur_val],    # switch to opposite
            'conf':         0.70,
            'block_len':    block_len,
            'pattern_name': f'uniform_blk_L{block_len}',
            'phase':        'at_boundary',
        }
    # cur_run > block_len — block is longer than pattern, no prediction
    return None


# ── 3-by-3 block pattern ────────────────────────────────────────────────────────
def detect_3x3_pattern(oe_arr, idx, block_len=3):
    """
    Detect a 3-by-3 alternating block pattern: EEEOOOEEE / OOOEEEOO...

    Fires when the previous block is EXACTLY block_len of the opposite value and
    the current run is 1..block_len.  Requires 2+ completed same-length pairs for
    the 'has_confirmed_pair' flag (higher confidence); works with just 1 pair.

    Returns:
      phase='in_progress'  (cur_run < block_len): predict CONTINUE current value
      phase='at_boundary'  (cur_run == block_len): predict SWITCH to opposite
    """
    arr = list(oe_arr[:idx + 1])
    n   = len(arr)
    cur_val = arr[idx]

    cur_run = 0
    for i in range(idx, max(-1, idx - 10), -1):
        if arr[i] == cur_val: cur_run += 1
        else: break

    if cur_run > block_len or n < cur_run + block_len:
        return None

    prev_val    = _FLIP[cur_val]
    prev_start  = n - cur_run - block_len
    if prev_start < 0:
        return None

    prev_block = arr[prev_start : n - cur_run]
    if prev_block != [prev_val] * block_len:
        return None

    # Ensure the previous block is EXACTLY block_len (not a longer run)
    if prev_start > 0 and arr[prev_start - 1] == prev_val:
        return None

    # Check whether the block before prev is also block_len of cur_val (confirmed pair)
    has_confirmed_pair = False
    pre_start = prev_start - block_len
    if pre_start >= 0:
        pre_block = arr[pre_start : prev_start]
        if pre_block == [cur_val] * block_len:
            if pre_start == 0 or arr[pre_start - 1] != cur_val:
                has_confirmed_pair = True

    if cur_run < block_len:
        base = 0.67 if has_confirmed_pair else 0.63
        conf = round(base + min(0.05, (cur_run - 1) * 0.025), 3)
        return {
            'pred_val':          cur_val,
            'conf':              conf,
            'pattern_name':      '3x3_in',
            'phase':             'in_progress',
            'cur_run':           cur_run,
            'has_confirmed_pair': has_confirmed_pair,
        }
    else:  # cur_run == block_len → at boundary
        conf = 0.68 if has_confirmed_pair else 0.65
        return {
            'pred_val':          _FLIP[cur_val],
            'conf':              conf,
            'pattern_name':      '3x3_boundary',
            'phase':             'at_boundary',
            'cur_run':           cur_run,
            'has_confirmed_pair': has_confirmed_pair,
        }


# ── Cycle detection ─────────────────────────────────────────────────────────────
def detect_cycle(blocks_with_meta):
    """
    Detect a repeating block-level cycle (period 2, 4, or 6).
    blocks_with_meta: list from _build_blocks() — last entry is the current (ongoing) block.
    Requires 2+ full completed cycles to confirm.

    Returns a dict with cycle info, or None.
    """
    if len(blocks_with_meta) < 5:
        return None

    # Completed blocks (exclude current ongoing)
    bl = [(v, l) for v, l, *_ in blocks_with_meta[:-1]]
    current_val = blocks_with_meta[-1][0]
    current_run = blocks_with_meta[-1][1]

    if len(bl) < 4:
        return None

    best = None

    # Only even periods are valid for ODD/EVEN block cycles (odd periods repeat
    # the same value at the cycle boundary, which _build_blocks would merge).
    for period in (2, 4, 6):
        if len(bl) < period * 2:
            continue

        template = bl[-period:]

        # Cycle boundary must alternate: last block val ≠ first block val
        if template[-1][0] == template[0][0]:
            continue

        # Count consecutive backward-matching cycles
        cycles_found = 1
        for cyc in range(1, len(bl) // period):
            end   = len(bl) - period * cyc
            start = end - period
            if start < 0:
                break
            chunk = bl[start:end]
            ok = all(
                tv == cv and (
                    tl == cl or (tl >= 2 and cl >= 2 and abs(tl - cl) <= 1)
                )
                for (tv, tl), (cv, cl) in zip(template, chunk)
            )
            if ok:
                cycles_found += 1
            else:
                break

        # Require 2+ confirmed cycles for all patterns — 1 cycle is too noisy
        lens = [l for _, l in template]
        if cycles_found < 2:
            continue

        # Position of current (ongoing) block in the cycle
        cur_pos          = len(bl) % period
        exp_val, exp_len = template[cur_pos]
        next_pos         = (cur_pos + 1) % period
        next_val, _      = template[next_pos]

        if current_val != exp_val:
            continue  # current block breaks the expected pattern

        overrun       = max(0, current_run - exp_len)
        is_transition = current_run >= exp_len
        pred_val      = next_val if is_transition else current_val

        if set(lens) == {2}:
            pname = 'double_alt'
        elif set(lens) <= {1, 2}:
            continue  # mixed_alt: too noisy, disabled
        elif set(lens) == {1}:
            continue  # single_alt: too noisy, disabled
        else:
            pname = f'cycle_p{period}'

        # 1-cycle detections get lower confidence (SKIP range); 2+ get BET range
        base_conf = 0.65 if cycles_found == 1 else 0.70
        conf = round(
            base_conf
            + min(0.15, max(0, cycles_found - 2) * 0.05)
            - (0.03 if is_transition else 0)
            - (0.04 * overrun),
            3
        )
        conf = max(0.52, conf)

        if best is None or cycles_found > best['cycles'] or (
            cycles_found == best['cycles'] and period < best['period']
        ):
            best = {
                'pred_val':      pred_val,
                'conf':          conf,
                'cycles':        cycles_found,
                'period':        period,
                'pattern_name':  pname,
                'cur_pos':       cur_pos,
                'exp_len':       exp_len,
                'current_run':   current_run,
                'is_transition': is_transition,
                'overrun':       overrun,
            }

    return best


# ── Feature engineering ─────────────────────────────────────────────────────────
def make_features(oe_arr, idx):
    if idx < 10:
        return None
    arr = list(oe_arr[:idx + 1])

    # Current run
    cur_val = arr[idx]
    cur_run = 0
    for i in range(idx, max(-1, idx - 60), -1):
        if arr[i] == cur_val:
            cur_run += 1
        else:
            break

    # Block sequence up to idx
    blocks = _build_blocks(arr)

    # Last 5 blocks (most recent first)
    b_lens = [0] * 5
    b_vals = [0] * 5
    for k, (bval, blen, _) in enumerate(reversed(blocks[-5:])):
        b_lens[k] = blen
        b_vals[k] = 1 if bval == 'ODD' else 0

    prev_len = b_lens[1] if len(blocks) >= 2 else 0
    mirror_ratio = min(cur_run / prev_len, 5.0) if prev_len > 0 else 0.0

    avg_block_len = float(np.mean([b[1] for b in blocks[-10:]])) if blocks else 0.0

    # Alternation rate last 8 rounds
    recent8 = arr[max(0, idx - 7): idx + 1]
    alt_cnt = sum(1 for i in range(1, len(recent8)) if recent8[i] != recent8[i - 1])
    alt_rate_8 = alt_cnt / (len(recent8) - 1) if len(recent8) > 1 else 0.0
    dom_o_8 = sum(1 for x in recent8 if x == 'ODD') / len(recent8) if recent8 else 0.5

    # Alternating run detection
    alt_run = 1
    for i in range(idx - 1, max(-1, idx - 30), -1):
        if arr[i] != arr[i + 1]:
            alt_run += 1
        else:
            break
    is_alt = 1 if alt_run >= 3 else 0
    alt_start_odd = (1 if arr[idx - alt_run + 1] == 'ODD' else 0) if is_alt else 0

    return [
        cur_run,                              # 0
        1 if cur_val == 'ODD' else 0,         # 1
        prev_len,                             # 2
        b_vals[1],                            # 3  prev_val
        mirror_ratio,                         # 4
        b_lens[0], b_lens[1], b_lens[2], b_lens[3], b_lens[4],   # 5-9
        b_vals[0], b_vals[1], b_vals[2], b_vals[3], b_vals[4],   # 10-14
        avg_block_len,                        # 15
        alt_rate_8,                           # 16
        dom_o_8,                              # 17
        1 if cur_run >= 4 else 0,             # 18  is_long_block
        1 if (prev_len >= 4 and cur_run < 4) else 0,  # 19  is_after_long
        is_alt,                               # 20
        alt_start_odd,                        # 21
    ]


# ── XGBoost model ───────────────────────────────────────────────────────────────
def _need_retrain():
    if not os.path.exists(XGB_MODEL_PATH) or not os.path.exists(XGB_META_PATH):
        return True
    with open(XGB_META_PATH) as f:
        return n_rows - json.load(f).get('trained_on', 0) >= RETRAIN_EVERY

def _train():
    X, y = [], []
    for i in range(10, len(oe_list) - 1):
        feats = make_features(oe_list, i)
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if oe_list[i + 1] == 'ODD' else 0)

    if len(X) < 100:
        return None

    X  = np.array(X, dtype=np.float32)
    y  = np.array(y, dtype=np.float32)
    rw = np.exp(np.linspace(-2, 0, len(X))).astype(np.float32)

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0, device='cpu'
        )
        model.fit(X, y, sample_weight=rw)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X, y, sample_weight=rw)

    import joblib
    joblib.dump(model, XGB_MODEL_PATH)
    with open(XGB_META_PATH, 'w') as f:
        json.dump({'trained_on': n_rows, 'n_samples': len(X)}, f)
    print(f'  XGBoost (DS1M) trained on {len(X)} samples', flush=True)
    return model


xgb_model = None
if _need_retrain():
    print('  Training XGBoost (DS1M)...', flush=True)
    xgb_model = _train()
else:
    if os.path.exists(XGB_MODEL_PATH):
        import joblib
        xgb_model = joblib.load(XGB_MODEL_PATH)


# ── Init new DB ─────────────────────────────────────────────────────────────────
def _init_new_db():
    conn = _sqlite3.connect(DB_NEW_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS rounds (
        id INTEGER PRIMARY KEY,
        disc1 TEXT, disc2 TEXT, disc3 TEXT, disc4 TEXT,
        pattern TEXT, oe TEXT,
        result TEXT DEFAULT "Not Predicted",
        pred_oe TEXT DEFAULT "", confidence REAL DEFAULT 0, bet INTEGER DEFAULT 0
    )''')
    conn.commit()

    src = _sqlite3.connect(DB_SRC_PATH, timeout=10)
    src.execute('PRAGMA journal_mode=WAL')
    rows = src.execute(
        'SELECT id, disc1, disc2, disc3, disc4, '
        'disc1||disc2||disc3||disc4, '
        "(CASE WHEN (LENGTH(disc1||disc2||disc3||disc4) - LENGTH(REPLACE(disc1||disc2||disc3||disc4,'R',''))) % 2 = 1 THEN 'ODD' ELSE 'EVEN' END) "
        'FROM rounds ORDER BY id'
    ).fetchall()
    src.close()

    existing = {r[0] for r in conn.execute('SELECT id FROM rounds').fetchall()}
    new_rows = [r for r in rows if r[0] not in existing]
    if new_rows:
        conn.executemany(
            'INSERT OR IGNORE INTO rounds(id,disc1,disc2,disc3,disc4,pattern,oe) VALUES(?,?,?,?,?,?,?)',
            new_rows
        )
        conn.commit()
    conn.close()

_init_new_db()


# ── Loss streak counter ─────────────────────────────────────────────────────────
def _loss_streak(log):
    scored = sorted(
        [(int(k), v) for k, v in log.items() if v.get('actual') and v.get('pred_oe')],
        key=lambda x: x[0], reverse=True
    )
    streak = 0
    for _, entry in scored:
        actual_oe = 'ODD' if entry['actual'].count('R') % 2 else 'EVEN'
        if entry['pred_oe'] != actual_oe:
            streak += 1
        else:
            break
    return streak


def _get_sticky_oe(log):
    """Return (pred_oe, confidence) to reuse if last scored round was WIN, else (None, None)."""
    scored = sorted(
        [(int(k), v) for k, v in log.items() if v.get('actual') and v.get('pred_oe')],
        key=lambda x: x[0], reverse=True
    )
    if not scored:
        return None, None
    _, last = scored[0]
    actual_oe = 'ODD' if last['actual'].count('R') % 2 else 'EVEN'
    if last['pred_oe'] == actual_oe:
        return last['pred_oe'], last.get('confidence', 0.65)
    return None, None


# ── Load log ────────────────────────────────────────────────────────────────────
log_new = {}
if os.path.exists(LOG_NEW_PATH):
    try:
        with open(LOG_NEW_PATH, encoding='utf-8-sig') as f:
            log_new = json.load(f)
    except Exception:
        pass

# Update actual results in log from main DB
_actual_map = df.set_index(df['id'].astype(str))['pattern'].to_dict()
_log_changed = False
for _rid, _entry in log_new.items():
    if _entry.get('actual') is None and _rid in _actual_map:
        _entry['actual'] = _actual_map[_rid]
        _log_changed = True

_cur_loss_streak = _loss_streak(log_new)


# ── Main predict function ───────────────────────────────────────────────────────
def predict_ds1m(oe_arr, idx):
    """
    Predict ODD/EVEN for position idx+1.
    Returns (pred_oe, p_odd, signal_info).
    """
    if idx < 10:
        return None, 0.5, {}

    arr     = list(oe_arr[:idx + 1])
    cur_val = arr[idx]

    # Current run length
    cur_run = 0
    for i in range(idx, max(-1, idx - 60), -1):
        if arr[i] == cur_val:
            cur_run += 1
        else:
            break

    # Previous block run length
    prev_val = arr[idx - cur_run] if idx >= cur_run else None
    prev_run = 0
    if prev_val and prev_val != cur_val:
        for i in range(idx - cur_run, max(-1, idx - cur_run - 60), -1):
            if arr[i] == prev_val:
                prev_run += 1
            else:
                break

    # Alternating run detection (unbroken OEOEO... ending at idx)
    alt_run = 1
    for i in range(idx - 1, max(-1, idx - 30), -1):
        if arr[i] != arr[i + 1]:
            alt_run += 1
        else:
            break
    alt_confirmed  = alt_run >= 3
    alt_start_val  = arr[idx - alt_run + 1] if alt_confirmed else None

    # Dominant direction last 8 rounds
    recent8 = arr[max(0, idx - 7): idx + 1]
    o8 = sum(1 for x in recent8 if x == 'ODD')
    e8 = len(recent8) - o8

    # XGBoost signal
    p_xgb = 0.5
    if xgb_model is not None:
        feats = make_features(oe_arr, idx)
        if feats is not None:
            p_xgb = float(xgb_model.predict_proba(
                np.array(feats, dtype=np.float32).reshape(1, -1)
            )[0][1])

    # Pre-block context analysis (for Rule 6)
    pre_context_dominant = None
    if cur_run < 4 and prev_run >= 6:
        blocks = _build_blocks(arr[:idx - cur_run])
        if len(blocks) >= 3:
            # Context before the long block
            ctx_blocks = blocks[:-1]  # exclude the long block
            ctx_vals   = []
            for bval, blen, _ in ctx_blocks[-3:]:
                ctx_vals.extend([bval] * blen)
            ctx_vals = ctx_vals[-8:]
            o_ctx = ctx_vals.count('ODD')
            e_ctx = len(ctx_vals) - o_ctx
            alt_ctx = sum(1 for i in range(1, len(ctx_vals)) if ctx_vals[i] != ctx_vals[i - 1])
            alt_rate_ctx = alt_ctx / (len(ctx_vals) - 1) if len(ctx_vals) > 1 else 0
            if alt_rate_ctx >= 0.5:
                pre_context_dominant = 'ODD' if o_ctx >= e_ctx else 'EVEN'
            else:
                pre_context_dominant = 'ODD' if o_ctx > e_ctx else 'EVEN'

    # ── DOMAIN RULES ─────────────────────────────────────────────────────────────

    applied_rule = 'xgb'
    p_odd = p_xgb

    # Cycle detection (used by Rule 7)
    all_blocks = _build_blocks(arr)
    cycle_info = detect_cycle(all_blocks) if len(all_blocks) >= 5 else None

    # Uniform-block alternating detection (EEE|OOO|EEE|OOO etc.) — used by Rule UB
    uniform_block_info = detect_uniform_block(all_blocks) if len(all_blocks) >= 3 else None

    # Mirror-tile detection (used by Rules 8 & 9)
    mirror_info    = detect_mirror_tile(oe_arr, idx)
    mirror_in_info = detect_mirror_tile_inprogress(oe_arr, idx) if mirror_info is None else None

    # 3-by-3 block pattern (EEEOOOEEE / OOOEEEOO...)
    pattern_3x3 = detect_3x3_pattern(oe_arr, idx)

    # Symmetric tile in-progress — inside OOO phase of EEEOOO
    sym_in_info = detect_sym_tile_inprogress(oe_arr, idx)

    # Symmetric tile precursor (used by Rule 10)
    sym_info = detect_sym_tile(oe_arr, idx) if (
        mirror_info is None and mirror_in_info is None
        and sym_in_info is None and pattern_3x3 is None
    ) else None

    # Post-symmetric tile in-progress — rounds 2-6 of OOEEOE after EEEOOO/OOOEEE (Rule 10b)
    post_sym_info = detect_post_sym_tile_inprogress(oe_arr, idx) if (
        mirror_info is None and mirror_in_info is None and sym_info is None
    ) else None

    # Known sequence tile — EEOEOOEO / OOEOEEOE (used by Rule 11)
    known_tile_info = detect_known_tile(oe_arr, idx) if mirror_in_info is None else None

    # Rule 8: Mirror-tile boundary — completed tile, next tile is complement
    # Always highest priority: boundary is a precise, point-in-time prediction
    if mirror_info is not None:
        pred_odd_m = (mirror_info['pred_val'] == 'ODD')
        m_conf     = mirror_info['conf']
        p_odd      = m_conf if pred_odd_m else (1.0 - m_conf)
        applied_rule = f'rule8_{mirror_info["pattern_name"]}'

    # Rule 9 (strong): 2+ confirmed tile pairs → tile structure overrides long-run
    elif mirror_in_info is not None and mirror_in_info.get('cycles', 0) >= 3:
        pred_odd_mi = (mirror_in_info['pred_val'] == 'ODD')
        mi_conf     = mirror_in_info['conf']
        p_odd       = mi_conf if pred_odd_mi else (1.0 - mi_conf)
        applied_rule = f'rule9_{mirror_in_info["pattern_name"]}'

    # Rule UB: Uniform-block alternating pattern (EEE|OOO|EEE|OOO...)
    # Must fire before Rule 1b (cur_run==3) so boundary switch takes priority over
    # naive "continue streak" logic.
    elif uniform_block_info is not None and mirror_info is None and mirror_in_info is None:
        pred_odd_ub = (uniform_block_info['pred_val'] == 'ODD')
        ub_conf     = uniform_block_info['conf']
        p_odd       = ub_conf if pred_odd_ub else (1.0 - ub_conf)
        applied_rule = f'rule_ub_{uniform_block_info["pattern_name"]}_{uniform_block_info["phase"]}'

    # Rule 3x3: Three-by-three block pattern (EEEOOOEEE / OOOEEEOO...)
    # Fires when:
    #   in_progress → cur_run 1-2 inside a new block whose predecessor was exactly L3
    #   at_boundary → cur_run == 3 and predecessor was exactly L3 → switch direction
    # Placed before Rule 1b/Rule 6 so the 3x3 structural signal beats generic trend rules.
    elif pattern_3x3 is not None:
        pred_odd_3x3 = (pattern_3x3['pred_val'] == 'ODD')
        conf_3x3     = pattern_3x3['conf']
        p_odd        = conf_3x3 if pred_odd_3x3 else (1.0 - conf_3x3)
        applied_rule = f'rule_3x3_{pattern_3x3["phase"]}'

    # Rule 1: Inside long block (run >= 4)
    # Data (15985 rounds): run=4→52% stay, run=5→52% stay, run=6+→53% switch.
    # Old code used 0.82 confidence which is far too aggressive for a ~52% signal.
    # Blend with XGBoost only when it agrees — disagreement means XGB is fighting a clear
    # structural signal, so trust the run instead of letting XGB drag the score below 0.5.
    elif cur_run >= 4:
        if cur_run <= 5:
            block_p = 0.60 if cur_val == 'ODD' else 0.40  # weak stay lean
        else:
            block_p = 0.44 if cur_val == 'ODD' else 0.56  # run>=6: lean switch
        rule_is_odd = block_p > 0.5
        if (p_xgb > 0.5) == rule_is_odd:
            p_odd = 0.55 * block_p + 0.45 * p_xgb
        else:
            p_odd = block_p  # XGBoost disagrees with streak — trust the run
        applied_rule = 'rule1_long_block'

    # Rule 1b: Medium streak (run == 3) — after 3 consecutive same, continue the streak.
    # Same XGBoost guard: an obvious 3-streak is not overridden by a contradicting model.
    elif cur_run == 3:
        stay_p = 0.60 if cur_val == 'ODD' else 0.40  # lean stay: continue current streak
        rule_is_odd = stay_p > 0.5
        if (p_xgb > 0.5) == rule_is_odd:
            p_odd = 0.55 * stay_p + 0.45 * p_xgb
        else:
            p_odd = stay_p  # XGBoost disagrees with 3-streak — trust the streak
        applied_rule = 'rule1b_stay'

    # Rule 6: Trend persist — after a dominant run (3+), stay on previous direction
    # until the counter reaches 3 consecutive. 1 or 2 counter-rounds are not a reversal.
    elif cur_run <= 2 and prev_run >= 3:
        if prev_run >= 5:
            eoe_conf = 0.65
        elif prev_run == 4:
            eoe_conf = 0.62
        else:  # prev_run == 3
            eoe_conf = 0.58
        p_odd = eoe_conf if prev_val == 'ODD' else (1.0 - eoe_conf)
        applied_rule = 'rule6_trend_persist'

    # Rule Maj20: When cur_run ≤ 2 (trend is changing), anchor prediction to the dominant
    # trend of the last 20 rounds. Replaces rule3_alternating and rule6b_short_lean —
    # alternating-flip and short-streak lean both cause losses during trend changes because
    # they react to the local direction rather than the sustained majority direction.
    elif cur_run <= 2:
        _r20   = arr[max(0, idx - 19): idx + 1]
        _odd20 = _r20.count('ODD')
        _n20   = len(_r20)
        _p_maj = _odd20 / _n20
        p_odd  = 0.60 * _p_maj + 0.40 * p_xgb
        applied_rule = 'rule_maj20'

    # Rule 7: Cyclic block pattern (period 2/4/6)
    elif cycle_info is not None:
        pred_odd_cy = (cycle_info['pred_val'] == 'ODD')
        cy_conf     = cycle_info['conf']
        p_odd       = cy_conf if pred_odd_cy else (1.0 - cy_conf)
        applied_rule = f'rule7_{cycle_info["pattern_name"]}'

    # Rule 9 (weak / single-pair) intentionally disabled — fires 37% of rounds at 50-56% loss

    # Rule 10: Symmetric 3+3 tile completed — predict start of SEMI tile
    elif sym_info is not None:
        pred_odd_s = (sym_info['pred_val'] == 'ODD')
        s_conf     = sym_info['conf']
        p_odd      = s_conf if pred_odd_s else (1.0 - s_conf)
        applied_rule = f'rule10_{sym_info["pattern_name"]}'

    # Rule 10b: Inside OOEEOE tile that follows a symmetric tile (rounds 2-6)
    elif post_sym_info is not None:
        pred_odd_ps = (post_sym_info['pred_val'] == 'ODD')
        ps_conf     = post_sym_info['conf']
        p_odd       = ps_conf if pred_odd_ps else (1.0 - ps_conf)
        applied_rule = f'rule10b_{post_sym_info["pattern_name"]}'

    # Rule 11: Known 8-round tile in progress (EEOEOOEO / OOEOEEOE)
    elif known_tile_info is not None:
        pred_odd_kt = (known_tile_info['pred_val'] == 'ODD')
        kt_conf     = known_tile_info['conf']
        p_odd       = kt_conf if pred_odd_kt else (1.0 - kt_conf)
        applied_rule = f'rule11_{known_tile_info["pattern_name"]}'

    # Rule 4: No confirmed pattern → pure XGBoost.
    # Analysis of 15985 rounds: 8-round window majority gives exactly 50.0% accuracy.
    # Removed dom_p component — it adds noise, not signal.
    # else: pure XGBoost (rule = 'xgb')

    p_odd = max(0.05, min(0.95, p_odd))

    # When no structural pattern was detected (xgb or rule4 fallback) and XGBoost is
    # very uncertain (within ±5% of 50%), apply a slight EVEN bias.
    # Basis: EVEN is 50.1% of all rounds; in uncertain conditions defaulting EVEN
    # avoids the over-aggressive ODD signals that caused 30% ODD precision.
    if applied_rule in ('xgb', 'rule4_dom_blend') and abs(p_odd - 0.5) < 0.05:
        p_odd = 0.47

    # Rule 5: Hard streak breaker (5+ consecutive losses → flip).
    # Only applies when no structural rule detected a clear pattern — flipping Rule 1/1b/6
    # signals inverts correct predictions and extends the loss streak instead of breaking it.
    _structural_prefixes = ('rule1_', 'rule1b_', 'rule6_', 'rule7_',
                            'rule8_', 'rule10_', 'rule10b_', 'rule11_', 'rule_ub_', 'rule_3x3')
    if _cur_loss_streak >= 5 and not any(applied_rule.startswith(p) for p in _structural_prefixes):
        p_odd = 1.0 - p_odd
        applied_rule += '+streak_flip'

    p_odd = max(0.05, min(0.95, p_odd))

    pred_oe = 'ODD' if p_odd > 0.5 else 'EVEN'
    conf    = max(p_odd, 1.0 - p_odd)

    sinfo = {
        'xgb':             round(p_xgb, 3),
        'cur_run':         cur_run,
        'cur_val':         cur_val,
        'prev_run':        prev_run,
        'alt_run':         alt_run,
        'alt_confirmed':   alt_confirmed,
        'dom_o8':          round(o8 / len(recent8), 3) if recent8 else 0.5,
        'rule':            applied_rule,
        'loss_streak':     _cur_loss_streak,
        'cycle_detected':  cycle_info is not None,
        'cycle_pattern':   cycle_info['pattern_name'] if cycle_info else None,
        'cycle_period':    cycle_info['period']       if cycle_info else None,
        'cycle_cycles':    cycle_info['cycles']       if cycle_info else None,
        'cycle_pred_val':  cycle_info['pred_val']     if cycle_info else None,
        'cycle_conf':      cycle_info['conf']         if cycle_info else None,
        'cycle_transition': cycle_info['is_transition'] if cycle_info else None,
        'cycle_exp_len':   cycle_info['exp_len']      if cycle_info else None,
        'mirror_info':        (mirror_info or mirror_in_info),
        'sym_info':           sym_info,
        'post_sym_info':      post_sym_info,
        'known_tile_info':    known_tile_info,
        'uniform_block_info': uniform_block_info,
        'pattern_3x3':        pattern_3x3,
    }
    return pred_oe, p_odd, sinfo


# ── Gap-fill: ensure every round in the last 300 has a prediction ───────────────
# Range includes n_rows-1 (last record) so the most recent round always gets pred_oe.
# Uses context up to _bi-1 (predict_ds1m(oe_list, _bi-1)) to predict round _bi — no future leak.
_GF_WINDOW = 300
_gf_start  = max(11, n_rows - _GF_WINDOW)
_missing   = [
    i for i in range(_gf_start, n_rows)
    if str(int(df.iloc[i]['id'])) not in log_new
    or not log_new[str(int(df.iloc[i]['id']))].get('pred_oe')
]
if _missing:
    print(f'  Filling {len(_missing)} missing predictions (last {_GF_WINDOW} rounds)…', flush=True)
    for _bi in _missing:
        _rid_b = str(int(df.iloc[_bi]['id']))
        _pe_b, _pp_b, _si_b = predict_ds1m(oe_list, _bi - 1)
        if _pe_b is None:
            continue
        _conf_b = max(_pp_b, 1.0 - _pp_b)
        log_new[_rid_b] = {
            'round_id':   _rid_b,
            'pred_oe':    _pe_b,
            'p_oe_odd':   round(_pp_b, 4),
            'confidence': round(_conf_b, 4),
            'actual':     _actual_map.get(_rid_b),
            'signals':    _si_b,
        }
    _log_changed = True
    print(f'  Gap fill done — {len(_missing)} rounds added.', flush=True)

if _log_changed:
    _tmp = LOG_NEW_PATH + '.tmp'
    with open(_tmp, 'w') as f:
        json.dump(log_new, f, indent=2)
    os.replace(_tmp, LOG_NEW_PATH)
    _log_changed = False

# Recompute loss streak after backfill
_cur_loss_streak = _loss_streak(log_new)

# ── Run prediction ──────────────────────────────────────────────────────────────
sig_idx  = len(oe_list) - 1

_sticky_oe, _sticky_conf = _get_sticky_oe(log_new)
if _sticky_oe is not None:
    pred_oe = _sticky_oe
    p_odd   = _sticky_conf if _sticky_oe == 'ODD' else (1.0 - _sticky_conf)
    sinfo   = {
        'rule': 'sticky_win', 'loss_streak': _cur_loss_streak,
        'xgb': 0.5, 'cur_run': 0, 'cur_val': '', 'prev_run': 0,
        'alt_run': 0, 'alt_confirmed': False, 'dom_o8': 0.5,
        'cycle_detected': False, 'cycle_pattern': None, 'cycle_period': None,
        'cycle_cycles': None, 'cycle_pred_val': None, 'cycle_conf': None,
        'cycle_transition': None, 'cycle_exp_len': None,
        'mirror_info': None, 'sym_info': None, 'post_sym_info': None,
        'known_tile_info': None, 'uniform_block_info': None,
    }
else:
    pred_oe, p_odd, sinfo = predict_ds1m(oe_list, sig_idx)

conf     = max(p_odd, 1.0 - p_odd)
next_rid = last_id + 1

# Update log
if pred_oe:
    key            = str(next_rid)
    already_scored = key in log_new and log_new[key].get('actual') is not None
    if not already_scored:
        log_new[key] = {
            'round_id':   key,
            'pred_oe':    pred_oe,
            'p_oe_odd':   round(p_odd, 4),
            'confidence': round(conf, 4),
            'actual':     None,
            'signals':    sinfo,
        }
        _log_changed = True

if _log_changed:
    _tmp = LOG_NEW_PATH + '.tmp'
    with open(_tmp, 'w') as f:
        json.dump(log_new, f, indent=2)
    os.replace(_tmp, LOG_NEW_PATH)


# ── Update new DB + CSV ─────────────────────────────────────────────────────────
def _update_new_db():
    # Sync any new rounds from source DB
    # DS1M: source and destination are the same DB — no sync needed
    conn = _sqlite3.connect(DB_NEW_PATH, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')

    # Write predictions from log
    pred_updates = []
    for rid_s, entry in log_new.items():
        if not entry.get('pred_oe'):
            continue
        p  = float(entry.get('p_oe_odd', 0.5))
        cv = max(p, 1.0 - p)
        pred_updates.append((entry['pred_oe'], round(cv, 4), 0, int(rid_s)))
    if pred_updates:
        conn.executemany(
            'UPDATE rounds SET pred_oe=?, confidence=?, bet=? WHERE id=?',
            pred_updates
        )

    # Recompute WIN/LOSS for ALL rows with pred_oe set (not just 'Not Predicted'),
    # so that updated predictions (from regeneration) get correct results.
    all_pred = conn.execute(
        "SELECT id, disc1||disc2||disc3||disc4, pred_oe "
        "FROM rounds WHERE pred_oe != '' AND pattern != ''"
    ).fetchall()
    result_updates = []
    for rid, pat, p_oe in all_pred:
        if not pat or len(pat) != 4:
            continue
        actual_oe = 'ODD' if pat.count('R') % 2 else 'EVEN'
        result_updates.append(('WIN' if p_oe == actual_oe else 'LOSS', rid))
    if result_updates:
        conn.executemany('UPDATE rounds SET result=? WHERE id=?', result_updates)

    conn.commit()

    # Sync CSV
    rows = conn.execute(
        'SELECT id,disc1,disc2,disc3,disc4,pattern,oe,result,pred_oe,confidence,bet '
        'FROM rounds ORDER BY id'
    ).fetchall()
    conn.close()

    import csv as _csv
    _tmp = CSV_NEW_PATH + '.tmp'
    with open(_tmp, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['id','disc1','disc2','disc3','disc4','pattern','oe','result','pred_oe','confidence','bet'])
        w.writerows(rows)
    try:
        os.replace(_tmp, CSV_NEW_PATH)
    except PermissionError:
        with open(CSV_NEW_PATH, 'w', newline='', encoding='utf-8') as f2:
            with open(_tmp, 'r', newline='', encoding='utf-8') as f3:
                f2.write(f3.read())
        try:
            os.remove(_tmp)
        except OSError:
            pass

_update_new_db()


# ── Write snapshot for server ───────────────────────────────────────────────────
_last_row  = df.iloc[-1]
_last_discs = str(_last_row['disc1']) + str(_last_row['disc2']) + str(_last_row['disc3']) + str(_last_row['disc4'])
_last_oe    = str(_last_row['odd'])

# Recent win rate from log (last 100 scored)
_scored_log = sorted(
    [(int(k), v) for k, v in log_new.items() if v.get('actual') and v.get('pred_oe')],
    key=lambda x: x[0]
)
_last100 = _scored_log[-100:]
_w100 = sum(1 for _, e in _last100 if e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN'))
_wr100 = round(_w100 / len(_last100), 4) if _last100 else None

# Per-round accuracy in log (for accuracy card)
_oe_hits = sum(1 for _, e in _scored_log if e['pred_oe'] == ('ODD' if e['actual'].count('R') % 2 else 'EVEN'))
_oe_acc  = round(_oe_hits / len(_scored_log), 4) if _scored_log else None

_first_pred_id = min(
    (int(k) for k, v in log_new.items() if v.get('pred_oe')), default=None
)

snap = {
    'timestamp':     datetime.datetime.now().isoformat(timespec='seconds'),
    'first_pred_id': _first_pred_id,
    'last_id':       last_id,
    'total_rows':    n_rows,
    'next_round_id': next_rid,
    'next_round':    next_rid,
    'pred_oe':       pred_oe,
    'p_oe_odd':      round(p_odd, 4),
    'confidence':    round(conf, 4),
    'last_discs':    _last_discs,
    'last_oe':       _last_oe,
    'loss_streak':   _cur_loss_streak,
    'total_scored':  len(_scored_log),
    'wr_last100':    _wr100,
    'wr_last100_n':  len(_last100),
    'wr_last100_w':  _w100,
    'oe_accuracy':   _oe_acc,
    'signals': {
        'xgb':             round(sinfo.get('xgb', 0.5), 4),
        'dom_o8':          round(sinfo.get('dom_o8', 0.5), 4),
        'cur_run':         sinfo.get('cur_run', 0),
        'cur_val':         sinfo.get('cur_val', ''),
        'prev_run':        sinfo.get('prev_run', 0),
        'alt_run':         sinfo.get('alt_run', 0),
        'alt_confirmed':   sinfo.get('alt_confirmed', False),
        'rule':            sinfo.get('rule', ''),
        'cycle_detected':  sinfo.get('cycle_detected', False),
        'cycle_pattern':   sinfo.get('cycle_pattern'),
        'cycle_period':    sinfo.get('cycle_period'),
        'cycle_cycles':    sinfo.get('cycle_cycles'),
        'cycle_pred_val':  sinfo.get('cycle_pred_val'),
        'cycle_conf':      sinfo.get('cycle_conf'),
        'cycle_transition': sinfo.get('cycle_transition'),
        'cycle_exp_len':   sinfo.get('cycle_exp_len'),
        'mirror_info':     sinfo.get('mirror_info'),
    },
}
_tmp = SNAP_NEW_PATH + '.tmp'
with open(_tmp, 'w') as f:
    json.dump(snap, f, indent=2)
os.replace(_tmp, SNAP_NEW_PATH)

print(
    f'  DS1M: Round {next_rid} → {pred_oe} '
    f'(conf={conf:.1%}, rule={sinfo["rule"]}, streak={_cur_loss_streak})',
    flush=True
)
