"""
Block pattern analysis — finds repeating OE sequences of various sizes,
clusters them, and shows what typically follows each pattern.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

CSV_PATH = r'data\1min-discs.csv'

df = pd.read_csv(CSV_PATH, names=['id','d1','d2','d3','d4'],
                 skiprows=1, usecols=[0,1,2,3,4])
df = df.sort_values('id').reset_index(drop=True)
df['r_count'] = (df['d1']+df['d2']+df['d3']+df['d4']).str.count('R')
df['oe'] = df['r_count'].apply(lambda r: 'O' if r%2 else 'E')
oe  = list(df['oe'])
n   = len(oe)
print(f"Total rounds: {n}\n")

# ── 1. Find high-frequency sliding-window patterns ────────────────────────────
print("=" * 60)
print("  REPEATING PATTERN FREQUENCY  (sliding window)")
print("=" * 60)

for win in [4, 6, 8, 10, 12]:
    freq  = Counter()
    after = defaultdict(list)          # pattern → list of next-4 OE chars

    for i in range(n - win - 4):
        pat = ''.join(oe[i:i+win])
        nxt = ''.join(oe[i+win:i+win+4])
        freq[pat]  += 1
        after[pat].append(nxt)

    total = sum(freq.values())
    top   = freq.most_common(8)

    print(f"\n-- Window {win}  ({total} windows, {len(freq)} unique) --")
    for pat, cnt in top:
        pct      = cnt / total * 100
        nexts    = Counter(after[pat])
        top_next = nexts.most_common(3)
        nxt_str  = '  '.join(f"{p}({c})" for p, c in top_next)
        print(f"  {pat}  {cnt:4d}x ({pct:.1f}%)  → next-4: {nxt_str}")

# ── 2. Current tail — match against high-frequency patterns ───────────────────
print("\n" + "=" * 60)
print("  CURRENT SEQUENCE MATCH")
print("=" * 60)

for win in [4, 6, 8, 10, 12, 16]:
    if n < win:
        continue
    current = ''.join(oe[-win:])
    freq2   = Counter()
    after2  = defaultdict(list)

    for i in range(n - win - 1):
        pat = ''.join(oe[i:i+win])
        freq2[pat] += 1
        after2[pat].append(oe[i+win])   # single next OE

    cnt = freq2.get(current, 0)
    if cnt == 0:
        print(f"\n  Last {win:2d}: {current}  — no historical match")
        continue

    nexts    = Counter(after2[current])
    total_n  = sum(nexts.values())
    o_pct    = nexts.get('O', 0) / total_n * 100
    e_pct    = nexts.get('E', 0) / total_n * 100
    top_next = nexts.most_common(2)
    print(f"\n  Last {win:2d}: {current}  (seen {cnt}x in history)")
    print(f"    Next ODD : {o_pct:.1f}%   Next EVEN: {e_pct:.1f}%   (n={total_n})")

# ── 3. Non-overlapping block repeat analysis ──────────────────────────────────
print("\n" + "=" * 60)
print("  NON-OVERLAPPING BLOCK REPEATS")
print("=" * 60)

for block in [4, 6, 8]:
    blocks = [''.join(oe[i:i+block]) for i in range(0, n - block, block)]
    freq3  = Counter(blocks)
    unique = len(freq3)
    total3 = len(blocks)
    top3   = freq3.most_common(6)

    # How many distinct patterns cover 50% of all occurrences?
    sorted_counts = sorted(freq3.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    cover50 = int(np.searchsorted(cumsum, total3 * 0.5)) + 1

    print(f"\n  Block-{block}: {total3} blocks, {unique} unique patterns")
    print(f"  Top {cover50} patterns cover 50% of all blocks")
    print(f"  Top 6:")
    for pat, cnt in top3:
        odds_in_pat  = pat.count('O')
        print(f"    {pat}  {cnt:4d}x ({cnt/total3:.1%})  ODD_count={odds_in_pat}")

# ── 4. Transition between consecutive blocks ──────────────────────────────────
print("\n" + "=" * 60)
print("  BLOCK TRANSITION MATRIX  (block-4)")
print("=" * 60)

block = 4
blocks4   = [''.join(oe[i:i+block]) for i in range(0, n - block, block)]
freq4     = Counter(blocks4)
top5_pats = [p for p, _ in freq4.most_common(5)]

trans = defaultdict(Counter)
for i in range(len(blocks4) - 1):
    trans[blocks4[i]][blocks4[i+1]] += 1

print(f"\n  Top-5 block patterns and their most likely successors:")
for pat in top5_pats:
    successors = trans[pat].most_common(3)
    total_t    = sum(trans[pat].values())
    suc_str    = '  '.join(f"{p}({c/total_t:.0%})" for p, c in successors)
    print(f"  {pat} ({freq4[pat]}x) -> {suc_str}")

# ── 5. Current block position and prediction ──────────────────────────────────
print("\n" + "=" * 60)
print("  CURRENT BLOCK STATE")
print("=" * 60)

block = 4
# Where are we within the current block?
pos_in_block = n % block
rounds_into_block = pos_in_block
rounds_left = block - pos_in_block if pos_in_block > 0 else 0

last_complete = ''.join(oe[-(pos_in_block + block):-pos_in_block]) if pos_in_block > 0 else ''.join(oe[-block:])
partial_cur   = ''.join(oe[-pos_in_block:]) if pos_in_block > 0 else ''

print(f"\n  Last complete block-4 : {last_complete}")
print(f"  Current partial       : {partial_cur or '(starting new block)'}")
print(f"  Rounds into block     : {rounds_into_block}")

# What block tends to follow the last complete block?
if last_complete in trans:
    successors = trans[last_complete].most_common(4)
    total_t    = sum(trans[last_complete].values())
    print(f"\n  After '{last_complete}', next block distribution:")
    for suc, cnt in successors:
        odd_c = suc.count('O')
        print(f"    {suc}  {cnt/total_t:.1%}  (ODD_count={odd_c})")
    # Based on current partial, narrow down which successor is likely
    if partial_cur:
        matching = [(s, c) for s, c in successors if s.startswith(partial_cur)]
        if matching:
            print(f"\n  Narrowed by current partial '{partial_cur}':")
            total_m = sum(c for _, c in matching)
            for suc, cnt in matching:
                remaining = suc[len(partial_cur):]
                print(f"    Completion: '{remaining}'  ({cnt/total_m:.1%})")
