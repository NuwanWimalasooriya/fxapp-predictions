import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv(r'data\1min-discs.csv', names=['id','d1','d2','d3','d4'],
                 skiprows=1, usecols=[0,1,2,3,4])
df['r_count'] = (df['d1']+df['d2']+df['d3']+df['d4']).str.count('R')
df['oe'] = df['r_count'].apply(lambda r: 'O' if r%2 else 'E')
oe = list(df['oe'])

# run-length encoding
runs = []
cur_val, cur_len = oe[0], 1
for v in oe[1:]:
    if v == cur_val:
        cur_len += 1
    else:
        runs.append((cur_val, cur_len))
        cur_val, cur_len = v, 1
runs.append((cur_val, cur_len))

print("=" * 60)
print("  CONTEXT-AWARE RUN ANALYSIS")
print("=" * 60)

# ── Q1: Does run length depend on previous run length? ────────────────────────
print("\n-- 1. Current run length vs previous run length --")
print("   (Does a short previous run predict a longer current run?)")
print()

# Group: if previous run was SHORT (1-2), what is current run length?
# If previous run was LONG (3+), what is current run length?
short_prev = []   # previous run len 1-2
long_prev  = []   # previous run len 3+

for i in range(1, len(runs)):
    prev_len = runs[i-1][1]
    cur_len  = runs[i][1]
    if prev_len <= 2:
        short_prev.append(cur_len)
    else:
        long_prev.append(cur_len)

print(f"  After short prev run (len 1-2): avg current run = {np.mean(short_prev):.2f}  "
      f"median = {np.median(short_prev):.0f}  n={len(short_prev)}")
print(f"  After long  prev run (len 3+) : avg current run = {np.mean(long_prev):.2f}  "
      f"median = {np.median(long_prev):.0f}  n={len(long_prev)}")
print()

# More granular: current run length distribution given previous run length
print("  Prev_len -> distribution of current run length (top 5):")
prev_to_cur = {}
for i in range(1, len(runs)):
    pl = runs[i-1][1]
    cl = runs[i][1]
    prev_to_cur.setdefault(pl, []).append(cl)

for pl in sorted(prev_to_cur)[:10]:
    vals  = prev_to_cur[pl]
    if len(vals) < 10:
        continue
    cnt   = Counter(vals)
    top5  = cnt.most_common(5)
    avg   = np.mean(vals)
    pct_long = sum(1 for v in vals if v >= 3) / len(vals)
    top_str  = '  '.join(f"L{l}({c})" for l,c in top5)
    print(f"  After L{pl}: avg={avg:.2f}  >=3: {pct_long:.0%}  -> {top_str}")

# ── Q2: After alternating pattern, does new run go longer? ────────────────────
print()
print("-- 2. Alternating context vs sustained context --")
print("   (User's observation: after EEOOOEEOO... a new EEEE run continues longer)")
print()

# Define "alternating context": last 4 runs are all length 1 or 2
# Define "sustained context": previous run was length 3+
alt_next_lens     = []  # run length after alternating context
sustained_next    = []  # run length after sustained context

for i in range(4, len(runs)):
    last_4_lens  = [runs[j][1] for j in range(i-4, i)]
    cur_len      = runs[i][1]
    if all(l <= 2 for l in last_4_lens):
        alt_next_lens.append(cur_len)
    elif runs[i-1][1] >= 3:
        sustained_next.append(cur_len)

print(f"  After 4 consecutive short runs (all len<=2):")
print(f"    avg next run = {np.mean(alt_next_lens):.2f}  "
      f"pct>=3: {sum(1 for x in alt_next_lens if x>=3)/len(alt_next_lens):.1%}  "
      f"pct>=4: {sum(1 for x in alt_next_lens if x>=4)/len(alt_next_lens):.1%}  "
      f"n={len(alt_next_lens)}")
c = Counter(alt_next_lens)
print(f"    distribution: {dict(sorted(c.items())[:10])}")

print(f"\n  After prev run >= 3 (sustained):")
print(f"    avg next run = {np.mean(sustained_next):.2f}  "
      f"pct>=3: {sum(1 for x in sustained_next if x>=3)/len(sustained_next):.1%}  "
      f"n={len(sustained_next)}")
c2 = Counter(sustained_next)
print(f"    distribution: {dict(sorted(c2.items())[:10])}")

# ── Q3: After seeing 2+ same in a row, probability of 3rd? ────────────────────
print()
print("-- 3. Continuation probability given context --")
print("   (Given we have N in a row, what was the context before this run started?)")
print()

# For current run of length >= 2, was the previous run short or long?
for cur_run_len in [2, 3, 4, 5]:
    short_ctx_cont = 0; short_ctx_switch = 0
    long_ctx_cont  = 0; long_ctx_switch  = 0

    for i in range(1, len(runs)-1):
        val, ln   = runs[i]
        prev_ln   = runs[i-1][1]
        nxt_val   = runs[i+1][0] if i+1 < len(runs) else None

        if ln == cur_run_len:
            # this run is exactly cur_run_len — it switched at this point
            if prev_ln <= 2:
                short_ctx_switch += 1
            else:
                long_ctx_switch += 1
        elif ln > cur_run_len:
            # this run is longer — it continued past cur_run_len
            if prev_ln <= 2:
                short_ctx_cont += 1
            else:
                long_ctx_cont += 1

    sc_total = short_ctx_cont + short_ctx_switch
    lc_total = long_ctx_cont  + long_ctx_switch
    sc_pct   = short_ctx_cont/sc_total if sc_total else 0
    lc_pct   = long_ctx_cont/lc_total  if lc_total else 0
    print(f"  At run length {cur_run_len} — continue probability:")
    print(f"    After short context (prev<=2): {sc_pct:.1%}  (n={sc_total})")
    print(f"    After long  context (prev>=3): {lc_pct:.1%}  (n={lc_total})")
    print()

# ── Q4: Show actual examples of pattern transitions ──────────────────────────
print("-- 4. Run-length sequences around long runs (len>=4) --")
print("   (What came before and after a long run?)")
long_run_contexts = []
for i in range(2, len(runs)-2):
    val, ln = runs[i]
    if ln >= 4:
        before2 = f"L{runs[i-2][1]}{runs[i-2][0]} L{runs[i-1][1]}{runs[i-1][0]}"
        after1  = f"L{runs[i+1][1]}{runs[i+1][0]}"
        long_run_contexts.append((before2, ln, val, after1))

ctx_counts = Counter((b, l, v) for b,l,v,_ in long_run_contexts)
print("  Before -> [LONG RUN] -> After  (top 15):")
for (before, ln, val), cnt in ctx_counts.most_common(15):
    after_runs = [a for b,l,v,a in long_run_contexts if b==before and l==ln and v==val]
    after_cnt  = Counter(after_runs)
    print(f"    {before} -> L{ln}{val} -> {after_cnt.most_common(3)}  ({cnt}x)")
