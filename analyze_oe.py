import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv(r'data\1min-discs.csv', names=['id','d1','d2','d3','d4'],
                 skiprows=1, usecols=[0,1,2,3,4])
df['r_count'] = (df['d1']+df['d2']+df['d3']+df['d4']).str.count('R')
df['oe'] = df['r_count'].apply(lambda r: 'O' if r%2 else 'E')
oe = list(df['oe'])
n  = len(oe)

# -- Run-length encoding --------------------------------------------------------
runs = []
cur_val, cur_len = oe[0], 1
for v in oe[1:]:
    if v == cur_val:
        cur_len += 1
    else:
        runs.append((cur_val, cur_len))
        cur_val, cur_len = v, 1
runs.append((cur_val, cur_len))

e_lens = sorted([l for v,l in runs if v=='E'])
o_lens = sorted([l for v,l in runs if v=='O'])

print("=" * 60)
print("  ODD/EVEN PATTERN ANALYSIS")
print("=" * 60)
print(f"  Records : {n}   Runs: {len(runs)}")
print(f"  ODD     : {oe.count('O')} ({oe.count('O')/n:.1%})")
print(f"  EVEN    : {oe.count('E')} ({oe.count('E')/n:.1%})")
print()

# -- 1. Run length distributions -----------------------------------------------
print("-- 1. Consecutive run lengths --------------------------")
print("  EVEN runs:")
ec = Counter(e_lens)
for l in sorted(ec):
    pct = ec[l]/len(e_lens)
    bar = '#' * int(pct*60)
    print(f"    {l:3d}: {ec[l]:4d} ({pct:4.1%}) {bar}")
print(f"  Avg EVEN run: {np.mean(e_lens):.2f}  Median: {np.median(e_lens):.0f}  Max: {max(e_lens)}")
print()
print("  ODD runs:")
oc = Counter(o_lens)
for l in sorted(oc):
    pct = oc[l]/len(o_lens)
    bar = '#' * int(pct*60)
    print(f"    {l:3d}: {oc[l]:4d} ({pct:4.1%}) {bar}")
print(f"  Avg ODD  run: {np.mean(o_lens):.2f}  Median: {np.median(o_lens):.0f}  Max: {max(o_lens)}")
print()

# -- 2. After a run of length N, does it continue or switch? -------------------
print("-- 2. Run continuation vs switch -----------------------")
for val in ['E', 'O']:
    label = 'EVEN' if val=='E' else 'ODD'
    print(f"  {label}: after N consecutive, does next continue or switch?")
    for length in range(1, 12):
        cont = sum(1 for i in range(len(runs)-1)
                   if runs[i][0]==val and runs[i][1]>length)
        switch = sum(1 for i in range(len(runs)-1)
                     if runs[i][0]==val and runs[i][1]==length)
        total = cont + switch
        if total < 5:
            break
        p_cont = cont/total
        bar_c = '#' * int(p_cont*30)
        bar_s = '-' * int((1-p_cont)*30)
        print(f"    after {length}: continue={p_cont:.0%} switch={1-p_cont:.0%}  |{bar_c}{bar_s}|  (n={total})")
    print()

# -- 3. Run-pair patterns -------------------------------------------------------
print("-- 3. Most common block patterns (EVEN_run + ODD_run) --")
pairs = []
i = 0
while i < len(runs)-1:
    if runs[i][0]=='E':
        pairs.append((runs[i][1], runs[i+1][1]))
        i += 2
    else:
        i += 1
pc = Counter(pairs)
print("  Top 20 (E_len + O_len) blocks:")
for (el,ol),cnt in pc.most_common(20):
    block = 'E'*el + 'O'*ol
    print(f"    E{el}+O{ol}: {cnt:4d} times  {block}")
print()

# -- 4. Transition matrix (what length run follows current run length) ----------
print("-- 4. What run length comes AFTER a run of length N ----")
for val in ['E', 'O']:
    opp = 'O' if val=='E' else 'E'
    label = 'EVEN' if val=='E' else 'ODD'
    print(f"  After {label} run ends -> {('ODD' if val=='E' else 'EVEN')} run length:")
    trans = {}
    for i in range(len(runs)-1):
        v, l = runs[i]
        nv, nl = runs[i+1]
        if v==val:
            trans.setdefault(l, []).append(nl)
    for l in sorted(trans)[:10]:
        vals = trans[l]
        if len(vals) < 3:
            continue
        cnt  = Counter(vals)
        top3 = cnt.most_common(3)
        avg  = np.mean(vals)
        top_str = '  '.join(f"{length}({c})" for length,c in top3)
        print(f"    after E{l}: avg_next={avg:.1f}  most common: {top_str}")
    print()

# -- 5. Balance analysis: ODD/EVEN ratio over rolling windows ------------------
print("-- 5. ODD/EVEN balance in rolling windows ---------------")
arr = np.array([1 if x=='O' else 0 for x in oe])
for w in [10, 20, 50, 100]:
    ratios = pd.Series(arr).rolling(w).mean().dropna()
    print(f"  Window {w:3d}: mean={ratios.mean():.3f} "
          f"std={ratios.std():.3f} "
          f"min={ratios.min():.2f} max={ratios.max():.2f}")
print()

# -- 6. Predictability: after same OE sequence of length N, how consistent? ----
print("-- 6. Sequence predictability (how often same window -> same next) -")
for seq_len in range(2, 9):
    votes = []
    counts_all = Counter()
    for i in range(seq_len, n-1):
        seq = tuple(oe[i-seq_len:i])
        nxt = oe[i]
        counts_all[(seq, nxt)] += 1
    seq_totals = Counter()
    for (seq, nxt), cnt in counts_all.items():
        seq_totals[seq] += cnt
    confidences = []
    for seq, total in seq_totals.items():
        if total < 5:
            continue
        o_cnt = counts_all.get((seq,'O'), 0)
        e_cnt = counts_all.get((seq,'E'), 0)
        conf  = max(o_cnt, e_cnt) / total
        confidences.append(conf)
    if confidences:
        avg_conf = np.mean(confidences)
        high_conf = sum(1 for c in confidences if c >= 0.65)
        print(f"  Seq len {seq_len}: avg confidence={avg_conf:.1%}  "
              f"sequences>=65% confident: {high_conf}/{len(confidences)}")
print()

# -- 7. Current state -----------------------------------------------------------
print("-- 7. Current state (last 60 records) -------------------")
last_run_val, last_run_len = runs[-1]
last60 = ''.join(oe[-60:])
print(f"  Last 60: {last60}")
print(f"  Current run: {last_run_val} x {last_run_len}")

# What happened historically after same run?
same_after = [(runs[i+1][0], runs[i+1][1])
              for i in range(len(runs)-1)
              if runs[i][0]==last_run_val and runs[i][1]==last_run_len]
if same_after:
    nv  = Counter(val for val,_ in same_after)
    nl  = Counter(lng for _,lng in same_after)
    tot = len(same_after)
    print(f"  After {last_run_val}x{last_run_len} ({tot} times in history):")
    for v,c in nv.most_common():
        print(f"    Next={v}: {c}/{tot} = {c/tot:.0%}")
    top_lens = nl.most_common(5)
    print(f"    Next run lengths: {[(l,c) for l,c in top_lens]}")
