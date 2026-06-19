"""
Backfill pred_hot_sums for all rounds in fst1m.db.

For each round X: top 5 from the 50 rounds ending AT X (inclusive),
newest-first -- matches the Hot sums (last 50) panel which uses ORDER BY id DESC LIMIT 50.
"""
import sqlite3, json
from collections import Counter

DB_PATH = 'data/fst1m.db'

conn = sqlite3.connect(DB_PATH)
conn.execute('PRAGMA journal_mode=WAL')

existing = {r[1] for r in conn.execute('PRAGMA table_info(rounds)').fetchall()}
if 'pred_hot_sums' not in existing:
    conn.execute('ALTER TABLE rounds ADD COLUMN pred_hot_sums TEXT DEFAULT "[]"')
    conn.commit()
    print('Added pred_hot_sums column.')

all_rounds = conn.execute('SELECT id, total FROM rounds ORDER BY id').fetchall()

db_updates = []
for idx, (round_id, _) in enumerate(all_rounds):
    # 50 rounds ending AT this round (inclusive), newest-first
    window = all_rounds[max(0, idx - 49):idx + 1][::-1]
    counts = Counter(r[1] for r in window)
    hot = [s for s, _ in sorted(counts.items(), key=lambda x: -x[1])[:5]]
    db_updates.append((json.dumps(hot), round_id))

conn.executemany('UPDATE rounds SET pred_hot_sums=? WHERE id=?', db_updates)
conn.commit()
conn.close()

print(f'DB: updated pred_hot_sums for {len(db_updates)} rows.')

conn2 = sqlite3.connect(DB_PATH)
check = conn2.execute('SELECT id, total, pred_hot_sums FROM rounds ORDER BY id DESC LIMIT 5').fetchall()
conn2.close()
for rid, tot, hs in check:
    print(f'  Round {rid}  actual={tot}  pred_hot_sums={hs}')
