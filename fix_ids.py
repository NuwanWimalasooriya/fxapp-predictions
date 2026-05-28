"""
Fix IDs in CSV: strip 2059000000 base from API issue numbers.
API issue 2059700332 -> display ID 700332.
"""
import csv, json, os

CSV_PATH = r'c:\Projects\docs\pocs\fxproapp\data\1min-discs.csv'
MAP_PATH = r'c:\Projects\docs\pocs\fxproapp\data\issue_map.json'
BASE     = 2059000000

# --- Fix CSV ---
rows = []
fixed = 0
with open(CSV_PATH, newline='') as f:
    for row in csv.reader(f):
        if row and row[0].isdigit():
            id_val = int(row[0])
            if id_val > BASE:
                row[0] = str(id_val - BASE)
                fixed += 1
        rows.append(row)

with open(CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerows(rows)

print(f"CSV: fixed {fixed} IDs")
print(f"  Sample new IDs: {[r[0] for r in rows[-5:]]}")

# --- Fix issue_map.json ---
if os.path.exists(MAP_PATH):
    with open(MAP_PATH) as f:
        m = json.load(f)
    new_m = {}
    for k, v in m.items():
        new_k = int(k) - BASE if int(k) > BASE else int(k)
        new_v = int(v) - BASE if int(v) > BASE else int(v)
        new_m[str(new_k)] = new_v
    with open(MAP_PATH, 'w') as f:
        json.dump(new_m, f)
    print(f"issue_map.json: updated {len(new_m)} entries")
