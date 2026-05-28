"""Add flag column to existing CSV marking WWWW and RRRR rows."""
import csv

CSV_PATH = r'c:\Projects\docs\pocs\fxproapp\data\1min-discs.csv'

rows = []
with open(CSV_PATH, newline='') as f:
    for row in csv.reader(f):
        if not row:
            continue
        if row[0] == 'id':
            rows.append(row + ['flag'])
            continue
        if len(row) >= 5 and row[0].isdigit():
            pattern = ''.join(row[1:5])
            flag = pattern if pattern in ('WWWW', 'RRRR') else ''
            rows.append(row[:5] + [flag])

with open(CSV_PATH, 'w', newline='') as f:
    csv.writer(f).writerows(rows)

flagged = sum(1 for r in rows if len(r) > 5 and r[5] in ('WWWW','RRRR'))
print(f"Done. {flagged} rows flagged.")
