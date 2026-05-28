"""Sort CSV by ID and remove duplicate rows."""
import csv

CSV_PATH = r'c:\Projects\docs\pocs\fxproapp\data\1min-discs.csv'

rows = {}
header = None
with open(CSV_PATH, newline='') as f:
    for row in csv.reader(f):
        if not row:
            continue
        if row[0] == 'id':
            header = row
            continue
        if row[0].isdigit():
            rows[int(row[0])] = row  # deduplicate by ID

sorted_rows = [rows[k] for k in sorted(rows.keys())]

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    if header:
        writer.writerow(header)
    writer.writerows(sorted_rows)

print(f"Done. {len(sorted_rows)} records written.")
print(f"  First ID : {sorted_rows[0][0]}")
print(f"  Last ID  : {sorted_rows[-1][0]}")
