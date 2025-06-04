# make_labels.py
import os
import csv

data_dir = 'data'        # relative path from project root
output_csv = 'labels.csv'

rows = []
for composer in ['chopin', 'beethoven', 'haydn', 'mozart']:
    folder = os.path.join(data_dir, composer)
    if not os.path.isdir(folder):
        continue
    for fn in os.listdir(folder):
        if fn.lower().endswith('.mid'):
            rows.append((fn, composer))

# write labels.csv in project root
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'composer'])
    writer.writerows(rows)

print(f"Wrote {len(rows)} entries to {output_csv}")
