"""Print paired fixed/dynamic B=1 decode throughput, including classifier overhead."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
with (a.root/'summary_rank000.csv').open() as f:
    rows = list(csv.DictReader(f))
best = max((r for r in rows if r['mode'].startswith('fixed_')), key=lambda r: float(r['decode_tok_s']))
baseline = float(best['decode_tok_s'])
print(f"Best fixed: {best['mode']}, {baseline:.2f} tok/s")
print(f"{'mode':36s} {'accepted':>9s} {'committed':>10s} {'tok/s':>10s} {'vs best':>10s} {'selector':>10s}")
for r in rows:
    if r['mode'] == 'ar':
        continue
    print(f"{r['mode']:36s} {float(r['mean_accepted']):9.3f} {float(r['mean_committed']):10.3f} "
          f"{float(r['decode_tok_s']):10.2f} {100*(float(r['decode_tok_s'])/baseline-1):+9.2f}% "
          f"{100*float(r.get('selector_time_fraction', 0)) :9.2f}%")
print(f'Results: {a.root}')
