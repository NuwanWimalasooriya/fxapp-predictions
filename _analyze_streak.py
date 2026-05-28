import json

log = json.load(open('data/pred_log.json', encoding='utf-8-sig'))
ids = ['2059800378','2059800379','2059800380','2059800381','2059800382','2059800383']

print(f"{'ID':<14} {'actual':6} {'pred':6} {'run':5} {'mom':5} {'bal':5} {'seq':5} {'ml':5} {'lstm':5}")
print('-'*70)
for rid in ids:
    e = log.get(rid, {})
    actual = e.get('actual', '?')
    pred_oe = e.get('pred_oe', '?')
    sigs = e.get('signals', {})
    if actual != '?':
        actual_oe = 'ODD' if actual.count('R') % 2 else 'EVEN'
    else:
        actual_oe = '?'
    def f(v):
        return f'{v:.2f}' if isinstance(v, float) else str(v)
    print(f"{rid:<14} {actual_oe:6} {pred_oe:6} {f(sigs.get('run','?')):5} {f(sigs.get('momentum','?')):5} {f(sigs.get('balance','?')):5} {f(sigs.get('seq','?')):5} {f(sigs.get('ml','?')):5} {f(sigs.get('lstm','?')):5}")
