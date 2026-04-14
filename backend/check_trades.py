import json
from collections import Counter

with open('C:/DDCHACK/PROYECTOS/APUESTAS/backend/trades.json') as f:
    data = json.load(f)

today = '2026-03-22'
today_filled = [t for t in data if t.get('status')=='FILLED' and t['timestamp'][:10]==today]
market_counts = Counter(t['market_id'] for t in today_filled)

print(f'Trades hoy: {len(today_filled)} en {len(market_counts)} mercados unicos')
print(f'Total gastado hoy: ${sum(t["size_usd"] for t in today_filled):.2f}')
print()
for mid, cnt in market_counts.most_common():
    q = next(t['question'] for t in today_filled if t['market_id']==mid)
    print(f'  x{cnt} | ${cnt:.0f} | {q[:60]}')
