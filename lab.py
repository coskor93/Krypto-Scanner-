#!/usr/bin/env python3
"""
Strategie-Labor: testet jede Regel einzeln auf 15m, 1h und 4h, mit verschiedenen
Ziel/Stop-Werten. Zeitraum wird halbiert (Training / Test), damit man nichts
zurechtbiegt. "Zufall" dient als Vergleich (keine Regel, nur Münzwurf).
Start:  python lab.py     (liegt neben krypto_scanner.py)
"""
import random
import krypto_scanner as ks

FEE = ks.FEE_ROUNDTRIP
COINS = 20
MIN_TRADES = 40
PLAN = {
    "15m": dict(candles=3000, hold=16, days=31, exits=[(1.0, 0.6), (1.5, 0.8), (2.0, 1.0)]),
    "1h":  dict(candles=2000, hold=24, days=83, exits=[(1.5, 1.0), (2.5, 1.25), (4.0, 2.0)]),
    "4h":  dict(candles=1500, hold=18, days=250, exits=[(3.0, 1.5), (5.0, 2.5), (8.0, 4.0)]),
}
RULES = ["Trend", "RSI-Umkehr", "Bollinger-Umkehr", "RSI+Bollinger", "Ausbruch", "Zufall"]


def rsi_series(c, n=14):
    out = [50.0] * len(c)
    if len(c) <= n:
        return out
    g = l = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0); l += max(-d, 0)
    g /= n; l /= n
    out[n] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        l = (l * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


def build_signals(h, l, c, v, seed):
    n = len(c)
    e9, e21, e50 = ks.ema(c, 9), ks.ema(c, 21), ks.ema(c, 50)
    rs = rsi_series(c)
    sig = {r: [0] * n for r in RULES}
    rnd = random.Random(seed)
    for i in range(60, n - 1):
        if e9[i - 1] <= e21[i - 1] and e9[i] > e21[i] and c[i] > e50[i]:
            sig["Trend"][i] = 1
        elif e9[i - 1] >= e21[i - 1] and e9[i] < e21[i] and c[i] < e50[i]:
            sig["Trend"][i] = -1
        if rs[i] < 30:
            sig["RSI-Umkehr"][i] = 1
        elif rs[i] > 70:
            sig["RSI-Umkehr"][i] = -1
        w = c[i - 19:i + 1]
        m = sum(w) / 20
        sd = (sum((x - m) ** 2 for x in w) / 20) ** 0.5
        if c[i] < m - 2 * sd:
            sig["Bollinger-Umkehr"][i] = 1
        elif c[i] > m + 2 * sd:
            sig["Bollinger-Umkehr"][i] = -1
        if rs[i] < 30 and c[i] < m - 2 * sd:
            sig["RSI+Bollinger"][i] = 1
        elif rs[i] > 70 and c[i] > m + 2 * sd:
            sig["RSI+Bollinger"][i] = -1
        hh, ll = max(h[i - 20:i]), min(l[i - 20:i])
        vr = v[i] / (sum(v[i - 20:i]) / 20 or 1)
        if c[i] > hh and vr > 1.5:
            sig["Ausbruch"][i] = 1
        elif c[i] < ll and vr > 1.5:
            sig["Ausbruch"][i] = -1
        if rnd.random() < 0.02:
            sig["Zufall"][i] = rnd.choice((1, -1))
    return sig


def simulate(side, tp, sl, o, h, l, c, i, hold):
    e = i + 1                      # Einstieg zur Eröffnung der NÄCHSTEN Kerze
    entry = o[e]
    if side == 1:
        tpp, slp = entry * (1 + tp / 100), entry * (1 - sl / 100)
    else:
        tpp, slp = entry * (1 - tp / 100), entry * (1 + sl / 100)
    last = min(e + hold - 1, len(c) - 1)
    for j in range(e, last + 1):
        if side == 1:
            if l[j] <= slp:
                return -sl - FEE, j
            if h[j] >= tpp:
                return tp - FEE, j
        else:
            if h[j] >= slp:
                return -sl - FEE, j
            if l[j] <= tpp:
                return tp - FEE, j
    pnl = (c[last] / entry - 1) * 100 if side == 1 else (1 - c[last] / entry) * 100
    return pnl - FEE, last


def run_combo(data, rule, tp, sl, hold):
    train, test = [], []
    for o, h, l, c, sg, split in data:
        s, n, i = sg[rule], len(c), 60
        while i < n - 1:
            if s[i]:
                pnl, j = simulate(s[i], tp, sl, o, h, l, c, i, hold)
                (train if i < split else test).append(pnl)
                i = j + 1
            else:
                i += 1
    return train, test


def avg(x):
    return sum(x) / len(x) if x else 0.0


def lab():
    syms = ks.top_symbols()[:COINS]
    alle_pos, alle_total = 0, 0
    for iv, cfg in PLAN.items():
        print(f"\n=== {iv}: lade Daten ===")
        data = []
        for k, s in enumerate(syms):
            try:
                kl = ks.klines_range(s, iv, cfg["candles"])[:-1]
            except Exception as e:
                print(s, "Fehler", e)
                continue
            if len(kl) < 200:
                continue
            o = [float(x[1]) for x in kl]; h = [float(x[2]) for x in kl]
            l = [float(x[3]) for x in kl]; c = [float(x[4]) for x in kl]
            v = [float(x[5]) for x in kl]
            data.append((o, h, l, c, build_signals(h, l, c, v, k), 60 + (len(c) - 60) // 2))
        if not data:
            continue
        rows, zufall = [], []
        for rule in RULES:
            for tp, sl in cfg["exits"]:
                tr, te = run_combo(data, rule, tp, sl, cfg["hold"])
                rows.append((rule, tp, sl, len(tr), avg(tr), len(te), avg(te)))
                if rule == "Zufall":
                    zufall += tr + te
        for r in rows:
            print(f"{iv} {r[0]:17} Z{r[1]}/S{r[2]}: Training n={r[3]} Ø{r[4]:+.2f}% | Test n={r[5]} Ø{r[6]:+.2f}%")
        echte = [r for r in rows if r[0] != "Zufall" and r[3] >= MIN_TRADES and r[5] >= MIN_TRADES]
        echte.sort(key=lambda r: min(r[4], r[6]), reverse=True)
        pos = [r for r in echte if r[4] > 0 and r[6] > 0]
        alle_pos += len(pos); alle_total += len(echte)
        txt = (f"Labor {iv} | {len(data)} Coins | ca. {cfg['days']} Tage | Gebühr {FEE}%\n"
               f"Einstieg zur nächsten Kerze, Zeit halbiert (Training/Test)\n"
               f"Zufall (Vergleich): Ø {avg(zufall):+.2f}% pro Trade\n\nBeste 5 (Ø netto, Training / Test):\n")
        for k, r in enumerate(echte[:5], 1):
            txt += f"{k}. {r[0]} Z{r[1]}/S{r[2]}: {r[4]:+.2f}% (n={r[3]}) / {r[6]:+.2f}% (n={r[5]})\n"
        txt += f"\nIn BEIDEN Hälften positiv: {len(pos)} von {len(echte)}"
        print("\n" + txt)
        ks.notify(txt)
    fazit = (f"Fazit: {alle_pos} von {alle_total} Regel-Kombinationen waren in beiden Zeitabschnitten "
             "positiv. Bei so vielen Tests sind einige Treffer reiner Zufall, "
             "echte Vorteile müssten deutlich über dem Zufalls-Vergleich liegen und in allen Zeitrahmen auftauchen.")
    print("\n" + fazit)
    ks.notify(fazit)


if __name__ == "__main__":
    lab()
