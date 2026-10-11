#!/usr/bin/env python3
"""
Trend-Labor: Mehrzeitrahmen-Trend (1h + 4h) + Kerzenmuster auf 15m, Long UND Short.
Prüft, ob Kerzenanalyse + Trendfilter einen messbaren Vorteil gegenüber Zufall bringt.
Braucht lab.py und krypto_scanner.py im selben Ordner.   Start: python lab3.py
"""
import random
import krypto_scanner as ks
import lab

COINS = 20
EXITS = [(1.0, 0.6), (1.5, 0.8), (2.0, 1.0)]
HOLD = 24                       # max. 24 x 15m = 6 Stunden
RULES = ["Pullback im Trend", "Engulfing im Trend", "Pin-Bar im Trend", "Ausbruch im Trend",
         "Engulfing ohne Trend", "Pin-Bar ohne Trend", "Zufall im Trend", "Zufall ohne Trend"]


def htf_trend(kl):
    c = [float(k[4]) for k in kl]
    e20, e50 = ks.ema(c, 20), ks.ema(c, 50)
    out = []
    for i, k in enumerate(kl):
        t = 0
        if i >= 50:
            if e20[i] > e50[i] and c[i] > e20[i]:
                t = 1
            elif e20[i] < e50[i] and c[i] < e20[i]:
                t = -1
        out.append((int(k[6]), t))
    return out


def align(close_times, htf):
    res, j = [], -1
    for ct in close_times:
        while j + 1 < len(htf) and htf[j + 1][0] <= ct:
            j += 1
        res.append(htf[j][1] if j >= 0 else 0)
    return res


def build(o, h, l, c, v, tr, seed):
    n = len(c)
    e21 = ks.ema(c, 21)
    sig = {r: [0] * n for r in RULES}
    rnd = random.Random(seed)
    for i in range(60, n - 1):
        t = tr[i]
        rng = h[i] - l[i]
        lower = min(o[i], c[i]) - l[i]
        upper = h[i] - max(o[i], c[i])
        bull_eng = c[i] > o[i] and c[i - 1] < o[i - 1] and c[i] >= o[i - 1] and o[i] <= c[i - 1]
        bear_eng = c[i] < o[i] and c[i - 1] > o[i - 1] and c[i] <= o[i - 1] and o[i] >= c[i - 1]
        hammer = rng > 0 and lower >= 0.6 * rng
        star = rng > 0 and upper >= 0.6 * rng
        if t == 1 and l[i] <= e21[i] < c[i] and c[i] > o[i]:
            sig["Pullback im Trend"][i] = 1
        elif t == -1 and h[i] >= e21[i] > c[i] and c[i] < o[i]:
            sig["Pullback im Trend"][i] = -1
        if bull_eng:
            sig["Engulfing ohne Trend"][i] = 1
            if t == 1:
                sig["Engulfing im Trend"][i] = 1
        elif bear_eng:
            sig["Engulfing ohne Trend"][i] = -1
            if t == -1:
                sig["Engulfing im Trend"][i] = -1
        if hammer and not star:
            sig["Pin-Bar ohne Trend"][i] = 1
            if t == 1:
                sig["Pin-Bar im Trend"][i] = 1
        elif star and not hammer:
            sig["Pin-Bar ohne Trend"][i] = -1
            if t == -1:
                sig["Pin-Bar im Trend"][i] = -1
        if t == 1 and c[i] > max(h[i - 12:i]):
            sig["Ausbruch im Trend"][i] = 1
        elif t == -1 and c[i] < min(l[i - 12:i]):
            sig["Ausbruch im Trend"][i] = -1
        if rnd.random() < 0.02:
            sig["Zufall ohne Trend"][i] = rnd.choice((1, -1))
        if t != 0 and rnd.random() < 0.04:
            sig["Zufall im Trend"][i] = t
    return sig


def fetch(sym, iv, total):
    return ks.klines_range(sym, iv, total)[:-1]


def run():
    syms = ks.top_symbols()[:COINS]
    data = []
    for k, s in enumerate(syms):
        try:
            k15, k1h, k4h = fetch(s, "15m", 5000), fetch(s, "1h", 1500), fetch(s, "4h", 500)
        except Exception as e:
            print(s, "Fehler", e)
            continue
        if len(k15) < 1000 or len(k1h) < 100 or len(k4h) < 60:
            continue
        cts = [int(x[6]) for x in k15]
        t1, t4 = align(cts, htf_trend(k1h)), align(cts, htf_trend(k4h))
        tr = [a if a == b else 0 for a, b in zip(t1, t4)]
        o = [float(x[1]) for x in k15]; h = [float(x[2]) for x in k15]
        l = [float(x[3]) for x in k15]; c = [float(x[4]) for x in k15]
        v = [float(x[5]) for x in k15]
        data.append((o, h, l, c, build(o, h, l, c, v, tr, k), 60 + (len(c) - 60) // 2))
        print(s, "fertig")
    if not data:
        ks.notify("Trend-Labor: keine Daten geladen.")
        return
    days = int(sum(len(d[3]) for d in data) / len(data) * 15 / 1440)
    rows, rnd_tr, rnd_free = [], [], []
    for rule in RULES:
        for tp, sl in EXITS:
            tr_, te_ = lab.run_combo(data, rule, tp, sl, HOLD)
            rows.append((rule, tp, sl, len(tr_), lab.avg(tr_), len(te_), lab.avg(te_)))
            if rule == "Zufall im Trend":
                rnd_tr += tr_ + te_
            if rule == "Zufall ohne Trend":
                rnd_free += tr_ + te_
    for r in rows:
        print(f"{r[0]:22} Z{r[1]}/S{r[2]}: Training n={r[3]} Ø{r[4]:+.2f}% | Test n={r[5]} Ø{r[6]:+.2f}%")
    echte = [r for r in rows if not r[0].startswith("Zufall") and r[3] >= lab.MIN_TRADES and r[5] >= lab.MIN_TRADES]
    echte.sort(key=lambda r: min(r[4], r[6]), reverse=True)
    pos = [r for r in echte if r[4] > 0 and r[6] > 0]
    txt = (f"Trend-Labor | {len(data)} Coins | ca. {days} Tage | 15m-Kerzen, Long+Short\n"
           f"Trend = 1h UND 4h gleiche Richtung. Gebühr {ks.FEE_ROUNDTRIP}%\n"
           f"Zufall mit Trend: Ø {lab.avg(rnd_tr):+.2f}% | Zufall ohne Trend: Ø {lab.avg(rnd_free):+.2f}%\n\n"
           "Beste 6 (Ø netto, Training / Test):\n")
    for k, r in enumerate(echte[:6], 1):
        txt += f"{k}. {r[0]} Z{r[1]}/S{r[2]}: {r[4]:+.2f}% (n={r[3]}) / {r[6]:+.2f}% (n={r[5]})\n"
    txt += f"\nIn BEIDEN Hälften positiv: {len(pos)} von {len(echte)}"
    print("\n" + txt)
    ks.notify(txt)


if __name__ == "__main__":
    run()
