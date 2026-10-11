#!/usr/bin/env python3
"""
Tages-Test: Trendfolge auf Tageskerzen (nur Long/Spot, Halten über Tage bis Wochen).
Vergleich mit Buy & Hold. Einstieg/Ausstieg zur Eröffnung des Folgetags, Gebühren je Seite.
Start: python lab2.py   (liegt neben krypto_scanner.py)
"""
import statistics as st
import krypto_scanner as ks

FEE_SIDE = ks.FEE_ROUNDTRIP / 2 / 100     # z.B. 0.1 % je Kauf/Verkauf
COINS = 25
MIN_DAYS = 800
WARM = 200
RULES = ["Buy&Hold", "Trend SMA50", "Trend SMA200", "SMA20>SMA50", "Ausbruch 20/10", "Momentum 90T"]


def sma(x, n):
    out, s = [None] * len(x), 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def positions(rule, o, h, l, c):
    n = len(c)
    p = [0] * n
    if rule == "Buy&Hold":
        return [1] * n
    s20, s50, s200 = sma(c, 20), sma(c, 50), sma(c, 200)
    state = 0
    for i in range(WARM, n):
        if rule == "Trend SMA50":
            p[i] = 1 if c[i] > s50[i] else 0
        elif rule == "Trend SMA200":
            p[i] = 1 if c[i] > s200[i] else 0
        elif rule == "SMA20>SMA50":
            p[i] = 1 if s20[i] > s50[i] else 0
        elif rule == "Momentum 90T":
            p[i] = 1 if c[i] > c[i - 90] else 0
        elif rule == "Ausbruch 20/10":
            if state == 0 and c[i] > max(h[i - 20:i]):
                state = 1
            elif state == 1 and c[i] < min(l[i - 10:i]):
                state = 0
            p[i] = state
    return p


def daily_returns(p, o, c):
    """Signal am Schlusskurs von Tag t-1, Ausführung zur Eröffnung von Tag t."""
    rets, expo = [], 0
    for t in range(WARM + 1, len(c)):
        cur, prev = p[t - 1], p[t - 2]
        if cur == 1 and prev == 1:
            r = c[t] / c[t - 1] - 1
        elif cur == 1 and prev == 0:
            r = c[t] / o[t] - 1 - FEE_SIDE
        elif cur == 0 and prev == 1:
            r = o[t] / c[t - 1] - 1 - FEE_SIDE
        else:
            r = 0.0
        if cur == 1:
            expo += 1
        rets.append(r)
    return rets, expo / max(1, len(rets))


def stats(rets):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return (eq - 1) * 100, mdd * 100


def run():
    syms = ks.top_symbols()[:40]
    data = []
    for s in syms:
        try:
            kl = ks.klines_range(s, "1d", 1000)[:-1]
        except Exception as e:
            print(s, "Fehler", e)
            continue
        if len(kl) < MIN_DAYS:
            continue
        data.append((s, [float(k[1]) for k in kl], [float(k[2]) for k in kl],
                     [float(k[3]) for k in kl], [float(k[4]) for k in kl]))
        if len(data) >= COINS:
            break
    if not data:
        ks.notify("Tages-Test: keine Daten geladen.")
        return
    tage = int(st.median(len(d[4]) for d in data))
    res = {r: dict(full=[], h1=[], h2=[], dd=[], ex=[], tr=[]) for r in RULES}
    for s, o, h, l, c in data:
        for r in RULES:
            p = positions(r, o, h, l, c)
            rets, ex = daily_returns(p, o, c)
            m = len(rets) // 2
            res[r]["full"].append(stats(rets)[0])
            res[r]["h1"].append(stats(rets[:m])[0])
            res[r]["h2"].append(stats(rets[m:])[0])
            res[r]["dd"].append(stats(rets)[1])
            res[r]["ex"].append(ex * 100)
            res[r]["tr"].append(sum(1 for i in range(WARM + 1, len(p)) if p[i] != p[i - 1]) / 2)
    med = st.median
    kopf = (f"Tages-Test | {len(data)} Coins | ca. {tage / 365:.1f} Jahre | Gebühr {FEE_SIDE * 100:.1f}% je Seite\n"
            "Nur Long (Spot). Median über alle Coins.\n")
    t1 = kopf + "\nGesamt: Rendite / schlimmster Rückgang / Zeit im Markt\n"
    for r in RULES:
        d = res[r]
        t1 += f"{r}: {med(d['full']):+.0f}% / {med(d['dd']):.0f}% / {med(d['ex']):.0f}%\n"
    t2 = "Rendite je Zeithälfte (1. / 2.), Median:\n"
    for r in RULES:
        d = res[r]
        t2 += f"{r}: {med(d['h1']):+.0f}% / {med(d['h2']):+.0f}%\n"
    bh = res["Buy&Hold"]
    t2 += "\nIn wie vielen Coins war die Regel besser als Buy&Hold?\n"
    for r in RULES[1:]:
        d = res[r]
        n_ret = sum(1 for a, b in zip(d["full"], bh["full"]) if a > b)
        n_dd = sum(1 for a, b in zip(d["dd"], bh["dd"]) if a > b)
        t2 += f"{r}: Rendite {n_ret}/{len(data)}, kleinerer Rückgang {n_dd}/{len(data)}\n"
    t2 += ("\nAchtung: Es sind nur Coins, die HEUTE viel gehandelt werden (Überlebende). "
           "Das verzerrt zugunsten von Buy&Hold und Co. Ein Zeitraum = ein Marktzyklus.")
    for r in RULES:
        print(r, {k: [round(x, 1) for x in v] for k, v in res[r].items() if k in ("full", "dd")})
    print("\n" + t1 + "\n" + t2)
    ks.notify(t1)
    ks.notify(t2)


if __name__ == "__main__":
    run()
