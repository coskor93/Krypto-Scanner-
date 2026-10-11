#!/usr/bin/env python3
"""
Krypto-Scanner mit Doppelmeldungs-Sperre, Bestätigungsfilter und Backtest.
  python krypto_scanner.py              Dauerbetrieb (alle SCAN_MINUTEN)
  python krypto_scanner.py --once       ein Scan (für GitHub Actions)
  python krypto_scanner.py --backtest   Strategie an historischen Daten testen
Nur Signale, keine Orders. Keine Anlageberatung, keine Gewinngarantie.
"""
import os
import sys
import json
import time
import requests
from datetime import datetime

# ---------------- Einstellungen ----------------
BASE = "https://data-api.binance.vision"
INTERVAL = "15m"            # 5m, 15m oder 1h
TOP_N = 40                  # so viele USDT-Paare scannen
SCAN_MINUTEN = 15
TAKE_PROFIT = 1.5           # Ziel in %
STOP_LOSS = 0.8             # Stop in %
MIN_ATR_PCT = 0.35          # Mindestbewegung pro Kerze
MIN_SCORE = 3               # Mindest-Signalstärke (max. 5)
CONFIRM_KERZE = True        # Filter: Long nur nach grüner, Short nur nach roter Kerze
COOLDOWN_MIN = 120          # gleicher Coin + Richtung wird so lange nicht erneut gemeldet
STATE_FILE = "state.json"
FEE_ROUNDTRIP = 0.2         # Gebühren für Ein+Ausstieg in % (Backtest)
HOLD_KERZEN = 16            # max. Haltedauer im Backtest (16 x 15m = 4 h)
BT_COINS = 20               # Coins im Backtest
BT_KERZEN = 3000            # Historie im Backtest (3000 x 15m = ca. 31 Tage)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# -----------------------------------------------

STABLES = ("USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "BUSD", "USD1")
MINUTEN = {"5m": 5, "15m": 15, "1h": 60}


def get(path, **params):
    r = requests.get(BASE + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def top_symbols():
    data = get("/api/v3/ticker/24hr")
    pairs = [d for d in data if d["symbol"].endswith("USDT")
             and not any(d["symbol"].startswith(s) for s in STABLES)
             and not d["symbol"].endswith(("UPUSDT", "DOWNUSDT"))]
    pairs.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:TOP_N]]


# ---------- Indikatoren ----------
def ema(values, n):
    k = 2 / (n + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes, n=14):
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    ag, al = gains / n, losses / n
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr_pct(h, l, c, n=14):
    trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
           for i in range(1, len(c))]
    return sum(trs[-n:]) / n / c[-1] * 100


def bollinger(closes, n=20, k=2):
    w = closes[-n:]
    m = sum(w) / n
    sd = (sum((x - m) ** 2 for x in w) / n) ** 0.5
    return m - k * sd, m, m + k * sd


# ---------- Signal-Logik (gleich für Live und Backtest) ----------
def signal(h, l, c, v):
    """Gibt Dict mit Richtung/Stärke zurück oder None. Nutzt nur fertige Kerzen."""
    if len(c) < 60:
        return None
    price = c[-1]
    a = atr_pct(h, l, c)
    if a < MIN_ATR_PCT:
        return None
    r = rsi(c)
    e9, e21, e50 = ema(c, 9)[-1], ema(c, 21)[-1], ema(c, 50)[-1]
    lo, _, up = bollinger(c)
    vol_ratio = v[-1] / (sum(v[-21:-1]) / 20 or 1)
    mom = (c[-1] / c[-4] - 1) * 100

    ls, ss, wl, ws = 0, 0, [], []
    if e9 > e21 > e50:
        ls += 1; wl.append("Aufwärtstrend")
    if e9 < e21 < e50:
        ss += 1; ws.append("Abwärtstrend")
    if r < 32:
        ls += 1; wl.append(f"RSI überverkauft ({r:.0f})")
    if r > 68:
        ss += 1; ws.append(f"RSI überkauft ({r:.0f})")
    if price <= lo:
        ls += 1; wl.append("unter Bollinger-Band")
    if price >= up:
        ss += 1; ws.append("über Bollinger-Band")
    if mom > 0.4 and vol_ratio > 1.5 and 50 < r < 68:
        ls += 1; wl.append(f"Ausbruch Vol x{vol_ratio:.1f}")
    if mom < -0.4 and vol_ratio > 1.5 and 32 < r < 50:
        ss += 1; ws.append(f"Abverkauf Vol x{vol_ratio:.1f}")
    if a >= 0.6:
        ls += 1; ss += 1
        wl.append(f"ATR {a:.2f}%"); ws.append(f"ATR {a:.2f}%")

    if ls >= ss and ls > 0:
        side, score, why = "LONG", ls, wl
        confirm = c[-1] > c[-2]       # grüne Kerze
    elif ss > ls:
        side, score, why = "SHORT", ss, ws
        confirm = c[-1] < c[-2]       # rote Kerze
    else:
        return None
    return dict(side=side, score=score, price=price, atr=a,
                why=", ".join(why), confirm=confirm)


def passes(sig, min_score=None, confirm=None):
    ms = MIN_SCORE if min_score is None else min_score
    cf = CONFIRM_KERZE if confirm is None else confirm
    return sig and sig["score"] >= ms and (sig["confirm"] or not cf)


def levels(side, price):
    if side == "LONG":
        return price * (1 + TAKE_PROFIT / 100), price * (1 - STOP_LOSS / 100)
    return price * (1 - TAKE_PROFIT / 100), price * (1 + STOP_LOSS / 100)


# ---------- Live-Scan ----------
def analyse(symbol):
    kl = get("/api/v3/klines", symbol=symbol, interval=INTERVAL, limit=120)[:-1]
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    v = [float(k[5]) for k in kl]
    sig = signal(h, l, c, v)
    if not passes(sig):
        return None
    sig["symbol"] = symbol
    sig["tp"], sig["sl"] = levels(sig["side"], sig["price"])
    return sig


def fmt(p):
    return f"{p:.6f}".rstrip("0").rstrip(".") if p < 1 else f"{p:.4f}"


def notify(text):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000]},
                          timeout=10)
        except Exception as e:
            print("Telegram-Fehler:", e)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def scan():
    print(f"\n=== Scan {datetime.now():%d.%m.%Y %H:%M} | {INTERVAL}-Kerzen ===")
    state = load_state()
    now = time.time()
    state = {k: t for k, t in state.items() if now - t < COOLDOWN_MIN * 60}
    results = []
    for s in top_symbols():
        try:
            res = analyse(s)
            if res:
                results.append(res)
        except Exception as e:
            print(f"{s}: Fehler {e}")
        time.sleep(0.1)
    results.sort(key=lambda x: (-x["score"], -x["atr"]))
    if not results:
        print("Keine klaren Signale.")
        save_state(state)
        return
    neu = []
    for r in results[:8]:
        key = f"{r['symbol']}_{r['side']}"
        line = (f"{r['side']:5} {r['symbol']:10} Stärke {r['score']}/5 | "
                f"Einstieg {fmt(r['price'])} | Ziel {fmt(r['tp'])} | "
                f"Stop {fmt(r['sl'])} | {r['why']}")
        if key in state:
            print("(schon gemeldet)", line)
        else:
            print(line)
            neu.append(line)
            state[key] = now
    save_state(state)
    if neu:
        notify("Krypto-Signale\n" + "\n".join(neu))
    else:
        print("Nur bereits gemeldete Signale – keine neue Nachricht.")


# ---------- Backtest ----------
def klines_range(symbol, interval, total):
    out, end = [], None
    while len(out) < total:
        params = dict(symbol=symbol, interval=interval, limit=1000)
        if end:
            params["endTime"] = end
        part = get("/api/v3/klines", **params)
        if not part:
            break
        out = part + out
        end = part[0][0] - 1
        if len(part) < 1000:
            break
    return out[-total:]


def simulate(side, entry, hi, lo, cl, i):
    tp, sl = levels(side, entry)
    last = min(i + HOLD_KERZEN, len(cl) - 1)
    for j in range(i + 1, last + 1):
        if side == "LONG":
            if lo[j] <= sl:
                return -STOP_LOSS - FEE_ROUNDTRIP, "SL", j
            if hi[j] >= tp:
                return TAKE_PROFIT - FEE_ROUNDTRIP, "TP", j
        else:
            if hi[j] >= sl:
                return -STOP_LOSS - FEE_ROUNDTRIP, "SL", j
            if lo[j] <= tp:
                return TAKE_PROFIT - FEE_ROUNDTRIP, "TP", j
    pnl = (cl[last] / entry - 1) * 100 if side == "LONG" else (1 - cl[last] / entry) * 100
    return pnl - FEE_ROUNDTRIP, "ZEIT", last


def report(name, tr):
    n = len(tr)
    if n == 0:
        return f"{name}\n  keine Trades"
    pct = lambda o: 100 * sum(1 for t in tr if t[1] == o) / n
    avg = sum(t[0] for t in tr) / n
    tot = sum(t[0] for t in tr)
    out = f"{name}\n  Trades {n} | Ziel {pct('TP'):.0f}% | Stop {pct('SL'):.0f}% | Zeit {pct('ZEIT'):.0f}%\n"
    out += f"  Ø netto {avg:+.2f}% pro Trade | Summe {tot:+.1f}%"
    for side in ("LONG", "SHORT"):
        s = [t for t in tr if t[2] == side]
        if s:
            out += f"\n  {side}: {len(s)} Trades, Ø {sum(t[0] for t in s) / len(s):+.2f}%"
    return out


def backtest():
    variants = {
        "A) Stärke>=3, ohne Bestätigung": (3, False),
        "B) Stärke>=3 + Bestätigungskerze": (3, True),
        "C) Stärke>=4 + Bestätigungskerze": (4, True),
    }
    trades = {k: [] for k in variants}
    syms = top_symbols()[:BT_COINS]
    print(f"Backtest über {len(syms)} Coins ...")
    for s in syms:
        try:
            kl = klines_range(s, INTERVAL, BT_KERZEN)[:-1]
        except Exception as e:
            print(s, "Fehler", e)
            continue
        h = [float(k[2]) for k in kl]
        l = [float(k[3]) for k in kl]
        c = [float(k[4]) for k in kl]
        v = [float(k[5]) for k in kl]
        n = len(c)
        sigs = {}
        for i in range(60, n - 2):
            a = max(0, i - 119)
            sigs[i] = signal(h[a:i + 1], l[a:i + 1], c[a:i + 1], v[a:i + 1])
        for name, (ms, cf) in variants.items():
            i = 60
            while i < n - 2:
                sg = sigs.get(i)
                if passes(sg, ms, cf):
                    pnl, res, j = simulate(sg["side"], c[i], h, l, c, i)
                    trades[name].append((pnl, res, sg["side"]))
                    i = j + 1          # keine überlappenden Trades pro Coin
                else:
                    i += 1
        print(s, "fertig")
    tage = len(syms) and BT_KERZEN * MINUTEN.get(INTERVAL, 15) / 1440
    kopf = (f"Backtest {INTERVAL}, {len(syms)} Coins, ca. {tage:.0f} Tage\n"
            f"Ziel {TAKE_PROFIT}% / Stop {STOP_LOSS}% / Gebühr {FEE_ROUNDTRIP}% / max. {HOLD_KERZEN} Kerzen\n"
            f"Break-even ca. {(STOP_LOSS + FEE_ROUNDTRIP) / (TAKE_PROFIT + STOP_LOSS) * 100:.0f}% Zielquote\n")
    text = kopf + "\n" + "\n\n".join(report(k, trades[k]) for k in variants)
    text += "\n\nHinweis: Vergangenheit ist keine Garantie. Bei Stop und Ziel in derselben Kerze wird Stop angenommen."
    print("\n" + text)
    notify(text)


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest()
        sys.exit(0)
    if "--once" in sys.argv:
        scan()
        sys.exit(0)
    while True:
        try:
            scan()
        except Exception as e:
            print("Scan-Fehler:", e)
        time.sleep(SCAN_MINUTEN * 60)
