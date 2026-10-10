#!/usr/bin/env python3
"""
Krypto-Scanner: zeigt regelmäßig Long-/Short-Kandidaten für ~1-2 % Kursbewegung.
Datenquelle: öffentliche Binance-Marktdaten (kein API-Key nötig).
Installation:  pip install requests
Start:         python krypto_scanner.py
KEIN Anlageberatung, KEINE Gewinngarantie. Nur Signale, es werden keine Orders ausgeführt.
"""
import os
import sys
import time
import requests
from datetime import datetime

# ---------------- Einstellungen ----------------
BASE = "https://data-api.binance.vision"   # öffentliche Marktdaten
INTERVAL = "15m"            # Kerzengröße: 5m, 15m, 1h
TOP_N = 40                  # so viele USDT-Paare mit höchstem Volumen scannen
SCAN_MINUTEN = 15           # Wiederholung
TAKE_PROFIT = 1.5           # Ziel in %
STOP_LOSS = 0.8             # Stop in %
MIN_ATR_PCT = 0.35          # Coin muss sich pro Kerze genug bewegen
MIN_SCORE = 3               # Mindest-Signalstärke (max. 5)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")   # Bot-Token von @BotFather
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")  # deine Chat-ID
# -----------------------------------------------

STABLES = ("USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "BUSD", "USD1")


def get(path, **params):
    r = requests.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def top_symbols():
    data = get("/api/v3/ticker/24hr")
    pairs = [d for d in data if d["symbol"].endswith("USDT")
             and not any(d["symbol"].startswith(s) for s in STABLES)
             and not d["symbol"].endswith(("UPUSDT", "DOWNUSDT"))]
    pairs.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in pairs[:TOP_N]]


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


def analyse(symbol):
    kl = get("/api/v3/klines", symbol=symbol, interval=INTERVAL, limit=120)
    kl = kl[:-1]  # unfertige Kerze ignorieren
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    v = [float(k[5]) for k in kl]

    price = c[-1]
    r = rsi(c)
    a = atr_pct(h, l, c)
    if a < MIN_ATR_PCT:
        return None
    e9, e21, e50 = ema(c, 9)[-1], ema(c, 21)[-1], ema(c, 50)[-1]
    lo, mid, up = bollinger(c)
    vol_ratio = v[-1] / (sum(v[-21:-1]) / 20 or 1)
    mom = (c[-1] / c[-4] - 1) * 100  # Momentum letzte 3 Kerzen

    long_s, short_s, why_l, why_s = 0, 0, [], []

    # Trendfolge
    if e9 > e21 > e50:
        long_s += 1; why_l.append("Aufwärtstrend (EMA9>21>50)")
    if e9 < e21 < e50:
        short_s += 1; why_s.append("Abwärtstrend (EMA9<21<50)")
    # Mean Reversion
    if r < 32:
        long_s += 1; why_l.append(f"RSI überverkauft ({r:.0f})")
    if r > 68:
        short_s += 1; why_s.append(f"RSI überkauft ({r:.0f})")
    if price <= lo:
        long_s += 1; why_l.append("unter unterem Bollinger-Band")
    if price >= up:
        short_s += 1; why_s.append("über oberem Bollinger-Band")
    # Momentum + Volumen
    if mom > 0.4 and vol_ratio > 1.5 and 50 < r < 68:
        long_s += 1; why_l.append(f"Ausbruch mit Volumen x{vol_ratio:.1f}")
    if mom < -0.4 and vol_ratio > 1.5 and 32 < r < 50:
        short_s += 1; why_s.append(f"Abverkauf mit Volumen x{vol_ratio:.1f}")
    # Volatilität ausreichend für Ziel
    if a >= 0.6:
        long_s += 1; short_s += 1
        why_l.append(f"ATR {a:.2f}%"); why_s.append(f"ATR {a:.2f}%")

    if long_s >= short_s and long_s >= MIN_SCORE:
        side, score, why = "LONG", long_s, why_l
        tp, sl = price * (1 + TAKE_PROFIT / 100), price * (1 - STOP_LOSS / 100)
    elif short_s > long_s and short_s >= MIN_SCORE:
        side, score, why = "SHORT", short_s, why_s
        tp, sl = price * (1 - TAKE_PROFIT / 100), price * (1 + STOP_LOSS / 100)
    else:
        return None
    return dict(symbol=symbol, side=side, score=score, price=price, tp=tp,
                sl=sl, rsi=r, atr=a, why=", ".join(why))


def fmt(p):
    return f"{p:.6f}".rstrip("0").rstrip(".") if p < 1 else f"{p:.4f}"


def notify(text):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        except Exception as e:
            print("Telegram-Fehler:", e)


def scan():
    print(f"\n=== Scan {datetime.now():%d.%m.%Y %H:%M} | {INTERVAL}-Kerzen ===")
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
        print("Keine klaren Signale – abwarten ist auch eine Position.")
        return
    msg = []
    for r in results[:8]:
        line = (f"{r['side']:5} {r['symbol']:10} Stärke {r['score']}/5 | "
                f"Einstieg {fmt(r['price'])} | Ziel {fmt(r['tp'])} | "
                f"Stop {fmt(r['sl'])} | {r['why']}")
        print(line)
        msg.append(line)
    notify("Krypto-Signale\n" + "\n".join(msg))


if __name__ == "__main__":
    if "--once" in sys.argv:   # für Cloud-Betrieb (GitHub Actions)
        scan()
        sys.exit(0)
    while True:
        try:
            scan()
        except Exception as e:
            print("Scan-Fehler:", e)
        time.sleep(SCAN_MINUTEN * 60)
