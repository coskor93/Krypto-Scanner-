"""Telegram-Signalbot für Aktien & Krypto (Daten via yfinance, kein API-Key nötig)."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------- Einstellungen ----------
STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "SAP.DE"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INTERVAL = "1h"      # Kerzen: 15m, 30m, 1h, 1d
PERIOD = "60d"       # genug Historie für EMA200
MIN_SCORE = 3        # Mindestpunktzahl für ein Signal
ATR_STOP = 1.5       # Stop-Loss = 1.5 x ATR
ATR_TARGET = 3.0     # Ziel = 3 x ATR (CRV 1:2)
STATE_FILE = Path("state.json")
# -----------------------------------

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def add_indicators(df):
    c = df["Close"]
    df["ema50"], df["ema200"] = ema(c, 50), ema(c, 200)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / down)
    macd = ema(c, 12) - ema(c, 26)
    df["macd_h"] = macd - ema(macd, 9)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = ma + 2 * sd, ma - 2 * sd
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df


def evaluate(df):
    """Bewertet die letzte ABGESCHLOSSENE Kerze. Gibt Signal-Dict oder None zurück."""
    if len(df) < 210:
        return None
    r, p = df.iloc[-2], df.iloc[-3]
    vol_ok = pd.notna(r["vol_ratio"]) and r["vol_ratio"] > 1.5

    for side in ("LONG", "SHORT"):
        long = side == "LONG"
        score, why = 0, []
        trigger = False

        if long and r.Close > r.ema200 and r.ema50 > r.ema200:
            score += 1; why.append("Aufwärtstrend (Kurs & EMA50 > EMA200)")
        if not long and r.Close < r.ema200 and r.ema50 < r.ema200:
            score += 1; why.append("Abwärtstrend (Kurs & EMA50 < EMA200)")

        if long and p.macd_h <= 0 < r.macd_h:
            score += 2; trigger = True; why.append("MACD bullisches Kreuz")
        if not long and p.macd_h >= 0 > r.macd_h:
            score += 2; trigger = True; why.append("MACD bärisches Kreuz")

        if long and p.rsi < 30 <= r.rsi:
            score += 2; trigger = True; why.append(f"RSI dreht aus überverkauft ({r.rsi:.0f})")
        if not long and p.rsi > 70 >= r.rsi:
            score += 2; trigger = True; why.append(f"RSI dreht aus überkauft ({r.rsi:.0f})")

        if long and r.Close > r.bb_up and p.Close <= p.bb_up:
            score += 2; trigger = True; why.append("Ausbruch über oberes Bollinger-Band")
        if not long and r.Close < r.bb_lo and p.Close >= p.bb_lo:
            score += 2; trigger = True; why.append("Bruch des unteren Bollinger-Bands")

        if vol_ok and trigger:
            score += 1; why.append(f"Volumen {r.vol_ratio:.1f}x über Schnitt")

        if trigger and score >= MIN_SCORE:
            price, atr = float(r.Close), float(r.atr)
            sign = 1 if long else -1
            return {
                "side": side, "score": score, "why": why, "price": price,
                "stop": price - sign * ATR_STOP * atr,
                "target": price + sign * ATR_TARGET * atr,
                "candle": str(df.index[-2]),
            }
    return None


def send(text):
    if not TOKEN or not CHAT_ID:
        print("[Kein Telegram-Token gesetzt] " + text)
        return
    resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                         data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    if not resp.ok:
        print("Telegram-Fehler:", resp.text)


def fmt(sym, sig):
    icon = "🟢" if sig["side"] == "LONG" else "🔴"
    kind = "Krypto" if sym in CRYPTO else "Aktie"
    d = 4 if sig["price"] < 10 else 2
    return (f"{icon} {sig['side']}-Signal: {sym} ({kind}, {INTERVAL})\n"
            f"Score: {sig['score']}\n"
            f"Preis: {sig['price']:.{d}f}\n"
            f"Stop-Loss: {sig['stop']:.{d}f}\n"
            f"Ziel: {sig['target']:.{d}f}\n\n"
            + "\n".join("• " + w for w in sig["why"])
            + "\n\n⚠️ Keine Anlageberatung.")


def main():
    if os.environ.get("TEST_MESSAGE"):
        send("✅ Signalbot läuft und ist mit Telegram verbunden.")
        return
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    for sym in STOCKS + CRYPTO:
        try:
            df = yf.Ticker(sym).history(period=PERIOD, interval=INTERVAL)
            if df.empty:
                print(sym, "keine Daten"); continue
            sig = evaluate(add_indicators(df))
        except Exception as e:
            print(sym, "Fehler:", e); continue
        if not sig:
            print(sym, "kein Signal"); continue
        key = f"{sig['side']}@{sig['candle']}"
        if state.get(sym) == key:
            print(sym, "bereits gesendet"); continue
        send(fmt(sym, sig))
        state[sym] = key
    STATE_FILE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    sys.exit(main())
