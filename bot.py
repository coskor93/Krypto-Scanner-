"""Telegram-Signalbot für Aktien & Krypto (Daten via yfinance, kein API-Key nötig)."""
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------- Einstellungen ----------
EXTRA_STOCKS = ["SAP.DE", "SIE.DE", "ALV.DE", "RHM.DE"]   # zusätzlich zum S&P 500
CRYPTO_TOP_N = 50    # Top-N Kryptos nach Marktkapitalisierung (CoinGecko)
BITVAVO = "https://api.bitvavo.com/v2"   # Krypto-Kurse direkt von Bitvavo (EUR-Märkte)
SHORTS_STOCKS = True   # True = bei Aktien auch SHORT-Signale senden
SHORTS_CRYPTO = True    # True = bei Krypto auch SHORT-/Verkaufs-Signale senden
MAX_SIGNALS = 10     # maximal so viele Nachrichten pro Lauf (beste Scores zuerst)
FALLBACK_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO"]
TIMEFRAMES = ["1h", "15m"]   # Zeitebenen, die gescannt werden (z. B. "15m", "30m", "1h", "1d")
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


def get_sp500():
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "Mozilla/5.0 signalbot"}, timeout=30)
        r.raise_for_status()
        tab = pd.read_html(StringIO(r.text))[0]
        syms = [x.replace(".", "-") for x in tab["Symbol"].astype(str)]
        print(f"S&P 500: {len(syms)} Symbole geladen")
        return syms
    except Exception as e:
        print("S&P-500-Liste nicht ladbar, nutze Fallback:", e)
        return FALLBACK_STOCKS


STABLE = {"USDT", "USDC", "DAI", "USDE", "FDUSD", "TUSD", "USDS", "PYUSD", "USD1", "USDD", "BUIDL", "USDTB"}


def get_top_crypto():
    """Top-N Coins (CoinGecko-Rang), die bei Bitvavo gegen EUR handelbar sind."""
    try:
        m = requests.get(f"{BITVAVO}/markets", timeout=30)
        m.raise_for_status()
        avail = {x["base"] for x in m.json()
                 if x.get("quote") == "EUR" and x.get("status") == "trading"}
    except Exception as e:
        print("Bitvavo-Marktliste nicht ladbar:", e)
        return []

    ranked = []
    try:
        r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                         params={"vs_currency": "usd", "order": "market_cap_desc",
                                 "per_page": 150, "page": 1}, timeout=30)
        r.raise_for_status()
        for c in r.json():
            sym, name = c["symbol"].upper(), c["name"].lower()
            if sym in STABLE or any(w in name for w in ("wrapped", "staked", "bridged", "usd", "stable")):
                continue
            if sym in avail:
                ranked.append(sym)
    except Exception as e:
        print("CoinGecko nicht ladbar, nutze Bitvavo-Handelsvolumen:", e)

    if not ranked:
        try:
            t = requests.get(f"{BITVAVO}/ticker/24h", timeout=30).json()
            t = [x for x in t if x["market"].endswith("-EUR")
                 and x["market"][:-4] in avail and x["market"][:-4] not in STABLE]
            t.sort(key=lambda x: float(x.get("volumeQuote") or 0), reverse=True)
            ranked = [x["market"][:-4] for x in t]
        except Exception as e:
            print("Bitvavo-Ticker nicht ladbar:", e)

    out = [f"{s}-EUR" for s in ranked[:CRYPTO_TOP_N]]
    print(f"Krypto: {len(out)} Bitvavo-Märkte geladen")
    return out


def fetch_bitvavo(markets, interval):
    """Holt 1h-Kerzen direkt von Bitvavo (öffentliche API, kein Key nötig)."""
    out = {}
    for mk in markets:
        try:
            params = {"interval": interval, "limit": 500}
            r = requests.get(f"{BITVAVO}/{mk}/candles", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(15)
                r = requests.get(f"{BITVAVO}/{mk}/candles", params=params, timeout=20)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                continue
            df = pd.DataFrame([row[:6] for row in rows],
                              columns=["ts", "Open", "High", "Low", "Close", "Volume"])
            df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
            out[mk] = df.set_index("ts").astype(float).sort_index()
        except Exception as e:
            print(mk, "Fehler:", e)
        time.sleep(0.1)
    return out


def download(symbols, interval, chunk_size=50):
    """Lädt Kerzen in Blöcken (viel schneller als einzeln). Gibt {symbol: DataFrame} zurück."""
    out = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=PERIOD, interval=interval, group_by="ticker",
                               threads=True, progress=False, auto_adjust=False)
        except Exception as e:
            print("Download-Fehler:", e); continue
        for sym in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.get_level_values(0):
                        continue
                    df = data[sym]
                else:
                    df = data
                df = df.dropna(subset=["Close"])
                if not df.empty:
                    out[sym] = df
            except Exception:
                continue
    return out


def send(text):
    if not TOKEN or not CHAT_ID:
        print("[Kein Telegram-Token gesetzt] " + text)
        return
    resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                         data={"chat_id": CHAT_ID, "text": text}, timeout=20)
    if not resp.ok:
        print("Telegram-Fehler:", resp.text)


def fmt(sym, sig, tf):
    icon = "🟢" if sig["side"] == "LONG" else "🔴"
    crypto = sym.endswith("-EUR")
    kind = "Krypto (Bitvavo)" if crypto else "Aktie"
    cur = "€" if sym.endswith(("-EUR", ".DE")) else "$"
    p = sig["price"]
    d = 2 if p >= 10 else 4 if p >= 0.1 else 6
    return (f"{icon} {sig['side']}-Signal: {sym}\n"
            f"⏱ Zeitebene: {tf}  |  {kind}\n"
            f"Score: {sig['score']}\n"
            f"Preis: {p:.{d}f} {cur}\n"
            f"Stop-Loss: {sig['stop']:.{d}f} {cur}\n"
            f"Ziel: {sig['target']:.{d}f} {cur}\n\n"
            + "\n".join("• " + w for w in sig["why"])
            + "\n\n⚠️ Keine Anlageberatung.")


def main():
    if os.environ.get("TEST_MESSAGE"):
        send("✅ Signalbot läuft und ist mit Telegram verbunden.")
        return
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    stocks = list(dict.fromkeys(get_sp500() + EXTRA_STOCKS))
    crypto = get_top_crypto()

    found = []
    for tf in TIMEFRAMES:
        data = {}
        data.update(download(stocks, tf))
        data.update(fetch_bitvavo(crypto, tf))
        print(f"[{tf}] Daten für {len(data)} von {len(stocks) + len(crypto)} Symbolen erhalten")

        for sym, df in data.items():
            try:
                sig = evaluate(add_indicators(df))
            except Exception as e:
                print(sym, tf, "Fehler:", e); continue
            if not sig:
                continue
            is_crypto = sym.endswith("-EUR")
            if sig["side"] == "SHORT" and not (SHORTS_CRYPTO if is_crypto else SHORTS_STOCKS):
                continue
            if state.get(f"{sym}|{tf}") == f"{sig['side']}@{sig['candle']}":
                continue
            found.append((sym, tf, sig))

    found.sort(key=lambda x: x[2]["score"], reverse=True)
    print(f"{len(found)} neue Signale, sende die besten {min(len(found), MAX_SIGNALS)}")
    for sym, tf, sig in found[:MAX_SIGNALS]:
        send(fmt(sym, sig, tf))
        state[f"{sym}|{tf}"] = f"{sig['side']}@{sig['candle']}"
        print("Signal:", sym, tf, sig["side"], sig["score"])
    STATE_FILE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    sys.exit(main())
