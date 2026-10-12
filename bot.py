"""Telegram-Signalbot für Aktien & Krypto (Daten via yfinance, kein API-Key nötig)."""
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------- Einstellungen ----------
EXTRA_STOCKS = ["SAP.DE", "SIE.DE", "ALV.DE", "RHM.DE"]   # zusätzlich zum S&P 500
CRYPTO_TOP_N = 50    # Top-N Kryptos nach Marktkapitalisierung (CoinGecko)
BITVAVO = "https://api.bitvavo.com/v2"   # Krypto-Kurse direkt von Bitvavo (EUR-Märkte)
SHORTS_STOCKS = True   # True = bei Aktien auch SHORT-Signale senden
SHORTS_CRYPTO = True    # True = bei Krypto auch SHORT-/Verkaufs-Signale senden
MAX_SIGNALS = 15     # maximal so viele Nachrichten pro Lauf (beste Scores zuerst)
FALLBACK_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "AVGO"]
STOCK_TF = ["1h", "1d"]    # Zeitebenen für Aktien (Yahoo: 15m, 30m, 1h, 1d)
CRYPTO_TF = ["1h", "4h"]   # Zeitebenen für Krypto (Bitvavo: 15m, 30m, 1h, 4h, 1d)
PERIODS = {"15m": "60d", "30m": "60d", "1h": "60d", "1d": "2y"}   # Historie für die Indikatoren (EMA200 braucht >210 Kerzen)
MIN_SCORE = 3        # Mindestpunktzahl für ein Signal
ATR_STOP = 1.5       # Stop-Loss = 1.5 x ATR
ATR_TARGET = 3.0     # Ziel = 3 x ATR (CRV 1:2)
STATE_FILE = Path("state.json")
# --- Preisstruktur-Filter ---
USE_SR = True          # Unterstützung/Widerstand berücksichtigen
SR_LOOKBACK = 50       # Hoch/Tief der letzten 50 Kerzen = Widerstand/Unterstützung
SR_NEAR_ATR = 1.0      # "nah" = innerhalb von 1 x ATR
USE_CANDLES = True     # Kerzenmuster (Hammer, Shooting Star, Engulfing) als Zusatzpunkt
BTC_FILTER = True      # Krypto-LONGs nur, wenn Bitcoin nicht im Abwärtstrend ist
# --- Trend-Filter ---
HTF_FILTER = True      # Signale nur in Richtung des Trends der nächsthöheren Zeitebene
HTF_STOCK = {"15m": "1h", "30m": "1h", "1h": "1d"}                # Aktien: Signal-Ebene -> höhere Ebene
HTF_CRYPTO = {"15m": "1h", "30m": "1h", "1h": "4h", "4h": "1d"}   # Krypto
USE_ADX = True         # ADX: Trendstärke
ADX_MIN = 20           # darunter = Seitwärtsphase: reine Trend-Signale (MACD/Bollinger) werden verworfen
ADX_STRONG = 25        # ab hier +1 Punkt für Trend-Signale
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

    # Unterstützung / Widerstand: Hoch/Tief der letzten Kerzen (ohne die aktuelle)
    df["res"] = df["High"].shift(1).rolling(SR_LOOKBACK).max()
    df["sup"] = df["Low"].shift(1).rolling(SR_LOOKBACK).min()

    # ADX (Trendstärke, Wilder)
    up_move, down_move = df["High"].diff(), -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Kerzenmuster
    o, h, l = df["Open"], df["High"], df["Low"]
    body, rng = (c - o).abs(), h - l
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    hammer = (rng > 0) & (lower >= 2 * body) & (upper <= 0.3 * rng)
    star = (rng > 0) & (upper >= 2 * body) & (lower <= 0.3 * rng)
    bull_eng = (c.shift() < o.shift()) & (c > o) & (o <= c.shift()) & (c >= o.shift())
    bear_eng = (c.shift() > o.shift()) & (c < o) & (o >= c.shift()) & (c <= o.shift())
    df["pat_bull"] = np.select([bull_eng, hammer], ["Bullish Engulfing", "Hammer"], "")
    df["pat_bear"] = np.select([bear_eng, star], ["Bearish Engulfing", "Shooting Star"], "")
    return df


def evaluate(df):
    """Bewertet die letzte ABGESCHLOSSENE Kerze. Gibt Signal-Dict oder None zurück."""
    if len(df) < 210:
        return None
    return signal_from_rows(df.iloc[-2], df.iloc[-3], str(df.index[-2]))


def signal_from_rows(r, p, candle):
    """Bewertet eine Kerze r (p = Vorgängerkerze). Wird auch vom Backtest genutzt."""
    vol_ok = pd.notna(r.vol_ratio) and r.vol_ratio > 1.5

    for side in ("LONG", "SHORT"):
        long = side == "LONG"
        score, why = 0, []
        trigger = trend_trig = rsi_trig = False

        if long and r.Close > r.ema200 and r.ema50 > r.ema200:
            score += 1; why.append("Aufwärtstrend (Kurs & EMA50 > EMA200)")
        if not long and r.Close < r.ema200 and r.ema50 < r.ema200:
            score += 1; why.append("Abwärtstrend (Kurs & EMA50 < EMA200)")

        if long and p.macd_h <= 0 < r.macd_h:
            score += 2; trigger = trend_trig = True; why.append("MACD bullisches Kreuz")
        if not long and p.macd_h >= 0 > r.macd_h:
            score += 2; trigger = trend_trig = True; why.append("MACD bärisches Kreuz")

        if long and p.rsi < 30 <= r.rsi:
            score += 2; trigger = rsi_trig = True; why.append(f"RSI dreht aus überverkauft ({r.rsi:.0f})")
        if not long and p.rsi > 70 >= r.rsi:
            score += 2; trigger = rsi_trig = True; why.append(f"RSI dreht aus überkauft ({r.rsi:.0f})")

        if long and r.Close > r.bb_up and p.Close <= p.bb_up:
            score += 2; trigger = trend_trig = True; why.append("Ausbruch über oberes Bollinger-Band")
        if not long and r.Close < r.bb_lo and p.Close >= p.bb_lo:
            score += 2; trigger = trend_trig = True; why.append("Bruch des unteren Bollinger-Bands")

        if vol_ok and trigger:
            score += 1; why.append(f"Volumen {r.vol_ratio:.1f}x über Schnitt")

        if not trigger:
            continue
        price, atr = float(r.Close), float(r.atr)

        if USE_ADX and pd.notna(r.adx):
            if r.adx < ADX_MIN and not rsi_trig:
                continue                              # Seitwärtsphase: Trend-Signale unzuverlässig
            if r.adx >= ADX_STRONG and trend_trig:
                score += 1; why.append(f"Starker Trend (ADX {r.adx:.0f})")

        if USE_SR and pd.notna(r.res) and pd.notna(r.sup) and atr > 0:
            near = SR_NEAR_ATR * atr
            if long:
                if 0 < r.res - price < near:
                    continue                      # Widerstand direkt über dem Kurs
                if 0 <= price - r.sup <= near:
                    score += 1; why.append("Nähe zur Unterstützung")
            else:
                if 0 < price - r.sup < near:
                    continue                      # Unterstützung direkt unter dem Kurs
                if 0 <= r.res - price <= near:
                    score += 1; why.append("Nähe zum Widerstand")

        if USE_CANDLES:
            pat = r.pat_bull if long else r.pat_bear
            if pat:
                score += 1; why.append(f"Kerzenmuster: {pat}")

        if score >= MIN_SCORE:
            sign = 1 if long else -1
            return {
                "side": side, "score": score, "why": why, "price": price,
                "res": float(r.res) if pd.notna(r.res) else None,
                "sup": float(r.sup) if pd.notna(r.sup) else None,
                "stop": price - sign * ATR_STOP * atr,
                "target": price + sign * ATR_TARGET * atr,
                "candle": candle,
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


def download(symbols, interval, chunk_size=50, period=None):
    """Lädt Kerzen in Blöcken (viel schneller als einzeln). Gibt {symbol: DataFrame} zurück."""
    out = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=period or PERIODS.get(interval, "60d"), interval=interval, group_by="ticker",
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


def trend_map(data):
    """Trend je Symbol auf einer Zeitebene: Schlusskurs der letzten abgeschlossenen Kerze über/unter EMA50."""
    out = {}
    for sym, df in data.items():
        try:
            if len(df) < 60:
                continue
            c = df["Close"]
            out[sym] = "up" if c.iloc[-2] > ema(c, 50).iloc[-2] else "down"
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
            f"Ziel: {sig['target']:.{d}f} {cur}\n"
            + (f"Widerstand: {sig['res']:.{d}f} {cur} | Unterstützung: {sig['sup']:.{d}f} {cur}\n"
               if sig.get("res") and sig.get("sup") else "")
            + "\n"
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
    jobs = [(tf, "stocks") for tf in STOCK_TF] + [(tf, "crypto") for tf in CRYPTO_TF]
    cache = {}

    def get_data(kind, tf):
        if (kind, tf) not in cache:
            cache[(kind, tf)] = download(stocks, tf) if kind == "stocks" else fetch_bitvavo(crypto, tf)
        return cache[(kind, tf)]

    for tf, kind in jobs:
        data = get_data(kind, tf)
        total = len(stocks) if kind == "stocks" else len(crypto)
        print(f"[{kind} {tf}] Daten für {len(data)} von {total} Symbolen erhalten")

        htf_tf = (HTF_STOCK if kind == "stocks" else HTF_CRYPTO).get(tf) if HTF_FILTER else None
        htf = trend_map(get_data(kind, htf_tf)) if htf_tf else {}
        if htf_tf:
            print(f"  Trendfilter {htf_tf}: {len(htf)} Symbole")

        btc_down = False
        if kind == "crypto" and BTC_FILTER and "BTC-EUR" in data:
            try:
                b = add_indicators(data["BTC-EUR"].copy()).iloc[-2]
                btc_down = bool(b.Close < b.ema200 and b.ema50 < b.ema200)
                print(f"BTC-Trend ({tf}): {'Abwärtstrend, Altcoin-LONGs gefiltert' if btc_down else 'kein Abwärtstrend'}")
            except Exception as e:
                print("BTC-Filter nicht berechenbar:", e)

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
            if btc_down and sig["side"] == "LONG" and sym != "BTC-EUR":
                continue
            t = htf.get(sym)
            if t:
                if (sig["side"] == "LONG") != (t == "up"):
                    continue                          # gegen den höheren Trend
                sig["why"].append(f"Höherer Trend ({htf_tf}): {'aufwärts' if t == 'up' else 'abwärts'}")
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
