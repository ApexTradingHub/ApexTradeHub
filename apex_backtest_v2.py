"""
ApexScan – Backtest
===================
Simuliert den Scanner auf historischen Daten.

Ablauf:
  1. Lädt Daten für alle Ticker (Standard: 2 Jahre, bei --start/--end: entsprechend mehr)
  2. Simuliert den Scanner an jedem Handelstag (rollierendes Fenster)
  3. Prüft ob Target/Stop in den Folgetagen erreicht wurde
  4. Baut Equity-Kurve und speichert Ergebnisse

Aufruf:
  py apex_backtest.py                              # letzte 120 Handelstage
  py apex_backtest.py --days 60                    # letzte 60 Tage
  py apex_backtest.py --start 2023-01-01 --end 2024-12-31   # fixer Zeitraum
  py apex_backtest.py --start 2023-01-01 --end 2024-12-31 --top 100  # schneller
  py apex_backtest.py --top 50                     # nur Top-50 Ticker (schnell)
"""

import sys
import io
import json
import argparse
import contextlib
import warnings
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Phase B: yfinance fundamental catalysts
try:
    from apex_catalysts import (
        get_catalyst_data,
        derive_catalyst_signals,
        score_delta_for_catalyst_signals,
        prewarm_catalysts,
    )
    CATALYSTS_AVAILABLE = True
except ImportError:
    CATALYSTS_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================
# CONFIG
# =============================================================
US_TICKER_FILE = "us_tickers.txt"
EU_TICKER_FILE = "eu_tickers.txt"
BACKTEST_DAYS  = 120       # wie viele Handelstage rückwirkend testen
SCAN_EVERY     = 5         # nur jeden N-ten Tag scannen (spart Zeit)
TRADE_SIZE     = 200.0     # USD pro Trade (virtuell)
MAX_OPEN_TRADES = 5        # max gleichzeitig offene Positionen
RESULTS_FILE   = "apex_backtest_results.json"
CHART_FILE     = "apex_backtest_chart.png"

MIN_DATA_DAYS     = 100
MIN_PRICE         = 2
MIN_AVG_VOLUME    = 50_000
MIN_DOLLAR_VOLUME = 500_000

# Hard scan-level score gate per setup (must match ApexScan.py)
SCAN_MIN_SCORE = {"BREAKOUT": 70, "REVERSAL": 50}

HORIZON_DAYS = {
    "1-3 weeks":  15,
    "2-6 weeks":  30,
    "4-12 weeks": 60,
}

DEAD_TICKERS = {
    "TWTR","FB","SIVB","FRC","YHOO","GGP","CELG","RTN","STI","VIAB",
    "APC","RHT","AGN","ETFC","NBL","ALXN","ATVI","XLNX","DISH","DXC",
    "ABMD","NKTR","DWDP","BHF","DRE","ARNC","COTY","FBHS","WFM","BBBY",
    "LEH","FNM","FRE","CFC","WB","JDSU","NOVL","GENZ","MFE","EK",
}

# Data-driven blacklist (2+ trades, WR <=25%, neg cumulative PnL)
BAD_PERFORMERS = {
    "VRSK","ALGN","DDOG","OWL","APTV","FICO","CMC","KSS",
    "IFF","GT","FSLR","MKTX","AFG","JKHY","J","KMI",
}


# =============================================================
# UTILS
# =============================================================
@contextlib.contextmanager
def suppress_output():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def safe_float(x, default=0.0):
    try:
        return default if pd.isna(x) else float(x)
    except Exception:
        return default


def percent_change(current, past):
    if pd.isna(current) or pd.isna(past) or past == 0:
        return None
    return ((float(current) / float(past)) - 1) * 100


def load_tickers(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [t.strip().upper().replace(".", "-")
                    for t in f if t.strip()
                    and t.strip().upper() not in DEAD_TICKERS
                    and t.strip().upper() not in BAD_PERFORMERS]
    except FileNotFoundError:
        return []


def unique(items):
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# =============================================================
# INDICATORS
# =============================================================
def rsi_wilder(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def atr_series(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# =============================================================
# CATALYSTS (mirror of ApexScan.py compute_technical_catalysts)
# =============================================================
def compute_technical_catalysts(df, atr14_now, vol_ratio):
    out = {
        "pocket_pivot_recent": False,
        "volume_climax": False,
        "up_gap_pct": 0.0,
        "gap_signal": False,
        "vcp_strength": 0.0,
        "vcp_signal": False,
    }
    if df is None or len(df) < 21:
        return out
    closes = df["Close"].values
    vols   = df["Volume"].values
    opens  = df["Open"].values
    n = len(closes)
    for i in range(1, 6):
        idx = n - i
        if idx < 11:
            break
        if closes[idx] <= closes[idx-1]:
            continue
        sub_close      = closes[idx-10:idx]
        sub_vol        = vols[idx-10:idx]
        sub_close_prev = closes[idx-11:idx-1]
        down_mask = sub_close < sub_close_prev
        if down_mask.any():
            max_down_vol = sub_vol[down_mask].max()
            if vols[idx] > max_down_vol:
                out["pocket_pivot_recent"] = True
                break
    out["volume_climax"] = vol_ratio >= 3.0
    if n >= 2 and closes[-2] > 0:
        out["up_gap_pct"] = (opens[-1] - closes[-2]) / closes[-2] * 100
        out["gap_signal"] = out["up_gap_pct"] > 2.0
    if "ATR14" in df.columns and n >= 21:
        atr_old = df["ATR14"].iloc[-21]
        if not pd.isna(atr_old) and atr_old > 0 and atr14_now > 0:
            contraction = 1 - (atr14_now / atr_old)
            out["vcp_strength"] = max(0.0, min(contraction, 1.0))
            out["vcp_signal"] = out["vcp_strength"] >= 0.30
    return out



# =============================================================
# TARGET CALCULATION
# =============================================================
def find_target(df, buy_above, atr14, setup, risk_pct=5.0):
    """
    Multi-method price target:
    1. Swing highs, 2. Fibonacci 1.272/1.618, 3. Round levels, 4. ATH buffer
    Fallback: variable RR per setup based on actual stop distance.
    """
    import math
    MIN_UPSIDE  = 4.0
    MAX_UPSIDE  = 25.0
    RR_FALLBACK = {"BREAKOUT": 2.0, "PRE-ROCKET": 2.5, "POSITION": 3.0}

    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(highs)
    candidates = []

    # Method 1: Swing highs
    SWING_W   = 3
    lookback  = min(252, n - 1)
    start_idx = max(SWING_W, n - lookback)
    for i in range(start_idx, n - SWING_W):
        h = highs[i]
        if h <= buy_above: continue
        left  = highs[max(0, i - SWING_W) : i]
        right = highs[i + 1 : i + SWING_W + 1]
        if len(left) and len(right) and h > max(left) and h > max(right):
            candidates.append(round(h, 2))

    # Method 2: Fibonacci extensions
    recent     = min(60, n - 1)
    swing_low  = float(min(lows[-recent:]))
    swing_high = float(max(highs[-recent:]))
    if swing_high > swing_low:
        move = swing_high - swing_low
        for fib in [1.272, 1.618]:
            lvl = round(swing_low + move * fib, 2)
            if lvl > buy_above:
                candidates.append(lvl)

    # Method 3: Round price levels
    if buy_above > 500:   steps = [10, 25, 50, 100]
    elif buy_above > 100: steps = [5, 10, 25, 50]
    elif buy_above > 50:  steps = [2.5, 5, 10, 25]
    elif buy_above > 10:  steps = [1, 2.5, 5, 10]
    else:                 steps = [0.5, 1, 2.5, 5]
    for step in steps:
        base = math.ceil(buy_above / step) * step
        for i in range(6):
            lvl = round(base + i * step, 2)
            if lvl > buy_above:
                candidates.append(lvl)

    # Method 4: ATH levels
    ath = float(max(highs))
    for mult in [1.0, 1.05, 1.10]:
        lvl = round(ath * mult, 2)
        if lvl > buy_above:
            candidates.append(lvl)

    # Enforce minimum target = ATR-based floor
    min_target_mult = {"BREAKOUT": 2.5, "PRE-ROCKET": 3.0, "POSITION": 4.0}
    atr_min_target  = buy_above + (atr14 * min_target_mult.get(setup, 2.5))
    atr_min_upside  = (atr_min_target / buy_above - 1) * 100
    effective_min   = max(MIN_UPSIDE, atr_min_upside)

    target = None
    for c in sorted(set(candidates)):
        upside = (c / buy_above - 1) * 100
        if effective_min <= upside <= MAX_UPSIDE:
            target = c
            break

    if target is None:
        rr_mult   = RR_FALLBACK.get(setup, 2.0)
        rr_target = round(buy_above * (1 + (risk_pct * rr_mult) / 100), 2)
        target    = max(rr_target, round(atr_min_target, 2))

    return target

# =============================================================
# SCANNER LOGIC (identical to ApexScan_v2, applied to a slice)
# =============================================================
def scan_slice(ticker, df_slice, relax=0, risk_on=True, scan_date=None):
    """
    Runs scanner logic on df_slice (history up to scan date).
    scan_date: datetime/Timestamp, used for point-in-time catalyst filtering.
    """
    if df_slice is None or len(df_slice) < MIN_DATA_DAYS:
        return None

    df = df_slice.copy()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA150"] = df["Close"].rolling(150).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = rsi_wilder(df["Close"], 14)
    df["ATR14"] = atr_series(df, 14)
    ml, sl, hist = macd(df["Close"])
    df["MACD_HIST"] = hist

    df = df.dropna(subset=["MA20","MA50","VOL20","RSI14","ATR14","MACD_HIST"]).copy()
    if len(df) < 30:
        return None

    l    = df.iloc[-1]
    prev = df.iloc[-2]

    close  = safe_float(l["Close"])
    high   = safe_float(l["High"])
    vol    = safe_float(l["Volume"])
    vol20  = safe_float(l["VOL20"])
    ma20   = safe_float(l["MA20"])
    ma50   = safe_float(l["MA50"])
    ma150  = safe_float(l["MA150"])
    rsi14  = safe_float(l["RSI14"])
    atr14  = safe_float(l["ATR14"])
    mh     = safe_float(l["MACD_HIST"])
    mh_p   = safe_float(prev["MACD_HIST"])

    if close < MIN_PRICE: return None
    avg_dv = close * vol20
    if vol20 < MIN_AVG_VOLUME or avg_dv < MIN_DOLLAR_VOLUME: return None

    prev_20_high = safe_float(df["High"].iloc[-21:-1].max())
    prev_10_high = safe_float(df["High"].iloc[-11:-1].max())
    low_10       = safe_float(df["Low"].iloc[-10:].min())
    high_10      = safe_float(df["High"].iloc[-10:].max())

    # 52-week high filter
    high_52w = safe_float(df["High"].iloc[-252:].max()) if len(df) >= 252 else safe_float(df["High"].max())

    perf_20  = percent_change(close, df["Close"].iloc[-21])
    perf_60  = percent_change(close, df["Close"].iloc[-61]) if len(df) >= 61 else None
    perf_120 = percent_change(close, df["Close"].iloc[-121]) if len(df) >= 121 else None

    if perf_20 is None: return None
    if perf_60  is None: perf_60  = perf_20
    if perf_120 is None: perf_120 = perf_60

    vol_ratio  = vol / vol20 if vol20 > 0 else 0
    base_range = percent_change(high_10, low_10)
    atr_pct    = (atr14 / close) * 100 if close > 0 else 0

    if base_range is None: return None

    # Phase A catalysts
    catalysts = compute_technical_catalysts(df, atr14, vol_ratio)

    # Phase B catalysts (yfinance, point-in-time correct via scan_date)
    catalyst_signals = None
    if CATALYSTS_AVAILABLE:
        try:
            _raw = get_catalyst_data(ticker)
            catalyst_signals = derive_catalyst_signals(_raw, as_of_date=scan_date)
        except Exception:
            catalyst_signals = None

    # 52-week high context (used in score only, not hard filter)
    near_52w_high = high_52w > 0 and close >= high_52w * 0.85

    macd_bull = mh > 0 and mh > mh_p
    higher_tf = ma150 > 0 and close >= ma150 * 0.95

    # Per-setup RSI zones
    if relax == 0:
        rsi_breakout  = 45 <= rsi14 <= 68
        rsi_prerocket = 40 <= rsi14 <= 65
        rsi_zone      = rsi_breakout
    else:
        rsi_breakout  = 40 <= rsi14 <= 72
        rsi_prerocket = 38 <= rsi14 <= 72
        rsi_zone      = 38 <= rsi14 <= 75

    if relax == 0:
        trend_ok  = close > ma20 and ma20 > ma50 * 0.99
        near_high = close >= prev_20_high * 0.93
        base_ok   = base_range <= 8
    elif relax == 1:
        trend_ok  = close > ma50
        near_high = close >= prev_20_high * 0.88
        base_ok   = base_range <= 15
    else:
        trend_ok  = close > ma150 * 0.90 if ma150 > 0 else close > ma50
        near_high = close >= prev_20_high * 0.80
        base_ok   = base_range <= 25

    breakout_touch = high >= prev_20_high * 0.995
    vol_ok         = vol_ratio >= (0.7 if relax >= 1 else 1.0)
    expansion_vol  = vol_ratio >= 1.1
    momentum_ok    = perf_20 > 0 and perf_60 > 0

    # REVERSAL: check before hard exits
    pct_from_52w  = ((close - high_52w) / high_52w * 100) if high_52w > 0 else 0
    macd_turning  = mh > mh_p and mh_p < 0
    not_near_high = close < prev_20_high * 0.92
    reversal_setup = (
        -40 <= pct_from_52w <= -12 and
        25 <= rsi14 <= 48 and
        (macd_turning or (mh > mh_p)) and
        vol_ratio >= 1.1 and
        (ma150 == 0 or close >= ma150 * 0.85) and
        perf_20 >= -35 and
        close > MIN_PRICE * 3 and
        not_near_high
    )

    if not reversal_setup:
        if not trend_ok:   return None
        if not vol_ok:     return None
        if not rsi_zone:   return None
        if relax == 0 and not momentum_ok: return None
        if relax == 0 and not base_ok:     return None
    if atr_pct > 15: return None

    # BREAKOUT: max 1% below entry
    buy_above_prev   = prev_20_high * 1.002
    breakout_close   = close >= buy_above_prev * 0.99
    breakout_setup   = breakout_touch and higher_tf and rsi_breakout and breakout_close

    # PRE-ROCKET: DISABLED — low win rate (40%), high SL rate
    pre_rocket = False

    # POSITION: DISABLED — underperforms in current market
    position_setup = False

    if not (breakout_setup or pre_rocket or position_setup or reversal_setup):
        return None

    # REVERSAL entry/stop/target
    if reversal_setup and not breakout_setup and not pre_rocket and not position_setup:
        setup     = "REVERSAL"
        horizon   = "3-8 weeks"
        buy_above = round(close * 1.005, 2)
        stop      = round(low_10 - (atr14 * 0.5), 2)
        risk_pct  = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0:
            stop = buy_above * 0.97; risk_pct = 3.0
        elif risk_pct > 10.0:
            stop = buy_above - (atr14 * 1.5)
            risk_pct = (buy_above - stop) / buy_above * 100
            if risk_pct > 10.0:
                return None
        fib_target = round(close + (high_52w - close) * 0.50, 2)
        target     = fib_target

    else:
        buy_above = prev_20_high * 1.002
        stop_candidate = low_10 * 0.995
        atr_pct_val    = (atr14 / buy_above) * 100 if buy_above > 0 else 5.0
        min_dist_pct   = max(3.0, atr_pct_val * 1.5)
        multiplier     = 1.0 if (position_setup and not breakout_setup) else 1.5
        candidate_dist = ((buy_above - stop_candidate) / buy_above * 100 if buy_above > 0 else 99)

        if candidate_dist < min_dist_pct:
            stop = buy_above * (1 - min_dist_pct / 100)
        elif candidate_dist > 10.0:
            stop = buy_above - (atr14 * multiplier)
        else:
            stop = stop_candidate

        risk_pct = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0:
            stop = buy_above * 0.97; risk_pct = 3.0
        elif risk_pct > 10.0:
            return None

        if breakout_setup:
            setup = "BREAKOUT"; horizon = "1-3 weeks"
        elif position_setup and not breakout_setup:
            setup = "POSITION"; horizon = "4-12 weeks"
        else:
            setup = "PRE-ROCKET"; horizon = "2-6 weeks"

        target = find_target(df, buy_above, atr14, setup, risk_pct)

    upside_pct = ((target / buy_above) - 1) * 100
    rr = upside_pct / risk_pct if risk_pct > 0 else 0
    if rr < 1.2: return None

    # ---- Setup-specific scoring (must mirror ApexScan.py) ----
    if setup == "REVERSAL":
        score = 20.0
        if 28 <= rsi14 <= 35:   score += 15
        elif 35 < rsi14 <= 42:  score += 10
        elif 42 < rsi14 <= 45:  score += 5
        macd_turning_sc = mh > mh_p and mh_p < 0
        score += 15 if macd_turning_sc else (8 if macd_bull else 0)
        if vol_ratio >= 1.5:    score += 10
        elif vol_ratio >= 1.3:  score += 7
        elif vol_ratio >= 1.1:  score += 4
        dip = abs(min(perf_20, 0))
        if 15 <= dip <= 30:    score += 12
        elif 10 <= dip < 15:   score += 8
        elif dip > 30:         score += 5
        depth_52w = abs(pct_from_52w) if pct_from_52w < 0 else 0
        if 15 <= depth_52w <= 35: score += 10
        elif 35 < depth_52w <= 40: score += 6
        score += min(rr, 5.0) * 4
        if upside_pct >= 20:    score += 10
        elif upside_pct >= 12:  score += 6
        elif upside_pct >= 8:   score += 3
        # No risk-regime bonus — initial small-sample suggested risk_off edge
        # (n=24 had +8pp WR) but did not replicate at n=124 (40.3% vs 39.7%)
        # Phase A catalysts (REVERSAL subset)
        if catalysts["pocket_pivot_recent"]: score += 5
        if catalysts["volume_climax"]:       score += 5
    else:
        score = 0.0
        score += 20 if breakout_setup else (15 if position_setup else 10)
        score += 8  if higher_tf else 0
        score += 8  if expansion_vol else (4 if vol_ratio >= 1.0 else 0)
        score += 8  if macd_bull else 0
        score += 6  if 48 <= rsi14 <= 68 else (3 if rsi_zone else 0)
        score += min(max(perf_20, 0), 20)  * 0.8
        score += min(max(perf_60, 0), 35)  * 0.5
        score += min(vol_ratio, 3.0) * 4
        score += min(rr, 5.0) * 4
        # Phase A catalysts (BREAKOUT — full set)
        if catalysts["pocket_pivot_recent"]: score += 10
        if catalysts["volume_climax"]:       score += 5
        if catalysts["gap_signal"]:          score += 8
        if catalysts["vcp_signal"]:          score += 5

    # Phase B catalysts (earnings only — short/analyst skipped in backtest_mode)
    if catalyst_signals is not None:
        score += score_delta_for_catalyst_signals(catalyst_signals, setup,
                                                   backtest_mode=True)

    # Hard score gate (per setup)
    if score < SCAN_MIN_SCORE.get(setup, 0):
        return None

    return {
        "ticker":    ticker,
        "setup":     setup,
        "horizon":   horizon,
        "price":     round(close, 2),
        "buy_above": round(buy_above, 2),
        "stop":      round(stop, 2),
        "target":    round(target, 2),
        "risk_pct":  round(risk_pct, 2),
        "rr":        round(rr, 2),
        "score":     round(score, 1),
    }


# =============================================================
# TRADE OUTCOME
# =============================================================
def evaluate_outcome(ticker, full_df, scan_idx, signal):
    """
    First waits for buy_above to be triggered (High >= entry).
    Only then tracks stop/target. If trigger never reached -> no trade.
    Returns (exit_price, exit_reason, exit_day, trigger_day) or (None,)*4.
    """
    hold_days   = HORIZON_DAYS.get(signal["horizon"], 21)
    entry       = signal["buy_above"]
    tp          = signal["target"]
    sl          = signal["stop"]

    future = full_df.iloc[scan_idx + 1 : scan_idx + 1 + hold_days]
    if future.empty:
        return None, None, None, None

    trigger_day = None
    # Live behavior: BUY-above order is stale after ~3 days. Backtest's old
    # default was the full hold_days window which inflated trade count via
    # late triggers (data shows trigger_day>=5 had 23% WR). Cap at 3.
    MAX_TRIGGER_DAYS = 3

    for i, (_, row) in enumerate(future.iterrows()):
        try:
            o = safe_float(row["Open"])
            h = safe_float(row["High"])
            l = safe_float(row["Low"])
        except Exception:
            continue

        # Wait for entry trigger
        if trigger_day is None:
            if i >= MAX_TRIGGER_DAYS:
                return None, None, None, None  # signal expired
            if h >= entry:
                trigger_day = i
                # Same candle also hit stop -> unreliable, skip
                if l <= sl:
                    return None, None, None, None
            else:
                continue  # not triggered yet

        # Trade is live
        hit_tp = h >= tp
        hit_sl = l <= sl

        if hit_tp and hit_sl:
            ep     = tp if o >= entry else sl
            reason = "Take Profit" if o >= entry else "Stop Loss"
            return ep, reason, i + 1, trigger_day + 1
        elif hit_tp:
            return tp, "Take Profit", i + 1, trigger_day + 1
        elif hit_sl:
            return sl, "Stop Loss",   i + 1, trigger_day + 1

    if trigger_day is None:
        return None, None, None, None  # trigger never reached

    last_close = safe_float(future["Close"].iloc[-1])
    return last_close, "Time Exit", min(hold_days, len(future)), trigger_day + 1


# =============================================================
# MARKET REGIME (simplified, on slice)
# =============================================================
def slice_regime(spy_slice, qqq_slice):
    def reg(df):
        if df is None or len(df) < 50: return "UNKNOWN"
        # squeeze() handles MultiIndex columns from yfinance batch downloads
        c = df["Close"].squeeze() if hasattr(df["Close"], "squeeze") else df["Close"]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        c = c.dropna()
        if len(c) < 50: return "UNKNOWN"
        lc     = safe_float(c.iloc[-1])
        lma20  = safe_float(c.rolling(20).mean().iloc[-1])
        lma50  = safe_float(c.rolling(50).mean().iloc[-1])
        lma200 = safe_float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else lma50
        if lc > lma20 > lma50 > lma200: return "STRONG"
        if lc > lma50 and lma50 > lma200: return "OK"
        return "WEAK"
    sr = reg(spy_slice)
    qr = reg(qqq_slice)
    return (sr in {"STRONG","OK"}) and (qr in {"STRONG","OK"})


# =============================================================
# MAIN BACKTEST
# =============================================================
def run_backtest(tickers, bt_days=None, top_n=None, start_date=None, end_date=None):
    # Determine download range
    if start_date and end_date:
        dl_start = (start_date - timedelta(days=400)).strftime("%Y-%m-%d")
        dl_end   = (end_date   + timedelta(days=90)).strftime("%Y-%m-%d")
        mode_str = f"{start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}"
        use_period = False
    else:
        use_period = True
        mode_str = f"letzte {bt_days} Handelstage"

    print(f"\nDownloading history for {len(tickers)} tickers ({mode_str})...")

    all_data = {}

    for batch in tqdm(list(chunked(tickers, 80)), desc="Downloading"):
        try:
            with suppress_output():
                if use_period:
                    raw = yf.download(
                        " ".join(batch), period="2y",
                        auto_adjust=True, progress=False,
                        threads=True, group_by="ticker"
                    )
                else:
                    raw = yf.download(
                        " ".join(batch), start=dl_start, end=dl_end,
                        auto_adjust=True, progress=False,
                        threads=True, group_by="ticker"
                    )
            for t in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0).tolist()
                        lvl1 = raw.columns.get_level_values(1).tolist()
                        if t in lvl0:   df = raw[t].copy()
                        elif t in lvl1: df = raw.xs(t, axis=1, level=1).copy()
                        else: continue
                    else:
                        df = raw.copy()
                    df = df[["Open","High","Low","Close","Volume"]].dropna()
                    if len(df) >= MIN_DATA_DAYS:
                        all_data[t] = df
                except Exception:
                    pass
        except Exception:
            pass

    print(f"Usable tickers: {len(all_data)}")

    if top_n:
        all_data = dict(list(all_data.items())[:top_n])
        print(f"Capped to top {top_n} tickers for speed")

    # Load SPY/QQQ for regime
    spy_full = qqq_full = None
    try:
        with suppress_output():
            if use_period:
                spy_full = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
                qqq_full = yf.download("QQQ", period="2y", auto_adjust=True, progress=False)
            else:
                spy_full = yf.download("SPY", start=dl_start, end=dl_end, auto_adjust=True, progress=False)
                qqq_full = yf.download("QQQ", start=dl_start, end=dl_end, auto_adjust=True, progress=False)
    except Exception:
        pass

    # Determine scan dates
    ref_df = spy_full if spy_full is not None and not spy_full.empty else list(all_data.values())[0]

    if start_date and end_date:
        mask = (ref_df.index.normalize() >= pd.Timestamp(start_date)) & \
               (ref_df.index.normalize() <= pd.Timestamp(end_date))
        all_dates = ref_df.index[mask]
    else:
        all_dates = ref_df.index[-bt_days:]

    scan_dates = all_dates[::SCAN_EVERY]
    # Pre-warm catalyst cache before scan loop (Phase B)
    if CATALYSTS_AVAILABLE:
        try:
            prewarm_catalysts(list(all_data.keys()), show_progress=True)
        except Exception as e:
            print(f"Catalyst pre-warm failed: {e} — continuing without")

    print(f"Scanning {len(scan_dates)} dates ({mode_str})...\n")

    # Pre-compute regime for every scan date to avoid per-loop tz issues
    # Normalize SPY/QQQ index once to date-only for reliable lookup
    if spy_full is not None and not spy_full.empty:
        spy_norm = spy_full.copy()
        qqq_norm = qqq_full.copy() if qqq_full is not None and not qqq_full.empty else spy_full.copy()
        spy_norm.index = spy_norm.index.normalize()
        qqq_norm.index = qqq_norm.index.normalize()
    else:
        spy_norm = qqq_norm = None

    regime_cache = {}
    if spy_norm is not None:
        for sd in scan_dates:
            sd_norm = pd.Timestamp(sd).normalize()
            mask = spy_norm.index <= sd_norm
            s_slice = spy_norm[mask]
            q_slice = qqq_norm[mask]
            regime_cache[sd_norm] = slice_regime(
                s_slice if len(s_slice) >= 50 else None,
                q_slice if len(q_slice) >= 50 else None,
            )

    trades     = []
    equity     = 0.0
    eq_curve   = [0.0]
    open_pos   = {}

    for scan_date in tqdm(scan_dates, desc="Backtesting"):
        sd_norm = pd.Timestamp(scan_date).normalize()
        risk_on = regime_cache.get(sd_norm, True)
        # Ensure risk_on is stored as proper bool in trades
        risk_on = bool(risk_on)

        signals_today = []

        for ticker, full_df in all_data.items():
            if ticker not in full_df.index and scan_date not in full_df.index:
                pass
            try:
                idx = full_df.index.get_loc(scan_date)
            except KeyError:
                continue

            if idx < MIN_DATA_DAYS:
                continue

            df_slice = full_df.iloc[:idx+1]

            # Strict-only: matches Live behavior where MIN_SIGNALS=1 is
            # always met by relax=0 across 3000+ tickers daily, so relax>0
            # paths effectively never run live. Per-ticker relax fallback
            # would mask the effect of any tightening change.
            sig = scan_slice(ticker, df_slice, relax=0, risk_on=risk_on,
                             scan_date=scan_date)

            if sig is None:
                continue

            # Skip BREAKOUT in RISK-OFF with weak score (REVERSAL exempted —
            # data shows REVERSAL has +8pp WR in risk_off vs risk_on)
            if not risk_on and sig.get("setup") != "REVERSAL" and sig["score"] < 50:
                continue

            # Skip if already in open position
            if ticker in open_pos:
                continue

            sig["scan_date"] = scan_date.strftime("%Y-%m-%d")
            sig["risk_on"]   = risk_on
            sig["full_idx"]  = idx
            sig["full_df_ref"] = ticker
            signals_today.append(sig)

        # Sort by score, take top MAX_OPEN_TRADES
        signals_today.sort(key=lambda x: x["score"], reverse=True)
        for sig in signals_today[:MAX_OPEN_TRADES]:
            ticker  = sig["ticker"]
            full_df = all_data[ticker]
            idx     = sig["full_idx"]

            ep, reason, eday, tday = evaluate_outcome(ticker, full_df, idx, sig)

            if ep is None:
                continue

            entry   = sig["buy_above"]
            pnl_pct = (ep - entry) / entry * 100
            pnl_usd = TRADE_SIZE * pnl_pct / 100
            equity += pnl_usd
            eq_curve.append(round(equity, 2))

            trades.append({
                "date":        sig["scan_date"],
                "ticker":      ticker,
                "setup":       sig["setup"],
                "entry":       round(entry, 2),
                "stop":        round(sig["stop"], 2),
                "target":      round(sig["target"], 2),
                "exit_price":  round(ep, 2),
                "exit_reason": reason,
                "exit_day":    eday,
                "trigger_day": tday,
                "pnl_pct":     round(pnl_pct, 2),
                "pnl_usd":     round(pnl_usd, 2),
                "rr":          sig["rr"],
                "score":       sig["score"],
                "risk_on":     sig["risk_on"],
                "equity":      round(equity, 2),
            })

    return trades, eq_curve


# =============================================================
# CHART
# =============================================================
def build_chart(trades, eq_curve):
    if not trades:
        print("Keine Trades für Chart.")
        return

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    tc = "#e0e0e0"; gc = "#1e2030"; green = "#4caf50"; red = "#ef5350"; blue = "#4a9eff"

    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = len(wins) / len(trades) * 100 if trades else 0
    avg_w  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(t["pnl_usd"] for t in wins) / sum(t["pnl_usd"] for t in losses)) if losses else 999

    # 1. Equity curve (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f1117")
    ax1.plot(eq_curve, color=blue, lw=1.5)
    ax1.fill_between(range(len(eq_curve)), eq_curve, alpha=0.12, color=blue)
    ax1.axhline(0, color=gc, lw=0.8)
    ax1.set_title("ApexScan – Backtest Equity Curve", color=tc, fontsize=13)
    ax1.set_xlabel("Closed Trades", color=tc, fontsize=10)
    ax1.set_ylabel("Equity ($)", color=tc, fontsize=10)
    ax1.tick_params(colors=tc); ax1.grid(True, color=gc, lw=0.5)
    for sp in ax1.spines.values(): sp.set_edgecolor(gc)
    final = eq_curve[-1] if eq_curve else 0
    ax1.annotate(f"  ${final:.2f}", xy=(len(eq_curve)-1, final),
                 color=green if final >= 0 else red, fontsize=10, fontweight="bold")

    # 2. PnL bars
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f1117")
    pnls   = [t["pnl_pct"] for t in trades]
    colors = [green if p > 0 else red for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, width=0.7)
    ax2.axhline(0, color=tc, lw=0.5)
    ax2.set_title("PnL per trade (%)", color=tc, fontsize=11)
    ax2.tick_params(colors=tc); ax2.grid(True, axis="y", color=gc, lw=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor(gc)

    # 3. Exit reasons pie
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f1117")
    reasons = Counter(t["exit_reason"] for t in trades)
    clrs    = [green if "Profit" in k else (red if "Loss" in k else "#888") for k in reasons]
    wedges, texts, autotexts = ax3.pie(
        reasons.values(), labels=reasons.keys(),
        colors=clrs, autopct="%1.0f%%", startangle=90,
        textprops={"color": tc, "fontsize": 10}
    )
    for at in autotexts: at.set_color("#0f1117"); at.set_fontsize(9)
    ax3.set_title("Exit reasons", color=tc, fontsize=11)

    # 4. Stats
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#0f1117"); ax4.axis("off")
    tp_c = sum(1 for t in trades if t["exit_reason"] == "Take Profit")
    sl_c = sum(1 for t in trades if t["exit_reason"] == "Stop Loss")
    te_c = sum(1 for t in trades if t["exit_reason"] == "Time Exit")
    setups = Counter(t["setup"] for t in trades)

    stats = [
        ("Trades",         str(len(trades))),
        ("Win rate",       f"{wr:.1f}%"),
        ("Avg win",        f"+{avg_w:.2f}%"),
        ("Avg loss",       f"{avg_l:.2f}%"),
        ("Profit factor",  f"{profit_factor:.2f}"),
        ("Take Profits",   str(tp_c)),
        ("Stop Losses",    str(sl_c)),
        ("Time Exits",     str(te_c)),
        ("BREAKOUT",       str(setups.get("BREAKOUT", 0))),
        ("PRE-ROCKET",     str(setups.get("PRE-ROCKET", 0))),
        ("POSITION",       str(setups.get("POSITION", 0))),
        ("Final equity",   f"${final:.2f}"),
    ]

    ax4.text(0.5, 1.02, "Backtest stats", color=tc, fontsize=11,
             ha="center", va="top", transform=ax4.transAxes, fontweight="bold")
    y = 0.92
    for label, val in stats:
        col = green if ("+" in val or (val.startswith("$") and final >= 0 and label == "Final equity")) \
              else (red if ("-" in val or (val.startswith("$") and final < 0)) else tc)
        ax4.text(0.05, y, label, color="#777", fontsize=9.5, transform=ax4.transAxes)
        ax4.text(0.65, y, val,   color=col,    fontsize=9.5, transform=ax4.transAxes, fontweight="bold")
        y -= 0.08

    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart gespeichert: {CHART_FILE}")


# =============================================================
# MAIN
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="ApexScan Backtest")
    parser.add_argument("--days",  type=int, default=BACKTEST_DAYS,
                        help=f"Handelstage zurück (default: {BACKTEST_DAYS}), ignoriert wenn --start/--end gesetzt")
    parser.add_argument("--start", type=str, default=None,
                        help="Startdatum YYYY-MM-DD (z.B. 2023-01-01)")
    parser.add_argument("--end",   type=str, default=None,
                        help="Enddatum YYYY-MM-DD (z.B. 2024-12-31)")
    parser.add_argument("--top",   type=int, default=None,
                        help="Nur erste N Ticker (schneller für Tests)")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date   = datetime.strptime(args.end,   "%Y-%m-%d") if args.end   else None

    us = load_tickers(US_TICKER_FILE)
    eu = load_tickers(EU_TICKER_FILE)
    tickers = unique(us + eu)
    print(f"Tickers: {len(us)} US + {len(eu)} EU = {len(tickers)} total")

    trades, eq_curve = run_backtest(
        tickers,
        bt_days=args.days,
        top_n=args.top,
        start_date=start_date,
        end_date=end_date,
    )

    if not trades:
        print("\nKeine Trades im Backtest gefunden.")
        return

    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    wr     = len(wins) / len(trades) * 100
    pf     = abs(sum(t["pnl_usd"] for t in wins) / sum(t["pnl_usd"] for t in losses)) if losses else 999

    period_str = f"{args.start} bis {args.end}" if args.start else f"{args.days} Handelstage"
    print(f"\n{'='*60}")
    print(f"  BACKTEST ERGEBNIS  ({period_str})")
    print(f"{'='*60}")
    print(f"  Trades total  : {len(trades)}")
    print(f"  Win rate      : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win       : +{sum(t['pnl_pct'] for t in wins)/len(wins):.2f}%" if wins else "  Avg win  : n/a")
    print(f"  Avg loss      : {sum(t['pnl_pct'] for t in losses)/len(losses):.2f}%" if losses else "  Avg loss : n/a")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Final equity  : ${eq_curve[-1]:.2f}")
    print(f"{'='*60}\n")

    print("Letzte 10 Trades:")
    for t in trades[-10:]:
        print(f"  {t['date']} | {t['ticker']:6} | {t['setup']:10} | "
              f"{t['pnl_pct']:>+6.2f}% | {t['exit_reason']:11} D+{t['exit_day']} | "
              f"Equity: ${t['equity']:.2f}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {RESULTS_FILE}")

    build_chart(trades, eq_curve)


if __name__ == "__main__":
    main()