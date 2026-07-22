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
SCAN_MIN_SCORE = {
    "BREAKOUT":      70,
    "VCP":           65,
    "SHORT_SQUEEZE": 60,
    "STAGE_2":       55,
}

# RSI-Range overrides via CLI (set in main() from --bo-rsi-min / --bo-rsi-max)
BO_RSI_MIN_OVERRIDE = None
BO_RSI_MAX_OVERRIDE = None
RESULTS_FILE_OVERRIDE = None
# Measurement: force a relax level for ALL tickers (default 0 = strict, live-historic).
# Set to 1 to measure the quality of relaxed signals (Telegram-gate study).
FORCE_RELAX = 0
# Restrict scan to a single setup for clean isolated measurement (e.g. "MEAN_REVERSION").
# None = all setups compete normally.
ONLY_SETUP = None
# MEAN_REVERSION oversold threshold (tuning knob via --mr-rsi-max).
MR_RSI_MAX = 38
# BREAKOUT relax=0 base_range cap (tuning knob via --bo-base-max). Default 22 (validated 2026-05-22;
# the 19.4 tightening to 8 hurt BREAKOUT: base<=22 gave WR 52.1%/PF 1.68 vs 48.4%/1.17). Others <=8.
BO_BASE_MAX = 22
# VCP ATR-contraction threshold (tuning knob via --vcp-atr-contraction). Default 0.20 (validated
# 2026-05-28: 0.30 fired ~0; 0.20 gave n=9 WR 88.9% PF 7.16 over 2yr). Mirrors live.
VCP_ATR_CONTRACTION = 0.20
# SHORT_SQUEEZE min short interest (tuning knob via --sq-short-min). Default 15.0%.
SQ_SHORT_MIN = 15.0
# Catalyst-Free Elite Penalty (Test 2026-06-04, opt-in via --catalyst-free-elite-penalty).
# Hypothese: BREAKOUT-Signale mit Score>=100 OHNE positiven Catalyst (PP, Gap>=2%,
# EarningsBeat) sind techn. Extremes ohne Bestätigung -> historisch im 100+ Bucket: 4 von 8
# Losern (FLS/PENN/HP/S) hatten 0 Catalysts. Default 0 = off, kein Verhaltens-Change.
CATALYST_FREE_ELITE_PENALTY = 0.0

# Score-Realign (Test 2026-06-11, opt-in via --score-realign).
# Aligniert BREAKOUT-Score an gemessene Live-WR-Kohorten (n=24-27 MED):
#   - RSI-Zone 48-68 -> 48-72 (RSI>=70 zeigt 75% WR n=12, addet auch Signale -> Count-Schutz)
#   - perf_120 0-25 Bucket: milde Penalty (-3) statt EMERGING +5
#     (gemessen 44% WR n=27 = groesstes Loser-Bucket; mild = re-rank statt gate-out)
#   - perf_120 25-50 Bucket: bleibt hoch (+15, gemessen 71% WR n=24)
# Analyst +3->0 und Winrate n>=4 sind NICHT hier (nicht backtestbar: analyst skip in
# backtest_mode, winrate braucht forward-data) -> separate Live-Validierung.
SCORE_REALIGN = False

# Score-Rebuild Hebel B (2026-06-19): Extension-Penalty mit Catalyst-Carve-Out.
# Zielt aufs ASML-Fade-Profil (extended + schwache Bestaetigung + KEIN starker Catalyst).
# Carve-Out schuetzt Semi/AI-Capex-Winner (what_to_replicate). Siehe SCORE_REBUILD_STRATEGY.md.
SCORE_REBUILD = False

# SCORE_V2 Stufe 2 (2026-07-11, opt-in via --score-v2): LogReg-Ranking aus
# score_v2_model.json (Stufe-1-Fit, Train <=2026-05-31). REINES RE-RANKING der
# Pick-Stufe — Signal-Generierung + Gates laufen unveraendert auf dem Original-Score.
# short_pct-Feature fehlt im Backtest -> Train-Mean (standardisiert 0, neutral).
SCORE_V2 = False
SCORE_V2_MODEL = None

# Sweet-Spot-Band aufs Pick-Ranking (opt-in via --pick-band). Reines Re-Ranking.
PICK_BAND = None  # z.B. (90, 120) wenn aktiv


def _score_v2_prob(sig):
    """LogReg-Wahrscheinlichkeit aus dem frozen Stufe-1-Modell. Within-Day-Sortierung
    nach prob = identisches Ranking wie Tages-Perzentil (monotone Transformation)."""
    import math as _math
    m = SCORE_V2_MODEL
    mu, sd, w, b = m["mu"], m["sd"], m["weights"], m["bias"]
    x = [
        float(sig.get("rsi") or 50),
        1.0 if sig.get("macd_bull") else 0.0,
        min(float(sig.get("vol_ratio") or 1), 5.0),
        min(float(sig.get("rr") or 2), 5.0),
        min(max(float(sig.get("perf_20") or 0), -20), 40),
        min(max(float(sig.get("perf_60") or 0), -30), 60),
        min(max(float(sig.get("perf_120") or 0), -40), 100),
        min(float(sig.get("base_range") or 15), 40),
        float(sig.get("movement_bonus") or 0),
        float(sig.get("closing_strength") or 0.5),
        1.0 if sig.get("cat_pocket_pivot") else 0.0,
        1.0 if sig.get("cat_vol_climax") else 0.0,
        min(max(float(sig.get("cat_gap_pct") or 0), -5), 10),
        mu[13],   # short_pct: im Backtest nicht verfuegbar -> neutral (Train-Mean)
    ]
    z = b + sum(wj * (xi - mui) / sdi for wj, xi, mui, sdi in zip(w, x, mu, sd))
    return 1 / (1 + _math.exp(-max(min(z, 30), -30)))
EXT_PENALTY   = 12.0   # Penalty-Hoehe (Sweep 2026-06-20: -12 = perfekte Monotonie -0pp, alle Signale)

HORIZON_DAYS = {
    "1-3 weeks":   15,   # BREAKOUT
    "2-4 weeks":   20,   # SHORT_SQUEEZE
    "2-6 weeks":   30,   # legacy
    "4-8 weeks":   40,   # VCP
    "4-12 weeks":  60,   # legacy
    "8-16 weeks":  80,   # STAGE_2
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
    "TSM",  # 2026-07-15: WR 25%/n=4, kum -5.2% (ASML NICHT: WR 40%/kum +5%, erfuellt nicht)
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


# 2026-07-11 AP4-Fix (BACKLOG #18): Punkte nur bei Class-Shares ersetzen,
# Boersen-Suffixe behalten (SAP.DE). Gleiche Logik wie ApexScan.normalize_ticker.
EXCHANGE_SUFFIXES = {"DE", "PA", "AS", "SW", "L", "MC", "MI"}


def _normalize_ticker(t):
    t = str(t).strip().upper()
    if "." in t:
        head, _, tail = t.rpartition(".")
        if tail in EXCHANGE_SUFFIXES:
            return head.replace(".", "-") + "." + tail
        return t.replace(".", "-")
    return t


def load_tickers(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [_normalize_ticker(t)
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
# PHASE G SETUPS (mirror of ApexScan.py)
# =============================================================
def detect_vcp_setup(df, atr14_now, vol_ratio, ma50, ma150, close, high, atr_pct):
    n = len(df)
    if n < 100: return None
    if not (close > ma150 > 0 and ma50 > ma150): return None
    if "ATR14" not in df.columns: return None
    atr_old = df["ATR14"].iloc[-25]
    if pd.isna(atr_old) or atr_old <= 0: return None
    contraction = 1 - (atr14_now / atr_old)
    if contraction < VCP_ATR_CONTRACTION: return None
    last20_high = df["High"].iloc[-21:-1].max()
    last20_low  = df["Low"].iloc[-21:-1].min()
    if last20_low <= 0: return None
    base_range_pct = (last20_high / last20_low - 1) * 100
    if base_range_pct > 12: return None
    last10_high = df["High"].iloc[-11:-1].max()
    breakout_level = last10_high * 1.002
    if high < breakout_level * 0.998: return None
    if vol_ratio < 1.2: return None
    if atr_pct > 6.0: return None
    return {"breakout_level": breakout_level, "base_low": last20_low,
            "contraction_pct": round(contraction*100, 1),
            "base_range_pct": round(base_range_pct, 1)}


def detect_short_squeeze_setup(df, short_pct, ma20, ma50, close, vol_ratio, rsi14):
    if short_pct is None or short_pct < SQ_SHORT_MIN: return None
    if not (close > ma20 and ma20 > ma50): return None
    if len(df) < 6: return None
    closes = df["Close"].values
    perf_5d = (closes[-1] / closes[-6] - 1) * 100
    if perf_5d < 3.0: return None
    if vol_ratio < 1.3: return None
    if not (50 <= rsi14 <= 78): return None
    return {"short_pct": round(short_pct, 1), "perf_5d": round(perf_5d, 1),
            "breakout_level": df["High"].iloc[-11:-1].max() * 1.003}


def detect_stage2_transition(df, ma150_series, close):
    n = len(df)
    if n < 252: return None
    last180_close = df["Close"].iloc[-180:]
    base_high = float(last180_close.max())
    base_low  = float(last180_close.min())
    if base_low <= 0 or (base_high / base_low) > 1.4: return None
    if ma150_series is None: return None
    try:
        ma150_now    = safe_float(ma150_series.iloc[-1])
        ma150_1m_ago = safe_float(ma150_series.iloc[-21])
        ma150_5m_ago = safe_float(ma150_series.iloc[-100])
    except Exception:
        return None
    if not (ma150_now and ma150_1m_ago and ma150_5m_ago): return None
    flat_change = abs(ma150_1m_ago / ma150_5m_ago - 1)
    if flat_change > 0.05: return None
    rise_change = ma150_now / ma150_1m_ago - 1
    if rise_change < 0.01: return None
    if close < base_high * 0.99: return None
    return {"base_high": round(base_high, 2), "base_low": round(base_low, 2),
            "ma150_rise_pct": round(rise_change*100, 2),
            "base_width_pct": round((base_high/base_low - 1)*100, 1)}


# =============================================================
# REVERSAL DETECTION (kept for compat, disabled in scan_slice)
# =============================================================
def compute_reversal_indicators(df, rsi_series):
    out = {"bullish_divergence": False, "hammer_candle": False,
           "bullish_engulfing": False, "higher_low_structure": False,
           "obv_divergence": False, "selling_climax": False}
    if df is None or len(df) < 30: return out
    closes = df["Close"].values; highs = df["High"].values
    lows = df["Low"].values; opens = df["Open"].values; vols = df["Volume"].values
    n = len(closes)
    # Bullish RSI divergence
    if rsi_series is not None and len(rsi_series) >= 30:
        rsi_vals = rsi_series.values
        old_low_idx = closes[-30:-10].argmin() + (n - 30)
        new_low_idx = closes[-10:].argmin() + (n - 10)
        if closes[new_low_idx] < closes[old_low_idx]:
            if not pd.isna(rsi_vals[new_low_idx]) and not pd.isna(rsi_vals[old_low_idx]):
                if rsi_vals[new_low_idx] > rsi_vals[old_low_idx]:
                    out["bullish_divergence"] = True
    # Hammer
    body = abs(closes[-1] - opens[-1])
    upper_shadow = highs[-1] - max(closes[-1], opens[-1])
    lower_shadow = min(closes[-1], opens[-1]) - lows[-1]
    total_range = highs[-1] - lows[-1]
    if total_range > 0 and body > 0:
        if lower_shadow >= 2 * body and upper_shadow <= body * 0.5 and total_range / closes[-1] > 0.015:
            out["hammer_candle"] = True
    # Bullish engulfing
    if n >= 2 and closes[-2] < opens[-2] and closes[-1] > opens[-1]:
        if closes[-1] >= opens[-2] and opens[-1] <= closes[-2]:
            out["bullish_engulfing"] = True
    # Higher-Low structure
    if n >= 30:
        old_seg = lows[-30:-15]; new_seg = lows[-15:]
        if new_seg.min() > old_seg.min():
            avg_close = closes[-30:-15].mean()
            if (old_seg.max() - old_seg.min()) / avg_close > 0.03:
                out["higher_low_structure"] = True
    # OBV divergence
    if n >= 30:
        import numpy as np
        obv_changes = []
        for i in range(1, n):
            if closes[i] > closes[i-1]: obv_changes.append(vols[i])
            elif closes[i] < closes[i-1]: obv_changes.append(-vols[i])
            else: obv_changes.append(0)
        obv = np.cumsum([0] + obv_changes)
        old_idx = closes[-30:-10].argmin() + (n - 30)
        new_idx = closes[-10:].argmin() + (n - 10)
        if closes[new_idx] < closes[old_idx] and obv[new_idx] > obv[old_idx]:
            out["obv_divergence"] = True
    # Selling climax
    if n >= 21:
        vol20_avg = vols[-20:].mean()
        if vol20_avg > 0:
            for i in range(-5, 0):
                if closes[i] < closes[i-1] and vols[i] / vol20_avg >= 2.5:
                    out["selling_climax"] = True; break
    return out


def compute_reversal_strength_score(rev_signals):
    rss = 0
    if rev_signals.get("bullish_divergence"):    rss += 25
    if rev_signals.get("obv_divergence"):        rss += 20
    if rev_signals.get("higher_low_structure"):  rss += 15
    if rev_signals.get("bullish_engulfing"):     rss += 15
    if rev_signals.get("hammer_candle"):         rss += 12
    if rev_signals.get("selling_climax"):        rss += 13
    return rss


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

    # Per-setup RSI zones (with optional CLI overrides for tuning experiments)
    if relax == 0:
        rsi_min = BO_RSI_MIN_OVERRIDE if BO_RSI_MIN_OVERRIDE is not None else 45
        rsi_max = BO_RSI_MAX_OVERRIDE if BO_RSI_MAX_OVERRIDE is not None else 68
        rsi_breakout  = rsi_min <= rsi14 <= rsi_max
        rsi_prerocket = 40 <= rsi14 <= 65
        rsi_zone      = rsi_breakout
    else:
        rsi_breakout  = 40 <= rsi14 <= 72
        rsi_prerocket = 38 <= rsi14 <= 72
        rsi_zone      = 38 <= rsi14 <= 75

    if relax == 0:
        trend_ok  = close > ma20 and ma20 > ma50 * 0.99
        near_high = close >= prev_20_high * 0.93
        base_ok   = base_range <= BO_BASE_MAX
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

    # Phase G: REVERSAL disabled. New setups: VCP, SHORT_SQUEEZE, STAGE_2 + BREAKOUT
    pct_from_52w = ((close - high_52w) / high_52w * 100) if high_52w > 0 else 0
    reversal_setup = False  # disabled
    rss = 0
    rev_indicators = {}

    # ---- MEAN_REVERSION setup (v0) — runs BEFORE breakout hard-exits ----
    # Uptrend pullback that is oversold and turning up. Postmortem-informed to
    # avoid REVERSAL's failures: (1) only in confirmed rising-MA150 uptrend
    # (no falling knives), (2) truly oversold RSI<38 (not neutral), (3) first
    # up-day confirmation, (4) earnings-blackout >=7d, (5) base-cap <15% (no
    # wide knives), (6) extension-cap perf120<80% (no momentum-unwinds).
    if ONLY_SETUP in (None, "MEAN_REVERSION"):
        ma150_prev = safe_float(df["MA150"].iloc[-11]) if len(df) >= 11 else 0.0
        prev_close = safe_float(prev["Close"])
        _en = (catalyst_signals or {}).get("earnings_next_days")
        mr_fire = (
            ma150 > 0 and ma150_prev > 0 and ma150 > ma150_prev and close > ma150  # uptrend
            and close < ma20                                                       # pulled back
            and rsi14 < MR_RSI_MAX                                                 # oversold
            and close > prev_close                                                 # turning up
            and base_range < 15                                                    # no wide knife
            and perf_120 < 80                                                      # not extended
            and ((_en is None) or (_en >= 7))                                      # earnings blackout
        )
        if mr_fire:
            mr_buy   = round(close * 1.002, 2)
            mr_stop  = round(low_10 * 0.985, 2)
            mr_riskp = (mr_buy - mr_stop) / mr_buy * 100 if mr_buy > 0 else 99
            mr_tgt   = round(max(ma20, high_10), 2)
            mr_rr    = (mr_tgt - mr_buy) / (mr_buy - mr_stop) if mr_buy > mr_stop else 0
            if 2.0 <= mr_riskp <= 10.0 and mr_rr >= 1.5:
                mr_score  = 50.0
                mr_score += max(0.0, 40 - rsi14) * 1.5            # oversold depth (fixed anchor)
                mr_score += min(mr_rr, 4.0) * 6                    # RR
                if close > ma50: mr_score += 8                     # shallow pullback = healthier
                if mh > mh_p:    mr_score += 6                     # MACD hist turning up
                if catalysts.get("pocket_pivot"): mr_score += 8
                mr_score += min(max(perf_120, 0.0), 40.0) * 0.2    # uptrend strength
                _dr = high - safe_float(l["Low"])
                _cs = (close - safe_float(l["Low"])) / _dr if _dr > 0 else 0.5
                _id = (high <= safe_float(prev["High"])) and (safe_float(l["Low"]) >= safe_float(prev["Low"]))
                return {
                    "ticker": ticker, "setup": "MEAN_REVERSION", "horizon": "2-4 weeks",
                    "price": round(close, 2), "buy_above": mr_buy, "stop": mr_stop,
                    "target": mr_tgt, "risk_pct": round(mr_riskp, 2), "rr": round(mr_rr, 2),
                    "score": round(mr_score, 1), "relax_level": relax,
                    "movement_class": "MEAN_REVERSION", "closing_strength": round(_cs, 2),
                    "inside_day": _id, "vcp_contraction": None, "squeeze_short_pct": None,
                    "stage2_ma150_rise": None,
                }
        if ONLY_SETUP == "MEAN_REVERSION":
            return None

    # ---- MOMO setup (v0, 2026-05-30) — Vertical-Momentum-Catcher ----
    # OPT-IN ONLY: 2026-05-30 baseline failed gate (PF 1.51 < 2.0, regime-sensitive,
    # +12% TP capped winners). Kept as opt-in research tool (--only-setup MOMO).
    # Original Philosophie: überkauft (RSI>60) ok, ignoriert base_range, vol>=2x.
    if ONLY_SETUP == "MOMO":
        if len(df) >= 6:
            perf_5d_val      = (close / safe_float(df["Close"].iloc[-6]) - 1) * 100
            prev_5d_high_val = safe_float(df["High"].iloc[-6:-1].max())
        else:
            perf_5d_val      = 0.0
            prev_5d_high_val = 0.0
        _momo_en = (catalyst_signals or {}).get("earnings_next_days")
        momo_fire = (
            perf_5d_val > 15.0                                  # vertical spike
            and vol_ratio >= 2.0                                # institutional confirmation
            and rsi14 > 60                                      # already in momentum
            and close > prev_5d_high_val                        # signal-day continuation
            and ma50 > 0 and close > ma50                       # at least short-term trend
            and ((_momo_en is None) or (_momo_en >= 5))         # earnings blackout (5d)
        )
        if momo_fire:
            momo_buy   = round(close * 1.002, 2)
            momo_stop  = round(close - 2.5 * atr14, 2)
            momo_riskp = (momo_buy - momo_stop) / momo_buy * 100 if momo_buy > 0 else 99
            momo_tgt   = round(close * 1.12, 2)
            momo_rr    = (momo_tgt - momo_buy) / (momo_buy - momo_stop) if momo_buy > momo_stop else 0
            if 5.0 <= momo_riskp <= 18.0:   # wide-stop band; no min RR (asymmetric by design)
                momo_score  = 50.0
                momo_score += min(vol_ratio, 5.0) * 5            # vol up to +25
                momo_score += min(perf_5d_val, 40.0) * 0.5       # perf_5d up to +20
                if rsi14 > 70: momo_score += 5
                if catalysts.get("pocket_pivot"): momo_score += 8
                if perf_20 > 30: momo_score += 5
                _mo_dr = high - safe_float(l["Low"])
                _mo_cs = (close - safe_float(l["Low"])) / _mo_dr if _mo_dr > 0 else 0.5
                _mo_id = (high <= safe_float(prev["High"])) and (safe_float(l["Low"]) >= safe_float(prev["Low"]))
                return {
                    "ticker": ticker, "setup": "MOMO", "horizon": "1-3 weeks",
                    "price": round(close, 2), "buy_above": momo_buy, "stop": momo_stop,
                    "target": momo_tgt, "risk_pct": round(momo_riskp, 2), "rr": round(momo_rr, 2),
                    "score": round(momo_score, 1), "relax_level": relax,
                    "movement_class": "MOMO", "closing_strength": round(_mo_cs, 2),
                    "inside_day": _mo_id, "vcp_contraction": None, "squeeze_short_pct": None,
                    "stage2_ma150_rise": None,
                }
        if ONLY_SETUP == "MOMO":
            return None

    # BREAKOUT detection (20d high) — computed early so the base-gate can be setup-specific
    buy_above_prev   = prev_20_high * 1.002
    breakout_close   = close >= buy_above_prev * 0.99
    breakout_setup   = breakout_touch and higher_tf and rsi_breakout and breakout_close

    # Hard exits (all 4 Phase G setups need trend/volume/etc)
    if not trend_ok:   return None
    if not vol_ok:     return None
    if not rsi_zone:   return None
    if relax == 0 and not momentum_ok: return None
    # Base-gate SETUP-SPECIFIC (mirror live): BREAKOUT uses BO_BASE_MAX (default 22), others <=8
    if relax == 0:
        base_cap = BO_BASE_MAX if breakout_setup else 8
        if base_range > base_cap: return None
    if atr_pct > 15: return None

    # Phase G new setups
    short_pct_val = (catalyst_signals or {}).get("short_pct_float")
    vcp_data     = detect_vcp_setup(df, atr14, vol_ratio, ma50, ma150, close, high, atr_pct)
    squeeze_data = detect_short_squeeze_setup(df, short_pct_val, ma20, ma50, close, vol_ratio, rsi14)
    stage2_data  = detect_stage2_transition(df, df.get("MA150"), close)

    pre_rocket = False
    position_setup = False

    # Priority: STAGE_2 > VCP > SHORT_SQUEEZE > BREAKOUT
    if stage2_data:
        chosen_setup = "STAGE_2"
    elif vcp_data:
        chosen_setup = "VCP"
    elif squeeze_data:
        chosen_setup = "SHORT_SQUEEZE"
    elif breakout_setup:
        chosen_setup = "BREAKOUT"
    else:
        return None

    # Isolated single-setup measurement (e.g. --only-setup BREAKOUT)
    if ONLY_SETUP and chosen_setup != ONLY_SETUP:
        return None

    # Entry/Stop/Target per setup
    if chosen_setup == "STAGE_2":
        setup = "STAGE_2"; horizon = "8-16 weeks"
        buy_above = round(stage2_data["base_high"] * 1.005, 2)
        stop      = round(max(ma150, stage2_data["base_low"]) * 0.97, 2)
        risk_pct  = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0: stop = buy_above * 0.92; risk_pct = 8.0
        elif risk_pct > 15.0: return None
        base_width = stage2_data["base_high"] - stage2_data["base_low"]
        target = round(min(stage2_data["base_high"] + base_width * 1.5, buy_above * 1.40), 2)

    elif chosen_setup == "VCP":
        setup = "VCP"; horizon = "4-8 weeks"
        buy_above = round(vcp_data["breakout_level"], 2)
        stop      = round(vcp_data["base_low"] * 0.99, 2)
        risk_pct  = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0: stop = buy_above * 0.95; risk_pct = 5.0
        elif risk_pct > 8.0: return None
        target = round(max(buy_above + (atr14 * 4.0), buy_above * 1.18), 2)

    elif chosen_setup == "SHORT_SQUEEZE":
        setup = "SHORT_SQUEEZE"; horizon = "2-4 weeks"
        buy_above = round(squeeze_data["breakout_level"], 2)
        stop      = round(ma20 * 0.98, 2)
        risk_pct  = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0: stop = buy_above * 0.96; risk_pct = 4.0
        elif risk_pct > 8.0: return None
        tgt_52w = high_52w * 0.95
        target = round(min(tgt_52w, buy_above * 1.20), 2) if tgt_52w > buy_above else round(buy_above * 1.15, 2)

    else:  # BREAKOUT
        setup = "BREAKOUT"; horizon = "1-3 weeks"
        buy_above = prev_20_high * 1.002
        stop_candidate = low_10 * 0.995
        atr_pct_val    = (atr14 / buy_above) * 100 if buy_above > 0 else 5.0
        min_dist_pct   = max(3.0, atr_pct_val * 1.5)
        multiplier     = 1.5
        candidate_dist = ((buy_above - stop_candidate) / buy_above * 100 if buy_above > 0 else 99)
        if candidate_dist < min_dist_pct:
            stop = buy_above * (1 - min_dist_pct / 100)
        elif candidate_dist > 10.0:
            stop = buy_above - (atr14 * multiplier)
        else:
            stop = stop_candidate
        risk_pct = (buy_above - stop) / buy_above * 100 if buy_above > 0 else 99
        if risk_pct < 3.0: stop = buy_above * 0.97; risk_pct = 3.0
        elif risk_pct > 10.0: return None
        target = find_target(df, buy_above, atr14, setup, risk_pct)

    upside_pct = ((target / buy_above) - 1) * 100
    rr = upside_pct / risk_pct if risk_pct > 0 else 0
    if rr < 1.2: return None

    # ---- Phase G setup-specific scoring (must mirror ApexScan.py) ----
    if setup == "STAGE_2":
        score = 30.0
        score += min(stage2_data["ma150_rise_pct"], 5) * 3
        score += 10 if stage2_data["base_width_pct"] < 25 else 5
        score += 8 if higher_tf else 0
        score += 8 if macd_bull else 0
        score += 6 if 48 <= rsi14 <= 68 else (3 if rsi_zone else 0)
        score += min(max(perf_20, 0), 20) * 0.5
        score += min(vol_ratio, 3.0) * 4
        score += min(rr, 5.0) * 4
        if catalysts["pocket_pivot_recent"]: score += 10
        if catalysts["volume_climax"]:       score += 5
        if catalysts["gap_signal"]:          score += 8
    elif setup == "VCP":
        score = 25.0
        score += vcp_data["contraction_pct"] * 0.2
        score += 10 if vcp_data["base_range_pct"] <= 8 else 5
        score += 8 if higher_tf else 0
        score += 8 if macd_bull else 0
        score += 6 if 48 <= rsi14 <= 68 else (3 if rsi_zone else 0)
        score += min(max(perf_60, 0), 35) * 0.4
        score += min(vol_ratio, 3.0) * 4
        score += min(rr, 5.0) * 4
        if catalysts["pocket_pivot_recent"]: score += 12
        if catalysts["volume_climax"]:       score += 5
        if catalysts["gap_signal"]:          score += 8
    elif setup == "SHORT_SQUEEZE":
        score = 20.0
        score += min(squeeze_data["short_pct"] - 15, 20) * 0.5
        score += min(squeeze_data["perf_5d"], 15) * 0.6
        score += 8 if higher_tf else 0
        score += 8 if macd_bull else 0
        score += 8 if expansion_vol else (4 if vol_ratio >= 1.0 else 0)
        score += 6 if 50 <= rsi14 <= 70 else (3 if 45 <= rsi14 <= 78 else 0)
        score += min(vol_ratio, 3.0) * 4
        score += min(rr, 5.0) * 4
        if catalysts["pocket_pivot_recent"]: score += 10
        if catalysts["volume_climax"]:       score += 8
        if catalysts["gap_signal"]:          score += 8
    else:  # BREAKOUT
        score = 0.0
        score += 20
        score += 8 if higher_tf else 0
        score += 8 if expansion_vol else (4 if vol_ratio >= 1.0 else 0)
        score += 8 if macd_bull else 0
        _rsi_hi = 72 if SCORE_REALIGN else 68   # Realign: RSI>=70 zeigt 75% WR (n=12)
        score += 6 if 48 <= rsi14 <= _rsi_hi else (3 if rsi_zone else 0)
        score += min(max(perf_20, 0), 20)  * 0.8
        score += min(max(perf_60, 0), 35)  * 0.5
        score += min(vol_ratio, 3.0) * 4
        score += min(rr, 5.0) * 4
        if catalysts["pocket_pivot_recent"]: score += 10
        if catalysts["volume_climax"]:       score += 5
        if catalysts["gap_signal"]:          score += 8
        if catalysts["vcp_signal"]:          score += 5

    # Phase B catalysts (earnings only — short/analyst skipped in backtest_mode)
    if catalyst_signals is not None:
        score += score_delta_for_catalyst_signals(catalyst_signals, setup,
                                                   backtest_mode=True)

    # Phase G: Movement Classification
    movement_class = "STANDARD"
    movement_bonus = 0
    if setup == "BREAKOUT":
        if SCORE_REALIGN:
            # Bucketed nach gemessenen Live-WR-Kohorten (mild = re-rank, nicht gate-out)
            if perf_120 < 0:
                movement_class = "WEAK_BREAKOUT"; movement_bonus = -15   # unveraendert (n=7 zu klein)
            elif perf_120 <= 25:
                movement_class = "DEADZONE_BREAKOUT"; movement_bonus = -3  # 44% WR n=27, war +5
            elif perf_120 <= 50:
                movement_class = "SWEET_BREAKOUT"; movement_bonus = 15     # 71% WR n=24
            else:  # >50 extended
                movement_class = ("POWER_BREAKOUT" if pct_from_52w > -2 else "EMERGING_BREAKOUT")
                movement_bonus = 8
        else:
            if perf_120 > 25 and pct_from_52w > -2:
                movement_class = "POWER_BREAKOUT"; movement_bonus = 15
            elif perf_120 >= 0:
                movement_class = "EMERGING_BREAKOUT"; movement_bonus = 5
            else:
                movement_class = "WEAK_BREAKOUT"; movement_bonus = -15
    elif setup == "VCP":
        movement_class = "VCP_TIGHT" if vcp_data["base_range_pct"] <= 8 else "VCP_WIDE"
        movement_bonus = 10 if movement_class == "VCP_TIGHT" else 5
    elif setup == "SHORT_SQUEEZE":
        movement_class = "SQUEEZE_HIGH" if short_pct_val and short_pct_val >= 20 else "SQUEEZE_MED"
        movement_bonus = 12 if movement_class == "SQUEEZE_HIGH" else 6
    elif setup == "STAGE_2":
        movement_class = "STAGE_2_BREAKOUT"; movement_bonus = 15

    # Phase E: Closing-Strength
    day_range = high - safe_float(l["Low"])
    closing_strength = (close - safe_float(l["Low"])) / day_range if day_range > 0 else 0.5
    closing_bonus = 5 if closing_strength > 0.75 else (-10 if closing_strength < 0.5 else 0)

    # Phase E: Inside-Day (BREAKOUT consolidation = often false breakout)
    inside_day = (high <= safe_float(prev["High"])) and (safe_float(l["Low"]) >= safe_float(prev["Low"]))
    inside_day_penalty = -8 if (inside_day and setup == "BREAKOUT") else 0

    score += movement_bonus + closing_bonus + inside_day_penalty

    # Hebel B (Score-Rebuild): Extension-Penalty mit Catalyst-Carve-Out.
    # Trifft NUR das Fade-Profil: stark gelaufen (perf_120>60) + schwache Bestaetigung
    # (vol<1.5 UND closing<0.6) + KEIN starker Catalyst. Carve-Out = Semi/AI-Capex-Winner
    # bleiben verschont (what_to_replicate). NB: analyst_upside in backtest_mode nicht da,
    # Carve-Out nutzt earnings_beat / PP+Vol-Climax / Gap>=5 (alle backtestbar).
    _gap = catalysts.get("up_gap_pct", 0) or 0
    strong_catalyst = bool(
        (catalyst_signals or {}).get("earnings_beat_recent") or
        (catalysts.get("pocket_pivot_recent") and catalysts.get("volume_climax")) or
        _gap >= 5.0
    )
    if SCORE_REBUILD and setup == "BREAKOUT":
        # KORRIGIERT 2026-06-20 nach Forensik: Treiber des 100-110-Trough ist perf_120-
        # Extension (Loser +61% vs Winner +33%), NICHT vol/closing (diskriminieren nicht).
        # Extended (perf_120>50) OHNE starken Catalyst = Penalty. Catalyst-Carve-Out schuetzt
        # Semi/AI-Capex-Winner (what_to_replicate).
        if perf_120 > 50 and not strong_catalyst:
            score -= EXT_PENALTY

    # (RSS removed in Phase G — REVERSAL disabled)

    # Test 2026-06-04: Catalyst-Free Elite Penalty (opt-in via CLI)
    # BREAKOUT-Signale mit Score>=100 ohne PP/Gap/Beat sind hypothetisch im
    # 100+ Bucket schlechter -> Penalty drueckt sie aus dem Elite-Bucket.
    if CATALYST_FREE_ELITE_PENALTY > 0 and setup == "BREAKOUT" and score >= 100:
        gap_pct = catalysts.get("up_gap_pct", 0) or 0
        has_positive_catalyst = (
            catalysts.get("pocket_pivot_recent") or
            catalysts.get("gap_signal") or
            gap_pct >= 2.0 or
            (catalyst_signals or {}).get("earnings_beat_recent")
        )
        if not has_positive_catalyst:
            score -= CATALYST_FREE_ELITE_PENALTY

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
        "relax_level":      relax,
        "movement_class":   movement_class,
        "closing_strength": round(closing_strength, 2),
        "inside_day":       inside_day,
        # Vola-Metadaten fuer Score-Forensik (Hebel B' / 2026-06-20)
        "vol_ratio":        round(vol_ratio, 2),
        "perf_20":          round(perf_20, 1),
        "perf_60":          round(perf_60, 1),
        "perf_120":         round(perf_120, 1),
        "rsi":              round(rsi14, 1),
        "strong_catalyst":  strong_catalyst,
        # SCORE_V2 Stufe 2 (2026-07-11): Features fuer LogReg-Ranking (reine Metadaten)
        "macd_bull":        bool(macd_bull),
        "base_range":       round(base_range, 1) if base_range is not None else None,
        "movement_bonus":   movement_bonus,
        "cat_pocket_pivot": bool(catalysts.get("pocket_pivot_recent")),
        "cat_vol_climax":   bool(catalysts.get("volume_climax")),
        "cat_gap_pct":      round(catalysts.get("up_gap_pct", 0) or 0, 2),
        # 2026-07-22: VCP-Katalysator-Staerke (ATR-Kontraktion) auch fuer NICHT-VCP-Setups
        # mitschreiben. Analyse zeigte cat_vcp_strength>0 auf BREAKOUT = +38pp WR (Live n=28),
        # war aber nie backtestbar weil hier nicht durchgereicht. Jetzt 2J-validierbar.
        "cat_vcp_strength": round(catalysts.get("vcp_strength", 0) or 0, 2),
        # Phase G metadata
        "vcp_contraction":   vcp_data["contraction_pct"] if chosen_setup == "VCP" else None,
        "squeeze_short_pct": squeeze_data["short_pct"] if chosen_setup == "SHORT_SQUEEZE" else None,
        "stage2_ma150_rise": stage2_data["ma150_rise_pct"] if chosen_setup == "STAGE_2" else None,
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
    MAX_TRIGGER_DAYS = 3
    setup_type = signal.get("setup", "BREAKOUT")
    # Phase F.3 REVERSAL exit management state
    rev_trailing_active = False
    rev_dynamic_sl = sl    # mutable stop, can be raised by trailing logic

    for i, (_, row) in enumerate(future.iterrows()):
        try:
            o = safe_float(row["Open"])
            h = safe_float(row["High"])
            l = safe_float(row["Low"])
            c = safe_float(row["Close"])
        except Exception:
            continue

        # Wait for entry trigger
        if trigger_day is None:
            if i >= MAX_TRIGGER_DAYS:
                return None, None, None, None
            if h >= entry:
                trigger_day = i
                if l <= sl:
                    return None, None, None, None
            else:
                continue

        # Days since entry triggered
        days_in_trade = i - trigger_day

        # --- Phase F.3: REVERSAL Exit Management ---
        if setup_type == "REVERSAL" and trigger_day is not None:
            # Rule 1: Trailing-Stop after Day 5 — once intraday hit +5%, lock breakeven
            if not rev_trailing_active and days_in_trade >= 5:
                if h >= entry * 1.05:
                    rev_dynamic_sl = max(rev_dynamic_sl, entry)  # breakeven
                    rev_trailing_active = True
            # Rule 2: Time-Stop at Day 14 if no progress
            if days_in_trade >= 14:
                pnl_pct_now = (c / entry - 1) * 100
                if pnl_pct_now <= 0:
                    return c, "Time Exit (REV-cut)", i + 1, trigger_day + 1

        # Trade is live — use dynamic stop for REVERSAL, static for others
        active_sl = rev_dynamic_sl if setup_type == "REVERSAL" else sl
        hit_tp = h >= tp
        hit_sl = l <= active_sl

        if hit_tp and hit_sl:
            ep     = tp if o >= entry else active_sl
            reason = "Take Profit" if o >= entry else ("Trailing Stop" if rev_trailing_active else "Stop Loss")
            return ep, reason, i + 1, trigger_day + 1
        elif hit_tp:
            return tp, "Take Profit", i + 1, trigger_day + 1
        elif hit_sl:
            reason = "Trailing Stop" if rev_trailing_active else "Stop Loss"
            return active_sl, reason, i + 1, trigger_day + 1

    if trigger_day is None:
        return None, None, None, None

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
            sig = scan_slice(ticker, df_slice, relax=FORCE_RELAX, risk_on=risk_on,
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
        # SCORE_V2: Pick-Ranking nach LogReg-prob (= Tages-Perzentil-Ranking).
        # Original-Score bleibt fuer Gates/Filter unveraendert (reines Re-Ranking).
        if SCORE_V2:
            for sig in signals_today:
                sig["v2_prob"] = round(_score_v2_prob(sig), 4)
            signals_today.sort(key=lambda x: x["v2_prob"], reverse=True)
        elif PICK_BAND:
            # 2026-07-15: Sweet-Spot-Band aufs Pick-Ranking (analog TG_SWEET_BAND Telegram).
            # BREAKOUT im Band [90,120) = Tier 1 (nach Score), ausserhalb = Tier 0 (nach
            # Naehe zum Band-Zentrum). Andere Setups behalten Score-Position. Reines
            # Re-Ranking der Pick-Stufe — Signal-Count identisch.
            def _band_rank(s):
                sc = s.get("score", 0)
                if s.get("setup") != "BREAKOUT":
                    return (1, sc)
                if PICK_BAND[0] <= sc < PICK_BAND[1]:
                    return (1, sc)
                return (0, -abs(sc - (PICK_BAND[0] + PICK_BAND[1]) / 2))
            signals_today.sort(key=_band_rank, reverse=True)
        else:
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
                "relax_level": sig.get("relax_level", 0),
                "risk_on":     sig["risk_on"],
                "equity":      round(equity, 2),
                # Vola-Metadaten fuer Score-Forensik (Hebel B')
                "vol_ratio":        sig.get("vol_ratio"),
                "perf_20":          sig.get("perf_20"),
                "perf_60":          sig.get("perf_60"),
                "perf_120":         sig.get("perf_120"),
                "rsi":              sig.get("rsi"),
                "movement_class":   sig.get("movement_class"),
                "closing_strength": sig.get("closing_strength"),
                "strong_catalyst":  sig.get("strong_catalyst"),
                "v2_prob":          sig.get("v2_prob"),
                # 2026-07-22: Katalysator-Flags durchreichen fuer Feature-Praediktivitaet
                "cat_vcp_strength": sig.get("cat_vcp_strength"),
                "cat_pocket_pivot": sig.get("cat_pocket_pivot"),
                "cat_vol_climax":   sig.get("cat_vol_climax"),
                "cat_gap_pct":      sig.get("cat_gap_pct"),
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

    # Breakeven (pnl_pct == 0, from Phase F.3 trailing stops at entry) excluded from WR
    wins   = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] < 0]   # strict <0
    decisive = len(wins) + len(losses)
    wr     = len(wins) / decisive * 100 if decisive > 0 else 0
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
    parser.add_argument("--bo-rsi-min", type=int, default=None,
                        help="Override BREAKOUT RSI-Gate min (default 45 relax=0, 40 relax>=1)")
    parser.add_argument("--bo-rsi-max", type=int, default=None,
                        help="Override BREAKOUT RSI-Gate max (default 68 relax=0, 72 relax>=1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Custom output filename for results JSON (default apex_backtest_results.json)")
    parser.add_argument("--force-relax", type=int, default=0, choices=[0, 1, 2],
                        help="Force relax level for ALL tickers (Telegram-gate study). 0=strict (default), 1=relaxed")
    parser.add_argument("--only-setup", type=str, default=None,
                        help="Restrict scan to ONE setup for isolated measurement, e.g. MEAN_REVERSION")
    parser.add_argument("--mr-rsi-max", type=float, default=38,
                        help="MEAN_REVERSION oversold RSI threshold (default 38; tuning knob)")
    parser.add_argument("--bo-base-max", type=float, default=22,
                        help="BREAKOUT relax=0 base_range cap (default 22; others fixed <=8)")
    parser.add_argument("--vcp-atr-contraction", type=float, default=0.20,
                        help="VCP min ATR-contraction over 25 bars (default 0.20; was 0.30 pre-2026-05-28)")
    parser.add_argument("--sq-short-min", type=float, default=15.0,
                        help="SHORT_SQUEEZE min short interest %% (default 15.0)")
    parser.add_argument("--catalyst-free-elite-penalty", type=float, default=0.0,
                        help="Penalty fuer BO score>=100 OHNE [PP, Gap>=2%%, EarningsBeat]. "
                             "Test-Hypothese: techn. Extreme ohne Bestaetigung sind Falle (default 0 = off)")
    parser.add_argument("--score-realign", action="store_true",
                        help="BREAKOUT-Score an gemessene WR-Kohorten aligniern: RSI 48-72, "
                             "perf_120 0-25 Penalty, 25-50 Sweet-Spot (default off)")
    parser.add_argument("--score-v2", action="store_true",
                        help="SCORE_V2 Stufe 2: Pick-Ranking via LogReg (score_v2_model.json)")
    parser.add_argument("--pick-band", type=str, default=None,
                        help="Sweet-Spot-Band aufs Pick-Ranking, z.B. '90,120' (reines Re-Ranking)")
    parser.add_argument("--score-rebuild", action="store_true",
                        help="Hebel B: Extension-Penalty mit Catalyst-Carve-Out (default off)")
    parser.add_argument("--ext-penalty", type=float, default=12.0,
                        help="Hoehe der Extension-Penalty fuer --score-rebuild (default 12, Sweep-Optimum)")
    args = parser.parse_args()

    # Stash CLI RSI overrides into module-level globals for scan_slice to pick up
    global BO_RSI_MIN_OVERRIDE, BO_RSI_MAX_OVERRIDE, RESULTS_FILE_OVERRIDE, FORCE_RELAX, ONLY_SETUP, MR_RSI_MAX, BO_BASE_MAX, VCP_ATR_CONTRACTION, SQ_SHORT_MIN, CATALYST_FREE_ELITE_PENALTY, SCORE_REALIGN, SCORE_REBUILD, EXT_PENALTY
    BO_RSI_MIN_OVERRIDE = args.bo_rsi_min
    BO_RSI_MAX_OVERRIDE = args.bo_rsi_max
    RESULTS_FILE_OVERRIDE = args.out
    FORCE_RELAX = args.force_relax
    ONLY_SETUP = args.only_setup
    MR_RSI_MAX = args.mr_rsi_max
    BO_BASE_MAX = args.bo_base_max
    VCP_ATR_CONTRACTION = args.vcp_atr_contraction
    SQ_SHORT_MIN = args.sq_short_min
    CATALYST_FREE_ELITE_PENALTY = args.catalyst_free_elite_penalty
    SCORE_REALIGN = args.score_realign
    SCORE_REBUILD = args.score_rebuild
    EXT_PENALTY = args.ext_penalty
    global SCORE_V2, SCORE_V2_MODEL, PICK_BAND
    SCORE_V2 = args.score_v2
    if args.pick_band:
        lo, hi = (float(x) for x in args.pick_band.split(","))
        PICK_BAND = (lo, hi)
        print(f"[TEST] PICK_BAND={PICK_BAND} — Pick-Ranking bevorzugt BREAKOUT im Band (Re-Ranking)")
    if SCORE_V2:
        with open("score_v2_model.json", encoding="utf-8") as f:
            SCORE_V2_MODEL = json.load(f)
        print(f"[TEST] SCORE_V2=ON (LogReg-Ranking, Train {SCORE_V2_MODEL['train_window']}, "
              f"n_train={SCORE_V2_MODEL['n_train']}) — Pick-Stufe re-ranked, Gates unveraendert")
    if SCORE_REBUILD:
        print(f"[TEST] SCORE_REBUILD=ON (Hebel B-korr: Extension-Penalty -{EXT_PENALTY:.0f} fuer "
              f"perf120>50 + kein-strong-Catalyst; Carve-Out: PP+VolClimax/EarnBeat/Gap>=5)")
    if CATALYST_FREE_ELITE_PENALTY:
        print(f"[TEST] CATALYST_FREE_ELITE_PENALTY={CATALYST_FREE_ELITE_PENALTY} "
              f"(BO score>=100 ohne PP/Gap/Beat -> -{CATALYST_FREE_ELITE_PENALTY})")
    if SCORE_REALIGN:
        print("[TEST] SCORE_REALIGN=ON (BO: RSI 48-72, perf_120 0-25 Penalty, 25-50 Sweet)")
    if FORCE_RELAX:
        print(f"[MEASURE] FORCE_RELAX={FORCE_RELAX} — measuring relaxed-signal quality")
    if ONLY_SETUP:
        print(f"[MEASURE] ONLY_SETUP={ONLY_SETUP} — isolated single-setup backtest")

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

    # Breakeven excluded from WR (Phase F.3 trailing-stops at entry)
    wins     = [t for t in trades if t["pnl_pct"] > 0]
    losses   = [t for t in trades if t["pnl_pct"] < 0]
    be       = [t for t in trades if t["pnl_pct"] == 0]
    decisive = len(wins) + len(losses)
    wr       = len(wins) / decisive * 100 if decisive > 0 else 0
    pf       = abs(sum(t["pnl_usd"] for t in wins) / sum(t["pnl_usd"] for t in losses)) if losses else 999

    period_str = f"{args.start} bis {args.end}" if args.start else f"{args.days} Handelstage"
    print(f"\n{'='*60}")
    print(f"  BACKTEST ERGEBNIS  ({period_str})")
    print(f"{'='*60}")
    print(f"  Trades total  : {len(trades)}  (+{len(be)} BE)" if be else f"  Trades total  : {len(trades)}")
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

    out_file = RESULTS_FILE_OVERRIDE or RESULTS_FILE
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)
    print(f"\nErgebnisse gespeichert: {out_file}")

    build_chart(trades, eq_curve)


if __name__ == "__main__":
    main()