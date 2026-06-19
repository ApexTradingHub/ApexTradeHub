"""
apex_trader.py — Paper-Trading-Engine (Phase B)

Liest apex_signals.json, waehlt taeglich Top-1 BREAKOUT nach Score, eroeffnet
Positionen wenn Trigger (high >= buy_above) erreicht und Cash verfuegbar.
Trailing-Stop: bei high >= Entry*1.08 -> SL springt auf Entry*1.05 (one-shot).
Schreibt apex_positions.json (State) + apex_trade_log.json (Journal).

Mode-Switch via TRADING_MODE env var ("paper"|"live"). Im "live"-Modus
ruft eToro-API auf (Stub, noch nicht implementiert).

USAGE:
    py apex_trader.py                 # einmaliger Run
    py apex_trader.py --reset         # State + Log loeschen, frisch starten
    py apex_trader.py --dry-run       # nichts schreiben, nur loggen
    py apex_trader.py --status        # aktuellen Stand anzeigen

DEPENDENCIES: yfinance, pandas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRADING_MODE = os.environ.get("TRADING_MODE", "paper").lower()  # "paper" | "live"
ETORO_API_KEY = os.environ.get("ETORO_API_KEY", "")
ETORO_ACCOUNT_ID = os.environ.get("ETORO_ACCOUNT_ID", "")

# Risk parameters
# 2026-06-12: Bumped 300->400, 5->7 fuer Rotation-Test (User-Request).
# Scope: testen ob mehr Slots + Momentum-Filler die Cash-Auslastung verbessern.
CAPITAL_INITIAL = 400.0         # Test-Portfolio Startkapital
POSITION_SIZE   = 50.0          # USD pro Trade
MAX_POSITIONS   = 7             # gleichzeitig offene Positionen
CASH_RESERVE    = 50.0          # Mindest-Cash-Reserve (Puffer)

# Setup-Filter — BREAKOUT only.
# 2026-06-12: STAGE_2 wieder raus. Widerspricht der Rotations-These (Hold 60d,
# blockiert Slot wochenlang). Schnelle ~5%-Spruenge sind das Momentum-Filler-Job.
ALLOWED_SETUPS  = {"BREAKOUT"}

# Telegram-aequivalentes Gate (mirrors ApexScan.py)
TG_MIN_RR       = 1.5
TG_MIN_UPSIDE   = 8.0
TG_MIN_SCORE    = {
    "BREAKOUT":      70,
    "VCP":           70,
    "SHORT_SQUEEZE": 65,
    "STAGE_2":       60,
    "MEAN_REVERSION": 70,
}
TG_MIN_SCORE_DEFAULT = 70

# Trailing-Stop — Leiter (3 Stufen) statt one-shot
# Phase-1 Update 2026-06-07: ersetzt single-step +8%->+5% durch progressive Stufen
TRAIL_LADDER = [
    (1.06, 1.02),   # Step 1: high >= entry*1.06 -> SL = entry*1.02  (+2% gesichert)
    (1.10, 1.06),   # Step 2: high >= entry*1.10 -> SL = entry*1.06  (+6% gesichert)
    (1.14, 1.10),   # Step 3: high >= entry*1.14 -> SL = entry*1.10  (+10% gesichert)
]
# Backward-Compat: alte Felder bleiben, werden aber nicht mehr genutzt
TRAILING_TRIGGER_MULT = 1.08
TRAILING_TARGET_MULT  = 1.05

# Stagnations-Exit — totes Kapital freigeben fuer neue Signale
# Phase-1 Update 2026-06-07
STAGNATION_DAYS    = 5      # nach 5 Tagen
STAGNATION_PNL_MIN = -2.0   # PnL zwischen -2 %
STAGNATION_PNL_MAX = 2.0    # und +2 % -> close

# Replacement-Logik — wenn alle Slots voll und neues Signal qualifiziert
# Phase-1 Update 2026-06-07
REPLACEMENT_MIN_SCORE       = 90.0   # neues Signal muss Elite-Bucket sein
REPLACEMENT_WEAKEST_MIN_PNL = 2.0    # schwaechste Position muss >= +2 % im Plus sein
                                     # (niemals Verlust realisieren fuer Replacement)
# Catalyst-Requirement: Pocket Pivot ODER Gap >= 2 % (CONFIRMED/HYPOTHESIS Positiv-Lift)

# Trigger-Window: Signal expired wenn nach N Tagen nicht getriggert
#  Mirrors apex_equity.py + apex_backtest_v2.py (= 3-day cap, matched 61.8 % BO-WR Messung).
#  Re-Validation (Ticker in heutiger Scan) refresht signal_date.
MAX_TRIGGER_DAYS = 3

# Hold-Window: Time-Exit pro Setup (matched apex_equity.py horizon_to_days)
HOLD_DAYS_PER_SETUP = {
    "BREAKOUT":       21,
    "VCP":            40,
    "STAGE_2":        60,
    "SHORT_SQUEEZE":  20,
    "MEAN_REVERSION": 20,
    "REVERSAL":       40,
    "MOMENTUM":        7,   # Filler: schneller Sprung, schnell wieder raus
    "INTRADAY":        1,   # Intraday-Catcher: same-day raus (EOD-Hardclose greift eh frueher)
}
HOLD_DAYS_DEFAULT = 21


def hold_days_for(setup: str) -> int:
    return HOLD_DAYS_PER_SETUP.get(setup, HOLD_DAYS_DEFAULT)

# Pfade
SCRIPT_DIR = Path(__file__).resolve().parent
SIGNALS_FILE   = SCRIPT_DIR / "apex_signals.json"
POSITIONS_FILE = SCRIPT_DIR / "apex_positions.json"
TRADE_LOG_FILE = SCRIPT_DIR / "apex_trade_log.json"
OVERRIDES_FILE = SCRIPT_DIR / "apex_manual_overrides.json"   # Phase 2: User/Claude-Overrides
MOMENTUM_CACHE = SCRIPT_DIR / "apex_momentum_cache.json"     # Momentum-Filler (Tages-Cache)
US_TICKERS     = SCRIPT_DIR / "us_tickers.txt"

# Momentum-Filler (2026-06-12) — fuellt Slots wenn Scanner-Signale nicht reichen
MOMENTUM_UNIVERSE_SIZE     = 200    # Top-N tickers aus us_tickers.txt
MOMENTUM_CACHE_MAX_AGE_H   = 6      # cache max 6h alt, dann refresh
MOMENTUM_PERF_5D_MIN       = 3.0    # mind. +3 % in 5 Handelstagen
MOMENTUM_RSI_MAX           = 72     # nicht ueberkauft
MOMENTUM_VOL_RATIO_MIN     = 1.2    # mind. 20 % ueber 20d-Durchschnitt
MOMENTUM_PRICE_MIN         = 5.0    # keine Penny-Stocks
MOMENTUM_MIN_SCORE         = 60     # eigene Score-Scale, NICHT mit Scanner vergleichbar
MOMENTUM_TP_PCT            = 0.06   # +6 % Target (User-These: schnelle 5%-Spruenge)
MOMENTUM_SL_PCT            = 0.04   # -4 % Stop
MOMENTUM_ENTRY_BUFFER      = 0.005  # buy_above = current * 1.005 (Trigger ueber jetzigem Kurs)

# ---------------------------------------------------------------------------
# Intraday-Momentum-Catcher (2026-06-18) — EXPERIMENT, opt-in via env-flag.
# User-These: Aktien die JETZT intraday laufen catchen, 5% mitnehmen, same-day raus.
# Bewusst risikoreich (MOMO-Profil, BACKLOG #2). Sauber instrumentiert + rollback-bar.
# Reaktiviert/scant NUR die bereits gefilterten Daily-Momentum-Kandidaten (leichter Fetch).
# ---------------------------------------------------------------------------
INTRADAY_ENABLED       = os.environ.get("INTRADAY_ENABLED", "0").strip() in ("1", "true", "True", "yes")
INTRADAY_MAX_POSITIONS = 3       # max gleichzeitige Intraday-Plays (eigenes Sub-Limit)
INTRADAY_GAIN_MIN      = 1.5     # min % vom Tages-Open (schon in Bewegung)
INTRADAY_GAIN_MAX      = 6.0     # max % vom Open (nicht schon erschoepft/zu spaet)
INTRADAY_TP_PCT        = 0.05    # +5 % Target (User-Ziel)
INTRADAY_SL_PCT        = 0.03    # -3 % Stop (intraday dreht schnell -> eng)
INTRADAY_PRICE_MIN     = 5.0     # keine Penny-Stocks
INTRADAY_RANGE_POS_MIN = 0.55    # last muss im oberen Teil der Tagesspanne sein (Momentum intakt)
INTRADAY_EOD_UTC       = "19:45" # Hard-Close ab dieser UTC-Zeit (15 Min vor US-Close 20:00)
INTRADAY_CACHE         = SCRIPT_DIR / "apex_intraday_cache.json"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def log(msg: str):
    print(f"[Trader] {msg}", flush=True)


def f(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def now_iso() -> str:
    """Returns UTC timestamp with explicit Z suffix (unambiguous for any TZ display)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def today_date() -> date:
    return date.today()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        log(f"WARN: cannot read {path.name}: {e}")
        return default


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# State init / I/O
# ---------------------------------------------------------------------------
def init_state() -> dict:
    return {
        "mode": TRADING_MODE,
        "capital_initial": CAPITAL_INITIAL,
        "cash": CAPITAL_INITIAL,
        "last_updated": now_iso(),
        "pending": [],   # signals seen, not yet triggered
        "open":    [],   # active positions
        "closed":  [],   # closed positions (history)
        "expired": [],   # signals that expired without trigger
        "stats": {
            "total_trades":    0,
            "open_trades":     0,
            "wins":            0,
            "losses":          0,
            "pnl_realized":    0.0,
            "pnl_unrealized":  0.0,
            "equity":          CAPITAL_INITIAL,
        },
    }


def load_state() -> dict:
    state = load_json(POSITIONS_FILE)
    if not state:
        return init_state()
    # Fill any new fields for forward compat
    for k, v in init_state().items():
        if k not in state:
            state[k] = v
    return state


def load_trade_log() -> list:
    return load_json(TRADE_LOG_FILE, default=[]) or []


def append_log(entries: list):
    log_data = load_trade_log()
    log_data.extend(entries)
    save_json(TRADE_LOG_FILE, log_data)


# ---------------------------------------------------------------------------
# Price feed (yfinance)
# ---------------------------------------------------------------------------
def _extract_series(df, ticker, single_ticker, col):
    """Holt eine Spalten-Series aus yfinance-DataFrame, robust gegen MultiIndex."""
    if df is None or df.empty:
        return None
    try:
        if single_ticker:
            # Single-Ticker: flat columns -> df[col]
            if col in df.columns:
                return df[col].dropna()
            return None
        # Multi-Ticker: MultiIndex ('Close', 'ADI') oder ('ADI', 'Close')
        if hasattr(df.columns, "levels"):
            if col in df.columns.get_level_values(0):
                return df[col][ticker].dropna() if ticker in df[col].columns else None
            if ticker in df.columns.get_level_values(0):
                return df[ticker][col].dropna() if col in df[ticker].columns else None
        return None
    except Exception:
        return None


def _yahoo_chart_api(ticker: str, interval: str = "1m", range_: str = "1d"):
    """Yahoo v8 chart endpoint - direkter HTTP-Fetch. Robuster als yfinance bei Throttle."""
    import urllib.request, json as _json
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval={interval}&range={range_}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = _json.loads(r.read().decode("utf-8"))
    res = data["chart"]["result"][0]
    meta = res.get("meta", {})
    return meta, res.get("indicators", {}).get("quote", [{}])[0]


def batch_prices(tickers: list[str]) -> dict[str, float]:
    """Return latest available price per ticker. Empty dict on failure."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
    except ImportError:
        log("yfinance not installed; cannot fetch prices")
        return {}

    result: dict[str, float] = {}
    single = (len(tickers) == 1)

    # 1. yfinance 1m intraday
    try:
        df = yf.download(
            tickers if not single else tickers[0],
            period="2d", interval="1m",
            progress=False, threads=False, auto_adjust=True,
        )
        for t in tickers:
            s = _extract_series(df, t, single, "Close")
            if s is not None and len(s):
                result[t] = float(s.iloc[-1])
        if result:
            log(f"prices: 1m yfinance OK ({len(result)}/{len(tickers)})")
    except Exception as e:
        log(f"1m yfinance failed: {e}")

    # 2. yfinance 5m fallback
    missing = [t for t in tickers if t not in result]
    if missing:
        try:
            m_single = (len(missing) == 1)
            df = yf.download(
                missing if not m_single else missing[0],
                period="2d", interval="5m",
                progress=False, threads=False, auto_adjust=True,
            )
            for t in missing:
                s = _extract_series(df, t, m_single, "Close")
                if s is not None and len(s):
                    result[t] = float(s.iloc[-1])
            log(f"prices: 5m fallback ({len(result)}/{len(tickers)})")
        except Exception as e:
            log(f"5m yfinance failed: {e}")

    # 3. Yahoo v8 chart API direkt (robusteste Methode)
    missing = [t for t in tickers if t not in result]
    if missing:
        log(f"prices: trying v8 chart API for {missing}")
        for t in missing:
            try:
                meta, _ = _yahoo_chart_api(t, "1m", "1d")
                px = meta.get("regularMarketPrice")
                if px:
                    result[t] = float(px)
                    log(f"  v8 chart OK: {t} = ${px}")
            except Exception as e:
                log(f"  v8 chart failed {t}: {e}")

    # 4. Daily-Close als letzte Notbremse
    missing = [t for t in tickers if t not in result]
    if missing:
        try:
            m_single = (len(missing) == 1)
            df = yf.download(
                missing if not m_single else missing[0],
                period="5d", interval="1d",
                progress=False, threads=False, auto_adjust=True,
            )
            for t in missing:
                s = _extract_series(df, t, m_single, "Close")
                if s is not None and len(s):
                    result[t] = float(s.iloc[-1])
                    log(f"  daily fallback: {t}")
        except Exception as e:
            log(f"daily fallback failed: {e}")

    if not result:
        log(f"WARN: NO price for any ticker {tickers}")
    return result


def get_today_high(tickers: list[str]) -> dict[str, float]:
    """Return today's intraday HIGH per ticker (for trigger check on intraday-runs)."""
    if not tickers:
        return {}
    result = {}
    try:
        import yfinance as yf
        single = (len(tickers) == 1)
        df = yf.download(
            tickers if not single else tickers[0],
            period="1d", interval="1m",
            progress=False, threads=False, auto_adjust=True,
        )
        for t in tickers:
            s = _extract_series(df, t, single, "High")
            if s is not None and len(s):
                result[t] = float(s.max())
    except Exception as e:
        log(f"high yfinance failed: {e}")
    # Fallback v8 chart API
    missing = [t for t in tickers if t not in result]
    for t in missing:
        try:
            meta, quote = _yahoo_chart_api(t, "1m", "1d")
            highs = [h for h in (quote.get("high") or []) if h is not None]
            if highs:
                result[t] = float(max(highs))
            elif meta.get("regularMarketDayHigh"):
                result[t] = float(meta["regularMarketDayHigh"])
        except Exception as e:
            log(f"  high v8 fail {t}: {e}")
    return result


# ---------------------------------------------------------------------------
# Telegram-aequivalentes Gate
# ---------------------------------------------------------------------------
def passes_tg_gate(sig: dict) -> bool:
    setup = sig.get("setup", "")
    if setup not in ALLOWED_SETUPS:
        return False
    min_score = TG_MIN_SCORE.get(setup, TG_MIN_SCORE_DEFAULT)
    return (
        f(sig.get("score"))      >= min_score and
        f(sig.get("rr"))         >= TG_MIN_RR and
        f(sig.get("upside_pct")) >= TG_MIN_UPSIDE
    )


# ---------------------------------------------------------------------------
# Momentum-Filler (2026-06-12) — fuellt Slots wenn Scanner-Signale nicht reichen
# Tages-Cache: yfinance Bulk-Download max 1× pro Run-Set, nicht jeden 15-Min-Run.
# ---------------------------------------------------------------------------
def _wilder_rsi(close_series, period: int = 14) -> float:
    """RSI(14) am letzten Punkt. Returns None bei zu wenig Daten."""
    if len(close_series) < period + 1:
        return None
    delta = close_series.diff().dropna()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    avg_up = up.rolling(period).mean()
    avg_dn = dn.rolling(period).mean()
    if avg_dn.iloc[-1] == 0:
        return 100.0
    rs = avg_up.iloc[-1] / avg_dn.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def fetch_momentum_universe() -> list:
    """Holt Top-N US-Tickers, computed Momentum-Filter + Score, cached.
    Returns liste von candidate-dicts (kompatibel zur pending-Struktur)."""
    if not US_TICKERS.exists():
        log("momentum: us_tickers.txt fehlt, skip")
        return []
    with open(US_TICKERS, "r", encoding="utf-8") as fh:
        tickers = [t.strip().upper() for t in fh if t.strip()][:MOMENTUM_UNIVERSE_SIZE]
    if not tickers:
        return []

    try:
        import yfinance as yf
    except ImportError:
        return []

    log(f"momentum: fetching {len(tickers)} tickers (1mo daily)...")
    try:
        df = yf.download(
            tickers, period="1mo", interval="1d",
            progress=False, threads=True, auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        log(f"momentum: yfinance fail {e}")
        return []

    today = today_date().isoformat()
    candidates = []
    for t in tickers:
        try:
            if len(tickers) == 1:
                tdf = df
            elif hasattr(df.columns, "levels"):
                if t not in df.columns.get_level_values(0):
                    continue
                tdf = df[t]
            else:
                continue
            close = tdf["Close"].dropna()
            vol = tdf["Volume"].dropna()
            if len(close) < 20 or len(vol) < 20:
                continue

            cur = float(close.iloc[-1])
            if cur < MOMENTUM_PRICE_MIN:
                continue

            perf_5d  = (cur / float(close.iloc[-5]) - 1) * 100
            perf_20d = (cur / float(close.iloc[-20]) - 1) * 100
            vol_5  = float(vol.iloc[-5:].mean())
            vol_20 = float(vol.iloc[-20:].mean())
            vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 0
            rsi = _wilder_rsi(close, 14)

            # Filter
            if rsi is None: continue
            if perf_5d < MOMENTUM_PERF_5D_MIN: continue
            if rsi > MOMENTUM_RSI_MAX: continue
            if vol_ratio < MOMENTUM_VOL_RATIO_MIN: continue

            # Momentum-Score (eigene Skala, NICHT mit Scanner vergleichbar)
            #   perf_5d: max +48 (bei 12 % capped)
            #   perf_20d positive: max +15
            #   vol_ratio: max +18
            #   RSI sweet 50-65: +10
            score = (
                min(perf_5d, 12) * 4 +
                min(max(perf_20d, 0), 30) * 0.5 +
                min(vol_ratio, 3.0) * 6 +
                (10 if 50 <= rsi <= 65 else 5 if rsi < 70 else 0)
            )
            if score < MOMENTUM_MIN_SCORE: continue

            candidates.append({
                "ticker": t,
                "setup": "MOMENTUM",
                "source": "momentum_filler",
                "date": today,
                "price": round(cur, 2),
                "buy_above": round(cur * (1 + MOMENTUM_ENTRY_BUFFER), 2),
                "stop": round(cur * (1 - MOMENTUM_SL_PCT), 2),
                "target": round(cur * (1 + MOMENTUM_TP_PCT), 2),
                "score": round(score, 1),
                "perf_5d": round(perf_5d, 1),
                "perf_20d": round(perf_20d, 1),
                "rsi": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "rr": round(MOMENTUM_TP_PCT / MOMENTUM_SL_PCT, 2),
                "upside_pct": round(MOMENTUM_TP_PCT * 100, 1),
                "sector": "Unknown",
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: -x["score"])
    candidates = candidates[:50]

    save_json(MOMENTUM_CACHE, {
        "updated_at": now_iso(),
        "candidates": candidates,
        "universe_size": len(tickers),
    })
    log(f"momentum: {len(candidates)} kandidaten gefunden (cached)")
    return candidates


def load_momentum_candidates() -> list:
    """Lade Cache wenn frisch, sonst refresh."""
    cache = load_json(MOMENTUM_CACHE)
    if cache and isinstance(cache, dict):
        try:
            ts = cache.get("updated_at", "").replace("Z", "+00:00")
            updated = datetime.fromisoformat(ts)
            age = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
            if age < MOMENTUM_CACHE_MAX_AGE_H:
                cands = cache.get("candidates", [])
                log(f"momentum: cache hit ({age:.1f}h alt, {len(cands)} kandidaten)")
                return cands
        except Exception:
            pass
    return fetch_momentum_universe()


def select_momentum_fillers(state: dict, free_slots: int) -> list:
    """Erstellt pending-Eintraege fuer Momentum-Filler.
    Wird nur aufgerufen wenn Scanner-Signale die Slots nicht voll fuellen."""
    if free_slots <= 0:
        return []
    candidates = load_momentum_candidates()
    if not candidates:
        return []

    busy = {p["ticker"] for p in state.get("open", [])}
    busy |= {p["ticker"] for p in state.get("pending", [])}

    # Skip historisch geschlossene desselben Tages (nicht zwei mal pro Tag)
    today = today_date().isoformat()
    tracked = {(p.get("ticker"), p.get("signal_date")) for p in
               state.get("closed", []) + state.get("expired", [])}

    fillers = []
    for c in candidates:
        if len(fillers) >= free_slots:
            break
        if c["ticker"] in busy:
            continue
        if (c["ticker"], c["date"]) in tracked:
            continue
        fillers.append({
            "id": f"PAPER_MOM_{c['ticker']}_{c['date']}",
            "ticker": c["ticker"],
            "setup": c["setup"],
            "source": c["source"],
            "sector": c.get("sector", "Unknown"),
            "signal_date": c["date"],
            "entry":        f(c["buy_above"]),
            "stop_initial": f(c["stop"]),
            "target":       f(c["target"]),
            "score":        f(c["score"]),
            "rr":           f(c["rr"]),
            "upside_pct":   f(c["upside_pct"]),
            "added_at": now_iso(),
            "status": "pending",
            "mode": TRADING_MODE,
        })
        busy.add(c["ticker"])
    return fillers


# ---------------------------------------------------------------------------
# Intraday-Momentum-Catcher (2026-06-18) — EXPERIMENT
# ---------------------------------------------------------------------------
def _is_eod_utc() -> bool:
    """True wenn aktuelle UTC-Zeit >= INTRADAY_EOD_UTC (Hard-Close-Fenster)."""
    try:
        hh, mm = (int(x) for x in INTRADAY_EOD_UTC.split(":"))
        now = datetime.now(timezone.utc)
        return (now.hour, now.minute) >= (hh, mm)
    except Exception:
        return False


def fetch_intraday_signals() -> list:
    """Scant die (bereits gefilterten) Daily-Momentum-Kandidaten auf INTRADAY-Momentum.
    Leichter Fetch: nur die ~50 Momentum-Namen, 5m-Bars von heute. Returns ranked candidates."""
    pool = load_momentum_candidates()  # ~50 vorgefilterte Namen (reuse, kein extra Universe-Fetch)
    if not pool:
        log("intraday: kein momentum-pool")
        return []
    tickers = [c["ticker"] for c in pool]

    try:
        import yfinance as yf
    except ImportError:
        return []

    log(f"intraday: scanning {len(tickers)} momentum-namen (5m bars heute)...")
    try:
        df = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            period="1d", interval="5m",
            progress=False, threads=True, auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        log(f"intraday: yfinance fail {e}")
        return []

    cands = []
    for t in tickers:
        try:
            if len(tickers) == 1:
                tdf = df
            elif hasattr(df.columns, "levels"):
                if t not in df.columns.get_level_values(0):
                    continue
                tdf = df[t]
            else:
                continue
            o = tdf["Open"].dropna()
            h = tdf["High"].dropna()
            l = tdf["Low"].dropna()
            c = tdf["Close"].dropna()
            v = tdf["Volume"].dropna()
            if len(c) < 4:   # zu wenig Bars heute
                continue

            open_today = float(o.iloc[0])
            last       = float(c.iloc[-1])
            hi_today   = float(h.max())
            lo_today   = float(l.min())
            if last < INTRADAY_PRICE_MIN or open_today <= 0:
                continue

            gain_from_open = (last / open_today - 1) * 100
            # VWAP heute
            typical = (h + l + c) / 3
            vwap = float((typical * v).sum() / v.sum()) if float(v.sum()) > 0 else last
            above_vwap = last >= vwap
            # Position in Tagesspanne (1.0 = am High, 0 = am Low)
            rng = hi_today - lo_today
            range_pos = (last - lo_today) / rng if rng > 0 else 0.5

            # === Intraday-Entry-Filter ===
            if not (INTRADAY_GAIN_MIN <= gain_from_open <= INTRADAY_GAIN_MAX):
                continue
            if not above_vwap:
                continue
            if range_pos < INTRADAY_RANGE_POS_MIN:
                continue

            # Score: gain + range-position + vwap-distanz (alles intraday)
            score = gain_from_open * 3 + range_pos * 20 + ((last / vwap - 1) * 100) * 2
            cands.append({
                "ticker": t,
                "last": round(last, 2),
                "gain_from_open": round(gain_from_open, 2),
                "range_pos": round(range_pos, 2),
                "above_vwap": above_vwap,
                "score": round(score, 1),
            })
        except Exception:
            continue

    cands.sort(key=lambda x: -x["score"])
    save_json(INTRADAY_CACHE, {"updated_at": now_iso(), "candidates": cands})
    log(f"intraday: {len(cands)} kandidaten (top: {[c['ticker'] for c in cands[:3]]})")
    return cands


def select_intraday_plays(state: dict, dry_run: bool = False) -> list:
    """Oeffnet Intraday-Plays DIREKT (market entry, kein pending/trigger).
    Nur waehrend Marktstunden, vor EOD-Fenster, mit freien Slots + Cash + Sub-Limit."""
    events = []
    if not INTRADAY_ENABLED:
        return events
    if _is_eod_utc():
        log("intraday: EOD-Fenster -> keine neuen Entries")
        return events

    open_pos = state.get("open", [])
    # Sub-Limit: max INTRADAY_MAX_POSITIONS gleichzeitig
    intraday_open = [p for p in open_pos if p.get("source") == "intraday_momentum"]
    if len(intraday_open) >= INTRADAY_MAX_POSITIONS:
        log(f"intraday: sub-limit erreicht ({len(intraday_open)}/{INTRADAY_MAX_POSITIONS})")
        return events
    # Globales Slot-Limit + Cash
    if len(open_pos) >= MAX_POSITIONS:
        log("intraday: globale slots voll")
        return events
    if state.get("cash", 0) < POSITION_SIZE:
        log("intraday: cash < position_size")
        return events

    cands = fetch_intraday_signals()
    if not cands:
        return events

    busy = {p["ticker"] for p in open_pos} | {p["ticker"] for p in state.get("pending", [])}
    # nicht denselben Ticker am selben Tag zweimal (auch nach Close)
    today = today_date().isoformat()
    tracked = {p.get("ticker") for p in state.get("closed", [])
               if str(p.get("closed_at", "")).startswith(today)
               and p.get("source") == "intraday_momentum"}

    n_free_global = MAX_POSITIONS - len(open_pos)
    n_free_intra  = INTRADAY_MAX_POSITIONS - len(intraday_open)
    can_open = min(n_free_global, n_free_intra)

    prices = batch_prices([c["ticker"] for c in cands[:can_open + 5]])
    opened = 0
    for c in cands:
        if opened >= can_open:
            break
        tk = c["ticker"]
        if tk in busy or tk in tracked:
            continue
        if state.get("cash", 0) < POSITION_SIZE:
            break
        cur = prices.get(tk, c["last"])
        if cur is None or cur < INTRADAY_PRICE_MIN:
            continue
        pending_like = {
            "id": f"PAPER_INTRA_{tk}_{today}_{datetime.now(timezone.utc).strftime('%H%M')}",
            "ticker": tk,
            "setup": "INTRADAY",
            "source": "intraday_momentum",
            "sector": "Unknown",
            "signal_date": today,
            "stop_initial": round(cur * (1 - INTRADAY_SL_PCT), 2),
            "target":       round(cur * (1 + INTRADAY_TP_PCT), 2),
            "score":        c["score"],
            "rr":           round(INTRADAY_TP_PCT / INTRADAY_SL_PCT, 2),
            "upside_pct":   round(INTRADAY_TP_PCT * 100, 1),
            "gain_from_open": c["gain_from_open"],
        }
        if not dry_run:
            open_position(state, pending_like, cur)
        events.append({
            "event": "intraday_open", "id": pending_like["id"], "ts": now_iso(),
            "ticker": tk, "entry": cur, "gain_from_open": c["gain_from_open"],
            "score": c["score"], "mode": TRADING_MODE,
        })
        log(f"  INTRADAY OPEN: {tk} @ ${cur:.2f} (gain_from_open {c['gain_from_open']:+.1f}%, "
            f"TP ${pending_like['target']}, SL ${pending_like['stop_initial']})")
        busy.add(tk)
        opened += 1
    return events


# ---------------------------------------------------------------------------
# Signal selection: Top-1 BREAKOUT per scan-day
# ---------------------------------------------------------------------------
def select_new_signals(state: dict, signals: list) -> list:
    """Return list of new pending entries to add (Top-1 BREAKOUT per scan-day)."""
    if not signals:
        return []

    # Freshness gate: nur Signale jünger als MAX_TRIGGER_DAYS aufnehmen
    today = today_date()
    cutoff = today - timedelta(days=MAX_TRIGGER_DAYS)

    # 2026-06-10: Top-1/Tag-Regel ABGESCHAFFT (fuehrte zu Unter-Auslastung:
    # 3 Slots leer + $200 idle waehrend mehrere Elite-Signale ignoriert wurden).
    # Neue Logik: ALLE qualifizierten BREAKOUTs der letzten MAX_TRIGGER_DAYS sammeln,
    # nach Score sortieren, freie Slots auffuellen.

    # Tracked-Keys (schon gesehen) + bereits gehaltene/pending Tickers
    tracked_keys = set()
    for bucket in ("pending", "open", "closed", "expired"):
        for p in state.get(bucket, []):
            tracked_keys.add((p.get("ticker"), p.get("signal_date")))
    busy_tickers = {p["ticker"] for p in state.get("open", [])}
    busy_tickers |= {p["ticker"] for p in state.get("pending", [])}

    # Alle qualifizierten, frischen Kandidaten flach sammeln
    candidates = []
    for s in signals:
        if not passes_tg_gate(s):
            continue
        d = s.get("date")
        if not d:
            continue
        try:
            d_dt = datetime.fromisoformat(d).date()
            if d_dt < cutoff:
                continue
        except Exception:
            continue
        ticker = s["ticker"]
        if ticker in busy_tickers:
            continue                          # schon offen oder pending
        if (ticker, d) in tracked_keys:
            continue                          # diesen (Ticker, Datum)-Eintrag schon getrackt
        candidates.append(s)

    if not candidates:
        return []

    # Dedup pro Ticker: behalte hoechsten Score (gleicher Ticker an mehreren Tagen)
    best_per_ticker = {}
    for s in candidates:
        t = s["ticker"]
        if t not in best_per_ticker or f(s.get("score")) > f(best_per_ticker[t].get("score")):
            best_per_ticker[t] = s
    candidates = list(best_per_ticker.values())

    # Nach Score sortieren (beste zuerst)
    candidates.sort(key=lambda x: f(x.get("score")), reverse=True)

    # Freie Slots berechnen: MAX_POSITIONS minus (open + pending)
    used_slots = len(state.get("open", [])) + len(state.get("pending", []))
    free_slots = max(0, MAX_POSITIONS - used_slots)
    if free_slots == 0:
        return []

    new_pending = []
    for top in candidates[:free_slots]:
        sigdate = top["date"]
        new_pending.append({
            "id": f"PAPER_{top['ticker']}_{sigdate}",
            "ticker": top["ticker"],
            "setup": top["setup"],
            "source": "scanner",
            "sector": top.get("sector", "Unknown"),
            "signal_date": sigdate,
            "entry":  f(top.get("buy_above")),
            "stop_initial": f(top.get("stop")),
            "target": f(top.get("target")),
            "score": f(top.get("score")),
            "rr":    f(top.get("rr")),
            "upside_pct": f(top.get("upside_pct")),
            "added_at": now_iso(),
            "status": "pending",
            "mode": TRADING_MODE,
        })
    return new_pending


# ---------------------------------------------------------------------------
# Manual Overrides (Phase 2) — User/Claude editiert apex_manual_overrides.json,
# Trader liest jeden Run und wendet noch-nicht-applied Eintraege an.
# ---------------------------------------------------------------------------
def apply_manual_overrides(state: dict, dry_run: bool = False) -> list:
    """Liest apex_manual_overrides.json, wendet alle Eintraege mit applied_at=null an.

    Schema (Key = Ticker, Wert = Override-Dict):
        {
          "AFRM": {
            "sl":    70.50,      // optional: SL setzen (max(old, new) -> niemals nach unten)
            "tp":    null,       // optional: TP ueberschreiben
            "close": false,      // optional: sofort schliessen
            "note":  "SL auf BE gezogen",
            "set_at":     "2026-06-07T15:30:00Z",
            "applied_at": null   // wird vom Trader gesetzt
          }
        }
    """
    overrides = load_json(OVERRIDES_FILE)
    if not overrides or not isinstance(overrides, dict):
        return []
    events = []
    changed = False
    for key, ov in overrides.items():
        if key.startswith("_"):
            continue   # _meta, _instructions, etc.
        if not isinstance(ov, dict):
            continue
        if ov.get("applied_at"):
            continue   # bereits angewandt

        # Position via Ticker finden (in open list)
        pos = next((p for p in state["open"] if p["ticker"] == key), None)
        if pos is None:
            log(f"  override: {key} - keine offene Position gefunden, skip")
            ov["applied_at"] = now_iso()
            ov["apply_result"] = "no_open_position"
            changed = True
            continue

        # SL
        if ov.get("sl") is not None:
            new_sl = f(ov["sl"])
            old_sl = pos["stop"]
            pos["stop"] = max(old_sl, new_sl)
            # Trailing-Ladder konsistent halten: wenn manuell hoeher als naechste Stufe -> Stufe upgrade
            entry = pos.get("entry_actual", pos.get("entry"))
            if entry:
                for step_idx, (_, sl_mult) in enumerate(TRAIL_LADDER, start=1):
                    if pos["stop"] >= entry * sl_mult:
                        pos["ladder_step"] = max(int(pos.get("ladder_step", 0)), step_idx)
                pos["trailing_active"] = pos["ladder_step"] > 0 or pos.get("trailing_active", False)
            events.append({
                "event": "manual_override", "id": pos["id"], "ts": now_iso(),
                "ticker": pos["ticker"], "field": "sl",
                "old": old_sl, "new": pos["stop"],
                "note": ov.get("note", ""),
            })
            log(f"  override: {key} SL ${old_sl:.2f} -> ${pos['stop']:.2f}  ({ov.get('note', '')})")

        # TP
        if ov.get("tp") is not None:
            new_tp = f(ov["tp"])
            old_tp = pos["target"]
            pos["target"] = new_tp
            events.append({
                "event": "manual_override", "id": pos["id"], "ts": now_iso(),
                "ticker": pos["ticker"], "field": "tp",
                "old": old_tp, "new": new_tp,
                "note": ov.get("note", ""),
            })
            log(f"  override: {key} TP ${old_tp:.2f} -> ${new_tp:.2f}  ({ov.get('note', '')})")

        # CLOSE NOW
        if ov.get("close") is True:
            cur = pos.get("current_price", pos["entry_actual"])
            if not dry_run:
                close_position(state, pos, cur, "Manual Close")
            events.append({
                "event": "manual_override", "id": pos["id"], "ts": now_iso(),
                "ticker": pos["ticker"], "field": "close",
                "exit_price": cur, "note": ov.get("note", ""),
            })
            log(f"  override: {key} CLOSE @ ${cur:.2f}  ({ov.get('note', '')})")

        ov["applied_at"] = now_iso()
        ov["apply_result"] = "ok"
        changed = True

    if changed and not dry_run:
        save_json(OVERRIDES_FILE, overrides)
    return events


# ---------------------------------------------------------------------------
# Replacement-Logik (Phase 1) — voller Slot, neues Top-Signal kommt
# ---------------------------------------------------------------------------
def find_signal_in_pool(ticker: str, signal_date: str) -> dict | None:
    """Lookup original signal dict aus apex_signals.json (fuer Catalyst-Check)."""
    sigs = load_json(SIGNALS_FILE) or []
    for s in sigs:
        if s.get("ticker") == ticker and s.get("date") == signal_date:
            return s
    return None


def is_replacement_eligible(pending: dict, state: dict) -> tuple[bool, str, dict | None]:
    """Prueft ob Pending-Signal eine bestehende Position ersetzen darf.
    Returns (eligible, reason, position_to_replace_or_None)."""
    open_pos = state.get("open") or []
    if len(open_pos) < MAX_POSITIONS:
        return False, "slot frei, kein Replacement noetig", None

    # 1. Neues Signal muss Score >= REPLACEMENT_MIN_SCORE
    new_score = f(pending.get("score"))
    if new_score < REPLACEMENT_MIN_SCORE:
        return False, f"score {new_score:.0f} < {REPLACEMENT_MIN_SCORE:.0f}", None

    # 2. Catalyst-Requirement: Pocket Pivot ODER Gap >= 2%
    sig = find_signal_in_pool(pending["ticker"], pending["signal_date"])
    if sig is None:
        return False, "Original-Signal nicht gefunden im apex_signals.json", None
    has_pp  = bool(sig.get("cat_pocket_pivot"))
    gap_pct = f(sig.get("cat_gap_pct"))
    has_gap = gap_pct >= 2.0
    if not (has_pp or has_gap):
        return False, "kein Pocket Pivot und kein Gap >= 2%", None

    # 3. Schwaechste Position finden (lowest current pnl_pct)
    weakest = min(open_pos, key=lambda p: f(p.get("pnl_pct")))
    weakest_pnl = f(weakest.get("pnl_pct"))

    # 4. Schwaechste muss im Plus sein (mind. REPLACEMENT_WEAKEST_MIN_PNL)
    if weakest_pnl < REPLACEMENT_WEAKEST_MIN_PNL:
        return False, (f"schwaechste {weakest['ticker']} bei {weakest_pnl:+.2f} % "
                       f"(< +{REPLACEMENT_WEAKEST_MIN_PNL} %)"), None

    return True, (f"OK: ersetze {weakest['ticker']} ({weakest_pnl:+.2f} %) "
                  f"durch {pending['ticker']} (score {new_score:.0f}, "
                  f"PP={has_pp}, Gap={gap_pct:.1f}%)"), weakest


# ---------------------------------------------------------------------------
# Pending -> Open (Trigger check)
# ---------------------------------------------------------------------------
def trigger_pending(state: dict, dry_run: bool = False) -> list:
    """Walk pending list, expire stale, trigger if price >= entry. Returns events."""
    if not state["pending"]:
        return []

    today = today_date()
    events = []
    still_pending = []

    # Bulk price fetch: today's high for ALL pending tickers
    tickers = list({p["ticker"] for p in state["pending"]})
    highs = get_today_high(tickers)
    prices = batch_prices(tickers)

    # Re-Validation-Lookup: Ticker+Setup, die in der LATEST Scan-Date erneut auftauchen,
    # gelten als "re-validiert" und kriegen ihren signal_date refreshed (kein Expire).
    latest_signals = load_json(SIGNALS_FILE) or []
    latest_scan_date = max((s.get("date") for s in latest_signals if s.get("date")),
                           default=None)
    revalidated_map = {}
    if latest_scan_date:
        for s in latest_signals:
            if (s.get("date") == latest_scan_date and
                s.get("setup") in ALLOWED_SETUPS and
                passes_tg_gate(s)):
                revalidated_map[(s["ticker"], s["setup"])] = s

    for p in state["pending"]:
        # 1. Expiry check
        try:
            sig_date = datetime.fromisoformat(p["signal_date"]).date()
            days_since = (today - sig_date).days
        except Exception:
            days_since = 0

        if days_since > MAX_TRIGGER_DAYS:
            # Re-Validierung: taucht (Ticker, Setup) in der neuesten Scan wieder auf?
            revalid = revalidated_map.get((p["ticker"], p["setup"]))
            if revalid and revalid.get("date") != p.get("signal_date"):
                # Frisch validiert -> Signal-Daten refreshen, signal_date update
                old_sd = p["signal_date"]
                p["signal_date"]   = revalid["date"]
                p["entry"]         = f(revalid.get("buy_above"))
                p["stop_initial"]  = f(revalid.get("stop"))
                p["target"]        = f(revalid.get("target"))
                p["score"]         = f(revalid.get("score"))
                p["rr"]            = f(revalid.get("rr"))
                p["upside_pct"]    = f(revalid.get("upside_pct"))
                p["revalidated_at"] = now_iso()
                events.append({
                    "event": "revalidated", "id": p["id"], "ts": now_iso(),
                    "ticker": p["ticker"], "old_signal_date": old_sd,
                    "new_signal_date": revalid["date"],
                })
                log(f"  revalidated: {p['ticker']} ({old_sd} -> {revalid['date']})")
                # Trigger-Check unten greift mit neuen Daten
            else:
                p["status"] = "expired"
                p["expired_at"] = now_iso()
                state["expired"].append(p)
                events.append({"event": "expired", "id": p["id"], "ts": now_iso(),
                               "ticker": p["ticker"],
                               "reason": f"no trigger after {days_since}d, nicht re-validiert"})
                log(f"  expired: {p['ticker']} (signal {p['signal_date']}, "
                    f"{days_since}d alt, nicht re-validiert)")
                continue

        # 2. Capacity & cash checks
        # === Phase-1 Replacement: wenn Slots voll, pruefe ob Pending qualifiziert ===
        if len(state["open"]) >= MAX_POSITIONS:
            eligible, reason, to_replace = is_replacement_eligible(p, state)
            if eligible and to_replace is not None:
                # Trigger-Preis trotzdem checken bevor wir was anfassen
                high_today = highs.get(p["ticker"])
                if high_today is None or high_today < p["entry"]:
                    still_pending.append(p)
                    continue
                # OK: schwaechste Position schliessen, Slot wird frei
                log(f"  REPLACEMENT: {reason}")
                cur_replace = prices.get(to_replace["ticker"], to_replace.get("entry_actual"))
                if not dry_run:
                    # Use close_position helper to keep stats clean
                    close_position(state, to_replace, cur_replace, "Replacement Exit")
                events.append({
                    "event": "replacement", "id": p["id"], "ts": now_iso(),
                    "ticker": p["ticker"], "replaced_ticker": to_replace["ticker"],
                    "replaced_pnl_pct": f(to_replace.get("pnl_pct")),
                    "reason": reason,
                })
                # state["open"] hat jetzt einen Slot frei -> fallthrough zur normalen Trigger-Logik unten
            else:
                still_pending.append(p)
                continue

        if state["cash"] < POSITION_SIZE:
            still_pending.append(p)
            continue

        # 3. Trigger check: today's HIGH must touch entry
        high_today = highs.get(p["ticker"])
        cur_price = prices.get(p["ticker"])
        if high_today is None or cur_price is None:
            still_pending.append(p)  # data fail: retry next run
            continue

        if high_today >= p["entry"]:
            # FIRE
            entry_actual = max(p["entry"], min(cur_price, p["entry"] * 1.005))
            # Simulate fill: at entry if high>=entry, but cap slippage at +0.5%
            if not dry_run:
                open_position(state, p, entry_actual)
            events.append({
                "event": "open", "id": p["id"], "ts": now_iso(),
                "ticker": p["ticker"], "entry_signal": p["entry"],
                "entry_actual": entry_actual, "mode": TRADING_MODE,
            })
            log(f"  TRIGGER: {p['ticker']} @ ${entry_actual:.2f} (signal entry ${p['entry']:.2f}, high ${high_today:.2f})")
        else:
            still_pending.append(p)

    state["pending"] = still_pending
    return events


def open_position(state: dict, pending: dict, entry_actual: float):
    """Move from pending to open."""
    shares = POSITION_SIZE / entry_actual
    pos = {
        **pending,
        "entry_actual": entry_actual,
        "shares": shares,
        "size_usd": POSITION_SIZE,
        "stop":  pending["stop_initial"],
        "high_since_entry": entry_actual,
        "trailing_active": False,
        "ladder_step": 0,             # Phase-1: noch keine Trail-Stufe aktiv
        "opened_at": now_iso(),
        "status": "open",
        "current_price": entry_actual,
        "pnl_pct": 0.0,
        "pnl_usd": 0.0,
        "hold_days": 0,
        "source": pending.get("source", "scanner"),
    }
    state["open"].append(pos)
    state["cash"] -= POSITION_SIZE

    if TRADING_MODE == "live":
        try:
            etoro_open_position(pos)
        except Exception as e:
            log(f"  ERR live-open: {e}")


# ---------------------------------------------------------------------------
# Open -> Closed (TP / SL / Trailing / Time)
# ---------------------------------------------------------------------------
def update_open_positions(state: dict, dry_run: bool = False) -> list:
    """Refresh prices, manage trailing, close if TP/SL/time hit."""
    if not state["open"]:
        return []

    events = []
    tickers = list({p["ticker"] for p in state["open"]})
    prices = batch_prices(tickers)
    highs = get_today_high(tickers)

    still_open = []
    for p in state["open"]:
        cur = prices.get(p["ticker"])
        high = highs.get(p["ticker"], cur)
        if cur is None:
            log(f"  no price for {p['ticker']} — keeping open")
            still_open.append(p)
            continue

        entry = p["entry_actual"]

        # === Intraday-Catcher: eigene Exit-Logik (TP/SL/EOD), KEIN Ladder/Stagnation/Time ===
        if p.get("source") == "intraday_momentum":
            high_i = high if high is not None else cur
            i_reason, i_px = None, cur
            if high_i >= p["target"]:
                i_reason, i_px = "Intraday TP", p["target"]
            elif cur <= p["stop"]:
                i_reason, i_px = "Intraday Stop", p["stop"]
            elif _is_eod_utc():
                i_reason, i_px = "Intraday Close (EOD)", cur
            if i_reason:
                if not dry_run:
                    close_position(state, p, i_px, i_reason)
                events.append({
                    "event": "close", "id": p["id"], "ts": now_iso(),
                    "ticker": p["ticker"], "exit_price": i_px, "reason": i_reason,
                    "pnl_pct": (i_px - entry) / entry * 100,
                })
                log(f"  CLOSE: {p['ticker']} @ ${i_px:.2f} ({i_reason}, "
                    f"{(i_px - entry) / entry * 100:+.2f}%)")
                continue
            p["current_price"] = cur
            p["pnl_pct"] = (cur - entry) / entry * 100
            p["pnl_usd"] = p["size_usd"] * p["pnl_pct"] / 100
            still_open.append(p)
            continue

        # Update high-since-entry (use today's high if available)
        if high is not None and high > p.get("high_since_entry", entry):
            p["high_since_entry"] = high

        # === Trailing-Ladder (Phase 1) — progressive Stufen ===
        # ladder_step: 0 = noch nicht aktiviert, 1/2/3 = Stufe erreicht
        current_step = int(p.get("ladder_step", 0))
        high_se = p["high_since_entry"]
        # Pruefe ob naechste Stufe erreicht ist
        for step_idx, (trigger_mult, sl_mult) in enumerate(TRAIL_LADDER, start=1):
            if step_idx <= current_step:
                continue   # diese Stufe ist schon aktiv
            if high_se >= entry * trigger_mult:
                new_stop = entry * sl_mult
                old_stop = p["stop"]
                p["stop"] = max(old_stop, new_stop)   # niemals nach unten
                p["ladder_step"] = step_idx
                p["trailing_active"] = True
                p["trailing_activated_at"] = p.get("trailing_activated_at") or now_iso()
                events.append({
                    "event": "trailing_activated", "id": p["id"], "ts": now_iso(),
                    "ticker": p["ticker"], "old_stop": old_stop, "new_stop": p["stop"],
                    "high": high_se, "ladder_step": step_idx,
                })
                log(f"  trailing step {step_idx}: {p['ticker']} SL ${old_stop:.2f} -> ${p['stop']:.2f}")
                current_step = step_idx
            else:
                break   # naechste Stufe nicht erreicht -> abbrechen

        # === Hold-Days berechnen (fuer Stagnation + Time-Exit) ===
        try:
            opened_str = p["opened_at"].replace("Z", "+00:00")
            opened = datetime.fromisoformat(opened_str)
            hold = (datetime.now(timezone.utc) - opened).days
            p["hold_days"] = hold
        except Exception:
            hold = p.get("hold_days", 0)

        # === Exit checks (Reihenfolge: TP > SL > Stagnation > Time) ===
        exit_reason = None
        exit_price = cur

        if high is not None and high >= p["target"]:
            exit_reason = "Take Profit"
            exit_price = p["target"]
        elif cur <= p["stop"]:
            exit_reason = "Stop Loss" + (" (Trailing)" if p.get("trailing_active") else "")
            exit_price = p["stop"]
        else:
            # Stagnations-Exit: ab Tag 5 mit flachem PnL (-2 % bis +2 %)
            pnl_pct = (cur - entry) / entry * 100
            if (hold >= STAGNATION_DAYS and
                STAGNATION_PNL_MIN <= pnl_pct <= STAGNATION_PNL_MAX):
                exit_reason = "Stagnation Exit"
                exit_price = cur
            else:
                # Time-Exit nach setup-spezifischem Hold-Limit
                hold_limit = hold_days_for(p.get("setup", ""))
                if hold >= hold_limit:
                    exit_reason = "Time Exit"
                    exit_price = cur

        if exit_reason:
            if not dry_run:
                close_position(state, p, exit_price, exit_reason)
            events.append({
                "event": "close", "id": p["id"], "ts": now_iso(),
                "ticker": p["ticker"], "exit_price": exit_price,
                "reason": exit_reason, "pnl_pct": (exit_price - entry) / entry * 100,
            })
            log(f"  CLOSE: {p['ticker']} @ ${exit_price:.2f} ({exit_reason}, "
                f"{(exit_price - entry) / entry * 100:+.2f}%)")
            continue

        # Still open: refresh live PnL
        p["current_price"] = cur
        p["pnl_pct"] = (cur - entry) / entry * 100
        p["pnl_usd"] = p["size_usd"] * p["pnl_pct"] / 100
        still_open.append(p)

    state["open"] = still_open
    return events


def close_position(state: dict, pos: dict, exit_price: float, reason: str):
    entry = pos["entry_actual"]
    pnl_pct = (exit_price - entry) / entry * 100
    pnl_usd = pos["size_usd"] * pnl_pct / 100
    closed = {
        **pos,
        "exit_price": exit_price,
        "exit_reason": reason,
        "closed_at": now_iso(),
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "status": "closed",
    }
    state["closed"].append(closed)
    # Cash flow: return position size + pnl
    state["cash"] += pos["size_usd"] + pnl_usd

    if TRADING_MODE == "live":
        try:
            etoro_close_position(pos, exit_price, reason)
        except Exception as e:
            log(f"  ERR live-close: {e}")


# ---------------------------------------------------------------------------
# Stats recompute
# ---------------------------------------------------------------------------
def recompute_stats(state: dict):
    closed = state["closed"]
    open_pos = state["open"]
    wins = sum(1 for c in closed if c.get("pnl_pct", 0) > 0)
    losses = sum(1 for c in closed if c.get("pnl_pct", 0) <= 0)
    pnl_real = sum(c.get("pnl_usd", 0) for c in closed)
    pnl_unr  = sum(p.get("pnl_usd", 0) for p in open_pos)
    state["stats"] = {
        "total_trades":   len(closed),
        "open_trades":    len(open_pos),
        "wins":           wins,
        "losses":         losses,
        "win_rate":       (wins / len(closed) * 100) if closed else 0.0,
        "pnl_realized":   pnl_real,
        "pnl_unrealized": pnl_unr,
        "equity":         state["cash"] + sum(p["size_usd"] + p.get("pnl_usd", 0)
                                              for p in open_pos),
    }


# ---------------------------------------------------------------------------
# eToro stubs (live mode)
# ---------------------------------------------------------------------------
def etoro_open_position(pos: dict):
    """Stub: POST /v1/portfolios/{account_id}/positions."""
    if not (ETORO_API_KEY and ETORO_ACCOUNT_ID):
        raise RuntimeError("ETORO_API_KEY / ETORO_ACCOUNT_ID missing")
    log(f"  [eToro] would OPEN {pos['ticker']} ${pos['size_usd']:.0f} "
        f"SL ${pos['stop']:.2f} TP ${pos['target']:.2f}")
    # TODO: implement actual eToro API call


def etoro_close_position(pos: dict, exit_price: float, reason: str):
    """Stub: DELETE /v1/portfolios/{account_id}/positions/{id}."""
    if not (ETORO_API_KEY and ETORO_ACCOUNT_ID):
        raise RuntimeError("ETORO_API_KEY / ETORO_ACCOUNT_ID missing")
    log(f"  [eToro] would CLOSE {pos['ticker']} @ ${exit_price:.2f} ({reason})")
    # TODO: implement actual eToro API call


# ---------------------------------------------------------------------------
# Status print
# ---------------------------------------------------------------------------
def print_status(state: dict):
    s = state["stats"]
    log(f"=== {state['mode'].upper()} STATUS ===")
    log(f"Cash:        ${state['cash']:.2f}")
    log(f"Equity:      ${s['equity']:.2f}")
    log(f"Open:        {s['open_trades']} positions")
    log(f"Closed:      {s['total_trades']} trades ({s['wins']}W/{s['losses']}L, "
        f"WR {s.get('win_rate', 0):.1f}%)")
    log(f"PnL real:    ${s['pnl_realized']:+.2f}")
    log(f"PnL unreal:  ${s['pnl_unrealized']:+.2f}")
    log(f"Pending:     {len(state['pending'])}")
    log(f"Expired:     {len(state['expired'])}")
    if state["open"]:
        log("--- OPEN ---")
        for p in state["open"]:
            trail = "T" if p.get("trailing_active") else " "
            log(f"  [{trail}] {p['ticker']:<6} "
                f"entry ${p['entry_actual']:.2f}  cur ${p.get('current_price', 0):.2f}  "
                f"SL ${p['stop']:.2f}  TP ${p['target']:.2f}  "
                f"PnL {p.get('pnl_pct', 0):+.2f}%")
    if state["pending"]:
        log("--- PENDING ---")
        for p in state["pending"]:
            log(f"  {p['ticker']:<6}  buy>=${p['entry']:.2f}  score {p['score']:.0f}  "
                f"signal {p['signal_date']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_trader(dry_run: bool = False):
    log(f"=== ApexTrader run ({TRADING_MODE.upper()} mode) ===")

    state = load_state()
    state["mode"] = TRADING_MODE
    state["last_updated"] = now_iso()

    all_events = []

    # 0. Manual Overrides anwenden (User/Claude via apex_manual_overrides.json)
    log("step 0: manual overrides")
    events0 = apply_manual_overrides(state, dry_run=dry_run)
    all_events.extend(events0)

    # 1. Update open positions (TP/SL/trailing/time-exit)
    log("step 1: update open positions")
    events1 = update_open_positions(state, dry_run=dry_run)
    all_events.extend(events1)

    # 2. Walk pending → trigger if cash + price ok
    log("step 2: trigger pending")
    events2 = trigger_pending(state, dry_run=dry_run)
    all_events.extend(events2)

    # 3a. Scanner-Signale (Prioritaet): BREAKOUT + STAGE_2 aus apex_signals.json
    log("step 3a: select scanner signals")
    signals = load_json(SIGNALS_FILE) or []
    new_pending = select_new_signals(state, signals)
    if new_pending:
        if not dry_run:
            state["pending"].extend(new_pending)
        for np_ in new_pending:
            all_events.append({
                "event": "pending_added", "id": np_["id"], "ts": now_iso(),
                "ticker": np_["ticker"], "signal_date": np_["signal_date"],
                "score": np_["score"], "source": np_.get("source", "scanner"),
            })
        log(f"  +{len(new_pending)} scanner pending: {[p['ticker'] for p in new_pending]}")
    else:
        log("  no scanner signals")

    # 3b. Momentum-Filler (Fallback): nur wenn Scanner-Signale + bestehende
    # Pending+Open die Slots nicht voll machen. yfinance Tages-Cache.
    used = len(state["open"]) + len(state["pending"])
    free_slots = max(0, MAX_POSITIONS - used)
    if free_slots > 0:
        log(f"step 3b: momentum fillers (free_slots={free_slots})")
        mom_pending = select_momentum_fillers(state, free_slots)
        if mom_pending:
            if not dry_run:
                state["pending"].extend(mom_pending)
            for np_ in mom_pending:
                all_events.append({
                    "event": "pending_added", "id": np_["id"], "ts": now_iso(),
                    "ticker": np_["ticker"], "signal_date": np_["signal_date"],
                    "score": np_["score"], "source": "momentum_filler",
                })
            log(f"  +{len(mom_pending)} momentum pending: {[p['ticker'] for p in mom_pending]}")
        else:
            log("  no momentum candidates")
    else:
        log("step 3b: skip momentum (slots voll)")

    # 3c. Intraday-Momentum-Catcher (EXPERIMENT, opt-in INTRADAY_ENABLED) — direkter Market-Entry
    if INTRADAY_ENABLED:
        log("step 3c: intraday momentum catcher")
        intra_events = select_intraday_plays(state, dry_run=dry_run)
        all_events.extend(intra_events)
        if not intra_events:
            log("  no intraday entries")
    else:
        log("step 3c: intraday disabled (INTRADAY_ENABLED=0)")

    # 4. Recompute stats
    recompute_stats(state)

    # 5. Persist
    if not dry_run:
        save_json(POSITIONS_FILE, state)
        # Always ensure trade-log file exists (even empty) for git workflow
        if all_events:
            append_log(all_events)
        elif not TRADE_LOG_FILE.exists():
            save_json(TRADE_LOG_FILE, [])
        log(f"saved: {POSITIONS_FILE.name} ({len(all_events)} events logged)")
    else:
        log(f"[DRY-RUN] would save with {len(all_events)} events")

    print_status(state)
    return state


def reset_state():
    if POSITIONS_FILE.exists():
        POSITIONS_FILE.unlink()
        log(f"deleted {POSITIONS_FILE.name}")
    if TRADE_LOG_FILE.exists():
        TRADE_LOG_FILE.unlink()
        log(f"deleted {TRADE_LOG_FILE.name}")
    log("state reset complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="loescht apex_positions.json + apex_trade_log.json")
    ap.add_argument("--dry-run", action="store_true", help="nichts schreiben, nur loggen")
    ap.add_argument("--status", action="store_true", help="aktuellen State anzeigen, kein Run")
    args = ap.parse_args()

    if args.reset:
        reset_state()
        return

    if args.status:
        state = load_state()
        print_status(state)
        return

    try:
        run_trader(dry_run=args.dry_run)
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
