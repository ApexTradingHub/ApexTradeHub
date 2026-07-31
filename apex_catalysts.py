"""
ApexScan — Catalyst Data from yfinance
=======================================
Earnings, short interest, analyst targets — cached to avoid repeated slow calls.

Public API:
  get_catalyst_data(ticker, force_refresh=False) -> dict (raw cached snapshot)
  derive_catalyst_signals(data, as_of_date=None)  -> dict (point-in-time signals)
  prewarm_catalysts(tickers)                       -> bulk-fill cache for backtest

Cache: catalyst_cache.json, TTL 24h.
For backtest: earnings_dates are point-in-time-correct (filter by date).
              short_pct, analyst_target are CURRENT snapshots (proxy with caveat).
"""

import contextlib
import io
import json
import sys
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CATALYST_CACHE_FILE = "catalyst_cache.json"
CATALYST_TTL_HOURS  = 24


@contextlib.contextmanager
def _suppress():
    o, e = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = o, e


def _load_cache():
    try:
        with open(CATALYST_CACHE_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CATALYST_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str, ensure_ascii=False)
    except Exception:
        pass


def _is_fresh(entry, ttl_h=CATALYST_TTL_HOURS):
    try:
        cached_at = datetime.fromisoformat(entry.get("_cached_at", "2000-01-01"))
        return (datetime.now() - cached_at).total_seconds() < ttl_h * 3600
    except Exception:
        return False


def fetch_catalyst_data(ticker: str) -> dict:
    """Pull fresh catalyst snapshot from yfinance. Returns dict (always serializable)."""
    out = {
        "_cached_at": datetime.now().isoformat(),
        "earnings": [],                    # [{date: ISO, surprise_pct: float|None}, ...]
        "short_pct_float": None,           # 0..100
        "analyst_target_upside_pct": None, # vs current price
    }

    try:
        with _suppress():
            t = yf.Ticker(ticker)

        # --- Earnings dates (historical + upcoming) ---
        try:
            with _suppress():
                ed = t.earnings_dates
            if ed is not None and not ed.empty:
                for idx, row in ed.iterrows():
                    try:
                        ts = pd.Timestamp(idx)
                        if ts.tz is not None:
                            ts = ts.tz_localize(None)
                        date_iso = ts.strftime("%Y-%m-%d")
                        surp = None
                        if "Surprise(%)" in ed.columns:
                            v = row.get("Surprise(%)")
                            if pd.notna(v):
                                surp = float(v)
                        out["earnings"].append({"date": date_iso, "surprise_pct": surp})
                    except Exception:
                        continue
        except Exception:
            pass

        # --- Info data: short interest, analyst target ---
        try:
            with _suppress():
                info = t.info
            if isinstance(info, dict):
                spf = info.get("shortPercentOfFloat")
                if spf is not None:
                    try:
                        out["short_pct_float"] = float(spf) * 100
                    except Exception:
                        pass
                target  = info.get("targetMeanPrice")
                current = info.get("currentPrice") or info.get("regularMarketPrice")
                if target and current:
                    try:
                        c = float(current)
                        if c > 0:
                            out["analyst_target_upside_pct"] = (float(target) / c - 1) * 100
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception:
        pass

    return out


def get_catalyst_data(ticker: str, force_refresh: bool = False) -> dict:
    """Cached fetch. Hits yfinance only if stale or forced."""
    cache = _load_cache()
    old = cache.get(ticker) if isinstance(cache.get(ticker), dict) else None
    if not force_refresh and old is not None and _is_fresh(old):
        return old
    data = fetch_catalyst_data(ticker)
    # MERGE-GUARD (2026-07-31): yfinance `t.earnings_dates` wird auf GitHub-Actions-
    # Datacenter-IPs geblockt und faellt in den stillen except -> earnings=[]. Der
    # committete Cache-Seed traegt gute Earnings-Termine; ein leerer Live-Fetch darf
    # sie NICHT ueberschreiben (sonst gehen -15 Blackout + +8 PEAD still verloren, was
    # ~30 Scan-Tage lang unbemerkt passiert ist). `t.info` (short/analyst) ueberlebt auf
    # Actions -> die werden normal aktualisiert, nur bei None faellt der Guard zurueck.
    if old is not None:
        if not data.get("earnings") and old.get("earnings"):
            data["earnings"] = old["earnings"]
            data["_earnings_stale"] = True   # audit: earnings aus altem Cache
        if data.get("short_pct_float") is None and old.get("short_pct_float") is not None:
            data["short_pct_float"] = old["short_pct_float"]
        if data.get("analyst_target_upside_pct") is None and old.get("analyst_target_upside_pct") is not None:
            data["analyst_target_upside_pct"] = old["analyst_target_upside_pct"]
    cache[ticker] = data
    _save_cache(cache)
    return data


def derive_catalyst_signals(catalyst_data: dict, as_of_date=None) -> dict:
    """
    Convert raw catalyst data → boolean signals for scoring.
    as_of_date: datetime or None (live=now). For backtest pass scan_date.
    """
    if as_of_date is None:
        as_of_date = datetime.now()
    if isinstance(as_of_date, str):
        try:
            as_of_date = datetime.fromisoformat(as_of_date)
        except Exception:
            as_of_date = datetime.now()
    if hasattr(as_of_date, 'to_pydatetime'):
        as_of_date = as_of_date.to_pydatetime()
    if hasattr(as_of_date, 'tzinfo') and as_of_date.tzinfo is not None:
        as_of_date = as_of_date.replace(tzinfo=None)

    out = {
        "earnings_in_blackout":    False,  # next earnings within 5 days
        "earnings_beat_recent":    False,  # last surprise >0% within last 30d
        "earnings_miss_recent":    False,  # last surprise <0% within last 30d
        "short_squeeze_setup":     False,  # short_pct_float >= 15%
        "analyst_bullish":         False,  # target_upside >= 15%
        "earnings_next_days":      None,   # int (days until next earnings)
        "earnings_last_surprise":  None,   # float (most recent surprise %)
        "short_pct_float":         None,
        "analyst_target_upside":   None,
    }

    # --- Earnings filtering ---
    try:
        future, past = [], []
        for e in catalyst_data.get("earnings", []) or []:
            try:
                ds = e.get("date")
                dt = datetime.fromisoformat(ds[:10] if isinstance(ds, str) else str(ds)[:10])
                surp = e.get("surprise_pct")
                if dt > as_of_date:
                    future.append((dt, surp))
                else:
                    past.append((dt, surp))
            except Exception:
                continue
        future.sort()
        past.sort(reverse=True)

        if future:
            next_dt, _ = future[0]
            days_to = (next_dt - as_of_date).days
            out["earnings_next_days"] = days_to
            if 0 <= days_to <= 5:
                out["earnings_in_blackout"] = True

        if past:
            # Beat/Miss + last_surprise auf die juengste Quartalszeile MIT bekannter
            # Surprise stuetzen. Die frisch gemeldete Zeile hat oft noch surprise=None
            # (Yahoo fuellt die Surprise erst Stunden/Tage nach dem Report) -> nur auf
            # past[0] zu schauen liess am Report-Tag jedes Beat-Flag ausfallen.
            # (past ist nach Datum absteigend sortiert -> erstes non-None = juengstes.)
            known = next(((d, s) for (d, s) in past if s is not None), None)
            if known is not None:
                k_dt, k_surp = known
                out["earnings_last_surprise"] = k_surp
                days_since = (as_of_date - k_dt).days
                if 0 <= days_since <= 30:
                    if k_surp > 0:
                        out["earnings_beat_recent"] = True
                    elif k_surp < 0:
                        out["earnings_miss_recent"] = True
    except Exception:
        pass

    # --- Short interest ---
    try:
        spf = catalyst_data.get("short_pct_float")
        if spf is not None:
            out["short_pct_float"] = float(spf)
            if float(spf) >= 15.0:
                out["short_squeeze_setup"] = True
    except Exception:
        pass

    # --- Analyst target ---
    try:
        upside = catalyst_data.get("analyst_target_upside_pct")
        if upside is not None:
            out["analyst_target_upside"] = float(upside)
            if float(upside) >= 15.0:
                out["analyst_bullish"] = True
    except Exception:
        pass

    return out


def prewarm_catalysts(tickers, show_progress=True):
    """Bulk-fill cache. Call once at backtest start. Skips fresh entries."""
    cache = _load_cache()
    todo = [t for t in tickers if t not in cache or not _is_fresh(cache[t])]
    if show_progress:
        print(f"Catalyst cache: {len(tickers)-len(todo)} fresh, {len(todo)} need fetch")
    if not todo:
        return
    try:
        from tqdm import tqdm as _tqdm
        iterator = _tqdm(todo, desc="Catalysts") if show_progress else todo
    except ImportError:
        iterator = todo
    for t in iterator:
        try:
            data = fetch_catalyst_data(t)
            cache[t] = data
        except Exception:
            cache[t] = {"_cached_at": datetime.now().isoformat(),
                        "earnings": [], "short_pct_float": None,
                        "analyst_target_upside_pct": None}
    _save_cache(cache)


def score_delta_for_catalyst_signals(signals: dict, setup: str,
                                     backtest_mode: bool = False) -> float:
    """
    Centralized score-delta computation.
    backtest_mode=True: skips anachronistic signals (short_pct, analyst_target
    are CURRENT-only snapshots — applying them to historical trades is
    look-ahead bias). Earnings dates are point-in-time correct, kept enabled.
    backtest_mode=False (live): all signals active.
    """
    delta = 0.0
    # ---- Point-in-time correct (earnings-based) ----
    # Earnings blackout: BO/REV both penalized (gap risk)
    if signals.get("earnings_in_blackout"):
        delta -= 15
    # PEAD bonus only for BREAKOUT (Bernard/Thomas 1989)
    # Reduced from +15 to +8 based on backtest data (small WR uplift +2.7pp)
    if signals.get("earnings_beat_recent") and setup == "BREAKOUT":
        delta += 8
    # Earnings miss penalty for both
    if signals.get("earnings_miss_recent"):
        delta -= 5

    # ---- Current-snapshot signals (skip in backtest to avoid look-ahead) ----
    if not backtest_mode:
        # Short squeeze potential — BREAKOUT only
        if signals.get("short_squeeze_setup") and setup == "BREAKOUT":
            delta += 5
        # Analyst bullish — both setups
        if signals.get("analyst_bullish"):
            delta += 3
    return delta


if __name__ == "__main__":
    # Quick test
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching {t}...")
    raw = get_catalyst_data(t, force_refresh=True)
    sig = derive_catalyst_signals(raw)
    print("Raw:", json.dumps(raw, indent=2, default=str))
    print("Signals:", json.dumps(sig, indent=2, default=str))
    print("BO delta:", score_delta_for_catalyst_signals(sig, "BREAKOUT"))
    print("REV delta:", score_delta_for_catalyst_signals(sig, "REVERSAL"))
