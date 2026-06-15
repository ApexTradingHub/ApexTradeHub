"""
ApexPostmortem — Per-Trade Postmortem-Datenbank
================================================
Aufgabe: für jeden geschlossenen Trade strukturierte Daten sammeln die
Claude später analysieren kann.

Was es macht:
1. Liest alle geschlossenen Trades aus apex_equity_results.json
2. Für jeden NICHT analysierten Trade (nicht in trade_postmortems.json):
   - Core trade data (entry, exit, pnl, etc.)
   - Signal metadata (RSI, vol, perf, catalysts, score)
   - Market context (SPY/QQQ/Sektor-ETF Performance während hold)
   - yfinance News um signal-date ± 5d
   - Sektor-Klassifikation
   - status: "pending" für Claude-Analyse
3. Speichert inkrementell in knowledge/trade_postmortems.json
4. Generiert reports/postmortem_summary.md (human-readable Index)

Claude füllt später:
- claude_analysis.primary_failure_cause
- claude_analysis.lesson_tags
- claude_analysis.what_would_have_helped
- claude_analysis.similar_trades
- claude_analysis.web_research (WebSearch findings)
- Setzt status: "complete"

Aufruf:
  py apex_postmortem.py                # alle neuen Trades analysieren
  py apex_postmortem.py --refresh-all  # re-analyze alle (Cache löschen)
  py apex_postmortem.py --ticker MU    # nur ein Ticker
  py apex_postmortem.py --summary      # nur Index re-generieren
"""

import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KNOWLEDGE_DIR = Path("knowledge")
REPORTS_DIR = Path("reports")
POSTMORTEM_FILE = KNOWLEDGE_DIR / "trade_postmortems.json"
SUMMARY_FILE = REPORTS_DIR / "postmortem_summary.md"

# Sektor → ETF mapping for context fetching
SECTOR_ETF = {
    "Energy":              "XLE",
    "Technology":          "XLK",
    "Healthcare":          "XLV",
    "Financial Services":  "XLF",
    "Financials":          "XLF",
    "Industrials":         "XLI",
    "Consumer Defensive":  "XLP",
    "Consumer Cyclical":   "XLY",
    "Basic Materials":     "XLB",
    "Utilities":           "XLU",
    "Real Estate":         "XLRE",
    "Communication Services": "XLC",
    "Communication":       "XLC",
}


@contextlib.contextmanager
def _suppress():
    o, e = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = o, e


# =============================================================
# DATA LOADING
# =============================================================
def load_postmortems():
    if POSTMORTEM_FILE.exists():
        with open(POSTMORTEM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_meta": {"created": datetime.now().isoformat(), "version": "1.0"},
            "trades": {}}


def save_postmortems(pm):
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    pm["_meta"]["updated"] = datetime.now().isoformat()
    pm["_meta"]["total_trades"] = len(pm.get("trades", {}))
    with open(POSTMORTEM_FILE, "w", encoding="utf-8") as f:
        json.dump(pm, f, indent=2, ensure_ascii=False, default=str)


def load_inputs():
    with open("apex_signals.json", "r", encoding="utf-8") as f:
        sigs = json.load(f)
    with open("apex_equity_results.json", "r", encoding="utf-8") as f:
        trades = json.load(f)
    try:
        with open("sector_cache.json", "r", encoding="utf-8") as f:
            sectors = json.load(f)
    except Exception:
        sectors = {}
    return sigs, trades, sectors


def trade_id(ticker, signal_date):
    return f"{ticker}_{signal_date}"


def join_signal_to_trade(trades, sigs):
    lookup = {(s["date"], s["ticker"]): s for s in sigs}
    out = []
    for t in trades:
        s = lookup.get((t["date"], t["ticker"]), {})
        out.append({**s, **t})
    return out


# =============================================================
# MARKET CONTEXT (pre-loaded, queried per trade)
# =============================================================
class MarketDataCache:
    """Pre-loads SPY, QQQ, all sector ETFs once. Queries per trade period."""
    def __init__(self, all_dates):
        if not all_dates:
            self.spy = self.qqq = None
            self.sectors = {}
            return
        start = min(all_dates)
        end_dt = datetime.strptime(max(all_dates), "%Y-%m-%d") + timedelta(days=120)
        end = end_dt.strftime("%Y-%m-%d")
        print(f"  Loading market data {start} → {end}...")
        tickers = " ".join(["SPY", "QQQ"] + list(set(SECTOR_ETF.values())))
        with _suppress():
            data = yf.download(tickers, start=start, end=end,
                               auto_adjust=True, progress=False,
                               group_by="ticker", threads=True)
        self._data = data

    def perf_during(self, ticker, start_date, end_date):
        """% change between start_date and end_date for given ticker."""
        try:
            if hasattr(self._data.columns, "levels"):
                df = self._data[ticker]
            else:
                df = self._data
            df = df["Close"].dropna()
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            sub = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if len(sub) < 2:
                return None
            return round((float(sub.iloc[-1]) / float(sub.iloc[0]) - 1) * 100, 2)
        except Exception:
            return None


# =============================================================
# NEWS FETCH (yfinance — basic, Claude extends with WebSearch)
# =============================================================
def fetch_yf_news(ticker, signal_date, hold_days=14):
    """Fetch news titles in window: signal_date - 5d to signal_date + hold_days."""
    try:
        sig_dt = datetime.strptime(signal_date, "%Y-%m-%d")
    except Exception:
        return []
    window_start = (sig_dt - timedelta(days=5)).timestamp()
    window_end = (sig_dt + timedelta(days=hold_days)).timestamp()
    try:
        with _suppress():
            news = yf.Ticker(ticker).news or []
        out = []
        for n in news:
            pubtime = n.get("providerPublishTime", 0)
            if not (window_start <= pubtime <= window_end):
                continue
            out.append({
                "date": datetime.fromtimestamp(pubtime).strftime("%Y-%m-%d"),
                "title": n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "link": n.get("link", ""),
            })
        # Sort newest first
        out.sort(key=lambda x: x["date"], reverse=True)
        return out[:8]  # max 8 articles
    except Exception:
        return []


# =============================================================
# POSTMORTEM BUILDER
# =============================================================

# Lazy macro-history cache (loaded once per run)
_MACRO_HISTORY = None

def _macro_lookup(target_date):
    """Look up FRED macro context (VIX/HY/yield-curve + regime) at a date.
    Returns None gracefully if apex_macro_history.json missing — postmortem
    still works, just without macro fields. Run `py apex_macro.py --backfill`
    once to populate the history file."""
    global _MACRO_HISTORY
    if not target_date:
        return None
    try:
        if _MACRO_HISTORY is None:
            from apex_macro import HISTORY_FILE, macro_at_date
            if not HISTORY_FILE.exists():
                _MACRO_HISTORY = {}  # mark as 'tried, missing'
                return None
            _MACRO_HISTORY = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if not _MACRO_HISTORY:
            return None
        from apex_macro import macro_at_date
        return macro_at_date(target_date, history=_MACRO_HISTORY)
    except Exception:
        return None


def compute_exit_date(signal_date, exit_day):
    """Approximate exit date = signal_date + exit_day trading days."""
    try:
        sig_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        # Approx: 1 trading day ≈ 1.4 calendar days
        calendar_days = int(round(exit_day * 1.4))
        return (sig_dt + timedelta(days=calendar_days)).strftime("%Y-%m-%d")
    except Exception:
        return None


def build_core_postmortem(merged, market_cache, sector_cache):
    """Build a postmortem entry from merged signal+trade data."""
    ticker = merged["ticker"]
    sig_date = merged["date"]
    exit_day = merged.get("exit_day", 0)
    exit_date = compute_exit_date(sig_date, exit_day)

    sector = sector_cache.get(ticker, "Unknown")
    sector_etf = SECTOR_ETF.get(sector, None)

    # Market context: SPY/QQQ/Sektor-ETF Perf während hold
    spy_perf = market_cache.perf_during("SPY", sig_date, exit_date) if exit_date else None
    qqq_perf = market_cache.perf_during("QQQ", sig_date, exit_date) if exit_date else None
    sec_perf = (market_cache.perf_during(sector_etf, sig_date, exit_date)
                if sector_etf and exit_date else None)

    # Sektor-Divergenz: Sektor signifikant schwächer als SPY
    sector_divergence = False
    if sec_perf is not None and spy_perf is not None:
        if (sec_perf - spy_perf) < -2.0:
            sector_divergence = True

    # News
    news = fetch_yf_news(ticker, sig_date, hold_days=max(exit_day, 15))

    pnl_pct = merged.get("pnl_pct", 0)
    outcome = "WIN" if pnl_pct > 0 else "LOSS"

    return {
        "core": {
            "ticker":      ticker,
            "signal_date": sig_date,
            "exit_date":   exit_date,
            "setup":       merged.get("setup", "?"),
            "entry":       merged.get("entry"),
            "stop":        merged.get("stop"),
            "target":      merged.get("target"),
            "exit_price":  merged.get("exit_price"),
            "exit_reason": merged.get("exit_reason"),
            "exit_day":    exit_day,
            "trigger_day": merged.get("trigger_day"),
            "pnl_pct":     pnl_pct,
            "pnl_usd":     merged.get("pnl_usd"),
            "outcome":     outcome,
        },
        "signal_metadata": {
            "score":           merged.get("score"),
            "rr":              merged.get("rr"),
            "rsi":             merged.get("rsi"),
            "macd_bull":       merged.get("macd_bull"),
            "vol_ratio":       merged.get("vol_ratio"),
            "avg_dv_m":        merged.get("avg_dv_m"),
            "perf_20d":        merged.get("perf_20d"),
            "perf_60d":        merged.get("perf_60d"),
            "perf_120d":       merged.get("perf_120d"),
            "base_range":      merged.get("base_range"),
            "horizon":         merged.get("horizon"),
            "movement_class":  merged.get("movement_class"),
            "closing_strength": merged.get("closing_strength"),
            "inside_day":      merged.get("inside_day"),
            "sector":          sector,
            "catalysts": {
                "pocket_pivot":     merged.get("cat_pocket_pivot"),
                "vol_climax":       merged.get("cat_vol_climax"),
                "gap_pct":          merged.get("cat_gap_pct"),
                "vcp_strength":     merged.get("cat_vcp_strength"),
                "earnings_blackout": merged.get("cat_earnings_blackout"),
                "earnings_beat":    merged.get("cat_earnings_beat"),
                "earnings_next_days": merged.get("cat_earnings_next_days"),
                "short_pct":        merged.get("cat_short_pct"),
                "analyst_upside":   merged.get("cat_analyst_upside"),
            },
        },
        "market_context": {
            "sector":           sector,
            "sector_etf":       sector_etf,
            "spy_perf_pct":     spy_perf,
            "qqq_perf_pct":     qqq_perf,
            "sector_etf_perf_pct": sec_perf,
            "sector_divergence": sector_divergence,
            "regime_at_signal": merged.get("risk_on"),
            "macro_at_signal":  _macro_lookup(sig_date),
            "macro_at_exit":    _macro_lookup(exit_date) if exit_date else None,
        },
        "news": {
            "yfinance_news": news,
            "web_research":  [],   # Claude fills via WebSearch
            "key_events":    [],   # Claude curates
        },
        "claude_analysis": {
            "status":                 "pending",
            "primary_failure_cause":  None,
            "primary_win_cause":      None,
            "lesson_tags":            [],
            "similar_trades":         [],
            "what_would_have_helped": [],
            "what_to_replicate":      [],
            "analyzed_at":            None,
            "confidence":             None,
        },
        "auto_meta": {
            "built_at": datetime.now().isoformat(),
            "version":  "1.0",
        },
    }


# =============================================================
# SUMMARY REPORT (human-readable index)
# =============================================================
def gen_summary(pm):
    trades = pm.get("trades", {})
    L = []
    L.append(f"# Postmortem Database — Summary")
    L.append("")
    L.append(f"_Updated: {pm.get('_meta', {}).get('updated', '?')} | "
             f"Total trades: {len(trades)}_")
    L.append("")

    # Status breakdown
    statuses = Counter(t["claude_analysis"]["status"] for t in trades.values())
    L.append(f"**Analyse-Status**: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    L.append("")

    # Data-quality guardrail: a trade marked 'complete' MUST have structured news
    # (web_research). Otherwise the news only lives as prose in claude_analysis and
    # is not machine-readable. Surface these loudly so future omissions get caught.
    missing_news = [
        f"{t['core']['ticker']}_{t['core']['signal_date']}"
        for t in trades.values()
        if t["claude_analysis"]["status"] == "complete"
        and not t["news"].get("web_research")
        and not t["news"].get("yfinance_news")
    ]
    if missing_news:
        L.append("## ⚠ Daten-Qualität: complete OHNE strukturierte News")
        L.append("")
        L.append(f"{len(missing_news)} Trade(s) sind als `complete` markiert, haben aber "
                 "ein leeres `news.web_research`-Feld. News bitte strukturiert nachtragen "
                 "(Datum/Titel/Quelle), nicht nur im Analyse-Text:")
        L.append("")
        for tid in missing_news:
            L.append(f"- `{tid}`")
        L.append("")

    # Lesson-Tag frequency
    all_tags = []
    for t in trades.values():
        all_tags.extend(t["claude_analysis"].get("lesson_tags", []))
    tag_counts = Counter(all_tags)
    if tag_counts:
        L.append("## Lesson-Tag Frequency (Claude curated)")
        L.append("")
        L.append("| Tag | Count |")
        L.append("|---|---|")
        for tag, n in tag_counts.most_common(20):
            L.append(f"| `{tag}` | {n} |")
        L.append("")

    # Sektor-Divergenz Loser
    sec_div_losers = [t for t in trades.values()
                      if t["core"]["outcome"] == "LOSS"
                      and t["market_context"].get("sector_divergence")]
    if sec_div_losers:
        L.append("## Sektor-Divergenz-Loser (Sektor-ETF >2pp schwächer als SPY)")
        L.append("")
        L.append("| Trade | Setup | Sektor | SPY% | Sektor% | PnL% |")
        L.append("|---|---|---|---|---|---|")
        for t in sec_div_losers[:15]:
            c = t["core"]; m = t["market_context"]
            L.append(f"| {c['ticker']}_{c['signal_date']} | {c['setup']} | "
                     f"{m['sector']} | {m['spy_perf_pct']}% | {m['sector_etf_perf_pct']}% | "
                     f"{c['pnl_pct']}% |")
        L.append("")

    # Top losers ranked by pnl
    by_pnl = sorted(trades.values(), key=lambda t: t["core"]["pnl_pct"])[:15]
    L.append("## Worst 15 Trades")
    L.append("")
    L.append("| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |")
    L.append("|---|---|---|---|---|---|---|")
    for t in by_pnl:
        c = t["core"]; m = t["signal_metadata"]; ca = t["claude_analysis"]
        analysis = ca.get("primary_failure_cause", "—") or "_pending_"
        L.append(f"| {c['ticker']}_{c['signal_date']} | {c['setup']} | "
                 f"{c['pnl_pct']}% | {c['exit_reason']} D+{c['exit_day']} | "
                 f"{m.get('score','?')} | {m.get('sector','?')} | {analysis} |")
    L.append("")

    # Top winners
    by_pnl_w = sorted(trades.values(), key=lambda t: -t["core"]["pnl_pct"])[:10]
    L.append("## Best 10 Trades")
    L.append("")
    L.append("| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |")
    L.append("|---|---|---|---|---|---|---|")
    for t in by_pnl_w:
        c = t["core"]; m = t["signal_metadata"]; ca = t["claude_analysis"]
        analysis = ca.get("primary_win_cause", "—") or "_pending_"
        L.append(f"| {c['ticker']}_{c['signal_date']} | {c['setup']} | "
                 f"{c['pnl_pct']}% | {c['exit_reason']} D+{c['exit_day']} | "
                 f"{m.get('score','?')} | {m.get('sector','?')} | {analysis} |")
    L.append("")

    # Pending analysis queue
    pending = [k for k, t in trades.items() if t["claude_analysis"]["status"] == "pending"]
    L.append(f"## Pending Claude-Analyse ({len(pending)})")
    L.append("")
    L.append("Diese Trades warten auf Claude WebSearch-Verifikation + Lesson-Tagging:")
    L.append("")
    for tid in pending[:20]:
        L.append(f"- `{tid}`")
    if len(pending) > 20:
        L.append(f"- _(+ {len(pending)-20} mehr)_")
    L.append("")

    return "\n".join(L)


# =============================================================
# MAIN
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-all", action="store_true",
                    help="Re-analyze all trades (clears existing postmortems)")
    ap.add_argument("--ticker", type=str, default=None,
                    help="Process only one ticker")
    ap.add_argument("--summary", action="store_true",
                    help="Only re-generate summary index, no new analysis")
    args = ap.parse_args()

    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    pm = load_postmortems()
    if args.refresh_all:
        print("--refresh-all: clearing existing postmortems")
        pm["trades"] = {}

    if args.summary:
        summary = gen_summary(pm)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary regenerated → {SUMMARY_FILE}")
        return

    sigs, trades, sectors = load_inputs()
    merged_all = join_signal_to_trade(trades, sigs)
    if args.ticker:
        merged_all = [m for m in merged_all if m.get("ticker") == args.ticker]

    # Identify which need processing
    existing_ids = set(pm["trades"].keys())
    to_process = []
    for m in merged_all:
        tid = trade_id(m["ticker"], m["date"])
        if tid in existing_ids:
            continue
        to_process.append((tid, m))

    print(f"Postmortem-DB: {len(pm['trades'])} existing | {len(to_process)} new to process")
    if not to_process:
        print("Nichts zu tun. Summary re-generieren mit --summary.")
        # Still regen summary for current state
        summary = gen_summary(pm)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary → {SUMMARY_FILE}")
        return

    # Pre-load market data for all dates in to_process
    all_dates = sorted({m["date"] for _, m in to_process})
    market_cache = MarketDataCache(all_dates)

    # Process trades
    print(f"Processing {len(to_process)} trades (yfinance news fetch per ticker)...")
    for i, (tid, m) in enumerate(to_process, 1):
        print(f"  [{i}/{len(to_process)}] {tid} ({m.get('setup','?')}) ...", end=" ")
        try:
            entry = build_core_postmortem(m, market_cache, sectors)
            pm["trades"][tid] = entry
            news_count = len(entry["news"]["yfinance_news"])
            sec_perf = entry["market_context"]["sector_etf_perf_pct"]
            sec_str = f"{sec_perf}%" if sec_perf is not None else "?"
            div = " [DIV]" if entry["market_context"]["sector_divergence"] else ""
            print(f"news={news_count} sec={sec_str}{div}")
        except Exception as e:
            print(f"ERROR: {e}")
        # Save incrementally so abort doesn't lose progress
        if i % 10 == 0:
            save_postmortems(pm)
        time.sleep(0.3)  # gentle on yfinance

    save_postmortems(pm)
    print(f"\nPostmortem-DB: {len(pm['trades'])} total entries → {POSTMORTEM_FILE}")

    summary = gen_summary(pm)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary → {SUMMARY_FILE}")
    print()
    print("Next: paste apex_learn report OR send postmortem_summary.md to Claude.")
    print("Claude will WebSearch news for pending trades + fill lesson_tags.")


if __name__ == "__main__":
    main()
