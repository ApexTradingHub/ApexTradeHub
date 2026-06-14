"""Schreibt AYI + TSM Postmortem-Eintraege ins trade_postmortems.json.

Quellen:
- apex_equity_results.json (Trade-Daten)
- apex_signals.json (Signal-Metadaten)
- WebSearch 2026-06-14 (Markt-Kontext, Analyst-Ratings, Macro-Selloff)
"""
import json
from datetime import datetime, timezone, timedelta

DB_PATH = "knowledge/trade_postmortems.json"
SIGNALS = "apex_signals.json"
EQUITY = "apex_equity_results.json"

WHEN = datetime.now(timezone.utc).isoformat()


def find_signal(sigs, ticker, date):
    for s in sigs:
        if s.get("ticker") == ticker and s.get("date", "").startswith(date):
            return s
    return None


def find_trade(eq, ticker, date):
    for t in eq:
        if t.get("ticker") == ticker and t.get("date", "").startswith(date):
            return t
    return None


def build_entry(ticker, sig, trade, claude):
    sd = trade["date"]
    # exit_date = entry + hold_days_until_stop (use exit_day)
    entry_dt = datetime.fromisoformat(sd)
    # exit_day is trading-day offset; approx with calendar days
    exit_dt = entry_dt + timedelta(days=trade["exit_day"] + 1)
    return {
        "core": {
            "ticker": ticker,
            "signal_date": sd,
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "setup": trade["setup"],
            "entry": trade["entry"],
            "stop": trade["stop"],
            "target": trade["target"],
            "exit_price": trade["exit_price"],
            "exit_reason": trade["exit_reason"],
            "exit_day": trade["exit_day"],
            "trigger_day": None,
            "pnl_pct": trade["pnl_pct"],
            "pnl_usd": trade["pnl_usd"],
            "outcome": "LOSS" if trade["pnl_pct"] < 0 else "WIN",
        },
        "signal_metadata": {
            "score": sig["score"],
            "rr": sig["rr"],
            "rsi": sig["rsi"],
            "macd_bull": sig["macd_bull"],
            "vol_ratio": sig["vol_ratio"],
            "avg_dv_m": sig.get("avg_dv_m"),
            "perf_20d": sig["perf_20d"],
            "perf_60d": sig["perf_60d"],
            "perf_120d": sig["perf_120d"],
            "base_range": sig["base_range"],
            "horizon": sig["horizon"],
            "movement_class": sig["movement_class"],
            "closing_strength": sig["closing_strength"],
            "inside_day": None,
            "sector": sig["sector"],
            "catalysts": {
                "pocket_pivot": bool(sig.get("cat_pocket_pivot", 0)),
                "vol_climax": bool(sig.get("cat_vol_climax", 0)),
                "gap_pct": sig.get("cat_gap_pct"),
                "vcp_strength": sig.get("cat_vcp_strength"),
                "earnings_blackout": bool(sig.get("cat_earnings_blackout", 0)),
                "earnings_beat": bool(sig.get("cat_earnings_beat", 0)),
                "earnings_next_days": sig.get("cat_earnings_next_days"),
                "analyst_upside": sig.get("cat_analyst_upside"),
            },
        },
        "market_context": {
            "sector": sig["sector"],
            "regime_at_signal": "MIXED",
            "note": "5.6.2026 Macro-Risk-Off-Event: NFP +172K (doppelter Konsens), Nasdaq -4%, Halbleiter -8% (TSM-Kontext).",
        },
        "news": {
            "web_research": claude["news_summary"],
            "key_events": claude["key_events"],
        },
        "claude_analysis": {
            "status": "complete",
            "primary_failure_cause": claude["primary_failure_cause"],
            "primary_win_cause": None,
            "lesson_tags": claude["lesson_tags"],
            "similar_trades": claude["similar_trades"],
            "what_would_have_helped": claude["what_would_have_helped"],
            "what_to_replicate": [],
            "analyzed_at": WHEN,
            "confidence": claude["confidence"],
            "data_sources": [
                "apex_signals.json (signal metadata)",
                "apex_equity_results.json (trade outcome)",
                "WebSearch 2026-06-14 (Yahoo Finance, Benzinga, GuruFocus, StockStory, TS2.tech)",
            ],
        },
        "auto_meta": {
            "built_at": WHEN,
            "version": "1.0",
            "note": "Equity-Tracker-simulierter Trade (kein Paper-Trader-Eintrag), aufgenommen 2026-06-14 retro.",
        },
    }


AYI_CLAUDE = {
    "primary_failure_cause": (
        "Macro-Risk-Off-Event 5.6.2026 (NFP +172K = doppelter Konsens, Nasdaq -4%) auf "
        "bereits angeschlagenem Setup: AYI war WEAK_BREAKOUT (perf_120d -17.4 %, vom 52w-High weit weg), "
        "post-Q2-Earnings vom 2.4.2026 hatten Wells Fargo (PT 370->310), TD Cowen (390->335) und Morgan "
        "Stanley (410->400) reduziert, Q2 verfehlte Revenue-Konsens ($1.06B vs $1.08B). 24 Tage "
        "vor naechstem Q3-Earnings-Call (25.6.) = klassische earnings_adjacency. Macro lieferte den "
        "Trigger, fundamentale Schwaeche lieferte das Niveau."
    ),
    "lesson_tags": [
        "macro_selloff_correlates_all_stocks",
        "earnings_adjacency_risk",
        "post_earnings_pt_cut_cycle",
        "weak_breakout_perf120_negative",
        "revenue_miss_demand_concern",
    ],
    "similar_trades": ["AFRM_2026-05-29", "IBKR_2026-06-01", "TSM_2026-06-01"],
    "what_would_have_helped": [
        "SCORE_REALIGN bereits live (06-14): WEAK_BREAKOUT bekommt jetzt -15 movement_bonus, RSI 64.7 wuerde noch in 48-72 sweet spot fallen. Score waere von 78.8 auf ~64 gefallen -> unter MIN_SCORE_NEW, geblockt.",
        "Earnings-Adjacency-Filter: BREAKOUT in T-30d vor Earnings sollte Score-Penalty bekommen, hier waren es 24d.",
        "Analyst-PT-Trend: 3 PT-Cuts in 60d Window vor Signal koennten als Negativ-Catalyst zaehlen.",
        "Macro-Calendar-Awareness: NFP-Release-Tag 5.6.2026 war bekannt, Penalty bei High-Impact-Events im Hold-Fenster.",
    ],
    "confidence": "high",
    "news_summary": (
        "AYI Q2 FY26 (2.4.2026): EPS Beat ($4.14 vs $4.00), Revenue Miss ($1.06B vs $1.08B). "
        "Stock -4.9 % Reaktion. Folgend PT-Cuts: Wells Fargo $370->$310, TD Cowen $390->$335, "
        "Morgan Stanley $410->$400. Avg PT-Trim double-digit. Naechste Earnings 25.6.2026."
    ),
    "key_events": [
        "2026-04-02: Q2 FY26 Revenue Miss, Stock -4.9 %",
        "2026-04 bis 05: Mehrere PT-Cuts (Wells, TD Cowen, MS)",
        "2026-06-01: BREAKOUT-Signal Entry $309.04",
        "2026-06-05: Macro Risk-Off (NFP +172K, Nasdaq -4 %)",
        "2026-06-07: Stop ausgeloest $295.85 (-4.27 %)",
        "2026-06-25: Q3 FY26 Earnings (lag ausserhalb Hold-Fenster)",
    ],
}

TSM_CLAUDE = {
    "primary_failure_cause": (
        "Sector-Rotation-Event: 5.6.2026 Macro-Risk-Off loeste 8 % Halbleiter-Selloff aus (Nasdaq -4 %). "
        "TSM fundamentaler Status STARK: May-Revenue +30 % YoY (am 10.6. berichtet), UBS-Top-Pick mit "
        "PT $470, 3nm-Preiserhoehung 15 % geplant. Aber: P/E TTM 34.66x vs 5y median 22.78x = "
        "Bewertungs-Stretch + AI-Profit-Taking nach starkem Run. EMERGING_BREAKOUT mit perf_120d +48.6 % "
        "= obere Haelfte der Bewegung. Klassischer Macro-vs-Setup-Konflikt: Fundamental intakt, aber "
        "ueberproportionale Beta in Sektor-Selloff."
    ),
    "lesson_tags": [
        "macro_selloff_correlates_all_stocks",
        "semiconductor_sector_beta_risk",
        "ai_profit_taking_unwind",
        "valuation_stretch_pullback_vulnerability",
        "fundamentals_intact_but_stopped",
    ],
    "similar_trades": ["AFRM_2026-05-29", "IBKR_2026-06-01", "AYI_2026-06-01"],
    "what_would_have_helped": [
        "Sektor-Beta-Awareness: Halbleiter mit -8 % Selloff vs Nasdaq -4 % = 2x amplifiziert. TSM-Stop bei -5.3 % war konsistent mit Sektor-Move, nicht idiosynkratisch.",
        "Macro-Calendar-Awareness: NFP-Release 5.6.2026 bekannt vor Entry 1.6. - High-Impact-Events im 7d-Window koennten Penalty triggern.",
        "Valuation-Stretch-Flag: P/E > 1.5x 5y median + perf_120 > 40 % koennte als Reduktion-im-Bonus zaehlen (anstatt Filter).",
        "Sektor-Cap (Backlog #3): waere hier nicht ausschlaggebend gewesen (TSM einziger Tech, AYI Industrials), aber Sektor-ETF-Divergenz war negativ.",
        "Hold-Through statt Stop: Fundamental war intakt - aber das wuerde unsere Risiko-Disziplin brechen. KEIN Aktion notwendig.",
    ],
    "confidence": "high",
    "news_summary": (
        "TSM zum Signal-Zeitpunkt: starke AI-Story, UBS-Top-Pick PT $470, 3nm-Preiserhoehung geplant, "
        "P/E TTM 34.66 vs 5y median 22.78 (Stretch). 5.6.2026: Nasdaq -4 %, Halbleiter-Index -8 % "
        "(Risk-Off + AI-Profit-Taking + Samsung-Strike-Geruechte + TSMC-Stake-Sale-Geruecht). "
        "Fundamental am 10.6. bestaetigt durch May-Revenue +30 % YoY."
    ),
    "key_events": [
        "2026-06-01: BREAKOUT-Signal Entry $431.41, Score 115.2 (EMERGING_BREAKOUT)",
        "2026-06-05: Macro Risk-Off, Halbleiter -8 %, Nasdaq -4 %",
        "2026-06-07: Stop $408.44 (-5.32 %)",
        "2026-06-10: TSM May-Revenue +30 % YoY (ex-post Bestaetigung Fundamentals)",
        "2026-07-10: TSM Juni-Revenue (lag ausserhalb Hold-Fenster)",
        "2026-07-16: TSM Q2 Earnings (lag ausserhalb Hold-Fenster)",
    ],
}


def main():
    db = json.load(open(DB_PATH, encoding="utf-8"))
    sigs = json.load(open(SIGNALS, encoding="utf-8"))
    eq = json.load(open(EQUITY, encoding="utf-8"))

    for ticker, claude in [("AYI", AYI_CLAUDE), ("TSM", TSM_CLAUDE)]:
        key = f"{ticker}_2026-06-01"
        sig = find_signal(sigs, ticker, "2026-06-01")
        trade = find_trade(eq, ticker, "2026-06-01")
        if not sig or not trade:
            print(f"SKIP {key}: signal={bool(sig)} trade={bool(trade)}")
            continue
        entry = build_entry(ticker, sig, trade, claude)
        if key in db["trades"]:
            existing = db["trades"][key]
            existing["claude_analysis"] = entry["claude_analysis"]
            existing["news"] = entry["news"]
            print(f"UPDATED claude_analysis on existing {key}")
        else:
            db["trades"][key] = entry
            print(f"ADDED new entry {key}")

    db["_meta"]["updated"] = WHEN
    db["_meta"]["analyzed_trades"] = sum(
        1
        for t in db["trades"].values()
        if isinstance(t, dict) and t.get("claude_analysis", {}).get("status") == "complete"
    )
    db["_meta"]["total_trades"] = len(db["trades"])

    json.dump(db, open(DB_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDB saved. total_trades={db['_meta']['total_trades']}, analyzed={db['_meta']['analyzed_trades']}")


if __name__ == "__main__":
    main()
