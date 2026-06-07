"""Write claude_analysis + news for AFRM_2026-05-29 and IBKR_2026-06-01."""
import json
from datetime import datetime

PATH = "knowledge/trade_postmortems.json"
with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

now = datetime.now().isoformat()

# ===== AFRM_2026-05-29 =====
afrm_key = "AFRM_2026-05-29"
afrm = data["trades"][afrm_key]
afrm["news"]["web_research"] = [
    {"date": "2026-05-08", "title": "AFRM Q3 FY2026 Earnings: GMV $11.6B (+35%), Revenue $1.039B (+33%), Net Income $100M, Adj OpMargin 27%, FY26 guidance raised",
     "src": "SEC 8-K, Shareholder Letter"},
    {"date": "2026-05-14", "title": "Mizuho raised AFRM PT to $100 from $95",
     "src": "Analyst Action"},
    {"date": "2026-05-27", "title": "Truist raised AFRM PT to $80 from $75",
     "src": "Analyst Action"},
    {"date": "2026-06-05", "title": "Nasdaq -4% as NFP +172K kills rate-cut hopes, 42.7% probability of Dec rate-hike",
     "src": "CNN Business / Yahoo Finance"},
    {"date": "2025-03-17", "title": "Klarna replaces Affirm as Walmart BNPL provider (5% GMV impact, known/priced but referenced in selloff narrative)",
     "src": "Retail Dive / Payments Dive"},
]
afrm["news"]["key_events"] = [
    "Q3 Earnings 8.5 STRONG: GMV +35%, NI $100M, guidance up - Fundamental intact zum Signal-Zeitpunkt",
    "NFP-Schock 5.6 (D+7): Nasdaq -4%, Rate-Hike-Fear, BNPL/Fintech disproportional getroffen",
    "Beta 3.7 mathematisch konsistent mit -7.46% Stop bei SPY ~-3% Zeitraum",
]
afrm["claude_analysis"] = {
    "status": "complete",
    "primary_failure_cause": "Macro-Risk-Off-Event am 5.6.2026 (NFP +172K = doppelter Konsens -> Rate-Hike-Fear, Nasdaq -4%). AFRM mit Beta 3.7 ueberproportional getroffen. Fundamental war intakt (Q3-Beat + Guidance-Raise 21d zuvor). Klassischer Macro-vs-Setup-Konflikt.",
    "primary_win_cause": None,
    "lesson_tags": [
        "high_beta_breakout_macro_risk",
        "fintech_consumer_credit_sensitivity",
        "fundamentals_intact_but_stopped",
        "rate_decision_window_risk",
    ],
    "similar_trades": [
        "IBKR_2026-06-01",
    ],
    "what_would_have_helped": [
        "BETA-Penalty: BREAKOUTs mit Beta > 2.5 koennten Score-Penalty -10 bekommen",
        "Macro-Calendar-Awareness: ApexCatalysts checkt upcoming NFP/CPI/FOMC, Penalty bei High-Impact-Events in 7d",
        "Regime-MIXED-Verschaerfung: bei Wechsel BULLISH->MIXED TG_MIN_SCORE 70->85 anheben",
        "Sektor-spezifisches Rate-Sensitivity-Flag (BNPL, REITs, Util) - bei steigender Rate-Erwartung pendalisieren",
    ],
    "what_to_replicate": [],
    "analyzed_at": now,
    "confidence": "high",
    "data_sources": ["Financial Modeling Prep (Company Profile)", "WebSearch (CNN, Yahoo Finance, Retail Dive)", "SEC 8-K"],
}

# ===== IBKR_2026-06-01 =====
ibkr_key = "IBKR_2026-06-01"
ibkr = data["trades"][ibkr_key]
ibkr["news"]["web_research"] = [
    {"date": "2026-04-15", "title": "IBKR Q1 2026 Earnings: Customer Accounts 4.75M (+31%), Margin Loans $90.2B (+40%), NII $966M (+20%), Commission $613M (+19%), Pretax Margin 77%",
     "src": "SEC 8-K"},
    {"date": "2026-06-05", "title": "Nasdaq -4% as NFP +172K kills rate-cut hopes - broad de-risking, IBKR caught despite zins-beneficiary profile",
     "src": "CNN Business / Yahoo Finance"},
    {"date": "2026-06-01", "title": "IBKR Entry $88.62 = 2.6% unter 52w-High $91.02 (sehr nahe Top, vulnerable Mean-Revert-Zone)",
     "src": "Financial Modeling Prep Company Profile"},
]
ibkr["news"]["key_events"] = [
    "Q1 Earnings April STARK (Account-Growth +31%, NII +20%, 77% Margin) - Fundamental intakt",
    "Entry naher 52w-High ($88.62 vs $91.02) = Late-Cycle BREAKOUT-Failure-Zone",
    "NFP-Schock 5.6: Broad Risk-Off ueberschreibt rationale Sektor-Logik (Zins-Gewinner faellt)",
]
ibkr["claude_analysis"] = {
    "status": "complete",
    "primary_failure_cause": "Late-Cycle BREAKOUT 2.6% unter 52w-High in classic Mean-Revert-Zone. Verschaerft durch Macro-Risk-Off-Event 5.6.2026 (Nasdaq -4%). Beta 1.33 x SPY-Drop ~3% = ca. -5%, exakt am -5.39% Stop. Paradox: IBKR ist zins-Gewinner (NII +20% Q1), aber bei pauschaler Risk-Off-Welle verliert ALLES kurzfristig.",
    "primary_win_cause": None,
    "lesson_tags": [
        "late_cycle_breakout_near_52w_high",
        "macro_selloff_correlates_all_stocks",
        "rate_beneficiary_paradox",
        "score_top_decile_no_protection_in_macro",
    ],
    "similar_trades": [
        "AFRM_2026-05-29",
    ],
    "what_would_have_helped": [
        "DISTANCE-TO-52W-HIGH-Filter: Entry < 3% unter 52w-High -> Score-Penalty -5 (Mean-Revert-Risk)",
        "Macro-Calendar-Awareness (gleicher Punkt wie AFRM): upcoming NFP/CPI/FOMC innerhalb 7d -> Score-Penalty",
        "MIXED-Regime-Schaerfung: TG_MIN_SCORE 70 -> 85 wenn Regime Wechsel BULLISH->MIXED",
        "Pretax-Margin-Profitability ist KEIN Macro-Schutz: hochprofitable Stocks fallen genauso bei broad selloff",
    ],
    "what_to_replicate": [],
    "analyzed_at": now,
    "confidence": "high",
    "data_sources": ["Financial Modeling Prep (Company Profile)", "WebSearch (CNN, SEC, Yahoo Finance)", "IBKR Q1 8-K"],
}

# Update meta analyzed-count
done = sum(1 for t in data["trades"].values()
           if t.get("claude_analysis", {}).get("status") == "complete")
data["_meta"]["analyzed_trades"] = done
data["_meta"]["updated"] = now

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

print(f"OK: AFRM + IBKR claude_analysis written. Total analyzed: {done}/{data['_meta']['total_trades']}")
