# Postmortem - AFRM (BREAKOUT) - 2026-05-29

**Outcome:** LOSS | **PnL:** -7.46% | **Exit:** Stop Loss D+3

## Trade-Eckdaten
- Entry $73.48 -> Exit $68.00 -> Ziel war $83.48
- Stop $68.00 | Score 123.6

## Primaere Ursache
Macro-Risk-Off-Event am 5.6.2026 (NFP +172K = doppelter Konsens -> Rate-Hike-Fear, Nasdaq -4%). AFRM mit Beta 3.7 ueberproportional getroffen. Fundamental war intakt (Q3-Beat + Guidance-Raise 21d zuvor). Klassischer Macro-vs-Setup-Konflikt.

## Lesson-Tags
- high_beta_breakout_macro_risk
- fintech_consumer_credit_sensitivity
- fundamentals_intact_but_stopped
- rate_decision_window_risk

## Was geholfen haette
- BETA-Penalty: BREAKOUTs mit Beta > 2.5 koennten Score-Penalty -10 bekommen
- Macro-Calendar-Awareness: ApexCatalysts checkt upcoming NFP/CPI/FOMC, Penalty bei High-Impact-Events in 7d
- Regime-MIXED-Verschaerfung: bei Wechsel BULLISH->MIXED TG_MIN_SCORE 70->85 anheben
- Sektor-spezifisches Rate-Sensitivity-Flag (BNPL, REITs, Util) - bei steigender Rate-Erwartung pendalisieren

## News-Research
- **2026-05-08** - AFRM Q3 FY2026 Earnings: GMV $11.6B (+35%), Revenue $1.039B (+33%), Net Income $100M, Adj OpMargin 27%, FY26 guidance raised *(SEC 8-K, Shareholder Letter)*
- **2026-05-14** - Mizuho raised AFRM PT to $100 from $95 *(Analyst Action)*
- **2026-05-27** - Truist raised AFRM PT to $80 from $75 *(Analyst Action)*
- **2026-06-05** - Nasdaq -4% as NFP +172K kills rate-cut hopes, 42.7% probability of Dec rate-hike *(CNN Business / Yahoo Finance)*
- **2025-03-17** - Klarna replaces Affirm as Walmart BNPL provider (5% GMV impact, known/priced but referenced in selloff narrative) *(Retail Dive / Payments Dive)*

## Aehnliche Trades
- IBKR_2026-06-01

---
_Powered by Bigdata.com (orchestration) - FMP (company data) - WebSearch. Confidence: high_