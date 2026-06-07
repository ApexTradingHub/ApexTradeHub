# Postmortem - IBKR (BREAKOUT) - 2026-06-01

**Outcome:** LOSS | **PnL:** -5.39% | **Exit:** Stop Loss D+3

## Trade-Eckdaten
- Entry $88.62 -> Exit $83.84 -> Ziel war $97.50
- Stop $83.84 | Score 123.3

## Primaere Ursache
Late-Cycle BREAKOUT 2.6% unter 52w-High in classic Mean-Revert-Zone. Verschaerft durch Macro-Risk-Off-Event 5.6.2026 (Nasdaq -4%). Beta 1.33 x SPY-Drop ~3% = ca. -5%, exakt am -5.39% Stop. Paradox: IBKR ist zins-Gewinner (NII +20% Q1), aber bei pauschaler Risk-Off-Welle verliert ALLES kurzfristig.

## Lesson-Tags
- late_cycle_breakout_near_52w_high
- macro_selloff_correlates_all_stocks
- rate_beneficiary_paradox
- score_top_decile_no_protection_in_macro

## Was geholfen haette
- DISTANCE-TO-52W-HIGH-Filter: Entry < 3% unter 52w-High -> Score-Penalty -5 (Mean-Revert-Risk)
- Macro-Calendar-Awareness (gleicher Punkt wie AFRM): upcoming NFP/CPI/FOMC innerhalb 7d -> Score-Penalty
- MIXED-Regime-Schaerfung: TG_MIN_SCORE 70 -> 85 wenn Regime Wechsel BULLISH->MIXED
- Pretax-Margin-Profitability ist KEIN Macro-Schutz: hochprofitable Stocks fallen genauso bei broad selloff

## News-Research
- **2026-04-15** - IBKR Q1 2026 Earnings: Customer Accounts 4.75M (+31%), Margin Loans $90.2B (+40%), NII $966M (+20%), Commission $613M (+19%), Pretax Margin 77% *(SEC 8-K)*
- **2026-06-05** - Nasdaq -4% as NFP +172K kills rate-cut hopes - broad de-risking, IBKR caught despite zins-beneficiary profile *(CNN Business / Yahoo Finance)*
- **2026-06-01** - IBKR Entry $88.62 = 2.6% unter 52w-High $91.02 (sehr nahe Top, vulnerable Mean-Revert-Zone) *(Financial Modeling Prep Company Profile)*

## Aehnliche Trades
- AFRM_2026-05-29

---
_Powered by Bigdata.com (orchestration) - FMP (company data) - WebSearch. Confidence: high_