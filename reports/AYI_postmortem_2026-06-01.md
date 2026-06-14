# Postmortem - AYI (BREAKOUT) - 2026-06-01

**Outcome:** LOSS | **PnL:** -4.27% | **Exit:** Stop Loss D+6

## Trade-Eckdaten
- Entry $309.04 -> Exit $295.85 -> Ziel war $340.00
- Stop $295.85 | Score 78.8 | RR 2.35 | Sektor Industrials
- Movement-Klasse: **WEAK_BREAKOUT** (perf_120d -17.4 %), RSI 64.7, vol_ratio 1.25

## Primaere Ursache
Macro-Risk-Off-Event 5.6.2026 (NFP +172K = doppelter Konsens, Nasdaq -4 %) auf bereits angeschlagenem Setup. AYI war WEAK_BREAKOUT (perf_120d -17.4 %, vom 52w-High weit weg), post-Q2-Earnings vom 2.4.2026 hatten Wells Fargo (PT $370 -> $310), TD Cowen ($390 -> $335) und Morgan Stanley ($410 -> $400) reduziert, Q2 verfehlte Revenue-Konsens ($1.06B vs $1.08B). 24 Tage vor naechstem Q3-Earnings-Call (25.6.) = klassische earnings_adjacency. Macro lieferte den Trigger, fundamentale Schwaeche lieferte das Niveau.

## Lesson-Tags
- macro_selloff_correlates_all_stocks
- earnings_adjacency_risk
- post_earnings_pt_cut_cycle
- weak_breakout_perf120_negative
- revenue_miss_demand_concern

## Was geholfen haette
- **SCORE_REALIGN bereits live (06-14)**: WEAK_BREAKOUT bekommt jetzt -15 movement_bonus, RSI 64.7 wuerde noch in 48-72 sweet spot fallen. Score waere von 78.8 auf ~64 gefallen -> unter MIN_SCORE -> geblockt.
- Earnings-Adjacency-Filter: BREAKOUT in T-30d vor Earnings sollte Score-Penalty bekommen, hier waren es 24 d.
- Analyst-PT-Trend: 3 PT-Cuts in 60d-Window vor Signal koennten als Negativ-Catalyst zaehlen.
- Macro-Calendar-Awareness: NFP-Release-Tag 5.6.2026 war im Hold-Fenster - High-Impact-Events koennten Penalty triggern.

## News-Research
- **2026-04-02** - AYI Q2 FY26: EPS Beat ($4.14 vs $4.00), Revenue Miss ($1.06B vs $1.08B), Stock -4.9 % *(Yahoo Finance / StockStory)*
- **2026-04 bis 05** - Mehrere PT-Cuts: Wells Fargo $370->$310, TD Cowen $390->$335, Morgan Stanley $410->$400 *(Benzinga Analyst Ratings)*
- **2026-06-01** - BREAKOUT-Signal Entry $309.04, Score 78.8, WEAK_BREAKOUT
- **2026-06-05** - Macro Risk-Off (NFP +172K, Nasdaq -4 %)
- **2026-06-07** - Stop ausgeloest $295.85 (-4.27 %)
- **2026-06-25** - Q3 FY26 Earnings (lag ausserhalb Hold-Fenster, aber Adjacency-Spannung bereits im Hold)

## Aehnliche Trades
- AFRM_2026-05-29 (gleiches Macro-Event)
- IBKR_2026-06-01 (gleiches Macro-Event)
- TSM_2026-06-01 (gleiches Macro-Event)

---
_Daten: apex_signals.json / apex_equity_results.json / WebSearch (Yahoo Finance, Benzinga, StockStory). Confidence: high_
