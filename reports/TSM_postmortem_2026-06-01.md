# Postmortem - TSM (BREAKOUT) - 2026-06-01

**Outcome:** LOSS | **PnL:** -5.32% | **Exit:** Stop Loss D+6

## Trade-Eckdaten
- Entry $431.41 -> Exit $408.44 -> Ziel war $470.00
- Stop $408.44 | Score 115.2 | RR 1.68 | Sektor Technology (Semiconductors)
- Movement-Klasse: **EMERGING_BREAKOUT** (perf_120d +48.6 %), RSI 65.8, vol_ratio 1.35

## Primaere Ursache
Sektor-Rotation-Event: 5.6.2026 Macro-Risk-Off loeste 8 % Halbleiter-Selloff aus (Nasdaq -4 %). TSM fundamentaler Status STARK: May-Revenue +30 % YoY (am 10.6. berichtet), UBS-Top-Pick mit PT $470, 3nm-Preiserhoehung 15 % geplant. Aber: P/E TTM 34.66x vs 5y median 22.78x = Bewertungs-Stretch + AI-Profit-Taking nach starkem Run. EMERGING_BREAKOUT mit perf_120d +48.6 % = obere Haelfte der Bewegung. Klassischer Macro-vs-Setup-Konflikt: Fundamental intakt, aber ueberproportionale Beta in Sektor-Selloff.

## Lesson-Tags
- macro_selloff_correlates_all_stocks
- semiconductor_sector_beta_risk
- ai_profit_taking_unwind
- valuation_stretch_pullback_vulnerability
- fundamentals_intact_but_stopped

## Was geholfen haette
- **Sektor-Beta-Awareness**: Halbleiter mit -8 % Selloff vs Nasdaq -4 % = 2x amplifiziert. TSM-Stop bei -5.3 % war konsistent mit Sektor-Move, nicht idiosynkratisch.
- Macro-Calendar-Awareness: NFP-Release 5.6.2026 bekannt vor Entry 1.6. - High-Impact-Events im 7d-Window koennten Penalty triggern.
- Valuation-Stretch-Flag: P/E > 1.5x 5y median + perf_120 > 40 % koennte als Reduktion-im-Bonus zaehlen (anstatt Hardfilter).
- Sektor-Cap (Backlog #3): waere hier nicht ausschlaggebend gewesen (TSM einziger Tech), aber Sektor-ETF-Divergenz war negativ.
- Hold-Through statt Stop: Fundamental war intakt - **aber das wuerde unsere Risiko-Disziplin brechen. KEINE Aktion notwendig.**

## News-Research
- **2026-06-01** - BREAKOUT-Signal Entry $431.41, Score 115.2 (EMERGING_BREAKOUT)
- **vor 06-01** - UBS PT $470 (Top-Pick AI-Theme), 3nm-Preiserhoehung 15 % geplant *(UBS / Tipranks)*
- **vor 06-01** - P/E TTM 34.66x vs 5y median 22.78x = stretched *(GuruFocus / SeekingAlpha)*
- **2026-06-05** - Macro Risk-Off, Halbleiter-Index -8 %, Nasdaq -4 %, AI-Profit-Taking *(GuruFocus / Quiver Quant)*
- **2026-06-07** - Stop $408.44 (-5.32 %)
- **2026-06-10** - TSM May-Revenue NT$416.98B (+30 % YoY) - ex-post Bestaetigung Fundamentals *(TSMC IR)*
- **2026-07-16** - TSM Q2 Earnings (ausserhalb Hold-Fenster)

## Aehnliche Trades
- AFRM_2026-05-29 (gleiches Macro-Event, Beta-Amplifikation)
- IBKR_2026-06-01 (gleiches Macro-Event)
- AYI_2026-06-01 (gleiches Macro-Event)

---
_Daten: apex_signals.json / apex_equity_results.json / WebSearch (Yahoo Finance, GuruFocus, TS2.tech, Quiver Quant). Confidence: high_
