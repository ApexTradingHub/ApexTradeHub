# Postmortem Database — Summary

_Updated: 2026-06-06T22:47:40.270730 | Total trades: 132_

**Analyse-Status**: complete=38, pending=94

## Lesson-Tag Frequency (Claude curated)

| Tag | Count |
|---|---|
| `sector_momentum_tailwind` | 9 |
| `energy_oil_surge_2026q1` | 9 |
| `sector_relative_strength` | 9 |
| `pocket_pivot_validation` | 7 |
| `earnings_adjacency_risk` | 6 |
| `fundamental_deterioration` | 6 |
| `reversal_after_analyst_downgrade` | 6 |
| `strong_closing_strength_confirmation` | 4 |
| `earnings_miss_during_hold` | 3 |
| `secular_ai_catalyst` | 3 |
| `geopolitical_premium_fade` | 3 |
| `oil_peak_chase` | 3 |
| `high_score_loss_85plus` | 3 |
| `ai_storage_momentum` | 2 |
| `post_earnings_beat_continuation` | 2 |
| `extended_but_catalyst_won` | 2 |
| `earnings_adjacency_breakout` | 2 |
| `dead_cat_bounce` | 2 |
| `duplicate_trap` | 2 |
| `no_pocket_pivot_warning` | 2 |

## Sektor-Divergenz-Loser (Sektor-ETF >2pp schwächer als SPY)

| Trade | Setup | Sektor | SPY% | Sektor% | PnL% |
|---|---|---|---|---|---|
| SHEL_2026-03-19 | BREAKOUT | Energy | 3.33% | -2.79% | -0.04% |
| AM_2026-03-24 | BREAKOUT | Energy | 0.32% | -3.07% | -4.83% |
| XEL_2026-04-09 | BREAKOUT | Utilities | 0.91% | -1.61% | -3.01% |
| D_2026-04-09 | BREAKOUT | Utilities | 0.91% | -1.61% | -3.01% |
| MDT_2026-03-26 | REVERSAL | Healthcare | 10.08% | 2.1% | -3.73% |
| CDP_2026-04-16 | BREAKOUT | Real Estate | 1.93% | -0.73% | -2.98% |
| MOS_2026-04-19 | REVERSAL | Basic Materials | 0.4% | -2.43% | -7.09% |
| ALL_2026-04-16 | BREAKOUT | Financial Services | 2.42% | 0.19% | -3.16% |
| CNM_2026-03-25 | REVERSAL | Industrials | 12.46% | 3.42% | -3.54% |
| CZR_2026-04-20 | BREAKOUT | Consumer Cyclical | 4.31% | -0.42% | -2.46% |
| AMTM_2026-04-01 | REVERSAL | Industrials | 9.98% | 5.19% | -8.01% |
| FTI_2026-04-27 | BREAKOUT | Energy | 3.14% | -1.88% | -4.52% |
| FLR_2026-04-27 | BREAKOUT | Industrials | 3.14% | 0.4% | -5.17% |
| PENN_2026-04-27 | BREAKOUT | Consumer Cyclical | 3.37% | 1.3% | -7.45% |
| ACI_2026-04-17 | REVERSAL | Consumer Defensive | 4.11% | 1.1% | -5.89% |

## Worst 15 Trades

| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |
|---|---|---|---|---|---|---|
| ENPH_2026-04-20 | REVERSAL | -9.96% | Stop Loss D+7 | 91.1 | Technology | Stop ausgeloest 1 Tag VOR Earnings-Beat - REVERSAL stoppt sich aus dem profitablen Move raus |
| SW_2026-05-04 | REVERSAL | -9.44% | Stop Loss D+11 | 64.6 | Unknown | REVERSAL gekauft 4 Tage vor Q1-2026-Earnings, die massiv verfehlten: EPS $0.33 vs $0.41 erwartet (-19.5% Surprise), Net Income -83% YoY auf $63M (vs $382M). Net-Marge brach von 5.0% auf 0.8% ein (hoehere Depreciation/Impairment/Restructuring + ~$65M Wetter-Headwinds + schwaechere NA-Volumina). 7-Tage-Verlustserie -9.5%, Stop am 19.5 bei -4.6% Tagesverlust. Oversold-RSI war Falle - der Abverkauf war fundamental getrieben. |
| WDC_2026-03-18 | BREAKOUT | -9.37% | Stop Loss D+3 | 94.4 | Technology | Stop zu eng fuer hochvolatilen AI-Storage-Stock (170% YTD, 845% 1y) - getoppt nach 3 Tagen, dann massive Continuation verpasst |
| HII_2026-05-12 | REVERSAL | -9.34% | Stop Loss D+13 | 90.9 | Industrials | _pending_ |
| CACI_2026-04-23 | REVERSAL | -9.28% | Stop Loss D+9 | 81.2 | Technology | Identisch zu CACI_2026-04-19 - REVERSAL #2 in Downgrade-Phase, kein Lerneffekt durch System |
| BTU_2026-05-01 | REVERSAL | -8.73% | Stop Loss D+3 | 82.2 | Energy | REVERSAL 4 Tage vor Q1 Earnings - Earnings am 5.5 zeigten Net Loss & EBITDA -43%, Production Cut |
| CACI_2026-05-01 | REVERSAL | -8.03% | Stop Loss D+3 | 72.6 | Technology | REVERSAL #3 fuer CACI in 12 Tagen - Phase E DUPLICATE_WINDOW (3d) hat nicht ausgereicht, weil wir alle 10-15 Tage neu signalisiert haben |
| AMTM_2026-04-01 | REVERSAL | -8.01% | Stop Loss D+23 | 59.7 | Industrials | DEAD_CAT_BOUNCE - Stock war -20% YTD, REVERSAL kaufte trotz fundamentaler Schwaeche; nahm 23 Tage zum Stop |
| APH_2026-05-12 | REVERSAL | -7.84% | Stop Loss D+5 | 86.5 | Unknown | REVERSAL in einen Valuation-Unwind eines Momentum-Stocks gekauft (+105% 1J). Trotz STARKER Q1-Earnings (Beat + Guidance-Raise auf +41-43% EPS-Wachstum) brach der Kurs ein wegen: Analyst-Downgrades Buy->Hold (BofA, Jefferies, Zacks; Bedenken zu Bewertung + Aerospace-Connectivity-Backlog-Slowdown) UND ~$250M Insider-Verkaeufe in 24-72h (CEO verkaufte $18.7M am 5.5). Stock -13.2% in einer Woche. base_range 27.6 = extrem breit = fallendes Messer. Stop nach nur 5 Tagen. Mean-Reversion eines parabolischen Momentum-Unwinds ist kein REVERSAL-Setup. |
| LKQ_2026-05-06 | REVERSAL | -7.61% | Stop Loss D+5 | 91.3 | Unknown | Fundamentale Margin-Compression + Securities-Lawsuit = REVERSAL traf strukturell geschwaechten Stock |
| AFRM_2026-05-29 | BREAKOUT | -7.46% | Stop Loss D+3 | 123.6 | Financial Services | _pending_ |
| PENN_2026-04-27 | BREAKOUT | -7.45% | Stop Loss D+10 | 110.2 | Consumer Cyclical | Post-Earnings-Spike-Chase - Q1 Earnings 23.4 mit 120% Surprise, Stock +16.7% pre-market; wir fired BREAKOUT 4 Tage spaeter = klassischer Late-Entry-Fade |
| MOS_2026-04-19 | REVERSAL | -7.09% | Stop Loss D+7 | 76.3 | Basic Materials | REVERSAL 10 Tage nach Brazilian-Phosphate-Idle-Announcement (9.4) + multiple Analyst-PT-Cuts - fundamental schwaechender Stock |
| S_2026-05-18 | BREAKOUT | -6.78% | Stop Loss D+8 | 122.4 | Unknown | SentinelOne BREAKOUT-Signal am 18.5. — Earnings am 28.5. = nur 10 Tage Abstand. Pre-Earnings-Nervositaet baute Pressure auf, Stop D+8 (also ~26.5., 2 Tage vor Earnings) traf den Drawdown. ABER: Das System hat uns vor dem 28.5.-Disaster gerettet — Q2-Guide-Miss + 8% Layoff-Ankuendigung loesten -18% Post-Earnings-Crash aus. Unser Stop bei -6.78% war besser als das Halten gewesen waere. KEIN Pocket-Pivot-Flag — zweiter Loser ohne PP (siehe HP). Score 122 + EMERGING_BREAKOUT = Setup sah gut aus, aber Earnings-Adjacency-Risk fuer BREAKOUT bestaetigt sich wieder (siehe FLS/BTU/XRAY). |
| GEV_2026-03-23 | BREAKOUT | -6.56% | Stop Loss D+5 | 89.0 | Industrials | BREAKOUT EXAKT am S&P-100-Inclusion-Tag (23.3) = index-buying war bereits gepriced, klassische 'sell the news' Fade |

## Best 10 Trades

| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |
|---|---|---|---|---|---|---|
| ENPH_2026-04-15 | REVERSAL | 31.6% | Take Profit D+20 | 60.9 | Technology | _pending_ |
| ENPH_2026-04-16 | REVERSAL | 31.6% | Take Profit D+19 | 61.1 | Technology | _pending_ |
| BLD_2026-04-01 | REVERSAL | 25.15% | Take Profit D+12 | 57.2 | Industrials | _pending_ |
| SCCO_2026-03-23 | REVERSAL | 19.55% | Take Profit D+13 | 67.2 | Basic Materials | _pending_ |
| NEM_2026-03-23 | REVERSAL | 17.99% | Take Profit D+11 | 64.4 | Basic Materials | _pending_ |
| RVTY_2026-05-04 | REVERSAL | 17.6% | Take Profit D+5 | 69.0 | Unknown | _pending_ |
| STX_2026-04-16 | BREAKOUT | 16.76% | Take Profit D+9 | 112.0 | Technology | BREAKOUT gewann +16.76% TROTZ extremer Extension (perf120 148%!) und niedrigem vol_ratio (0.78). Treiber: saekularer AI-Storage-Catalyst - Q3-FY26 (29.4) crushte (+44% Rev, EPS $4.10 vs $3.51), High-End-Kapazitaet bis FY2027 ausgebucht, Analyst-Upgrades auf $750. Plus Sektor-Tailwind (XLK +4.66%). Lehre: Extension allein ist KEIN Ausschlusskriterium - ein starker, bestaetigter Catalyst schlaegt die Extension (Gegenbeispiel zu WDC, das mit zu engem Stop trotz gleichem AI-Storage-Thema verlor). |
| CIEN_2026-03-18 | BREAKOUT | 16.64% | Take Profit D+5 | 105.5 | Technology | BREAKOUT gewann +16.64% TP D+5. Entry 18.3 lag 13 Tage NACH dem Q1-Earnings-Beat (5.3: Rekord-Rev $1.43B +33%) - also Post-Earnings-Momentum-Continuation auf einem bereits BESTAETIGTEN Catalyst (AI-Optical-Networking). Stock +27% in 30d, Analyst-PT-Raises (MS, JPM auf $380-400). Extension (perf120 178%) war kein Hindernis, weil der Catalyst (AI) real und bestaetigt war. Der GUTE Gegenpol zu FLS/XRAY: Earnings BEKANNT (Beat), nicht davor gewettet. |
| APP_2026-05-26 | BREAKOUT | 16.18% | Take Profit D+2 | 103.4 | Unknown | BREAKOUT mit dem perfekten Catalyst-Stack: Q1-2026-Beat ($1.84B Rev vs $1.78B Est), Q2-Guidance ueber Konsens ($1.92-1.95B Rev, $1.62B EBITDA), Analyst-PT-Raise-Welle (Morgan Stanley auf $720, UBS, DB, Macquarie, Wedbush, Oppenheimer, Jefferies bullish), PLUS AXON-Self-Serve-Plattform-Launch im Juni, PLUS Meta-Competition-Relief (kein non-IDFA-Bid). Score 103 + Pivot + Analyst-Flag = strong-conviction-breakout. Gewann TP D+2 trotz schwacher closing_strength 0.41 — die fundamentalen Treiber waren so dominant, dass intraday-Spike-Fade-Warnsignal irrelevant. |
| CLF_2026-05-26 | BREAKOUT | 13.64% | Take Profit D+1 | 103.0 | Unknown | Trump verkuendete am Freitag 23.5. nach US-Close die Verdopplung der Stahl-Tariffs von 25% auf 50%. CLF gappte am Montag 26.5. premarket +35% hoch. Reiner Policy-Tarif-Tailwind: Cleveland-Cliffs ist ein puer US-Stahlproduzent, profitiert maximal von Importzoellen. closing_strength 0.92 = sehr starker Tagesabschluss = der Move war institutionell, kein retail-fade. TP D+1, sauberer +13.6%-Win. |

## Pending Claude-Analyse (94)

Diese Trades warten auf Claude WebSearch-Verifikation + Lesson-Tagging:

- `XOM_2026-03-18`
- `MNST_2026-03-22`
- `PFE_2026-03-17`
- `DOW_2026-03-18`
- `VLO_2026-03-18`
- `UI_2026-03-18`
- `EXE_2026-03-19`
- `HP_2026-03-19`
- `EQT_2026-03-19`
- `PFE_2026-03-19`
- `SEE_2026-03-19`
- `SHEL_2026-03-19`
- `SEE_2026-03-22`
- `DE_2026-03-22`
- `SEE_2026-03-23`
- `EMR_2026-03-23`
- `BUD_2026-03-23`
- `NEM_2026-03-23`
- `AVT_2026-03-23`
- `CTVA_2026-03-24`
- _(+ 74 mehr)_
