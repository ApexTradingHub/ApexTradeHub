# Postmortem Database — Summary

_Updated: 2026-07-22T09:17:05.737968 | Total trades: 225_

**Analyse-Status**: complete=216, pending=9

## Lesson-Tag Frequency (Claude curated)

| Tag | Count |
|---|---|
| `sector_momentum_tailwind` | 34 |
| `breakout_no_follow_through` | 34 |
| `sector_relative_strength` | 29 |
| `post_earnings_beat_continuation` | 29 |
| `high_score_loss_85plus` | 26 |
| `reversal_after_oversold_trap` | 24 |
| `duplicate_trap` | 22 |
| `reversal_win_external_catalyst` | 21 |
| `secular_ai_catalyst` | 20 |
| `defensive_low_beta_drift` | 18 |
| `oversold_bounce_with_catalyst` | 18 |
| `oil_peak_chase` | 15 |
| `energy_oil_surge_2026q1` | 15 |
| `fundamental_deterioration` | 12 |
| `geopolitical_premium_fade` | 12 |
| `earnings_adjacency_risk` | 11 |
| `low_score_loss` | 10 |
| `reversal_no_turn` | 9 |
| `pocket_pivot_validation` | 9 |
| `ai_capex_supercycle` | 8 |

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
| COST_2026-05-17 | STAGE_2 | -12.3% | Stop Loss D+31 | 101.6 | Consumer Defensive | STAGE_2 (einziger Nicht-BREAKOUT) -12.3% Stop nach D+31. Costco meldete Blowout-Q3 (Net Income +15.2%, Sales +11.6%) am 28.5 — aber Aktie FIEL (Sell-the-News: 52x P/E zu teuer) + Tariff-Class-Action-Klagen (IEEPA). STAGE_2's langer 60d-Hold liess einen -12% Drawdown entstehen. |
| ENPH_2026-04-20 | REVERSAL | -9.96% | Stop Loss D+7 | 91.1 | Technology | Stop ausgeloest 1 Tag VOR Earnings-Beat - REVERSAL stoppt sich aus dem profitablen Move raus |
| UPST_2026-06-29 | BREAKOUT | -9.55% | Stop Loss D+6 | 102.2 | Financial Services | Macro-Selloff Ende Juni 2026 (steigende Bond-Yields + Oil-Spike revived Inflations-Fear + hawkishe Fed) traf High-Beta-Long-Duration-Growth ueberproportional. UPST bereits -56%/Jahr (perf_120 -30.9%) = BREAKOUT auf tief-negativer Momentum-Basis. Fundamental intakt (Origination +48% YoY), reiner Macro-vs-Setup-Konflikt. D+6 Stop -9.6%. |
| SW_2026-05-04 | REVERSAL | -9.44% | Stop Loss D+11 | 64.6 | Unknown | REVERSAL gekauft 4 Tage vor Q1-2026-Earnings, die massiv verfehlten: EPS $0.33 vs $0.41 erwartet (-19.5% Surprise), Net Income -83% YoY auf $63M (vs $382M). Net-Marge brach von 5.0% auf 0.8% ein (hoehere Depreciation/Impairment/Restructuring + ~$65M Wetter-Headwinds + schwaechere NA-Volumina). 7-Tage-Verlustserie -9.5%, Stop am 19.5 bei -4.6% Tagesverlust. Oversold-RSI war Falle - der Abverkauf war fundamental getrieben. |
| WDC_2026-03-18 | BREAKOUT | -9.37% | Stop Loss D+3 | 94.4 | Technology | Stop zu eng fuer hochvolatilen AI-Storage-Stock (170% YTD, 845% 1y) - getoppt nach 3 Tagen, dann massive Continuation verpasst |
| HII_2026-05-12 | REVERSAL | -9.34% | Stop Loss D+13 | 90.9 | Industrials | REVERSAL-Lehrbuch-Versagen: oversold RSI 35.6 war Falle. Gekauft ~7 Tage NACH Q1-Earnings (5.5) in fundamentale Deterioration: Segment-Margin 5.6% (von 6.3%), neg. FCF -$461M, $390M Cash-Burn, Insider-Form-144 am 5.5. Stock -12% post-earnings, weiter -4% am 29.5. analyst_upside 19.4% (>15 = ANTI-predictive bestaetigt). Score 90.9 = High-Score-Loss. Kombi aller REVERSAL-Failure-Modes. |
| CACI_2026-04-23 | REVERSAL | -9.28% | Stop Loss D+9 | 81.2 | Technology | Identisch zu CACI_2026-04-19 - REVERSAL #2 in Downgrade-Phase, kein Lerneffekt durch System |
| KLAC_2026-06-29 | BREAKOUT | -9.28% | Stop Loss D+3 | 117.8 | Technology | Semi-Sektor-Selloff Ende Juni 2026 (Memory-Kosten-Fear nach Apple/Microsoft-Preiserhoehungen, AI-Spending-Scrutiny, TSMC-Stake-Sale, Samsung-Streik-News). LRCX verlor >100B Marktwert. Hoch-Score Tech-BREAKOUT lief in einen sektorweiten Baerenmove; XLK/Semi divergierte negativ TROTZ positivem SPY. Quick-Stop D+3-4. |
| LRCX_2026-06-29 | BREAKOUT | -9.2% | Stop Loss D+3 | 125.5 | Technology | Semi-Sektor-Selloff Ende Juni 2026 (Memory-Kosten-Fear nach Apple/Microsoft-Preiserhoehungen, AI-Spending-Scrutiny, TSMC-Stake-Sale, Samsung-Streik-News). LRCX verlor >100B Marktwert. Hoch-Score Tech-BREAKOUT lief in einen sektorweiten Baerenmove; XLK/Semi divergierte negativ TROTZ positivem SPY. Quick-Stop D+3-4. |
| BTU_2026-05-01 | REVERSAL | -8.73% | Stop Loss D+3 | 82.2 | Energy | REVERSAL 4 Tage vor Q1 Earnings - Earnings am 5.5 zeigten Net Loss & EBITDA -43%, Production Cut |
| ANET_2026-07-08 | BREAKOUT | -8.57% | Stop Loss D+7 | 137.0 | Technology | _pending_ |
| CACI_2026-05-01 | REVERSAL | -8.03% | Stop Loss D+3 | 72.6 | Technology | REVERSAL #3 fuer CACI in 12 Tagen - Phase E DUPLICATE_WINDOW (3d) hat nicht ausgereicht, weil wir alle 10-15 Tage neu signalisiert haben |
| AMTM_2026-04-01 | REVERSAL | -8.01% | Stop Loss D+23 | 59.7 | Industrials | DEAD_CAT_BOUNCE - Stock war -20% YTD, REVERSAL kaufte trotz fundamentaler Schwaeche; nahm 23 Tage zum Stop |
| APH_2026-05-12 | REVERSAL | -7.84% | Stop Loss D+5 | 86.5 | Unknown | REVERSAL in einen Valuation-Unwind eines Momentum-Stocks gekauft (+105% 1J). Trotz STARKER Q1-Earnings (Beat + Guidance-Raise auf +41-43% EPS-Wachstum) brach der Kurs ein wegen: Analyst-Downgrades Buy->Hold (BofA, Jefferies, Zacks; Bedenken zu Bewertung + Aerospace-Connectivity-Backlog-Slowdown) UND ~$250M Insider-Verkaeufe in 24-72h (CEO verkaufte $18.7M am 5.5). Stock -13.2% in einer Woche. base_range 27.6 = extrem breit = fallendes Messer. Stop nach nur 5 Tagen. Mean-Reversion eines parabolischen Momentum-Unwinds ist kein REVERSAL-Setup. |
| LKQ_2026-05-06 | REVERSAL | -7.61% | Stop Loss D+5 | 91.3 | Unknown | Fundamentale Margin-Compression + Securities-Lawsuit = REVERSAL traf strukturell geschwaechten Stock |

## Best 10 Trades

| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |
|---|---|---|---|---|---|---|
| ENPH_2026-04-15 | REVERSAL | 31.6% | Take Profit D+20 | 60.9 | Technology | REVERSAL kaufte oversold (RSI 35) bei perf_120 -13.2% = echter beaten-down Stock. Win aus 60%-Rally off late-April-Lows, getrieben von Q1-2026-Beat + IQ9S-3P Commercial-Microinverter-Launch (Pre-Orders +13% am 13.5) + IQ-Solid-State-Transformer AI-Story. Naeher an echtem Reversal als SCCO/NEM, aber dennoch Catalyst-getrieben. Held 20d. |
| ENPH_2026-04-16 | REVERSAL | 31.6% | Take Profit D+19 | 61.1 | Technology | Duplikat-Signal zu ENPH_2026-04-15 (gleicher Entry $32.34, gleicher Exit $42.56, 1 Tag spaeter). Selbe Win-Ursache. ZEIGT DUPLICATE-TRAP: System signalisierte denselben Trade 2x in 2 Tagen, im Live-System doppelte Kapital-Allokation. |
| BLD_2026-04-01 | REVERSAL | 25.15% | Take Profit D+12 | 57.2 | Industrials | REINER M&A-GLUECKSFALL, KEIN Setup-Edge. QXO kuendigte 18.4.2026 $14.3B-Buyout zu $505/Share an. Entry $371, Exit $464 (+25%) = Buyout-Pop. REVERSAL-Logik (RSI 43.5, perf_120 -15.4%) hat NICHTS vorhergesagt, ein Downtrend-Stock wurde zufaellig uebernommen. NICHT replizierbar. |
| SCCO_2026-03-23 | REVERSAL | 19.55% | Take Profit D+13 | 67.2 | Basic Materials | REVERSAL fing zufaellig den Metalle-Bullenmarkt (Maerz 2026 Silber/Zink/Kupfer-Rekorde). Win kam aus Sektor-Tailwind (Q4-2025 Silber-Umsatz +106%, Rekord-Net-Sales $13.4B +17% YoY), NICHT aus Mean-Reversion-Logik. RSI 36 war oversold, aber perf_120 bereits +35.4% = kein Falling-Knife sondern Pullback im intakten Bull. |
| NEM_2026-03-23 | REVERSAL | 17.99% | Take Profit D+11 | 64.4 | Basic Materials | REVERSAL fing den historischen Gold-Rally (Realized-Gold >$3500/oz Ende 2025). RSI 31 oversold + perf_120 +16.7%. Win aus Sektor-Tailwind (Gold-Rekord) + Ahafo-North-Mine-Start Okt 2025, NICHT aus Reversal-Setup. Identisches Muster wie SCCO: Pullback im Commodity-Bull. |
| RVTY_2026-05-04 | REVERSAL | 17.6% | Take Profit D+5 | 69.0 | Unknown | REVERSAL Revvity, TP +17.60% D+5. Schneller starker Bounce Life-Sciences. Score 69 niedrig (REVERSAL-Score anti-prediktiv). Oversold-Bounce mit Turn. |
| STX_2026-04-16 | BREAKOUT | 16.76% | Take Profit D+9 | 112.0 | Technology | BREAKOUT gewann +16.76% TROTZ extremer Extension (perf120 148%!) und niedrigem vol_ratio (0.78). Treiber: saekularer AI-Storage-Catalyst - Q3-FY26 (29.4) crushte (+44% Rev, EPS $4.10 vs $3.51), High-End-Kapazitaet bis FY2027 ausgebucht, Analyst-Upgrades auf $750. Plus Sektor-Tailwind (XLK +4.66%). Lehre: Extension allein ist KEIN Ausschlusskriterium - ein starker, bestaetigter Catalyst schlaegt die Extension (Gegenbeispiel zu WDC, das mit zu engem Stop trotz gleichem AI-Storage-Thema verlor). |
| CIEN_2026-03-18 | BREAKOUT | 16.64% | Take Profit D+5 | 105.5 | Technology | BREAKOUT gewann +16.64% TP D+5. Entry 18.3 lag 13 Tage NACH dem Q1-Earnings-Beat (5.3: Rekord-Rev $1.43B +33%) - also Post-Earnings-Momentum-Continuation auf einem bereits BESTAETIGTEN Catalyst (AI-Optical-Networking). Stock +27% in 30d, Analyst-PT-Raises (MS, JPM auf $380-400). Extension (perf120 178%) war kein Hindernis, weil der Catalyst (AI) real und bestaetigt war. Der GUTE Gegenpol zu FLS/XRAY: Earnings BEKANNT (Beat), nicht davor gewettet. |
| APP_2026-05-26 | BREAKOUT | 16.18% | Take Profit D+2 | 103.4 | Unknown | BREAKOUT mit dem perfekten Catalyst-Stack: Q1-2026-Beat ($1.84B Rev vs $1.78B Est), Q2-Guidance ueber Konsens ($1.92-1.95B Rev, $1.62B EBITDA), Analyst-PT-Raise-Welle (Morgan Stanley auf $720, UBS, DB, Macquarie, Wedbush, Oppenheimer, Jefferies bullish), PLUS AXON-Self-Serve-Plattform-Launch im Juni, PLUS Meta-Competition-Relief (kein non-IDFA-Bid). Score 103 + Pivot + Analyst-Flag = strong-conviction-breakout. Gewann TP D+2 trotz schwacher closing_strength 0.41 — die fundamentalen Treiber waren so dominant, dass intraday-Spike-Fade-Warnsignal irrelevant. |
| LRCX_2026-06-11 | BREAKOUT | 14.87% | Time Exit D+15 | 161.0 | Technology | BREAKOUT-WIN +14.87% (Time-Exit D+15, Score 161 = hoechster). Record-Q3-FY26 (Rev $5.84B +24% YoY, EPS +41%) getrieben von AI-WFE-Nachfrage; +28.7% in 30d. Micron-Q3 hob den ganzen Semi-Sektor. Analyst-PT-Welle (Wells Fargo $450, Citi $450, Oppenheimer $400). WICHTIG: Score 161 HAT hier geliefert (Gegenbeispiel zu high_score_loss — mit echtem AI-Catalyst skaliert hoher Score zum grossen Win). |

## Recent (30d) — Worst 5 / Best 5  (n=43)

| Trade | Setup | PnL% | Exit | Score | Sektor | Tags |
|---|---|---|---|---|---|---|
| UPST_2026-06-29 | BREAKOUT | -9.55% | Stop Loss D+6 | 102.2 | Financial Services | macro_selloff_correlates_all_stocks,negative_perf120_breakout,high_beta_macro_risk |
| KLAC_2026-06-29 | BREAKOUT | -9.28% | Stop Loss D+3 | 117.8 | Technology | semi_selloff_2026q2,high_score_loss_85plus,sector_divergence_loser |
| LRCX_2026-06-29 | BREAKOUT | -9.2% | Stop Loss D+3 | 125.5 | Technology | semi_selloff_2026q2,high_score_loss_85plus,sector_divergence_loser |
| ANET_2026-07-08 | BREAKOUT | -8.57% | Stop Loss D+7 | 137.0 | Technology | — |
| TRIP_2026-07-16 | BREAKOUT | -7.08% | Stop Loss D+3 | 114.8 | Consumer Cyclical | — |
| RHI_2026-07-13 | BREAKOUT | 13.58% | Take Profit D+3 | 141.3 | Industrials | high_score_winner,clean_breakout_profile,pick_band_cost |
| SE_2026-07-01 | BREAKOUT | 13.31% | Take Profit D+5 | 102.8 | Consumer Cyclical | post_earnings_beat_continuation,oversold_bounce_with_catalyst,catalyst_beats_extension |
| TECH_2026-06-24 | BREAKOUT | 10.77% | Take Profit D+1 | 90.5 | Healthcare | ma_buyout_catalyst,duplicate_trap,reversal_win_external_catalyst |
| MPC_2026-07-10 | BREAKOUT | 8.42% | Take Profit D+4 | 101.4 | Energy | sector_momentum_tailwind,catalyst_beats_extension,geopolitical_premium |
| BBY_2026-07-10 | BREAKOUT | 8.38% | Take Profit D+4 | 120.2 | Consumer Cyclical | post_earnings_beat_continuation,catalyst_beats_extension,product_catalyst |

## Pending Claude-Analyse (9)

Diese Trades warten auf Claude WebSearch-Verifikation + Lesson-Tagging:

- `CMG_2026-07-13`
- `ILMN_2026-07-02`
- `BAX_2026-07-02`
- `HOOD_2026-07-02`
- `ANET_2026-07-08`
- `GS_2026-07-14`
- `PCAR_2026-07-06`
- `TRIP_2026-07-16`
- `LW_2026-07-07`
