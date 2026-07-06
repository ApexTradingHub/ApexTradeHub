# ApexLearn Report — 2026-06-24 22:13

Window: last 30 days (since 2026-05-25)  |  Filter: setup=ALL, ticker=ALL
Market: MIXED — SPY=OK | QQQ=OK | Market=MIXED

**Knowledge base** — 98 lifetime trades  |  WR 56.1%  |  PF 2.265  |  Range 2026-03-17 → 2026-06-19

## 1. Window Performance

- **7d**: n=1 | WR 0.0% | PF 0.00 | AvgW +0.00% | AvgL -6.58% | Total $-13.16
- **14d**: n=5 | WR 20.0% | PF 0.42 | AvgW +10.11% | AvgL -5.98% | Total $-27.65
- **30d**: n=30 | WR 50.0% | PF 1.80 | AvgW +8.25% | AvgL -4.58% | Total $+109.90
- **90d**: n=68 | WR 54.4% | PF 2.33 | AvgW +8.98% | AvgL -4.60% | Total $+379.91

## 2. Confirmed Facts (Knowledge Base)

**✅ CONFIRMED (n≥30) — vertrauenswürdige Fakten:**
- **🔵 Breakout**: lifetime WR 56% (n=98, PF 2.265)
- **cat_pocket_pivot**: +12% WR-Lift (with 64% vs without 52%, n_present=31)

**⚠️ TENTATIVE (n=15-30) — Trend, brauche mehr Daten:**
- _(keine)_

**🔍 HYPOTHESIS (n<15) — nur Auffälligkeit, kein Beweis:**
- **gap_gt_2pct**: +12% WR-Lift (with 67% vs without 55%, n_present=9)
- **analyst_upside_gt_15pct**: +46% WR-Lift (with 100% vs without 54%, n_present=5)
- **🔵 Breakout × perf_120_lt_0**: +22% vs setup-baseline (WR 78%, n=9)
- **🔵 Breakout × rsi_50_60**: +10% vs setup-baseline (WR 67%, n=9)
- **🔵 Breakout × rsi_70plus**: +19% vs setup-baseline (WR 75%, n=12)

## 3. Per-Setup Window-Performance

| Setup | n | WR | PF | AvgWin | AvgLoss |
|---|---|---|---|---|---|
| 🔵 Breakout | 30 | 50.0% | 1.80 | +8.25% | -4.58% |

## 4. Per-Setup Lifetime-Performance (Knowledge)

| Setup | n | WR | PF | AvgWin | AvgLoss | Confidence |
|---|---|---|---|---|---|---|
| 🔵 Breakout | 98 | 56.1% | 2.27 | +8.28% | -4.67% | HIGH |

## 5. Catalyst Effectiveness (Lifetime)

| Catalyst | n_present | WR with | WR without | Lift | Confidence |
|---|---|---|---|---|---|
| analyst_upside_gt_15pct | 5 | 100.0% | 53.8% | +46.2% | LOW |
| cat_pocket_pivot | 31 | 64.5% | 52.2% | +12.3% | HIGH |
| gap_gt_2pct | 9 | 66.7% | 55.1% | +11.6% | LOW |

## 6. Score-Gate Calibration (Lifetime — validiert Gate-Setting)

**🔵 Breakout:**
| Score-Bucket | n | actual WR | conf |
|---|---|---|---|
| 100-999 | 47 | 61.7% | HIGH |
| 60-70 | 6 | 33.3% | LOW |
| 70-80 | 13 | 38.5% | LOW |
| 80-90 | 12 | 50.0% | LOW |
| 90-100 | 20 | 65.0% | MED |

## 7. Per-Ticker Heatmap (Lifetime, n≥3)

**🏆 Top-Performer (sortiert WR):**

| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |
|---|---|---|---|---|---|---|
| MUR | 3 | 100% | +11.14% | +12.4% | +10.0% | 🔵 Breakout |
| SEE | 3 | 100% | +0.29% | +0.3% | +0.2% | 🔵 Breakout |
| ASML | 5 | 40% | +1.00% | +9.4% | -6.6% | 🔵 Breakout |
| TSM | 3 | 33% | +0.30% | +9.6% | -5.3% | 🔵 Breakout |

**💀 Worst-Performer (sortiert WR):**

| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |
|---|---|---|---|---|---|---|
| TSM | 3 | 33% | +0.30% | +9.6% | -5.3% | 🔵 Breakout |
| ASML | 5 | 40% | +1.00% | +9.4% | -6.6% | 🔵 Breakout |
| MUR | 3 | 100% | +11.14% | +12.4% | +10.0% | 🔵 Breakout |
| SEE | 3 | 100% | +0.29% | +0.3% | +0.2% | 🔵 Breakout |

## 8. Failure-Modes (Lifetime)

| Failure Pattern | count | % of losses | avg pnl | by setup |
|---|---|---|---|---|
| quick_stop_1_3d | 13 | 30% | -5.41% | 🔵 Breakout:13 |
| slow_stop_4plus | 24 | 56% | -5.03% | 🔵 Breakout:24 |
| time_exit_negative | 6 | 14% | -1.66% | 🔵 Breakout:6 |
| high_score_loss_85plus | 27 | 63% | -5.10% | 🔵 Breakout:27 |

## 9. Window-Trade-Detail (letzte 30)

- **2026-06-19** ASML   🔵 Breakout         | 🔴 L -6.58% | Stop Loss D+2 | score 144.4 GAP
- **2026-06-16** EMR    🔵 Breakout         | 🔴 L -4.45% | Stop Loss D+4 | score 98.1
- **2026-06-15** JBL    🔵 Breakout         | 🔴 L -6.48% | Stop Loss D+4 | score 123.3 GAP
- **2026-06-15** ASML   🔵 Breakout         | 🔴 L -6.42% | Stop Loss D+5 | score 146.3 PP
- **2026-06-12** LUV    🔵 Breakout         | 🟢 W +10.11% | Take Profit D+5 | score 115.5 PP
- **2026-06-09** ARE    🔵 Breakout         | 🔴 L -6.06% | Stop Loss D+6 | score 104.9
- **2026-06-09** SWK    🔵 Breakout         | 🟢 W +8.35% | Take Profit D+8 | score 101.3 PP
- **2026-06-09** ASML   🔵 Breakout         | 🔴 L -0.25% | Time Exit D+15 | score 132.4 PP
- **2026-06-09** CARR   🔵 Breakout         | 🟢 W +1.11% | Time Exit D+15 | score 115.6
- **2026-06-08** MOH    🔵 Breakout         | 🟢 W +0.66% | Time Exit D+15 | score 112.7
- **2026-06-04** AXTA   🔵 Breakout         | 🟢 W +8.97% | Take Profit D+7 | score 106.1 PP GAP
- **2026-06-03** JCI    🔵 Breakout         | 🔴 L -4.43% | Stop Loss D+5 | score 105.7 PEAD
- **2026-06-02** ARE    🔵 Breakout         | 🟢 W +1.29% | Time Exit D+15 | score 124.6 PP
- **2026-06-01** IBKR   🔵 Breakout         | 🔴 L -5.39% | Stop Loss D+3 | score 123.3 PP
- **2026-06-01** AYI    🔵 Breakout         | 🔴 L -4.27% | Stop Loss D+6 | score 78.8 PP
- **2026-06-01** TSM    🔵 Breakout         | 🔴 L -5.32% | Stop Loss D+6 | score 115.2 PP
- **2026-05-29** AFRM   🔵 Breakout         | 🔴 L -7.46% | Stop Loss D+3 | score 123.6 PP
- **2026-05-29** AVGO   🔵 Breakout         | 🟢 W +10.55% | Take Profit D+3 | score 115.1 PP
- **2026-05-29** ODFL   🔵 Breakout         | 🟢 W +9.25% | Take Profit D+4 | score 110.4 PP
- **2026-05-28** ORCL   🔵 Breakout         | 🟢 W +11.35% | Take Profit D+1 | score 109.8 PP
- **2026-05-28** CZR    🔵 Breakout         | 🟢 W +1.58% | Time Exit D+15 | score 110.5 PP VC
- **2026-05-27** CBT    🔵 Breakout         | 🔴 L -4.46% | Stop Loss D+7 | score 133.9 PP
- **2026-05-27** GM     🔵 Breakout         | 🔴 L -1.16% | Time Exit D+15 | score 115.0 PP
- **2026-05-27** TSM    🔵 Breakout         | 🔴 L -3.33% | Time Exit D+15 | score 103.9 GAP
- **2026-05-26** CLF    🔵 Breakout         | 🟢 W +13.64% | Take Profit D+1 | score 103.0 PP GAP
- **2026-05-26** APP    🔵 Breakout         | 🟢 W +16.18% | Take Profit D+2 | score 103.4 PP GAP
- **2026-05-26** AMAT   🔵 Breakout         | 🟢 W +10.30% | Take Profit D+6 | score 132.1 GAP
- **2026-05-26** KLAC   🔵 Breakout         | 🟢 W +11.51% | Take Profit D+7 | score 126.3 GAP
- **2026-05-26** BC     🔵 Breakout         | 🔴 L -2.70% | Time Exit D+15 | score 97.9 PP
- **2026-05-25** ASML   🔵 Breakout         | 🟢 W +8.86% | Time Exit D+15 | score 104.7

## 10. Offene Positionen (6)

_Filtered: 44 legacy, 7 expired-hold, 29 stale-no-trigger_

| Signal | D+ | Ticker | Setup | Entry | Stop | Target | Score |
|---|---|---|---|---|---|---|---|
| 2026-06-22 | D+2 | TSM | 🔵 Breakout | $466.15 | $439.03 | $520.71 | 123.7 |
| 2026-06-22 | D+2 | SW | 🔵 Breakout | $45.67 | $43.04 | $51.6 | 114.8 |
| 2026-06-22 | D+2 | ABBV | 🔵 Breakout | $230.93 | $221.72 | $250.0 | 98.7 |
| 2026-06-22 | D+2 | D | 🚀 Trend | $68.84 | $59.65 | $87.37 | 92.4 |
| 2026-06-22 | D+2 | JCI | 🔵 Breakout | $149.68 | $142.62 | $164.1 | 91.3 |
| 2026-06-22 | D+2 | NVO | 🔵 Breakout | $46.14 | $43.9 | $50.0 | 83.3 |

## 12. Window-vs-Lifetime Drift (worth investigating?)

- TOTAL WR last-30d (50.0%) vs lifetime (56.1%): 📉 worse by 6.1pp

---
_Report-Ende. Knowledge-File: `knowledge\apex_knowledge.json`. Diesen Output an Claude füttern für datengestützte Verbesserungen._