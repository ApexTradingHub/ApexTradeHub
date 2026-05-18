# ApexLearn Report — 2026-05-18 21:48

Window: last 30 days (since 2026-04-18)  |  Filter: setup=ALL, ticker=ALL
Market: BULLISH — SPY=STRONG | QQQ=STRONG | Market=BULLISH

**Knowledge base** — 97 lifetime trades  |  WR 43.3%  |  PF 1.582  |  Range 2026-03-17 → 2026-05-13

## 1. Window Performance

- **7d**: n=2 | WR 0.0% | PF 0.00 | AvgW +0.00% | AvgL -4.59% | Total $-18.36
- **14d**: n=12 | WR 33.3% | PF 0.95 | AvgW +10.53% | AvgL -5.53% | Total $-4.29
- **30d**: n=31 | WR 22.6% | PF 0.51 | AvgW +10.23% | AvgL -5.89% | Total $-139.56
- **90d**: n=97 | WR 43.3% | PF 1.58 | AvgW +10.25% | AvgL -4.95% | Total $+316.67

## 2. Confirmed Facts (Knowledge Base)

**✅ CONFIRMED (n≥30) — vertrauenswürdige Fakten:**
- **BREAKOUT**: lifetime WR 55% (n=53, PF 1.993)
- **REVERSAL**: lifetime WR 30% (n=44, PF 1.306)

**⚠️ TENTATIVE (n=15-30) — Trend, brauche mehr Daten:**
- **BREAKOUT × perf_120_0_25**: -27% vs setup-baseline (WR 28%, n=18)
- **BREAKOUT × perf_120_25_50**: +16% vs setup-baseline (WR 71%, n=24)
- **BREAKOUT × rsi_60_65**: -16% vs setup-baseline (WR 39%, n=23)
- **BREAKOUT × vol_lt_1**: -11% vs setup-baseline (WR 44%, n=23)
- **REVERSAL × perf_120_lt_0**: -11% vs setup-baseline (WR 18%, n=22)
- **BREAKOUT score 90-100**: 71% WR (n=17) — elite zone

**🔍 HYPOTHESIS (n<15) — nur Auffälligkeit, kein Beweis:**
- **analyst_upside_gt_15pct**: -36% WR-Lift (with 11% vs without 47%, n_present=9)
- **BREAKOUT × rsi_70plus**: +25% vs setup-baseline (WR 80%, n=10)
- **REVERSAL × perf_120_25_50**: +42% vs setup-baseline (WR 71%, n=7)
- **REVERSAL × vol_15_25**: +20% vs setup-baseline (WR 50%, n=12)

## 3. Per-Setup Window-Performance

| Setup | n | WR | PF | AvgWin | AvgLoss |
|---|---|---|---|---|---|
| REVERSAL | 18 | 5.6% | 0.17 | +17.60% | -6.23% |
| BREAKOUT | 13 | 46.2% | 1.52 | +9.01% | -5.07% |

## 4. Per-Setup Lifetime-Performance (Knowledge)

| Setup | n | WR | PF | AvgWin | AvgLoss | Confidence |
|---|---|---|---|---|---|---|
| BREAKOUT | 53 | 54.7% | 1.99 | +7.50% | -4.55% | HIGH |
| REVERSAL | 44 | 29.5% | 1.31 | +16.39% | -5.26% | HIGH |

## 5. Catalyst Effectiveness (Lifetime)

| Catalyst | n_present | WR with | WR without | Lift | Confidence |
|---|---|---|---|---|---|
| analyst_upside_gt_15pct | 9 | 11.1% | 46.6% | -35.5% | LOW |
| cat_pocket_pivot | 7 | 42.9% | 43.3% | -0.5% | LOW |

## 6. Score-Gate Calibration (Lifetime — validiert Gate-Setting)

**BREAKOUT:**
| Score-Bucket | n | actual WR | conf |
|---|---|---|---|
| 100-999 | 7 | 71.4% | LOW |
| 60-70 | 6 | 33.3% | LOW |
| 70-80 | 12 | 41.7% | LOW |
| 80-90 | 11 | 45.5% | LOW |
| 90-100 | 17 | 70.6% | MED |

**REVERSAL:**
| Score-Bucket | n | actual WR | conf |
|---|---|---|---|
| 40-60 | 7 | 14.3% | LOW |
| 60-70 | 11 | 72.7% | LOW |
| 70-80 | 13 | 30.8% | LOW |
| 80-90 | 8 | 0.0% | LOW |

## 7. Per-Ticker Heatmap (Lifetime, n≥3)

**🏆 Top-Performer (sortiert WR):**

| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |
|---|---|---|---|---|---|---|
| MUR | 3 | 100% | +11.14% | +12.4% | +10.0% | BREAKOUT |
| SEE | 3 | 100% | +0.29% | +0.3% | +0.2% | BREAKOUT |
| ENPH | 3 | 67% | +17.75% | +31.6% | -10.0% | REVERSAL |
| CACI | 3 | 0% | -7.00% | -3.7% | -9.3% | REVERSAL |

**💀 Worst-Performer (sortiert WR):**

| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |
|---|---|---|---|---|---|---|
| CACI | 3 | 0% | -7.00% | -3.7% | -9.3% | REVERSAL |
| ENPH | 3 | 67% | +17.75% | +31.6% | -10.0% | REVERSAL |
| MUR | 3 | 100% | +11.14% | +12.4% | +10.0% | BREAKOUT |
| SEE | 3 | 100% | +0.29% | +0.3% | +0.2% | BREAKOUT |

## 8. Failure-Modes (Lifetime)

| Failure Pattern | count | % of losses | avg pnl | by setup |
|---|---|---|---|---|
| quick_stop_1_3d | 18 | 33% | -5.23% | BREAKOUT:8, REVERSAL:10 |
| slow_stop_4plus | 31 | 56% | -5.30% | REVERSAL:17, BREAKOUT:14 |
| time_exit_negative | 3 | 6% | -2.01% | BREAKOUT:2, REVERSAL:1 |
| high_score_loss_85plus | 16 | 29% | -5.63% | BREAKOUT:9, REVERSAL:7 |

## 9. Window-Trade-Detail (letzte 31)

- **2026-05-13** TEL    REVERSAL       | 🔴 L -4.98% | Stop Loss D+3 | score 76.2
- **2026-05-11** XRAY   REVERSAL       | 🔴 L -4.20% | Stop Loss D+2 | score 100.0 PP
- **2026-05-09** AMTM   REVERSAL       | 🔴 L -5.80% | Stop Loss D+3 | score 86.7
- **2026-05-08** RIVN   REVERSAL       | 🔴 L -5.67% | Stop Loss D+6 | score 88.0
- **2026-05-07** DG     REVERSAL       | 🔴 L -5.30% | Stop Loss D+2 | score 78.9 PP
- **2026-05-06** BKNG   REVERSAL       | 🔴 L -5.01% | Stop Loss D+3 | score 77.7 GAP
- **2026-05-06** LKQ    REVERSAL       | 🔴 L -7.61% | Stop Loss D+5 | score 91.3 PP
- **2026-05-06** HD     REVERSAL       | 🔴 L -5.70% | Stop Loss D+5 | score 81.2 PP GAP
- **2026-05-05** DVA    BREAKOUT       | 🟢 W +8.27% | Take Profit D+1 | score 104.9 PP SHORT
- **2026-05-05** CBT    BREAKOUT       | 🟢 W +8.11% | Take Profit D+1 | score 90.7 PP
- **2026-05-05** BWA    BREAKOUT       | 🟢 W +8.13% | Take Profit D+4 | score 99.2 PP
- **2026-05-04** RVTY   REVERSAL       | 🟢 W +17.60% | Take Profit D+5 | score 69.0
- **2026-05-01** XRAY   REVERSAL       | 🔴 L -6.49% | Stop Loss D+2 | score 65.2
- **2026-05-01** BTU    REVERSAL       | 🔴 L -8.73% | Stop Loss D+3 | score 82.2
- **2026-05-01** CACI   REVERSAL       | 🔴 L -8.03% | Stop Loss D+3 | score 72.6
- **2026-05-01** BWA    BREAKOUT       | 🟢 W +9.53% | Take Profit D+6 | score 90.7
- **2026-04-29** HII    REVERSAL       | 🔴 L -5.03% | Stop Loss D+4 | score 78.3
- **2026-04-27** FLS    BREAKOUT       | 🔴 L -6.18% | Stop Loss D+3 | score 110.9
- **2026-04-27** MGM    BREAKOUT       | 🔴 L -5.28% | Stop Loss D+3 | score 98.0
- **2026-04-27** FTI    BREAKOUT       | 🔴 L -4.52% | Stop Loss D+8 | score 91.5
- **2026-04-27** FLR    BREAKOUT       | 🔴 L -5.17% | Stop Loss D+9 | score 77.3
- **2026-04-27** PANW   BREAKOUT       | 🟢 W +10.45% | Take Profit D+9 | score 96.8
- **2026-04-27** PENN   BREAKOUT       | 🔴 L -7.45% | Stop Loss D+10 | score 110.2
- **2026-04-23** CACI   REVERSAL       | 🔴 L -9.28% | Stop Loss D+9 | score 81.2
- **2026-04-22** HUBB   BREAKOUT       | 🔴 L -4.43% | Stop Loss D+6 | score 93.5
- **2026-04-22** TSM    BREAKOUT       | 🟢 W +9.55% | Time Exit D+15 | score 100.4
- **2026-04-20** ENPH   REVERSAL       | 🔴 L -9.96% | Stop Loss D+7 | score 91.1
- **2026-04-20** CZR    BREAKOUT       | 🔴 L -2.46% | Time Exit D+15 | score 76.3
- **2026-04-19** CACI   REVERSAL       | 🔴 L -3.68% | Stop Loss D+3 | score 91.5
- **2026-04-19** HSY    REVERSAL       | 🔴 L -3.40% | Stop Loss D+6 | score 84.6

## 10. Offene Positionen (1)

_Filtered: 57 legacy, 3 expired-hold, 0 stale-no-trigger_

| Signal | D+ | Ticker | Setup | Entry | Stop | Target | Score |
|---|---|---|---|---|---|---|---|
| 2026-05-17 | D+1 | COST | STAGE_2 | $1054.19 | $924.56 | $1350.96 | 101.6 |

## 12. Window-vs-Lifetime Drift (worth investigating?)

- TOTAL WR last-30d (22.6%) vs lifetime (43.3%): 📉 worse by 20.7pp
- REVERSAL last-30d 5.6% vs lifetime 29.5%: 📉 worse by 23.9pp

---
_Report-Ende. Knowledge-File: `knowledge\apex_knowledge.json`. Diesen Output an Claude füttern für datengestützte Verbesserungen._