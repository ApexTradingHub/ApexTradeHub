# Postmortem Database — Summary

_Updated: 2026-05-18T22:29:23.981193 | Total trades: 97_

**Analyse-Status**: complete=11, pending=86

## Lesson-Tag Frequency (Claude curated)

| Tag | Count |
|---|---|
| `reversal_after_analyst_downgrade` | 4 |
| `fundamental_deterioration` | 3 |
| `earnings_adjacency_risk` | 2 |
| `dead_cat_bounce` | 2 |
| `duplicate_trap` | 2 |
| `earnings_miss_during_hold` | 1 |
| `high_volatility_tight_stop` | 1 |
| `vertical_move_breakout_fail` | 1 |
| `ai_storage_momentum` | 1 |
| `index_inclusion_fade` | 1 |
| `post_announcement_fade` | 1 |
| `extended_stock_breakout` | 1 |
| `institutional_rerating_in_progress` | 1 |
| `defense_sector_specific` | 1 |
| `guidance_reduction` | 1 |
| `reversal_stopped_before_catalyst` | 1 |
| `stop_too_tight_pre_earnings` | 1 |
| `downtrend_stock_reversal_fail` | 1 |
| `slow_bleed_stop` | 1 |
| `no_cooldown_after_recent_loss` | 1 |

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
| WDC_2026-03-18 | BREAKOUT | -9.37% | Stop Loss D+3 | 94.4 | Technology | Stop zu eng fuer hochvolatilen AI-Storage-Stock (170% YTD, 845% 1y) - getoppt nach 3 Tagen, dann massive Continuation verpasst |
| CACI_2026-04-23 | REVERSAL | -9.28% | Stop Loss D+9 | 81.2 | Technology | Identisch zu CACI_2026-04-19 - REVERSAL #2 in Downgrade-Phase, kein Lerneffekt durch System |
| BTU_2026-05-01 | REVERSAL | -8.73% | Stop Loss D+3 | 82.2 | Energy | REVERSAL 4 Tage vor Q1 Earnings - Earnings am 5.5 zeigten Net Loss & EBITDA -43%, Production Cut |
| CACI_2026-05-01 | REVERSAL | -8.03% | Stop Loss D+3 | 72.6 | Technology | REVERSAL #3 fuer CACI in 12 Tagen - Phase E DUPLICATE_WINDOW (3d) hat nicht ausgereicht, weil wir alle 10-15 Tage neu signalisiert haben |
| AMTM_2026-04-01 | REVERSAL | -8.01% | Stop Loss D+23 | 59.7 | Industrials | DEAD_CAT_BOUNCE - Stock war -20% YTD, REVERSAL kaufte trotz fundamentaler Schwaeche; nahm 23 Tage zum Stop |
| LKQ_2026-05-06 | REVERSAL | -7.61% | Stop Loss D+5 | 91.3 | Unknown | Fundamentale Margin-Compression + Securities-Lawsuit = REVERSAL traf strukturell geschwaechten Stock |
| PENN_2026-04-27 | BREAKOUT | -7.45% | Stop Loss D+10 | 110.2 | Consumer Cyclical | Post-Earnings-Spike-Chase - Q1 Earnings 23.4 mit 120% Surprise, Stock +16.7% pre-market; wir fired BREAKOUT 4 Tage spaeter = klassischer Late-Entry-Fade |
| MOS_2026-04-19 | REVERSAL | -7.09% | Stop Loss D+7 | 76.3 | Basic Materials | REVERSAL 10 Tage nach Brazilian-Phosphate-Idle-Announcement (9.4) + multiple Analyst-PT-Cuts - fundamental schwaechender Stock |
| GEV_2026-03-23 | BREAKOUT | -6.56% | Stop Loss D+5 | 89.0 | Industrials | BREAKOUT EXAKT am S&P-100-Inclusion-Tag (23.3) = index-buying war bereits gepriced, klassische 'sell the news' Fade |
| XRAY_2026-05-01 | REVERSAL | -6.49% | Stop Loss D+2 | 65.2 | Healthcare | _pending_ |
| UI_2026-03-18 | BREAKOUT | -6.27% | Stop Loss D+8 | 86.2 | Technology | _pending_ |
| FLS_2026-04-27 | BREAKOUT | -6.18% | Stop Loss D+3 | 110.9 | Industrials | _pending_ |
| HP_2026-03-19 | BREAKOUT | -5.98% | Stop Loss D+9 | 82.6 | Energy | _pending_ |
| CNH_2026-03-26 | REVERSAL | -5.91% | Stop Loss D+20 | 71.5 | Industrials | _pending_ |

## Best 10 Trades

| Trade | Setup | PnL% | Exit | Score | Sektor | Analyse |
|---|---|---|---|---|---|---|
| ENPH_2026-04-15 | REVERSAL | 31.6% | Take Profit D+20 | 60.9 | Technology | _pending_ |
| ENPH_2026-04-16 | REVERSAL | 31.6% | Take Profit D+19 | 61.1 | Technology | _pending_ |
| BLD_2026-04-01 | REVERSAL | 25.15% | Take Profit D+12 | 57.2 | Industrials | _pending_ |
| SCCO_2026-03-23 | REVERSAL | 19.55% | Take Profit D+13 | 67.2 | Basic Materials | _pending_ |
| NEM_2026-03-23 | REVERSAL | 17.99% | Take Profit D+11 | 64.4 | Basic Materials | _pending_ |
| RVTY_2026-05-04 | REVERSAL | 17.6% | Take Profit D+5 | 69.0 | Unknown | _pending_ |
| STX_2026-04-16 | BREAKOUT | 16.76% | Take Profit D+9 | 112.0 | Technology | _pending_ |
| CIEN_2026-03-18 | BREAKOUT | 16.64% | Take Profit D+5 | 105.5 | Technology | _pending_ |
| MUR_2026-03-19 | BREAKOUT | 12.43% | Take Profit D+12 | 84.2 | Energy | _pending_ |
| AKAM_2026-04-16 | REVERSAL | 12.01% | Take Profit D+13 | 71.0 | Technology | _pending_ |

## Pending Claude-Analyse (86)

Diese Trades warten auf Claude WebSearch-Verifikation + Lesson-Tagging:

- `XOM_2026-03-18`
- `CIEN_2026-03-18`
- `DVN_2026-03-17`
- `MUR_2026-03-17`
- `CTRA_2026-03-17`
- `CTRA_2026-03-18`
- `MUR_2026-03-18`
- `COP_2026-03-18`
- `MNST_2026-03-22`
- `PFE_2026-03-17`
- `DOW_2026-03-18`
- `VLO_2026-03-18`
- `UI_2026-03-18`
- `EXE_2026-03-19`
- `HAL_2026-03-19`
- `MUR_2026-03-19`
- `HP_2026-03-19`
- `EQT_2026-03-19`
- `PFE_2026-03-19`
- `SEE_2026-03-19`
- _(+ 66 mehr)_
