# ApexMacro Backtest — 2026-06-14 17:25

**Joined 143 of 143 trades** with FRED VIX + HY-OAS at entry date.

**Hypothese:** BREAKOUT-WR fällt bei VIX≥22 oder HY-OAS≥4.0 (macro selloff correlates).
**Akzeptanz:** n≥30 für CONFIRMED, ≥15 für TENTATIVE. Lift ≥10pp WR vs baseline = signal.

---

## Baseline (alle Setups)

- n=143 | WR 45.5% | PF 1.81 | avg +2.11% | [HIGH]

## BREAKOUT — Baseline: n=85 | WR 56.5% | PF 2.43 | avg +2.87% | [HIGH]

### VIX-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| VIX <16 (Quiet) | 5 | 80.0% | 4.39 | +5.05% | LOW |
| VIX 16-20 (Normal) | 49 | 53.1% | 2.54 | +3.21% | HIGH |
| VIX 20-22 (Elevated) | 0 | — | — | — | — |
| VIX 22-25 (Stress) | 13 | 53.8% | 2.45 | +2.50% | LOW |
| VIX 25+ (Panic) | 18 | 61.1% | 1.74 | +1.60% | MED |

### HY-OAS-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| HY <3.0 (Tight) | 54 | 55.6% | 2.67 | +3.38% | HIGH |
| HY 3.0-3.5 (Normal) | 31 | 58.1% | 2.00 | +1.98% | HIGH |
| HY 3.5-4.0 (Elevated) | 0 | — | — | — | — |
| HY 4.0-5.0 (Stress) | 0 | — | — | — | — |
| HY 5.0+ (Crisis) | 0 | — | — | — | — |

## MEAN_REVERSION — Baseline: n=2 | WR 0.0% | PF 0.00 | avg -2.59% | [LOW]

### VIX-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| VIX <16 (Quiet) | 1 | 0.0% | 0.00 | -2.81% | LOW |
| VIX 16-20 (Normal) | 1 | 0.0% | 0.00 | -2.38% | LOW |
| VIX 20-22 (Elevated) | 0 | — | — | — | — |
| VIX 22-25 (Stress) | 0 | — | — | — | — |
| VIX 25+ (Panic) | 0 | — | — | — | — |

### HY-OAS-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| HY <3.0 (Tight) | 2 | 0.0% | 0.00 | -2.59% | LOW |
| HY 3.0-3.5 (Normal) | 0 | — | — | — | — |
| HY 3.5-4.0 (Elevated) | 0 | — | — | — | — |
| HY 4.0-5.0 (Stress) | 0 | — | — | — | — |
| HY 5.0+ (Crisis) | 0 | — | — | — | — |

## REVERSAL — Baseline: n=56 | WR 30.4% | PF 1.32 | avg +1.12% | [HIGH]

### VIX-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| VIX <16 (Quiet) | 0 | — | — | — | — |
| VIX 16-20 (Normal) | 41 | 22.0% | 0.92 | -0.32% | HIGH |
| VIX 20-22 (Elevated) | 0 | — | — | — | — |
| VIX 22-25 (Stress) | 2 | 50.0% | 3.14 | +8.57% | LOW |
| VIX 25+ (Panic) | 13 | 53.8% | 3.40 | +4.52% | LOW |

### HY-OAS-Bucket

| Bucket | n | WR | PF | AvgPnL | Conf |
|---|---:|---:|---:|---:|---|
| HY <3.0 (Tight) | 41 | 22.0% | 0.92 | -0.32% | HIGH |
| HY 3.0-3.5 (Normal) | 15 | 53.3% | 3.34 | +5.06% | MED |
| HY 3.5-4.0 (Elevated) | 0 | — | — | — | — |
| HY 4.0-5.0 (Stress) | 0 | — | — | — | — |
| HY 5.0+ (Crisis) | 0 | — | — | — | — |

## Combined Macro-Regime

Regime-Definition: RISK_OFF wenn VIX≥25 ODER HY≥5.0 | ELEVATED wenn VIX≥20 ODER HY≥3.5 | sonst RISK_ON

| Regime | All Setups | BREAKOUT only |
|---|---|---|
| **RISK_ON** | n=97 | WR 40.2% | PF 1.59 | avg +1.69% | [HIGH] | n=54 | WR 55.6% | PF 2.67 | avg +3.38% | [HIGH] |
| **ELEVATED** | n=15 | WR 53.3% | PF 2.63 | avg +3.31% | [MED] | n=13 | WR 53.8% | PF 2.45 | avg +2.50% | [LOW] |
| **RISK_OFF** | n=31 | WR 58.1% | PF 2.38 | avg +2.82% | [HIGH] | n=18 | WR 61.1% | PF 1.74 | avg +1.60% | [MED] |

---

## Interpretation

- WR-Lift ≥10pp + n≥30 in einem Bucket → potentieller Score-Penalty- oder Gate-Kandidat
- Niedrige n (<15) = nicht aussagekräftig, nur als Trend
- Vor Live-Integration: in apex_backtest_v2.py mit `--only-setup BREAKOUT --exclude-regime RISK_OFF` re-validieren