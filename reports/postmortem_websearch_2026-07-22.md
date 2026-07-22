# Postmortem-WebSearch — 9 Pending-Trades (2026-07-22)

Recherchiert via WebSearch (MCP-News-Tool auf diesem FMP-Plan gesperrt). Zum Nachtragen
in `knowledge/trade_postmortems.json` sobald die Pipeline die Trades einspeist
(`news.web_research` + `claude_analysis.lesson_tags`).

## Verlierer (trieben die -$38 eToro-Realverlust)

**ANET_2026-07-08** | BREAKOUT score 137.0 | Stop Loss D+7 -8.57%
- Insider-Verkäufe **$584,6M in 3 Monaten** ohne einen einzigen Kauf. Cloud-Titan-
  Konzentration (Microsoft/Meta) → jede Capex-Guidance-Änderung drückt sofort. Networking-
  Hardware-De-Risking sektorweit. Hoher Score sah die Distribution NICHT.
- tags: `high_score_loss_85plus`, `insider_distribution`, `customer_concentration_risk`

**TRIP_2026-07-16** | BREAKOUT score 114.8 (SHORT-Flag!) | Stop Loss D+3 -7.08%
- BTIG-Downgrade Buy→Neutral (AI-Exposure-Risiko). **Short-Interest 29,45% = meistgeshortete
  Communications-Aktie**. Strukturell kaputt: -65% über 5 Jahre, TheFork-Verkauf entfernte
  den Wachstumsmotor. Das SHORT-Flag war im Signal PRÄSENT und wurde ignoriert.
- tags: `short_gt_15pct_confirmed`, `analyst_downgrade`, `structural_decline`
- **→ direkte Evidenz für WR-Hebel B (SHORT-Gate)**

**GS_2026-07-14** | BREAKOUT score 123.3 | Stop Loss D+3 -4.62%
- KEIN fundamentales Versagen: GS meldete Rekord-Q2 (EPS $20.98, Rev +39,5%), Aktie **+9%
  am 14.07.**. Wir kauften den Post-Earnings-Spike am 15.07. (Ask 1143,80) und wurden auf
  dem normalen Pullback bei 1079 ausgestoppt. Whipsaw.
- tags: `post_earnings_spike_chase`, `stop_too_tight_post_earnings`

**CMG_2026-07-13** | BREAKOUT score 89.7 | Stop Loss D+3 -5.19%
- Echte Margin-Erosion: Operating-Margin 16,7%→12,9% (Beef-Preise, Lohn-Inflation, Steuer).
  Analyst-Downgrades, -4% YTD vs S&P +10%. Strukturell schwach in die Earnings.
- tags: `fundamental_deterioration`, `margin_compression`

## Time-Exit-Drifts (die "Wins" die die WR beschönigen)

**HOOD_2026-07-02** | BREAKOUT score 104.0 (GAP) | Time Exit D+15 -5.94% (VERLUST!)
- Früh-Juli-Breakout scheiterte, CEO Tenev verkaufte 375k Aktien ab 06.07., Consumer-Finance-
  Scrutiny (ABS-Bond), Fintech-Selloff 17.07. 15 Tage seitwärts → Time-Exit im Minus.
- tags: `breakout_no_follow_through`, `insider_distribution`

**ILMN_2026-07-02** | BREAKOUT score 141.1 | Time Exit D+15 +2.69%
**BAX_2026-07-02** | BREAKOUT score 132.4 | Time Exit D+15 +5.09%
- Beide langsame Grinder: Analyst-PT-Anhebungen, intakt, aber erreichten den TP nicht in
  15 Tagen. Der Hold-Limit deckelt langsame Winner (bekanntes Muster, BACKLOG #7).
- tags: `hold_limit_caps_slow_winner`, `high_score_slow_grind`

**PCAR_2026-07-06** | score 89.0 | Time Exit +0.85%  → flacher Drift
**LW_2026-07-07** | score 87.7 | Time Exit +0.81%  → flacher Drift
- tags: `flat_drift_time_exit`

## Muster über alle 9

1. **Hohe Scores schützen nicht** (ANET 137, ILMN 141, BAX 132) — im BEARISH-Regime driften
   oder stoppen sie. Bestätigt PICK_BAND/TG_SWEET_BAND (130+ nicht besser).
2. **Insider-Distribution** bei ANET + HOOD — kein Feature in der Pipeline (SEC Form 4).
3. **SHORT-Flag anti-prädiktiv** — TRIP bestätigt Learn (n=8, -14,5% WR).
4. **Post-Earnings-Spike-Chase** (GS) — Kauf des Gaps, Stop auf dem Pullback.
