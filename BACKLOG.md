# ApexNext — Backlog (zurückgestellte Ideen)

Zurückgestellte Features/Ideen, die NICHT vergessen werden sollen. Jede mit genug
Kontext, um sie kalt (ohne Vorwissen) aufzugreifen.

---

## 3. Sektor-Cap im Trader (max 2 pro Sektor) — zurückgestellt 2026-06-12

**Motivation:** AFRM (Fintech) + IBKR (Financial Services) gleichzeitig im Portfolio
beim Macro-Schock 5.6.2026 = beide gestoppt, identische Loss-Cluster. Postmortem-Lesson
`macro_selloff_correlates_all_stocks` (siehe knowledge/trade_postmortems.json).

**Idee aus CLAUDE_CODE_TRADER_HYBRID.md (User-Brief):** Max 2 Positionen pro Sektor,
gilt für Scanner-Signale UND Momentum-Filler zusammen. Wird im `trigger_pending`-Schritt
geprüft, bevor `open_position` gerufen wird.

**Vorgeschlagene Implementierung:**
```python
def sector_count(ticker, open_positions, sector_cache):
    sec = sector_cache.get(ticker, "Unknown")
    return sum(1 for p in open_positions if sector_cache.get(p["ticker"], "Unknown") == sec)

# in trigger_pending vor open_position:
if sector_count(p["ticker"], state["open"], sector_cache) >= 2:
    still_pending.append(p)  # nicht expirien, evtl. spaeter Slot
    continue
```

**Warum zurückgestellt:** User-Entscheidung 2026-06-12 — erst Hybrid-Trader-Test ohne Cap
laufen lassen, dann ggf. nachziehen wenn Cluster-Verluste sich wiederholen.

**Trigger zum Anpacken:** ein zweites Loss-Cluster (mind. 2 Positions desselben Sektors
in <3 Tagen gestoppt). Oder User-Wunsch.

---

## 1. Trigger-Status für offene Signale ("Pending" vs "Offen") — zurückgestellt 2026-05-29

**Problem:** Im Dashboard werden frische Signale pauschal als "offen" angezeigt, obwohl
sie noch nicht getriggert haben (buy_above noch nicht erreicht). Die "offen"-Logik ist
nur eine Approximation (`nicht-geschlossen + im Hold-Fenster`), kein echter Trigger-Check.

**Gewünschtes Verhalten:** Neues Signal steht auf "⏳ Pending / nicht getriggert", bis der
Trigger (`high >= buy_above`) erreicht wird — meist beim nächsten US-Open — und springt
dann mit dem **nächsten Equity-Lauf** auf "✅ Offen / getriggert".

**Designvorschlag (analysiert, machbar, mittlerer Aufwand):**
- `apex_equity.py` schreibt neue Datei `apex_open_positions.json` mit jedem offenen Signal
  + Flag `triggered: true/false` (= `high seit Signal-Datum >= buy_above`, innerhalb
  `MAX_TRIGGER_DAYS=3`). Die Trigger-Maschinerie existiert dort schon (nur Output fehlt).
- `dashboard.html` liest das File → zeigt "⏳ Pending" vs "✅ Offen" statt pauschal "offen".
- Warum nur Equity-Tracker das kann: Dashboard ist statisch (kein Kurs-Feed im Browser);
  nur der Equity-Lauf hat yfinance-Kurse.

**Sonderfälle entfallen (vereinfacht es):**
- "US-Börse offen beim Signal" → passiert nie: Scanner läuft 20:30 UTC, nach US-Close (20:00).
- "Entry = Kurs / BUY now" → alle aktiven Setups nutzen "BUY above X" (REVERSAL/BUY-now disabled).
- → Eine simple Regel reicht: `triggered = (high seit Signal-Datum >= buy_above)`.

**Risiko:** fasst Produktions-Equity-Tracker an → vorsichtig + Sanity-Check umsetzen.

---

## 2. MOMO-Setup (Vertical-Momentum-Catch) — zurückgestellt 2026-05-29

**Problem / Motivation:** Unsere bestehenden Setups (BREAKOUT, VCP, STAGE_2, MR) sind alle
Trend-Continuation- oder Mean-Reversion-Plays mit **engen Filtern für Volatilität**. Sie
**filtern parabolische Vertical-Moves explizit raus** — und zwar by design, weil die
Postmortems zeigen dass die meisten Vertical-Chases verlieren (WDC -9.4 %, PENN -7.4 %,
APH -7.8 %, etc.).

**Konkretes Beispiel das wir verpasst haben:** DELL +188 % in 60 Tagen (März-Ende Mai 2026),
allein am 29.5. ein +33 %-Gap. Filter-Forensik zeigte: RSI **85.1** (>>68-cap), base_range
**44.2 %** (>>22-cap), close +164 % über MA150. Mehrere strict-Gates ausgehebelt. **Richtig
geblockt statistisch** — aber die seltene Ausnahme, die runs, schmerzt.

**Idee:** Ein eigenständiges **MOMO-Setup** mit explizit anderem Filter-Profil, gebaut für
parabolische Verticals — nicht statt der bestehenden Setups, sondern als zusätzlicher Kanal.

**Designskizze (NICHT validiert, Ausgangshypothese):**
- **Trigger:** vertikaler Spike — z.B. perf_5d > 15 %, perf_20d > 30 %, oder Gap-up > 10 %
- **RSI:** **70+ erlaubt** (Gegenpol zu BREAKOUT's 45-68); keine Obergrenze, da Momentum
- **Stops:** **breit** (z.B. 12-18 % statt 3-10 %), weil Vola explodiert
- **Target/Horizont:** **kurz** (1-2 Wochen), TP klein (+8-12 %), oder Trail-Stop
- **base_range:** komplett ignorieren (vertical = breit per definition)
- **Vol:** vol_ratio >= 2.0 (echte Momentum-Bestätigung)
- **Catalyst:** Pivot oder Vol-Climax als Bonus, kein Hard-Filter

**Erwartete Profile-WR/PF:** typisch 35-45 % WR, hohe PF (~2.5+) wenn die wenigen Winner
laufen, brutaler durch viele kleine Verlierer. **Asymmetrisches Outcome**, nicht
Trefferquoten-Setup.

**Validierungsweg (wenn umgesetzt):**
1. Detector in `apex_backtest_v2.py` bauen
2. `--only-setup MOMO` Test (Infrastruktur steht von früheren Tuning-Experimenten)
3. Akzeptanz: WR >= 35 %, PF >= 2.0 bei n >= 30 über 2 Jahre, sonst verworfen
4. Live-Port wie üblich nur bei bestandenem Gate

**Wichtige Caveats:**
- **Nicht zu früh bauen** — ist explizit ein *zusätzlicher* Kanal, der gegen die WR-Disziplin
  der anderen Setups spielt (Trefferquote tief, einzelne Trades dominieren)
- **Risiko-Diziplin Pflicht:** ohne harte Stops killt es das Konto bei Reversals
- **Cross-Ref Postmortem-Lessons:** `vertical_move_breakout_fail` (WDC), `extended_stock_breakout`,
  `high_volatility_tight_stop` — alles Hinweise dass das Setup eine andere Stop-Philosophie braucht

**Trigger zum Anpacken:** wenn der User explizit Vertical-Catches will, oder DELL-artige Misses
sich häufen.

**STATUS 2026-05-30 — v0 GETESTET & VERWORFEN:**
- Detector implementiert (apex_backtest_v2.py, opt-in via `--only-setup MOMO`)
- Baseline 2J: n=72, WR 51.4 %, PF 1.51 — **PF-Gate (≥2.0) verfehlt**
- Diagnose: +12 %-TP cappt die DELL-artigen Verticals (gemeint waren +25-50 %), Top-Winner
  enden alle bei exakt +11.78 %. Asymmetrisches Profil wurde durch zu engen TP ausgehebelt.
- Regime-Sensitivität real: 2025H2 PF 0.67 (verlustbringend), 2026YTD PF 3.62 (sehr stark).
- Verworfen statt v1-Iteration (Overfitting-Risiko auf demselben Sample).
- Code bleibt als opt-in Research-Tool in apex_backtest_v2.py (greift nur bei
  `--only-setup MOMO`, sonst inaktiv).

---

## 4. Macro-VIX/HY-Gate für BREAKOUT — getestet & FALSIFIED 2026-06-14

**Hypothese:** BREAKOUT-WR fällt bei VIX≥22 oder HY-OAS≥4.0 zum Entry-Zeitpunkt.
Motivation: AFRM/IBKR-Postmortems mit Lesson `macro_selloff_correlates_all_stocks`.

**Setup:** `apex_macro.py` zieht FRED-Series (VIXCLS, BAMLH0A0HYM2, T10Y2Y, DFF, DTB3) →
`apex_macro.json`. `apex_macro_backtest.py` joint 143 Lifetime-Trades aus
`apex_equity_results.json` mit VIX + HY-OAS am Entry-Datum.

**Ergebnis: Hypothese widerlegt, sogar gegenteilig.**

BREAKOUT-WR nach Combined Regime (n=85):
| Regime | n | WR | vs Baseline 56.5 % |
|---|---:|---:|---:|
| RISK_ON  | 54 | 55.6 % | -0.9pp |
| ELEVATED | 13 | 53.8 % | -2.7pp |
| **RISK_OFF** | 18 | **61.1 %** | **+4.6pp** |

VIX 25+ hat das **stärkste WR-Bucket** (61.1 %, n=18). Ein VIX-Gate würde genau die
besten Signale killen → verletzt Signal-Protection-Regel.

**AFRM/IBKR-Recency-Bias:** Aus 2 schmerzhaften Trades hatten wir Pattern induziert,
das im n=18 RISK_OFF-Bucket Noise ist (61 % WR insgesamt).

**Sample-Caveat:** Nur 3 Monate (März-Juni 2026), VIX 20-22-Bucket leer, bimodale
Verteilung. Für sauberen Re-Test 2+ Jahre Daten nötig (haben wir nicht).

**Was bleibt:**
- `apex_macro.py` als Situational-Awareness-Tool aktiv (daily snapshot, kein Live-Filter)
- `apex_macro_backtest.py` als Re-Test-Tool bei ~300+ Lifetime-Trades wieder anwerfen
- Side-Finding: REVERSAL × HY 3.0-3.5 = 53.3 % WR n=15 (vs Setup-Baseline 30.4 %),
  TENTATIVE — siehe Punkt 5

**Trigger zum Re-Test:** Trade-DB ≥ 300, oder echter Crash mit >50 Trades im VIX>30-Bereich.

---

## 5. REVERSAL-Reaktivierung in Stress-Regime — Watch (nicht handeln) 2026-06-14

**Befund aus Macro-Backtest:**
- REVERSAL × HY 3.0-3.5 (Normal-Stress): WR **53.3 %** (n=15, MED) vs Setup-Baseline 30.4 % (+23pp)
- REVERSAL × VIX 25+ (Panic): WR **53.8 %** (n=13, LOW) vs Setup-Baseline 30.4 % (+23pp)

**Warum trotzdem nicht reaktivieren (yet):**
- n=13-15 ist TENTATIVE — User-Regel: nur Code-Changes bei n≥30 + CONFIRMED
- REVERSAL ist strukturell defekt (Postmortems: fundamental-driven drops, Earnings-Miss,
  Analyst-Downgrade-Cascade, Insider-Distribution) — gleicher Mechanismus auch in Stress
- Lift kommt vermutlich aus dem Mean-Reversion-Floor nach Panik, nicht aus reparierter
  Setup-Logik

**Trigger zum Anpacken:** n≥30 in REVERSAL×HY 3.0-3.5 ODER User entscheidet:
"reaktivieren nur im stress regime, Hard-Gate `hy_oas ≥ 3.0`". Wäre **machbar** als
opt-in: REVERSAL bleibt im Scanner aktiv, aber Telegram-Gate filtert wenn HY < 3.0.
Risiko: Cherry-Pick auf kleiner Sample, MR-Floor verschwindet wenn Stress sich auflöst.

**Realismus-Score:** 3/10 für Reaktivierung in den nächsten 6 Monaten. Setup-Tod
ist fundamental, nicht regime-bedingt.
