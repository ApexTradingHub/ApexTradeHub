# ApexNext — Backlog (zurückgestellte Ideen)

Zurückgestellte Features/Ideen, die NICHT vergessen werden sollen. Jede mit genug
Kontext, um sie kalt (ohne Vorwissen) aufzugreifen.

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
