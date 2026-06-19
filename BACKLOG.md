# ApexNext — Backlog (zurückgestellte Ideen)

Zurückgestellte Features/Ideen, die NICHT vergessen werden sollen. Jede mit genug
Kontext, um sie kalt (ohne Vorwissen) aufzugreifen.

---

## 8. Duplicate-Trap: DUPLICATE_WINDOW_DAYS=3 zu kurz — beobachtet 2026-06-17

**Befund aus Postmortem-Vollanalyse (147/147 analyzed):** `duplicate_trap` ist mit
**n=15** das auffälligste neue Lesson-Tag. Dasselbe Signal wurde mehrfach innerhalb
weniger Tage emittiert: SEE ×3 (19./22./23.3), AKAM ×2, TPL ×2, BWA ×2, HII ×2,
PFE ×2, CBT ×2, ENPH ×2.

**Problem:** `DUPLICATE_WINDOW_DAYS = 3` (ApexScan.py L45) filtert nur Re-Emissionen
innerhalb 3 Tagen. Re-Signale alle 10-20 Tage rutschen durch. Im Equity-Tracker zählen
sie als separate Trades (verzerrt n + WR), im Live-Paper-Trader hätten wir mehrfach
Kapital auf denselben Trade allokiert (SEE = 3 Slots für denselben flachen Trade).

**Mögliche Fixes (NICHT umgesetzt — User: keine Action 2026-06-17):**
- DUPLICATE_WINDOW von 3d auf ~10d erhöhen (einfachster Fix)
- Same-Ticker-within-Hold-Window-Block im Paper-Trader (sauberer, da hold-aware)
- Im Equity-Tracker: Duplikate dedupen vor WR-Berechnung

**Caveat (Signal-Protection-Regel):** Window-Erhöhung senkt Signal-Count. Vor Umsetzung
prüfen wie viele LEGITIME Re-Entries (Trade geschlossen, sauberes Neu-Signal) dabei
mitgefiltert würden. Nicht blind erhöhen.

**Trigger zum Anpacken:** vor Go-Live (verzerrt Live-Kapital-Allokation) ODER beim
nächsten Equity-Refactor. Aktuell nur beobachtet.

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

---

## 6. evaluate_trade Same-Day-Trigger-Stop-Ambiguität — bug found 2026-06-15

**Problem:** `apex_equity.py::evaluate_trade` gibt `None` zurück (= Trade wird nicht
simuliert, Signal bleibt forever-open) wenn am Trigger-Tag SOWOHL `high >= entry`
ALS AUCH `low <= stop` zutrifft. Konservative Sicherheit — Intraday-Reihenfolge
nicht bekannt — aber zu aggressiv.

**Aufgedeckt durch:** CDNS 2026-04-27 BREAKOUT. D+1 (2026-04-28):
- Open $329.70 (< Entry $335.48 → kein Trigger am Open)
- Low $317.07 < Stop $317.38 ✓
- High $338.55 ≥ Entry $335.48 ✓ (Trigger)
- Close $325.31

Realistische Intraday-Sequenz: Open → Drop zu Low (kein Trigger, noch nicht in Trade)
→ Up zu High (Trigger fires bei $335.48) → Close. Hätte funktioniert, Stop wurde nie
gehittet weil wir nicht im Trade waren als Low erreicht wurde.

**Manuell gefixt 2026-06-15:** CDNS in `apex_equity_results.json` als TP-Hit
nachgetragen (D+19, $370, +10.29%, +$20.58). Equity-Cumulativ recomputed.

**Vorschlag-Fix (nicht implementiert):**
```python
# Heuristik: wenn Open >= entry, trigger und stop sind klar separable.
# Wenn Open < entry: nehmen wir an Drop kam zuerst (Open->Low->High).
# In dem Fall kein Stop weil nicht im Trade beim Low.
if h >= entry:
    if o >= entry:
        # Open ueber entry: getriggert sofort am Open
        # Bei Low <= sl im gleichen Tag -> Stop loss
        if l <= sl: exit ... # gleicher Tag
    else:
        # Open unter entry: Drop wahrscheinlich zuerst, Trigger danach
        # Stop wird nicht gehittet weil nicht im Trade
        trigger_day = i  # weiter im Loop
```

**Trigger zum Anpacken:** weitere forever-open-Signale auftauchen (User-Frage). Oder
beim nächsten Equity-Refactor mitnehmen.

---

## 7. HORIZON_DAYS-Konflikt: Code 15 vs Doku 21 für BREAKOUT — discovered 2026-06-15

**Problem:** Drei Quellen-of-Truth für BREAKOUT-Hold widersprechen sich:

| Quelle | BREAKOUT hold | Aktuell |
|---|---|---|
| `apex_equity.py` HORIZON_DAYS["1-3 weeks"] | **15** | Code-Wahrheit |
| CONTEXT.md Konsistenz-Konstanten | **21** | dokumentierte Intention |
| `apex_trader.py` HOLD_DAYS_PER_SETUP.BREAKOUT | **21** | Paper-Trader |
| Dashboard SETUP_META.BREAKOUT.hold | 21 (seit 2026-06-04) | Frontend |

**Konsequenz:** Die gesamte `apex_equity_results.json` (147 Trades) wurde mit
15-Tage-Hold für BREAKOUT simuliert. Time-Exits feuern 6 Tage früher als
beabsichtigt. Realer Edge ist möglicherweise unterschätzt — siehe CDNS, hätte
mit 21-Tage-Hold organisch TP gehittet.

**Optionen:**
1. **Code anpassen:** `HORIZON_DAYS["1-3 weeks"] = 21` → konsistent mit Trader/Backtest.
   Alle Lifetime-Stats werden besser (mehr TP-Hits, weniger Time-Exits).
2. **Doku anpassen:** CONTEXT.md ehrlich auf 15 setzen, Paper-Trader nachziehen.
   System wird strenger.

**Empfehlung:** Option 1. Aber **Backtest-First!** — `apex_backtest_v2.py` mit beiden
Werten laufen lassen (15 vs 21 für BREAKOUT) über 2J, sehen welcher Wert besser
performt. Falls 21 deutlich besser → Code anpassen + Knowledge-Refresh + Equity-
Recompute (alle BREAKOUTs aus Lifetime mit 21 neu evaluieren).

**Risiko:** Im Code-Pfad steckt evtl. mehr Logik die auf 15 Tage hängt
(Backtest, Knowledge-Eval, etc.) — vor Live-Wechsel sauber tracen.

**Trigger zum Anpacken:** vor nächstem Tuning-Backtest. Oder wenn weitere Trades
in der "TP-knapp-verpasst"-Zone auftauchen.

---

## 9. Win-Magnitude zu klein — Gewinner laufen lassen — entdeckt 2026-06-18

**Befund aus Paper-Trader-Analyse (11 closed, WR 72.7%):**
- **Avg-Winner +3.34% < Avg-Loser -4.13%** — unser durchschnittlicher Gewinner ist
  KLEINER als unser durchschnittlicher Verlierer.
- **Kein einziger BREAKOUT erreichte den vollen TP (~+10%)** — alle exiteten via
  Trailing-Ladder (+2-6%) oder Stagnation (+2%), bevor sie die +8.5% Lifetime-AvgWin
  erreichten. Beispiele: CCL war +7.5%, raus bei +6%. Stagnation-Exits AXTA/MOH/BAX/CARR
  alle bei +1.6-1.9%.
- **Fragilität:** PF (~2.16) wird KOMPLETT von der hohen WR getragen. Bei n=11 regресst
  72% → ~58% (Lifetime). Mit diesen Magnituden faellt PF dann auf ~1.1 = kaum profitabel.

**Root-Cause:** Exit-Mechanik (Stagnation ±2% nach 5d + enge Trailing-Ladder 1.06→1.02)
ist auf Kapitalerhalt getrimmt, erstickt aber die +10%-Winner die im Backtest die PF tragen.
System ist exzellenter Verlust-Vermeider, mittelmaessiger Gewinn-Maximierer.

**Fix-Optionen (NICHT umgesetzt, Backtest-First):**
1. **Partial-Profit-Taking:** halbe Position bei +6% (Trailing-Step-2), Rest mit weiterem
   Stop laufen lassen Richtung TP. Sichert + laesst Runner laufen.
2. **Trailing-Ladder weiter:** Steps lockern (z.B. 1.08→1.03, 1.14→1.08, 1.20→1.14) damit
   normale Pullbacks nicht bei +2% rausstoppen.
3. **Stagnation-Schwelle anheben:** statt ±2% nach 5d → erst nach 8d oder bei engerem Band.

**Akzeptanz (vorher fixieren):** AvgWin muss ueber AvgLoss steigen OHNE dass WR signifikant
faellt UND ohne dass Max-Drawdown steigt. Gegen aktuelle Logik in apex_backtest_v2.py.
Ein-Knopf-Pro-Lauf (nicht alle 3 Optionen buendeln).

**Trigger:** naechster Tuning-Tag. Groesster Einzelhebel fuer Equity-Wachstum.

---

## 10. Rotation / Replacement-Logik miskalibriert — entdeckt 2026-06-18

**Befund:** Die Replacement-Logik EXISTIERT (`is_replacement_eligible` in apex_trader.py,
seit 2026-06-07), aber **feuert praktisch nie** wegen einer rueckwaerts-kalibrierten
Bedingung.

**Aktuelle Gates (ALLE noetig):**
1. Alle 7 Slots voll
2. Neues Signal Score ≥ 90
3. Neues Signal hat Pocket Pivot ODER Gap ≥ 2%
4. **Schwaechste offene Position ≥ +2%** ← DER FEHLER

**Warum #4 falsch ist:** Die schwaechste Position (= min pnl) muss ≥ +2% sein, damit
getauscht wird. Aber es gibt fast IMMER eine Position die flach oder leicht rot ist
(aktuelles Buch: DAL -1.9%, ASML -2.1%, SW -1.0%, RL -0.9%). → Replacement feuert NUR
wenn das GANZE Buch ≥ +2% gruen ist — also genau dann wenn man NICHT rotieren will.
Wenn man am meisten rotieren WILL (ein -2% Laggard blockiert einen Slot waehrend ein
Elite-PP-Signal kommt), weigert sich die Logik weil der Laggard im Minus ist.

**Das ist die Antwort auf "kann der Trader Positionen austauschen":** Ja, der Code ist da,
aber er schuetzt Laggards statt sie zu ersetzen. Rotation = totes/schwaches Geld schneiden
und in frische Momentum-Signale umschichten — genau das verhindert #4.

**Fix-Optionen (NICHT umgesetzt, Backtest-First):**
1. **#4 umdrehen:** Replace erlauben wenn schwaechste Position UNTER einer Schwelle ist
   (z.B. flach-bis-leicht-negativ, < +1% UND ≥ 3d gehalten ohne Progress) = dead money
   schneiden, nicht Winner stoeren.
2. **Score-Delta-Gate:** Neues Signal muss Original-Score der schwaechsten Position um
   Margin schlagen (z.B. +20 Punkte) = nur nach OBEN tauschen, kein Churn.
3. **Loss-Toleranz:** kleine realisierte Verluste (z.B. bis -3%) fuer klar bessere Signale
   erlauben — ABER in echt (eToro) Spreads/Fees bedenken, sonst Churn-Bleed.

**Spannung mit Disziplin:** Aggressive Rotation = Overtrading-Risiko + "grass is greener"-
Falle (guten Dip verkaufen fuer neuen Namen der dann failt). Quality-Gate (PP/Gap + Score)
ist gut, nur #4 ist kaputt. Vorsichtig + Backtest.

**Verbindung zu #9:** Bessere Rotation = weniger Dead-Money-Drag + mehr Throughput +
schnellerer Redeploy in frische Winner. #9 (Winner laufen lassen) + #10 (Laggards schneller
schneiden) zusammen = Magnitude-UND-Frequenz-Hebel.

**Trigger:** zusammen mit #9 im naechsten Trader-Tuning-Tag.
