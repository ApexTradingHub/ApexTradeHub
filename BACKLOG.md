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

**STATUS 2026-07-17: THESE WIDERLEGT bei n=61 — Punkt geschlossen.**
Der Befund stammte aus **n=11**. Nachgemessen am selben Trader mit **n=61**:

| | 06-18 (n=11) | **07-17 (n=61)** |
|---|---|---|
| AvgWin | +3.34% | **+4.01%** |
| AvgLoss | -4.13% | **-3.44%** |
| These "AvgWin < AvgLoss" | ja | **WIDERLEGT** |

- **Give-back nur +0.97pp/Winner** (Peak Ø +4.98% -> Exit Ø +4.01%). Der behauptete
  "groesste Einzelhebel" ist ein Rundungsfehler. **Option 2 (Ladder lockern) waere sogar
  die falsche Richtung** — siehe unten.
- **Fix-Option 1/2/3 alle obsolet.** Getestete Alternative (BE-Stufe `(1.04, 1.00)` vor
  Step 1, Retro-Sim auf echten Trades, Akzeptanz vorab fixiert): **NO-GO** — nur 4/32
  Swing-Trades betroffen, Netto-Delta +8.2pp < Bar +10pp, und davon waren +12.1pp ein
  Sim-Artefakt (PENG lief in der Sim auf den Initialstop -12.1%, real via eToro bei -4.7%
  geschlossen) -> real ~+0.8pp = nichts. Runner (PAY) blieben unbeschaedigt, aber es gab
  nichts zu gewinnen.

**WO der echte Befund sitzt** (dieselben Daten, nach Quelle aufgeschluesselt):

| Quelle | n | davon Peak>=+4% | endete rot | Quote |
|---|---:|---:|---:|---:|
| scanner | 20 | 9 | 1 | **11%** |
| momentum_filler | 12 | 9 | 3 | 33% |
| **intraday_rescued** | 14 | 5 | 3 | **60%** |
| intraday_momentum | 8 | 0 | 0 | — |

**Die Swing-TRAIL_LADDER funktioniert** (11% Ausfall bei scanner). Der Cluster sitzt im
**Rescue-Pfad** (rote EOD->SWING-Konvertierungen) — das ist **BACKLOG #16**, dort bereits
G3-Stop getestet+falsifiziert und seit 07-11 Shadow-Logging aktiv (range_pos_eod etc.,
wartet auf n>=15). Intradays tauchen gar nicht auf, weil ihr TP schon bei +5% greift —
eine BE-Stufe bei +4% waere dort per Konstruktion sinnlos.

**Lehre:** Der 06-18-Befund war Small-Sample-Rauschen (n=11), das sich als Struktur-Aussage
tarnte — und die vorgeschlagene Fix-Richtung war gegenlaeufig zur Realitaet. Erst messen
(n aktuell?), dann simulieren, dann bauen. Kein Code angefasst.

---

## 24. Backtest kennt unsere Exit-Mechanik NICHT — Blindstelle (2026-07-17)

**Befund (bei der #9-Pruefung aufgefallen):** `apex_backtest_v2.py` simuliert fuer BREAKOUT
einen **statischen** Stop:
```python
active_sl = rev_dynamic_sl if setup_type == "REVERSAL" else sl   # ~L1092
```
Keine `TRAIL_LADDER`, kein Stagnation-Exit, keine EOD-Konvertierung, kein Rescue-Pfad.
Der 2J-Backtest (WR 50.2%, PF 2.02, +454%) simuliert damit **einen Trader den wir nicht
haben** — er misst reine Signal-Qualitaet mit TP/SL/Time-Exit.

**Konsequenz — wichtig fuer die Interpretation aller bisherigen Backtests:**
- Fuer **Signal-Fragen** (SCORE_V2 #17, PICK_BAND, Sektor-RS-Gate #20, Extension #13) bleibt
  alles gueltig: beide Vergleichsseiten sind gleich betroffen, der Vergleich ist fair.
- Fuer **Exit-/Money-Management-Fragen** (#9, #10, Trailing, Stagnation, Rescue) ist der
  Backtest **blind**. "Backtest-First" ist dort nicht durchfuehrbar, ohne vorher die
  Exit-Mechanik nachzubauen — was 2026-07-17 der Grund war, auf Retro-Sim auszuweichen
  (n=32 Swings, mittlere Abweichung 2.55pp/Trade = nur Richtungsindikator).

**Wenn angepackt:** Trader-Exit-Logik als opt-in Flag in apex_backtest_v2.py (Muster:
`--score-v2`), damit der Default-Backtest vergleichbar bleibt. Aufwand mittel; Risiko:
Nachbau divergiert vom echten Trader -> die Konstanten (TRAIL_LADDER, STAGNATION_*) aus
apex_trader.py **importieren** statt kopieren.

**Trigger:** wenn eine Exit-/MM-Frage ernsthaft entschieden werden soll (#10 Rotation ist
der naechste Kandidat) — vorher ist der Nachbau Selbstzweck.

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

**STATUS 2026-07-17: NO-GO — der "Bug" ist ein Schutz. Punkt geschlossen (kein Code).**

**1. Die Diagnose war falsch.** Nicht Bedingung #4 blockiert — der Pfad dorthin ist dicht.
Zwei Stellen zaehlen unterschiedlich:

| Stelle | zaehlt | heute (6 open + 1 pending) | Folge |
|---|---|---|---|
| `select_new_signals` L1301 | open **+ pending** | 7 -> `free_slots=0` | neue Signale **sterben** (nie pending) |
| `is_replacement_eligible` L1384 | nur `len(open)` | 6 >= 7 = **False** | "slot frei" -> **#4 wird nie erreicht** |

Deadlock: fuer neue Signale ist das Buch voll, fuer die Rotation ist es leer. Und
`is_replacement_eligible` wird NUR aus `trigger_pending` gerufen — ein neues Signal kann
per Konstruktion **nie** ein Replacement ausloesen, egal wie gut es ist (JBHT score 138.9
starb so am 17.07.). Empirie: **0 Replacement-Events** in der gesamten Historie (53
pending_added, 43 open) — konsistent.

**2. Und es ist gut so.** Die Kernfrage ist nicht "feuert Rotation", sondern "haetten die
Verworfenen Geld gebracht". Der Equity-Tracker simuliert JEDES Signal — also messbar
(ab 06-04, Trader-Start):

| | n | WR | avg | PF | Summe |
|---|---:|---:|---:|---:|---:|
| vom Trader **genommen** | 24 | 54.2% | +1.43% | 1.69 | **+34.4pp** |
| mangels Slot **liegengelassen** | 38 | 36.8% | -0.34% | 0.89 | **-12.9pp** |

**Delta -1.77pp/Trade — die Slot-Knappheit hat uns geschuetzt.** Rotation haette in diesem
Zeitraum Geld verbrannt. Der Schaden hat eine Adresse:

| liegengelassen | n | WR | Summe |
|---|---:|---:|---:|
| im Band 90-120 | 22 | 36.4% | +1.0pp |
| **ueber 120** | 10 | 30.0% | **-14.8pp** |
| unter 90 | 6 | 50.0% | +0.9pp |

**Der komplette Verlust sitzt im 120+-Bucket** (ASML 144 -> -6.6%, TSM 137 -> -6.1%,
CGNX 127 -> -6.7%, LRCX 125 -> -9.2%, KLAC 118 -> -9.3% = die Semi-Selloff-Loser).
**Unabhaengige Bestaetigung von PICK_BAND/TG_SWEET_BAND** aus einer ganz anderen Richtung:
die Slot-Knappheit hat zufaellig genau das gefiltert, was das Band absichtlich filtert.
Nebenbefund: das Pick-Ranking funktioniert — die Genommenen schlagen die Liegengelassenen
um 18pp WR.

**Caveats (ehrlich):**
- **LRCX score 161 -> +14.87%** ist der eine grosse verpasste Gewinner. Bei n=38 kein Beweis.
- **Ein Regime** (Juni-Juli 2026, BEARISH, 30d-WR 34%). In einem BULLISH-Regime koennte die
  Knappheit teuer werden — dann waeren die Verworfenen die besseren Trades.
- Equity-Sim nutzt statischen Stop (#24), nicht die echte Trader-Mechanik.
- "liegengelassen" mischt zwei Gruende: schlechter gerankt (free_slots>0) und Buch voll
  (free_slots=0). Fuer die Richtung reicht es, fuer eine Praezisionsaussage nicht.

**Konsequenz:** Fix-Optionen 1/2/3 alle NICHT umsetzen. Die Zaehl-Inkonsistenz **nicht
"aufraeumen"** — sie ist aktuell der Schutz. Wer sie glattzieht, oeffnet die Rotation
genau in das 120+-Bucket hinein.

**Re-Test-Trigger:** Regime-Wechsel auf BULLISH (SPY+QQQ) **und** 30d-WR wieder >= 50%.
Dann dieselbe Messung wiederholen (Skript-Rezept: scratchpad/bl10_worth.py, Session 07-17)
— wenn "liegengelassen" dann POSITIV dreht, ist Rotation ein echter Hebel und der Deadlock
gehoert gefixt. Vorher waere es Geldverbrennen mit Extra-Schritten.

---

## 11. perf_120-Unification (80-90-Bucket / WEAK-Seite) — geparkt 2026-06-20

**Befund (Ultracode-Forensik):** Der BREAKOUT-Score ist bei perf_120 nicht-monoton — `perf_120`
verhaelt sich wie eine **umgekehrte U-Kurve** (WR ganzes 250d-Sample):
- perf_120 < 0 (WEAK): **41 %** 🔴
- perf_120 0-25 (DEADZONE): 53 %
- perf_120 25-50 (SWEET): 56 %
- perf_120 > 50 (POWER): **39 %** 🔴

**Schlecht an beiden Extremen, gut in der Mitte.** Der Score-Rebuild (2026-06-20) fixte das
RECHTE Extrem (POWER/extended, catalyst-gated -12). Das LINKE Extrem (WEAK, perf_120<0 =
Breakout im Abwaertstrend = Dead-Cat-Bounce, vgl. Postmortem `dead_cat_bounce` AMTM) ist offen.

**Warum 80-90 das Loch ist (42 % WR, schlechter als BEIDE Nachbarn):** Mischzone — sammelt die
schlechten Raender von WEAK + POWER, die durch starke Kurzfrist-Terme (vol_ratio, perf_60) auf
80-90 inflationiert werden. Gegenprobe: 80-90 ohne Deadzone-Signale = nur 29 % WR (die
Nicht-Deadzone-Signale sind das Problem). 90-100 ohne Deadzone = 72 %.

**Loesung B (elegant, empfohlen):** perf_120 als echte umgekehrte-U-Kurve statt 3 separater
Mechanismen (Realign-Buckets + Score-Rebuild-Penalty + WEAK -15). Eine kohaerente, kalibrierte
Kurve die BEIDE Extreme stufenlos straft, die Mitte belohnt. Encodiert die Replicate-Lehre
F_2026-05-22 direkt („EMERGING/Mitte = Sweet-Spot"). POWER-Carve-Out (Semi/AI-Capex) muss
erhalten bleiben.

**Warum geparkt (niedrige Prio):** Der Score zaehlt fuers Ranking (Telegram-Top-2, Trader-Picks)
— und gepickt wird aus 90-100+, NICHT aus 80-90. 80-90-Signale werden nur an ruhigen Tagen
gehandelt. Geringer praktischer Payoff. Der Score-Rebuild fixte bereits das Bucket das zaehlt.

**Trigger zum Anpacken:** naechster substanzieller Score-Tag (dann Loesung B als
perf_120-Unification, Backtest-First, POWER-Carve-Out erhalten). Quick-Alternative falls
gezielt: WEAK-Penalty -15->-20 ODER Trend-Konflikt-Penalty (perf_120<0 UND perf_20>0).

---

## 12. Sektor-Bonus entfernen (kein prädiktiver Wert) — getestet 2026-06-26

**Befund:** Die Sektor-RS-Hypothese (Sektor-ETF-Momentum vs SPY als Score-Faktor) wurde an
83 echten BREAKOUT-Trades getestet — **FALSIFIZIERT**. WR nach Sektor-RS-Bucket ist reines
Rauschen, nicht-monoton (schwächster Sektor-Bucket hatte die *höchste* WR 67 %). Trennschärfe:
Sektor schwach (<-2pp vs SPY) = 56 % WR vs Sektor stark (≥+2pp) = 54 % — **kein Edge.**

Die 27+27 `sector_relative_strength`/`sector_momentum_tailwind`-Postmortem-Tags waren
**Post-hoc-Narrativ** (Gewinner rückblickend gelabelt; Verlierer hatten denselben „Rückenwind").
Hindsight-Bias — Tag-Häufigkeit ≠ prädiktiver Edge. Klassischer Backtest-First-Save.

**Eigentliche Action:** Der **bereits existierende absolute Sektor-Bonus** in ApexScan.py
(~Zeile 1319-1335, `sec_perf >= 5 → +6` etc.) zeigt im selben Test **auch keine Trennung**
(flat ~54 % WR über alle Buckets). Er addiert Noise auf den Score und verwässert die
Kalibrierung. **Vorschlag: entfernen** — reiner Vereinfachungs-Gewinn (kein Feature-Verlust).

**Warum geparkt:** Niedrige Prio, schadet nicht akut. Vor dem Entfernen kurzer Backtest
(Score mit/ohne Sektor-Bonus, Signal-Count darf nicht sinken — Signal-Protection). Der
Stock-vs-SPY-RS-Bonus (Zeile 1311-1317) NICHT anfassen — der ist eine andere Metrik (eigene
Aktie vs Markt) und nicht mitgetestet.

**STATUS 2026-07-11: UMGESETZT (Deepsearch-Session).** Sektor-Bonus aus ApexScan entfernt.
Zusatz-Evidenz: Bonus war prozyklische Inflations-Quelle (71/169 Signale +6; Energy-Mai-
Cluster HAL/HP/SM alle 130+ am Sektor-Top, alle Verlierer). Signal-Count-Check via
Rekonstruktion: 163->162 Signale ueber Gate 70 (-0.6%, Protection OK). RS-Bonus unangetastet.

---

## 13. Score-Anti-Monotonie 130+ — Root-Cause + Fix (2026-07-08, Backtest-validiert)

**Problem:** BREAKOUT-Score-Bucket 130+ hatte WR 17% (n=12) vs Sweet-Spot 90-130 (68-72%).

**FALSIFIZIERTE Hypothesen (Daten):**
- **Extension-Filter (perf_120/perf_20):** FALSCH. Die 2 Wins im 130+ (LRCX +134%, AMAT +79%)
  sind die AM STÄRKSTEN extended; groesster Loser FLR hatte niedrigste Extension (+31%).
  perf_120-Sweep: nur +1pp WR bei 14-19% Signal-Loss. Kein sauberer Loser-Cluster.
- **Sektor-Concentration-Cap:** FALSCH. CAP=2 gab NULL WR-Lift (57.1%->57.1%), droppte LRCX-Winner,
  35% Signal-Loss. Gedroppte Trades WR 52-57% (=Baseline). BACKLOG #3 damit auch erledigt/verworfen.
- **Broad-Regime-Gate (SPY bearish):** zu grob. BREAKOUT bearish PF 1.62 aber immer noch profitabel
  (avg +1.38%) — pauschaler Stopp wirft Profit weg.

**ROOT-CAUSE (echt):** Tech-Breakouts wenn QQQ selbst schwach = Semi-Reversal-Failure-Mode.
Kreuztabelle Sektor x QQQ-Regime:
- TECH + QQQ<0: WR 14% PF 0.56 (n=7) 💀
- TECH + QQQ>0: WR 54% PF 2.37
- NON-TECH + QQQ<0: WR 57% PF 1.72 (fine!)
- NON-TECH + QQQ>0: WR 62% PF 2.97

**FIX (live):** TECH_QQQ_GATE_ENABLED in ApexScan.py — skip BREAKOUT wenn Sektor Tech/Comm
UND qqq_perf_20<0. Effekt: WR 57.1->59.8%, PF 2.29->2.53, Signal-Loss 6%.
Vorbehalt: n=7, Monitoring noetig. Rollback = Flag False.

---

## 14. Yahoo-Screener als Intraday-Zusatzquelle (2026-07-10, zurueckgestellt)

**Idee:** Intraday-Mover zusaetzlich aus einem Web-Screener ziehen (yfinance
`Screener().set_predefined_body('...')`), nicht nur aus dem kuratierten us_tickers.txt
(719) + eu_tickers.txt (107). Bringt Small-/Mid-Caps mit Heute-Explosion die nicht in
unserer Blue-Chip-Liste stehen.

**Warum NICHT gebaut (2026-07-10):**
- `day_gainers` / `most_actives` liefern das **Peak-Universum** (+8% bis +30% intraday).
  Die fallen bei INTRADAY_GAIN_MAX=6% ALLE raus — wir wollen bewusst NICHT im Blow-off kaufen.
- Viele day_gainers sind Micro-Caps (Pump-and-Dump-Risiko) und bei eToro-Demo eh nicht handelbar.
- Wir haben stattdessen Ansatz B gebaut: volles us+eu-Universum (826) + 5 Anti-Peak-Chase-Filter
  (siehe #15). Erst dieses kuratierte Universum ausreizen.

**Wenn spaeter reaktiviert:** NICHT `day_gainers` (Peak), sondern `small_cap_gainers` oder
`undervalued_growth_stocks` als Screener-Slug. Die 25 Screener-Ticker in den us_tickers-Pool
mergen, dann derselbe 2-stufige Filter (1d-Pre-Filter -> 5m-Deep-Scan). Trigger fuer
Reaktivierung: wenn kuratiertes Universum dauerhaft < 1-2 Intras/Tag liefert UND wir mehr wollen.

---

## 15. Intraday-Anti-Peak-Filter — Kalibrierung beobachten (2026-07-10)

**Gebaut (Commit 83da9d3):** 5 Filter im Intraday-Deep-Scan gegen "im Hoch kaufen":
1. RANGE_POS_MAX=0.90 (nicht am absoluten Top)
2. VOL_RATIO 1.3-5.0x (Bestaetigung, kein Blow-off)
3. Konsolidation: letzter 5m-Bar kein neues Tageshoch (HARD-SKIP — aggressivster Filter)
4. Time-Gate 10:00 ET (Open-Volatility raus)
5. Gap-Filter 3% vs Vortag-Close (News-Spike raus)

**Validierung:** Alle 4 heute (13:35Z) gekauften Intras (AMAT/JCI/LII/F) hatten range_pos>0.90
= am Top gekauft, alle 4 rot. Neuer Filter haette alle 4 geskippt.

**Spannung Quantitaet vs Qualitaet:** User will viele Intras/Tag. Filter reduzieren Durchsatz.
ABER: Durchsatz-Treiber ist das Universum (1->826 Ticker), nicht der Filter. Peak-Buy-Durchsatz
= negativer EV (heute 4/4 rot). Reconcile-Hebel wenn nach 3-4 Tagen zu duenn (grep intraday: ~/trader.log):
1. Filter 3 (Konsolidation) Hard-Skip -> Score-Penalty (throughput-killer, schwaechst begruendet)
2. RANGE_POS_MAX 0.90 -> 0.92
3. VOL_RATIO_MIN 1.3 -> 1.2

**Monitoring:** In 3-4 Tagen Intra-Count/Tag pruefen. Ziel grob 1-3/Tag. Bei 0/Tag -> Hebel 1 ziehen.

---

## 16. G3-Tagestief-Stop fuer Rot-Rescues — FALSIFIZIERT (2026-07-11)

**Hypothese (Brief AP1 Schritt 1-NEU):** Rote EOD->SWING-Rescues bekommen statt
pauschal `entry*0.96` den engeren Stop `max(entry*0.96, tagestief_conv_tag*0.995)`.
Recovery-These "haelt das heutige Tief" — Falsifikation des Turns statt 4% Ausbluten.

**Retro-Simulation (6 Rot-Rescues, Daily-Bars, Akzeptanz vorab fixiert):**
- (a) PAY + FRSH ueberleben: JA (beide Tiefs nie unterschritten) ✓
- (b) Bleeder MRCY/VERX/PLTR Verbesserung: nur **+0.35pp/Trade** (Soll >= +1.5pp) ✗
- (c) Netto-Delta: +1.05pp (>= 0) ✓
- **NO-GO nach Kriterium (b).**

**Warum es nicht traegt:** Die Conv-Tagestiefs lagen nur 1-3.5% unter Entry — der
G3-Stop hebt kaum an (MRCY +0.86, VERX +-0.00, PLTR +0.18). Die Bleeder fielen durch
BEIDE Stop-Level durch. Tagestief-Naehe ist keine Turn-Information.

**Konsequenz:** -4%-Stop bleibt (nachgewiesener Preis der Runner-Option: PAY +18-21%).
Stattdessen Shadow-Logging live (AP1 Schritt 1b, seit 2026-07-11): range_pos_eod,
above_vwap_eod, dist_to_day_low_pct, high_since_entry_pct im intraday_to_swing-Event.
Bei n>=15 neuen Konvertierungen pruefen ob range_pos_eod<0.4 (Spike-Fade) Turner von
Bleedern trennt. Sim-Skript: scratchpad g3_sim.py (Session 07-11), Rezept im Brief §AP1.

---

## 17. SCORE_V2 (LogReg-Rekalibrierung) — FALSIFIZIERT in Stufe 2 (2026-07-11)

**Kontext:** Brief AP2 (CLAUDE_CODE_BRIEF_2026-07-10_SIGNAL_QUALITY.md) — Score war
rangpraediktiv tot (Spearman -0.066 auf Live-Join n=137). Dreistufiger Plan mit
vorab fixierten Gates.

**Stufe 1 (Offline-Join-Replay, Commit 05ce1fc):**
- V2a (Ballast raus ohne Fit): OOS-Spearman -0.119 -> NO-GO.
  **Lehre: Ballast-Entfernung allein heilt das Ranking nicht.**
- Baseline-Perzentil (score_pct): -0.202 -> **Inflation ist nicht das Kernproblem.**
- V2b_pct (LogReg + Tages-Perzentil): +0.233, TopQ-WR 69.2% -> formal GO.
  ABER Ablation: GO hing KOMPLETT am negativ gelernten movement_bonus-Gewicht
  (ohne: -0.013). Die Ein-Regime-Inversion vor der der Brief warnte.

**Stufe 2 (2J-Backtest, --score-v2 Flag, 504 Handelstage, Pick-Stufe Top-5/Tag):**
| | n | WR | PF | Sum |
|---|---|---|---|---|
| Baseline (--score-realign) | 203 | 50.2% | 2.02 | +454% |
| SCORE_V2 | 209 | 45.5% | 1.57 | +266% |
- WR-Kriterium (>= +2pp) FAIL, PF-Kriterium FAIL -> **NO-GO, verworfen.**

**Kern-Lehren:**
1. Der movement_bonus-Flip generalisiert NICHT — 2026-Regime-Artefakt. Brief-Warnung
   ("Vorzeichen NICHT flippen") war korrekt, der 2J-Test hat sie bestaetigt.
2. WICHTIGE NUANCE: Der aktuelle Score funktioniert als PICK-RANKER ueber 2 Jahre
   weiterhin (Top-5/Tag: WR 50.2%, PF 2.02, profitabel). Die tote Spearman-Korrelation
   im 2026-Live-Sample ist womoeglich Regime-Rauschen, nicht struktureller Verfall.
   Score-Anfassen hat aktuell KEINE datengestuetzte Grundlage.
3. Walk-Forward auf EIN Regime (Maerz-Juli 2026) reicht nicht als Gate — der
   2J-Backtest als Stufe-2-Pflicht hat den Fehlschluss abgefangen. Prozess beibehalten.

**Artefakte (opt-in, kein Live-Effekt):** --score-v2 Flag + Feature-Metadaten in
apex_backtest_v2.py, score_v2_model.json (frozen, mit Caveat), apex_score_v2_stage1.py,
apex_score_v2_stage2_compare.py. Ergebnis-Files bt_baseline_2y.json / bt_scorev2_2y.json.

**AP5 (Soft-Diversity-Nudge) Status:** war auf "nach AP2-Entscheid" gated. Entscheid =
Score bleibt. Nudge-Rationale (10 Score-Punkte tragen ~keine Info) gilt weiterhin —
falls angepackt, gegen den BESTEHENDEN Score backtesten (Replay-Plan im Brief §AP5).

---

## 18. EU-Universe-Bug — GEFIXT: normalize_ticker zerstoerte Suffixe (2026-07-11, AP4)

**Root-Cause (Brief §1.6 "0 EU-Signale in 4 Monaten"):** `normalize_ticker()` ersetzte
ALLE Punkte durch Bindestriche (gebaut fuer US-Class-Shares BRK.B -> BRK-B). Damit wurde
`SAP.DE` zu `SAP-DE` — bei Yahoo unbekannt -> **0/106 EU-Frames seit Tag 1**. Die
Liquiditaets-Hypothese war falsch (P10 avg_dv = $46M, 0/102 unter der $500k-Schwelle).
Derselbe Bug war in apex_backtest_v2.py load_tickers -> **alle bisherigen Backtests
liefen faktisch US-only** (Vergleiche bleiben gueltig, beide Seiten gleich betroffen).

**Fix:** EXCHANGE_SUFFIXES-Whitelist {DE,PA,AS,SW,L,MC,MI} in ApexScan.normalize_ticker
+ apex_backtest_v2._normalize_ticker. Class-Shares weiter BRK.B->BRK-B, Suffixe bleiben.
Diagnose-Tool: apex_eu_diagnose.py (pro-Ticker Drop-Grund + avg_dv-Verteilung).

**Nach Fix (Diagnose-Lauf 07-11):** 102/106 Frames OK. Drop-Gruende sind NORMALE
Tagesfilter (fail_trend 54, fail_volume 43 = Setup-Vol-Confirm). Stale Ticker bereinigt:
CRH.L raus (jetzt NYSE, in us_tickers), AHOG.AS->AD.AS, DSMFP.AS->DSFIR.AS, ROG.SW->RO.SW.

**Offene Punkte (Monitoring):**
1. **30 EU-Signale sammeln, dann getrennt auswerten** (Edge unbekannt — es gab nie eins).
2. **eToro-Handelbarkeit pruefen** beim ersten EU-Trigger: resolve_ticker nutzt
   internalSymbolFull (SAP.DE funktioniert lt. reference_etoro_api, andere Suffixe
   ungetestet). Resolve-Fail = graceful skip (Paper-only), aber beobachten.
3. **Handelszeiten-Ueberlappung:** Trader-Cron laeuft 13-21 UTC (US-Stunden), Xetra
   schliesst 15:30 UTC — EU-Trigger feuern nur im Overlap-Fenster, Exits auf stale
   Kursen moeglich. Bei >5 EU-Positionen: EU-spezifisches Zeitfenster diskutieren.

---

## 19. Ticker-Winrate-Bonus (±10) — nie validiert, Inflations-Verdacht (2026-07-11)

**Kontext (Deepsearch):** `score += (wr - 50) * 0.2` aus ticker_winrate.json (ApexScan
~Z. 1330). Existiert NICHT im Backtest (pfadabhaengig auf Live-Knowledge) — also nie
gegen 2 Jahre validiert. Mechanisch prozyklisch: Ticker die zuletzt gewonnen haben
kriegen +Bonus -> Re-Entry am Top wird belohnt. Waechst mit der Knowledge-DB =
plausible Quelle der Score-Inflation (Median Maerz 85.7 -> Juni 115.3).

**Pruef-Rezept (eigene Session, ~2h):**
1. Git-History von ticker_winrate.json: Snapshots pro Signal-Datum auschecken
   (`git log --format='%H %ad' -- ticker_winrate.json`), Bonus pro Signal rekonstruieren.
2. Join mit equity_results: Lift der Bonus-Traeger vs Nicht-Traeger (WR-Differenz).
3. Wenn Lift <= 0: entfernen (wie Sektor-Bonus #12). Signal-Count-Check Pflicht
   (Bonus bis ±10, Gate-Naehe pruefen).

**Warum geparkt:** Sektor-Bonus-Entfernung + TG_SWEET_BAND (beide 2026-07-11 live)
zuerst 2-4 Wochen wirken lassen — nicht drei Score-Aenderungen gleichzeitig, sonst
ist Attribution unmoeglich.

---

## 20. Sektor-relative-Strength-Gate für BREAKOUT (2026-07-15, HEUTE bauen)

**Root-Cause (Learn 07-15 + Postmortem 25 Pending):** Der 30d-WR-Einbruch (32.6% vs
Lifetime 51%) kommt v.a. aus dem **Semi-Selloff Ende Juni** (LRCX -9.2, KLAC -9.3,
CGNX -6.7, alle Tech, alle D+3-4). Web-verifiziert: sektorweiter Baerenmove (Memory-
Kosten, AI-Spending-Scrutiny, TSMC-Stake, Samsung-Streik). Gemeinsamer Nenner:
**Sektor-ETF divergierte -2.6 bis -3.65% negativ TROTZ positivem SPY.**

**Warum TECH_QQQ_GATE das NICHT deckt:** feuert nur bei `qqq_perf_20 < 0`. Im Semi-
Selloff war QQQ teils noch positiv/flach, aber XLK/Semi brach ein. = die "neue Zelle"
vor der der Fable-Brief (§1.5) warnte.

**Hypothese (falsifizierbar):** BREAKOUT skippen wenn Sektor-ETF-Momentum (sector_etf
_perf_20) < Schwelle X UND negativ vs SPY divergiert (sector_perf - spy_perf < -Y).
Datenbeleg: Postmortem-Sektor-Divergenz-Loser-Tabelle + die 3 Semi-Trades.

**Backtest-Plan (Backtest-First, VOR Live):**
1. Join Signale mit equity_results, Feature sector_etf_perf_20 (aus market_context /
   Sektor-ETF-Historie rekonstruieren, wie in apex_score_v2_stage1.py gemacht).
2. Sweep Schwellen (sector_perf_20 < 0 / < -2 / divergenz < -2pp / -5pp).
3. Akzeptanz VORAB: WR-Lift der gepickten Trades >= +3pp UND Signal-Count >= 95%
   Baseline (Signal-Protection) UND kein Winner-Cluster gedroppt (SE/STX-Typ mit
   Catalyst muss durch — Catalyst-Carve-Out wie SCORE_REBUILD pruefen).
4. NUR wenn GO: Flag SECTOR_RS_GATE_ENABLED in ApexScan.py, analog TECH_QQQ_GATE.

**WICHTIG — der Catalyst-Konflikt (nicht ignorieren):** Learn sagt `Breakout x perf_120
<0` = 70% WR (n=20), aber die perf_120<0-Loser (HRB/GIS/DT) hatten KEINEN Catalyst,
SE (perf_120 -26.5, WON +13%) HATTE einen (Earnings-Beat). Der Bucket ist bimodal —
Catalyst trennt. Ein pauschales Sektor/Momentum-Gate darf die Catalyst-Winner nicht
killen. Carve-Out einbauen + im Backtest die gedroppten Winner zaehlen.

**Blacklist-Nebenaktion:** TSM (25% WR/4), ASML (40%/5) sind chronische Semi-ADR-
Underperformer -> Blacklist-Kandidaten (wie BAD_PERFORMERS-Set).

**BACKTEST-ERGEBNIS (2026-07-15, apex_sector_rs_gate_backtest.py):**
Baseline (144 BREAKOUT-Trades): WR 51.4%, PF 1.77, sum +255%.
Beste Variante **TECH sec<0 + Catalyst-Carve-Out**: WR 54.5% (+3.1pp), PF 2.10,
sum +296% (Profit STEIGT), Retention 92%. Droppt 12 Trades: 10 Loser / 2 Winner
(83% Loser-Anteil). Fängt exakt den Ziel-Cluster: WDC/KLAC/LRCX/CGNX/UI/DT/TKO/
FFIV/CSCO/XYZ (alle Tech in schwachem Sektor).

**FORMAL NO-GO** an den vorab fixierten Kriterien: Retention 92% < 95%-Signal-
Protection-Bar. KEINE Variante schafft beide (Cluster fangen = ~8% Signal-Loss).

**ABER differenziert:** Der 95%-Bar hat die Escape-Hatch "Loser-Anteil rechtfertigen"
(feedback-signal-protection) — 83% Loser-Drop erfüllt das. PF steigt deutlich (1.77->2.10),
Profit STEIGT trotz Drop. Der Haken: droppt **CIEN +16.6%** (Post-Earnings-Winner, den
der Carve-Out verpasste — Beat war 13d alt, außerhalb des recent-beat-Fensters).

**Vorbehalt Regime:** Effekt v.a. vom Juni-Semi-Selloff getrieben = event-spezifisch,
Generalisierung unklar (nur 2026-Sample, kein 2J-Backtest möglich da Sektor-ETF-Historie).

**OFFENE ENTSCHEIDUNG (User):** (a) TECH sec<0 hart schalten (8% Signal-Loss akzeptieren,
Loser-Anteil rechtfertigt), (b) SOFT-Penalty statt Hard-Gate (Sektor-schwacher Tech-BO
kriegt Score-Malus -> rankt niedriger in Pick/Telegram, KEIN Signal-Loss, wie TG_SWEET_BAND),
(c) verwerfen (regime-spezifisch, CIEN-Kosten). Empfehlung: (b) — signal-protection-neutral.

**STATUS 2026-07-15: LIVE geschaltet** (SECTOR_RS_GATE_ENABLED=True, ApexScan.py). Entscheidung: Hard-Skip am Scan-Ursprung (= no-buy bis Sektor gruen via taegliches Re-Scan), NICHT Score-Malus (haette mit TG_SWEET_BAND kollidiert: gedrueckter Score landet im bevorzugten 90-120-Band). TECH-scoped + Catalyst-Carve. Monitoring: in 2-3 Wochen Signal-Loss + WR gegen Learn pruefen; CIEN-Typ-Winner (13d-alter Beat) beobachten.

**STATUS 2026-07-15: INSTRUMENTIERT.** Git-Rekonstruktion war infeasible (ticker_winrate.json nur 1 Commit). Ledger-Point-in-Time nur n=13 (bedeutungslos). Footprint 25% der BREAKOUT-Signale, Mittel-Bonus +0.5 (ausbalanciert, KEINE Inflations-Quelle wie befuerchtet). Praediktivitaet aktuell NICHT messbar -> Bonus BLEIBT. Instrumentierung eingebaut: apex_signals.json loggt jetzt winrate_bonus/winrate_pit/winrate_n bei Emission. In ~3 Mon saubere Point-in-Time-Daten -> dann sauber beurteilen (Split winrate_pit>50 vs <50 gegen Outcome).

---

## 21. eToro-Close-Backfill — GEFIXT (2026-07-16, RHI-Bug)

**Befund (User):** RHI sprang bei Open durch den TP, eToro verkaufte 15:32 mit ~13.3%, aber
es stand NICHT im eToro-Tab und das Paper buchte nur +13.01%.

**Root-Cause (Race):** eToro-TP feuerte 13:32 UTC (Fill 40.45 UEBER TP 40.37). Beim 13:35-Run
zeigte eToros Portfolio-API RHI noch als offen (Lag ~3min) -> Sync tat nichts. Danach schloss
der Paper-Trader selbst bei seinem theoretischen Target 40.40 (+13.01%) und rief
etoro_close_position -> scheiterte (auf eToro laengst zu) -> KEIN Event. **Und der Sync schaut
nie wieder auf geschlossene Positionen** -> echte Close-Rate fuer immer verloren.
Gegenprobe: ANET + CMG (beide eToro-SL, Paper hatte NICHT selbst geschlossen) wurden korrekt
via close_from_history erfasst. Nur wenn Paper das Rennen gewinnt, geht es verloren.

**Fix:** sync_etoro_positions backfillt jetzt bereits geschlossene Live-Trades ohne
etoro_close_rate (letzte 7 Tage) aus der History: setzt echte close_rate/net_profit, korrigiert
pnl_pct (aus eToros openRate) + pnl_usd (= netProfit, echtes Geld), leitet den echten Reason
ab (TP/SL/closed) und loggt ein close_backfill-Event -> erscheint im eToro-Tab. Audit-Felder
paper_exit_price/paper_exit_reason/paper_pnl_pct bleiben erhalten. History-Lag ist egal: wenn
noch nicht da, versucht es der naechste Run erneut.

**RHI-Effekt:** Take Profit 40.40/+13.01%/$6.50 -> eToro TP 40.45/+13.21%/$6.60.

---

## 22. Intraday-Reject-Log — sperren wir die staerksten Mover aus? (2026-07-16, MESSEN)

**Widerspruechliche Evidenz, n winzig auf beiden Seiten:**
- **10.07.:** 4 Intradays bei range_pos>0.90 gekauft -> ALLE 4 rot. Daraufhin Anti-Peak-Filter gebaut (RANGE_POS_MAX=0.90 etc.).
- **16.07.:** RHI +7.5%, MAT +6.5%, DXCM +5.7% liefen den GANZEN Tag und hielten ihre Hochs (MAT schloss AUF dem Tageshoch, range_pos 1.00). Gekauft haben wir nur IR (+3.1% Peak, verblasste auf -1.1%) — einer der Schwaechsten. RHI war durch den Close-Cooldown gesperrt (korrekt), MAT/DXCM vermutlich durch RANGE_POS_MAX=0.90.

**Konkreter Verdacht (2 Schwellen):**
1. `RANGE_POS_MAX = 0.90` — starke Trends laufen per Definition am Hoch entlang und machen staendig neue Highs. Genau das filtern wir raus.
2. `GAIN_MAX = 6.0` — Mover wachsen aus dem Vorfilter HERAUS wenn sie am staerksten laufen (RHI +7.5%, MAT +6.5% ueberschritten es).

**NICHT geaendert** — 4 Trades gegen 1 Tag ist keine Datenbasis. Stattdessen instrumentiert:
`apex_intraday_rejects.json` loggt tages-dedupliziert jeden abgelehnten Kandidaten mit Grund
(gain_too_high / range_pos_too_high / vol_ratio_out / below_vwap / before_10et / gap_too_large)
+ gain/range_pos/vol_ratio/last. Retention 14d.

**Auswertung in 2-3 Wochen:** Join der Rejects mit dem Tagesverlauf (was wurde aus ihnen bis
Close?) -> WR/avg-Move der Abgelehnten je Ablehnungsgrund. Wenn `range_pos_too_high` +
`gain_too_high` systematisch Gewinner aussperren -> Schwellen lockern (0.90->0.95, 6.0->8.0).
Wenn sie ueberwiegend Fader treffen -> bestaetigt, so lassen.

**ACHTUNG run_trader.sh:** apex_intraday_rejects.json MUSS in die git-add-Liste, sonst bleibt
das Log auf der VM (wie apex_intraday_cache.json, das nie committed wurde).

**AUSWERTUNG 2026-07-22 (4 Tage, 848 Rejects, 199 Anti-Peak):** Fuer jeden Anti-Peak-Reject
den hypothetischen Intraday-Play ab Ablehnungspreis simuliert (TP+5/SL-3/EOD, echte 5m-Bars):

| Filter | n | WR | avg | TP-Rate | Ø-max danach |
|---|---:|---:|---:|---:|---:|
| range_pos_too_high | 194 | 31% | -0.63% | 1% | +0.38% |
| gain_too_high | 5 | 80% | +1.31% | 20% | +2.55% |

**ERGEBNIS: RANGE_POS_MAX=0.90 ist VALIDIERT — filtert Fader, nicht Gewinner.** Abgelehnte
Mover verblassen (-0.63% avg, nur 1% haetten TP erreicht, Best-Case-Move nur +0.38%). range_pos
>0.90 = Kauf am Tageshoch = Mean-Reversion. **NICHT lockern.** Damit ist die 10.07.-vs-16.07.-
Frage (RHI/MAT) entschieden: die 4 Peak-Kaeufe am 10.07. (alle rot) waren die Regel, nicht MAT.
`GAIN_MAX=6.0`: n=5 zu klein (1 Continuation SMCI +5.2%), weiter beobachten, kein Change.
Nebenschluss: der Intraday-CATCHER ist nicht das Problem — Verluste sitzen im MOMENTUM-Subtyp
+ intraday_rescued (28% WR). Reject-Log-Monitoring kann runterprioren; Fokus auf #16 (Rescue).

---

## 23. EU-Grundsatzentscheid: "messen statt bauen" — umgesetzt 2026-07-17

**Die Frage (User, 16.07.):** EU und US abkapseln (eigener Cron/Filter/Zeitfenster) ODER EU
so anpassen dass es in unsere Signal-Logik passt?

**Der eigentliche Befund — es ist kein Filter-, sondern ein TAKT-Problem:**
EU-Boersen schliessen **15:30 UTC**, unser Trader-Cron laeuft **13:00-21:00 UTC** = nur 2.5h
echte Ueberlappung. Und der Trigger-Mechanismus macht es schlimmer: `trigger_pending` prueft
`high_today >= entry` (Tageshoch), **kauft aber zum aktuellen Preis**. Ein SAP-Signal das um
08:00 UTC durch `buy_above` springt wird erst um 13:00 gesehen — der Trigger traegt, aber
ausgefuehrt wird 5h spaeter zu einem beliebigen Kurs. Nach 15:30 UTC liefert yfinance den
Close-Bar unveraendert weiter, ohne ihn als stale zu markieren -> Entry/Stop-Check auf 2.5h
alten Daten. Nie passiert (0 EU-Positionen seit Tag 1), aber strukturell real.

**Warum "anpassen" nicht traegt:** Begrenzt man EU-Trigger auf das 2.5h-Fenster, verpasst man
jede Morgen-Bewegung und die Signale expiren an `MAX_TRIGGER_DAYS`. Der Takt ist Physik,
kein Parameter.

**Warum "abkapseln" (eigener EU-Cron 07:00-15:30) nicht JETZT:** hoher Aufwand (zweiter Cron,
EU-Gates ueberall, eToro-EU-Handelbarkeit ungetestet) fuer einen Edge den wir bei **0 Signalen
in 4 Monaten** noch nie gesehen haben. Doppelte Komplexitaet bei $400 Kapital.

**ENTSCHEID (User 17.07.): messen statt bauen.** Der Hebel: **der Equity-Tracker simuliert
alle Signale auf Daily-Bars — voellig unabhaengig vom Live-Takt.** Wir koennen den EU-Edge
also sauber vermessen, ohne einen EU-Euro zu riskieren und ohne einen zweiten Cron.

**Umgesetzt (apex_trader.py):**
1. **EU bleibt im Scanner** — unveraendert. Signale werden emittiert + vom Equity-Tracker
   simuliert = die Datenquelle fuer den Entscheid.
2. `INTRADAY_EU_ENABLED = False` — EU raus aus dem Intraday-Universum (825 -> 719 Ticker).
   Dort ist der Takt-Konflikt am schaerfsten: nach 15:30 UTC eingefrorene 5m-Bars, der
   Vorfilter liest den KOMPLETTEN EU-Tag als `gain_from_open`, das Time-Gate (et_hour>=10)
   winkt die letzte EU-Bar durch -> Peak-Kauf auf totem Chart.
3. `EU_GUARD_ENABLED = True` + `_eu_entry_blocked()` — LIVE-Entry in EU-Titel nur im Fenster
   07:00-15:15 UTC (15 Min Puffer vor Close), nicht am Wochenende. Sitzt in `trigger_pending`
   NACH dem Expiry-Check (Signal darf normal verfallen) und VOR beiden Kauf-Pfaden
   (Replacement + regulaerer Trigger). Blockiert nur den Entry, das Pending wartet auf ein
   Fenster mit echten Preisen. Fail-closed bei Exception.

**Auswertung — Trigger: n>=30 EU-Trades in apex_equity_results.json.** Zaehlen mit:
```python
import json; from apex_trader import _is_eu_ticker
r = json.load(open('apex_equity_results.json'))
eu = [t for t in (r if isinstance(r, list) else r.get('trades', [])) if _is_eu_ticker(t.get('ticker',''))]
print(len(eu), 'EU-Trades')
```
Dann: WR/PF der EU-Trades vs US-Baseline (Lifetime 51.7% / PF 1.78).
- **EU-Edge >= US-Baseline** -> eigener EU-Cron lohnt sich, dann sauber bauen (Option
  "abkapseln" mit Daten statt Bauchgefuehl).
- **EU-Edge deutlich drunter** -> EU-Ticker ganz raus, Scan wird schneller.
- **Immer noch ~0 Signale nach 3 Monaten** -> Antwort ist eh "raus".

**Erwartete Dauer:** bei US-Pass-Rate (0.478%/Ticker-Tag) x 106 EU-Ticker ~ 0.5 Signale/Tag
-> n=30 in grob 60 Handelstagen (~3 Monate, also ca. Oktober 2026). Kostet bis dahin nichts.

**Rollback:** `EU_GUARD_ENABLED = False` / `INTRADAY_EU_ENABLED = True`.

### 23b. EU-Intraday-Bucket — geprueft, GEPARKT (2026-07-17, User-Frage)

**Frage:** Statt EU ganz aus dem Intraday zu nehmen — gibt es ein eigenes Zeitfenster
("Bucket") in dem EU-Intras funktionieren?

**Struktur (machbar, aber mit Bedingungen):**
- `INTRADAY_MIN_ET_HOUR=10` (=14:00 UTC) ist fuer EU **sinnlos**: es soll die erste
  US-Handelsstunde abfangen, trifft bei EU aber eine Session die seit 7h laeuft. Es
  schneidet 5 von 8.5 EU-Stunden weg, aus einem Grund der fuer EU nicht gilt.
- **Der Exit ist das echte Problem, nicht der Entry:** `INTRADAY_EOD_UTC=19:45` liegt
  4h15 NACH EU-Close. Ein EU-Bucket braucht zwingend einen **eigenen EOD-Cutoff (~15:15
  UTC)**, sonst haelt man "Intraday"-Plays over-night auf eingefrorenen Kursen.
- Option A: Bucket im bestehenden Cron, Fenster 13:00-14:45 (~7 Laeufe/Tag), Aufwand
  klein-mittel, kein neuer Cron. Option B: Cron auf 07-21 vorziehen -> Fenster 08:00-14:45
  (7h15), aber VM-Last x2 und faktisch der "zweiter Trader"-Weg durch die Hintertuer.

**Daten (Mini-Backtest, echte 5m-Bars, 60d = yfinance-Limit, Filter aus apex_trader.py
importiert):**

| | EU (105 Ticker) | US (Stichprobe 152) |
|---|---:|---:|
| Kandidaten/Tag | 27.1 | 42.5 |
| **Follow-Through** (Entry->Close, ohne TP/SL) | **+0.026%** | **-0.189%** |
| Anteil positiv | 50.5% | 42.5% |
| WR mit TP+5/SL-3 | 50.1% | 42.2% |
| PF | 1.00 | 0.67 |
| Top-1/Tag nach vol_ratio | -0.160% | -0.286% |

**Befund 1: Kandidaten gibt es reichlich** (27/Tag) — der Filter ist NICHT der Engpass.
**Befund 2: EU ist nicht schlechter als US — in dieser Messung sogar besser.** Die These
"EU-Blue-Chips haben kein Intraday-Momentum" ist damit NICHT belegt; US hat unter
identischen Filtern einen *negativen* Follow-Through.
**Befund 3: der +5%-TP wird in beiden Maerkten fast nie erreicht** (EU 7/1706, US 16/2550).

**WARNUNG — die Sim ist zu grob, sie bildet unseren Trader NICHT ab:** 16 TPs/2550 (Sim)
vs **4 TPs/8 (echter Trader, +1.74%/Trade)**. Die Diskrepanz ist zu gross fuer Zufall.
Fehlend: (a) **Konsolidations-Regel** (letzter 5m-Bar kein neues Tageshoch = HARD-SKIP,
der aggressivste Filter ueberhaupt), (b) der **Vorfilter** (Top-100 nach gain aus 826),
(c) `vol_ratio` ist hier "5m-Bar vs Tagesschnitt", im Trader "Tagesvolumen vs Vortage-
Schnitt" = eine andere Metrik. Der echte Trader ist 10-40x selektiver. Die Zahlen oben
beschreiben die **Signal-Klasse**, nicht unser System.

**ENTSCHEID: GEPARKT.** Nicht weil EU schlecht waere, sondern weil das Fundament fehlt —
**wir wissen nicht ob der Intraday-Kanal ueberhaupt einen Edge hat (n=8 live, siehe #25).**
Einen EU-Bucket auf ein unvalidiertes Konzept zu setzen verdoppelt die Unsicherheit statt
Edge zu addieren.

**Trigger zum Anpacken:** validierter US-Intraday-Edge (#25) — dann ist der Bucket
Option A in ~2h gebaut (EU-Time-Gate + EU-EOD-Cutoff 15:15 + `INTRADAY_EU_ENABLED=True`).
Rezepte: scratchpad/eu_bucket_{1..4}_*.py (Session 07-17), Kandidaten-Caches
eu_cands.json / us_cands.json.

---

## 25. Der Intraday-Kanal ist nie validiert worden (2026-07-17)

**Aufgefallen bei der EU-Bucket-Pruefung.** Wir haben einen kompletten Signal-Kanal
(Intraday-Momentum-Catcher, seit 06-18) der nie gegen eine Stichprobe validiert wurde:
- **Live: n=8 geschlossene Intradays** — 4x TP (+4.99%), 2x Stop (-3.02%), 2x EOD (-0.02%)
  = +1.74%/Trade. Sieht gut aus, ist aber **statistisch nichts** (4/8 TP kann Zufall sein).
- **Der Backtest kann ihn nicht pruefen** (#24 + yfinance liefert 5m nur 60 Tage).
- **Eine grobe 60d-Sim (ohne Konsolidations-Filter/Vorfilter) findet KEINEN Edge**:
  US-Follow-Through **-0.189%**, nur 42.5% positiv, Top-1/Tag sogar -0.286% (= je
  "staerker" das Signal, desto schlechter -> Mean-Reversion statt Momentum).
  **Das ist kein Beweis** (die Sim ist 10-40x weniger selektiv als der Trader), aber es
  ist auch **keine Entwarnung** — es gibt schlicht keinen Nachweis in beide Richtungen.

**Warum das zaehlt:** Der Kanal hat 3 von 7 Slots reserviert (`INTRADAY_RESERVED_SLOTS=3`)
und darf Swings verdraengen. Falls er keinen Edge hat, kostet er nicht nur seine eigenen
Trades, sondern auch die verdraengten Scanner-BREAKOUTs (WR 52% lifetime).

**Naechster Schritt (laeuft schon):** Das **Reject-Log (#22, ab 17.07. live)** sammelt
genau die noetigen Daten — welche Kandidaten lehnen wir ab, was wird aus ihnen. Zusammen
mit den Live-Intras ergibt das in 2-3 Wochen die erste echte Datenbasis. Dann:
1. Sperren `range_pos_too_high`/`gain_too_high` die Gewinner aus? (#22-Frage)
2. **Hat der Kanal ueberhaupt einen Edge?** (diese Frage) — wenn n>=20 und WR/avg klar
   unter den Scanner-BREAKOUTs liegen: Slot-Reserve reduzieren oder Kanal abschalten.

**Nicht vorher anfassen** — n=8 rechtfertigt weder Ausbau (EU-Bucket #23b) noch Abbau.
