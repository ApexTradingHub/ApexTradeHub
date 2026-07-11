# Claude-Code-Brief — Signal-Qualität, Score-Rekalibrierung & Rescue-Pfad-Fix

**Erstellt:** 2026-07-10 (Research-Analyse-Session, Desktop-Claude)
**Für:** Claude Code Agent (Projektleitung ApexNext)
**Datenstand der Zahlen:** 2026-07-10 ~18:00 UTC — `apex_positions.json` (52 closed),
`apex_equity_results.json` (205 Trades), `apex_signals.json` (201 Signale).
**⚠ Wenn seit Erstellung Tage vergangen sind: Zahlen mit dem Join-Rezept (§5) neu rechnen, bevor du handelst.**

**Pflicht-Lektüre vorab (Projekt-Konvention):** `CONTEXT.md` (Live-Config, Konstanten),
`BACKLOG.md` (#12–#15 direkt relevant). Goldene Regeln gelten uneingeschränkt:
Backtest-First, Ein-Knopf-pro-Lauf, Anti-Cherry-Pick (Akzeptanzkriterien in DIESEM
Dokument sind vorab fixiert — nicht nachträglich anpassen), Signal-Protection
(Signal-Count ≥95 % Baseline oder Loser-Anteil-Nachweis).

---

## 0. TL;DR

Forensik vom 2026-07-10 hat vier verifizierte Befunde produziert:

1. **KRITISCH — Rescue-Pfad kontaminiert die Bücher:** Die EOD→Swing-Konvertierung
   überschreibt `source` auf `"momentum_filler"` (apex_trader.py, Funktion
   `update_open_positions`, Rescue-Block — heute Zeile ~1474). Die 9 konvertierten
   Trades haben WR 22 % und verstecken sich in der Filler-Statistik. Echter Filler:
   WR 58,3 % (gesund). INTRADAY ehrlich gerechnet: WR 41,2 % statt ausgewiesener 62,5 %.
   **Nachtrag (Abend-Session, §1.2b):** Buchführungs-Fix bleibt Pflicht, aber die
   Policy selbst NICHT kippen — inklusive des noch OFFENEN PAY-Runners (+20,5 %,
   ein Rot-Rescue!) ist die Rot-Rescue-Bilanz netto POSITIV (+2,2pp/Trade).
   Der Hebel ist der Stop (Variante G3 in AP1), nicht das Schließen.
2. **Score ist als Ranking rangwertlos:** Spearman Score→PnL = **−0.066** (n=137
   BREAKOUT-Trades). Dazu Score-Inflation: Median-Score März 85,7 → Juni 115,3.
   ~63 von ~160 Score-Punkten liegen auf Komponenten mit null/negativem prädiktivem Wert.
3. **Score-Sektor-Tilt blockiert Diversifikation:** Tech avgScore 114 / WR 48 %,
   Healthcare avgScore 99,5 / WR 85 %. Top-Score-Picking kauft systematisch den
   schwächsten Sektor. Score-Fix = Diversifikations-Fix.
4. **30d-WR-Einbruch zerlegt:** Tech-Cluster (WR 9,1 %, Summe −45,2 % im 30d-Slice)
   + Score-Inflation (Masse der Juni-Signale in den toxischen Buckets 110+)
   + Rescue-Pfad im Paper-Buch. Exec/eToro: Beifang, kein Treiber.

**Arbeitspakete in Reihenfolge:** AP1 (Rescue-Fix, ~1–2 h) → AP2 (SCORE_V2, ~1 Tag)
→ AP3/AP4 als Beifang → AP5 nur nach AP2.

---

## 1. Verifizierte Befunde (alle selbst gerechnet, reproduzierbar via §5)

### 1.1 WR-Trend, zwei Populationen getrennt

| Fenster | Paper-Ledger (n=52 closed) | Signal-Ebene (equity_results, n=205) |
|---|---|---|
| Lifetime | 57,7 % WR, PF 1,70 | 44,4 % WR, PF 1,56 |
| 30d | 58,0 % (n=50), PF 1,84 | **35,6 % (n=45), PF 0,87** |
| 14d | **38,1 % (n=21), PF 0,97** | **10,0 % (n=10)** |
| 7d | 40,0 % (n=15) | n=1 |

### 1.2 Befund A — Rescue-Pfad (KRITISCH, AP1)

Code: `apex_trader.py`, Rescue-Block in der Intraday-Exit-Logik (Branch
`if p.get("source") == "intraday_momentum":`, EOD-Zweig). Seit 2026-06-26 werden
ALLE überlebenden Intradays am EOD konvertiert (grün → Breakeven-Stop, rot → −4 %-Stop),
dabei: `p["source"] = "momentum_filler"`, `p["setup"] = "MOMENTUM"`, Flag
`p["intraday_rescued"] = True`. Der Intraday-Score (eigene Skala, Werte 20–40)
bleibt stehen — daher die mysteriösen „Sub-60-MOMENTUM-Trades" (Floor
`MOMENTUM_MIN_SCORE=60` wurde nie verletzt, die Trades sind keine Filler).

Dekontaminierte Zahlen (Flag `intraday_rescued`):

| Kohorte | n | WR | PF | Summe |
|---|---|---|---|---|
| Echter Momentum-Filler | 12 | **58,3 %** | 1,54 | +12,2 % |
| Intraday-Rescues | 9 | **22,2 %** | 0,63 | −4,6 % |
| INTRADAY ausgewiesen | 8 | 62,5 % | 3,16 | +13,9 % |
| **INTRADAY ehrlich (inkl. Rescues)** | 17 | **41,2 %** | **1,48** | +9,2 % |

Rescue-Ticker: FROG, HSAI, AYI, VERX, RIVN, MRCY, FRSH, NKTR, PLTR (alle geöffnet
2026-06-26 bis 07-06). 13 der 21 Paper-Closes der letzten 14 Tage waren „MOMENTUM" —
davon 9 Rescues. Der Rescue-Pfad ist der größte aktive Bleeder unter den CLOSED-Trades —
**aber siehe §1.2b, bevor du die Policy anfasst.**

### 1.2b Nachtrag — Event-Level-Retro (AP1 Schritt 1, bereits ausgeführt 2026-07-10 Abend)

Alle 10 `intraday_to_swing`-Events aus `apex_trade_log.json`, gejoint mit finalem Outcome:

| Ticker | conv-PnL | Modus | final PnL | Exit | Δ (final−conv) |
|---|---|---|---|---|---|
| AYI | −0,56 % | rot→−4 % | −0,72 % | Stagnation | −0,16 |
| **PAY** | **−0,55 %** | **rot→−4 %** | **+20,49 % OFFEN** | Runner, Ladder 3 | **+21,04** |
| FROG | +0,02 % | grün→BE | −0,00 % | Stop (BE) | −0,02 |
| FRSH | −0,30 % | rot→−4 % | +1,03 % | Time Exit | +1,33 |
| HSAI | +0,55 % | grün→BE | +0,00 % | Stop (BE) | −0,55 |
| MRCY | −0,35 % | rot→−4 % | −4,00 % | Stop Loss | −3,65 |
| VERX | −0,50 % | rot→−4 % | −4,00 % | Stop Loss | −3,50 |
| PLTR | −1,84 % | rot→−4 % | −4,00 % | Stop Loss | −2,16 |
| RIVN | +1,79 % | grün→BE | +0,00 % | Stop (BE) | −1,79 |
| NKTR | +3,82 % | grün→BE | +7,04 % | order_dropped | +3,22 |

**Kernzahlen:**
- Rot-Rescues, nur closed (n=5): Δ **−8,14pp** gesamt ≈ −1,63pp/Trade → sah nach „kippen" aus.
- Rot-Rescues **inkl. offenem PAY** (n=6): Δ **+12,9pp** ≈ **+2,15pp/Trade → Policy ist netto positiv.**
- Grün-Rescues (n=4): Δ +0,86pp ≈ Wash (NKTR-Runner zahlt für 3 Breakeven-Stops). Unverändert lassen.

**⚠ Lehre für alle künftigen Retro-Messungen (Zensur-Falle):** Die ursprüngliche
AP1-Entscheidungsregel war auf CLOSED-only definiert — mit dem größten Gewinner noch
offen hätte sie die Policy fälschlich gekillt. **Offene Positionen müssen in jede
Exit-Policy-Bewertung rein (mark-to-market), sonst systematischer Bias gegen Runner.**

**Mechanik-Diagnose:** Das Verlustprofil entsteht nicht durch das Rescuen, sondern
durch den **−4 %-Stop ab Entry**: 3 der 5 closed Rot-Rescues (MRCY/VERX/PLTR) waren
bei Konvertierung flach-rot (−0,35/−0,50/−1,84) und bluteten dann die volle Distanz
bis −4 % aus. Turner (PAY/FRSH) und Bleeder (MRCY/VERX) sind über die conv-PnL-Tiefe
NICHT trennbar (FRSH −0,30 ≈ MRCY −0,35) — ein reines Tiefe-Gate funktioniert nicht.

### 1.3 Befund B — Score rangprädiktiv wertlos + inflationär (AP2)

- **Spearman Score→PnL = −0.066** (n=137 gejointe BREAKOUT-Trades, März–Juli 2026).
- Quartile: Q1 (Score 60–87) WR 38,2 % | Q2 (87–103) **70,6 %** | Q3 (103–115) 58,8 %
  | Q4 (115–161) **37,1 %**. Umgekehrtes U, Sweet-Spot **90–110** (nicht 110–130).
- Buckets lifetime: <90: 40,5 % (n=37) | 90–110: **69,8 %, avg +5,50 %** (n=43) |
  110–130: 52,3 % (n=44) | 130+: 15,4 % (n=13).
- **Score-Inflation:** Median BREAKOUT-Score pro Monat: März 85,7 → April 91,1 →
  Mai 111,0 → Juni 115,3 → Juli 102,4. Absolute Schwellen (TG_MIN_SCORE,
  REPLACEMENT_MIN_SCORE=90, Bucket-Erwartungen) sind nicht stationär.

**Komponenten-Autopsie** (Gewichte: BREAKOUT-Block in `ApexScan.py` ~Z. 1307–1320
+ movement_class ~Z. 1205–1219 + Sektor-Bonus ~Z. 1356–1360; Lifts aus dem Join n=137):

| Komponente | Max-Punkte | Gemessener Lift (WR) | Urteil |
|---|---|---|---|
| gap ≥2 % | +8 | **+17,9pp** (n=18) | trägt, eher untergewichtet |
| vol_ratio ≥1,5 | ~12 | +10,8pp (n=32) | trägt |
| pocket_pivot | +10 | +6,8pp (n=62) | trägt, klein |
| rsi ≥70 | (+6 Zone) | +20pp (n=13, klein) | Richtung stimmt |
| **rr × 4** | **20** | rr<2,0 ist **+13,7pp BESSER** | **falsches Vorzeichen** |
| **perf-Stack 20/60/120** | **~43** | perf_20>10: −3,8pp · POWER: −8,3pp | null bis negativ |
| macd_bull | +8 | 130/137 Signale haben es | keine Diskriminierung |
| risk_on | +5 | für alle Signale eines Tages gleich | reine Inflations-Komponente |
| Sektor-Bonus | ±6 | falsifiziert (BACKLOG #12) | steht noch im Code |
| WEAK-Penalty (movement) | −15 | WEAK real **+22,3pp** (n=17) | im Sample invertiert — siehe Caveat |

**Caveat (wichtig, nicht überspringen):** WEAK +22,3pp widerspricht dem
2-Jahres-Backtest (41 % WR). Das Join-Sample ist EIN Regime (März–Juli 2026).
**Vorzeichen NICHT flippen** — nur Gewichte der Momentum-Terme reduzieren.
Gleiches gilt für rsi≥70 (n=13) — Richtung notieren, nicht überdrehen.

### 1.4 Befund C — Score-Sektor-Tilt = Diversifikations-Blocker (AP2 + AP5)

| Sektor | avg Signal-Score | reale WR (equity) |
|---|---|---|
| Technology | **114,0** (n=38 Signale) | 48,4 % (n=31) |
| Basic Materials | 108,8 | **75,0 %** (n=8) |
| Industrials | 108,1 | 44,0 % (n=25) |
| Healthcare | **99,5** | **84,6 %** (n=13) |
| Energy | 98,3 | 50,0 % (n=20) |
| Consumer Cyclical | 97,1 | 50,0 % (n=24) |

Der Score rankt invers zur realisierten WR über Sektoren. Telegram-Top-2 und
Trader-Slot-Füllung picken nach Score → strukturelle Tech-Übergewichtung,
Healthcare/BasicMat werden verdrängt. **Kein neues Setup nötig — der Score-Fix
ist der Diversifikations-Fix.**

Paper-Ledger: 29/52 = 56 % Sektor „Unknown" — reines Labeling: Filler und
Intraday setzen hardcoded `"sector": "Unknown"` (apex_trader.py ~Z. 645 bzw. ~992),
obwohl `sector_cache.json` existiert. → AP3.

### 1.5 Befund D — Zerlegung des 30d/14d-Einbruchs (kein weiterer AP nötig, nur Monitoring)

1. **Tech-Cluster (Signal-Ebene):** 30d-Slice: Technology WR 9,1 % (n=11),
   Summe −45,2 % — ohne Tech wäre der 30d-Slice **positiv** (+25,7 %). Dazu
   ConsDef 0/3 (−20,2 %). Ursache: Semi-Selloff 23./24.06. Das TECH_QQQ_GATE
   (live 07-08) adressiert exakt diese Zelle, kam aber zu spät für die Juni-Trades
   und feuert nur bei `qqq_perf_20 < 0` — **QQQ ist aktuell wieder STRONG, das
   Gate ist schlafend.** Monitoring-Auftrag: Wenn Tech-BREAKOUTs bei QQQ>0
   weiter unterperformen, ist das eine NEUE Zelle, die das Gate nicht deckt.
2. **Score-Inflation:** 30d-Verluste konzentriert in Bucket 110–130 (WR 33 %,
   n=21, Summe −29 %) + 130+ (WR 14 %, n=7) = 28 von 43 BREAKOUT-Exits.
3. **Rescue-Pfad** (§1.2) im Paper-Buch.
4. **Exec/eToro:** 3 eToro-SL + 2 order_dropped von 21 Closes in 14d — Beifang.
   Slippage-These war mechanisch schon durch Fix A widerlegt (SL wird prozentual
   auf den Ask rebased, Distanz bleibt erhalten).

### 1.6 Befund E — EU-Universe liefert 0 Signale (AP4)

`eu_tickers.txt` (106 Zeilen, Format `SAP.DE`) wird geladen und übersteht
`clean_universe()` (kein Punkt-Filter drin). Trotzdem: **0/201 Signalen und
0/205 Trades sind EU.** Die Pipeline hat noch NIE ein EU-Signal produziert —
Root-Cause unbekannt (Kandidaten: `avg_dv_m`-Liquiditätsschwelle, NaN-Slices im
gemischten yf-Batch-Download, Filter-Kaskade). Entweder toter Ballast oder stiller Bug.

### 1.7 Nebenbefunde (dokumentieren, NICHT handeln)

- **analyst_upside ist instabil:** Knowledge-Base sagte −19pp (n=20), heutiger Join
  sagt **+11,5pp** (n=13) und analyst_upside<5 sogar −15,7pp (n=40). Das Merkmal
  flippt je nach Sample. **Kein Score-Change in keine Richtung** — als instabil in
  der Watchlist markieren, erst bei n≥40 konsistent neu bewerten.
- **Learn-Report-Datenbruch:** `reports/learn_latest.md` (07-09) zeigt
  `Market: UNKNOWN — Market filter unavailable -> defaulting to RISK-OFF`.
  Der Regime-Join in `apex_learn.py` ist kaputt → vor dem nächsten Sa-Cron fixen,
  sonst ist die nächste Regime-Auswertung Müll. (Mini-Fix, kein eigenes AP.)
- MOMENTUM-Filler-Floor 60 funktioniert korrekt; Trending-Quelle läuft durch
  dieselbe Loop. Kein Handlungsbedarf am Filler selbst.

---

## 2. Arbeitspakete (Akzeptanzkriterien VORAB fixiert — Anti-Cherry-Pick)

### AP1 — Rescue-Pfad: Retro-Messung + Fix (PRIO 1, ~1–2 h)

**Ziel:** Den Bleeder stoppen und die Buchführung ehrlich machen.

**Schritt 1 — Retro-Messung: ERLEDIGT (Ergebnis in §1.2b).**
Die ursprüngliche Entscheidungsregel („Rot-Rescues ≥1pp schlechter → rote am EOD
schließen") ist **SUPERSEDED** — sie war auf closed-only definiert und hätte wegen
des offenen PAY-Runners (+20,5 %) falsch entschieden (Zensur-Falle, §1.2b).
**Neue Entscheidung: Policy behalten** („kein hartes Banken", User-Wunsch 06-26,
empirisch bestätigt: Rot-Rescues inkl. PAY +2,15pp/Trade). Der Fix zielt auf den
Bleed-Mechanismus, nicht auf das Rescuen:

**Schritt 1-NEU — G3-Retro-Simulation: Tagestief-Stop statt entry−4 % (vor Code-Change):**
Rot-Rescues bekommen statt `stop = entry × 0.96` künftig
`stop = max(entry × 0.96, tagestief_conv_tag × 0.995)` (max auf Preisebene = engerer Stop).
- **Warum das die „verifiziert noch turnen"-Frage operationalisiert:** Die
  Recovery-These eines roten Intraday-Trades ist „hält das heutige Tief". Schließt
  der Trade AM Tagestief (Spike-Fade), liegt der Stop ≈ EOD-Kurs → konvergiert von
  selbst gegen Hard-Close, genau für die schlechtesten Fälle. Schließt er ÜBER dem
  Tief (Turn-Evidenz), bekommt er Raum — aber nur bis zur Falsifikation der These
  (Tief unterschritten), nicht pauschal 4 %.
- **Simulation (nur Daily-Bars nötig, kein 60d-Limit):** Für die 6 Rot-Rescues
  (AYI/PAY/FRSH/MRCY/VERX/PLTR): Tagestief des Konvertierungstages holen, Pfad
  forward mit Daily-Lows/Highs simulieren (Momentum-Exits: TP +6 %, Hold 7d,
  Ladder), hypothetisches Outcome vs. reales Outcome.
- **Akzeptanz (vorab fixiert):** GO für G3 nur wenn ALLE drei gelten:
  (a) **PAY und FRSH überleben** den Tagestief-Stop (Turner werden nicht gekillt),
  (b) MRCY/VERX/PLTR-Verluste verbessern sich im Mittel um ≥1,5pp,
  (c) Netto-Δ der 6 Trades ≥ Status quo (+12,9pp).
  **Wenn PAY unter G3 stirbt → G3 verwerfen**, −4 %-Stop behalten: der breite Stop
  ist dann der nachgewiesene Preis der Runner-Option. Ergebnis so oder so in
  BACKLOG dokumentieren.
- n=6 ist klein — Paper-Modus, aggressive Iteration OK. Nach Änderung: Eval bei
  n≥10 neuen Rot-Konvertierungen gegen dieselben Kriterien.

**Schritt 1b — Shadow-Logging der EOD-Features (immer, unabhängig von G3):**
Im `intraday_to_swing`-Event zusätzlich loggen: `range_pos_eod` (Position in
Tagesspanne), `above_vwap_eod` (bool), `high_since_entry_pct`, `dist_to_day_low_pct`,
`pnl_now`. Kosten: ~5 Zeilen (die 5m-Bars liegen im Intraday-Scan eh vor).
**Policy während der Sammlung NICHT zusätzlich ändern** — bei Alles-Rescuen werden
BEIDE Zweige real beobachtet, das Kontrafaktische ist gratis. Bei n≥15 neuen
Konvertierungen: prüfen, ob ein Feature-Gate (Kandidat: `range_pos_eod < 0.4` =
Spike-Fade, deckt sich mit CONFIRMED-Pattern `closing_strength < 0.5` aus CONTEXT §9)
Turner von Bleedern trennt. Erst dann ggf. Gate — vorher nicht.
- Optional: 5m-Retro der 10 Alt-Events (yfinance hält 5m-Bars nur ~60 Tage —
  AYI vom 26.06. verfällt ~Ende August; wenn gewünscht, zeitnah ziehen).
- **Population-Shift beachten:** Die Anti-Peak-Filter (BACKLOG #15, live 07-10)
  verändern die Rot-EOD-Kohorte (weniger Peak-Käufe). Shadow-Daten ab 07-10
  spiegeln die neue Population — noch ein Grund für Messen-statt-Kippen.

**Schritt 2 — Source-Rewrite entfernen (immer, unabhängig von Schritt 1):**
- In der Konvertierung `p["source"] = "momentum_filler"` **löschen**. `source` bleibt
  wahrheitsgemäß `"intraday_momentum"`.
- **Achtung Seiteneffekt:** Die Intraday-Exit-Logik brancht auf
  `p.get("source") == "intraday_momentum"`. Damit konvertierte Positionen nicht
  wieder in die Intraday-Exits laufen, Branch-Bedingung erweitern:
  `if p.get("source") == "intraday_momentum" and not p.get("intraday_rescued"):`
- Ebenso prüfen: Slot-Zählung (`swing_open` zählt `source != "intraday_momentum"`)
  — konvertierte Positionen belegen Swing-Slots, also dort auf
  `source != "intraday_momentum" or intraday_rescued` umstellen. **Alle
  `source`-Abfragen im File durchgehen (grep `intraday_momentum`), bevor committed wird.**
- `setup = "MOMENTUM"`-Umschreibung kann bleiben (Management-Logik), das
  `intraday_rescued`-Flag ist der Attributions-Anker für alle Auswertungen.

**Schritt 3 — Einmalige Datenreparatur:**
Repair-Skript: in `apex_positions.json` → `closed[]` bei den 9 Rescue-Trades
(erkennbar an `intraday_rescued: true`) `source` zurück auf `"intraday_momentum"`
setzen. Danach Auswertungen (Dashboard/Learn) prüfen, ob sie nach source gruppieren.

**Rollback:** Policy-Änderung = eine Bedingung im EOD-Zweig, trivial revertierbar.
**Signal-Protection:** nicht berührt (Trader-Exit-Policy, kein Scanner-Filter).
**Beifang:** AP3 im selben Commit miterledigen.

---

### AP2 — SCORE_V2: Offline-Rekalibrierung (PRIO 2, ~0,5–1 Tag, dreistufig)

**Ziel:** Score wieder rangprädiktiv machen. Reines Re-Ranking — **kein Signal wird
geskippt, Signal-Count bleibt identisch. Signal-Protection per Konstruktion erfüllt.**

**Stufe 1 — Join-Replay (Offline-Skript, kein Live-Code):**
- Join-Rezept aus §5. Baseline: geloggter `score` aus `apex_signals.json`.
- **V2a (Ballast raus, KEIN Fit — Anker-Variante):** Score neu berechnen aus den
  geloggten Feldern mit diesen Änderungen: rr-Term (rr×4) raus, macd_bull-Bonus raus,
  risk_on-Bonus raus, Sektor-Bonus raus (deckt auch BACKLOG #12 ab),
  perf-Stack-Gewichte halbieren (perf_20×0,4 / perf_60×0,25 / perf_120×0,1).
  movement_class-Buckets UNVERÄNDERT lassen (2-Jahres-validiert). Catalyst-Boni
  (PP/gap/vol_climax) unverändert.
- **V2b (Fit-Variante):** Logistic Regression (Outcome win/loss) auf den
  verfügbaren Komponenten, **trainiert NUR auf Signale mit date ≤ 2026-05-31**,
  getestet auf Juni–Juli (n≈50 OOS).
- **Perzentil-Normalisierung:** beide Varianten zusätzlich als Cross-Sectional-Rank
  pro Scan-Tag evaluieren (behebt die Inflations-Komponente).
- **Ehrlichkeits-Fußnote (in den Report schreiben):** Die Komponenten-Auswahl für
  V2a hat das volle Sample gesehen (diese Analyse) — Juni-Juli ist für V2a nur
  „quasi-OOS". Für V2b ist der Walk-Forward sauber. Deshalb ist Stufe 2 Pflicht.

**Akzeptanz Stufe 1 (vorab fixiert):**
- OOS-Spearman (Juni–Juli) ≥ **+0,15** (Baseline: −0,066) UND
- Top-Quartil-WR (nach V2-Ranking) ≥ Baseline-Top-Quartil-WR **+5pp**.
- Wenn V2a ≈ V2b (±0,05 Spearman): **V2a nehmen** (weniger Overfitting-Fläche).
- Wenn beide failen: verwerfen, als BACKLOG-Eintrag dokumentieren, KEIN Live-Change.

**Stufe 2 — 2-Jahres-Backtest (nur wenn Stufe 1 GO):**
- V2 als Flag `--score-v2` in `apex_backtest_v2.py` implementieren
  (Konvention wie `--score-realign` / `--score-rebuild`).
- 2 Jahre laufen lassen, Vergleich Baseline vs. V2: (a) WR/PF gesamt,
  (b) Simulation der Pick-Stufe (Top-2/Tag bzw. Trader-Top-Score): WR/PF der
  GEPICKTEN Trades, (c) Sektor-Verteilung der gepickten Trades (erwarteter
  Nebeneffekt: weniger Tech-Konzentration).
- **Akzeptanz Stufe 2 (vorab fixiert):** Signal-Count identisch (Re-Ranking!),
  gepickte-Trades-WR ≥ Baseline +2pp, PF ≥ Baseline. Ein-Knopf-Regel beachten:
  V2a und V2b als GETRENNTE Läufe, nicht mischen.

**Stufe 3 — Live-Port (nur wenn Stufe 2 GO):**
- `ApexScan.py` Score-Block anpassen, Flag-gated (`SCORE_V2_ENABLED`), Rollback = Flag.
- **Pflicht-Nacharbeit:** Absolute Schwellen auf Perzentil-Äquivalente mappen —
  `TG_MIN_SCORE` (Telegram-Gate), `REPLACEMENT_MIN_SCORE=90`. Sonst bricht das
  Gate, sobald die Score-Verteilung sich verschiebt (genau das Problem, das wir fixen).
- `CONTEXT.md` §2 + Konstanten-Tabelle aktualisieren, `SCORE_V2_STRATEGY.md` anlegen
  (Konvention wie `SCORE_REBUILD_STRATEGY.md`).

**Worst-Case:** Overfitting auf ein Regime → dagegen stehen Walk-Forward (Stufe 1),
2-Jahres-Backtest (Stufe 2) und V2a-als-Anker. Wenn V2 im 2-Jahres-Test nicht
generalisiert, ist die Antwort „verwerfen und dokumentieren" — nicht „Kriterien lockern".

---

### AP3 — Sektor-Enrichment für Filler + Intraday (Beifang zu AP1, ~30 min)

**Ziel:** 56 % „Unknown" im Paper-Buch eliminieren. Reine Instrumentierung, kein Backtest.
- In `apex_trader.py`: `sector_cache.json` laden (Pattern existiert im Scanner),
  bei Kandidaten-Erstellung Momentum (~Z. 645) und Intraday (~Z. 992)
  `sector_cache.get(ticker, "Unknown")` statt hardcoded `"Unknown"`.
- Einmalig closed-Trades backfillen (Repair-Skript, gleiche Lookup-Logik).
- Fallback bleibt „Unknown" (Cache ist selbstheilend seit 06-01).
**Wert:** Ohne das bleibt jede künftige Sektor-Auswertung (inkl. AP5) auf halbem Buch blind.

---

### AP4 — EU-Universe-Diagnose (1 Scan-Lauf, ~1 h)

**Ziel:** Klären, warum 107 EU-Ticker in 4 Monaten 0 Signale produziert haben.
- Diagnose-Skript (oder Debug-Flag in `ApexScan.py`): für die EU-Kohorte
  **per-Filter-Drop-Counter** loggen — wie viele der 107 sterben an: Download/NaN,
  `avg_dv_m`-Schwelle, base_range, RSI-Zone, vol_ratio, Rest.
- **Erst diagnostizieren, dann entscheiden:**
  - Bug (z. B. NaN im Batch-Download) → fixen. Jedes entstehende EU-Signal ist
    Netto-PLUS beim Signal-Count (Signal-Protection freut sich).
  - Legitime Filter (z. B. Liquidität) → Entscheidung an User: EU-spezifische
    Schwelle (Vorsicht: neue Stellschraube) ODER `eu_tickers.txt` aus dem Scanner
    nehmen (tote Konfiguration raus; Intraday nutzt das File separat weiter —
    BACKLOG #14-Kontext beachten).
- **Kein Edge-Versprechen:** Ob EU-Signale performen, weiß niemand — es gab nie eins.
  Falls nach Fix EU-Signale entstehen: 30 Signale sammeln, dann getrennt auswerten
  (eToro-Handelbarkeit der EU-Namen vorher prüfen, sonst Paper-only taggen).

---

### AP5 — Soft-Diversity-Nudge (NACH AP2, optional, ~2–3 h)

**Ziel:** Sektor-Cluster-Risiko senken ohne die falsifizierte harte Sektor-Cap
(BACKLOG #3/#13: Cap droppte Winner, 0pp Lift).

**Mechanik:** Beim Slot-Füllen, wenn mehrere qualifizierte Signale konkurrieren:
bevorzuge das Signal aus einem NICHT im Buch vertretenen Sektor, **sofern
Score-Differenz ≤10 Punkte**. Kein Signal wird gedroppt — nur Reihenfolge.
Datenbasis: Bei Spearman −0,07 (bzw. auch nach V2 zu prüfen) tragen 10
Score-Punkte Differenz keine Information — der Nudge kostet statistisch ~nichts.

**Warum NACH AP2:** Wenn V2 funktioniert, trägt der Score wieder Information —
dann muss die 10-Punkte-Schwelle gegen die V2-Verteilung neu begründet werden
(z. B. „innerhalb desselben V2-Perzentil-Dezils").

**Backtest (vorab fixierte Akzeptanz):** Replay der Trader-Slot-Füllung über die
Signal-Historie mit/ohne Nudge (Join-Daten aus §5). GO wenn: Portfolio-MaxDD −20 %
oder besser UND WR-Änderung innerhalb ±2pp. **Worst-Case ist real und bekannt:**
In Sektor-Trend-Phasen (März-Energy-Cluster: 9 simultane Wins) kostet der Nudge
Gewinne — genau das muss das Replay beziffern, bevor gebaut wird.

---

## 3. Was NICHT tun (Falsifikations-Gedächtnis + heutige Erkenntnisse)

- **Keine neuen Setup-Typen jetzt** (PEAD/Earnings-Drift ist der einzige spätere
  Kandidat — erst wenn AP1+AP2 verdaut sind). MEAN_REVERSION bleibt zu.
- **WEAK-Penalty-Vorzeichen NICHT flippen** (n=17, ein Regime, widerspricht 2J-Backtest).
- **analyst_upside NICHT anfassen** — in beide Richtungen instabil (§1.7).
- Bereits falsifiziert, nicht wieder vorschlagen: Score-Cap 130+, harte Sektor-Cap,
  Extension-Filter, Broad-Regime-Gate, VIX/Macro-Gate, Inverse-ETF, Sektor-RS-Bonus.
- **Keine Filter-Änderung ohne Signal-Count-Nachweis** (≥95 % Baseline oder
  Loser-Anteil-Rechtfertigung). AP2 und AP5 sind bewusst als Re-Ranking konstruiert.
- Trader-Cron committet alle 5–15 min State-Files — bei Git-Konflikten auf
  Auto-Gen-Files: `git checkout HEAD -- <file>`, nie manuell mergen (CONTEXT.md §1).

---

## 4. Reihenfolge & Abhängigkeiten

```
AP1 (Rescue-Fix)  ──┬─→ AP3 (Sektor-Enrichment, selber Commit)
                    └─→ danach: 2 Wochen Monitoring der EOD-Policy
AP2 (SCORE_V2)    ── Stufe 1 → Stufe 2 → Stufe 3 (jede Stufe hartes Gate)
                    └─→ AP5 (Nudge) erst nach AP2-Entscheid
AP4 (EU-Diagnose) ── unabhängig, irgendwann diese Woche
Mini-Fix: apex_learn.py Regime-Join (vor Samstag-Cron 06:47 UTC)
```

Empfohlener Einstieg für die erste Session: **AP1 komplett + AP3 + Learn-Mini-Fix**
(ein Commit-Block, ~2–3 h), dann AP2 Stufe 1 als eigene Session.

---

## 5. Reproduzierbarkeit — Join-Rezept

Alle Zahlen dieses Briefs entstehen aus diesem Join (Python, Repo-Root):

```python
import json
sigs = json.load(open("apex_signals.json", encoding="utf-8"))      # Features
eqt  = json.load(open("apex_equity_results.json", encoding="utf-8"))  # Outcomes

eq_by_key = {(t["ticker"], str(t["date"])[:10]): t for t in eqt}
joined = [(s, eq_by_key[(s["ticker"], str(s["date"])[:10])])
          for s in sigs
          if (s["ticker"], str(s["date"])[:10]) in eq_by_key
          and eq_by_key[(s["ticker"], str(s["date"])[:10])].get("pnl_pct") is not None]
# Stand 07-10: 138 Paare, davon 137 BREAKOUT.
# Outcome: t["pnl_pct"]; Win = pnl_pct > 0.
# Paper-Ledger separat: apex_positions.json -> closed[] (Felder: setup, source,
# score, sector, pnl_pct, exit_reason, closed_at, intraday_rescued).
```

Metriken: WR = Anteil pnl_pct>0; PF = Summe Gewinne / |Summe Verluste|;
Spearman handgestrickt über Ranks (kein scipy nötig). Fenster-Slices über
`closed_at` (Ledger) bzw. `date + exit_day` (Signal-Ebene).

**Wichtigste Populations-Falle:** Paper-Ledger (Trader-Exits: Ladder/Stagnation)
und Signal-Ebene (simulierte TP/SL/Time-Exits) NIE mischen — sie erzählen
verschiedene Geschichten (30d: 58 % WR vs. 35,6 % WR). Immer beide ausweisen.

---

*Ende des Briefs. Bei Widersprüchen zwischen diesem Dokument und dem Code:
Code ist autoritativ, Zeilennummern via grep verifizieren (Trader wurde am
07-10 mehrfach committed). Bei Fragen zur Herleitung: Alle Befunde sind mit
§5 in <5 min nachrechenbar.*
