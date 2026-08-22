# ApexNext — Session Context & Active State

**Zweck:** Persistenter Schnappschuss des aktuellen Stands. Wenn das Chat-Kontextfenster
komprimiert wird, kann eine neue Session diese Datei lesen und **kalt aufgreifen** ohne den
ganzen Verlauf zu kennen. Wird laufend aktualisiert.

**Letztes Update:** 2026-08-22 (**Rescue nur noch gruen** (rot am EOD zu, +20.4pp) · **Zerlegung: Rohkante nur +1.3%/Trade, Auswahl -0.9, Management -0.9** -> weniger Mechanik · seit eToro-Aus PF 1.40 ueber n=31 · WR ist KEINE Stellschraube (BACKLOG #28) · Regime wieder BEARISH)

---

## 1. Workflow-Goldene Regeln (NICHT verletzen)

- **WELCHE DATENQUELLE FUER WELCHE FRAGE (Fund 2026-08-07, wichtig):** Backtest und Live-/Equity-
  Daten messen NICHT dieselbe Population. Im identischen Fenster (17.03.-30.06.26): LIVE n=121 vs
  BACKTEST n=27; **Pocket-Pivot-Anteil 42% vs 78%, VCP>0 14% vs 56%, Score>=100 52% vs 26%**.
  Ursachen: (a) der Backtest scannt nur **96 Termine in 2 Jahren = ~woechentlich**, live scannen wir
  **taeglich**; (b) `apex_backtest_v2.py` hat **kein sector_momentum** -> TECH_QQQ_GATE und
  SECTOR_RS_GATE laufen dort NICHT.
  -> **BACKTEST = Mechanik-Pruefstand.** Nur fuer A/B DERSELBEN Trades unter verschiedenen Regeln
     (Haltedauer, Trailing-Leitern, Exit-Logik). Der Sampling-Unterschied kuerzt sich in beiden Armen weg.
  -> **LIVE/EQUITY = Wahrheit ueber den tatsaechlichen Signalfluss.** Fuer Kalibrierungsfragen
     (Score-Gates, Katalysator-Effekte, Sektor-Verhalten), weil dort alle Gates + die echte Kadenz drin sind.
  -> **NIE Katalysator- oder Score-Effekte zwischen den beiden Quellen vergleichen.** Genau das hat
     2026-08-06/07 zu zwei Schein-Widerspruechen gefuehrt (Gate 80 und Pocket Pivot).

- **CONFIRMED-only-Code-Änderungen**: Keine System-Änderungen aufgrund von TENTATIVE/MED-Findings.
  Brauche `n ≥ 30` UND einen klaren Effekt-Größenwert für Code-Changes an Filtern/Gates.
- **Anti-Cherry-Pick**: Akzeptanzkriterien vor jedem Tuning-Backtest fixieren, NICHT danach.
- **Backtest-First, Live-Second**: Neue Setups/Filter werden im Backtest validiert
  (`apex_backtest_v2.py` mit `--only-setup` Flag), bevor `ApexScan.py` (Live) angefasst wird.
- **Ein-Knopf-Pro-Lauf**: Beim Tuning nur eine Stellschraube pro Backtest, nicht bündeln.
- **Reports statt JSONs für State-Check**: Bei „wie steht's"-Fragen die `.md` aus `reports/`
  senden lassen, JSONs sind zu groß für Kontext. JSONs lese ich gezielt von Platte.
- **Auto-Gen-Files**: `apex_signals.json`, `apex_equity_*.json`, `apex_market.json`,
  `company_names.json`, `sector_cache.json` werden vom Cron geschrieben. Bei Git-Konflikten
  immer mit `git checkout HEAD -- <file>` resetten, NICHT manuell mergen. Cron regeneriert
  beim nächsten Lauf.
- **Telegram-/Scanner-Cron**: Scan läuft 20:30 UTC (22:30 CEST), NACH US-Close. Signale immer
  auf fertigem Tagesbar, Trigger immer am Folgetag.
- **Secrets**: TELEGRAM_TOKEN/CHAT_ID nur aus GitHub Secrets, KEIN hardcoded Fallback.

---

## 2. Aktive Setups (Live in `ApexScan.py`)

| Setup | Status | Charakteristik | Aktueller WR/PF (lifetime) |
|---|---|---|---|
| **BREAKOUT** 🔵 | CONFIRMED, **TECH_QQQ_GATE live 07-08** | 20d-High-Breakout | WR **57 % / PF 2.29** (n=119) · Nach Gate: WR **59.8 % / PF 2.53** |
| ~~STAGE_2~~ 🚀 Trend | **DISABLED 2026-07-08** (User: 0 Wins bisher) | Flag `STAGE_2_ENABLED = False` — Detektor bleibt für Re-Enable |
| **VCP** 🔹 Bounceback | aktiv (gelockert 28.5.) | Backtest WR 88.9 % n=9 |
| **SHORT_SQUEEZE** 🔥 Bet | aktiv (strict, ≥15 % short) | n=0 lifetime |
| ~~MEAN_REVERSION~~ 🟢 Dip | **DISABLED 2026-06-17** (User: "kaufe ich eh nicht") | n=4 30d-window, WR **0 %**, AvgLoss -2.70 %. Score weakly ANTI-predictive in-sample. Flag `MEAN_REVERSION_ENABLED=False` in ApexScan.py. Code-Pfad bleibt fuer Re-Enable. |
| **MOMENTUM** ⚡ (Filler) | **NEU 06-12** Paper-Trader-only · **BEARISH-Skip live 07-08** (14d WR 30%) | Lifetime WR 41 %, PF 1.29 (n=17) |
| **INTRADAY** ⚡⚡ Catcher | **NEU 06-18 EXPERIMENT, opt-in `INTRADAY_ENABLED=1`** Paper-only | n=0 (Test, default OFF) |
| ~~REVERSAL~~ | **disabled** | Legacy, 28 % WR strukturell defekt | sterbende Legacy-Positions in n |

**INTRADAY-Catcher (EXPERIMENT, User-Wunsch 2026-06-18, default OFF):**
- Code in `apex_trader.py` (Step 3c). Scant die ~50 Daily-Momentum-Kandidaten auf
  INTRADAY-Momentum (5m-Bars heute): gain_from_open 1.5-6 %, über VWAP, oberer Teil
  Tagesspanne (range_pos ≥0.55). Direkter Market-Entry (kein pending/trigger).
- Exit: TP **+5 %**, Stop **-3 %**, Hard-Close ab **19:45 UTC** (same-day, kein Overnight).
- Sub-Limit **max 3** gleichzeitige Intraday-Plays, $50/Pos, eigene Exit-Logik (KEIN
  Ladder/Stagnation). Source-Tag `intraday_momentum`, setup `INTRADAY`, Exit-Reasons
  `Intraday TP/Stop/Close (EOD)` → sauber separat auswertbar.
- **Aktivierung VM:** `export INTRADAY_ENABLED=1` in ~/.bashrc + Cron `*/15`→`*/5` für
  schnellere Kadenz. Default OFF = ändert Live-Trader nicht bis aktiviert.
- **Ziel/These (User):** ~$20/Tag durch schnelle 5%-Intraday-Spruenge + mehr Rotation.
- **Risiko bewusst:** MOMO-Profil (BACKLOG #2 = PF 1.51 verworfen). Experiment, Rollback
  = Flag auf 0. **Eval nach ~1-2 Wochen:** Intraday-Trades isoliert (source-Tag), bringt es
  netto + nach Würde-Fees? Falls nein → zurückbauen.

**BREAKOUT-Tuning aktuell (Stand 2026-06-15):**
- **RSI-Zone (REALIGN): 48-72** (war 48-68, +6 voll im erweiterten Bereich)
- **~~RSI 60-65 DEADZONE~~ — FALSIFIZIERT 2026-07-31, KEINE Action.** Der Juni-Fund (n=40, -10pp)
  hat sich **out-of-sample umgedreht**: seit 06-15 ist 60-65 **+4.6pp** (WR 44.1 vs Baseline 39.5),
  waehrend 65-70 von +5.4 auf **-7.0pp** kippte — Buckets tauschen die Plaetze = Rauschen/Regime,
  keine Struktur. Ueber alle 174 gejointen BREAKOUT-Trades: 60-65 = **-3.0pp**, 65-70 = **-3.3pp**
  (praktisch identisch -> es gibt gar kein 60-65-"Loch"). Echtes Muster ist die **U-Kurve**: Mitte
  (60-70) mittelmaessig, Raender gut (50-60 +12.5pp n=11, 70+ **+22.2pp** n=15).
  **KEINE Score-Penalty** (die frueher hier notierte "-5 nach Backtest" ist damit erledigt), weil:
  (1) Befund out-of-sample gedreht, (2) 60-65 ≈ 65-70, eine Strafe nur auf 60-65 waere willkuerlich,
  (3) **97% unserer Signale liegen in 60-70** -> jede Penalty dort trifft die Haelfte aller Signale
  und drueckt sie mit Gate 80 unter die Schwelle = Signal-Protection-Verstoss.
  **KERNEINSICHT: RSI ist in unserem Signalset fast eine KONSTANTE, keine Variable** — ein 20d-Hoch-
  Ausbruch erzeugt mechanisch RSI 60-70. Deshalb kann RSI kaum diskriminieren; nicht daran tunen.
  Die guten Raender haetten Edge, aber wir bekommen dort schlicht keine Signale (1 bzw. 0 in den
  letzten 10 Scan-Tagen). **Nicht erneut aufrollen ohne fundamental neue Evidenz.**
- **perf_120 Buckets (REALIGN):** <0 = -15 (WEAK), 0-25 = **-3 (DEADZONE)**, 25-50 = +15 (SWEET), >50 = +8
- **TECH_QQQ_GATE live (2026-07-08):** Skip BREAKOUT wenn Sektor Tech/Communication UND
  `market_regime.qqq_perf_20 < 0`. Backtest: Tech+QQQ<0 = WR 14%/PF 0.56 (n=7) → nach Gate
  WR 57.1→59.8%, PF 2.29→2.53, Signal-Loss 6%. Flag `TECH_QQQ_GATE_ENABLED = True`.
  Vorbehalt n=7 klein, Monitoring nötig.
- **SECTOR_RS_GATE live (2026-07-15, BACKLOG #20):** Skip Tech/Comm-BREAKOUT wenn der
  SEKTOR-ETF (XLK/XLC) `sector_momentum < 0` UND kein starker Catalyst (Earnings-Beat/
  PP+VolClimax/Gap≥5). Ergaenzt TECH_QQQ_GATE (deckt Sektor-Schwaeche die QQQ verpasst,
  z.B. Semi-Selloff). Backtest: kept-WR 51.4→54.5%, PF 1.77→2.10, Profit STEIGT, droppt
  10L/2W (83% Loser). Flag `SECTOR_RS_GATE_ENABLED = True`. Scan-Time = no-buy bis Sektor
  gruen (Re-Scan re-emittiert). Restkosten: CIEN +16.6 (13d-Beat, Carve verpasst).
- **SCORE_REBUILD live (2026-06-20):** Extension-Penalty **-12** für perf_120>50 OHNE starken
  Catalyst (Catalyst-Carve-Out: earnings_beat / analyst_upside>15 / PP+Vol-Climax / Gap≥5).
  Backtest 250d: Plateau WR(100+) 47→54 %, **Monotonie -15pp→-0pp**, alle 122 Signale erhalten,
  PF 1.53→1.60. EXT_PENALTY=12 = Sweep-Optimum. Trifft ASML-Juni-Profil (146→134), verschont
  Sweet-Zone (FLR perf_120 31 unberührt) + Semi/AI-Capex-Winner (what_to_replicate).
  Backtest-Flag `--score-rebuild --ext-penalty 12`, Live hardcoded. Doku: `SCORE_REBUILD_STRATEGY.md`.
- `base_range`: ≤22 BREAKOUT, ≤8 andere Setups (28.5.)
- `vol_ratio`: ≥1.0 (relax=0), ≥0.7 (relax=1)

**Telegram-Gate (post-2026-05-22 Fix):**
- Score-basiert, NICHT relax-basiert: `TG_MIN_SCORE` per Setup + RR ≥ 1.5 + Upside ≥ 8 %
- Top-2 nach Score über alle Setups (NICHT 2 pro Setup — Diskussion offen)
- Catalyst-Flags ⚡/📈/🎯/🔥/⚠ werden gerendert
- MR ist als Setup-Group im Telegram-Builder wired ("🟢 Dip / Mean-Reversion")

---

## 3. Watchlist: Findings approaching CONFIRMED (NICHT handeln, beobachten)

| Finding | n | Status | Lift | Action falls CONFIRMED |
|---|---|---|---|---|
| **⚡ Pocket Pivot Edge** | 24 | MED → CONFIRMED bei n≥30 | **+20pp WR** (cooled von +25, naehert sich n=30) | Hard-Filter-Kandidat: skip BREAKOUT ohne PP |
| **🎯 analyst_upside>15 NEGATIV** | 20 | MED → CONFIRMED bei n≥30 | **−19pp WR** (was -22, leicht moderiert) | Catalyst-Score-Delta entfernen oder invertieren |
| **Score-Cap-Hypothese** | 90-100: n=18 (72.2 %)<br>100+: **n=29 (69.0 %)** | TENTATIVE (n=29 bei 100+ fast bei CONFIRMED-threshold) | 100+ KEINE bessere WR als 90-100 — verfestigt | Telegram-Ranking-Score-Cap bei 100 |
| **🔵 BREAKOUT × RSI≥70** | 12 | HYPOTHESIS | **+15pp WR (75 %)** | RSI-Obergrenze 68→72 lockern für BREAKOUT |
| **⚡ Gap ≥2 %** | 7 | LOW (n→ noch klein) | **+27pp WR (71 %)** — starkes Signal | Gap-Score-Boost (aktuell +8), evtl. erhoehen |
| **🔵 BREAKOUT × perf_120 0-25** | 27 | MED | -15pp WR (44 %, war -17) | Score-Penalty |
| **🔵 BREAKOUT × vol_lt_1** | 27 | MED | -12pp WR (48 %) | vol≥1.0-Gate validiert |
| **🚀 STAGE_2 Surge im MIXED-Regime** | n=10 open, 0 closed | HYPOTHESIS (2026-06-08, neu) | aktuelle PnL alle ≤+4 % | **Beobachten:** kommt Defensive-Rotation? STAGE_2-Stocks (LIN/EW/COST/KIM/EQR/ASB/GL) im Watch. Bei n≥10 closed schauen. Falls WR <40 % → STAGE_2 in MIXED-Regime deaktivieren oder Score-Penalty |
| **🔵 BREAKOUT × perf_120 25-50** | 24 | MED | +13pp WR (71 %) | Score-Bonus für diese Range |
| **🔵 BREAKOUT × perf_120 0-25** | 24 | MED | −16pp WR (42 %) | Score-Penalty |
| **🔵 BREAKOUT × vol_lt_1** | 24 | MED | −16pp WR (42 %) | vol≥1.0-Gate validiert |
| **Sektor-Divergenz-Loser** | 15+ Trades | TENTATIVE | 78 % loss rate | Sektor-relative-Stärke-Filter |
| **closing_strength<0.5 Penalty** | n=1 (SM) | HYPOTHESIS | Spike-Fade-Warner | Aktuell -10 Score-Penalty, evtl. Hard-Skip <0.35 |

---

## 4. Aktive Live-Positionen (eToro Demo, TRADING_MODE=live seit 07-06)

**eToro-Portfolio ($100k virtuell):**
- **PCAR** offen, entry $124.47 (echt), SL $118, TP $135 (BREAKOUT scanner)
- **META** offen, entry $620.41 (echt via Fix A), SL $595, TP $658 (MOMENTUM)
- Cash: ~$99.9k
- geschlossen: **NKTR TP +5.30%** (entry $71.39 → close $75.07, netProfit +$2.58)
- gedropped: **AVNT** (order_dropped bei eToro — echt nie zustande gekommen; Hypothese
  Demo-Restriction oder SL/TP-Spread-Reject, volle Response wird jetzt geloggt)

**Paper-Legacy** (vor Live-Zeit, ohne etoro_order_id, laufen im Paper aus):
- PAY (Runner +18%, Trailing-continuous aktiv), FRSH, MRCY, PLTR
- Migrieren sich über 1-2 Wochen weg (TP/SL/Stagnation)

## 4b. Aktive Diskretionäre Positionen / Watch

- **APP** (gekauft $560 am 27.5., +250 USD) — TP $600 gehittet, „buy & hold to $669"-Plan,
  Stop auf $560 (Break-Even) hochziehen empfohlen
- **GEV** (Diskussion vom 1.6.): Buy-Zone unter $950 erreicht ($944 aktuell), aber Pullback noch
  im Gange (RSI 29, schloss am Tagestief). Empfehlung: halbe Position @$944 mit Stop $895, oder
  warten auf grünen Tagesschluss als MR-Turn-Bestätigung.
- **DIS** (Watch seit 2026-06-18, cross-sector idea-gen): enge Base $98.6-$104, löst gerade auf
  (+3.1% am 18.6, $103.97). Comm-Services/Entertainment = bewusst raus aus AI/Power-Chase.
  Setup: Trigger Schluss **>$105.25** (20d-High), Stop unter Base ~$98.50 (-5.7%) o. enger MA20
  ~$101, Ziel 52w-High $124 (+19%, R/R ~3:1). Katalysator: Q2-FY26-Beat (6.5: EPS $1.57>$1.50,
  Streaming+Experiences führen, Double-Digit-EPS-Guide FY26/27). Earnings erst ~4.-12.8 (kein
  Adjacency-Risk). Insider clean (16.6 Form-4 = nur RSU-Vesting+Tax-Withholding, keine Sales).
  Kein Spin/Split. NICHT scanner-validiert, diskretionär. Wenn Trigger nicht kommt: Pullback MA20.

---

## 5. Offene Backlog-Items (siehe `BACKLOG.md`)

1. **Pending/Triggered-Status für offene Signale** — durch Phase B teilweise gelöst.
2. **MOMO-Setup** — verworfen (PF 1.51), Code opt-in belassen.
3. ~~**Sektor-Concurrency-Cap**~~ — **FALSIFIZIERT 2026-07-08** (BACKLOG #13): 0pp WR-Lift,
   wirft Winner raus, 35% Signal-Loss. Nicht mehr verfolgen.
4. **BACKLOG #13** — dokumentiert alle falsifizierten Hypothesen zum 130+-Score-Bucket
   und den gewonnenen TECH_QQQ_GATE-Fix. Auch: Duplicate-Trap (WINDOW=3d zu kurz, BACKLOG #8).
5. **OFFEN: Bearish-Kandidat #2 (Exposure-Reduktion)** — nach Falsifikation von Inverse-ETF
   (nur +8.5%/5yr, 19% Whipsaw-WR) noch nicht spezifiziert. Idee: bei BEARISH weniger Slots
   (5→2) + kleinere Size. Nicht gebaut.

### Aktive Roadmap (`CLAUDE_CODE_BRIEF.md`)
- ✅ **Phase A — Obsidian Brain** (`apex_brain.py`) — shipped 2026-06-03 (`0c0a93b`)
- ✅ **Phase B — Paper Trader** (`apex_trader.py`) — shipped 2026-06-04
  - BREAKOUT only, Top-1 nach Score pro Scan-Tag, Telegram-äquivalentes Gate
  - $300 Kapital, $50 × max 5 Positionen (= $250 deployed + $50 Cash-Reserve)
  - Trailing: high ≥ Entry×1.08 → SL auf Entry×1.05 (einmaliger Sprung)
  - **Cron: `*/5 13-21 * * 1-5` auf Oracle-VM** (07-22 verifiziert: */5, nicht */15) (GH-Workflow geloescht 2026-06-05)
  - Freshness-Gate: Signale älter als MAX_TRIGGER_DAYS=3d werden gar nicht erst aufgenommen
  - State: `apex_positions.json` (pending/open/closed/expired + stats)
  - Journal: `apex_trade_log.json` (append-only, alle Events)
  - eToro-API als Stub (TRADING_MODE env var: paper|live)
  - Löst Backlog-Item 1 (Pending-Status) als Side-Effect
- ✅ **Phase C — Dashboard Paper-Tab** (`dashboard.html`) — Redesign 2026-06-05
  - Open + Closed als ausklappbare Zeilen (vorher 12-Spalten-Tabelle)
  - **Activity Log** rendert apex_trade_log.json (open/close/trailing/expired/etc.)
  - Mode-Karte mit Status-Pill, Equity-Karte mit Δ zum Start-Kapital
  - Mobile 2-Zeilen-Layout via Flex (sw.js v23)
- ⏳ **Phase D — Equity-Research-Plugin** (optional, hängt von A)

### Infrastruktur 2026-06-05+
- **Trader** läuft auf **Oracle Always-Free VM** (Ubuntu 22.04, E2.1.Micro,
  1 GB RAM + 2 GB Swap, Public IP). `~/run_trader.sh` = git pull + python +
  git push. Cron `*/5 13-21 * * 1-5` (alle 5 Min, 07-22 verifiziert). Robust gegen GH-Throttling.

#### ⚠️ `~/run_trader.sh` — liegt AUSSERHALB des Repos (Keys!), nur auf der VM
Enthält `ETORO_API_KEY`/`ETORO_USER_KEY` im Klartext → **darf nie ins Git**. Damit ist die
Datei aber auch **nicht versioniert**: stirbt die VM, ist sie weg. Struktur zum Rekonstruieren
(Stand 2026-07-17, Backups liegen als `~/run_trader.sh.bak-<datum>` daneben):

```
#!/bin/bash                      # KEIN "set -e" — siehe Leitsatz unten
cd ~/ApexTradeHub || exit 1
if ! git diff --cached --quiet; then git commit -m "Trader recover ..." || true; fi   # dirty-Index-Selbstheilung
git checkout -- <state-files> 2>/dev/null || true
git pull --rebase origin master || echo "WARN: pull fail -> lokaler Stand"
export INTRADAY_ENABLED=1 / ETORO_API_KEY / ETORO_USER_KEY / ETORO_ENV=demo / TRADING_MODE=live
python3 apex_trader.py >> ~/trader.log 2>&1 ; PY_RC=$?      # <- laeuft IMMER
for f in apex_positions apex_trade_log apex_manual_overrides apex_etoro_events apex_intraday_rejects (.json); do
  [ -f "$f" ] || continue                                   # noch nicht da -> kein fatal
  if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet "$f"; then
    git add "$f" || true; NEED_PUSH=1; fi                   # trackt neue Logs automatisch
done
[ NEED_PUSH ] && { git commit || true; git pull --rebase || true; git push || echo "WARN: push fail"; }
echo "$(date) Trader run done (py_rc=$PY_RC)" >> ~/trader.log    # laeuft IMMER
```

**LEITSATZ (2026-07-17, nach dem 35-Min-Freeze am 16.07.): Git darf NIE die Positions-Logik
killen.** Das alte `set -e` galt global — ein `git add <nicht-existente-datei>` killte das
Script nach dem Staging und vor dem Commit → dirty Index → jeder Folge-Run scheiterte am
`git pull --rebase` → 35 Min ungemanagte Positionen. Drei Konsequenzen, alle oben verbaut:
1. **Kein `set -e`**, jeder Git-Call mit `|| true` / Fallback-Log. Python läuft immer.
2. **`[ -f ]` + `ls-files`-Check**: Dateien dürfen in die Liste, bevor sie existieren — und
   tracken sich beim ersten Auftauchen selbst. (`git diff --quiet` sieht untracked Files
   NICHT — die Falle die `apex_etoro_events.json` und `apex_intraday_cache.json` erwischt hat.)
3. **`git pull --rebase` direkt VOR dem Push** (zwischen dem Pull oben und dem Push unten
   liegt der ganze Python-Lauf → Race mit Scanner-/Equity-Cron und Windows-Pushes).
4. **`Trader run done` läuft immer** → die Zeile heisst jetzt "Python lief", nicht "Git war
   zufrieden". Vorher log das Monitoring bei jedem Git-Fehler einen toten Trader vor.
- **Scanner, Equity, Knowledge** weiter auf GitHub Actions, aber Push-Step
  gehärtet: `/tmp`-Backup statt Stash, 5x Retry-Loop, Conflict-Resolution
  bevorzugt Worker's Files.
- **Brain** lokal mit auto git-pull. Vault gitignored.

---

## 6. System-Files & Daten-Flow

**Persistente Quellen (in Repo, committed):**
- `knowledge/apex_knowledge.json` — Aggregat-Stats, von `apex_learn.py` erstellt
- `knowledge/trade_postmortems.json` — Per-Trade-DB, von mir manuell ergänzt mit `claude_analysis`
- `apex_signals.json` — alle Scanner-Signale (cron-managed)
- `apex_equity_results.json` — alle geschlossenen Trades (cron-managed)
- `apex_equity_top2.json` — Telegram-Pushed Trades (cron-managed, neue Filter-Logik seit 5/27)
- `sector_cache.json` — Ticker→Sektor (selbstheilend seit 6/1 Fix)

**Reports (auto-generiert, NICHT manuell editieren):**
- `reports/learn_latest.md` — `py apex_learn.py`
- `reports/postmortem_summary.md` — `py apex_postmortem.py --summary`
- `reports/learn_YYYYMMDD_HHMM.md` — historisch

**Beim Postmortem-Schreiben:**
1. Ich update `trade_postmortems.json` (`claude_analysis` + `news.web_research` + `key_events`)
2. Guardrail in `apex_postmortem.py --summary` warnt bei `complete OHNE strukturierte News`
3. User regeneriert Summary lokal mit `py apex_postmortem.py --summary`

---

## 7. Aktueller Daten-Stand (2026-06-14)

- **Lifetime Trades:** 132 | WR 46.2 % | PF 1.84 (Knowledge-Snapshot, nächster Refresh Mo 06:47 UTC)
- **Postmortems analysiert:** **40/132** (92 pending) ← +2 (AFRM/IBKR via Bigdata-Workflow)
- **Market Regime aktuell:** **MIXED** (SPY=OK | QQQ=OK) — war BULLISH bis 06-04
- **CONFIRMED Setups:** BREAKOUT (n=77, WR 59.7 %, PF 2.70), REVERSAL (n=54, disabled)
- **30d Window:** WR 54.5 % / PF 2.49 (n=33) — Drift +8.3pp vs lifetime
- **Paper-Trader (Hybrid live, 2026-06-12):** Equity **$402.09 (+$2.09 / +0.5 %)**
  - 5 offene BREAKOUTs: MOH/ASML/CARR/BAX/ARE | Cash $147.95
  - ASML Trail-Step 1 hat live gefeuert (SL $1818, +2 % gesichert)
  - 2 closed lifetime: ADI Stop −6.05 % / AXTA Stagnation +1.95 %
- **Trader-Config:** MAX_POSITIONS=7, CAPITAL=$400 (incl. $100 virtual deposit 2026-06-12),
  BREAKOUT-only Scanner + Momentum-Filler Backup
- **🚀 STAGE_2 Anomalie:** weiter beobachten, 10 offen im Equity-Tracker (NICHT im Paper),
  0 closed lifetime, Defensive-Rotation-Hypothese (LIN/EW/COST/KIM/EQR/ASB/GL)
- **Score-Realign live (2026-06-14):** Backtest 2J: WR 51.9 → 53.8 %, PF 1.66 → 1.78,
  Total PnL +11 %. 13 weniger Trades, 77 % davon waren Loser (aktiver Filter).
- **BREAKOUT 30d:** **WR 70.8 %** (n=24) — Drift +11pp vs lifetime, weiter stark aber abgekühlt
- **MEAN_REVERSION:** erster geschlossener Trade SBUX -2.38 % D+2 SL
- **REVERSAL 30d:** WR 10.0 % (n=10) — Legacy stirbt
- **Offene Positionen (Equity-Sicht, nach IBKR/AFRM-Closes):** 5 BREAKOUT
  (AXTA/ADI/FANG/JCI/ARE)
- **Paper-Trader-Sicht:** AXTA open (D+1 via Oracle-Cron getriggert), ADI Stop Loss

---

## 8. Recent Major Code-Changes (chronologisch, für Re-Bauchgefühl)

- **2026-08-20** **Rescue-Regel: nur noch GRUEN retten, ROT am EOD schliessen** (commit f0bbf2b5,
  BACKLOG #16b). `RESCUE_REQUIRE_GREEN = True`. Ursache-Analyse: der gruene Zweig (Breakeven-Stop)
  ist gemessen OPTIMAL — jede Lockerung auf -1/-2/-3/-4% verschlechtert monoton. Die 0.00%-Closes
  sind sein PREIS, nicht sein Fehler. Das Loch sass im ROTEN Zweig: -4% "Raum zum Erholen" kostete
  ueber n=13 **-30.00%**. Realistisch (inkl. WULF-Guard): -4% -30.00 | -2% -12.18 | **EOD-Close
  -9.64%** (+20.4pp, in 10/13 Faellen besser). Rollback = Flag auf False, der -4%-Code steht noch.
  **KORREKTUR 22.08.:** Der gruene Zweig verdient NICHT +29.28% wie zunaechst berichtet — die
  Identifikation ueber Stop/Entry hatte Runner mitgezaehlt, deren Stop die Trailing-Leiter hochzog.
  Sauber ueber das Event-Log (mode="gruen->Breakeven"): **+7.20% ueber n=19**, davon fast alles aus
  den komfortabel gruenen (>=+1%: n=7, +6.42%). Die **hauchduenn gruenen (<+1%, n=12) bringen
  zusammen +0.78%** und belegen Slots — GE/TSLA/NDSN sind die Live-Faelle. Vorschlag (NICHT gebaut):
  Rescue-Schwelle von >0% auf >=+1% anheben (+4.31pp gemessen, aber n=12 und von wenigen Trades
  getragen). Begruendung waere mechanisch: ein Breakeven-Stop auf +0.4% ist kein Stop, sondern ein
  verzoegerter Verkauf mit Gap-Risiko ohne Aufwaertsoption.

- **2026-08-22** **WARUM WIR NICHT PROFITABEL WAREN — die Zerlegung** (keine Code-Aenderung, aber
  die wichtigste Erkenntnis). Arithmetik: Ø Gewinn +4.13% vs Ø Verlust -5.34% bei ~50% WR.
  **Nur 13% der Trades erreichen ihr Ziel, Median-MFE +2.62%**, und von 26 Gewinnern sind 13 flache
  Ausstiege bei ~+1%. Die Kante geht in drei Schichten verloren (alle mit derselben Methode gemessen):
  | Ebene | Ø/Trade |
  |---|---|
  | alle BREAKOUT-Signale (180, reine TP/SL-Sim) | **+1.32%** |
  | davon die, die der Trader PICKT (37) | **+0.40%** |
  | was er tatsaechlich erzielt (54) | **-0.51%** |
  Auswahl kostet ~0.9pp, Management ~0.9pp. **Die Rohkante ist nur +1.3%/Trade — zu duenn fuer zwei
  Schichten Reibung.** Deshalb hat kein Einzel-Tuning geholfen.
  **Kernsatz: In einem System mit 13% Zielerreichung zerstoert jeder Mechanismus, der einen Gewinner
  kappt, mehr als er schuetzt** (PPC +14% wiegt vier Stops auf). -> Kuenftig WENIGER Mechanik, nicht mehr.
  **Erste saubere Zahlen seit dem eToro-Aus (07.08.):** alle Setups n=31, WR 51.6%, **PF 1.40**,
  +13.4pp (Ø +0.43%); nur Scanner-BREAKOUTs n=10, WR 70%, PF 1.97. Das ist profitabel — aber
  n=31 ueber zwei Wochen in einem anderen Regime, **kein Beweis**. Der Vergleich "eToro-Aera vs
  Paper-Aera" ist zudem verzerrt (eToro fuehrte die STOPS aus, unsere Logik die TPs).

- **2026-08-05/06** **Gap-Gate auch in Backtest + top2; Reichweite des Bugs eingegrenzt**
  (commits 79e9d39b, 15e9df13): `apex_backtest_v2.evaluate_outcome` hatte denselben Fehler wie der
  Equity-Tracker (Fill zum Trigger, egal wie weit die Aktie darueber EROEFFNETE) -> gefixt, gleiche
  3%-Schwelle wie apex_equity + apex_trader. Alle drei messen jetzt dasselbe Universum.
  `apex_equity_top2.json` (Telegram-Linie der Equity-Kurve) ebenfalls bereinigt: 13 von 86 raus,
  86->73 Trades, WR 55.8->54.8%, +172.1->+128.9pp.
  **DASHBOARD-SCOPE geprueft:** Die Signal-LISTE (`renderToday`) ist NICHT betroffen (nur Levels/
  Score, keine Ergebnisse). Betroffen sind aber die **Equity-Kurve (eqChart) und die Zaehler auf dem
  Signale-Reiter**, der Trade-History-Reiter, das Exit-Doughnut und der Vergleich alle-vs-Telegram.
  **ENTWARNUNG zur Reichweite (A/B-Kontrolllauf, `BT_GAP_GATE_PCT=999` schaltet das Gate ab):**
  Im Backtest entfernt das Gate nur **17 von 223 Trades (7.6%)** und aendert den Gate-80-Lift um
  **0.5pp** (-6.1 -> -5.6pp). **Der Gap-Bug hat die backtest-basierten Entscheidungen (Gate 80,
  VCP A+B, Hold 30d) also NICHT entwertet.** Im Equity-Tracker war er dagegen gravierend (11% der
  Trades = 31% des gebuchten Gewinns) — der Unterschied kommt daher, dass der Backtest ueber 2 Jahre
  und 779 Ticker mittelt.
  **NEUER, UNABHAENGIGER BEFUND (nicht gehandelt):** In diesem 2J-Fenster steigt die Score-Ordnung
  NICHT monoton — 70-80: 55.6% | 80-90: 54.5% | 90-100: 46.5% | 100+: 46.4%. **Gate 80 wirft damit
  den besten Bucket weg (Lift -6.1pp, mit UND ohne Gap-Gate)**, was der Original-Validierung
  ("sub-80 = 33-37% WR") widerspricht. **NICHT zurueckgedreht**, weil (a) die Original-Validierung
  moeglicherweise auf Live-Signalen statt der Backtest-Engine lief, (b) n=45 im Schluessel-Bucket,
  (c) die Live-Kalibrierung aus der Wissensbasis stammt, die heute bereinigt wurde -> **der naechste
  Learn-Lauf liefert die erste ehrliche Score-Kalibrierung. Vorher nichts an Gate 80 aendern.**
  METHODEN-WARNUNG: ein erster Vergleich gegen `bt_gate_study_2y.json` war UNGUELTIG (anderes
  Startdatum, alle Setups statt BREAKOUT-only, null gemeinsame Trades). Nur der gematchte
  Kontrolllauf zaehlt — bei Backtest-Vergleichen IMMER Zeitraum/Setup/Flags gegenpruefen.

- **2026-08-05** **Equity-Tracker bekommt das GAP-GATE — zentrale Metrik war systematisch zu gut**
  (commit 1f805a70): Der Tracker unterstellte IMMER einen Fill zum Trigger-Preis, auch wenn die
  Aktie weit darueber EROEFFNETE — diesen Preis gab es dann nie mehr. Musterfall MHK 31.07.:
  Trigger 122.39, Open 127.85 (+4.5%) nach +42.8% EPS-Surprise, gebucht **+9.43%**, realistisch
  ~+4.8%. Der LIVE-Trader lehnt genau das per `GapTooLargeError` ab (gleiche 3%-Schwelle)
  -> **Tracker und Trader massen unterschiedliche Universen.** Das erklaert die lange offene Frage,
  warum eToro bei -11.7% steht waehrend der Tracker positiv aussieht.
  **Umfang (ganze Historie, nicht neu!): 27 von 245 pruefbaren Ergebnissen (11%), +106.2pp von
  +337.4pp = fast ein Drittel des ausgewiesenen Gewinns.** Durchgehend seit April, Mai am
  schlimmsten (18%). Maerz war zufaellig sauber.
  **Fix:** `GAP_GATE_PCT = 3.0` in apex_equity.py; `evaluate_trade` skippt, wenn der Trigger-Tag
  >3% ueber dem Trigger eroeffnet. Zusaetzlich die 27 Alt-Ergebnisse **purged** (already_saved
  haette sie nie neu bewertet -> sonst halb alt/halb neu), Equity-Kurve neu gerechnet.
  **Neue Basis: 250 -> 223 Trades, WR 46.0 -> 44.4%, Summe +355.4 -> +249.1pp.** Schlechter, aber
  vereinnahmbar. Knowledge/Learn/Score-Gate-Kalibrierung bauen darauf auf und ziehen beim naechsten
  Cron nach — **alte Learn-Zahlen vor dem 05.08. sind entsprechend zu hoch.**
  Backup: apex_equity_results.json.bak-gapgate. Verifiziert: MHK geskippt, SNOW (+1.77% Gap) bleibt.
  **NEBENBEFUND (ungetestet):** Trades mit wirklich erreichbarem Entry (Gap<=0.5%, n=40 seit 01.06.)
  haben WR 32.5% / -59.5pp — die positive Bilanz kam ueberwiegend aus dem geschenkten Gap.

- **2026-07-31** **Trailing-Fruehstufe +4% -> +2% gesichert** (commit c7c882a, beide Ladders):
  User-Beobachtung "viele duempeln bei +2-4%". Befund: unter +6% war **NICHTS geschuetzt** — nur
  **15% von 92 Trades erreichten je Step 1**, bei 85% feuerte die Ladder nie. 12 Trades liefen
  >=+3% und endeten <=+1% (**-30.4pp**). **Neues Tool `trail_sweep.py`**: rekonstruiert Trade-Pfade
  aus Tagesbars (Trailing ist in apex_backtest_v2 NICHT simulierbar — der kennt nur TP/SL/Zeit),
  adverse-first, Horizont = echtes Exit-Datum (exogene Exits bleiben, nur der Trail variiert).
  Validiert: CURRENT-Sim -29.8pp vs. real -21.8pp (Delta = Tagesbar-Granularitaet + TP-Gap-Fills,
  wirkt auf alle Varianten gleich). **Ergebnis +4%->+2%: -29.8 -> -17.1pp, WR 46.7->50.0%,
  Trail-Stops 3->13** (= mehr Rotation). **Die 9 grossen Gewinner (+84.8pp von -21.8pp gesamt!)
  bleiben unberuehrt** — PAY/RHI/HAS/ABBV/NKTR/JBL in JEDER Variante identisch, weil ein Trail nur
  bei Rueckkehr feuert; wer ueber +4% weiter ausbricht laeuft ungebremst. Nur NVO -5.8pp.
  **FALSIFIZIERT dabei:** (a) zeit-konditionale Variante (nur stagnierende Trades ab Tag 3/5 eng
  stellen, um schnelle Runner zu schonen) bringt NICHTS (-29.1 vs -29.8) — der Give-Back passiert
  in den **ersten 1-2 Tagen**, Schutz muss sofort greifen; (b) aggressiv +2.5%->+1% (in-sample
  besser mit -8.7pp, aber Rauschen + kostete hier schon 2 Runner).
  **⚠ REGIME-KONDITIONAL:** Sample = Baer-/Chop-Phase Jun-Jul 2026. Bei extremem Positiv-Skew
  (9 Trades tragen alles) wuerde die Fruehstufe im BULLENMARKT genau die Runner kappen.
  **Bei Regime-Wechsel auf BULLISH neu bewerten. Rollback = Zeile `(1.04, 1.02)` raus.**
  Akzeptanz-Kriterien (vorab fixiert): nach ~20 neuen Trades WR hoeher UND Gesamt-PnL nicht
  schlechter; wenn viele Winner gekappt werden -> raus.

- **2026-07-31** **Phantom-TP-Fix: yfinance-Bad-Print-Guard im Equity-Tracker** (commit 4467d63):
  User sah VOD.L im Signale-/History-Tab als "Take Profit +11.12%", real aber −2% & bei eToro noch
  OFFEN. Ursache: yfinance lieferte fuer VOD.L (.L = London, Pence) am 30.07. einen **korrupten
  Tagesbar High=12047.50** statt ~120 (100x-LSE-Skalierungs-Glitch). `evaluate_trade`s `h >= tp`
  feuerte auf dem Muell-High -> Phantom-TP. Der Live-Trader nutzt den Intraday-Last (sauber) und liess
  korrekt offen -> Divergenz. **Fix:** `_sanitize_ohlc()` (apex_equity.py) verwirft Bars mit
  High > 2x Body-Top bzw. Low < Body-Bottom/2 (nie real fuer Large/Mid-Cap; faengt 100x trivial),
  angewandt in evaluate_trade + compute_open_positions. Konstante `BADPRINT_MAX_RATIO=2.0`. Frozen
  VOD.L-Result (already_saved re-evaluiert nie) manuell entfernt + Equity-Kurve neu berechnet ->
  naechster Equity-Run haelt es raus (Guard: re-eval -> None). Sibling-Scan: nur AMP.MI als EU-TP
  uebrig, verifiziert ECHT (High 12.58 >= Target 12.36). LEHRE: .L/EU-Ticker sind yfinance-100x-Prone.
  **NACHTRAG (commit f36e9cc): 2. Phantom-Klasse — Signal-Bar-Look-Ahead.** User-Frage "AVY heute?":
  AVY meldete am Signal-Tag (07-30) Earnings, echter Intraday-Spike auf High 187.91 = der SIGNAL-Bar.
  yfinance lieferte bei `start=sig_date+1` wegen Timezone-Boundary gelegentlich den Signal-Bar selbst
  zurueck -> `h>=target` feuerte -> Phantom-TP 186.09 (real ab 07-31 nur 174.17). Fix: evaluate_trade
  filtert jetzt strikt `data.index.date > sig_date.date()`. AVY-Phantom entfernt. **Full-Re-Eval letzte
  30d: nur VOD.L + AVY waren echte Phantome.** Die restlichen "Time Exit->TP"-Diffs = Hold-Aenderung
  15->30d (erwartet, KEINE Bad-Prints); LW 07-23 = Same-Bar-Trigger+Stop-Eigenheit (Target real erreicht,
  behalten). Merke: bei so einem Re-Eval NICHT Hold-Staleness mit Phantomen verwechseln.

- **2026-07-31** **Earnings-Schicht war ~30 Scan-Tage LIVE TOT — reaktiviert (Root-Cause: fehlendes lxml)**
  (commits 4d3bc33 + **7024f1a**): User-Frage zu Earnings deckte auf: `cat_earnings_next_days` bei
  **98-100% aller BREAKOUT-Signale = None**, 0% beat/blackout-Flags in letzten 30 Scan-Tagen. Also
  **-15 Blackout (Gap-Schutz) UND +8 PEAD (backtest-validiert!) live komplett offline** — wir kauften
  ungeschuetzt in Earnings-Gaps (ILMN 07-30) und der validierte Post-Beat-Edge wurde nie vergeben.
  **ROOT-CAUSE (korrigiert — NICHT der zuerst vermutete IP-Block):** yfinance `t.earnings_dates`
  braucht **lxml**, das in der Actions-pip-install-Zeile fehlte (`yfinance pandas requests pytz tqdm
  matplotlib`, kein lxml). Ein unpinntes yfinance-Upgrade vor ~1 Monat machte earnings_dates
  lxml-pflichtig -> `ImportError` -> stiller `except: pass` -> earnings=[]. `t.info` (analyst/short)
  braucht kein lxml -> ueberlebte -> maskierte den Ausfall. Lokaler Dev hat lxml 6.0.2 -> lokaler
  Cache sah gesund aus. VM-Probe (`t.earnings_dates` -> "Missing optional dependency 'lxml'") war der
  Beweis. **FIX (7024f1a): `lxml` in alle 3 Cron-Workflows** (nur ApexScan fetcht, aber Zeilen konsistent)
  -> Actions baut den Kalender wieder selbst pro Lauf, **kein Cron/Seed/Chore noetig**. Der zuerst
  committete Cache-Seed (4d3bc33) wurde in 7024f1a **wieder ent-trackt** (gitignored). BEHALTEN aus
  4d3bc33: Merge-Guard in `get_catalyst_data` + derive-Fallback auf juengste Quartalszeile MIT bekannter
  Surprise (frisch gemeldete = `surprise=None`, Yahoo-Lag) — harmlose Defensive. `apex_catalysts` nur
  von ApexScan+Backtest importiert, NICHT apex_trader. **VERIFY: naechster Scan (Fr 20:42 UTC) muss
  wieder `cat_earnings_next_days != None` / beat/blackout-Flags zeigen.** LEHRE: unpinntes yfinance in CI
  kann still Deps nachziehen; jeder in CI genutzte yfinance-Endpoint braucht seine Parsing-Deps explizit.
  **+ Visibility-Guard (commit 48f2816):** apex_catalysts zaehlt earnings-Fetch-Outcomes (`_FETCH_STATS`,
  `get_fetch_stats()`), ApexScan druckt am Runde-Ende `Earnings-Health: X ok / Y empty / Z errors of N`
  und ALARMIERT bei >=20 Fetches & 0 ok ("EARNINGS-LAYER TOT") — damit so ein stiller 30-Tage-Ausfall
  nie wieder unbemerkt bleibt. Wenn die Health-Zeile im Scan-Log 0 ok zeigt: sofort lxml/pip-install pruefen.

- **2026-07-29/30** **eToro-Trailing-Endpoint-FIX (kritisch) + Self-Healing-SL-Reconcile**:
  update_sl_tp gab seit Live-Start **[404] RouteNotFound** (still verschluckt) -> KEIN Trailing-
  Stop erreichte je eToro, jede Position lief mit dem Initial-Stop (User-Report SJM). Reiner
  unbemerkter Kapitalverlust (MMM: Close 175.64 statt Trailing-Stop 176.94 = 5-Min-Paper-Close-
  Slippage weil eToro-SL nie gepflegt). Root-Cause: Endpoint war nie verifizierter Platzhalter.
  Korrekt (Prober 3 Runden, 202 mit operationId): `PATCH /api/v2/trading/{env}/positions/{positionId}`
  camelCase `{stopLossRate, takeProfitRate, stopLossType:"fixed"}`. ALLES war falsch (v1/PascalCase/
  id-im-Body/kein-id-im-Pfad). ACHTUNG eToro-Inkonsistenz: Open-Order will PascalCase, Update-Position
  camelCase! Fix commit d294588 + Reconcile aller 7 Positionen (etoro_reconcile_stops.py, 7 OK).
  **Self-Healing** (3f6c104, SL_RECONCILE_ENABLED): sync_etoro_positions gleicht jeden Run eToros
  stopLossRate mit Paper-Stop ab (aus Portfolio-Response, kein Extra-Call), re-pusht bei Drift >0.3%
  -> keine stille Divergenz > 1 Run. 2 Close-Wege: Paper-Stop-Hit -> Trader schliesst aktiv (Market);
  eToro-server-SL-Hit -> eToro schliesst selbst, Trader liest via close_from_history. Prober-Tool:
  etoro_probe_updatesltp.py. Endpoint in reference_etoro_api-Memory korrigiert.

- **2026-07-29** **BREAKOUT-Haltedauer 15/21 -> 30 (Punkt 4)**: Haltedauer-Sweep (`--hold-sweep`,
  alle Holds in einem Lauf via evaluate_outcome(hold_override)): 30d Optimum (+60pp/2J vs 15d,
  45d nur +2pp mehr), robust Bull +45/Baer +15, WR-Kosten ~2pp. Behebt BACKLOG #7 (Trader 21,
  Equity/Backtest 15 -> alle 30). TP/SL selbst solide. 24% Time-Exits (79% positiv, 70-96% zum TP)
  erreichen jetzt ihren TP. commit dd47fd3. Vorbehalt: Live-Stagnations-Exit (5d) kappt flache
  Trades -> Live-Benefit partiell.

- **2026-07-29** **Regime-Bremse FALSIFIZIERT** (BACKLOG #4, 2. Anlauf): User wollte Baer-Bremse.
  219 BREAKOUT-Trades nach Regime-TIEFE gesplittet: der TIEFSTE Baer (beide WEAK + SPY<MA200, wo
  wir JETZT sind) hat die HOECHSTE WR (76.9%), kein Bucket netto negativ. Baer-Breakouts sind
  Survivor. KEINE Bremse gebaut. Aktuelle Verluste = Small-Sample (30d n=37) + z.T. out-of-sample.
  **KEINE dritte Regime-Bremse ohne fundamental neue Evidenz.** Skript regime_depth.py.

- **2026-07-23** **VCP-Fix: Schema A (Score) + B (Pick-Prioritaet)** (commit 59e5368, 2J-validiert):
  Gewicht-Sweep (bt_all_candidates, `--emit-all-candidates`): VCP gehoert in die PICK-PRIORITAET,
  nicht den Score. **A** (klein, +0.6pp): vcp_signal>=0.30 -> vcp_strength>0, Gewicht +5->+8
  (ApexScan L1376 + Backtest). **B** (gross, +2.4pp): VCP-first im _pick_rank (apex_trader) +
  _tg_rank (ApexScan) + _band_rank (Backtest). A+B-Kombi 2J: WR 54.4->55.7%, PF ->1.85, +395pp,
  VCP-Anteil 35->58%. Schema-B-allein waere 56.8%/2.03/+361 (User: A+B laeuft, Schema-A-Rollback
  als Hebel gemerkt). LEHRE: Score-Gewicht aendert Top-N-Ranking kaum (wie rr), Prioritaet ist der Hebel.

- **2026-07-22/23** **Gate 80 + eToro-Reason-Fix + Rescue-VWAP-Gate** (commit 89302ac/d2ed7c4):
  **Gate 80** (TG_MIN_SCORE BREAKOUT 70->80, ApexScan+Trader): sub-80 = 33-37% WR, Equity +2.2pp/
  2J 54.4->56.2%, 88% Retention. SCAN_MIN_SCORE bleibt 70 (Equity misst sub-80 weiter). **eToro-
  Reason-Fix**: sl_ref/tp_ref-Fallback auf pos["stop"]/target wenn eToro keine Rate liefert (WULF).
  **Rescue-VWAP-Gate** (RESCUE_REQUIRE_ABOVE_VWAP): Rescue war netto -1.01pp (n=18, 13/18 schaden)
  -> nur noch rescuen wenn EOD ueber VWAP. Dashboard-Badge-Fix (37/61 falsch als TE, SW v48).
  **Intraday-Analyse**: Reject-Filter RANGE_POS_MAX=0.90 validiert (filtert Fader), Rescue der Bluter.

- **2026-07-22/29** **Diagnose + Postmortem-Pipeline**: eToro real -$48/WR 28% = REGIME (voll BEARISH,
  90d noch +). TE-Beschoenigung: ehrliche Scanner-WR ~30% statt 52%. Winner = Post-Earnings-Beat
  (ABT/HAS/LW), Loser = Baer-Stops (high-score schuetzt nicht). **Pipeline-Bug gefixt** (d001f59):
  trade_postmortems.json war nicht in Knowledge-Cron-git-add -> Pendings persistierten nie (9->23);
  jetzt drin, 240 complete/0 pending. **MCP-News gesperrt (FMP) -> WebSearch.**


- **2026-07-17** **run_trader.sh Push-Guard — Git darf die Positions-Logik nicht killen**:
  - Konsequenz aus dem 35-Min-Freeze (16.07.). `set -e` galt global → ein `git add` auf eine
    noch nicht existente Datei killte das Script zwischen Staging und Commit → dirty Index →
    Dauer-Freeze am `git pull --rebase`. Details + rekonstruierbare Struktur: §Infrastruktur.
  - Vier Fixes: (1) kein `set -e`, Python läuft immer; (2) `[ -f ]` + `ls-files`-Check →
    Dateien dürfen vor ihrer Existenz in der Liste stehen **und tracken sich selbst** beim
    ersten Auftauchen; (3) `git pull --rebase` direkt vor dem Push (Race-Fix); (4) dirty-Index-
    Selbstheilung am Run-Start; (5) `Trader run done (py_rc=$?)` läuft immer = ehrliches
    Monitoring.
  - **Damit ist BACKLOG #22 entschärft**: `apex_intraday_rejects.json` steht bereits in der
    Liste und trackt sich beim ersten Deep-Scan selbst — kein manueller `git add`, keine
    Reihenfolge-Falle mehr.
  - Verifiziert auf der VM: `bash ~/run_trader.sh` → EXIT 0, `Trader run done (py_rc=0)`,
    Index sauber, Push OK — **mit der exakten Freeze-Konstellation** (rejects.json in der
    Liste, existiert nicht) → folgenlos übersprungen. Backups: `~/run_trader.sh.bak-<datum>`.

- **2026-07-17** **EU-Grundsatzentscheid: "messen statt bauen"** (BACKLOG #23):
  - **Der Befund war kein Filter-, sondern ein TAKT-Problem.** EU-Boersen schliessen 15:30 UTC,
    Trader-Cron laeuft 13:00–21:00 = 2.5h Ueberlappung. Verschaerft durch die Trigger-Mechanik:
    `trigger_pending` prueft `high_today >= entry`, **kauft aber zum aktuellen Preis** — ein
    EU-Signal das 08:00 UTC triggert wird 13:00 gesehen und 5h zu spaet ausgefuehrt. Nach EU-Close
    liefert yfinance den Close-Bar weiter, ohne ihn als stale zu markieren.
  - **"Anpassen" traegt nicht** (Trigger aufs 2.5h-Fenster begrenzen = jede Morgen-Bewegung
    verpasst, Signale expiren). **"Abkapseln"** (eigener EU-Cron) = hoher Aufwand fuer einen Edge
    den wir bei **0 Signalen in 4 Monaten** nie gesehen haben.
  - **Der Hebel:** Der Equity-Tracker simuliert alle Signale auf **Daily-Bars, unabhaengig vom
    Live-Takt** → EU-Edge messbar ohne EU-Euro und ohne zweiten Cron.
  - Umgesetzt: EU bleibt im **Scanner** (= Datenquelle), `INTRADAY_EU_ENABLED=False` (825→719
    Ticker), `EU_GUARD_ENABLED=True` (`_eu_entry_blocked`: Live-Entry nur 07:00–15:15 UTC, kein
    WE, fail-closed). Guard sitzt nach dem Expiry- und vor beiden Kauf-Pfaden → Pending wartet
    auf echte Preise statt zu verfallen.
  - **Entscheid vertagt auf Daten:** bei n>=30 EU-Trades im Equity-Tracker (~3 Mon, ca. Okt 2026)
    WR/PF gegen US-Baseline (51.7%/1.78). Danach: eigener Cron / EU raus / bestaetigt.
  - Verhaltens-Tests: Suffix-Erkennung (7 EU-Boersen, BRK-B negativ), Cutoff-Grenzen 15:14 vs
    15:15, Wochenende, Rollback-Flag, Universum EU-frei. Alle PASS.

- **2026-07-16** **eToro-Close-Backfill + Reject-Log + Postmortem-Wartung**:
  - **eToro-Close-Backfill live** (apex_trader.py `sync_etoro_positions`): Race gefixt — wenn
    die Paper-Exit-Bedingung im selben Run feuert, in dem eToro schon geschlossen hat
    (Portfolio-API-Lag ~3min), gewann Paper das Rennen → `etoro_close_rate`/`net` blieben leer,
    kein Event, Sync schaute nie wieder auf `closed[]`. **RHI 16.07.**: eToro-TP feuerte 13:32
    (Fill 40.45 ÜBER TP 40.37 = Gap durch den TP), Paper buchte 13:35 seinen theoret. Target
    40.40 (+13.01%/$6.50) — real +13.21%/$6.60. Fix backfillt `closed[]`-Live-Trades ohne
    Close-Daten aus der History (7d-Fenster), korrigiert pnl auf eToros Wahrheit (netProfit =
    echtes Geld), loggt `close_backfill`-Event. Audit-Felder `paper_exit_price/reason/pnl_pct`
    bleiben. **RHI wurde 18:35 automatisch korrigiert** ✓. BACKLOG #21.
  - **Intraday-Reject-Log** (`apex_intraday_rejects.json`, tages-dedupliziert, 14d-Retention):
    loggt jeden abgelehnten Kandidaten + Grund. Offene Frage (BACKLOG #22): sperren
    `RANGE_POS_MAX=0.90` / `GAIN_MAX=6.0` die STÄRKSTEN Mover aus? 16.07. liefen RHI +7.5%,
    MAT +6.5%, DXCM +5.7% den ganzen Tag, gekauft wurde nur IR (+3.1% Peak → verblasste).
    Gegen-Evidenz 10.07.: 4 Peak-Käufe bei range_pos>0.90 = alle rot. n winzig beidseitig →
    messen statt raten. ~~TODO: Datei einmalig tracken + in run_trader.sh-Liste~~ **erledigt
    07-17: der neue Push-Guard trackt sie beim ersten Auftauchen selbst (ls-files-Check).**
  - **Postmortem 216 complete / 0 pending**, News jetzt via **MCP `news`-Tool** statt WebSearch
    (strukturiert, datumsgenau, mappt direkt auf `web_research`). Learn-Sektion **5b** neu
    (perf_120<0 × Catalyst-Split). 2 Report-Bugs gefixt (Heatmap-Duplikate, Recent-30d-Sektion).
  - **EU weiterhin 0 Signale** — aber statistisch unauffällig: Fix erst 3 Scan-Tage alt,
    erwartet wären ~1.5 (P(0)=22%). 105/105 Frames laden, Liquidität kein Problem.
    **Offen: Grundsatzentscheid EU abkapseln vs. anpassen.**
  - **Trader-Freeze 35min** (19:25–20:00 UTC): `sed` trug `apex_intraday_rejects.json` in die
    git-add-Liste ein BEVOR die Datei existierte → `git add` fatal → `set -e` killt Script nach
    dem Staging, vor dem Commit → dirty Index → Folge-Runs scheitern am `git pull`. **Lehre:
    Cron stoppen bevor am Live-System gearbeitet wird** (Memory-Regel).

- **2026-07-15** **Pick-Band + Blacklist + Analyst-Analyse**:
  - **PICK_BAND live** (apex_trader.py `select_new_signals`): Trader-Pick sortiert jetzt via Sweet-Spot-Band [90,120) statt rohem Score — vereinheitlicht mit Telegram-Band (schon live). BREAKOUT im Band bevorzugt, 130+/120+ ans Ende. 2J-Backtest WR 52.9->54.1%, PF 1.59->1.68, Profit +36pp (Re-Ranking, kein Signal-Loss). Formal knapp unter +2pp-Bar aber Backtest unterschaetzt (fehlende Live-Boni = kein 130+-Bucket im BT). Rollback = PICK_BAND=None.
  - **TSM blacklisted** (BAD_PERFORMERS, WR 25%/kum -5.2%). ASML NICHT (40%/+5%).
  - **Analyst +3 geprueft, BEHALTEN** (User-Entscheid): instabil (+10.6pp gesamt nur aus 6 fruehen Zufalls-Wins, juengere Haelfte +0.9pp) aber klein, bleibt.

- **2026-07-15** **Slot-Option-B + Score-Prio + Trader-Fixes (KW29-Analyse)**:
  - **Slot-Option-B** (apex_trader.py): Scanner-BREAKOUT (`select_new_signals`) nutzt jetzt
    ALLE physisch freien Slots (`MAX_POSITIONS - open - pending`) statt SWING_MAX=4. Grund:
    0/6 BREAKOUTs in KW29 gekauft, weil rescued Intradays (PAY, F) das Swing-Budget fuellten.
    Momentum-Filler bleibt SWING_MAX-gedeckelt. Intraday verdraengt bei Vollbelegung die
    schwaechste Swing-Position (`_find_displaceable_swing`: nur momentum_filler/rescued, nie
    BREAKOUT/Runner/gruen, max 1/Run) bis zur Reserve-Floor (3). Prio = WR: BREAKOUT>Intraday>Filler.
  - **Score-Prio** (ApexScan.py): Sektor-Momentum-Bonus (±6) ENTFERNT (prozyklische Inflation,
    BACKLOG #12). Telegram-Top-2 bevorzugt `TG_SWEET_BAND=(90,120)` — walk-forward +8pp WR OOS.
    Trader-Pick (Top-1) noch NICHT auf Band (AP5-Pfad). Winrate-Bonus-Pruefung = BACKLOG #19.
  - **Trader-Fixes**: (a) EOD-Rescue pusht neuen SL zu eToro (WBD-Bug: Paper/eToro-Divergenz);
    (b) `INTRADAY_ENTRY_CUTOFF_UTC=19:15` — keine Intraday-Entries mehr kurz vor EOD (EQNR-Fall).
  - **Score-V2 FALSIFIZIERT** (BACKLOG #17): 2J-Backtest WR 45.5% vs Baseline 50.2% -> Score bleibt.
  - **EU-Universe-Bug gefixt** (BACKLOG #18): normalize_ticker zerstoerte .DE-Suffixe -> 0 EU-Signale.
  - **Rescue-Attribution ehrlich** (BACKLOG-Brief AP1): source bleibt intraday_momentum + Flag.

- **2026-07-08** **Signal-System Robustheit — TECH_QQQ_GATE + STAGE_2/Momentum disabled**:
  - **STAGE_2 (Trend-Setup) DEAKTIVIERT** in ApexScan.py — kein historischer Edge (n<3
    Lifetime, User-Beobachtung: 0 Wins). Flag `STAGE_2_ENABLED = False`.
  - **Momentum-Filler pausiert im BEARISH-Regime** (apex_trader.py `select_momentum_fillers`
    liest apex_market.json.mode). Daten: 14d bearish WR 30% vs Lifetime 41%. Intraday-Catcher
    (57% bearish WR) bleibt UNANGETASTET.
  - **TECH_QQQ_GATE live** (ApexScan.py L~1091, `TECH_QQQ_GATE_ENABLED = True`): Skip BREAKOUT
    wenn Sektor Tech/Communication UND qqq_perf_20 < 0. Backtest-validiert:
    - Failure-Mode: Tech+QQQ<0 = **WR 14% / PF 0.56** (n=7, ASML×3/TSM/LRCX-Loss/KLAC)
    - Nach Gate: WR 57.1→**59.8%**, PF 2.29→**2.53**, Signal-Loss nur 6%
    - Non-Tech+QQQ<0 (57%) und Tech+QQQ>0 (54%) unberuehrt
  - **FALSIFIZIERT (BACKLOG #13 dokumentiert)**: Extension-Filter (perf_120/perf_20) →
    Winner-Drop; Sektor-Concurrency-Cap → 0pp Lift; Broad-Regime-Gate → zu grob.
  - **Inverse-ETF-Backtest VERWORFEN**: PSQ+QQQ<MA200&mom20<0 = nur +8.5%/5yr,
    Whipsaw-WR 19%, −17% DD. Cash halten schlaegt Inverse-Timing. Kandidat #2
    (Bearish-Exposure-Reduktion) noch OFFEN.

- **2026-07-06/07** **eToro LIVE (Demo-Portfolio) + kompletter API-Roundtrip**:
  - **`etoro_client.py`** — REST-Wrapper mit ALLEN Endpoints: `resolve_ticker`,
    `search_instrument` (internalSymbolFull), `get_rates` (`/market-data/instruments/rates`,
    live bid/ask), `get_balance` (`/trading/info/{env}/portfolio`), `get_positions`
    (aus balance-response), `open_position` (POST `/api/v2/trading/execution/{env}/orders`,
    PascalCase-Body: InstrumentID/Amount/IsBuy/StopLossRate/TakeProfitRate/StopLossType=fixed),
    `close_position` (POST `/trading/execution/{env}/market-close-orders/positions/{pid}`,
    Body: InstrumentId+UnitsToDeduct), `cancel_order`, `update_sl_tp`, `get_history`
    (`/trading/info/trade/{env}/history?minDate=`).
  - **Auth-Erkenntnis**: Portal-Labels sind VERDREHT zu API-Headern:
    - `x-api-key` = **"Öffentlicher Schlüssel"** aus Portal (nicht der generierte Key!)
    - `x-user-key` = **generierter API-Schlüssel-Wert** (nur 1x angezeigt)
    - Cloudflare bannt Python-urllib default UA → **User-Agent-Header muss gesetzt sein**
  - **Trader wired**: `TRADING_MODE = paper | live_dry | live` (env-var).
    `etoro_open_position` (mit **Fix A**: holt eToro-Ask VOR order, rebased entry/SL/TP/shares
    auf echten Preis — vorher 0.3-0.6% Divergenz yfinance vs Fill),
    `etoro_close_position` (via etoro_position_id oder order_id),
    `etoro_update_sl_tp` (nach jedem trailing_activated/continuous).
  - **sync_etoro_positions**: bei jedem Run Portfolio+History fetchen. Positions gefunden
    → openRate/positionID mergen. Nicht gefunden → history checken → wenn dort: `close_from_history`
    mit netProfit; sonst `phantom_close` (order_dropped).
  - **Retro-Fix Script** `apex_etoro_retrofit.py`: korrigiert falsch-gelabelte "order_dropped"
    Closes durch History-Lookup. Angewandt: NKTR korrigiert ($71.29 yfinance → $71.39 real
    → TP $75.07 +5.30% netP +$2.58). AVNT bestätigt "wirklich order_dropped" (nie in eToro).
  - **VM-Setup**: run_trader.sh hat ETORO_API_KEY/USER_KEY/ENV/TRADING_MODE exports. Push-Loop
    committet auch apex_etoro_events.json. git checkout --  erweiterte Liste gegen Dirty-State-Freeze.
  - **Live-Bilanz 07-08**: NKTR (TP +5.3%, retrofit), PCAR (offen), META (offen), AVNT (dropped).
    Slippage META $0.55 yfinance→ask war Anlass fuer Fix A. eToro-Gebuehr $1/Trade im
    normalen Demo — Live Smart-Portfolio angeblich fee-free (User-Info, unverifiziert).

- **2026-07-03** **Dashboard eToro-Tab + Trader-Fixes**:
  - Neuer Tab `#page-etoro` mit Mode-Badge, Live-Positionen-Tabelle (filter etoro_position_id,
    nicht order_id — sonst Phantoms sichtbar), Bubble-Chart (Zeit × PnL% × Size), Activity-Log.
  - **Trader-Fixes**: (1) Trailing-Continuous Compare auf **round(new_stop,2)** statt raw
    (PAY spammte 65 Events/24h — 27.1848>27.18 True, gerundet identisch). (2) Auto-Exits
    (Stagnation/Time/EOD) gegated auf **market_is_open_now()** — verhindert AYI-Feiertags-
    Stagnation-auf-stalem-Preis (kritisch fuer Live). TP/SL bleiben aktiv (preis-getriggert).
  - **Momentum-Trailing verbessert**: kontinuierliches Trailing NACH Ladder-Ende (>+15%):
    `SL = high × (1 - MOMENTUM_TRAIL_GIVEBACK)` — kein +11.5%-Cap mehr.
  - `sw.js` bis v41 gebumpt (mehrere Iterationen im UI).
  - Claude's Picks Tab + apex_etoro_take.json ENTFERNT (User: brauchen wir nicht).

- **2026-07-02** **Learn-Stand + Postmortem-Batch (187 Trades, alle analysiert)**:
  - **Performance (Knowledge, nur getradete Setups):** BREAKOUT lifetime **WR 57% (n=119, PF 2.29)**,
    90d +$469. Markt BULLISH. ABER **7d/14d schwach** (14d WR 30%) — der 23./24.6-Semi-Selloff +
    High-Score-Stops (FLR 148, TSM 137, TGT 137 alle Loss). Kein Systemfehler, Regime-Delle.
  - **NEUES validiertes Thema — AI-Power-für-Data-Center** (`ai_power_datacenter`): CMI (Circe
    2GW-Gensets-Deal), CAT (Project Kilby, Picks-Board), CEG (Nuklear) = der **Nicht-Chip-Weg**
    zum AI-Trade. Mehrfach als Gewinn-Treiber + what_to_replicate. Rückgrat des AI-Buildouts.
  - **Sell-the-News auf Rekord-Earnings:** COST (STAGE_2 -12.3%/31d, Blowout-Q3 aber 52x-P/E fiel)
    + CCL (Rekord-Q2, fiel auf Guidance). Muster: Rekord + Extension/vorsichtige Guidance = Fade.
  - **`analyst_upside` Anti-Prediktivität RE-bestätigt:** TGT hatte 4 PT-Raises + Upgrade → verlor.
    Deckt die Carve-Out-Entfernung (06-24). Learn zeigt zwar +46% (n=7) = Kleine-Stichprobe-Rauschen,
    NICHT drauf reinfallen. Confirmed-Edges: pocket_pivot +16% (n=50 HIGH), gap_gt_2pct +21% (n=16).
  - **STAGE_2-Risiko notiert:** 60d-Hold liess COST -12% Drawdown entstehen → engerer Stop/Trailing erwägen.

- **2026-06-24/26** **Scoring/Learn/Trader/Board-Sammelupdate**:
  - **Scanner:** `analyst_upside>15` aus SCORE_REBUILD-Carve-out ENTFERNT (anti-prädiktiv, n=28).
  - **Learn:** Reversal/MEAN_REVERSION aus Aggregation gefiltert (`join_signal_to_trade` → ACTIVE_SETUPS);
    echte WR 46→57%. Reversal-Cleanup auch in Dashboard/apex_signals.json/apex_equity (überall raus).
  - **Trader:** Stagnation zählt **Handelstage** (nicht Kalender); Momentum läuft höher via **Trailing-Ladder**
    statt hartem +6%-Cut; **Intraday-EOD→Momentum-Swing** (grün=Breakeven-Stop, rot=-4%, statt Force-Close);
    **Market-Hours-Guard** `market_is_open_now()` (NYSE 9:30-16:00 ET, blockt Pre-Market-Entries — KRITISCH Live).
  - **Sektor-RS-Bonus** getestet (n=83) → **FALSIFIZIERT** (kein Edge, Post-hoc-Narrativ). Existierender
    absoluter Sektor-Bonus auch nutzlos → BACKLOG #12 (entfernen).
  - **Claude's Picks kuratiert:** NVDA raus (worst-Semi 2026, China/GPU-Deflation) + DIS raus; LRCX + CAT rein.
  - **Dashboard:** History- + Paper-Closed-Liste einklappbar; Activity-Log formatiert intraday_open/_to_swing;
    `dashboard.html` network-first in sw.js (Updates kamen sonst nie an); sw v30.

- **2026-06-23** **Trader: Slot-Auslastung — Cooldown + Trending + Stagnation-Gate**:
  - **Close-Cooldown** (`recently_closed_tickers`, CLOSE_COOLDOWN_DAYS=5): gerade geschlossener
    Ticker darf 5 Tage NICHT re-geoeffnet werden — in ALLEN 4 Entry-Pfaden (select_new_signals,
    trigger_pending, momentum, intraday). **Fix fuer ASML-Duplicate-Churn:** ASML an 5 Daten
    emittiert; nach Stagnation-Close (15.6-Version) oeffnete die 19.6-Version 5 Min spaeter am
    alten buy_above ($1942) ueber Marktkurs. = BACKLOG #8 als realer Schaden. Trigger_pending hat
    zusaetzlichen Anti-Churn-Guard (offen ODER Cooldown -> Pending expired).
  - **yfinance-Trending als 2. Momentum-Quelle** (`fetch_trending_universe`): day_gainers +
    most_actives gemergt in die Momentum-Universe (durchlaufen DIESELBEN Filter, kein Loosen).
    Fuellt idle Cash mit in-motion-Namen wenn statisches Top-200-Universe dünn ist (war 0 -> 2-3
    Kandidaten). Bewusst spekulativer (Small-Cap-Mover wie BWIN/RCUS/KLRA), aber Stop -4%/Hold 7d.
    Funktioniert: BWIN/RCUS je +5.5% TP, KLRA -4.5% SL = netto positiv.
  - **Stagnation-Gate** (`update_open_positions(allow_stagnation=...)`): Stagnation-Exit nur wenn
    Ersatz in Pipeline (fresh-scanner ODER momentum). Pipeline leer -> flache Position HALTEN
    statt Slot fuer nichts zu leeren (User-Wunsch: Slots voll halten, kein idle Cash). Log:
    `replacement-check: fresh=X momentum=Y -> Stagnation erlaubt/GESPERRT`.

- **2026-06-22** **Claude's Picks — diskretionaeres Conviction-Board (NEU, 3. System-Layer)**:
  - `claude_picks.html` + `claude_picks.json` — von Claude kuratierte Top-Picks, **kein Scanner/
    Trader**, rein diskretionaer auf User-Zuruf aktualisiert. Quellen: Knowledge + WebSearch +
    SEC EDGAR + yfinance-Charts. Dark-Trading-Terminal-Design: Heatmap-Kacheln, Ticker-Logos
    (FMP-Image-URL), Sparklines (30d), Ticker-Tape (Indizes/Crypto/VIX/Gold), SVG-Daumen-Verdikte.
  - Pro Pick: `take` (Einordnung in einfacher Sprache, „kaufen/abwarten weil…"), `entries`
    (mehrere Optionen), `stop`/`target`, conviction 1-5, horizon swing/long, `why` (dated facts).
  - **Dashboard-Tab** „🎯 Claude's Picks" (iframe, lazy-load). sw.js **v25** + claude_picks.html
    auf **network-first** (Board-Iterationen ohne Versions-Bump sichtbar). Outcome-Tracking
    (Δ seit Aufnahme) zur Selbstkontrolle.
  - Update via „update die Picks": Claude pflegt JSON + zieht frische Kurse/Sparklines. Picks
    duerfen reifen (kein Tageszwang). Aktuell: Cross-Sektor (NVDA/VRTX/PLD/CEG/V/KO/DIS).

- **2026-06-20** **SCORE_REBUILD live** (Details in §2 BREAKOUT-Tuning): catalyst-gated
  perf_120>50-Penalty (-12), Backtest-validiert (Plateau-Monotonie -15pp->-0pp). 80-90-Bucket
  diagnostiziert (umgekehrte-U-Kurve, WEAK-Seite offen) + geparkt (BACKLOG #11).

- **2026-06-18/19** **Trader: Intraday-Catcher + Option-B + Holiday-Guard + 2 Bugfixes**:
  - **Intraday-Momentum-Catcher** (`apex_trader.py` Step 3c, EXPERIMENT, opt-in `INTRADAY_ENABLED=1`):
    scant die ~50 Daily-Momentum-Namen intraday (5m-Bars): gain_from_open 1.5-6 %, über VWAP,
    range_pos ≥0.55. Direkter Market-Entry (kein pending/trigger). Exit TP **+5 %** / Stop **-3 %** /
    Hard-Close ab **19:45 UTC**. Tags `source=intraday_momentum`, setup `INTRADAY`. User-Ziel:
    ~$20/Tag durch schnelle Intraday-Sprünge. MOMO-Risiko bewusst (BACKLOG #2). Eval nach 1-2 Wo,
    Rollback = Flag auf 0. **VM:** `export INTRADAY_ENABLED=1` in run_trader.sh (NICHT ~/.bashrc —
    Cron sourct das nicht!), Cron `*/15`→`*/5`.
  - **Option B Slot-Split:** `SWING_MAX_POSITIONS=5` (Scanner+Momentum), `INTRADAY_RESERVED_SLOTS=2`,
    total MAX_POSITIONS=7. Intraday zählt NICHT gegen Swing-Budget → Catcher wird nicht mehr vom
    vollen BREAKOUT-Buch ausgehungert. (Antwort auf: "bei genug BREAKOUTs öffnet nie ein Momentum/
    Intraday-Trade" — stimmte, Filler+Intraday waren beide hinter freien Slots gated.)
  - **BUGFIX Manual-Close-Doppelzählung:** `apply_manual_overrides` CLOSE rief `close_position`
    (→ closed + Cash) aber entfernte die Pos NICHT aus `open` → doppelt gezählt (Equity inflated).
    Gefunden via LUV-Close (Equity sprang fälschlich auf $468). Fix: `state["open"]` filtern.
  - **BUGFIX Holiday-Guard:** Cron `*/5 13-21 1-5` kennt keine US-Feiertage → an Juneteenth (19.6)
    triggerte FLR fälschlich auf stalem Donnerstags-Hoch. Neue `market_open_today()` (SPY letztes
    Bar-Datum == heute ET?) gated Step 2+3. Bei zu Börse: nur Mgmt+Overrides, keine Entries/Trigger.
  - State-Repairs: LUV (Doppelzählung raus) + FLR (Feiertags-Open → zurück pending).

- **2026-06-15** **FRED-Macro-Integration live (Telegram-Header + Postmortem-Context)**:
  - `apex_macro.py` — pullt FRED daily (VIXCLS, BAMLH0A0HYM2, T10Y2Y, DFF, DTB3), schreibt
    `apex_macro.json`. 3-State Regime: RISK_ON 🟢 Good / ELEVATED 🟡 Mid / RISK_OFF 🔴 Bad.
    Threshold: VIX ≥25 oder HY-OAS ≥5.0 = stress; VIX ≥20 oder HY ≥3.5 = elevated. Worst-of regiert.
  - `apex_macro_history.json` (2y backfill via `--backfill`) — fuer Postmortem-Lookups
    am Entry/Exit-Datum.
  - **ApexScan.py-Patch:** Telegram-Header bekommt 2. Zeile `🟢 Macro: Good · VIX 17.7 (-1.8) · HY 2.71`.
    Graceful fallback wenn Macro-File fehlt.
  - **apex_postmortem.py-Patch:** `market_context` enthaelt jetzt `macro_at_signal` +
    `macro_at_exit` ({vix, hy_oas, yield_curve, regime, date_used}).
  - **Oracle-VM-Cron NEU:** `15 6 * * * /home/ubuntu/run_macro.sh` — daily 06:15 UTC.
    Pulls, runs apex_macro.py, commits+pushes apex_macro.json wenn changed. `FRED_API_KEY`
    in `~/.bashrc` als env var (NICHT in Repo).
  - **Macro-Backtest-Hypothese FALSIFIED** (siehe BACKLOG #4): BREAKOUT-WR sinkt NICHT bei
    VIX ≥22 — RISK_OFF-Bucket zeigt sogar **+4.6pp WR** (61.1% n=18). Macro-Gate killen
    waere kontraproduktiv. apex_macro_backtest.py bleibt als opt-in Re-Test-Tool fuer
    n≥300 in 6+ Monaten.
  - **Side-Finding (BACKLOG #5):** REVERSAL × HY 3.0-3.5 = 53.3% WR n=15 (vs 30.4% baseline)
    — TENTATIVE, nicht reaktivieren bis n≥30.

- **2026-06-12** **Hybrid-Trader live + STAGE_2-Rollback**:
  - `ALLOWED_SETUPS = {BREAKOUT}` (STAGE_2 testweise drin, dann raus — Hold 60d widerspricht
    Rotations-These und blockiert Slots wochenlang. STAGE_2 bleibt im Equity-Tracker als
    Beobachtungs-Sample mit 10 offen, 0 closed.)
  - `MAX_POSITIONS 5→7, CAPITAL_INITIAL 300→400` ($100 virtual deposit, in trade_log geloggt
    als event `capital_deposit`)
  - **Momentum-Filler NEU** (`fetch_momentum_universe`, `select_momentum_fillers`):
    yfinance Top-200 US-Tickers (us_tickers.txt vorsortiert nach Marktkap), 1mo daily,
    Filter: perf_5d≥3 %, RSI≤72, vol_ratio≥1.2, price≥$5.
    Score: eigene Skala (perf_5d*4 + perf_20d*0.5 + vol_ratio*6 + RSI-sweet), min_score=60.
    Stop/TP: −4 % / +6 %, Hold 7d. Cache 6h TTL in `apex_momentum_cache.json` →
    max 2 yfinance-Downloads/Tag, kein Throttling.
  - **Priorität:** Scanner-BREAKOUTs ZUERST, Momentum nur wenn Slots übrig nach Scanner.
  - `source: "scanner"|"momentum_filler"` auf jeder Position für spätere Auswertung.
  - **NICHT umgesetzt (BACKLOG #3):** Sektor-Cap max 2 pro Sektor (User-Entscheidung,
    erst nach Hybrid-Test ggf. nachziehen).
- **2026-06-11** **Multi-Signal Slot-Filling in apex_trader.py** (Top-1/Tag-Regel weg):
  - Vorher: Scanner pickte nur Top-1 BREAKOUT pro Scan-Tag → chronische Unter-Auslastung
    (3 Slots leer, $198 idle bei 4 verfügbaren Elite-Signalen).
  - Jetzt: Alle qualifizierten BREAKOUTs der letzten 3 Tage gesammelt, dedup pro Ticker,
    nach Score sortiert, freie Slots aufgefüllt. Cash-Gate (`cash ≥ $50`) bleibt.
- **2026-06-14** **SCORE_REALIGN live in ApexScan.py** (Backtest 2 Jahre validiert):
  - BREAKOUT RSI-Zone 48-68 → **48-72** (RSI≥70 zeigt 75 % WR n=12)
  - perf_120 0-25 movement_bonus +5 → **−3 DEADZONE** (44 % WR n=27, größtes Loser-Bucket)
  - perf_120 25-50 → **+15 SWEET** (71 % WR n=24)
  - perf_120 >50 → **+8** (vorher Power +15 oder Emerging +5)
  - Backtest-Ergebnis: WR 51.9 → **53.8 %** (+1.9pp), PF 1.66 → **1.78**, total PnL +11 %
  - n −6 % (195 vs 208), **77 % der weggefallenen Trades waren Loser** = aktiver Filter
  - Gate 1 (n≥95 %) strict gefailed (94 %), aber Loser-Anteil rechtfertigt GO. User OK.
  - Backtest-Flag `--score-realign` bleibt für Reproducability, Live ist hardcoded.



- **2026-06-08** **ApexKnowledge Cron 06:30 → 06:47 UTC** (off-peak slot, war 2-6h
  delayed durch GH-:30-Throttle-Zone). Plus `apex_postmortem.py` lief ohne `--summary`
  (full mode) — addet neue closed trades zu trade_postmortems.json (aktuell 0 new).
- **2026-06-07** **Trader Phase 2: Manual Override System** (`apex_manual_overrides.json`)
  - Schema: `{ticker: {sl, tp, close, note, set_at, applied_at}}`
  - User/Claude editiert, Trader liest jeden Run, wendet noch-nicht-`applied_at` an
  - SL: max(old, new) - niemals nach unten, Trail-Ladder konsistent gehalten
  - TP: direktes Überschreiben | CLOSE: `"Manual Close"` exit mit current_price
  - Events ins trade_log (`event: manual_override` mit field/old/new/note)
  - VM-Script `run_trader.sh` updated (apex_manual_overrides.json in git add list)
- **2026-06-07** **Trader Phase 1: Trailing-Ladder + Stagnation + Replacement**
  - **Trailing-Ladder** ersetzt one-shot Trail: 3 Stufen
    - Step 1: high ≥ entry×1.06 → SL = entry×1.02 (+2 % gesichert)
    - Step 2: high ≥ entry×1.10 → SL = entry×1.06 (+6 % gesichert)
    - Step 3: high ≥ entry×1.14 → SL = entry×1.10 (+10 % gesichert)
    Position bekommt `ladder_step` Feld (0/1/2/3).
  - **Stagnations-Exit:** ≥ 5 Tage held + PnL zwischen ±2 % → close mit "Stagnation Exit"
  - **Replacement-Logik:** wenn Slots voll + neues Pending qualifiziert:
    - Score ≥ 90 + (Pocket Pivot OR Gap ≥2 %) + schwächste Pos ≥ +2 % im Plus
    - → schwächste mit "Replacement Exit" close, neue open im selben Run
  - **NICHT übernommen aus User-Brief:** MAX_HOLD 7d (Daten sagen 21d), Score-Sweet-Spot
    70-80 (Daten sagen 90-100=72 % WR, 70-80=42 % WR)
- **2026-06-07** **Bigdata.com MCP-Skills installiert** vom User. Workflow für
  Postmortem-Batches: bigdata-com:financial-research-analyst orchestriert FMP +
  WebSearch. Test mit AFRM + IBKR erfolgreich (Phase 2 ergänzt 40/132 analyzed).
- **2026-06-07** **Postmortems AFRM + IBKR** (Batch 5 v2): Macro-Risk-Off-Theme:
  - AFRM_2026-05-29: high_beta_breakout_macro_risk, fintech_consumer_credit_sensitivity,
    fundamentals_intact_but_stopped, rate_decision_window_risk
  - IBKR_2026-06-01: late_cycle_breakout_near_52w_high, macro_selloff_correlates_all_stocks,
    rate_beneficiary_paradox, score_top_decile_no_protection_in_macro
  - Output: knowledge/trade_postmortems.json + reports/{AFRM,IBKR}_postmortem_*.md (MD)
    + reports/{AFRM,IBKR}_company_brief_2026-06.docx (Word mit inline attribution)
- **2026-06-06** **Workflow-Hardening + Market-Regime-Backup:**
  - `apex_scan.yml`, `apex_equity.yml`, `apex_knowledge.yml`: Push-Step von
    `git stash` auf `/tmp`-Backup umgestellt (Stash-Pop-Conflict vermieden).
    Plus Push-Retry-Loop (5x mit exponential backoff). Bei Conflict im
    Push: `pull --rebase -X theirs` (Worker's Files gewinnen).
  - Scan-Cron 30→42 (off-peak, weg von GH-:30-Drossel-Zone).
  - **`apex_equity.maybe_refresh_market_regime()`**: wenn `apex_market.json.updated`
    >18h alt (= Scanner-Fail Donnerstag/Freitag), berechnet Equity das Regime
    via importierter `ApexScan.get_market_regime()`. Single-Source-of-Truth.
  - Root-Cause 2026-06-05 Push-Fail: Equity pushte 22:00 `apex_market.json`,
    Scanner stashte + pull-rebase + stash pop → CONFLICT auf market.json.
- **2026-06-05** **Trader-Migration auf Oracle Cloud Always-Free VM:**
  - Ubuntu 22.04 + E2.1.Micro (1 CPU, 1 GB RAM + 2 GB Swap, Public IP)
  - GitHub Deploy-Key fuer Push, `~/run_trader.sh` + cron `*/5 13-21 1-5`
  - Verlaesslicher als GH-Actions-Cron (echtes Linux-Cron, kein Throttling)
  - GH-Workflow `.github/workflows/apex_trader.yml` GELOESCHT (kein
    doppelter Trader). Andere Workflows (Scanner, Equity, Knowledge) bleiben
    auf GH.
- **2026-06-05** **Paper-Tab Komplett-Redesign:**
  - Open + Closed als kompakte ausklappbare Zeilen (vorher 12-Spalten-Tabelle).
    Header zeigt Logo+Ticker+Setup, Stats (Wert/Δ%/PnL), Chevron. Klick
    klappt Detail-Panel aus mit Entry/TP/SL/Shares/Trailing/Hold/etc.
  - **Pending + Verfallen-Sektionen** aus Paper-Tab entfernt (interner State,
    Dashboard zeigt nur Open+Closed).
  - **NEU: Activity Log** rendert `apex_trade_log.json` als lesbare Events
    (⏳ pending_added, 🟢 open, ✅/❌ close, 🟡 trailing, ⚫ expired, 🔄 revalidated)
  - Mode-Karte: Status-Pill mit pulsing Dot, cyan/orange Top-Border
  - Equity-Karte: 28px Hero-Number + ▲/▼ Delta zum Start-Kapital
  - Mobile: 2-Zeilen-Layout via Flex (vorher Grid-Overflow), Ticker-Logos via
    FMP-Image-URL, sw.js bis v23.
- **2026-06-04** **`apex_open_positions.json` Single-Source-of-Truth fuer Signal-Status:**
  - apex_equity.compute_open_positions() schreibt fuer jedes nicht-geschlossene
    Signal: status (pending/open/expired) + trigger_day + current_price + PnL%.
  - Dashboard History-Tab nutzt das File statt Alters-Heuristik. ARE/JCI/etc.
    zeigen jetzt echten Status (open D+1) statt heuristisch (pending).
  - SETUP_META.BREAKOUT.hold 15→21 (Frontend matched Backend).
- **2026-06-04** **yfinance 5m→1m Bars + Multi-Step-Fallback in Trader:**
  Realtime-Lag von ~4 Min auf <1 Min reduziert. Fallback-Kette: 1m → 5m →
  Yahoo v8 Chart API → daily. Plus expliziter Bugfix: `group_by='ticker'`
  zerschiess Single-Ticker-Schema → batch_prices returnte leer → current_price
  wurde seit ADI-Open NIE aktualisiert. Fix mit `_extract_series` Helper.
- **2026-06-04** **Konsistenz-Pass Trigger+Hold-Windows system-wide:**
  - `MAX_TRIGGER_DAYS = 3` (Paper-Trader zurück 1→3 = matched Equity/Backtest = 61.8 % BO-WR
    Messung). Re-Validation refresht signal_date bei wiederholter Emission.
  - `HOLD_DAYS_PER_SETUP` dict in Paper-Trader (BO=21, VCP=40, STAGE_2=60, SQ=20, MR=20, REV=40)
    statt fix 30 für alle. Matched apex_equity.py horizon_to_days.
  - Dashboard History Status-Logik: age ≤ 3d → ⏳ Pending, 3d<age≤hold → 🟢 Offen,
    age>hold → ⚫ Expired (vorher: hold cutoff hat alles abgedeckt = falsch).
  - Filter-Dropdown bekommt Pending-Option.
- **2026-06-04** **Trader hardening:** Re-Validation-Logik:
  alte Pendings nur überleben wenn (Ticker,Setup) in heutiger Scan erneut auftaucht →
  signal_date refresht. Sonst expired. Timestamps jetzt UTC mit Z-Suffix.
  Dashboard zeigt alle Trader-Zeiten in Europe/Berlin via toLocaleString. sw.js v12→v14.
  Erstanwendung: IBKR (3d) + ARE (2d) expired, ADI (1d frisch) getriggert @$437.58.
- **2026-06-04** **Phase C: Dashboard Paper Trading Tab** — neuer 3. Tab im `dashboard.html`
  liest `apex_positions.json`. Status-Header (Mode/Cash/Equity/PnL), 4 Tabellen:
  Open (mit live PnL + Trail-Status), Pending (warten auf Trigger), Closed
  (Entry/Exit/Reason), Expired (max 30). sw.js v11→v12, PF zu DATA_FILES.
- **2026-06-04** **Phase B: Paper Trader (`apex_trader.py`)** — autonome Trading-Engine
  in Python + GH Actions Workflow `apex_trader.yml`. Liest `apex_signals.json`, wählt
  Top-1 BREAKOUT pro Scan-Tag (Telegram-äquivalentes Gate), schreibt Pending in
  `apex_positions.json`. Bei high≥buy_above → Trigger, Position auf @ Entry+max 0.5 %
  Slippage, $50 abgezogen von Cash. TP/SL/Trailing/Time-Exit jeden 20-Min-Run.
  Trailing: high≥Entry×1.08 → SL springt auf Entry×1.05 (one-shot). MAX_HOLD=30d,
  MAX_TRIGGER_DAYS=3. Test: 3 Pendings (IBKR/ARE/ADI) → alle getriggert Lauf 2,
  Cash $300→$150. eToro-Mode via TRADING_MODE env var (Stub für live).
  **Löst Backlog-Item 1** (Pending-Status für Dashboard automatisch).
- **2026-06-03** **Phase A: Obsidian Brain (`apex_brain.py`)** — autonomer Markdown-Writer
  in lokales Vault `./vault/` (gitignored). Liest `apex_signals.json`,
  `apex_equity_results.json`, `knowledge/trade_postmortems.json`, `apex_market.json`.
  Modi: `--signals` (idempotent), `--postmortems` (regen), `--weekly`, `--market`,
  `--learnings`. Erstrun: 205 trade-notes, 38 postmortems, 1 weekly, 1 market,
  68 lesson-tag-Aggregate. **Keine Eingriffe in Live-Code.**
  `.gitignore` UTF-16→UTF-8 fixed (war fuer git unlesbar).
- **2026-06-01** Sektor-Enrichment-Fix: retry "Unknown", cache nur Erfolge
- **2026-05-30** Dashboard Light-Mode Setup-Bar-Bug gefixed (drawSetup mit `sigsAll` statt `allR`),
  MOMO im Backtest auf opt-in
- **2026-05-29** Telegram-Builder mit Catalyst-Flags + MR-Group ergänzt
- **2026-05-29** Dashboard renderToday: zeigt letzten Scan persistent (nicht strikt heute)
- **2026-05-28** History-Tab: Catalyst-Badges auf jeder Row
- **2026-05-28** Track 2 = Telegram-Pushed (top-2/day) statt broad-quality. Compare-Card zurück.
- **2026-05-28** VCP ATR-Kontraktion 30 → 20 (live + Backtest-Default)
- **2026-05-28** History Entry-Spalte zeigt „Entry → Ziel"
- **2026-05-27** SW-Cache-Bumps + history Status-Filter (Open/Closed/Expired)
- **2026-05-22** **BREAKOUT base-Cap 8 → 22 setup-spezifisch** (Live + Backtest)
- **2026-05-22** **Telegram-Gate von relax-basiert → score-basiert**, TG_MIN_RR 2.0 → 1.5
- **2026-05-22** **Phase H: Mean-Reversion Setup live**, equity ACTIVE_SETUPS expanded
- **2026-05-21** Scan-Schedule 17:30 UTC → 20:30 UTC (nach US-Close)

---

## 9b. Konsistenz-Konstanten (System-weit, NICHT divergieren)

| Konzept | Wert | Quelle der Wahrheit | Wo verwendet |
|---|---|---|---|
| **TRIGGER_WINDOW** | 3 Trading-Days | apex_equity.py L100 | Paper-Trader, Dashboard History, Backtest v2 |
| **HOLD_DAYS BREAKOUT** | **30** | apex_trader.py + apex_equity.py + Backtest | **07-29 15/21->30** (Hold-Sweep-Optimum, +60pp/2J, behebt #7). ALLE DREI Quellen jetzt 30 |
| **HOLD_DAYS VCP** | 40 | dito | dito |
| **HOLD_DAYS STAGE_2** | 60 | dito | dito |
| **HOLD_DAYS SHORT_SQUEEZE** | 20 | dito | dito |
| **HOLD_DAYS MEAN_REVERSION** | 20 | dito | dito |
| **DUPLICATE_WINDOW_DAYS** | 3 | ApexScan.py L45 | Scanner — skipped Signals die in 3d schon emittiert wurden |
| **MAX_POSITIONS** (Paper, total) | 7 | apex_trader.py | bumped 5→7 für Hybrid-Test 2026-06-12 |
| **SWING_MAX_POSITIONS** | **4** (= 7−3) | apex_trader.py | 07-15 Option-B: gilt nur noch fuer Momentum-Filler — **Scanner-BREAKOUT nutzt ALLE freien Slots** |
| **INTRADAY_RESERVED_SLOTS** | **3** | apex_trader.py | 07-10 2→3; Intraday darf bis dahin schwache Swings verdraengen (max 1/Run) |
| **CLOSE_COOLDOWN_DAYS** | 5 | apex_trader.py | 06-23: gerade geschlossener Ticker 5d gegen Re-Entry gesperrt (Anti-Churn) |
| **CAPITAL_INITIAL** | $400 | apex_trader.py | bumped 300→400 + $100 virtual deposit |
| **HOLD_DAYS_PER_SETUP.MOMENTUM** | 7 | apex_trader.py | Momentum-Filler-Hold, schnelle Rotation |
| **Momentum-Filler-Cache** | 6h | apex_trader.py MOMENTUM_CACHE_MAX_AGE_H | yfinance-Schutz, max 2 Downloads/Tag |
| **Source-Field** | "scanner" \| "momentum_filler" | Position-Dict + pending-Dict | für Performance-Trennung |
| **TG-Send-Modus** | „no signal"-Message wenn 0 neue | ApexScan.py L1875-1878 | Falls Telegram-Channel still ist: Scanner OK, nur alle Tickers in 3d-Duplicate-Filter |
| **TRADING_MODE** | `paper` \| `live_dry` \| `live` | env-var, run_trader.sh | seit 07-06 auf `live` (eToro Demo-Portfolio) |
| **ETORO_ENV** | `demo` \| `live` | env-var | derzeit `demo` — virtuelles $100k Konto |
| **STAGE_2_ENABLED** | False | ApexScan.py L51 | 2026-07-08 disabled, kein Edge |
| **TECH_QQQ_GATE_ENABLED** | True | ApexScan.py L56 | 07-08 live: skip Tech/Comm-BO wenn `qqq_perf_20 < 0` |
| **SECTOR_RS_GATE_ENABLED** | True | ApexScan.py L~58 | **07-15 live**: skip Tech/Comm-BO wenn Sektor-ETF (XLK/XLC) `sector_momentum < 0` UND kein starker Catalyst. Deckt die Zelle die QQQ verpasst (Semi-Selloff). Rollback = False |
| **TG_SWEET_BAND** | (90, 120) | ApexScan.py L~71 | **07-11 live**: Telegram-Top-2 bevorzugt BREAKOUT im Band; 130+ ans Ende |
| **PICK_BAND** | (90.0, 120.0) | apex_trader.py L~48 | **07-15 live**: Trader-Pick-Ranking analog TG_SWEET_BAND (vereinheitlicht). Rollback = None |
| **TG_MIN_SCORE BREAKOUT (Gate 80)** | **80** | ApexScan.py + apex_trader.py | **07-22 70->80**: sub-80 = 33-37% WR, +2.2pp. SCAN_MIN_SCORE bleibt 70 (Equity misst sub-80 weiter). VCP/SQUEEZE-Gates unveraendert (70/65) -> tauchen an BO-armen Tagen in TG auf (kosmetisch, Trader handelt nur BREAKOUT) |
| **VCP_PICK_PRIORITY** | True | apex_trader.py | **07-23 live**: VCP-Kandidaten (cat_vcp_strength>0) VOR dem Band gepickt (_pick_rank/_tg_rank/_band_rank). +2.4pp WR, VCP-Anteil 35->58%. Schema A (Score) auch: vcp_strength>0 @ +8. Rollback = False |
| **RESCUE_REQUIRE_ABOVE_VWAP** | True | apex_trader.py | **07-22 live**: EOD->SWING-Rescue nur wenn ueber VWAP (Rescue war netto -1.01pp). Rollback = False |
| **SL_RECONCILE_ENABLED** | True | apex_trader.py sync | **07-29 live**: gleicht eToro-SL jeden Run mit Paper-Stop ab, re-pusht bei Drift >0.3%. Absicherung nach dem update_sl_tp-404-Bug. Rollback = False |
| **eToro Update-SL/TP-Endpoint** | `PATCH /api/v2/trading/{env}/positions/{positionId}` | etoro_client.py | **07-29 FIX** (war v1 -> [404], jeder Trailing-Push verschluckt seit Live-Start). camelCase-Body {stopLossRate,takeProfitRate,stopLossType}. Erfolg 202. ACHTUNG: Open-Order will PascalCase, Update camelCase! |
| **EU_GUARD / INTRADAY_EU_ENABLED** | True / False | apex_trader.py | 07-17: EU-Live-Entry nur 07:00-15:15 UTC; EU raus aus Intraday-Universum. EU-Edge via Equity-Tracker messen (BACKLOG #23) |
| **INTRADAY_MAX_POSITIONS** | 4 | apex_trader.py | 07-10 2→4 |
| **INTRADAY_GAIN_MIN/MAX** | 1.0 / 6.0 % | apex_trader.py | **GAIN_MAX unter Verdacht** (BACKLOG #22): Mover wachsen aus dem Vorfilter heraus wenn sie am staerksten laufen |
| **INTRADAY_RANGE_POS_MAX** | 0.90 | apex_trader.py | Anti-Peak (07-10). **Unter Verdacht** (BACKLOG #22): starke Trends laufen am Hoch |
| **INTRADAY_ENTRY_CUTOFF_UTC** | "19:15" | apex_trader.py | 07-15: keine neuen Intradays kurz vor EOD (EQNR-Fall). EOD-Close-Cutoff bleibt 19:45 |
| **eToro-Close-Backfill** | 7d-Fenster | apex_trader.py `sync_etoro_positions` | **07-16 live**: holt echte Close-Rate/netProfit nach wenn Paper das Rennen gegen den API-Lag gewann (RHI-Bug) |
| **EU_GUARD_ENABLED** | True | apex_trader.py `_eu_entry_blocked` | **07-17 live**: EU-Live-Entry nur 07:00–15:15 UTC (EU-Boerse schliesst 15:30, wir laufen bis 21:00 → sonst Kauf auf 2.5h altem Close-Bar). Sitzt in `trigger_pending` nach dem Expiry-, vor dem Kauf-Check. Rollback = False |
| **INTRADAY_EU_ENABLED** | **False** | apex_trader.py | **07-17**: EU raus aus dem Intraday-Universum (825→719 Ticker) — 5m-Bars nach EU-Close eingefroren = Peak-Kauf auf totem Chart. EU-Edge wird stattdessen ueber den Equity-Tracker vermessen (BACKLOG #23). Rollback = True |
| **MOMENTUM-Bearish-Skip** | aktiv | apex_trader.py Step 3b | liest apex_market.json.mode, skip wenn BEARISH |
| **eToro-Auth-Mapping** | x-api-key = "Öffentlicher Schlüssel" · x-user-key = generierter Schlüssel-Wert | etoro_client.py | **VERDREHT vs Portal-Labels!** |
| **eToro-Fee** | ~$1 open + $1 close (normal Demo) | eToro | Live Smart-Portfolio angeblich fee-free (unverifiziert) |

---

## 9. Konstante Wahrheiten (Postmortem-Lehren, alle CONFIRMED-Pattern)

- **Earnings-Adjacency-Risk gilt für ALLE Setups** (SW/BTU/XRAY/FLS/S) — BREAKOUT-Earnings-Blackout
  fehlt als Hard-Filter, ist nur Score-Komponente
- **Sector-Momentum-Tailwind** funktioniert (März-Energy-Cluster 9 Wins simultan: WTI $55→$93)
- **closing_strength < 0.5 = Spike-Fade-Warner** (SM, einige Bezüge)
- **High-Score (>120) ≠ Conviction** — oft Vola-Extrem. 38 % aller Verluste haben Score ≥85
- **Pocket Pivot = institutionelle Akkumulations-Bestätigung** (7/7 Winners in Batch 4)
- **Analyst-Downgrade-Cascade vor Signal = Falling-Knife-Risk** (CACI, HCA, APH)
- **Stop-Adjusts für extended Stocks**: tight stop tötet bei vola-Stocks (WDC)
- **REVERSAL strukturell defekt** weil fundamental-getriebene Drops gekauft werden
  (Earnings-Miss, Analyst-Downgrade, Insider-Distribution)

---

*Diese Datei lebt. Bei Inkonsistenzen mit den Code-Files: Code-Files sind autoritativ,
diese Datei wird aktualisiert.*
