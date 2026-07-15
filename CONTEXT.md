# ApexNext — Session Context & Active State

**Zweck:** Persistenter Schnappschuss des aktuellen Stands. Wenn das Chat-Kontextfenster
komprimiert wird, kann eine neue Session diese Datei lesen und **kalt aufgreifen** ohne den
ganzen Verlauf zu kennen. Wird laufend aktualisiert.

**Letztes Update:** 2026-07-15 (eToro LIVE · Brief AP1-4 done · Score: Sektor-Bonus raus + TG-Band 90-120 · Slot-Option-B: Scanner nutzt alle freien Slots + Intraday-Verdraengung · Intraday-Entry-Cutoff 19:15)

---

## 1. Workflow-Goldene Regeln (NICHT verletzen)

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
- **⚠ RSI 60-65 = neue DEADZONE (CONFIRMED 2026-06-15, n=40, WR 47.5% / −10pp lift)** — innerhalb der erlaubten Zone gibt's ein Mid-Range-Loch. Action ausstehend: Score-Penalty (~-5) **nach Backtest**. NICHT als Hard-Skip (Loser-Anteil 52.5% vs Baseline 42.5% = killt fast 1:1 Winner mit). Pattern: U-Kurve, RSI 50-60 (+10pp) und RSI 70+ (+18pp) ggf. Bonus-Kandidaten bei groesserem n.
- **perf_120 Buckets (REALIGN):** <0 = -15 (WEAK), 0-25 = **-3 (DEADZONE)**, 25-50 = +15 (SWEET), >50 = +8
- **TECH_QQQ_GATE live (2026-07-08):** Skip BREAKOUT wenn Sektor Tech/Communication UND
  `market_regime.qqq_perf_20 < 0`. Backtest: Tech+QQQ<0 = WR 14%/PF 0.56 (n=7) → nach Gate
  WR 57.1→59.8%, PF 2.29→2.53, Signal-Loss 6%. Flag `TECH_QQQ_GATE_ENABLED = True`.
  Vorbehalt n=7 klein, Monitoring nötig.
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
  - **Cron: `*/15 13-21 * * 1-5` auf Oracle-VM** (GH-Workflow geloescht 2026-06-05)
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
  git push. Cron `*/15 13-21 * * 1-5`. Robust gegen GH-Throttling.
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
  - GitHub Deploy-Key fuer Push, `~/run_trader.sh` + cron `*/15 13-21 1-5`
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
| **HOLD_DAYS BREAKOUT** | 21 | apex_equity.py horizon_to_days | Paper-Trader, Dashboard, Backtest |
| **HOLD_DAYS VCP** | 40 | dito | dito |
| **HOLD_DAYS STAGE_2** | 60 | dito | dito |
| **HOLD_DAYS SHORT_SQUEEZE** | 20 | dito | dito |
| **HOLD_DAYS MEAN_REVERSION** | 20 | dito | dito |
| **DUPLICATE_WINDOW_DAYS** | 3 | ApexScan.py L45 | Scanner — skipped Signals die in 3d schon emittiert wurden |
| **MAX_POSITIONS** (Paper, total) | 7 | apex_trader.py | bumped 5→7 für Hybrid-Test 2026-06-12 |
| **SWING_MAX_POSITIONS** | 5 | apex_trader.py | Option B 06-19: Scanner+Momentum max 5 (= 7−2) |
| **INTRADAY_RESERVED_SLOTS** | 2 | apex_trader.py | Option B 06-19: für Intraday-Catcher reserviert |
| **CLOSE_COOLDOWN_DAYS** | 5 | apex_trader.py | 06-23: gerade geschlossener Ticker 5d gegen Re-Entry gesperrt (Anti-Churn) |
| **CAPITAL_INITIAL** | $400 | apex_trader.py | bumped 300→400 + $100 virtual deposit |
| **HOLD_DAYS_PER_SETUP.MOMENTUM** | 7 | apex_trader.py | Momentum-Filler-Hold, schnelle Rotation |
| **Momentum-Filler-Cache** | 6h | apex_trader.py MOMENTUM_CACHE_MAX_AGE_H | yfinance-Schutz, max 2 Downloads/Tag |
| **Source-Field** | "scanner" \| "momentum_filler" | Position-Dict + pending-Dict | für Performance-Trennung |
| **TG-Send-Modus** | „no signal"-Message wenn 0 neue | ApexScan.py L1875-1878 | Falls Telegram-Channel still ist: Scanner OK, nur alle Tickers in 3d-Duplicate-Filter |
| **TRADING_MODE** | `paper` \| `live_dry` \| `live` | env-var, run_trader.sh | seit 07-06 auf `live` (eToro Demo-Portfolio) |
| **ETORO_ENV** | `demo` \| `live` | env-var | derzeit `demo` — virtuelles $100k Konto |
| **STAGE_2_ENABLED** | False | ApexScan.py L51 | 2026-07-08 disabled, kein Edge |
| **TECH_QQQ_GATE_ENABLED** | True | ApexScan.py L53 | 2026-07-08 live, verhindert Tech-Breakouts bei QQQ<0 |
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
