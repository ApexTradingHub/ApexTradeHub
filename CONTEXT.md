# ApexNext — Session Context & Active State

**Zweck:** Persistenter Schnappschuss des aktuellen Stands. Wenn das Chat-Kontextfenster
komprimiert wird, kann eine neue Session diese Datei lesen und **kalt aufgreifen** ohne den
ganzen Verlauf zu kennen. Wird laufend aktualisiert.

**Letztes Update:** 2026-06-08 (Trader Phase 1+2 + Bigdata-Skills installed + Cron-Slot-Shift)

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
| **BREAKOUT** 🔵 | CONFIRMED | 20d-High-Breakout, base-Cap setup-spezifisch ≤22 | WR 58 % / PF 2.56 (n=67) ✅ |
| **STAGE_2** 🚀 Trend | aktiv | Weinstein Long-Base | n=4 signals lifetime — rar by design |
| **VCP** 🔹 Bounceback | aktiv (gelockert 28.5.) | Minervini, ATR-Kontraktion 30→20 | Backtest WR 88.9 % n=9 |
| **SHORT_SQUEEZE** 🔥 Bet | aktiv (strict, ≥15 % short) | praktisch nie feuernd | n=0 lifetime |
| **MEAN_REVERSION** 🟢 Dip | aktiv | RSI<38 Pullback im Uptrend | n=0 lifetime (zu jung) |
| ~~REVERSAL~~ | **disabled** | Legacy, 30 % WR strukturell defekt | sterbende Legacy-Positions in n |

**Aktuelle Tuning-State der BREAKOUT-Hard-Exits (relax=0):**
- `base_range`: ≤22 wenn BREAKOUT-Kandidat, ≤8 für andere Setups (setup-spezifisch, 28.5.)
- `vol_ratio`: ≥1.0 (relax=0), ≥0.7 (relax=1)
- RSI-Zone: 45-68 (relax=0), 40-72 (relax=1) — TENTATIVE: RSI 60-65 underperforms

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

## 4. Aktive Diskretionäre Positionen / Watch

- **APP** (gekauft $560 am 27.5., +250 USD) — TP $600 gehittet, „buy & hold to $669"-Plan,
  Stop auf $560 (Break-Even) hochziehen empfohlen
- **GEV** (Diskussion vom 1.6.): Buy-Zone unter $950 erreicht ($944 aktuell), aber Pullback noch
  im Gange (RSI 29, schloss am Tagestief). Empfehlung: halbe Position @$944 mit Stop $895, oder
  warten auf grünen Tagesschluss als MR-Turn-Bestätigung.

---

## 5. Offene Backlog-Items (siehe `BACKLOG.md`)

1. **Pending/Triggered-Status für offene Signale** — wird durch **Phase B (Paper Trader)**
   gelöst: Trader läuft alle 20 min, schreibt `triggered`-Flag in `apex_positions.json`.
   Dashboard liest direkt. Item bleibt Backlog bis Phase B live.
2. **MOMO-Setup** — getestet (PF 1.51 < 2.0 = verworfen), Code als opt-in (`--only-setup MOMO`)
   in `apex_backtest_v2.py` belassen. Re-test bei klaren DELL-artigen Misses.

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

## 7. Aktueller Daten-Stand (2026-06-08)

- **Lifetime Trades:** 132 | WR 46.2 % | PF 1.84 (unverändert seit 06-03, keine neuen Closes)
- **Postmortems analysiert:** **40/132** (92 pending) ← +2 (AFRM/IBKR via Bigdata-Workflow)
- **Market Regime aktuell:** **MIXED** (SPY=OK | QQQ=OK) — war BULLISH bis 06-04, seit Macro-Risk-Off 06-05 MIXED
- **CONFIRMED Setups:** BREAKOUT (n=77, WR 59.7 %, PF 2.70), REVERSAL (n=54, disabled)
- **30d Window:** **WR 54.5 % / PF 2.49** (n=33) — Drift +8.3pp vs lifetime, weiter stark
- **14d Window:** WR 63.6 % / PF 4.20 (n=11) — unverändert
- **7d Window:** WR 0.0 % (n=2) — IBKR/AFRM Stops fielen ins Macro-Schock-Fenster
- **🚀 STAGE_2 Anomalie (2026-06-08):** 10 STAGE_2 offen (defensive Sektor-Names:
  LIN/EW/COST/KIM/EQR/ASB/GL), Signal-Welle in MIXED-Regime begonnen.
  **0 STAGE_2 jemals geschlossen lifetime** — wir wissen nicht wie sie performen.
  Aktuelle PnL alle leicht negativ bis +4 %. Defensive-Rotation-Hypothese.
- **BREAKOUT 30d:** **WR 70.8 %** (n=24) — Drift +11pp vs lifetime, weiter stark aber abgekühlt
- **MEAN_REVERSION:** erster geschlossener Trade SBUX -2.38 % D+2 SL
- **REVERSAL 30d:** WR 10.0 % (n=10) — Legacy stirbt
- **Offene Positionen (Equity-Sicht, nach IBKR/AFRM-Closes):** 5 BREAKOUT
  (AXTA/ADI/FANG/JCI/ARE)
- **Paper-Trader-Sicht:** AXTA open (D+1 via Oracle-Cron getriggert), ADI Stop Loss

---

## 8. Recent Major Code-Changes (chronologisch, für Re-Bauchgefühl)

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
| **TG-Send-Modus** | „no signal"-Message wenn 0 neue | ApexScan.py L1875-1878 | Falls Telegram-Channel still ist: Scanner OK, nur alle Tickers in 3d-Duplicate-Filter |

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
