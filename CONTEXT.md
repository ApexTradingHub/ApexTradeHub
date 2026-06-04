# ApexNext — Session Context & Active State

**Zweck:** Persistenter Schnappschuss des aktuellen Stands. Wenn das Chat-Kontextfenster
komprimiert wird, kann eine neue Session diese Datei lesen und **kalt aufgreifen** ohne den
ganzen Verlauf zu kennen. Wird laufend aktualisiert.

**Letztes Update:** 2026-06-04 (Phase A + B + C shipped)

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
| **⚡ Pocket Pivot Edge** | 21 | MED → CONFIRMED bei n≥30 | **+25pp WR** | Hard-Filter-Kandidat: skip BREAKOUT ohne PP |
| **🎯 analyst_upside>15 NEGATIV** | 19 | MED → CONFIRMED bei n≥30 | **−22.7pp WR** (vorher -17, **stärker negativ**) | Catalyst-Score-Delta entfernen oder invertieren |
| **Score-Cap-Hypothese** | 90-100: n=18 (72.2 %)<br>100+: n=24 (70.8 %) | TENTATIVE | 100+ KEINE bessere WR als 90-100 (bestätigt) | Telegram-Ranking-Score-Cap bei 100 |
| **🔵 BREAKOUT × RSI≥70** | 12 | HYPOTHESIS (neu 2026-06-04) | **+16pp WR (75 %)** | RSI-Obergrenze 68→72 lockern für BREAKOUT |
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
  - Cron: `*/20 13-21 * * 1-5` (alle 20 min während US-Markt, GH Actions)
  - Freshness-Gate: Signale älter als MAX_TRIGGER_DAYS=3d werden gar nicht erst aufgenommen
  - State: `apex_positions.json` (pending/open/closed/expired + stats)
  - Journal: `apex_trade_log.json` (append-only, alle Events)
  - eToro-API als Stub (TRADING_MODE env var: paper|live)
  - Löst Backlog-Item 1 (Pending-Status) als Side-Effect
- ✅ **Phase C — Dashboard Paper-Tab** (`dashboard.html`) — shipped 2026-06-04
  - Neuer 3. Tab "📈 Paper Trading", liest `apex_positions.json`
  - Status-Header (Mode/Cash/Equity/PnL), 4 Tabellen (Open/Pending/Closed/Expired)
  - sw.js v11→v12, PF zu DATA_FILES
- ⏳ **Phase D — Equity-Research-Plugin** (optional, hängt von A)

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

## 7. Aktueller Daten-Stand (2026-06-04)

- **Lifetime Trades:** 125 | WR 45.6 % | PF 1.81
- **Postmortems analysiert:** 38/125 (87 pending)
- **CONFIRMED Setups:** BREAKOUT (n=71, WR 59.2 %, PF 2.66), REVERSAL (n=54, disabled)
- **30d Window:** WR 47.2 % / PF 1.95 (n=36) — flat lifetime
- **14d Window:** **WR 90.9 % / PF 16.37 (n=11)** ⚡ BREAKOUT-Welle, regime-getrieben
- **BREAKOUT 30d:** **WR 76.2 %** (n=21) — Drift +17pp vs lifetime, neuer Rekord
- **REVERSAL 30d:** WR 6.7 % (n=15) — Drift -21pp, Legacy stirbt schneller als erwartet
- **Offene Positionen (Equity-Sicht):** 7 BREAKOUT + 1 STAGE_2 (ADI/FANG/JCI/ARE/IBKR/TSM/AYI/EW)
- **Paper-Trader-Sicht:** 1 BREAKOUT open (ADI), 2 expired (IBKR/ARE), 0 pending
  - Diskrepanz ggü Equity ist erwartet (Trader = Top-1/Tag + Re-Validation, Equity = alle Signale + strict 3d)

---

## 8. Recent Major Code-Changes (chronologisch, für Re-Bauchgefühl)

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
