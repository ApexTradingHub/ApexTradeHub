# ApexNext — Roadmap zum Go-Live (eToro Agent-Portfolio)

**Erstellt:** 2026-06-17
**Ziel:** Hybrid-Trader produktiv live auf eToro Agent-Portfolio, mit echtem Geld.
**Frühestens-Datum:** Q4 2026 (Okt-Nov) bei striktem Kurs, realistisch Q1 2027.

---

## Aktueller Stand (Baseline 2026-06-17)

- Hybrid-Trader live seit 06-12 (5 Tage), **n=10 closed** (Paper)
- WR 70 % (n=10 → statistisch Rauschen), Equity ~$410 (von $400)
- Lifetime-Knowledge: 150 Trades, BREAKOUT WR 58 %, PF 2.55
- Markt-Regime: BULLISH durchgehend seit Trader-Start
- FRED-Macro-Awareness live, MEAN_REVERSION disabled

---

## Go-Live-Gates (ALLE müssen ✅ sein)

| # | Gate | Aktuell | Trigger zum Check |
|---|---|---|---|
| 1 | **n ≥ 50 Hybrid-Closed-Trades** | 10/50 | mit jedem Trader-Close hochzählen |
| 2 | **30d-Rolling WR ≥ 55 %** | 70 % (n=10, noise) | nach Gate 1 sinnvoll messbar |
| 3 | **30d-Rolling PF ≥ 1.5** | n.a. (Sample zu klein) | mit Gate 1 |
| 4 | **Min. 1 Regime-Wechsel überlebt** (BULLISH → MIXED/RISK_OFF und zurück) | nein, durchgehend BULLISH | wann immer Markt wechselt |
| 5 | **BACKLOG #6 entschieden** (Same-Day-Trigger-Stop-Ambiguität) | offen | Phase 1 unten |
| 6 | **BACKLOG #7 entschieden** (Horizon 15-vs-21-Konflikt) | offen | Phase 1 unten |
| 7 | **Pre-Market-Trigger-Artefakt entschieden** (Cron 13:15 UTC vs Markt-Open 13:30) | "bleiben erstmal so" | wenn Live, MUSS entschieden sein |
| 8 | **Schatten-Live (dry_run=True) ≥ 1 Woche grün** | nicht gestartet | Phase 4 |

---

## Phase 1 — Bug-Closure (Soft-Deadline: 2026-07-15)

Vor jedem Live-Schritt: die offenen System-Bugs entweder fixen oder explizit als "akzeptiert mit dokumentiertem Impact" abhaken.

### 1.1 Same-Day-Trigger-Stop (BACKLOG #6)
- **Entscheidung:** Heuristik bauen (`if open >= entry: trigger-then-stop`; sonst nicht-im-Trade-wenn-Low) ODER explizit akzeptieren mit Workaround "CDNS-artige Trades manuell schließen"
- **Empfehlung:** Heuristik einbauen (1 Std), saubere Equity-Stats
- **Aufwand:** 1-2 Std + Backtest-Validierung

### 1.2 Horizon-Konflikt 15 vs 21 (BACKLOG #7)
- **Entscheidung:** Code anpassen `HORIZON_DAYS["1-3 weeks"] = 21` ODER Doku auf 15 zurücksetzen
- **Empfehlung:** Backtest beide Werte (2J, Ein-Knopf-Pro-Lauf-Regel), Gewinner-Wert ins Code
- **Aufwand:** 1 Tag (Backtest + Live-Anpassung + Knowledge-Recompute)

### 1.3 Pre-Market-Trigger-Artefakt
- **Entscheidung 1:** Cron-Start auf 13:35 UTC (= 5 Min nach Markt-Open) — yfinance liefert dann echte 1m-Bars vom Open
- **Entscheidung 2:** Code-Patch — explizit `pre_market_skip` Check + `prepost=False`
- **Entscheidung 3:** Realistische Fill-Sim — `entry_actual = max(signal_entry, today_open)` bei Gap-Up
- **Empfehlung:** Erst Cron-Shift (cheap fix), dann Fill-Sim für Realismus
- **Aufwand:** 30 Min Cron + 2 Std Fill-Sim

---

## Phase 2 — Daten-Akkumulation (Soft-Deadline: 2026-09-30)

Parallel zur Infrastruktur-Arbeit: einfach laufen lassen.

- Hybrid-Trader weiter live (Paper)
- Wöchentlicher Check der Gates 1-4
- Bei jedem signifikanten Drawdown (>10 %) → Postmortem-Batch
- Knowledge-Refresh wöchentlich (Cron läuft Sa 06:47 UTC)
- Bei Regime-Wechsel: explizit dokumentieren wie Trader sich verhält

**Erwartet:** ~5-8 Closes/Woche × 8-12 Wochen = 40-100 weitere Closes. n=50 sollte erreichbar sein bis ~Anfang August, n=80 bis Oktober.

---

## Phase 3 — Infrastructure-Setup (parallel, kann sofort starten)

eToro-Integration bauen, aber **NICHT** aktivieren. Code-Pfad parallel zum Paper-Mode.

### 3.1 eToro-Setup (Account-Seite)
- [ ] eToro Builders Portal Account registrieren (`builders.etoro.com`)
- [ ] App registrieren (Production-Tier, falls möglich; sonst Personal-API-Key)
- [ ] API-Key + Secret in VM `~/.bashrc` (`ETORO_API_KEY`, `ETORO_API_SECRET`)
- [ ] Smart-KI / Agent-Portfolio bei eToro öffnen (NICHT funden!)
- [ ] Portfolio-ID notieren

### 3.2 etoro_client.py bauen
- [ ] REST-Wrapper mit 5-6 Endpoints:
  - `get_portfolio_balance()` → cash + positions value
  - `get_open_positions()` → ticker, qty, entry, current
  - `open_position(ticker, size_usd, sl, tp)` → POST /api/v2/trading/execution/orders
  - `close_position(position_id)` → soft close
  - `update_sl_tp(position_id, sl, tp)` → für trailing-ladder
  - `get_quote(ticker)` → realtime price (alternativ via WebSocket)
- [ ] Rate-Limit-Handling, retry-loop (wie GH-Workflows)
- [ ] Dry-Run-Mode: alle Calls loggen, nichts ausführen

### 3.3 apex_trader.py erweitern
- [ ] `TRADING_MODE = "paper" | "live_dry" | "live_etoro"`
- [ ] `live_dry`: Trader rechnet normal, `etoro_client` Calls werden ge-mocked + geloggt
- [ ] `live_etoro`: echte Calls
- [ ] **Reconciliation pro Run:** Apex-State (apex_positions.json) vs eToro-State diff'en. Bei Mismatch → ALERT + skip run
- [ ] Hard-Cutoff: bei Drawdown ≥ 50 % vom Initial-Capital → automatisch `TRADING_MODE = "paper"` zurück

**Aufwand:** 3-5 Tage, kann zwischen Trader-Runs entwickelt werden ohne Live-Risiko.

---

## Phase 4 — Schatten-Live / Dry-Run (1-2 Wochen)

Nach Phase 1+3 abgeschlossen + Gates 1-4 ≥ 80 % erfüllt.

- `TRADING_MODE = "live_dry"` auf VM-Trader-Cron
- Trader macht weiter Paper-Decisions, aber gleichzeitig:
  - Würde-Calls werden als JSON in `etoro_dry_log.json` geschrieben
  - Tatsächliche eToro-Portfolio-Daten werden gepullt (read-only ok)
  - Daily-Reconciliation: was hätte Trader gemacht vs was hätte eToro gefüllt
- **Was wir lernen:** echte Slippage, Fill-Verfügbarkeit, Pre-Market-Verhalten am realen Broker
- **Gate zum nächsten Schritt:** Reconciliation-Diff < 5 % auf Win-Rate über 1 Woche

---

## Phase 5 — Go-Live (Trigger-basiert, kein Datum)

**Alle Gates 1-8 grün. Sonst nicht.**

### 5.1 Initial-Kapital
- **Fixe Zahl, nicht %:** Empfehlung **$1000-$2000**
- Sei ehrlich: würde dich Totalverlust schmerzen? Wenn ja → reduzier
- Position-Sizing-Logik bleibt: $X / 7 Slots
- Bei $1000: $143/Position. Bei $2000: $286/Position.

### 5.2 Switch-Procedure
1. Letzten Knowledge-Refresh
2. Postmortem aller pending Trades
3. `TRADING_MODE = "live_etoro"`
4. Erster Live-Trade: visuell observieren (nicht aus Telegram heraus, sondern manuell auf eToro-Dashboard)
5. Erste Woche: täglich Reconciliation prüfen

### 5.3 Hard-Limits (NICHT verhandelbar)
- **DD ≥ 50 % vom Initial:** automatisch zurück auf Paper, Audit-Pause
- **3 Stop-Losses in Folge:** Telegram-Alert, manuelle Review
- **Unbekannter eToro-Error:** Position-Close-Versuch, dann Paper-Fallback
- **VM down ≥ 2 Std:** auto-skip Trigger (kein Recovery-Trading auf nachgeholten Signalen)

### 5.4 Review-Kadenz
- **Täglich:** Telegram-Macro-Header + offene Positionen
- **Wöchentlich:** Equity-Delta vs Paper-Sim
- **Monatlich:** Postmortem-Batch + Knowledge-Refresh
- **Quartalsweise:** Strategie-Audit (Gates noch erfüllt?)

---

## Kill-Switch-Kriterien (Live abbrechen, zurück auf Paper)

- 30d-Rolling-WR fällt unter 45 %
- 3 Wochen am Stück negativ
- DD ≥ 30 % vom Initial-Capital (vor 50 %-Hard-Cutoff)
- Du verlierst Schlaf wegen offener Positionen — psychologische Linie
- Major-Bug entdeckt (z.B. eToro-API-Verhalten anders als simuliert)

---

## Was JETZT machbar ist (kein Wait-State)

Du musst NICHT warten bis n=50 um produktiv zu sein. Heute machbar:
1. **Phase 1.3 Cron-Shift** (5 Min) — Pre-Market-Trigger sofort weg
2. **Phase 3.1 eToro-Builder-Account** (15 Min) — Build-Portal Registrierung
3. **Phase 3.1 Smart-KI-Portfolio öffnen** (10 Min) — leer, ohne Funding

Sobald du Builder-Account hast → ich kann etoro_client.py-Skizze bauen (Phase 3.2).

---

*Diese Roadmap lebt. Bei Status-Updates: direkt hier bearbeiten, BACKLOG.md für neue Findings.*
