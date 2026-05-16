# Dashboard Update — Prompt für Claude

Kopiere diesen Prompt in eine neue Claude-Code-Session im Ordner `C:\Users\Niklas\TradeBot\ApexNext\`.

---

## Aufgabe

Update das `dashboard.html` (Vanilla-HTML/JS-PWA mit Chart.js) für das neue Phase-G-Setup-System UND füge eine 2. Page hinzu.

## Kontext: Was sich in ApexScan.py geändert hat

Wir haben das alte Setup-System (BREAKOUT + REVERSAL + PRE-ROCKET + POSITION) ersetzt durch ein **4-Setup-System**. REVERSAL ist komplett raus.

**Neue Setup-Typen** (Priority-Order STAGE_2 > VCP > SHORT_SQUEEZE > BREAKOUT):

| Setup-Key | Anzeige-Name | Emoji | Charakteristik |
|---|---|---|---|
| `STAGE_2` | Stage-2 Breakout | 🚀 | Weinstein long-base, 8-16 Wochen, big moves |
| `VCP` | VCP Breakout | 🔹 | Minervini volatility contraction, 4-8 Wochen |
| `SHORT_SQUEEZE` | Short Squeeze | 🔥 | >15% short interest, 2-4 Wochen |
| `BREAKOUT` | Breakout | 🔵 | 20d-high breakout, 1-3 Wochen |

**Neue Felder in apex_signals.json** (zusätzlich zu den bekannten ticker/setup/price/buy_above/stop/target/rr/rsi/score):
- `movement_class`: STAGE_2_BREAKOUT / VCP_TIGHT / VCP_WIDE / SQUEEZE_HIGH / SQUEEZE_MED / POWER_BREAKOUT / EMERGING_BREAKOUT / WEAK_BREAKOUT
- `vcp_contraction`, `vcp_base_range` (nur bei VCP)
- `squeeze_short_pct`, `squeeze_perf_5d` (nur bei SHORT_SQUEEZE)
- `stage2_ma150_rise`, `stage2_base_width` (nur bei STAGE_2)
- `closing_strength` (0-1 — wo close im day-range)
- `inside_day` (boolean)
- Catalyst-Felder: `cat_pocket_pivot`, `cat_vol_climax`, `cat_gap_pct`, `cat_vcp_strength`, `cat_earnings_blackout`, `cat_earnings_beat`, `cat_earnings_next_days`, `cat_short_pct`, `cat_analyst_upside`

## Was zu tun ist

### A) Page 1 (existing dashboard) — Updates

1. **Setup-Filter/Stats** — die Code-Stellen die `BREAKOUT/REVERSAL/POSITION/PRE-ROCKET` referenzieren sind veraltet. Suche nach:
   ```js
   groups = {"BREAKOUT": [], "POSITION": [], "PRE-ROCKET": [], "REVERSAL": []}
   ```
   und ersetze mit den neuen 4 Setups.
   
2. **Stat-Zähler** wie `s-breakout`, `s-reversal` → ersetze REVERSAL durch z.B. `s-stage2`, `s-vcp`, `s-squeeze`.

3. **Setup-Badges** (CSS-Klassen `badge-blue` für BREAKOUT, `badge-teal` für REVERSAL) erweitern:
   - STAGE_2 → grüner/lila Badge (rare/special)
   - VCP → blauer Badge (quality)
   - SHORT_SQUEEZE → orange/rot Badge (momentum)
   - BREAKOUT → bestehender blauer Badge

4. **Optional**: Catalyst-Badges anzeigen (z.B. "⚡ Pocket Pivot", "🔥 Earnings Beat", "📈 +Analyst") wenn das jeweilige Feld true ist.

### B) Page 2 (NEU) — Trade History

Komplette Trade-History-Tabelle. Daten kommen aus `apex_equity_results.json` (geschlossene Trades) JOIN mit `apex_signals.json` (für signal_date, score, catalysts).

**Tabelle mit folgenden Spalten:**

| Spalte | Quelle |
|---|---|
| Signal Date | `apex_signals.json` → `date` |
| Ticker | `ticker` |
| Setup | `setup` mit Badge |
| Entry $ | `entry` (von equity) oder `buy_above` (von signal) |
| Exit Date | `apex_equity_results.json` → berechnet: `date` + `exit_day` Trading-Tage |
| Exit Reason | `exit_reason` (Take Profit / Stop Loss / Time Exit / Trailing Stop) |
| PnL % | `pnl_pct` (grün >0, rot <=0) |
| PnL $ | `pnl_usd` |
| Hold Days | `exit_day` |
| Score | von signal |

**Features:**
- Sortierbar (alle Spalten, Default: Signal Date desc)
- Filter: Setup (Multi-Select), Date Range, Exit Reason
- Summary-Stats oben: Total Trades, WR, PF, Total PnL, Avg Win, Avg Loss
- Per-Setup Mini-Stats (alleine WR pro Setup)
- Mobile-responsive

### C) Navigation

Tabs oder Top-Nav zwischen Page 1 ("Today" / "Aktuelle Signale") und Page 2 ("History" / "Trade-Log"). 
Empfehlung: einfache Tab-Bar oben mit URL-Hash-Routing (`#today` / `#history`) damit Page reloads keinen Tab verlieren.

## Wichtige Konstraints

- **NICHT brechen**: bestehende Stats-Cards, Equity-Charts, Sector-Doughnuts, Today-Signal-Liste sollen weiterhin funktionieren. Nur erweitern/ersetzen wo nötig.
- **Service Worker** (`sw.js`) updaten falls Page 2 separate Assets braucht — aktuell cacht er nur `dashboard.html` + 4 JSON-Files.
- **Mobile-First**: PWA wird mobil genutzt. Trade-History-Tabelle muss horizontal scrollbar sein auf kleinen Screens.
- **NaN-Safety**: `apex_signals.json` hat manche Felder als `null`. JS muss das handlen ohne crash.

## Test-Workflow

1. Code-Änderungen lokal machen
2. `python -m http.server 8080` in Repo-Ordner
3. Öffne `http://localhost:8080/dashboard.html`
4. Beide Pages testen
5. Wenn OK: commit + push zu master
6. Live-Dashboard ist `https://apextradinghub.github.io/ApexTradeHub/dashboard.html`

## Files

- **Bearbeiten:** `dashboard.html` (Page 1 updates + neue Page 2)
- **Bearbeiten:** `sw.js` (Service Worker — falls neue Cache-Files)
- **NICHT anfassen:** ApexScan.py, apex_backtest_v2.py, apex_equity.py, apex_catalysts.py, alle JSON-Files (das sind Daten, kommen vom Scanner)

## Aktueller Setup-Stand für Referenz

```
Backtest 2024-01 to 2026-04 (Phase G):
  Total:        179 trades | WR 59.8% | PF 2.04
  STAGE_2:       93 trades | WR 63.4% | PF 2.33 | AvgWin +10.01%
  BREAKOUT:      78 trades | WR 55.1% | PF 1.35 | AvgWin +4.38%
  SHORT_SQUEEZE:  6 trades | WR 66.7% | PF 4.02 | AvgWin +10.08%
  VCP:            2 trades | WR 50.0% | PF 28   | AvgWin +2.24%
```

Start mit Page 2 da das die größere neue Arbeit ist, dann Page 1 Updates für die neuen Setup-Typen.
