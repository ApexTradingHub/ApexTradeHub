# ApexScan — Score-Rebuild Strategiepapier

**Erstellt:** 2026-06-19 (Ultracode-Session)
**Status:** ENTWURF zur Diskussion — noch nichts implementiert, Backtest-First Pflicht
**Ziel-Owner:** Niklas + Claude

---

## 0. Auftrag (in einem Satz)

Der BREAKOUT-Score soll wieder **monoton** werden (höherer Score = höhere reale WR) und
Aktien **beim Ausbruch aus der Base** finden — nicht erst nachdem sie schon +80 % gelaufen sind.
ASML/ETN/FLR sollen einen Top-Score bekommen, wenn sie an der Resistance **kippen**, nicht am Hoch.

---

## 1. Diagnose: was kaputt ist

### 1.1 Der harte Beweis — Score-Kalibrierung plateaut
Lifetime-WR pro BREAKOUT-Score-Bucket (`knowledge/apex_knowledge.json`):

| Bucket | n | WR | |
|---|---|---|---|
| 60-70 | 6 | 33 % | monoton ↑ |
| 70-80 | 13 | 39 % | monoton ↑ |
| 80-90 | 12 | 50 % | monoton ↑ |
| **90-100** | 19 | **68 %** | **Peak** |
| **100+** | 39 | **64 %** | **Plateau/↓** |

→ Der Score **funktioniert sauber bis 100**. Darüber trägt jeder Extrapunkt **keine** zusätzliche
Trefferwahrscheinlichkeit, sogar leicht negativ. Ein 146 ist nicht besser als ein 98.

### 1.2 Die Ursache — die `perf`-Terme belohnen Extension linear
Aktuelle BREAKOUT-Formel (Auszug, `ApexScan.py` ~L1281-1335):

```
score += min(max(perf_20, 0), 20)  * 0.8     # max +16
score += min(max(perf_60, 0), 35)  * 0.5     # max +17.5
score += min(max(perf_120, 0), 50) * 0.2     # max +10   -> Summe perf max +43.5
score += min(vol_ratio, 3.0) * 4             # max +12
score += min(rr, 5.0) * 4                     # max +20
+ movement_bonus (perf_120-realign: <0=-15, 0-25=-3, 25-50=+15, >50=+8)
+ closing_bonus (+5 stark / 0 / -10 <0.5)
+ pocket_pivot +10, gap>2% +8, vcp +5, vol_climax +5
+ RS-vs-SPY (+8/+5/+2/-2/-5), Sektor-Mom (+6/+3/-2/-5)
+ historical_wr_bonus (wr-50)*0.2
```

**Die ~43.5 Punkte aus `perf_20/60/120` sind der größte Einzelblock — und sie belohnen genau
das was wir NICHT wollen: eine Aktie die schon weit gelaufen ist.** Die perf_120-Realign
(06-14) korrigiert nur die *120-Tage*-Achse grob (POWER >50 nur +8 statt +15). Aber `perf_20`
und `perf_60` sind weiter **linear-bis-Cap** — eine Aktie +26 % in 20d und +39 % in 60d maxt
beide Terme aus, egal ob sie jetzt überdehnt ist.

### 1.3 Live-Beleg: ASML 146 vs FLR 149 (gleicher Score, ungleiche Qualität)

| Merkmal | FLR (149) | ASML (146) | Was es bedeutet |
|---|---|---|---|
| vol_ratio | **2.71** | 1.22 | FLR = echtes Volumen, ASML = kein Schub |
| closing_strength | **0.88** | 0.57 | FLR = starker Schluss, ASML = Fade-Risiko |
| perf_120 | +31 % (SWEET) | **+80 %** | ASML extrem überdehnt |
| perf_20 / perf_60 | +23.6 / +13 | +26 / **+38.8** | ASML maxt perf-Terme aus |
| analyst_upside | -5.5 % | **-9.9 %** | ASML 10 % über Zielen |

**Beide ~147 — aber ASML ist auf Volumen + Schluss schwächer UND viel extender.** Der Score
verwechselt „weit gelaufen" mit „gut". Das ist der Defekt in einem Bild.

### 1.4 Warum das zum Go-Live-Risiko wird
Telegram-Top-2 + Paper-Trader picken nach Score. Wenn der Score überdehnte Late-Entries an die
Spitze setzt, kaufen wir systematisch **nach** dem Move (ASML/FLR/SNDK-Muster) statt beim
Ausbruch. Bei echtem Geld = Entry am Hoch, Stop kurz drauf.

---

## 2. Designprinzipien (woran wir den neuen Score messen)

1. **Monotonie:** höherer Score ⇒ höhere reale WR über ALLE Buckets, auch 100+.
2. **Früh statt spät:** Score peakt beim **Base→Breakout-Übergang** (enge Base kippt mit Volumen +
   starkem Schluss), nicht bei Post-Run-Extension.
3. **Confirmation > Rohmomentum:** Volumen-Schub, closing_strength, Base-Tightness sollen mehr
   wiegen als „+X % gelaufen".
4. **Signal-Protection (HARTE Leitplanke):** Signal-Count ≥ 95 % Baseline. Re-Ranking, NICHT
   Wegschneiden. Kein Change der einen bestehenden Winner (+TP-Trade) rausfiltert.

---

## 3. Vorgeschlagene Hebel (3 + 1, jeder einzeln backtestbar)

### Hebel A — Degressive `perf`-Kurve (statt linear-bis-Cap)
**Problem:** perf_20/60 belohnen die ersten % genauso wie die letzten % vor der Erschöpfung.
**Idee:** Kurve die die **ersten 10-15 %** Momentum voll belohnt und darüber **abflacht** (und ggf.
ins Negative dreht). Catcht den frühen Ausbruch, hört auf Extension zu füttern.

Skizze (zu kalibrieren):
```
def perf_score(p, sweet_hi, peak_pts):
    if p <= 0: return 0
    if p <= sweet_hi:            # z.B. sweet_hi=12 %
        return (p / sweet_hi) * peak_pts        # linear hoch bis Peak
    # darüber: abflachen, dann leicht abbauen (Extension-Dämpfung)
    over = p - sweet_hi
    return peak_pts - min(over * decay, peak_pts * 0.6)
```
→ perf_20 peakt bei ~12 %, perf_60 bei ~20 %. Eine Aktie +26 %/20d bekommt dann **weniger** als
heute, nicht das Maximum. **Kalibrierung gegen die Bucket-WRs, nicht erfunden.**

### Hebel B — Extension-Penalty (komposit, MIT Catalyst-Carve-Out)
**Wichtig:** nicht alle extended Namen strafen — nur das **ASML-Fade-Muster** (extended UND schwache
Bestätigung UND kein starker Catalyst). FLR (vol 2.71, closing 0.88) bliebe verschont.

**KRITISCH (Knowledge-Check 2026-06-19):** Die `what_to_replicate`-Felder sind explizit und 8×
wiederholt: „BREAKOUT mit starkem sekularem Catalyst (Semi/AI-Capex, AI-Storage/Networking)
gewinnt TROTZ Extension — nicht blind perf120 bestrafen wenn Catalyst stark" (STX, APP, +6×
Semi-Cluster inkl. ASML-Mai-Winner). Ein Extension-Penalty OHNE Carve-Out wäre im Krieg mit
unserer #1-Replicate-Lehre. → **Carve-Out ist Pflicht, nicht optional.**

```
strong_catalyst = (cat_earnings_beat
                   or (cat_analyst_upside is not None and cat_analyst_upside > 15)
                   or (cat_pocket_pivot and cat_vol_climax)   # PP + Vol-Climax = inst. Conviction
                   or cat_gap_pct >= 5)                        # echter Catalyst-Gap

if perf_120 > 60 and vol_ratio < 1.5 and closing_strength < 0.6 and not strong_catalyst:
    score -= EXT_PENALTY            # z.B. -12 bis -20, im Backtest kalibriert
# optional Stufen: je mehr der 3 Kriterien zutreffen, desto höher
```
→ **ASML-Juni** (perf_120 80, vol 1.22, closing 0.57, KEIN earnings_beat, analyst -9.9, nur PP
ohne Vol-Climax) = voller Penalty (= der Late-Weak-Entry, korrekt).
→ **STX/ASML-Mai-Winner** (perf_120 extended ABER earnings_beat / Capex-Catalyst) = Carve-Out
greift, KEIN Penalty (= bleibt Top-Score, korrekt — entspricht Replicate-Lehre).
→ **FLR** (vol 2.71, closing 0.88) = trifft die 3-Kriterien-Gate eh nicht.

**Empirie-Check:** mit der 3-Kriterien-Gate (ohne Carve-Out) würden 0 historische BREAKOUT-Winner
gestraft. Der Carve-Out schützt zusätzlich gegen ZUKÜNFTIGE Semi/AI-Capex-Winner (die Replicate-
Kategorie). Doppelt abgesichert.

### Hebel C — Hard-Score-Cap → **VERWORFEN** (mit Begründung)
Ein Cap bei z.B. 110 würde viele Signale auf 110 stauchen → Rangfolge unter den besten
kollabiert, Telegram-Top-2 wird Zufall (User-Einwand, korrekt). **Der Cap behandelt das Symptom
(Plateau), nicht die Ursache (Extension-Reward).** Hebel A+B reparieren die Ursache und stellen
Monotonie ohne Cap wieder her. → Cap NICHT bauen.

### Hebel D (optional) — Early-Base-Breakout-Boost
**Idee:** den sauberen frühen Ausbruch zusätzlich belohnen, damit frische Breakouts überdehnte
überholen:
```
if base_range <= 12 and breakout_from_base and vol_ratio >= 1.5 and closing_strength >= 0.7:
    score += EARLY_BONUS    # z.B. +8
```
→ macht den Score „früh-affin". Nur wenn A+B allein die Monotonie nicht voll herstellen.

---

## 4. Validierungsplan (Akzeptanz VORHER fixiert — Anti-Cherry-Pick)

**Tool:** `apex_backtest_v2.py`, neuer Flag `--score-rebuild` (analog `--score-realign`).
**Fenster:** 2 Jahre (für n) + Cross-Check letzte 6 Monate.
**Disziplin:** Ein-Hebel-pro-Lauf (A, dann B, dann ggf. D — nie bündeln).

**GO nur wenn ALLE erfüllt:**
| Kriterium | Schwelle |
|---|---|
| **Monotonie wiederhergestellt** | WR(100+) ≥ WR(90-100) — der Kern-Test |
| **Signal-Count** | ≥ 95 % Baseline (Signal-Protection) |
| **WR gesamt** | ≥ Baseline |
| **PF gesamt** | ≥ Baseline |
| **Winner-Erhalt** | 0 bestehende TP-Winner durch Change rausgefiltert (explizit gelistet + geprüft) |
| **Replicate-Erhalt** | jeder Change gegen `what_to_replicate` geprüft — KEIN Penalty der eine Replicate-Kategorie trifft (v.a. Semi/AI-Capex-Extension-Winner). Carve-Out validiert. |
| **Per-Bucket-Drift** | kein Bucket fällt >5pp WR ggü. Baseline |

**Diagnose-Output je Lauf:** Score-Kalibrierungs-Tabelle vorher/nachher (die 5 Buckets), plus
Liste der Top-10-Score-Signale vorher/nachher mit ihren tatsächlichen Outcomes.

---

## 5. Risiko + Rollback

- **Flag-gated:** `--score-rebuild` im Backtest, Live erst nach GO als hardcoded (wie SCORE_REALIGN).
- **Rollback:** Flag raus / movement-Block auf Realign-Stand zurück. Score-Logik ist isoliert in
  einem Block (`ApexScan.py` ~L1281-1345), kein Seiteneffekt auf Entry/Exit.
- **Knowledge-Recompute:** nach Live-Switch `apex_learn.py` + ggf. Equity-Re-Eval (Scores ändern
  Telegram-Ranking, nicht die simulierten Trades selbst — die hängen an Entry/Stop/Target).
- **Größtes Restrisiko:** Überfitting auf das ASML/FLR-Beispiel. Gegenmittel: Akzeptanz auf
  Aggregat (Bucket-WRs + Signal-Count), nicht auf Einzelnamen.

---

## 6. Offene Entscheidungen für Niklas

1. **Hebel-Reihenfolge:** A zuerst (degressive perf) oder B zuerst (Extension-Penalty)?
   Empfehlung: **B zuerst** — chirurgischer, kleineres Signal-Count-Risiko, schneller Beweis ob
   das Fade-Profil real schlechter ist.
2. **Extension-Penalty-Schwellen:** perf_120 > 60 % / vol < 1.5 / closing < 0.6 — als
   Startwerte ok, oder anders ankern?
3. **Hebel D bauen** oder erst schauen ob A+B die Monotonie allein herstellen?
4. **Sweet-Spot perf-Peak:** 12 % für perf_20, 20 % für perf_60 — Startannahme, im Backtest kalibriert.

---

## 7. Nächster Schritt (wenn freigegeben)

1. `apex_backtest_v2.py`: Baseline-Lauf, Score-Kalibrierungs-Tabelle als Referenz festhalten.
2. Hebel B implementieren hinter `--score-rebuild`, Backtest, Akzeptanz prüfen.
3. Falls GO: Hebel A, dann ggf. D — je einzeln.
4. Aggregat-GO → Live-Hardcode + Knowledge-Refresh.

*Backtest-First. Akzeptanz vor dem Lauf. Re-Ranking statt Wegschneiden. Das ist die Disziplin.*
