# Score-Redesign — Was macht einen Breakout-Kandidaten? (2026-07-22)

Datengetriebene Analyse: Welche Features trennen in UNSEREN Daten Winner von Losern,
und wo gehört im Score Gewicht hin. Quellen: 160 Live-Signale (apex_signals × equity),
169 BREAKOUT-Trades aus dem frischen 2J-Backtest (2024-07..2026-07). Wo beide Quellen
übereinstimmen = hart; wo nur Live vorliegt = Hypothese (Backtest berechnet das Feature nicht).

---

## 1. Das aktuelle Gewicht (BREAKOUT, SCORE_REALIGN) — wo es JETZT hingeht

| Feature | Formel | max Punkte | Trennschärfe in Daten |
|---|---|---:|---|
| **rr (Risk/Reward)** | `min(rr,5)×4` | **20** | **ANTI-prädiktiv** ⛔ |
| perf_60 | `min(perf_60,35)×0.5` | 17.5 | schwach |
| perf_20 | `min(perf_20,20)×0.8` | 16 | Sweet-Spot 5-10, sonst flach |
| Base | fix | 20 | — |
| vol_ratio | `min(vr,3)×4` | 12 | schwach, Climax stark aber selten |
| perf_120 | `min(perf_120,50)×0.2` | 10 | **falsch herum** (beaten-down gewinnt!) |
| Pocket Pivot | +10 | 10 | +5pp (schwach-mittel) |
| winrate-Bonus | ±10 | 10 | nicht messbar (#19) |
| macd_bull | +8 | 8 | flach/Rauschen |
| RSI 48-72 | +6 | 6 | flach (55-65 = 65-72) |
| **VCP** | +5 | **5** | **+38pp — der stärkste Prädiktor!** ⭐ |
| gap, RS, higher_tf, movement… | div. | div. | gemischt |

**Kernproblem in einem Satz:** Das mit Abstand größte Gewicht (`rr`, 20 Punkte) ist
anti-prädiktiv, während der mit Abstand stärkste Prädiktor (`VCP`, +38pp) nur 5 Punkte bekommt.

---

## 2. Feature-Prädiktivität — was TRENNT Winner von Losern (160 Live-Signale, Baseline-WR 50.6%)

### Die Sterne (echte Trennschärfe)
| Feature | n | WR | vs Baseline | Quelle |
|---|---:|---:|---:|---|
| **VCP-Pattern** (cat_vcp_strength>0) | 28 | **82.1%** | **+31.5pp** | Live only |
| **perf_120 < 0** (beaten-down) | 22 | 63.6% | +13.0pp | Live + Learn 5b |
| **perf_20 5-10%** (Sweet-Momentum) | 42 | 59.5% | +8.9pp | Live |
| vol_ratio > 2.5 (Climax) | 5 | 80.0% | +29pp | Live (n klein) |
| Pocket Pivot | 81 | 53.1% | +5.0pp | Live + Learn |
| Gap ≥ 2% | 22 | 54.5% | +4.5pp | Live + Learn |

### Das Gewinner-Profil (Kombination)
| Profil | n | WR |
|---|---:|---:|
| **VCP ODER beaten-down** | 44 | **72.7%** |
| KEIN VCP, KEIN beaten-down (= der Bulk) | 116 | **42.2%** |
| VCP innerhalb der schlechtesten perf_120-Deadzone (0-25) | 13 | 84.6% |
| Deadzone OHNE VCP (= der echte Müll) | 53 | 35.8% |

**VCP-Robustheit geprüft:** 23/28 Winner, verteilt über Mai/Jun/Jul (kein Regime-Cluster),
auch ohne die 5 größten Wins noch 78.3% WR. Loser bei -6% gedeckelt, Winner bis +13.6% =
saubere Asymmetrie. Mechanisch = Minervini-VCP: Volatilitäts-Kontraktion = Angebot erschöpft.

### Das tote Gewicht (rr) — in BEIDEN Quellen bestätigt
| rr-Bucket | Live WR | 2J-Backtest WR |
|---|---:|---:|
| < 1.5 | 75.0% (n=4) | 62.3% (n=53) |
| 1.5-2.0 | — | 51.6% (n=91) |
| 2.0-2.5 | 45.5% | 45.5% (n=22) |
| ≥ 2.5 | — | 66.7% (n=3) |

**Höheres rr → niedrigere WR, monoton.** Mechanik: hohes rr = fernes Target ODER enger Stop.
Beides senkt die Trefferquote. Wir belohnen das mit 20 Punkten — dem größten Gewicht.

### Rauschen (Punkte ohne Gegenwert)
- **RSI-Band** (55-65 = 50%, 65-72 = 52%) → +6 Punkte für ~nichts
- **macd_bull** (mit 49.3% vs ohne 75%, aber n=8 ohne = Rauschen) → +8 für ~nichts
- **base_range** jenseits des VCP-Flags → flach über alle Buckets

---

## 3. Alternative Score-Schemata (Vorschläge, wo Gewicht hingehört)

Grundprinzip: **Gewicht von anti-prädiktiv (rr) + Rauschen (RSI/MACD) hin zu den
robusten Prädiktoren (VCP, beaten-down+Catalyst, Sweet-Momentum, Volumen-Climax).**

### Schema A — "VCP-Weighted" (Umallokation, minimal-invasiv)
Behält die Struktur, verschiebt nur Gewicht:
```
rr:          min(rr,5)×4  →  min(rr,3)×1.5   (20 → ~5 Punkte; nicht ganz raus, Filter-Info)
VCP:         +5           →  cat_vcp_strength-gestuft, bis +18   ⭐ der Star
beaten-down: neu          →  perf_120<0 UND Catalyst: +8         (Learn 5b + Daten)
perf_20:     linear ×0.8  →  Sweet-Fenster 5-12%: +8, sonst gedämpft
RSI/MACD:    +6/+8        →  je +3 (Rauschen abschmelzen)
vol_climax:  +5           →  +10 (selten, aber 80% WR)
```
Netto: ~15 Punkte von rr/RSI/MACD → VCP/beaten-down/Momentum-Sweet/Climax.

### Schema B — "Profile-Tier" (2-stufig, an PICK_BAND angelehnt)
Statt linearer Addition ein Qualitäts-Tier über dem Basis-Score:
```
Tier-1 (+20): VCP-Pattern ODER (perf_120<0 UND Catalyst) ODER vol_climax
Tier-2 (+0):  der Rest (der 42%-WR-Bulk muss sich den Score anders verdienen)
```
Bildet den 72.7% vs 42.2% Split direkt ab. Konsistent mit der Band-Philosophie.
Vorteil: robust gegen Overfitting (nur binäre, mechanisch begründete Gates).

### Schema C — "Empirical-Fit" (LogReg) — **NICHT empfohlen als Primärweg**
Regression Features→Outcome, Koeffizienten als Gewichte. **Das haben wir schon versucht:
SCORE_V2 (BACKLOG #17) — FALSIFIZIERT im 2J-Backtest** (WR 45.5 vs 50.2), weil der
gelernte movement_bonus-Flip nur EIN Regime abbildete. Lehre: naiver Fit generalisiert nicht.
Nur als Gegen-Check zu A/B, nie als Live-Mechanik ohne 2J-Bestätigung.

---

## 4. Was einen Breakout-Kandidaten ausmacht (Synthese Daten + Theorie)

Daten und klassische Literatur (Minervini VCP, O'Neil CANSLIM, Weinstein Stage-2) decken sich:

1. **Enge Konsolidierung vor dem Move (VCP)** — Volatilität kontrahiert, Angebot versiegt.
   Unser #1-Prädiktor (+31.5pp). *Das ist die Signatur eines echten Breakouts.*
2. **Volumen-Expansion auf dem Ausbruch** — echte Nachfrage-Bestätigung (Climax 80% WR).
3. **Gesunde Basis, nicht überdehnt** — entweder beaten-down mit Raum (perf_120<0 + Catalyst)
   ODER moderates Momentum (perf_20 5-10%). Überdehnt (perf_120>100, perf_20>20) = Late-Entry.
4. **Relative Stärke vs Markt** — Literatur stark, in unseren Daten nicht testbar (Feld leer im
   Equity-Join). Bonus existiert (+8), Prädiktivität hier offen.
5. **NICHT: ein weites Target / enger Stop (rr)** — die Trefferquoten-Illusion.

---

## 5. Disziplin & nächste Schritte

- **Sofort umsetzbar & in BEIDEN Quellen bestätigt:** Score-Gate 70→80 (separate Analyse,
  +2pp WR, Profit steigt). Das ist der erste, sicherste Schritt.
- **rr-Gewicht senken** ist in beiden Quellen gedeckt (anti-prädiktiv) — aber es verändert
  das Ranking breit → **Backtest-First** mit vorab fixierten Kriterien.
- **VCP-Aufwertung** ist der größte Hebel, aber **Live-only (n=28)** — der Backtest berechnet
  cat_vcp_strength NICHT. Zwei Wege: (a) VCP-Feature IN den Backtest bauen (dann 2J-testbar),
  (b) live-forward validieren (VCP-Signale markieren, 30-40 sammeln). Kein Blind-Flug.
- **Kein naiver Gesamt-Refit** (SCORE_V2-Lehre). Ein Feature-Gewicht pro Backtest ändern,
  Attribution sauber halten.

**Reihenfolge-Empfehlung:** (1) Gate 80 [bestätigt], (2) VCP-Feature in den Backtest bauen
→ dann VCP-Aufwertung 2J-testen, (3) rr-Gewicht-Reduktion backtesten, (4) Schema B als
Alternative gegen A backtesten. Jeder Schritt einzeln, Backtest-First.
