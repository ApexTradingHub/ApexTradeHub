"""SCORE_V2 Stufe 2 — Vergleich Baseline vs V2 (Brief AP2 Stufe 2, 2026-07-11).

Nimmt zwei Backtest-Result-Files (gleiches Fenster, gleiche Config bis auf --score-v2)
und prueft die VORAB fixierten Akzeptanzkriterien:
  - Signal-Count ~identisch (Re-Ranking, kein Gate) — Toleranz ±5% (Pick-Pfad-Divergenz
    durch unterschiedliche open_pos ist inhaerent)
  - Gepickte-Trades-WR >= Baseline +2pp
  - PF >= Baseline
Zusaetzlich: Sektor-Verteilung der Picks (erwarteter Nebeneffekt: weniger Tech-Cluster).

Usage: py apex_score_v2_stage2_compare.py baseline.json v2.json
"""
import json
import sys
from collections import Counter


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d if isinstance(d, list) else d.get("trades", d)


def stats(trades):
    pnls = [t["pnl_pct"] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n * 100 if n else 0
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
    return n, wr, pf, sum(pnls)


def sector_dist(trades):
    try:
        cache = json.load(open("sector_cache.json", encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return Counter(cache.get(t["ticker"], "Unknown") for t in trades)


def main():
    base_f, v2_f = sys.argv[1], sys.argv[2]
    base, v2 = load(base_f), load(v2_f)

    nb, wrb, pfb, sumb = stats(base)
    nv, wrv, pfv, sumv = stats(v2)

    print(f"{'':14} {'n':>5} {'WR':>7} {'PF':>6} {'Sum-PnL':>9}")
    print(f"{'Baseline':14} {nb:>5} {wrb:>6.1f}% {pfb:>6.2f} {sumb:>+8.1f}%")
    print(f"{'SCORE_V2':14} {nv:>5} {wrv:>6.1f}% {pfv:>6.2f} {sumv:>+8.1f}%")

    print("\n=== Sektor-Verteilung der Picks (Top 8) ===")
    sb, sv = sector_dist(base), sector_dist(v2)
    all_secs = sorted(set(sb) | set(sv), key=lambda s: -(sb.get(s, 0) + sv.get(s, 0)))
    print(f"{'Sektor':24} {'Baseline':>9} {'V2':>6}")
    for s in all_secs[:8]:
        print(f"{s:24} {sb.get(s,0):>9} {sv.get(s,0):>6}")

    # --- Akzeptanz (vorab fixiert) ---
    count_ok = abs(nv - nb) / max(nb, 1) <= 0.05
    wr_ok = wrv >= wrb + 2
    pf_ok = pfv >= pfb
    print(f"\n=== AKZEPTANZ (vorab fixiert, Brief §AP2 Stufe 2) ===")
    print(f"  Trade-Count ±5%:      {nv} vs {nb}  -> {'OK' if count_ok else 'FAIL'}")
    print(f"  WR >= Baseline +2pp:  {wrv:.1f}% vs {wrb:.1f}%+2  -> {'OK' if wr_ok else 'FAIL'}")
    print(f"  PF >= Baseline:       {pfv:.2f} vs {pfb:.2f}  -> {'OK' if pf_ok else 'FAIL'}")
    go = count_ok and wr_ok and pf_ok
    print(f"\n==> STUFE 2 {'GO -> Stufe 3 (Live-Port, Flag-gated)' if go else 'NO-GO -> verwerfen + BACKLOG, KEIN Live-Change'}")


if __name__ == "__main__":
    main()
