# -*- coding: utf-8 -*-
"""Sektor-RS-Gate Backtest (BACKLOG #20, 2026-07-15).

Hypothese: BREAKOUT skippen wenn der Sektor-ETF schwach ist / negativ vs SPY divergiert
faengt den Semi-Selloff-Verlust-Cluster (LRCX/KLAC/CGNX) ohne die Catalyst-Winner (SE) zu killen.

Rekonstruiert Sektor-ETF-perf_20 + SPY-perf_20 pro Signal-Datum (exakt Scanner-Logik).
Sweept Gate-Varianten x Scope (all/Tech-Comm) x Catalyst-Carve-Out.

Akzeptanz (VORAB fixiert, Anti-Cherry-Pick):
  - WR der GEHALTENEN Trades >= Baseline +3pp
  - Signal-Count >= 95% Baseline (Signal-Protection)
  - Netto-PnL-Summe der gehaltenen >= Baseline (kein Profit-Verlust)
  - Gedroppte Winner-Rate < gedroppte Loser-Rate (wir droppen ueberwiegend Loser)
GO nur wenn ALLE vier. Sonst verwerfen + BACKLOG.
"""
import json
import math

SECTOR_ETFS = {
    "Energy": "XLE", "Technology": "XLK", "Healthcare": "XLV",
    "Financial Services": "XLF", "Industrials": "XLI",
    "Consumer Defensive": "XLP", "Consumer Cyclical": "XLY",
    "Basic Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}
TECH_COMM = {"Technology", "Communication Services"}


def wr_pf(pnls):
    n = len(pnls)
    if not n:
        return 0.0, 0.0, 0.0
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n * 100
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 99.0
    return wr, pf, sum(pnls)


def strong_catalyst(s):
    # analog SCORE_REBUILD-Carve-Out: PP+VolClimax ODER Earnings-Beat ODER Gap>=5
    return bool(
        s.get("cat_earnings_beat") or
        (s.get("cat_pocket_pivot") and s.get("cat_vol_climax")) or
        (float(s.get("cat_gap_pct") or 0) >= 5.0)
    )


def main():
    sigs = json.load(open("apex_signals.json", encoding="utf-8"))
    eqt = json.load(open("apex_equity_results.json", encoding="utf-8"))
    eq = {(t["ticker"], str(t["date"])[:10]): t for t in eqt}

    rows = []
    for s in sigs:
        if s.get("setup") != "BREAKOUT":
            continue
        key = (s["ticker"], str(s["date"])[:10])
        t = eq.get(key)
        if not t or t.get("pnl_pct") is None:
            continue
        rows.append({"date": key[1], "ticker": key[0], "sector": s.get("sector", ""),
                     "pnl": float(t["pnl_pct"]), "score": float(s.get("score") or 0),
                     "catalyst": strong_catalyst(s), "sig": s})
    print(f"Join: {len(rows)} BREAKOUT-Paare")

    # Sektor-ETF + SPY perf_20 pro Datum rekonstruieren
    import yfinance as yf
    import pandas as pd
    symbols = ["SPY"] + list(SECTOR_ETFS.values())
    print("Lade SPY + 11 Sektor-ETFs...")
    df = yf.download(symbols, start="2026-01-01", end="2026-07-16",
                     progress=False, auto_adjust=True, group_by="ticker")
    closes = {sym: df[sym]["Close"].dropna() for sym in symbols}

    def perf20(sym, d):
        h = closes[sym].loc[:pd.Timestamp(d)]
        if len(h) < 21:
            return 0.0
        return float((h.iloc[-1] / h.iloc[-21] - 1) * 100)

    for r in rows:
        etf = SECTOR_ETFS.get(r["sector"])
        r["sec_perf20"] = perf20(etf, r["date"]) if etf else 0.0
        r["spy_perf20"] = perf20("SPY", r["date"])
        r["diverg"] = r["sec_perf20"] - r["spy_perf20"]

    # Baseline
    base_wr, base_pf, base_sum = wr_pf([r["pnl"] for r in rows])
    n_base = len(rows)
    print(f"\nBASELINE: n={n_base} WR={base_wr:.1f}% PF={base_pf:.2f} sum={base_sum:+.1f}%\n")

    # Gate-Varianten: (label, predicate(r) -> True=SKIP)
    def mk(scope_tech, sec_thr=None, div_thr=None):
        def pred(r):
            if scope_tech and r["sector"] not in TECH_COMM:
                return False
            cond = True
            if sec_thr is not None:
                cond = cond and (r["sec_perf20"] < sec_thr)
            if div_thr is not None:
                cond = cond and (r["diverg"] < div_thr)
            return cond
        return pred

    variants = []
    for scope, sname in [(False, "ALL"), (True, "TECH")]:
        variants += [
            (f"{sname} sec<0",            mk(scope, sec_thr=0)),
            (f"{sname} sec<-2",           mk(scope, sec_thr=-2)),
            (f"{sname} div<-2",           mk(scope, div_thr=-2)),
            (f"{sname} div<-5",           mk(scope, div_thr=-5)),
            (f"{sname} sec<0 & div<-2",   mk(scope, sec_thr=0, div_thr=-2)),
        ]

    print(f"{'Variante':26} {'Carve':5} {'kept':>5} {'ret%':>5} {'WR':>6} {'dWR':>6} {'PF':>5} {'sum':>7} {'drop':>4} {'dW/dL':>7} {'GO?':>4}")
    results = []
    for label, pred in variants:
        for carve in (False, True):
            kept, dropped = [], []
            for r in rows:
                skip = pred(r) and not (carve and r["catalyst"])
                (dropped if skip else kept).append(r)
            kp = [r["pnl"] for r in kept]
            wr, pf, ssum = wr_pf(kp)
            ret = len(kept) / n_base * 100
            dW = sum(1 for r in dropped if r["pnl"] > 0)
            dL = sum(1 for r in dropped if r["pnl"] <= 0)
            # Akzeptanz
            go = (wr >= base_wr + 3 and ret >= 95 and ssum >= base_sum and dL >= dW and len(dropped) > 0)
            dwl = f"{dW}/{dL}"
            print(f"{label:26} {'ja' if carve else 'nein':5} {len(kept):>5} {ret:>5.0f} {wr:>5.1f}% {wr-base_wr:>+5.1f} {pf:>5.2f} {ssum:>+6.0f}% {len(dropped):>4} {dwl:>7} {'GO' if go else '-':>4}")
            results.append((label, carve, go, wr, ret, ssum, dropped))

    # Detail fuer die beste GO-Variante (oder beste nach WR-Lift wenn keine GO)
    gos = [r for r in results if r[2]]
    print("\n" + "="*70)
    if gos:
        best = max(gos, key=lambda x: (x[3], x[5]))
        print(f"BESTE GO-Variante: {best[0]} (carve={best[1]}) WR={best[3]:.1f}% ret={best[4]:.0f}% sum={best[5]:+.0f}%")
    else:
        print("KEINE Variante besteht alle 4 Akzeptanzkriterien -> NO-GO.")
        best = max(results, key=lambda x: x[3])
        print(f"Bester WR-Lift (nicht GO): {best[0]} carve={best[1]} WR={best[3]:.1f}% (Baseline {base_wr:.1f})")
    # Welche Trades gedroppt?
    print(f"\nGedroppte Trades der Referenz-Variante '{best[0]}' carve={best[1]}:")
    for r in sorted(best[6], key=lambda x: x["pnl"]):
        wl = "W" if r["pnl"] > 0 else "L"
        cat = " [CATALYST]" if r["catalyst"] else ""
        print(f"  {wl} {r['ticker']:6} {r['date']} {r['sector'][:18]:18} pnl={r['pnl']:+6.1f}% sec20={r['sec_perf20']:+5.1f} spy20={r['spy_perf20']:+4.1f} div={r['diverg']:+5.1f}{cat}")


if __name__ == "__main__":
    main()
