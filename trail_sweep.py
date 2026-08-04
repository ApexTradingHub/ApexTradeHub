# -*- coding: utf-8 -*-
"""
Trailing-Ladder-Sweep auf Tagesbars (2026-07-31).

Warum: Trailing ist im apex_backtest_v2 NICHT simulierbar (der kennt nur TP/SL/Zeit).
Dieses Skript rekonstruiert den Pfad jedes geschlossenen Trades aus yfinance-Tagesbars
und simuliert verschiedene Trailing-Leitern gegeneinander.

Methodik (bewusst konservativ):
  - Horizont = tatsaechliche Haltedauer (opened_at .. closed_at). Exogene Exits
    (Stagnation/Time/EOD) bleiben damit erhalten; NUR der Trailing-Stop variiert.
  - Pro Tag zuerst Stop gegen LOW pruefen (adverse-first), DANN High/Ladder updaten.
    Verhindert, dass ein Intraday-High den Stop rettet, bevor das Low ihn reisst.
  - Bars strikt NACH dem Entry-Tag (kein Pre-Entry-Low-Artefakt).
  - Feuert nichts bis zum Ende -> tatsaechlicher Exit-Preis (Realitaet).
VALIDIERUNG: Variante "CURRENT" muss die echte Gesamt-PnL ungefaehr treffen.

⚠ GRENZE DIESES WERKZEUGS (Fund 2026-08-04): Tagesbars sind NUR fuer Givebacks >= ~2pp
brauchbar. Der Stop wird hier einmal pro Tag gegen das Tagestief geprueft, mit dem Stand vom
VORTAGESHOCH — ein enger Stop kann also gar nicht intraday feuern. Live ratcht der Trader alle
5 Minuten nach. Konkret gemessen fuer "1pp-Giveback ab +3%": Tagesbars +21.2pp, 5-Min-Bars
-10.3pp (Modell LOW). Fuer enge Leitern IMMER trail_sweep_intraday.py nehmen.
"""
import json, warnings, sys
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

BREAKOUT_LADDERS = {
    "CURRENT  (6->2, 10->6, 14->10)": [(1.06, 1.02), (1.10, 1.06), (1.14, 1.10)],
    "B  +3->1 Fruehstufe":            [(1.03, 1.01), (1.06, 1.02), (1.10, 1.06), (1.14, 1.10)],
    "C  +4->2 Fruehstufe":            [(1.04, 1.02), (1.06, 1.04), (1.10, 1.07), (1.14, 1.11)],
    "D  aggressiv (2.5->1)":          [(1.025, 1.01), (1.05, 1.03), (1.08, 1.05), (1.12, 1.09)],
    "E  nur BE-Schutz ab +3":         [(1.03, 1.00), (1.06, 1.02), (1.10, 1.06), (1.14, 1.10)],
}
MOMO_LADDERS = {
    "CURRENT  (6->2, 10->6, 14->10)": [(1.06, 1.035), (1.10, 1.075), (1.15, 1.115)],
    "B  +3->1 Fruehstufe":                  [(1.03, 1.01), (1.06, 1.035), (1.10, 1.075), (1.15, 1.115)],
    "C  +4->2 Fruehstufe":                  [(1.04, 1.02), (1.06, 1.04), (1.10, 1.075), (1.15, 1.115)],
    "D  aggressiv (2.5->1)":                [(1.025, 1.01), (1.05, 1.03), (1.08, 1.05), (1.12, 1.09)],
    "E  nur BE-Schutz ab +3":               [(1.03, 1.00), (1.06, 1.035), (1.10, 1.075), (1.15, 1.115)],
}
MOMO_GIVEBACK = 0.06


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def load_trades():
    p = json.load(open("apex_positions.json", encoding="utf-8"))
    out = []
    for c in p.get("closed", []):
        if not isinstance(c, dict):
            continue
        e = f(c.get("entry_actual")) or f(c.get("entry_price"))
        pnl = f(c.get("pnl_pct"))
        op, cd = c.get("opened_at"), c.get("closed_at")
        if not e or pnl is None or not op or not cd:
            continue
        try:
            od = datetime.fromisoformat(op.replace("Z", "")).date()
            xd = datetime.fromisoformat(cd.replace("Z", "")).date()
        except Exception:
            continue
        out.append({
            "ticker": c["ticker"], "setup": c.get("setup"), "source": c.get("source"),
            "rescued": bool(c.get("intraday_rescued")),
            "entry": e, "stop0": f(c.get("stop_initial")) or e * 0.95,
            "target": f(c.get("target")) or e * 1.10,
            "open_d": od, "exit_d": xd,
            "actual_pnl": pnl, "actual_exit": f(c.get("exit_price")),
            "actual_reason": c.get("exit_reason", ""),
        })
    return out


def simulate(tr, bars, ladder, giveback=None, hard_tp=True):
    """Gibt (pnl_pct, exit_kind) zurueck.
    hard_tp=False fuer MOMENTUM: der Trader hat dort KEINEN harten TP (apex_trader L1912-1914,
    'Momentum: KEIN harter TP — ausbrechen lassen'), die Trail-Ladder sichert progressiv.
    Ohne dieses Flag wuerden alle Runner am Target gekappt -> die Kosten von engem Trailing
    wuerden systematisch UNTERschaetzt (PAY: real +19.5%, gekappt +6.0%)."""
    entry, stop, target = tr["entry"], tr["stop0"], tr["target"]
    high_se = entry
    step = 0
    if bars is None or bars.empty:
        return tr["actual_pnl"], "actual(no-bars)"
    for _, row in bars.iterrows():
        lo, hi = float(row["Low"]), float(row["High"])
        # 1) adverse-first: Stop gegen Low
        if lo <= stop:
            return (stop / entry - 1) * 100, ("trail-stop" if step > 0 else "initial-stop")
        # 2) TP (nur Nicht-Momentum)
        if hard_tp and hi >= target:
            return (target / entry - 1) * 100, "target"
        # 3) High + Ladder updaten
        high_se = max(high_se, hi)
        for i, (trig, slm) in enumerate(ladder, start=1):
            if high_se >= entry * trig and i > step:
                stop = max(stop, round(entry * slm, 2))
                step = i
        if giveback and step >= len(ladder) and high_se >= entry * ladder[-1][0]:
            stop = max(stop, round(high_se * (1 - giveback), 2))
    return tr["actual_pnl"], "actual(held)"


def main():
    trades = load_trades()
    print(f"Closed-Trades mit Datum+Entry: {len(trades)}")
    tickers = sorted({t["ticker"] for t in trades})
    start = min(t["open_d"] for t in trades) - timedelta(days=3)
    print(f"Lade Bars fuer {len(tickers)} Ticker ab {start} ...")
    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                       auto_adjust=False, progress=False, group_by="ticker")

    def bars_for(tr):
        tk = tr["ticker"]
        try:
            df = data[tk] if isinstance(data.columns, pd.MultiIndex) else data
            df = df[["Open", "High", "Low", "Close"]].dropna()
            m = (df.index.date > tr["open_d"]) & (df.index.date <= tr["exit_d"])
            return df[m]
        except Exception:
            return None

    cache = {id(t): bars_for(t) for t in trades}
    names = list(BREAKOUT_LADDERS.keys())
    print(f"\n{'Variante':38} {'Gesamt-PnL':>11} {'WR':>7} {'AvgWin':>8} {'AvgLoss':>8} {'TrailStops':>11}")
    print("-" * 88)
    results = {}
    for name in names:
        bo_l, mo_l = BREAKOUT_LADDERS[name], MOMO_LADDERS[name]
        tot, wins, losses, wsum, lsum, tstops = 0.0, 0, 0, 0.0, 0.0, 0
        per = []
        for t in trades:
            is_momo = (t["setup"] == "MOMENTUM") or t["rescued"] or t["source"] == "intraday_momentum"
            lad = mo_l if is_momo else bo_l
            gb = MOMO_GIVEBACK if is_momo else None
            pnl, kind = simulate(t, cache[id(t)], lad, gb, hard_tp=not is_momo)
            per.append((t, pnl, kind))
            tot += pnl
            if kind == "trail-stop":
                tstops += 1
            if pnl > 0:
                wins += 1; wsum += pnl
            elif pnl < 0:
                losses += 1; lsum += pnl
        n = wins + losses
        wr = 100 * wins / n if n else 0
        aw = wsum / wins if wins else 0
        al = lsum / losses if losses else 0
        results[name] = (tot, per)
        print(f"{name:38} {tot:+10.1f}pp {wr:6.1f}% {aw:+7.2f}% {al:+7.2f}% {tstops:10}")

    # Detail: was passiert mit den grossen Gewinnern?
    print("\n=== Wirkung auf die 9 grossen Gewinner (actual >= +6%) ===")
    base = {t["ticker"]: p for t, p, k in results[names[0]][1]}
    print(f"{'Ticker':8} {'ACTUAL':>8} " + " ".join(f"{n.split()[0]:>9}" for n in names))
    for t, p, k in sorted(results[names[0]][1], key=lambda x: -x[0]["actual_pnl"]):
        if t["actual_pnl"] < 6:
            continue
        row = f"{t['ticker']:8} {t['actual_pnl']:+7.2f}% "
        for nm in names:
            pv = dict((tt["ticker"], pp) for tt, pp, kk in results[nm][1])[t["ticker"]]
            row += f" {pv:+8.2f}%"
        print(row)

    print("\n=== Wirkung auf die 12 Give-Back-Trades (MFE>=3%, Ende<=+1%) ===")
    gb_tk = {"WULF", "CMI", "SW", "PLTR", "PENG", "AYI", "HRB", "HSAI", "KEEL", "FROG", "RIVN", "MRCY"}
    for t, p, k in results[names[0]][1]:
        if t["ticker"] not in gb_tk:
            continue
        row = f"{t['ticker']:8} {t['actual_pnl']:+7.2f}% "
        for nm in names:
            pv = dict((tt["ticker"], pp) for tt, pp, kk in results[nm][1])[t["ticker"]]
            row += f" {pv:+8.2f}%"
        print(row)


if __name__ == "__main__":
    main()
