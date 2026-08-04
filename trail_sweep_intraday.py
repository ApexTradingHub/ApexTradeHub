# -*- coding: utf-8 -*-
"""Trailing-Sweep auf 5-MINUTEN-Bars (2026-08-04).

Warum: trail_sweep.py rechnet auf Tagesbars und prueft den Stop einmal pro Tag gegen das
Tagestief — mit dem Stop vom VORTAGESHOCH. Der echte Trader zieht den Stop alle 5 Minuten
intraday nach. Je ENGER der Giveback, desto staerker unterschaetzt die Tagesbar-Simulation
die Auslösehaeufigkeit: eine Aktie, die intraday auf +5% laeuft und auf +3.9% zurueckfaellt,
wird live bei +4% ausgestoppt — auf Tagesbars passiert nichts.

Dieses Skript rekonstruiert den 5-Min-Pfad und ratcht den Stop bar-fuer-bar, also genau wie live.

Zwei Stop-Modelle (Spanne statt Scheingenauigkeit):
  LOW   = Stop feuert wenn das Bar-TIEF ihn reisst  -> realistisch, weil eToros Server-SL
          seit dem Endpoint-Fix (29.07.) auf JEDEN Tick reagiert.
  CLOSE = Stop feuert nur wenn der Bar-SCHLUSS drunter liegt -> Paper-Trader-Verhalten
          (prueft `cur` alle 5 Min, verpasst Sub-5-Min-Dips).
Die Wahrheit liegt bei LOW, solange die eToro-SLs gepflegt sind.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
import yfinance as yf
import pandas as pd
import trail_sweep as ts

MOMO_GIVEBACK = ts.MOMO_GIVEBACK


def ladder(start_pct, give, top=30, step=1):
    out, p = [], start_pct
    while p <= top:
        out.append((round(1 + p / 100, 4), round(1 + (p - give) / 100, 4)))
        p += step
    return out


LIVE_BO = [(1.04, 1.02), (1.06, 1.04), (1.08, 1.06), (1.10, 1.07), (1.14, 1.11)]
LIVE_MO = [(1.04, 1.02), (1.06, 1.04), (1.08, 1.06), (1.10, 1.075), (1.15, 1.115)]

VARIANTS = {
    "A) LIVE (4/6/8/10/14)":        (LIVE_BO, LIVE_MO),
    "B) 1pp ab +3 (Idee)":          (ladder(3, 1), ladder(3, 1)),
    "C) 1.5pp ab +3":               (ladder(3, 1.5), ladder(3, 1.5)),
    "D) 2pp ab +3":                 (ladder(3, 2), ladder(3, 2)),
    "E) 2pp ab +4":                 (ladder(4, 2), ladder(4, 2)),
}


def simulate_5m(tr, bars, lad, giveback, hard_tp, mode="LOW"):
    """Bar-fuer-Bar wie der Live-Trader: erst Stop pruefen, dann High/Leiter nachziehen."""
    entry, stop, target = tr["entry"], tr["stop0"], tr["target"]
    high_se, step = entry, 0
    if bars is None or len(bars) == 0:
        return tr["actual_pnl"], "actual(no-bars)"
    for _, row in bars.iterrows():
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
        trigger = lo if mode == "LOW" else cl
        if trigger <= stop:
            return (stop / entry - 1) * 100, ("trail" if stop > tr["stop0"] else "init")
        if hard_tp and hi >= target:
            return (target / entry - 1) * 100, "tp"
        if hi > high_se:
            high_se = hi
            for i, (t_, s_) in enumerate(lad, start=1):
                if high_se >= entry * t_ and i > step:
                    stop = max(stop, round(entry * s_, 2))
                    step = i
            if giveback and step >= len(lad) and high_se >= entry * lad[-1][0]:
                stop = max(stop, round(high_se * (1 - giveback), 2))
    return tr["actual_pnl"], "actual(held)"


def main():
    trades = [t for t in ts.load_trades()]
    tickers = sorted({t["ticker"] for t in trades})
    print(f"{len(trades)} Trades / {len(tickers)} Ticker — lade 5m-Bars (60d-Limit) ...")
    store = {}
    for i, tk in enumerate(tickers, 1):
        try:
            df = yf.download(tk, period="60d", interval="5m", auto_adjust=False,
                             progress=False, threads=False)
            if df is not None and len(df):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                store[tk] = df[["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            pass
        if i % 25 == 0:
            print(f"   {i}/{len(tickers)} ...")
    print(f"5m-Daten fuer {len(store)}/{len(tickers)} Ticker\n")

    def bars_for(tr):
        df = store.get(tr["ticker"])
        if df is None:
            return None
        try:
            idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
            m = (idx.date > tr["open_d"]) & (idx.date <= tr["exit_d"])
            return df[m]
        except Exception:
            return None

    cache = {id(t): bars_for(t) for t in trades}
    covered = sum(1 for t in trades if cache[id(t)] is not None and len(cache[id(t)]))
    print(f"Trades mit 5m-Abdeckung: {covered}/{len(trades)} "
          f"(Rest faellt auf das tatsaechliche Ergebnis zurueck)\n")

    for mode in ("LOW", "CLOSE"):
        print(f"=== Stop-Modell {mode} "
              f"({'eToro-Server-SL, jeder Tick' if mode=='LOW' else 'Paper-Trader, 5-Min-Check'}) ===")
        print(f"{'Variante':26}{'Gesamt':>10}{'WR':>7}{'Ø Win':>9}{'Stops':>7}")
        print("-" * 60)
        for nm, (bo, mo) in VARIANTS.items():
            tot = w = n = tst = 0
            ws = []
            for t in trades:
                is_m = (t["setup"] == "MOMENTUM") or t["rescued"] or t["source"] == "intraday_momentum"
                pnl, kind = simulate_5m(t, cache[id(t)], mo if is_m else bo,
                                        MOMO_GIVEBACK if is_m else None,
                                        hard_tp=not is_m, mode=mode)
                tot += pnl
                n += 1
                if pnl > 0:
                    w += 1
                    ws.append(pnl)
                if kind == "trail":
                    tst += 1
            print(f"{nm:26}{tot:+9.1f}pp{100*w/n:6.1f}%{sum(ws)/max(1,len(ws)):+8.2f}%{tst:7}")
        print()


if __name__ == "__main__":
    main()
