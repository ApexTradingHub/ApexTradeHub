"""Einmalige Datenreparatur 2026-07-11 (Brief AP1 Schritt 3 + AP3 Backfill).

1. Rescued Trades (intraday_rescued=True) in open[] + closed[]:
   source zurueck auf "intraday_momentum" (war durch den Rescue-Pfad-Bug
   auf "momentum_filler" umgeschrieben — kontaminierte die Filler-Statistik).
2. Sektor-Backfill: alle Positionen mit sector "Unknown" via sector_cache.json,
   Misses via yfinance nachschlagen. Neue Sektoren werden auch in
   sector_cache.json ergaenzt (einmalig, damit Scanner+Trader sie kuenftig haben).

Idempotent — mehrfaches Ausfuehren ist harmlos.
"""
import json
import time

POS_FILE = "apex_positions.json"
CACHE_FILE = "sector_cache.json"


def main():
    state = json.load(open(POS_FILE, encoding="utf-8"))
    cache = json.load(open(CACHE_FILE, encoding="utf-8"))

    # --- 1. Source-Repair der Rescues ---
    fixed_source = 0
    for pool in (state.get("open", []), state.get("closed", [])):
        for p in pool:
            if p.get("intraday_rescued") and p.get("source") == "momentum_filler":
                p["source"] = "intraday_momentum"
                fixed_source += 1
                print(f"  source-fix: {p['ticker']:6} ({p.get('id','?')})")
    print(f"1. Source-Repair: {fixed_source} Trades korrigiert\n")

    # --- 2. Sektor-Backfill ---
    unknown = [p for pool in (state.get("open", []), state.get("closed", []))
               for p in pool if p.get("sector") == "Unknown"]
    tickers = sorted({p["ticker"] for p in unknown})
    print(f"2. Sektor-Backfill: {len(unknown)} Positionen, {len(tickers)} unique Ticker")

    resolved = {}
    misses = []
    for tk in tickers:
        sec = cache.get(tk)
        if not sec:
            try:
                import yfinance as yf
                sec = yf.Ticker(tk).info.get("sector")
                time.sleep(0.3)   # rate-limit-freundlich
            except Exception as e:
                print(f"  yf-fail {tk}: {e}")
                sec = None
        if sec:
            resolved[tk] = sec
            if tk not in cache:
                cache[tk] = sec
        else:
            misses.append(tk)

    filled = 0
    for p in unknown:
        sec = resolved.get(p["ticker"])
        if sec:
            p["sector"] = sec
            filled += 1
    print(f"   aufgeloest: {len(resolved)}/{len(tickers)} Ticker -> {filled} Positionen gefuellt")
    if misses:
        print(f"   weiterhin Unknown: {misses}")

    json.dump(state, open(POS_FILE, "w", encoding="utf-8"), indent=2)
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(f"\nGespeichert: {POS_FILE} + {CACHE_FILE}")


if __name__ == "__main__":
    main()
