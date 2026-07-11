"""AP4 — EU-Universe-Diagnose (Brief §1.6, 2026-07-11).

Frage: Warum haben 107 EU-Ticker in 4 Monaten 0 Signale produziert?
Methode: exakt die Scanner-Pipeline (Download-Batch, extract_ticker_frame,
scan_ticker relax=0), aber pro Ticker ein frischer Debug-Counter -> wir sehen
fuer JEDEN EU-Ticker den exakten Drop-Grund. Dazu avg_dv-Verteilung fuer die
Liquiditaets-Hypothese (MIN_DOLLAR_VOLUME=500k).
"""
import json
from collections import Counter, defaultdict

import ApexScan as sc


def main():
    eu = sc.clean_universe(sc.load_tickers(sc.EU_TICKER_FILE))
    print(f"EU-Universe nach clean_universe: {len(eu)} Ticker")

    print("Market-Regime holen...")
    regime = sc.get_market_regime()
    print(f"  {regime['summary']}")
    sector_cache = sc.load_sector_cache()

    # Download exakt wie run_scan
    import yfinance as yf
    frames = {}
    for batch in sc.chunked(eu, sc.BATCH_SIZE):
        raw = None
        try:
            with sc.suppress_output():
                raw = yf.download(tickers=" ".join(batch), period=sc.HISTORY_PERIOD,
                                  auto_adjust=True, progress=False, threads=True,
                                  group_by="ticker")
        except Exception as e:
            print(f"  batch download fail: {e}")
        for t in batch:
            f = sc.extract_ticker_frame(raw, t)
            if f is not None:
                frames[t] = f
    no_frame = [t for t in eu if t not in frames]
    print(f"Usable frames: {len(frames)}/{len(eu)}  (Download/NaN-Drops: {len(no_frame)})")
    if no_frame:
        print(f"  ohne Frame: {no_frame}")

    # Pro Ticker: frischer Counter -> exakter Drop-Grund
    reason_count = Counter()
    reason_examples = defaultdict(list)
    passed = []
    liquidity = []   # (ticker, avg_dv_moegl)
    for t, f in frames.items():
        dbg = Counter()
        res = sc.scan_ticker(t, f, regime, dbg, sector_cache, relax=0)
        if res is not None:
            passed.append(t)
            reason_count["PASSED"] += 1
            continue
        reason = dbg.most_common(1)[0][0] if dbg else "unknown"
        reason_count[reason] += 1
        if len(reason_examples[reason]) < 8:
            reason_examples[reason].append(t)
        # avg_dv unabhaengig vom Drop-Grund erfassen (Liquiditaets-Verteilung)
        try:
            close = float(f["Close"].iloc[-1])
            vol20 = float(f["Volume"].rolling(20).mean().iloc[-1])
            liquidity.append((t, close * vol20))
        except Exception:
            pass

    print(f"\n=== Drop-Gruende ({len(frames)} EU-Ticker, relax=0) ===")
    for reason, n in reason_count.most_common():
        ex = ", ".join(reason_examples.get(reason, [])[:8])
        print(f"  {reason:22} {n:4}  {('z.B. ' + ex) if ex else ''}")
    if passed:
        print(f"\nPASSED heute: {passed}")

    # Liquiditaets-Verteilung
    if liquidity:
        liquidity.sort(key=lambda x: x[1])
        vals = [v for _, v in liquidity]
        n = len(vals)
        print(f"\n=== avg_dv-Verteilung (EUR~USD, n={n}) — Schwelle MIN_DOLLAR_VOLUME=${sc.MIN_DOLLAR_VOLUME:,} ===")
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            print(f"  P{int(q*100):2}: ${vals[int(q*(n-1))]:,.0f}")
        below = sum(1 for v in vals if v < sc.MIN_DOLLAR_VOLUME)
        print(f"  unter Schwelle: {below}/{n}")
        print(f"  duennste 5: {[(t, f'${v:,.0f}') for t, v in liquidity[:5]]}")

    json.dump({
        "date": __import__('datetime').datetime.now().isoformat()[:16],
        "universe": len(eu), "frames": len(frames), "no_frame": no_frame,
        "drop_reasons": dict(reason_count),
        "examples": {k: v for k, v in reason_examples.items()},
        "passed": passed,
    }, open("apex_eu_diagnose_result.json", "w", encoding="utf-8"), indent=2)
    print("\nGespeichert: apex_eu_diagnose_result.json")


if __name__ == "__main__":
    main()
