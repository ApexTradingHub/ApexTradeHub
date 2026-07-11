"""SCORE_V2 Stufe 1 — Offline-Join-Replay (Brief AP2, 2026-07-11).

Frage: Kann eine rekalibrierte Score-Variante das Ranking wieder praediktiv machen?
Baseline: geloggter Score (Spearman Score->PnL = -0.066 auf n=137, Brief §1.3).

V2a (Anker, KEIN Fit): Ballast raus — rr-Term (rr x4), macd_bull (+8), risk_on (+5),
     Sektor-Bonus (±6) — und perf-Stack-Gewichte halbiert (0.8/0.5/0.2 -> 0.4/0.25/0.1).
     movement_class + Catalyst-Boni unveraendert (2J-validiert).
     risk_on + Sektor-Bonus sind nicht pro Signal geloggt -> Rekonstruktion aus
     SPY/QQQ/Sektor-ETF-Historie mit exakt der Scanner-Logik (get_market_regime).
V2b (Fit): Logistic Regression auf geloggten Komponenten, Train date <= 2026-05-31,
     Test Juni-Juli (Walk-Forward, sauber OOS).
Beide zusaetzlich als Cross-Sectional-Perzentil pro Scan-Tag (heilt Score-Inflation).

Akzeptanz (VORAB fixiert, Anti-Cherry-Pick — Brief §AP2):
  - OOS-Spearman (Juni-Juli) >= +0.15  UND
  - Top-Quartil-WR (V2-Ranking) >= Baseline-Top-Quartil-WR + 5pp
  - V2a ~ V2b (±0.05 Spearman) -> V2a nehmen (weniger Overfitting-Flaeche)
  - beide failen -> verwerfen, BACKLOG-Eintrag, KEIN Live-Change

Ehrlichkeits-Fussnote: V2a-Komponentenwahl hat das volle Sample gesehen (Forensik
2026-07-10) — Juni-Juli ist fuer V2a nur quasi-OOS. Fuer V2b ist der Walk-Forward sauber.
"""
import json
import math
from datetime import datetime

OOS_START = "2026-06-01"
TRAIN_END = "2026-05-31"

SECTOR_ETFS = {
    "Energy": "XLE", "Technology": "XLK", "Healthcare": "XLV",
    "Financial Services": "XLF", "Industrials": "XLI",
    "Consumer Defensive": "XLP", "Consumer Cyclical": "XLY",
    "Basic Materials": "XLB", "Utilities": "XLU",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


# ---------------------------------------------------------------- helpers
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def top_quartile_wr(scored):
    """scored: list of (ranking_value, pnl). Top 25% nach ranking_value."""
    s = sorted(scored, key=lambda x: -x[0])
    q = max(1, len(s) // 4)
    top = s[:q]
    return sum(1 for _, p in top if p > 0) / len(top) * 100, q


def percentile_per_day(rows, key):
    """Cross-Sectional-Rank pro Scan-Tag: 0..100 innerhalb des Tages."""
    from collections import defaultdict
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(r)
    for day, group in by_day.items():
        vals = sorted(r[key] for r in group)
        n = len(vals)
        for r in group:
            if n == 1:
                r[key + "_pct"] = 50.0
            else:
                below = sum(1 for v in vals if v < r[key])
                r[key + "_pct"] = below / (n - 1) * 100


# ---------------------------------------------------------------- market context
def build_market_context(dates):
    """Rekonstruiert risk_on + sector_momentum pro Datum (exakt Scanner-Logik)."""
    import yfinance as yf
    import pandas as pd

    symbols = ["SPY", "QQQ"] + list(SECTOR_ETFS.values())
    df = yf.download(symbols, start="2025-05-01", end="2026-07-12", interval="1d",
                     progress=False, auto_adjust=True, group_by="ticker")

    closes = {}
    for sym in symbols:
        c = df[sym]["Close"].dropna()
        closes[sym] = c

    def regime_at(c, ts):
        h = c.loc[:ts]
        if len(h) < 50:
            return "UNKNOWN"
        lc = float(h.iloc[-1])
        l20 = float(h.rolling(20).mean().iloc[-1])
        l50 = float(h.rolling(50).mean().iloc[-1])
        l200 = float(h.rolling(200).mean().iloc[-1]) if len(h) >= 200 else None
        if l200 is not None and not math.isnan(l200):
            if lc > l20 > l50 > l200:
                return "STRONG"
            if lc > l50 and l50 > l200:
                return "OK"
        else:
            if lc > l20 > l50:
                return "STRONG"
            if lc > l50:
                return "OK"
        return "WEAK"

    def perf20_at(c, ts):
        h = c.loc[:ts]
        if len(h) < 21:
            return 0.0
        return float((h.iloc[-1] / h.iloc[-21] - 1) * 100)

    ctx = {}
    for d in sorted(set(dates)):
        ts = pd.Timestamp(d)
        sr = regime_at(closes["SPY"], ts)
        qr = regime_at(closes["QQQ"], ts)
        risk_on = sr in {"STRONG", "OK"} and qr in {"STRONG", "OK"}
        sec_mom = {sec: perf20_at(closes[etf], ts) for sec, etf in SECTOR_ETFS.items()}
        ctx[d] = {"risk_on": risk_on, "sector_momentum": sec_mom}
    return ctx


def sector_bonus(sec_perf):
    if sec_perf >= 5:
        return 6
    if sec_perf >= 2:
        return 3
    if sec_perf < -5:
        return -5
    if sec_perf < -2:
        return -2
    return 0


# ---------------------------------------------------------------- V2b LogReg
FEATURES = ["rsi", "macd", "vol_ratio", "rr", "perf_20d", "perf_60d", "perf_120d",
            "base_range", "movement_bonus", "closing_strength", "pp", "vol_climax",
            "gap_pct", "short_pct"]


def featurize(r):
    s = r["sig"]
    return [
        float(s.get("rsi") or 50),
        1.0 if s.get("macd_bull") == "YES" else 0.0,
        min(float(s.get("vol_ratio") or 1), 5.0),
        min(float(s.get("rr") or 2), 5.0),
        min(max(float(s.get("perf_20d") or 0), -20), 40),
        min(max(float(s.get("perf_60d") or 0), -30), 60),
        min(max(float(s.get("perf_120d") or 0), -40), 100),
        min(float(s.get("base_range") or 15), 40),
        float(s.get("movement_bonus") or 0),
        float(s.get("closing_strength") or 0.5),
        1.0 if s.get("cat_pocket_pivot") else 0.0,
        1.0 if s.get("cat_vol_climax") else 0.0,
        min(max(float(s.get("cat_gap_pct") or 0), -5), 10),
        min(float(s.get("cat_short_pct") or 0), 25),
    ]


def logreg_fit(X, y, l2=1.0, lr=0.05, iters=4000):
    n, m = len(X), len(X[0])
    w = [0.0] * m
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * m
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            e = p - yi
            for j in range(m):
                gw[j] += e * xi[j]
            gb += e
        for j in range(m):
            w[j] -= lr * (gw[j] / n + l2 * w[j] / n)
        b -= lr * gb / n
    return w, b


def logreg_predict(w, b, X):
    return [1 / (1 + math.exp(-max(min(b + sum(wj * xj for wj, xj in zip(w, xi)), 30), -30)))
            for xi in X]


# ---------------------------------------------------------------- main
def main():
    sigs = json.load(open("apex_signals.json", encoding="utf-8"))
    eqt = json.load(open("apex_equity_results.json", encoding="utf-8"))

    eq_by_key = {(t["ticker"], str(t["date"])[:10]): t for t in eqt}
    rows = []
    for s in sigs:
        if s.get("setup") != "BREAKOUT":
            continue
        key = (s["ticker"], str(s["date"])[:10])
        t = eq_by_key.get(key)
        if not t or t.get("pnl_pct") is None:
            continue
        rows.append({"date": key[1], "ticker": key[0], "sig": s,
                     "pnl": float(t["pnl_pct"]), "score": float(s.get("score") or 0)})
    print(f"Join: {len(rows)} BREAKOUT-Paare "
          f"({sum(1 for r in rows if r['date'] <= TRAIN_END)} Train / "
          f"{sum(1 for r in rows if r['date'] >= OOS_START)} OOS Juni-Juli)")

    # --- Markt-Kontext rekonstruieren (risk_on + Sektor-Momentum pro Datum) ---
    print("Rekonstruiere Markt-Kontext (SPY/QQQ/Sektor-ETFs)...")
    ctx = build_market_context([r["date"] for r in rows])

    # --- V2a ---
    for r in rows:
        s = r["sig"]
        c = ctx[r["date"]]
        rr_term = min(float(s.get("rr") or 0), 5.0) * 4
        macd_term = 8.0 if s.get("macd_bull") == "YES" else 0.0
        risk_term = 5.0 if c["risk_on"] else 0.0
        sec_perf = c["sector_momentum"].get(s.get("sector", ""), 0.0)
        sec_term = float(sector_bonus(sec_perf)) if s.get("sector") in SECTOR_ETFS else 0.0
        p20 = min(max(float(s.get("perf_20d") or 0), 0), 20) * 0.8
        p60 = min(max(float(s.get("perf_60d") or 0), 0), 35) * 0.5
        p120 = min(max(float(s.get("perf_120d") or 0), 0), 50) * 0.2
        half_perf = 0.5 * (p20 + p60 + p120)
        r["v2a"] = r["score"] - rr_term - macd_term - risk_term - sec_term - half_perf

    # --- Perzentil-Varianten (pro Scan-Tag) ---
    percentile_per_day(rows, "score")
    percentile_per_day(rows, "v2a")

    # --- V2b: LogReg walk-forward ---
    train = [r for r in rows if r["date"] <= TRAIN_END]
    test = [r for r in rows if r["date"] >= OOS_START]
    Xtr = [featurize(r) for r in train]
    ytr = [1.0 if r["pnl"] > 0 else 0.0 for r in train]
    # Standardisieren mit Train-Statistik
    m = len(Xtr[0])
    mu = [sum(x[j] for x in Xtr) / len(Xtr) for j in range(m)]
    sd = [max(math.sqrt(sum((x[j] - mu[j]) ** 2 for x in Xtr) / len(Xtr)), 1e-9) for j in range(m)]
    Xtr_z = [[(x[j] - mu[j]) / sd[j] for j in range(m)] for x in Xtr]
    w, b = logreg_fit(Xtr_z, ytr)
    for r in rows:
        xz = [(v - mu[j]) / sd[j] for j, v in enumerate(featurize(r))]
        r["v2b"] = logreg_predict(w, b, [xz])[0]
    percentile_per_day(rows, "v2b")

    print("\nV2b-Gewichte (standardisiert, |w| = Einfluss):")
    for name, wj in sorted(zip(FEATURES, w), key=lambda x: -abs(x[1])):
        print(f"  {name:18} {wj:+.3f}")

    # --- Evaluation ---
    def evaluate(rows_, label):
        pnls = [r["pnl"] for r in rows_]
        out = {}
        for key in ["score", "score_pct", "v2a", "v2a_pct", "v2b", "v2b_pct"]:
            vals = [r[key] for r in rows_]
            sp = spearman(vals, pnls)
            wr, q = top_quartile_wr(list(zip(vals, pnls)))
            out[key] = (sp, wr, q)
        print(f"\n=== {label} (n={len(rows_)}) ===")
        print(f"{'Variante':12} {'Spearman':>9} {'TopQ-WR':>8} {'nQ':>4}")
        for key, (sp, wr, q) in out.items():
            print(f"{key:12} {sp:>+9.3f} {wr:>7.1f}% {q:>4}")
        return out

    evaluate(rows, "FULL SAMPLE (Referenz)")
    res = evaluate(test, "OOS Juni-Juli (ENTSCHEIDUNGSFENSTER)")

    # --- Entscheidung nach vorab fixierten Kriterien ---
    base_sp, base_wr, _ = res["score"]
    print(f"\n=== ENTSCHEIDUNG (Akzeptanz: OOS-Spearman >= +0.15 UND TopQ-WR >= Baseline {base_wr:.1f}% + 5pp) ===")
    verdicts = {}
    for key in ["v2a", "v2a_pct", "v2b", "v2b_pct"]:
        sp, wr, _ = res[key]
        ok = sp >= 0.15 and wr >= base_wr + 5
        verdicts[key] = (sp, wr, ok)
        print(f"  {key:8} Spearman {sp:+.3f} {'OK' if sp >= 0.15 else 'FAIL':4} | "
              f"TopQ {wr:.1f}% {'OK' if wr >= base_wr + 5 else 'FAIL':4} "
              f"=> {'GO' if ok else 'NO-GO'}")

    go_a = verdicts["v2a"][2] or verdicts["v2a_pct"][2]
    go_b = verdicts["v2b"][2] or verdicts["v2b_pct"][2]
    if go_a and go_b:
        best_a = max(verdicts["v2a"][0], verdicts["v2a_pct"][0])
        best_b = max(verdicts["v2b"][0], verdicts["v2b_pct"][0])
        pick = "V2a" if best_b - best_a <= 0.05 else "V2b"
        print(f"\nBeide GO. |Diff| Spearman {abs(best_b-best_a):.3f} -> {pick} "
              f"({'V2a bevorzugt bei Gleichstand' if pick == 'V2a' else 'V2b klar besser'})")
    elif go_a:
        print("\nV2a GO, V2b NO-GO -> V2a in Stufe 2 (2J-Backtest).")
    elif go_b:
        print("\nV2b GO, V2a NO-GO -> V2b in Stufe 2 (2J-Backtest). "
              "Caveat: Fit-Variante, Overfitting-Flaeche beachten.")
    else:
        print("\nBEIDE NO-GO -> verwerfen, BACKLOG-Eintrag, KEIN Live-Change (Brief §AP2).")


if __name__ == "__main__":
    main()
