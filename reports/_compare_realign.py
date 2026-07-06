"""Vergleicht Baseline vs Score-Realign Backtest, prueft die 3 Akzeptanz-Gates.
Usage: python reports/_compare_realign.py apex_backtest_baseline.json apex_backtest_realign.json
"""
import json, sys

def stats(rows):
    bo = [r for r in rows if r.get("setup") == "BREAKOUT"]
    if not bo:
        return None
    wins = [r for r in bo if r["pnl_pct"] > 0]
    losses = [r for r in bo if r["pnl_pct"] <= 0]
    gw = sum(r["pnl_usd"] for r in wins)
    gl = abs(sum(r["pnl_usd"] for r in losses))
    return {
        "n": len(bo),
        "wr": len(wins) / len(bo) * 100,
        "pf": gw / gl if gl > 0 else float("inf"),
        "total_pnl": sum(r["pnl_usd"] for r in bo),
        "avg_pnl_pct": sum(r["pnl_pct"] for r in bo) / len(bo),
    }

def bucket_wr(rows, lo, hi):
    """WR fuer score-bucket [lo, hi)."""
    bo = [r for r in rows if r.get("setup") == "BREAKOUT" and lo <= r.get("score", 0) < hi]
    if not bo:
        return (0, 0.0)
    wins = sum(1 for r in bo if r["pnl_pct"] > 0)
    return (len(bo), wins / len(bo) * 100)

base = json.load(open(sys.argv[1]))
real = json.load(open(sys.argv[2]))
b, r = stats(base), stats(real)

print("=== BREAKOUT OVERALL ===")
print(f"Baseline: n={b['n']:>4}  WR={b['wr']:.1f}%  PF={b['pf']:.2f}  avgPnL={b['avg_pnl_pct']:+.2f}%  total=${b['total_pnl']:.0f}")
print(f"Realign:  n={r['n']:>4}  WR={r['wr']:.1f}%  PF={r['pf']:.2f}  avgPnL={r['avg_pnl_pct']:+.2f}%  total=${r['total_pnl']:.0f}")
print(f"Delta:    n={r['n']-b['n']:+d} ({r['n']/b['n']*100:.0f}%)  WR={r['wr']-b['wr']:+.1f}pp  PF={r['pf']-b['pf']:+.2f}")
print()

print("=== SCORE-BUCKETS (WR by bucket) ===")
for lo, hi, lbl in [(70,80,"70-80"),(80,90,"80-90"),(90,100,"90-100"),(100,999,"100+")]:
    bn, bw = bucket_wr(base, lo, hi)
    rn, rw = bucket_wr(real, lo, hi)
    print(f"  {lbl:>7}: base n={bn:>3} WR={bw:.0f}%  |  realign n={rn:>3} WR={rw:.0f}%")
print()

print("=== AKZEPTANZ-GATES ===")
gates = {
    "1. Signal-Count >= 95% Baseline (USER-CONSTRAINT)": r["n"] >= b["n"] * 0.95,
    "2. Overall BO WR >= Baseline +1.5pp":               (r["wr"] - b["wr"]) >= 1.5,
    "3. PF >= Baseline -0.1":                            (r["pf"] - b["pf"]) >= -0.1,
}
for name, ok in gates.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print()
verdict = "GO (live nach ApexScan.py)" if all(gates.values()) else "NO-GO"
print(f"OVERALL: {verdict}  ({sum(gates.values())}/3 gates)")
