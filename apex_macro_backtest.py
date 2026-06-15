"""
ApexMacroBacktest — joint apex_equity_results.json mit FRED-Series (VIX, HY-OAS)
am Entry-Datum und buckets WR/PF.

Hypothese aus AFRM/IBKR-Postmortems: BREAKOUT-WR sinkt bei VIX>=22 oder HY-OAS>=4.0
zum Entry-Zeitpunkt (macro_selloff_correlates_all_stocks).

Run:
    $env:FRED_API_KEY = "dein_key"
    py apex_macro_backtest.py
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY = os.environ.get("FRED_API_KEY", "")

REPO = Path(__file__).parent
EQUITY_FILE = REPO / "apex_equity_results.json"
OUT_FILE = REPO / "reports" / "macro_backtest.md"

# Series we fetch full history for
SERIES = {
    "vix":    "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
}

# Buckets
VIX_BUCKETS = [
    ("VIX <16 (Quiet)",   None, 16.0),
    ("VIX 16-20 (Normal)", 16.0, 20.0),
    ("VIX 20-22 (Elevated)", 20.0, 22.0),
    ("VIX 22-25 (Stress)",  22.0, 25.0),
    ("VIX 25+ (Panic)",    25.0, None),
]

HY_BUCKETS = [
    ("HY <3.0 (Tight)",    None, 3.0),
    ("HY 3.0-3.5 (Normal)", 3.0, 3.5),
    ("HY 3.5-4.0 (Elevated)", 3.5, 4.0),
    ("HY 4.0-5.0 (Stress)",  4.0, 5.0),
    ("HY 5.0+ (Crisis)",    5.0, None),
]


def fetch_history(series_id: str, start: str = "2024-01-01") -> dict:
    """Vollstaendige tagesweise FRED-Serie als {YYYY-MM-DD: float}."""
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    url = f"{FRED_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ApexNext/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    result = {}
    for o in data.get("observations", []):
        v = o.get("value")
        if v in (".", "", None):
            continue
        try:
            result[o["date"]] = float(v)
        except ValueError:
            continue
    return result


def lookup_on_or_before(series: dict, target: str, max_back_days: int = 7) -> float | None:
    """Hole Wert am Datum oder davor (VIX skippt WE)."""
    d = datetime.strptime(target, "%Y-%m-%d").date()
    for i in range(max_back_days):
        key = (d - timedelta(days=i)).isoformat()
        if key in series:
            return series[key]
    return None


def bucketize(value: float | None, buckets) -> str | None:
    if value is None:
        return None
    for label, lo, hi in buckets:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return label
    return None


def stats(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": None, "pf": None, "avg_pnl": None}
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    return {
        "n": n,
        "wr": len(wins) / n * 100,
        "pf": pf,
        "avg_pnl": sum(t["pnl_pct"] for t in trades) / n,
        "avg_win": (gross_win / len(wins)) if wins else 0,
        "avg_loss": -(gross_loss / len(losses)) if losses else 0,
    }


def fmt_stats(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    conf = "HIGH" if s["n"] >= 30 else "MED" if s["n"] >= 15 else "LOW"
    pf = f"{s['pf']:.2f}" if s["pf"] is not None else "∞"
    return f"n={s['n']} | WR {s['wr']:.1f}% | PF {pf} | avg {s['avg_pnl']:+.2f}% | [{conf}]"


def render_table(title: str, buckets, bucketed_trades: dict) -> list:
    lines = [f"### {title}", "", "| Bucket | n | WR | PF | AvgPnL | Conf |", "|---|---:|---:|---:|---:|---|"]
    for label, _, _ in buckets:
        trades = bucketed_trades.get(label, [])
        s = stats(trades)
        if s["n"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — |")
            continue
        conf = "HIGH" if s["n"] >= 30 else "MED" if s["n"] >= 15 else "LOW"
        pf = f"{s['pf']:.2f}" if s["pf"] is not None else "∞"
        lines.append(f"| {label} | {s['n']} | {s['wr']:.1f}% | {pf} | {s['avg_pnl']:+.2f}% | {conf} |")
    return lines


def main():
    if not FRED_KEY:
        print("ERROR: FRED_API_KEY env var nicht gesetzt.", file=sys.stderr)
        sys.exit(1)

    # Load trades
    trades = json.loads(EQUITY_FILE.read_text(encoding="utf-8"))
    earliest = min(t["date"] for t in trades)
    print(f"Loaded {len(trades)} trades, earliest {earliest}")

    # Fetch FRED history starting 60 days before earliest trade
    start_date = (datetime.strptime(earliest, "%Y-%m-%d").date() - timedelta(days=60)).isoformat()
    print(f"Fetching FRED series from {start_date}...")
    series_data = {}
    for label, series_id in SERIES.items():
        series_data[label] = fetch_history(series_id, start=start_date)
        print(f"  {label} ({series_id}): {len(series_data[label])} observations")

    # Enrich each trade with vix/hy at entry
    enriched = []
    missed = 0
    for t in trades:
        vix = lookup_on_or_before(series_data["vix"], t["date"])
        hy = lookup_on_or_before(series_data["hy_oas"], t["date"])
        if vix is None and hy is None:
            missed += 1
            continue
        enriched.append({
            **t,
            "vix_at_entry": vix,
            "hy_at_entry": hy,
            "vix_bucket": bucketize(vix, VIX_BUCKETS),
            "hy_bucket": bucketize(hy, HY_BUCKETS),
        })
    print(f"Enriched {len(enriched)} / {len(trades)} trades ({missed} missed FRED-lookup)")

    # Group by setup
    by_setup = defaultdict(list)
    for t in enriched:
        by_setup[t["setup"]].append(t)

    # Build report
    out = [
        f"# ApexMacro Backtest — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Joined {len(enriched)} of {len(trades)} trades** with FRED VIX + HY-OAS at entry date.",
        "",
        "**Hypothese:** BREAKOUT-WR fällt bei VIX≥22 oder HY-OAS≥4.0 (macro selloff correlates).",
        "**Akzeptanz:** n≥30 für CONFIRMED, ≥15 für TENTATIVE. Lift ≥10pp WR vs baseline = signal.",
        "",
        "---",
        "",
    ]

    # All trades baseline
    baseline = stats(enriched)
    out.append(f"## Baseline (alle Setups)")
    out.append("")
    out.append(f"- {fmt_stats(baseline)}")
    out.append("")

    # Per-setup VIX + HY breakdown
    for setup in sorted(by_setup.keys()):
        setup_trades = by_setup[setup]
        setup_baseline = stats(setup_trades)
        out.append(f"## {setup} — Baseline: {fmt_stats(setup_baseline)}")
        out.append("")

        by_vix = defaultdict(list)
        by_hy = defaultdict(list)
        for t in setup_trades:
            if t["vix_bucket"]:
                by_vix[t["vix_bucket"]].append(t)
            if t["hy_bucket"]:
                by_hy[t["hy_bucket"]].append(t)

        out.extend(render_table("VIX-Bucket", VIX_BUCKETS, by_vix))
        out.append("")
        out.extend(render_table("HY-OAS-Bucket", HY_BUCKETS, by_hy))
        out.append("")

    # Combined regime
    out.append("## Combined Macro-Regime")
    out.append("")
    out.append("Regime-Definition: RISK_OFF wenn VIX≥25 ODER HY≥5.0 | ELEVATED wenn VIX≥20 ODER HY≥3.5 | sonst RISK_ON")
    out.append("")
    by_regime = defaultdict(list)
    for t in enriched:
        vix = t.get("vix_at_entry")
        hy = t.get("hy_at_entry")
        if (vix and vix >= 25) or (hy and hy >= 5.0):
            r = "RISK_OFF"
        elif (vix and vix >= 20) or (hy and hy >= 3.5):
            r = "ELEVATED"
        else:
            r = "RISK_ON"
        by_regime[r].append(t)

    out.append("| Regime | All Setups | BREAKOUT only |")
    out.append("|---|---|---|")
    for r in ["RISK_ON", "ELEVATED", "RISK_OFF"]:
        rt = by_regime.get(r, [])
        bo = [t for t in rt if t["setup"] == "BREAKOUT"]
        all_s = fmt_stats(stats(rt))
        bo_s = fmt_stats(stats(bo))
        out.append(f"| **{r}** | {all_s} | {bo_s} |")

    out.append("")
    out.append("---")
    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append("- WR-Lift ≥10pp + n≥30 in einem Bucket → potentieller Score-Penalty- oder Gate-Kandidat")
    out.append("- Niedrige n (<15) = nicht aussagekräftig, nur als Trend")
    out.append("- Vor Live-Integration: in apex_backtest_v2.py mit `--only-setup BREAKOUT --exclude-regime RISK_OFF` re-validieren")

    report = "\n".join(out)
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport: {OUT_FILE}")
    print()
    print(report)


if __name__ == "__main__":
    main()
