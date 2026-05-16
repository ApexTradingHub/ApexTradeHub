"""
ApexLearn — Live-Performance Analysis Tool
============================================
Analysiert geschlossene Trades + offene Positionen, fokussiert auf:
  - Was war das Setup, was waren die Catalysts, was die Score-Komponenten?
  - Warum hat der Stop ausgelöst (oder TP getroffen)?
  - Welche Patterns wiederholen sich in Wins/Losses?

Aufruf:
  py apex_learn.py                    # standard report (letzte 30 Tage)
  py apex_learn.py --days 14          # nur letzte 2 Wochen
  py apex_learn.py --setup STAGE_2    # filter by setup
  py apex_learn.py --ticker MU        # einzelner ticker
  py apex_learn.py --news             # plus aktuelle yfinance-News für Loser
  py apex_learn.py --md > report.md   # markdown-output zum dump in Claude

Output ist ein strukturierter Report, den Claude lesen + analysieren kann,
um Verbesserungen am Scanner abzuleiten.
"""

import argparse
import json
import sys
import io
import contextlib
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Force UTF-8 stdout on Windows (default cp1252 chokes on → / emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@contextlib.contextmanager
def _suppress():
    o, e = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = o, e


def load_data():
    with open("apex_signals.json", "r", encoding="utf-8") as f:
        sigs = json.load(f)
    try:
        with open("apex_equity_results.json", "r", encoding="utf-8") as f:
            trades = json.load(f)
    except FileNotFoundError:
        trades = []
    try:
        with open("apex_market.json", "r", encoding="utf-8") as f:
            market = json.load(f)
    except FileNotFoundError:
        market = {}
    return sigs, trades, market


def join_signal_to_trade(trades, sigs):
    """Merge signal-data into each trade for full context."""
    lookup = {(s["date"], s["ticker"]): s for s in sigs}
    merged = []
    for t in trades:
        s = lookup.get((t["date"], t["ticker"]), {})
        m = {**s, **t}
        merged.append(m)
    return merged


def summarize_period(trades, days):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    sub = [t for t in trades if t["date"] >= cutoff]
    if not sub:
        return None
    wins = [t for t in sub if t["pnl_pct"] > 0]
    losses = [t for t in sub if t["pnl_pct"] <= 0]
    gw = sum(abs(t.get("pnl_usd", 0)) for t in wins)
    gl = sum(abs(t.get("pnl_usd", 0)) for t in losses)
    return {
        "n": len(sub), "wins": len(wins), "losses": len(losses),
        "wr": len(wins) / len(sub) * 100,
        "avg_w": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
        "avg_l": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        "pf": gw / gl if gl > 0 else 999,
        "total_pnl_pct": sum(t["pnl_pct"] for t in sub),
        "total_pnl_usd": sum(t.get("pnl_usd", 0) for t in sub),
    }


def fetch_news(tickers, limit=3):
    """Pull most recent yfinance news per ticker. Slow — only for loser deep-dive."""
    out = {}
    for tk in tickers:
        try:
            with _suppress():
                n = yf.Ticker(tk).news
            if n:
                out[tk] = [{"title": x.get("title"), "publisher": x.get("publisher"),
                            "link": x.get("link"),
                            "date": datetime.fromtimestamp(x.get("providerPublishTime", 0)).strftime("%Y-%m-%d")}
                           for x in n[:limit]]
            else:
                out[tk] = []
        except Exception:
            out[tk] = []
    return out


def gen_report(args):
    sigs, trades, market = load_data()
    merged = join_signal_to_trade(trades, sigs)

    if args.setup:
        merged = [t for t in merged if t.get("setup") == args.setup]
    if args.ticker:
        merged = [t for t in merged if t.get("ticker") == args.ticker]

    cutoff_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    recent = [t for t in merged if t["date"] >= cutoff_date]
    recent.sort(key=lambda x: x["date"], reverse=True)

    L = []  # lines
    L.append(f"# ApexLearn Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"")
    L.append(f"Window: last {args.days} days (since {cutoff_date})")
    L.append(f"Market: {market.get('mode','?')} — {market.get('summary','')}")
    L.append(f"Filter: setup={args.setup or 'ALL'}, ticker={args.ticker or 'ALL'}")
    L.append(f"")

    # === Section 1: Headlines ===
    L.append("## 1. Performance Headlines")
    L.append("")
    for days_back, label in [(7, "Last 7 days"), (14, "Last 14 days"),
                              (30, "Last 30 days"), (90, "Last 90 days")]:
        s = summarize_period(merged, days_back)
        if s:
            L.append(f"- **{label}**: {s['n']} trades | WR {s['wr']:.1f}% | "
                     f"PF {s['pf']:.2f} | AvgW {s['avg_w']:+.2f}% | AvgL {s['avg_l']:+.2f}% | "
                     f"Total ${s['total_pnl_usd']:+.2f}")
    L.append("")

    # === Section 2: Per-Setup-WR ===
    L.append("## 2. Per-Setup Performance (window)")
    L.append("")
    by_setup = defaultdict(list)
    for t in recent: by_setup[t.get("setup", "?")].append(t)
    L.append("| Setup | n | WR | PF | AvgWin | AvgLoss |")
    L.append("|---|---|---|---|---|---|")
    for setup, ts in sorted(by_setup.items(), key=lambda x: -len(x[1])):
        w = [t for t in ts if t["pnl_pct"] > 0]
        l = [t for t in ts if t["pnl_pct"] <= 0]
        gw = sum(abs(t.get("pnl_usd", 0)) for t in w)
        gl = sum(abs(t.get("pnl_usd", 0)) for t in l)
        pf = gw / gl if gl > 0 else 999
        aw = sum(t["pnl_pct"] for t in w) / len(w) if w else 0
        al = sum(t["pnl_pct"] for t in l) / len(l) if l else 0
        L.append(f"| {setup} | {len(ts)} | {len(w)/len(ts)*100:.1f}% | {pf:.2f} | "
                 f"{aw:+.2f}% | {al:+.2f}% |")
    L.append("")

    # === Section 3: Exit-Reason Breakdown ===
    L.append("## 3. Exit-Reason Verteilung")
    L.append("")
    ec = Counter(t.get("exit_reason", "?") for t in recent)
    L.append("| Exit | n | % | Avg PnL |")
    L.append("|---|---|---|---|")
    for r, n in ec.most_common():
        sub = [t for t in recent if t.get("exit_reason") == r]
        avg = sum(t["pnl_pct"] for t in sub) / len(sub) if sub else 0
        L.append(f"| {r} | {n} | {n/len(recent)*100:.0f}% | {avg:+.2f}% |")
    L.append("")

    # === Section 4: Quick-Stop-Detail (Setup-Quality-Hint) ===
    L.append("## 4. Quick-Stop-Analyse (Day 1-3 Stops = wahrscheinlich schlechtes Setup)")
    L.append("")
    sls = [t for t in recent if t.get("exit_reason") == "Stop Loss"]
    quick = [t for t in sls if t.get("exit_day", 99) <= 3]
    slow = [t for t in sls if t.get("exit_day", 99) > 3]
    L.append(f"- Total Stops: {len(sls)}")
    L.append(f"- Quick-Stops (1-3d): {len(quick)} ({len(quick)/max(len(sls),1)*100:.0f}% aller Stops)")
    L.append(f"- Slow Stops (>3d): {len(slow)}")
    if quick:
        L.append("")
        L.append("**Quick-Stop-Tickers (signal-quality-issue?):**")
        for t in sorted(quick, key=lambda x: x["pnl_pct"]):
            cats = [c for c in ["cat_pocket_pivot", "cat_vol_climax", "cat_earnings_beat",
                                 "cat_gap_pct"] if t.get(c)]
            cat_str = " " + " ".join(cats) if cats else ""
            L.append(f"- {t['ticker']} ({t['date']}, {t.get('setup','?')}) "
                     f"score={t.get('score','?')} → {t['pnl_pct']:+.1f}% D+{t.get('exit_day','?')}"
                     f"{cat_str}")
    L.append("")

    # === Section 5: Detail per closed trade ===
    L.append(f"## 5. Trade Detail (last {len(recent)} closed)")
    L.append("")
    for t in recent[:50]:
        outcome = "🟢 WIN" if t["pnl_pct"] > 0 else "🔴 LOSS"
        L.append(f"### {t['date']} | **{t['ticker']}** | {t.get('setup','?')} | {outcome} {t['pnl_pct']:+.2f}%")
        L.append(f"- Exit: **{t.get('exit_reason','?')}** at D+{t.get('exit_day','?')} "
                 f"(trigger D+{t.get('trigger_day','?')})")
        L.append(f"- Entry ${t.get('entry','?')} → Exit ${t.get('exit_price','?')}, "
                 f"Stop was ${t.get('stop','?')}, Target was ${t.get('target','?')}")
        L.append(f"- Score: {t.get('score','?')} | RR: {t.get('rr','?')} | "
                 f"RSI: {t.get('rsi','?')} | Sector: {t.get('sector','?')}")
        # Catalysts
        cat_marks = []
        if t.get("cat_pocket_pivot"): cat_marks.append("⚡ Pocket Pivot")
        if t.get("cat_vol_climax"):   cat_marks.append("📊 Vol Climax")
        if t.get("cat_gap_pct", 0) and t.get("cat_gap_pct", 0) > 2: cat_marks.append(f"↗️ Gap {t['cat_gap_pct']:+.1f}%")
        if t.get("cat_earnings_beat"): cat_marks.append("🎯 Earnings Beat")
        if (t.get("cat_short_pct") or 0) >= 15: cat_marks.append(f"🔥 Short {t['cat_short_pct']:.0f}%")
        if (t.get("cat_analyst_upside") or 0) >= 15: cat_marks.append(f"📈 Analyst +{t['cat_analyst_upside']:.0f}%")
        if cat_marks:
            L.append(f"- Catalysts: {', '.join(cat_marks)}")
        # Movement class
        if t.get("movement_class"):
            L.append(f"- Movement-Class: `{t['movement_class']}` (bonus {t.get('movement_bonus','?')})")
        # Setup-specific data
        if t.get("vcp_contraction") is not None:
            L.append(f"- VCP: contraction {t['vcp_contraction']:.1f}%, base_range {t.get('vcp_base_range','?')}%")
        if t.get("squeeze_short_pct") is not None:
            L.append(f"- SQUEEZE: short {t['squeeze_short_pct']:.1f}%, perf_5d {t.get('squeeze_perf_5d','?')}%")
        if t.get("stage2_ma150_rise") is not None:
            L.append(f"- STAGE_2: MA150 rise {t['stage2_ma150_rise']:.2f}%, base_width {t.get('stage2_base_width','?')}%")
        L.append("")

    # === Section 6: News for losers (slow, optional) ===
    if args.news:
        losers = [t for t in recent if t["pnl_pct"] <= 0][:10]
        if losers:
            L.append("## 6. News-Kontext für Loser (yfinance, optional)")
            L.append("")
            news = fetch_news([t["ticker"] for t in losers])
            for t in losers:
                items = news.get(t["ticker"], [])
                L.append(f"### {t['ticker']} ({t['date']}, {t['pnl_pct']:+.1f}%)")
                if items:
                    for n in items:
                        L.append(f"- [{n.get('date','?')}] {n.get('title','')} ({n.get('publisher','')})")
                else:
                    L.append("- (keine News abrufbar)")
                L.append("")

    # === Section 7: Open positions (filtered to ACTIVE setups + within hold window) ===
    # Phase G: only these setups are alive. Old PRE-ROCKET/POSITION/REVERSAL are
    # legacy artifacts that shouldn't show as "open".
    ACTIVE_SETUPS = {"BREAKOUT", "VCP", "SHORT_SQUEEZE", "STAGE_2"}
    HOLD_DAYS = {
        "1-3 weeks": 15, "2-4 weeks": 20, "2-6 weeks": 30,
        "3-8 weeks": 40, "4-8 weeks": 40, "4-12 weeks": 60, "8-16 weeks": 80,
    }
    MAX_TRIGGER_DAYS = 3

    closed_keys = {(t["date"], t["ticker"]) for t in trades}
    open_sigs = []
    expired_count = 0
    legacy_count = 0
    never_triggered_count = 0
    today_dt = datetime.now()
    for s in sigs:
        if (s["date"], s["ticker"]) in closed_keys:
            continue
        # 1. Filter legacy setup types (Phase G removed them)
        if s.get("setup") not in ACTIVE_SETUPS:
            legacy_count += 1
            continue
        # 2. Check age
        try:
            sig_dt = datetime.strptime(s["date"], "%Y-%m-%d")
        except Exception:
            continue
        days_old = (today_dt - sig_dt).days
        hold = HOLD_DAYS.get(s.get("horizon", ""), 30)
        # 3. Past hold period -> would be Time Exit, not "open"
        if days_old > hold + 5:
            expired_count += 1
            continue
        # 4. Past trigger window without becoming a trade -> stale, never triggered
        if days_old > MAX_TRIGGER_DAYS + 1 and (s["date"], s["ticker"]) not in closed_keys:
            never_triggered_count += 1
            continue
        open_sigs.append(s)

    L.append(f"## 7. Offene Positionen ({len(open_sigs)})")
    L.append("")
    if legacy_count or expired_count or never_triggered_count:
        L.append(f"_Filtered out: {legacy_count} legacy setup signals, "
                 f"{expired_count} past hold-period, "
                 f"{never_triggered_count} never-triggered (>3d alt, kein Entry)_")
        L.append("")
    if open_sigs:
        L.append("| Signal Date | Days Old | Ticker | Setup | Entry | Stop | Target | Score |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in sorted(open_sigs, key=lambda x: x["date"], reverse=True):
            try:
                age = (today_dt - datetime.strptime(s["date"], "%Y-%m-%d")).days
            except Exception:
                age = "?"
            L.append(f"| {s['date']} | D+{age} | {s['ticker']} | {s.get('setup','?')} | "
                     f"${s.get('buy_above','?')} | ${s.get('stop','?')} | ${s.get('target','?')} | "
                     f"{s.get('score','?')} |")
        L.append("")

    # === Section 8: Hypothesis seeds for Claude ===
    L.append("## 8. Hypothesen / Verbesserungsfragen für Claude")
    L.append("")
    L.append("Anregungen die Claude im Code prüfen könnte:")
    L.append("")
    # Quick-stop concentration
    if sls and len(quick) / len(sls) > 0.4:
        L.append(f"- ⚠️  **{len(quick)/len(sls)*100:.0f}% der Stops sind Quick-Stops** "
                 f"→ Setup-Quality-Filter prüfen (Closing-Strength, Inside-Day, …)")
    # Setup-specific loss patterns
    for setup, ts in by_setup.items():
        w = sum(1 for t in ts if t["pnl_pct"] > 0)
        if len(ts) >= 5 and w / len(ts) < 0.4:
            L.append(f"- ⚠️  **{setup} hat nur {w/len(ts)*100:.0f}% WR** (n={len(ts)}) "
                     f"→ Detection-Logik dieses Setups neu prüfen")
    # High-score losers
    hi_score_losers = [t for t in recent if t["pnl_pct"] <= 0 and t.get("score", 0) >= 85]
    if hi_score_losers:
        L.append(f"- 🔍 **{len(hi_score_losers)} High-Score-Loser** (score≥85): "
                 f"{', '.join(t['ticker'] for t in hi_score_losers[:5])} "
                 f"→ welche Score-Komponente predicted falsch?")
    # Sector concentration in losers
    loser_secs = Counter(t.get("sector", "?") for t in recent if t["pnl_pct"] <= 0)
    if loser_secs:
        worst_sec = loser_secs.most_common(1)[0]
        if worst_sec[1] >= 3:
            L.append(f"- 🔍 **Sektor {worst_sec[0]} produziert {worst_sec[1]} Loser** in der Periode "
                     f"→ sektor-spezifische Filter erwägen?")

    L.append("")
    L.append("---")
    L.append(f"_Report-Ende. Diesen Output an Claude füttern für Verbesserungs-Vorschläge._")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="Lookback window in days")
    ap.add_argument("--setup", type=str, default=None, help="Filter by setup name")
    ap.add_argument("--ticker", type=str, default=None, help="Filter by ticker")
    ap.add_argument("--news", action="store_true", help="Fetch news for losers (slow)")
    ap.add_argument("--md", action="store_true", help="Markdown to stdout only (no extras)")
    args = ap.parse_args()

    report = gen_report(args)
    if args.md:
        print(report)
        return
    print(report)
    # Also save to file for easy attach
    out_path = f"apex_learn_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[saved to {out_path}]")
    except Exception as e:
        print(f"[save failed: {e}]")


if __name__ == "__main__":
    main()
