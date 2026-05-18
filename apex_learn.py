"""
ApexLearn — Persistent Performance Knowledge Base + Report Generator
=====================================================================
Aufgabe: aus geschlossenen Live-Trades langfristige Fakten extrahieren,
nicht nur Snapshot-Hypothesen. Jeder Run akkumuliert Wissen in
knowledge/apex_knowledge.json.

Aufruf:
  py apex_learn.py                      # standard (last 30 days window + knowledge update)
  py apex_learn.py --days 14            # short window
  py apex_learn.py --setup STAGE_2      # filter window by setup
  py apex_learn.py --ticker MU          # einzelner ticker
  py apex_learn.py --news               # plus yfinance news für loser
  py apex_learn.py --no-update          # report nur, keine knowledge update
  py apex_learn.py --md                 # markdown to stdout (kein file)

Output:
  reports/learn_latest.md               (immer überschrieben)
  reports/learn_YYYYMMDD_HHMM.md        (historisch)
  knowledge/apex_knowledge.json         (akkumuliertes Wissen über alle Trades)

Knowledge contents:
  - Per-Ticker: Trades/WR/AvgPnL/Best/Worst pro Symbol
  - Per-Setup: Lifetime WR + Feature-Korrelationen (perf_120, rsi, vol_ratio)
  - Catalysts: WR mit/ohne jeden Catalyst, Lift, Confidence-Level
  - Score-Calibration: Score-Bucket → Actual WR (validiert Score-Gate)
  - Failure Modes: Kategorisierte Loss-Patterns
"""

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPORTS_DIR   = Path("reports")
KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_FILE = KNOWLEDGE_DIR / "apex_knowledge.json"

# Confidence thresholds for sample sizes
CONF_HIGH = 30
CONF_MED  = 15

# Active setups (Phase G) — used in open-positions filter
ACTIVE_SETUPS = {"BREAKOUT", "VCP", "SHORT_SQUEEZE", "STAGE_2"}

# User-facing display names (mirror of dashboard.html SETUP_META)
# Internal data still uses raw codes — only display layer is changed.
SETUP_DISPLAY = {
    "STAGE_2":       "🚀 Trend",
    "VCP":           "🔹 Bounceback",
    "SHORT_SQUEEZE": "🔥 Bet",
    "BREAKOUT":      "🔵 Breakout",
    "REVERSAL":      "Reversal",
}
def disp(setup_code):
    """Return user-facing display name for a setup code (fallback: raw code)."""
    return SETUP_DISPLAY.get(setup_code, setup_code)
HOLD_DAYS = {
    "1-3 weeks": 15, "2-4 weeks": 20, "2-6 weeks": 30,
    "3-8 weeks": 40, "4-8 weeks": 40, "4-12 weeks": 60, "8-16 weeks": 80,
}
MAX_TRIGGER_DAYS = 3


@contextlib.contextmanager
def _suppress():
    o, e = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = o, e


# =============================================================
# DATA LOADING
# =============================================================
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
    lookup = {(s["date"], s["ticker"]): s for s in sigs}
    merged = []
    for t in trades:
        s = lookup.get((t["date"], t["ticker"]), {})
        m = {**s, **t}
        merged.append(m)
    return merged


# =============================================================
# KNOWLEDGE BASE — persistent, akkumuliert über alle Trades
# =============================================================
def conf_level(n):
    if n >= CONF_HIGH: return "HIGH"
    if n >= CONF_MED:  return "MED"
    return "LOW"


def _wr(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t.get("pnl_pct", 0) > 0) / len(trades)


def _pf(trades):
    gw = sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_pct", 0) > 0)
    gl = sum(abs(t.get("pnl_usd", 0)) for t in trades if t.get("pnl_pct", 0) <= 0)
    return round(gw / gl, 3) if gl > 0 else None


def compute_ticker_stats(merged):
    """Per-Ticker: alle Trades je Symbol mit WR/Avg/Best/Worst."""
    by_ticker = defaultdict(list)
    for t in merged:
        by_ticker[t["ticker"]].append(t)
    out = {}
    for tk, ts in by_ticker.items():
        if len(ts) < 2:
            continue  # Singleton-Trades sind kein Signal
        wins = [t for t in ts if t.get("pnl_pct", 0) > 0]
        out[tk] = {
            "n": len(ts),
            "wins": len(wins),
            "wr": round(_wr(ts), 3),
            "avg_pnl_pct": round(sum(t.get("pnl_pct", 0) for t in ts) / len(ts), 2),
            "total_pnl_usd": round(sum(t.get("pnl_usd", 0) for t in ts), 2),
            "best_pnl": round(max(t.get("pnl_pct", 0) for t in ts), 2),
            "worst_pnl": round(min(t.get("pnl_pct", 0) for t in ts), 2),
            "setups_seen": sorted(set(t.get("setup", "?") for t in ts)),
            "confidence": conf_level(len(ts)),
        }
    return out


def compute_setup_stats(merged):
    """Per-Setup: Lifetime-WR + Feature-Buckets."""
    by_setup = defaultdict(list)
    for t in merged:
        by_setup[t.get("setup", "?")].append(t)

    out = {}
    for setup, ts in by_setup.items():
        if len(ts) < 5:
            continue
        base_wr = _wr(ts)
        wins = [t for t in ts if t.get("pnl_pct", 0) > 0]
        losses = [t for t in ts if t.get("pnl_pct", 0) <= 0]

        # Feature-Buckets
        feature_buckets = {}

        # perf_120d buckets
        for label, fn in [
            ("perf_120_lt_0",  lambda t: (t.get("perf_120d") or 0) < 0),
            ("perf_120_0_25",  lambda t: 0 <= (t.get("perf_120d") or 0) < 25),
            ("perf_120_25_50", lambda t: 25 <= (t.get("perf_120d") or 0) < 50),
            ("perf_120_50plus", lambda t: (t.get("perf_120d") or 0) >= 50),
        ]:
            sub = [t for t in ts if fn(t)]
            if len(sub) >= 5:
                feature_buckets[label] = {
                    "n": len(sub),
                    "wr": round(_wr(sub), 3),
                    "lift_vs_setup_baseline": round(_wr(sub) - base_wr, 3),
                    "confidence": conf_level(len(sub)),
                }

        # RSI buckets
        for label, fn in [
            ("rsi_lt_50",   lambda t: (t.get("rsi") or 0) < 50),
            ("rsi_50_60",   lambda t: 50 <= (t.get("rsi") or 0) < 60),
            ("rsi_60_65",   lambda t: 60 <= (t.get("rsi") or 0) < 65),
            ("rsi_65_70",   lambda t: 65 <= (t.get("rsi") or 0) < 70),
            ("rsi_70plus",  lambda t: (t.get("rsi") or 0) >= 70),
        ]:
            sub = [t for t in ts if fn(t)]
            if len(sub) >= 5:
                feature_buckets[label] = {
                    "n": len(sub),
                    "wr": round(_wr(sub), 3),
                    "lift_vs_setup_baseline": round(_wr(sub) - base_wr, 3),
                    "confidence": conf_level(len(sub)),
                }

        # vol_ratio buckets
        for label, fn in [
            ("vol_lt_1",     lambda t: (t.get("vol_ratio") or 0) < 1.0),
            ("vol_1_15",     lambda t: 1.0 <= (t.get("vol_ratio") or 0) < 1.5),
            ("vol_15_25",    lambda t: 1.5 <= (t.get("vol_ratio") or 0) < 2.5),
            ("vol_25plus",   lambda t: (t.get("vol_ratio") or 0) >= 2.5),
        ]:
            sub = [t for t in ts if fn(t)]
            if len(sub) >= 5:
                feature_buckets[label] = {
                    "n": len(sub),
                    "wr": round(_wr(sub), 3),
                    "lift_vs_setup_baseline": round(_wr(sub) - base_wr, 3),
                    "confidence": conf_level(len(sub)),
                }

        out[setup] = {
            "n": len(ts),
            "wr": round(base_wr, 3),
            "pf": _pf(ts),
            "avg_win_pct":  round(sum(t.get("pnl_pct", 0) for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss_pct": round(sum(t.get("pnl_pct", 0) for t in losses) / len(losses), 2) if losses else 0,
            "confidence": conf_level(len(ts)),
            "feature_buckets": feature_buckets,
        }
    return out


def compute_catalyst_effectiveness(merged):
    """For each catalyst: WR with vs without, lift, confidence."""
    out = {}
    boolean_cats = [
        "cat_pocket_pivot", "cat_vol_climax", "cat_earnings_beat",
        "cat_earnings_blackout", "inside_day",
    ]
    for cat in boolean_cats:
        present = [t for t in merged if t.get(cat) is True]
        absent  = [t for t in merged if t.get(cat) is False or t.get(cat) is None]
        if len(present) < 5 or len(absent) < 5:
            continue
        wr_p = _wr(present)
        wr_a = _wr(absent)
        out[cat] = {
            "n_present": len(present),
            "n_absent":  len(absent),
            "wr_when_present": round(wr_p, 3),
            "wr_when_absent":  round(wr_a, 3),
            "lift": round(wr_p - wr_a, 3),
            "confidence": conf_level(len(present)),
        }

    # Numerical catalysts (thresholds)
    for cat, threshold, label in [
        ("cat_gap_pct", 2.0, "gap_gt_2pct"),
        ("cat_short_pct", 15.0, "short_gt_15pct"),
        ("cat_analyst_upside", 15.0, "analyst_upside_gt_15pct"),
    ]:
        present = [t for t in merged if (t.get(cat) or 0) >= threshold]
        absent  = [t for t in merged if (t.get(cat) or 0) < threshold]
        if len(present) < 5 or len(absent) < 5:
            continue
        wr_p = _wr(present)
        wr_a = _wr(absent)
        out[label] = {
            "n_present": len(present),
            "n_absent":  len(absent),
            "wr_when_present": round(wr_p, 3),
            "wr_when_absent":  round(wr_a, 3),
            "lift": round(wr_p - wr_a, 3),
            "confidence": conf_level(len(present)),
        }

    return out


def compute_score_calibration(merged):
    """Per-Setup: actual WR by score bucket — validiert Score-Gate-Setting."""
    by_setup = defaultdict(list)
    for t in merged:
        by_setup[t.get("setup", "?")].append(t)

    out = {}
    for setup, ts in by_setup.items():
        if len(ts) < 10:
            continue
        buckets = {}
        for low, high in [(40, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 999)]:
            sub = [t for t in ts if low <= (t.get("score") or 0) < high]
            if len(sub) >= 5:
                buckets[f"{low}-{high}"] = {
                    "n": len(sub),
                    "wr": round(_wr(sub), 3),
                    "confidence": conf_level(len(sub)),
                }
        if buckets:
            out[setup] = buckets
    return out


def compute_failure_modes(merged):
    """Kategorisierte Loss-Patterns."""
    losses = [t for t in merged if t.get("pnl_pct", 0) <= 0]
    if not losses: return {}

    modes = {
        "quick_stop_1_3d": [t for t in losses
                             if t.get("exit_reason") == "Stop Loss" and t.get("exit_day", 99) <= 3],
        "slow_stop_4plus": [t for t in losses
                             if t.get("exit_reason") == "Stop Loss" and t.get("exit_day", 99) > 3],
        "time_exit_negative": [t for t in losses if t.get("exit_reason") == "Time Exit"],
        "high_score_loss_85plus": [t for t in losses if (t.get("score") or 0) >= 85],
    }
    out = {}
    for label, sub in modes.items():
        if not sub: continue
        out[label] = {
            "count": len(sub),
            "pct_of_losses": round(len(sub) / len(losses), 3),
            "avg_pnl": round(sum(t.get("pnl_pct", 0) for t in sub) / len(sub), 2),
            "by_setup": dict(Counter(t.get("setup", "?") for t in sub)),
        }
    return out


def update_knowledge_base(merged):
    """Akkumuliere ALLE Trade-Daten in einer persistenten Knowledge-Datei."""
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    knowledge = {
        "meta": {
            "updated":      datetime.now().isoformat(),
            "total_trades": len(merged),
            "date_range":   [min(t["date"] for t in merged), max(t["date"] for t in merged)] if merged else None,
            "lifetime_wr":  round(_wr(merged), 3),
            "lifetime_pf":  _pf(merged),
        },
        "tickers":            compute_ticker_stats(merged),
        "setups":             compute_setup_stats(merged),
        "catalysts":          compute_catalyst_effectiveness(merged),
        "score_calibration":  compute_score_calibration(merged),
        "failure_modes":      compute_failure_modes(merged),
    }
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False, default=str)
    return knowledge


# =============================================================
# REPORT GENERATION
# =============================================================
def summarize_period(trades, days):
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    sub = [t for t in trades if t.get("date", "") >= cutoff]
    if not sub: return None
    wins = [t for t in sub if t.get("pnl_pct", 0) > 0]
    losses = [t for t in sub if t.get("pnl_pct", 0) <= 0]
    gw = sum(abs(t.get("pnl_usd", 0)) for t in wins)
    gl = sum(abs(t.get("pnl_usd", 0)) for t in losses)
    return {
        "n": len(sub), "wins": len(wins), "losses": len(losses),
        "wr": len(wins) / len(sub) * 100,
        "avg_w": sum(t.get("pnl_pct", 0) for t in wins) / len(wins) if wins else 0,
        "avg_l": sum(t.get("pnl_pct", 0) for t in losses) / len(losses) if losses else 0,
        "pf": gw / gl if gl > 0 else 999,
        "total_pnl_pct": sum(t.get("pnl_pct", 0) for t in sub),
        "total_pnl_usd": sum(t.get("pnl_usd", 0) for t in sub),
    }


def fetch_news(tickers, limit=3):
    out = {}
    for tk in tickers:
        try:
            with _suppress():
                n = yf.Ticker(tk).news
            if n:
                out[tk] = [{"title": x.get("title"), "publisher": x.get("publisher"),
                            "date": datetime.fromtimestamp(x.get("providerPublishTime", 0)).strftime("%Y-%m-%d")}
                           for x in n[:limit]]
            else:
                out[tk] = []
        except Exception:
            out[tk] = []
    return out


def gen_confirmed_findings(knowledge):
    """Extrahiere bestätigte Fakten aus Knowledge mit Confidence-Filter."""
    findings = {"CONFIRMED": [], "TENTATIVE": [], "HYPOTHESIS": []}

    def bucket(text, conf):
        if conf == "HIGH":   findings["CONFIRMED"].append(text)
        elif conf == "MED":  findings["TENTATIVE"].append(text)
        else:                findings["HYPOTHESIS"].append(text)

    # Setup-Lifetime
    for setup, d in knowledge.get("setups", {}).items():
        bucket(f"**{disp(setup)}**: lifetime WR {d['wr']:.0%} (n={d['n']}, PF {d['pf']})", d["confidence"])

    # Catalyst-Lifts (only meaningful lifts)
    for cat, d in knowledge.get("catalysts", {}).items():
        if abs(d["lift"]) < 0.05:
            continue  # too small to mention
        sign = "+" if d["lift"] > 0 else ""
        bucket(
            f"**{cat}**: {sign}{d['lift']:.0%} WR-Lift "
            f"(with {d['wr_when_present']:.0%} vs without {d['wr_when_absent']:.0%}, "
            f"n_present={d['n_present']})",
            d["confidence"],
        )

    # Setup-Feature-Buckets (lifts >= 10pp)
    for setup, d in knowledge.get("setups", {}).items():
        for feat, fd in d.get("feature_buckets", {}).items():
            if abs(fd.get("lift_vs_setup_baseline", 0)) < 0.10:
                continue
            sign = "+" if fd["lift_vs_setup_baseline"] > 0 else ""
            bucket(
                f"**{disp(setup)} × {feat}**: {sign}{fd['lift_vs_setup_baseline']:.0%} "
                f"vs setup-baseline (WR {fd['wr']:.0%}, n={fd['n']})",
                fd["confidence"],
            )

    # Score-Calibration outliers
    for setup, sb in knowledge.get("score_calibration", {}).items():
        for bucket_name, bd in sb.items():
            if bd["wr"] >= 0.70 and bd["confidence"] in ("HIGH", "MED"):
                bucket(
                    f"**{disp(setup)} score {bucket_name}**: {bd['wr']:.0%} WR (n={bd['n']}) — elite zone",
                    bd["confidence"],
                )

    # Best/Worst Tickers (n>=4)
    tickers = knowledge.get("tickers", {})
    for tk, d in sorted(tickers.items(), key=lambda x: x[1]["wr"], reverse=True)[:5]:
        if d["n"] >= 4 and d["wr"] >= 0.7:
            bucket(f"**{tk}** historisch: {d['wr']:.0%} WR über {d['n']} Trades (avg {d['avg_pnl_pct']:+.2f}%)",
                   d["confidence"])
    for tk, d in sorted(tickers.items(), key=lambda x: x[1]["wr"])[:5]:
        if d["n"] >= 4 and d["wr"] <= 0.3:
            bucket(f"**{tk}** historisch: nur {d['wr']:.0%} WR über {d['n']} Trades — blacklist-Kandidat",
                   d["confidence"])

    return findings


def gen_report(args, knowledge):
    sigs, trades, market = load_data()
    merged = join_signal_to_trade(trades, sigs)

    if args.setup:
        merged_filt = [t for t in merged if t.get("setup") == args.setup]
    elif args.ticker:
        merged_filt = [t for t in merged if t.get("ticker") == args.ticker]
    else:
        merged_filt = merged

    cutoff = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    recent = sorted([t for t in merged_filt if t.get("date", "") >= cutoff],
                    key=lambda x: x["date"], reverse=True)

    L = []
    L.append(f"# ApexLearn Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append("")
    L.append(f"Window: last {args.days} days (since {cutoff})  |  Filter: "
             f"setup={args.setup or 'ALL'}, ticker={args.ticker or 'ALL'}")
    L.append(f"Market: {market.get('mode','?')} — {market.get('summary','')}")
    L.append("")
    meta = knowledge.get("meta", {})
    L.append(f"**Knowledge base** — {meta.get('total_trades',0)} lifetime trades  |  "
             f"WR {meta.get('lifetime_wr', 0):.1%}  |  PF {meta.get('lifetime_pf', '?')}  |  "
             f"Range {meta.get('date_range', ['?','?'])[0]} → {meta.get('date_range', ['?','?'])[1]}")
    L.append("")

    # === 1. Window Snapshot ===
    L.append("## 1. Window Performance")
    L.append("")
    for d, label in [(7, "7d"), (14, "14d"), (30, "30d"), (90, "90d")]:
        s = summarize_period(merged_filt, d)
        if s:
            L.append(f"- **{label}**: n={s['n']} | WR {s['wr']:.1f}% | PF {s['pf']:.2f} | "
                     f"AvgW {s['avg_w']:+.2f}% | AvgL {s['avg_l']:+.2f}% | Total ${s['total_pnl_usd']:+.2f}")
    L.append("")

    # === 2. CONFIRMED FACTS (from knowledge) ===
    findings = gen_confirmed_findings(knowledge)
    L.append("## 2. Confirmed Facts (Knowledge Base)")
    L.append("")
    L.append("**✅ CONFIRMED (n≥30) — vertrauenswürdige Fakten:**")
    if findings["CONFIRMED"]:
        for f in findings["CONFIRMED"]:
            L.append(f"- {f}")
    else:
        L.append("- _(noch keine HIGH-Confidence Findings — Sample-Größen zu klein)_")
    L.append("")
    L.append("**⚠️ TENTATIVE (n=15-30) — Trend, brauche mehr Daten:**")
    if findings["TENTATIVE"]:
        for f in findings["TENTATIVE"]:
            L.append(f"- {f}")
    else:
        L.append("- _(keine)_")
    L.append("")
    L.append("**🔍 HYPOTHESIS (n<15) — nur Auffälligkeit, kein Beweis:**")
    if findings["HYPOTHESIS"]:
        for f in findings["HYPOTHESIS"][:5]:
            L.append(f"- {f}")
        if len(findings["HYPOTHESIS"]) > 5:
            L.append(f"- _(+ {len(findings['HYPOTHESIS'])-5} weitere)_")
    L.append("")

    # === 3. Per-Setup-Window ===
    L.append("## 3. Per-Setup Window-Performance")
    L.append("")
    by_setup_win = defaultdict(list)
    for t in recent: by_setup_win[t.get("setup", "?")].append(t)
    L.append("| Setup | n | WR | PF | AvgWin | AvgLoss |")
    L.append("|---|---|---|---|---|---|")
    for setup, ts in sorted(by_setup_win.items(), key=lambda x: -len(x[1])):
        w = [t for t in ts if t["pnl_pct"] > 0]
        l = [t for t in ts if t["pnl_pct"] <= 0]
        gw = sum(abs(t.get("pnl_usd", 0)) for t in w)
        gl = sum(abs(t.get("pnl_usd", 0)) for t in l)
        pf = gw / gl if gl > 0 else 999
        aw = sum(t["pnl_pct"] for t in w) / len(w) if w else 0
        al = sum(t["pnl_pct"] for t in l) / len(l) if l else 0
        L.append(f"| {disp(setup)} | {len(ts)} | {len(w)/len(ts)*100:.1f}% | {pf:.2f} | {aw:+.2f}% | {al:+.2f}% |")
    L.append("")

    # === 4. Per-Setup-Lifetime ===
    L.append("## 4. Per-Setup Lifetime-Performance (Knowledge)")
    L.append("")
    L.append("| Setup | n | WR | PF | AvgWin | AvgLoss | Confidence |")
    L.append("|---|---|---|---|---|---|---|")
    for setup, d in sorted(knowledge.get("setups", {}).items(), key=lambda x: -x[1]["n"]):
        pf_str = f"{d['pf']:.2f}" if d['pf'] is not None else "—"
        L.append(f"| {disp(setup)} | {d['n']} | {d['wr']:.1%} | {pf_str} | "
                 f"{d['avg_win_pct']:+.2f}% | {d['avg_loss_pct']:+.2f}% | {d['confidence']} |")
    L.append("")

    # === 5. Catalyst Effectiveness ===
    L.append("## 5. Catalyst Effectiveness (Lifetime)")
    L.append("")
    L.append("| Catalyst | n_present | WR with | WR without | Lift | Confidence |")
    L.append("|---|---|---|---|---|---|")
    cats_sorted = sorted(knowledge.get("catalysts", {}).items(),
                         key=lambda x: -abs(x[1]["lift"]))
    for cat, d in cats_sorted:
        sign = "+" if d["lift"] > 0 else ""
        L.append(f"| {cat} | {d['n_present']} | {d['wr_when_present']:.1%} | "
                 f"{d['wr_when_absent']:.1%} | {sign}{d['lift']:.1%} | {d['confidence']} |")
    L.append("")

    # === 6. Score Calibration ===
    L.append("## 6. Score-Gate Calibration (Lifetime — validiert Gate-Setting)")
    L.append("")
    for setup, buckets in knowledge.get("score_calibration", {}).items():
        L.append(f"**{disp(setup)}:**")
        L.append("| Score-Bucket | n | actual WR | conf |")
        L.append("|---|---|---|---|")
        for b, d in sorted(buckets.items()):
            L.append(f"| {b} | {d['n']} | {d['wr']:.1%} | {d['confidence']} |")
        L.append("")

    # === 7. Top + Bottom Tickers (Lifetime) ===
    L.append("## 7. Per-Ticker Heatmap (Lifetime, n≥3)")
    L.append("")
    tickers = {k: v for k, v in knowledge.get("tickers", {}).items() if v["n"] >= 3}
    if tickers:
        top = sorted(tickers.items(), key=lambda x: (x[1]["wr"], x[1]["n"]), reverse=True)[:10]
        bot = sorted(tickers.items(), key=lambda x: (x[1]["wr"], -x[1]["n"]))[:10]
        L.append("**🏆 Top-Performer (sortiert WR):**")
        L.append("")
        L.append("| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |")
        L.append("|---|---|---|---|---|---|---|")
        for tk, d in top:
            setups_disp = ",".join(disp(s) for s in d['setups_seen'])
            L.append(f"| {tk} | {d['n']} | {d['wr']:.0%} | {d['avg_pnl_pct']:+.2f}% | "
                     f"{d['best_pnl']:+.1f}% | {d['worst_pnl']:+.1f}% | {setups_disp} |")
        L.append("")
        L.append("**💀 Worst-Performer (sortiert WR):**")
        L.append("")
        L.append("| Ticker | n | WR | Avg PnL% | Best | Worst | Setups |")
        L.append("|---|---|---|---|---|---|---|")
        for tk, d in bot:
            setups_disp = ",".join(disp(s) for s in d['setups_seen'])
            L.append(f"| {tk} | {d['n']} | {d['wr']:.0%} | {d['avg_pnl_pct']:+.2f}% | "
                     f"{d['best_pnl']:+.1f}% | {d['worst_pnl']:+.1f}% | {setups_disp} |")
        L.append("")

    # === 8. Failure-Modes ===
    L.append("## 8. Failure-Modes (Lifetime)")
    L.append("")
    fm = knowledge.get("failure_modes", {})
    if fm:
        L.append("| Failure Pattern | count | % of losses | avg pnl | by setup |")
        L.append("|---|---|---|---|---|")
        for mode, d in fm.items():
            by_s = ", ".join(f"{disp(k)}:{v}" for k, v in d.get("by_setup", {}).items())
            L.append(f"| {mode} | {d['count']} | {d['pct_of_losses']:.0%} | "
                     f"{d['avg_pnl']:+.2f}% | {by_s} |")
        L.append("")

    # === 9. Window-Trade-Detail (compact) ===
    L.append(f"## 9. Window-Trade-Detail (letzte {len(recent)})")
    L.append("")
    for t in recent[:30]:
        out = "🟢 W" if t["pnl_pct"] > 0 else "🔴 L"
        cats = []
        if t.get("cat_pocket_pivot"): cats.append("PP")
        if t.get("cat_vol_climax"):   cats.append("VC")
        if (t.get("cat_gap_pct") or 0) > 2: cats.append("GAP")
        if t.get("cat_earnings_beat"): cats.append("PEAD")
        if (t.get("cat_short_pct") or 0) >= 15: cats.append("SHORT")
        cat_str = " " + " ".join(cats) if cats else ""
        setup_d = disp(t.get('setup','?'))
        L.append(f"- **{t['date']}** {t['ticker']:6} {setup_d:18} | "
                 f"{out} {t['pnl_pct']:+5.2f}% | {t.get('exit_reason','?')} D+{t.get('exit_day','?')} | "
                 f"score {t.get('score','?')}{cat_str}")
    L.append("")

    # === 10. Open Positions (filtered) ===
    closed_keys = {(t["date"], t["ticker"]) for t in trades}
    today_dt = datetime.now()
    open_sigs = []
    legacy = expired = stale = 0
    for s in sigs:
        if (s["date"], s["ticker"]) in closed_keys:
            continue
        if s.get("setup") not in ACTIVE_SETUPS:
            legacy += 1; continue
        try:
            sig_dt = datetime.strptime(s["date"], "%Y-%m-%d")
        except Exception:
            continue
        days_old = (today_dt - sig_dt).days
        hold = HOLD_DAYS.get(s.get("horizon", ""), 30)
        if days_old > hold + 5:
            expired += 1; continue
        if days_old > MAX_TRIGGER_DAYS + 1:
            stale += 1; continue
        open_sigs.append(s)

    L.append(f"## 10. Offene Positionen ({len(open_sigs)})")
    L.append("")
    if legacy or expired or stale:
        L.append(f"_Filtered: {legacy} legacy, {expired} expired-hold, {stale} stale-no-trigger_")
        L.append("")
    if open_sigs:
        L.append("| Signal | D+ | Ticker | Setup | Entry | Stop | Target | Score |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in sorted(open_sigs, key=lambda x: x["date"], reverse=True):
            try:
                age = (today_dt - datetime.strptime(s["date"], "%Y-%m-%d")).days
            except Exception:
                age = "?"
            L.append(f"| {s['date']} | D+{age} | {s['ticker']} | {disp(s.get('setup','?'))} | "
                     f"${s.get('buy_above','?')} | ${s.get('stop','?')} | ${s.get('target','?')} | "
                     f"{s.get('score','?')} |")
        L.append("")

    # === 11. News (optional) ===
    if args.news:
        losers = [t for t in recent if t["pnl_pct"] <= 0][:8]
        if losers:
            L.append("## 11. News-Kontext für Loser (yfinance)")
            L.append("")
            news = fetch_news([t["ticker"] for t in losers])
            for t in losers:
                items = news.get(t["ticker"], [])
                L.append(f"**{t['ticker']}** ({t['date']}, {t['pnl_pct']:+.1f}%)")
                if items:
                    for n in items:
                        L.append(f"- [{n.get('date','?')}] {n.get('title','')} ({n.get('publisher','')})")
                else:
                    L.append("- (keine News)")
                L.append("")

    # === 12. Window-vs-Lifetime Anomalies ===
    L.append("## 12. Window-vs-Lifetime Drift (worth investigating?)")
    L.append("")
    win_30 = summarize_period(merged_filt, 30)
    life_wr = knowledge.get("meta", {}).get("lifetime_wr", 0) * 100
    if win_30 and abs(win_30["wr"] - life_wr) >= 5:
        direction = "📈 better" if win_30["wr"] > life_wr else "📉 worse"
        L.append(f"- TOTAL WR last-30d ({win_30['wr']:.1f}%) vs lifetime ({life_wr:.1f}%): "
                 f"{direction} by {abs(win_30['wr']-life_wr):.1f}pp")
    for setup, d in by_setup_win.items():
        if len(d) < 5: continue
        win_wr = sum(1 for t in d if t["pnl_pct"]>0) / len(d) * 100
        life_d = knowledge.get("setups", {}).get(setup)
        if not life_d: continue
        life_setup_wr = life_d["wr"] * 100
        if abs(win_wr - life_setup_wr) >= 10:
            direction = "📈 better" if win_wr > life_setup_wr else "📉 worse"
            L.append(f"- {disp(setup)} last-30d {win_wr:.1f}% vs lifetime {life_setup_wr:.1f}%: "
                     f"{direction} by {abs(win_wr-life_setup_wr):.1f}pp")
    L.append("")

    L.append("---")
    L.append(f"_Report-Ende. Knowledge-File: `{KNOWLEDGE_FILE}`. "
             f"Diesen Output an Claude füttern für datengestützte Verbesserungen._")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",   type=int, default=30)
    ap.add_argument("--setup",  type=str, default=None)
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--news",   action="store_true")
    ap.add_argument("--no-update", action="store_true",
                    help="Skip knowledge-base update (read-only report)")
    ap.add_argument("--md",     action="store_true",
                    help="Print to stdout only, no file save")
    args = ap.parse_args()

    # Load + merge
    sigs, trades, market = load_data()
    merged = join_signal_to_trade(trades, sigs)

    # Update knowledge base (unless --no-update)
    if args.no_update and KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    else:
        knowledge = update_knowledge_base(merged)
        print(f"[knowledge updated → {KNOWLEDGE_FILE} | {len(merged)} trades]")

    # Generate report
    report = gen_report(args, knowledge)

    if args.md:
        print(report)
        return

    print(report)

    # Save to reports/ folder
    REPORTS_DIR.mkdir(exist_ok=True)
    latest = REPORTS_DIR / "learn_latest.md"
    dated  = REPORTS_DIR / f"learn_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    try:
        with open(latest, "w", encoding="utf-8") as f:
            f.write(report)
        with open(dated, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[report saved → {dated} + {latest}]")
    except Exception as e:
        print(f"[save failed: {e}]")


if __name__ == "__main__":
    main()
