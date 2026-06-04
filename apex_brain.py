"""
apex_brain.py — Obsidian Vault Writer

Liest die Auto-Gen-Files (apex_signals.json, apex_equity_results.json,
knowledge/trade_postmortems.json, apex_market.json) und schreibt strukturierte
Markdown-Notes in einen lokalen Obsidian-Vault. Greift NICHT in Live-Scanner /
Equity-Tracker / Postmortem ein — komplett autark.

USAGE:
    py apex_brain.py                # alle Modi
    py apex_brain.py --signals      # nur Signal-Notes (idempotent: ueberschreibt nicht)
    py apex_brain.py --postmortems  # nur Postmortem-Notes (regeneriert)
    py apex_brain.py --weekly       # nur Weekly-Summary
    py apex_brain.py --market       # nur Marktphasen-Note
    py apex_brain.py --learnings    # Lesson-Tag-Aggregat
    py apex_brain.py --force        # ueberschreibt auch Signal-Notes
    py apex_brain.py --vault PATH   # ueberschreibt Vault-Pfad

VAULT-STRUKTUR (Default ./vault/ relativ zum Script):
    vault/
      trades/        # Signal-Notes (eine pro Signal, optional User-Editierbar)
      postmortems/   # Trade-Postmortems (regeneriert aus JSON)
      weekly/        # Woche-Summary
      market/        # Marktphasen
      learnings/     # Aggregierte Lessons nach lesson_tag
      index.md       # Vault-Root-Index
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAULT = SCRIPT_DIR / "vault"

SIGNALS_FILE = SCRIPT_DIR / "apex_signals.json"
EQUITY_FILE = SCRIPT_DIR / "apex_equity_results.json"
POSTMORTEMS_FILE = SCRIPT_DIR / "knowledge" / "trade_postmortems.json"
MARKET_FILE = SCRIPT_DIR / "apex_market.json"

# ---------------------------------------------------------------------------
# Setup-Meta (mirrors dashboard SETUP_META)
# ---------------------------------------------------------------------------
SETUP_META = {
    "BREAKOUT":       {"label": "BREAKOUT",       "emoji": "🔵"},
    "VCP":            {"label": "VCP Bounceback", "emoji": "🔹"},
    "SHORT_SQUEEZE":  {"label": "Short Squeeze",  "emoji": "🔥"},
    "STAGE_2":        {"label": "Stage 2 Trend",  "emoji": "🚀"},
    "MEAN_REVERSION": {"label": "Mean Reversion", "emoji": "🟢"},
    "REVERSAL":       {"label": "Reversal (legacy)", "emoji": "⚫"},
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"WARN: cannot read {path.name}: {e}")
        return None


def ensure_dirs(vault: Path):
    for sub in ("trades", "postmortems", "weekly", "market", "learnings"):
        (vault / sub).mkdir(parents=True, exist_ok=True)


def write_md(path: Path, content: str, overwrite: bool = True) -> bool:
    """Write markdown file. Returns True if written, False if skipped."""
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def log(msg: str):
    print(f"[Brain] {msg}", flush=True)


def safe_ticker(t: str) -> str:
    """Sanitize ticker for filename use."""
    return str(t).replace("/", "_").replace("\\", "_").replace(":", "_")


def f(val, default=0.0):
    """Coerce None/missing values to a numeric default (for format strings)."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Catalyst rendering
# ---------------------------------------------------------------------------
def catalyst_flags(sig: dict) -> list[str]:
    """Return list of catalyst flag strings present in this signal."""
    flags = []
    if sig.get("cat_pocket_pivot"):
        flags.append("⚡ Pocket Pivot")
    if sig.get("cat_vol_climax"):
        flags.append("📈 Vol Climax")
    if sig.get("cat_vcp_strength", 0) and sig.get("cat_vcp_strength", 0) >= 0.4:
        flags.append("🎯 VCP")
    short_pct = sig.get("cat_short_pct") or 0
    if short_pct >= 15:
        flags.append(f"🔥 Short {short_pct:.1f}%")
    if sig.get("cat_earnings_blackout"):
        flags.append("⚠ Earnings Blackout")
    if sig.get("cat_earnings_beat"):
        flags.append("✅ Earnings Beat")
    upside = sig.get("cat_analyst_upside")
    if upside is not None and upside > 15:
        flags.append(f"🎯 Analyst +{upside:.0f}%")
    gap = sig.get("cat_gap_pct") or 0
    if gap >= 3:
        flags.append(f"⬆ Gap +{gap:.1f}%")
    return flags


# ---------------------------------------------------------------------------
# Signal-Notes (trades/YYYY-MM-DD_TICKER_SETUP.md)
# ---------------------------------------------------------------------------
def write_signal_notes(vault: Path, signals: list, force: bool = False) -> tuple[int, int]:
    """Write a markdown note per signal. Idempotent: skips existing unless force."""
    if not signals:
        log("no signals to process")
        return 0, 0

    written = 0
    skipped = 0
    market = load_json(MARKET_FILE) or {}
    market_mode = market.get("mode", "?")
    market_summary = market.get("summary", "")

    # Find equity closures keyed by (ticker, signal-date) to backfill outcomes
    eq = load_json(EQUITY_FILE) or []
    closed_lookup = {}
    for t in eq:
        key = (t.get("ticker"), t.get("date"))
        closed_lookup[key] = t

    for sig in signals:
        ticker = sig.get("ticker")
        setup = sig.get("setup", "UNKNOWN")
        sigdate = sig.get("date") or datetime.now().date().isoformat()
        if not ticker:
            continue

        filename = f"{sigdate}_{safe_ticker(ticker)}_{setup}.md"
        path = vault / "trades" / filename

        if path.exists() and not force:
            skipped += 1
            continue

        meta = SETUP_META.get(setup, {"label": setup, "emoji": ""})
        flags = catalyst_flags(sig)
        closed = closed_lookup.get((ticker, sigdate))

        # Frontmatter (YAML)
        fm = [
            "---",
            f"date: {sigdate}",
            f"ticker: {ticker}",
            f"setup: {setup}",
            f"sector: {sig.get('sector', 'Unknown')}",
            f"score: {f(sig.get('score')):.1f}",
            f"entry: {f(sig.get('buy_above', sig.get('price'))):.2f}",
            f"stop: {f(sig.get('stop')):.2f}",
            f"target: {f(sig.get('target')):.2f}",
            f"rr: {f(sig.get('rr')):.2f}",
            f"upside_pct: {f(sig.get('upside_pct')):.1f}",
            f"rsi: {f(sig.get('rsi')):.1f}",
            f"relax_level: {int(f(sig.get('relax_level')))}",
            f"cat_pocket_pivot: {bool(sig.get('cat_pocket_pivot'))}",
            f"cat_analyst_upside: {f(sig.get('cat_analyst_upside')):.1f}",
            f"cat_earnings_blackout: {bool(sig.get('cat_earnings_blackout'))}",
            f"closed: {bool(closed)}",
            "tags:",
            f"  - {setup.lower()}",
            f"  - {sig.get('sector', 'unknown').replace(' ', '_').lower()}",
            "---",
            "",
        ]

        # Body
        body = [
            f"# {ticker} — {meta['label']} — {sigdate}",
            "",
            "## Signal",
            f"- **Entry:** ${f(sig.get('buy_above')):.2f}  "
            f"→ **Ziel:** ${f(sig.get('target')):.2f} (+{f(sig.get('upside_pct')):.1f} %)",
            f"- **Stop:** ${f(sig.get('stop')):.2f} (−{f(sig.get('risk_pct')):.1f} %)",
            f"- **RR:** {f(sig.get('rr')):.2f}  |  **RSI:** {f(sig.get('rsi')):.1f}  "
            f"|  **Score:** {f(sig.get('score')):.1f}",
            f"- **Sektor:** {sig.get('sector', 'Unknown')}  "
            f"|  **Horizon:** {sig.get('horizon', '?')}",
            f"- **Vol-Ratio:** {f(sig.get('vol_ratio')):.2f}  "
            f"|  **Perf 20/60/120d:** "
            f"{f(sig.get('perf_20d')):+.1f}% / "
            f"{f(sig.get('perf_60d')):+.1f}% / "
            f"{f(sig.get('perf_120d')):+.1f}%",
            "",
        ]
        if flags:
            body.append("## Catalysts")
            for fl in flags:
                body.append(f"- {fl}")
            body.append("")

        body.extend([
            "## Marktkontext",
            f"- **Regime:** {market_mode}  |  {market_summary}",
            f"- Siehe: [[{datetime.fromisoformat(sigdate).strftime('%Y-%m')}_Marktphase]]",
            "",
        ])

        # Outcome section (filled if closed)
        body.append("## Ergebnis")
        if closed:
            body.extend([
                f"- **Exit:** ${f(closed.get('exit_price')):.2f} "
                f"({closed.get('exit_reason', '?')})",
                f"- **PnL:** {f(closed.get('pnl_pct')):+.2f} % "
                f"(${f(closed.get('pnl_usd')):+.2f})",
                f"- **Hold:** {int(f(closed.get('hold_days')))} Tage",
                f"- **Equity nach Trade:** ${f(closed.get('equity')):.2f}",
            ])
            pmkey = f"{ticker}_{sigdate}"
            body.append(f"- Postmortem: [[{pmkey}]]")
        else:
            body.append("- *Trade noch offen oder nicht getriggert.*")
        body.append("")

        body.extend([
            "## Notizen",
            "<!-- Eigene Notes / Beobachtungen hier ergaenzen. -->",
            "",
        ])

        content = "\n".join(fm + body)
        if write_md(path, content, overwrite=True):
            written += 1

    log(f"signal-notes: {written} written, {skipped} skipped (existing)")
    return written, skipped


# ---------------------------------------------------------------------------
# Postmortem-Notes (postmortems/TICKER_YYYY-MM-DD.md)
# ---------------------------------------------------------------------------
def write_postmortem_notes(vault: Path, pm_data: dict) -> int:
    if not pm_data or "trades" not in pm_data:
        log("no postmortem data")
        return 0
    trades = pm_data["trades"]
    if not isinstance(trades, dict):
        log("postmortem trades wrong type")
        return 0

    written = 0
    for key, pm in trades.items():
        ca = pm.get("claude_analysis", {}) or {}
        if ca.get("status") != "complete":
            continue
        core = pm.get("core", {}) or {}
        sig_meta = pm.get("signal_metadata", {}) or {}
        mkt = pm.get("market_context", {}) or {}
        news = pm.get("news", {}) or {}

        ticker = core.get("ticker")
        sigdate = core.get("signal_date")
        if not (ticker and sigdate):
            continue

        filename = f"{safe_ticker(ticker)}_{sigdate}.md"
        path = vault / "postmortems" / filename

        outcome = core.get("outcome", "?")
        pnl = f(core.get("pnl_pct"))
        lesson_tags = ca.get("lesson_tags", []) or []
        similar = ca.get("similar_trades", []) or []

        fm = [
            "---",
            f"date: {sigdate}",
            f"ticker: {ticker}",
            f"setup: {core.get('setup', '?')}",
            f"sector: {sig_meta.get('sector', '?')}",
            f"outcome: {outcome}",
            f"pnl_pct: {pnl:.2f}",
            f"exit_reason: {core.get('exit_reason', '?')}",
            f"hold_days: {core.get('exit_day', 0)}",
            f"score: {f(sig_meta.get('score')):.1f}",
            f"rr: {f(sig_meta.get('rr')):.2f}",
            f"confidence: {ca.get('confidence', 'unknown')}",
            "tags:",
            f"  - postmortem",
            f"  - {outcome.lower()}",
            f"  - {core.get('setup', 'unknown').lower()}",
        ]
        for t in lesson_tags:
            fm.append(f"  - {t}")
        fm.extend(["---", ""])

        body = [
            f"# Postmortem — {ticker} ({core.get('setup', '?')}) — {sigdate}",
            "",
            f"**Outcome:** {outcome}  |  **PnL:** {pnl:+.2f} %  "
            f"|  **Exit:** {core.get('exit_reason', '?')} "
            f"nach {core.get('exit_day', 0)} Tagen",
            "",
            "## Trade-Eckdaten",
            f"- Entry ${f(core.get('entry')):.2f}  "
            f"→ Exit ${f(core.get('exit_price')):.2f}  "
            f"→ Ziel ${f(core.get('target')):.2f}",
            f"- Stop ${f(core.get('stop')):.2f}  "
            f"|  RR {f(sig_meta.get('rr')):.2f}  "
            f"|  Score {f(sig_meta.get('score')):.1f}",
            f"- Sektor: **{sig_meta.get('sector', '?')}**  "
            f"|  RSI {f(sig_meta.get('rsi')):.1f}  "
            f"|  Vol-Ratio {f(sig_meta.get('vol_ratio')):.2f}",
            "",
            "## Marktkontext",
            f"- SPY: {f(mkt.get('spy_perf_pct')):+.2f}%  "
            f"|  QQQ: {f(mkt.get('qqq_perf_pct')):+.2f}%  "
            f"|  {mkt.get('sector_etf', '?')}: {f(mkt.get('sector_etf_perf_pct')):+.2f}%",
            f"- Sektor-Divergenz: **{mkt.get('sector_divergence', False)}**",
            "",
        ]

        # Failure / Win analysis
        if outcome == "LOSS":
            cause = ca.get("primary_failure_cause", "")
            if cause:
                body.extend(["## Primärursache (Loss)", cause, ""])
        else:
            cause = ca.get("primary_win_cause", "")
            if cause:
                body.extend(["## Primärursache (Win)", cause, ""])

        # Lessons & replicate
        helped = ca.get("what_would_have_helped", []) or []
        if helped:
            body.append("## Was geholfen hätte")
            for x in helped:
                body.append(f"- {x}")
            body.append("")
        replicate = ca.get("what_to_replicate", []) or []
        if replicate:
            body.append("## Was wir replizieren wollen")
            for x in replicate:
                body.append(f"- {x}")
            body.append("")

        # Lesson tags as backlinks
        if lesson_tags:
            body.append("## Lesson-Tags")
            for t in lesson_tags:
                body.append(f"- [[{t}]]")
            body.append("")

        # News / events
        events = news.get("key_events", []) or []
        if events:
            body.append("## Key Events")
            for ev in events:
                body.append(f"- {ev}")
            body.append("")

        web = news.get("web_research", []) or []
        if web:
            body.append("## News-Research")
            for n in web:
                d = n.get("date", "?")
                title = n.get("title", "?")
                src = n.get("src", "?")
                body.append(f"- **{d}** — {title}  *({src})*")
            body.append("")

        # Similar trades
        if similar:
            body.append("## Ähnliche Trades")
            for s in similar:
                body.append(f"- [[{s}]]")
            body.append("")

        # Backlink to signal note
        try:
            sigfile = f"{sigdate}_{safe_ticker(ticker)}_{core.get('setup', '?')}"
            body.append(f"---")
            body.append(f"Siehe Signal: [[{sigfile}]]")
            body.append("")
        except Exception:
            pass

        content = "\n".join(fm + body)
        if write_md(path, content, overwrite=True):
            written += 1

    log(f"postmortem-notes: {written} written")
    return written


# ---------------------------------------------------------------------------
# Weekly-Summary (weekly/YYYY-WWW_Summary.md)
# ---------------------------------------------------------------------------
def write_weekly_summary(vault: Path, equity: list, signals: list) -> int:
    """Write last-complete-week summary based on closed trades."""
    if not equity:
        log("no equity data for weekly")
        return 0

    today = date.today()
    # Compute last Monday → last Sunday window
    # If today is Mon, last week = Mon-7 .. Sun-1
    days_since_mon = today.weekday()  # Mon=0
    last_sun = today - timedelta(days=days_since_mon + 1)
    last_mon = last_sun - timedelta(days=6)

    iso_year, iso_week, _ = last_mon.isocalendar()
    label = f"{iso_year}-W{iso_week:02d}"

    def in_window(d):
        try:
            dt = datetime.fromisoformat(d).date()
            return last_mon <= dt <= last_sun
        except Exception:
            return False

    # Filter closed trades by exit-date if available, else signal-date
    week_trades = []
    for t in equity:
        exit_d = t.get("date")  # signal-date in equity_results
        # No exit_date in equity_results — approximate via date+hold_days
        try:
            exit_d_dt = datetime.fromisoformat(t["date"]).date() + timedelta(days=int(t.get("hold_days", 0)))
            if last_mon <= exit_d_dt <= last_sun:
                week_trades.append(t)
        except Exception:
            continue

    if not week_trades:
        log(f"weekly: no closed trades in window {last_mon}..{last_sun}")
        # Still write an empty summary
        wins = losses = 0
        wr = 0.0
        pnl_total = 0.0
        avg_win = avg_loss = 0.0
    else:
        wins = sum(1 for t in week_trades if f(t.get("pnl_pct")) > 0)
        losses = len(week_trades) - wins
        wr = (wins / len(week_trades)) * 100 if week_trades else 0
        pnl_total = sum(f(t.get("pnl_usd")) for t in week_trades)
        win_pnls = [f(t.get("pnl_pct")) for t in week_trades if f(t.get("pnl_pct")) > 0]
        loss_pnls = [f(t.get("pnl_pct")) for t in week_trades if f(t.get("pnl_pct")) <= 0]
        avg_win = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0
        avg_loss = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0

    # New signals in this week
    new_sigs = [s for s in (signals or []) if in_window(s.get("date", ""))]

    path = vault / "weekly" / f"{label}_Summary.md"
    fm = [
        "---",
        f"week: {label}",
        f"window_start: {last_mon}",
        f"window_end: {last_sun}",
        f"closed_trades: {len(week_trades)}",
        f"wins: {wins}",
        f"losses: {losses}",
        f"win_rate: {wr:.1f}",
        f"pnl_usd_total: {pnl_total:.2f}",
        f"new_signals: {len(new_sigs)}",
        "tags:",
        "  - weekly",
        "---",
        "",
    ]

    body = [
        f"# Woche {label} — {last_mon} bis {last_sun}",
        "",
        "## Performance",
        f"- **Trades geschlossen:** {len(week_trades)}",
        f"- **Win-Rate:** {wr:.1f} %  ({wins}W / {losses}L)",
        f"- **PnL realisiert:** ${pnl_total:+.2f}",
        f"- **Avg Win:** {avg_win:+.2f}%  |  **Avg Loss:** {avg_loss:+.2f}%",
        f"- **Neue Signale:** {len(new_sigs)}",
        "",
    ]

    if week_trades:
        # Sort by PnL desc
        sorted_trades = sorted(week_trades, key=lambda x: f(x.get("pnl_pct")), reverse=True)
        body.append("## Beste Trades")
        for t in sorted_trades[:3]:
            if f(t.get("pnl_pct")) <= 0:
                break
            sigfile = f"{t['date']}_{safe_ticker(t['ticker'])}_{t.get('setup', '?')}"
            body.append(f"- [[{sigfile}|{t['ticker']}]] "
                        f"({t.get('setup', '?')}) {f(t.get('pnl_pct')):+.2f}%")
        body.append("")

        body.append("## Schlechteste Trades")
        for t in sorted(week_trades, key=lambda x: f(x.get("pnl_pct")))[:3]:
            if f(t.get("pnl_pct")) > 0:
                break
            sigfile = f"{t['date']}_{safe_ticker(t['ticker'])}_{t.get('setup', '?')}"
            body.append(f"- [[{sigfile}|{t['ticker']}]] "
                        f"({t.get('setup', '?')}) {f(t.get('pnl_pct')):+.2f}%")
        body.append("")

        # Setup-Verteilung
        setup_count = Counter(t.get("setup", "?") for t in week_trades)
        body.append("## Setup-Verteilung")
        for s, c in setup_count.most_common():
            wins_s = sum(1 for t in week_trades if t.get("setup") == s and f(t.get("pnl_pct")) > 0)
            body.append(f"- **{s}:** {c} Trades ({wins_s}W)")
        body.append("")

    if new_sigs:
        body.append("## Neue Signale dieser Woche")
        for s in new_sigs[:20]:
            sigfile = f"{s['date']}_{safe_ticker(s['ticker'])}_{s.get('setup', '?')}"
            body.append(f"- [[{sigfile}|{s['ticker']}]] "
                        f"({s.get('setup', '?')}) Score {f(s.get('score')):.1f}")
        body.append("")

    body.extend([
        "## Reflexion",
        "<!-- Eigene Lehren / Beobachtungen ergaenzen. -->",
        "",
    ])

    content = "\n".join(fm + body)
    if write_md(path, content, overwrite=True):
        log(f"weekly-summary: {label} written")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Market-Phase Note (market/YYYY-MM_Marktphase.md)
# ---------------------------------------------------------------------------
def write_market_phase(vault: Path) -> int:
    m = load_json(MARKET_FILE) or {}
    if not m:
        return 0
    today = date.today()
    label = today.strftime("%Y-%m")
    path = vault / "market" / f"{label}_Marktphase.md"

    fm = [
        "---",
        f"month: {label}",
        f"mode: {m.get('mode', '?')}",
        f"risk_on: {m.get('risk_on', False)}",
        f"updated: {m.get('updated', '?')}",
        "tags:",
        "  - market",
        "---",
        "",
    ]
    body = [
        f"# Marktphase {label}",
        "",
        f"- **Regime:** {m.get('mode', '?')}",
        f"- **Summary:** {m.get('summary', '')}",
        f"- **Risk-On:** {m.get('risk_on', False)}",
        f"- **Letztes Update:** {m.get('updated', '?')}",
        "",
        "## Notes",
        "<!-- Marktbeobachtungen, Themes, Sector-Rotation hier. -->",
        "",
    ]
    write_md(path, "\n".join(fm + body), overwrite=True)
    log(f"market-phase: {label} written")
    return 1


# ---------------------------------------------------------------------------
# Lesson-Aggregat (learnings/<tag>.md)
# ---------------------------------------------------------------------------
def write_learnings(vault: Path, pm_data: dict) -> int:
    if not pm_data or "trades" not in pm_data:
        return 0
    trades = pm_data["trades"]
    if not isinstance(trades, dict):
        return 0

    # Aggregate: lesson_tag -> list of (ticker, sigdate, outcome, pnl_pct, primary_cause)
    by_tag = defaultdict(list)
    for key, pm in trades.items():
        ca = pm.get("claude_analysis", {}) or {}
        if ca.get("status") != "complete":
            continue
        core = pm.get("core", {}) or {}
        for tag in ca.get("lesson_tags", []) or []:
            by_tag[tag].append({
                "ticker": core.get("ticker"),
                "date": core.get("signal_date"),
                "setup": core.get("setup"),
                "outcome": core.get("outcome"),
                "pnl_pct": f(core.get("pnl_pct")),
                "cause": (ca.get("primary_failure_cause") or ca.get("primary_win_cause") or "")[:200],
            })

    written = 0
    for tag, items in by_tag.items():
        path = vault / "learnings" / f"{tag}.md"
        wins = sum(1 for i in items if i["outcome"] == "WIN")
        losses = sum(1 for i in items if i["outcome"] == "LOSS")
        total = len(items)
        wr = (wins / total * 100) if total else 0

        fm = [
            "---",
            f"lesson_tag: {tag}",
            f"occurrences: {total}",
            f"wins: {wins}",
            f"losses: {losses}",
            f"win_rate: {wr:.1f}",
            "tags:",
            "  - learning",
            f"  - {tag}",
            "---",
            "",
        ]
        body = [
            f"# Lesson: `{tag}`",
            "",
            f"- **Vorkommen:** {total} ({wins}W / {losses}L, WR {wr:.1f} %)",
            "",
            "## Trades mit diesem Muster",
            "",
            "| Datum | Ticker | Setup | Outcome | PnL % |",
            "|---|---|---|---|---|",
        ]
        for i in sorted(items, key=lambda x: x["date"] or "", reverse=True):
            pmkey = f"{safe_ticker(i['ticker'])}_{i['date']}"
            body.append(
                f"| {i['date']} | [[{pmkey}\\|{i['ticker']}]] | "
                f"{i['setup']} | {i['outcome']} | {f(i['pnl_pct']):+.2f} % |"
            )
        body.extend([
            "",
            "## Beobachtungen",
            "<!-- Pattern-Beschreibung, Hypothesen, Gegenmittel. -->",
            "",
        ])
        write_md(path, "\n".join(fm + body), overwrite=True)
        written += 1

    log(f"learnings: {written} lesson-tags aggregated")
    return written


# ---------------------------------------------------------------------------
# Root index
# ---------------------------------------------------------------------------
def write_root_index(vault: Path):
    path = vault / "index.md"
    fm = [
        "---",
        f"updated: {datetime.now().isoformat(timespec='seconds')}",
        "tags:",
        "  - index",
        "---",
        "",
    ]
    body = [
        "# ApexScan Vault",
        "",
        "Automatisch generiert von `apex_brain.py`. Nicht manuell editieren.",
        "",
        "## Bereiche",
        "- 📡 [Trade-Signale](trades/)",
        "- 📋 [Postmortems](postmortems/)",
        "- 📈 [Weekly Summaries](weekly/)",
        "- 🌐 [Marktphasen](market/)",
        "- 💡 [Learnings](learnings/)",
        "",
        "## Aktive Setups",
    ]
    for k, v in SETUP_META.items():
        body.append(f"- {v['emoji']} **{v['label']}**")
    body.extend([
        "",
        f"_Letztes Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ])
    write_md(path, "\n".join(fm + body), overwrite=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=str(DEFAULT_VAULT),
                    help="Vault-Pfad (default ./vault relativ zum Script)")
    ap.add_argument("--signals", action="store_true", help="nur Signal-Notes")
    ap.add_argument("--postmortems", action="store_true", help="nur Postmortems")
    ap.add_argument("--weekly", action="store_true", help="nur Weekly")
    ap.add_argument("--market", action="store_true", help="nur Marktphase")
    ap.add_argument("--learnings", action="store_true", help="nur Lesson-Aggregat")
    ap.add_argument("--force", action="store_true",
                    help="ueberschreibt auch existierende Signal-Notes")
    args = ap.parse_args()

    vault = Path(args.vault).resolve()
    ensure_dirs(vault)
    log(f"vault: {vault}")

    # Wenn kein Modus gewaehlt: alle laufen lassen
    run_all = not any([args.signals, args.postmortems, args.weekly,
                       args.market, args.learnings])

    if args.signals or run_all:
        sigs = load_json(SIGNALS_FILE) or []
        write_signal_notes(vault, sigs, force=args.force)

    if args.postmortems or run_all:
        pm = load_json(POSTMORTEMS_FILE) or {}
        write_postmortem_notes(vault, pm)

    if args.weekly or run_all:
        eq = load_json(EQUITY_FILE) or []
        sigs = load_json(SIGNALS_FILE) or []
        write_weekly_summary(vault, eq, sigs)

    if args.market or run_all:
        write_market_phase(vault)

    if args.learnings or run_all:
        pm = load_json(POSTMORTEMS_FILE) or {}
        write_learnings(vault, pm)

    # Always refresh root index
    write_root_index(vault)
    log("done.")


if __name__ == "__main__":
    main()
