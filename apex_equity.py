"""
ApexScan – Equity Tracker
=========================
Workflow:
  1. ApexScan.py läuft -> speichert Signale in apex_signals.json
  2. Dieses Script prüft ob Target/Stop erreicht wurde
  3. Ergebnisse werden in apex_equity_results.json gespeichert
  4. Equity-Kurve + Stats-Chart werden als PNG gespeichert

Aufruf: py apex_equity.py
"""

import json
import warnings
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# =============================================================
# CONFIG
# =============================================================
SIGNALS_FILE       = "apex_signals.json"
RESULTS_FILE       = "apex_equity_results.json"       # alle geschlossenen Trades
RESULTS_FILE_TOP2  = "apex_equity_top2.json"          # nur Top-2 Qualitätssignale
OPEN_POSITIONS_FILE = "apex_open_positions.json"      # aktive: pending/open/expired
CHART_FILE        = "apex_equity_chart.png"
CHART_FILE_TOP2   = "apex_equity_top2_chart.png"
TRADE_SIZE        = 200.0
START_CAPITAL     = 0.0

# Qualitätsfilter — muss identisch zu ApexScan.py sein
# TG-Gate constants — MIRROR ApexScan.py (post-2026-05-22 changes)
TG_MIN_RR     = 1.5    # was 2.0; live lowered after backtest showed >=2.0 had no WR edge
TG_MIN_UPSIDE = 8.0
TG_MIN_SCORE  = {
    "BREAKOUT":      70,
    "VCP":           70,
    "SHORT_SQUEEZE": 65,
    "STAGE_2":       60,
    "MEAN_REVERSION": 70,
}
TG_MIN_SCORE_DEFAULT = 70


def telegram_pushed_signals(signals):
    """Replicate ApexScan's Telegram-push: top-2 by score per scan-date among TG-quality.
    Mirrors ApexScan.py post-2026-05-22 gate (score-based, per-setup TG_MIN_SCORE).
    Used to define Track-2 = the ACTUAL Telegram universe (not the broad quality threshold)."""
    from collections import defaultdict
    by_date = defaultdict(list)
    for s in signals:
        d = s.get("date")
        if not d:
            continue
        min_score = TG_MIN_SCORE.get(s.get("setup", ""), TG_MIN_SCORE_DEFAULT)
        try:
            ok = (float(s.get("rr", 0))                                   >= TG_MIN_RR
                  and float(s.get("upside_pct", s.get("upside", 0)))      >= TG_MIN_UPSIDE
                  and float(s.get("score", 0))                            >= min_score)
        except (TypeError, ValueError):
            ok = False
        if ok:
            by_date[d].append(s)
    pushed = []
    for d, sigs_d in by_date.items():
        sigs_d.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        pushed.extend(sigs_d[:2])
    return pushed


def is_quality_signal(signal):
    """Returns True if signal would have passed the Telegram quality filter."""
    min_score = TG_MIN_SCORE.get(signal.get("setup", ""), TG_MIN_SCORE_DEFAULT)
    try:
        return (
            float(signal.get("rr", 0))                                   >= TG_MIN_RR and
            float(signal.get("upside_pct", signal.get("upside", 0)))     >= TG_MIN_UPSIDE and
            float(signal.get("score", 0))                                >= min_score
        )
    except (TypeError, ValueError):
        return False

# Horizon -> maximale Haltezeit in Handelstagen
HORIZON_DAYS = {
    "1-3 weeks":   15,   # BREAKOUT
    "2-4 weeks":   20,   # SHORT_SQUEEZE
    "2-6 weeks":   30,   # legacy
    "3-8 weeks":   40,   # legacy REVERSAL (disabled in Phase G)
    "4-8 weeks":   40,   # VCP
    "4-12 weeks":  60,   # legacy
    "8-16 weeks":  80,   # STAGE_2
}
DEFAULT_HOLD = 21

# Maximum days to wait for buy_above trigger before signal expires.
# Mirrors apex_backtest_v2.py's MAX_TRIGGER_DAYS so live tracking matches
# backtest assumptions (61.8% BO WR was measured with 3-day cap).
MAX_TRIGGER_DAYS = 3


# =============================================================
# HELPERS
# =============================================================
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def maybe_refresh_market_regime():
    """Backup-Logik: wenn apex_market.json.updated > 18h alt (= Scanner ist ausgefallen
    z.B. Donnerstag/Freitag durch GH-Cron-Drosselung), berechnet Equity das Regime
    selbst via ApexScan-Funktion und schreibt mode/risk_on/summary/updated neu.
    Normal-Betrieb (Scanner laeuft taeglich 20:42 UTC): nicht-stale, kein Update."""
    from datetime import datetime, timedelta
    try:
        with open("apex_market.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        updated_str = m.get("updated", "1970-01-01 00:00")
        last = datetime.strptime(updated_str, "%Y-%m-%d %H:%M")
        if datetime.now() - last < timedelta(hours=18):
            return False  # noch frisch, kein Eingriff
    except Exception as e:
        print(f"  market check fail: {e} -> refresh trotzdem")

    print("  Market regime stale (>18h) -> Equity-Backup berechnet Regime...")
    try:
        import sys
        sys.path.insert(0, ".")
        from ApexScan import get_market_regime, save_market_regime
        regime = get_market_regime()
        save_market_regime(regime)
        print(f"  Equity-Backup: market_regime aktualisiert -> {regime.get('mode')}")
        return True
    except Exception as e:
        print(f"  market backup compute fail: {e}")
        return False


def save_json(path, data):
    # allow_nan=False zwingt valid JSON (kein NaN/Infinity) — Browser-fetch sicher.
    # Bei Erfolg wird normal geschrieben; bei NaN-Fehler vorher per _clean_nan absichern.
    with open(path, "w", encoding="utf-8") as f:
        try:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        except ValueError:
            # Fallback: NaN-Inhalt cleanen
            import math as _math
            def _clean(o):
                if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
                if isinstance(o, list): return [_clean(x) for x in o]
                if isinstance(o, float) and (_math.isnan(o) or _math.isinf(o)): return None
                return o
            f.seek(0); f.truncate()
            json.dump(_clean(data), f, indent=2, ensure_ascii=False, allow_nan=False)


def horizon_to_days(horizon: str) -> int:
    for key, days in HORIZON_DAYS.items():
        if key in horizon:
            return days
    return DEFAULT_HOLD


def already_saved(signal, saved):
    return any(
        r["date"] == signal["date"] and r["ticker"] == signal["ticker"]
        for r in saved
    )


# =============================================================
# TRADE EVALUATION
# =============================================================
def evaluate_trade(signal, today):
    """
    Simuliert den Trade ab dem Tag nach dem Signal.
    Gibt dict mit Ergebnis zurück oder None wenn noch offen.
    """
    ticker     = signal["ticker"]
    entry      = float(signal["buy_above"])   # Trigger-Preis als Entry
    tp         = float(signal["target"])
    sl         = float(signal["stop"])
    horizon    = signal.get("horizon", "2-6 weeks")
    hold_days  = horizon_to_days(horizon)
    sig_date   = datetime.strptime(signal["date"], "%Y-%m-%d")
    end_date   = sig_date + timedelta(days=hold_days + 5)   # etwas Puffer

    try:
        data = yf.download(
            ticker,
            start=(sig_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            end=min(end_date, datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None

    if data is None or data.empty:
        return None

    # Flatten MultiIndex falls vorhanden
    if isinstance(data.columns, tuple) or hasattr(data.columns, "levels"):
        try:
            data.columns = data.columns.get_level_values(0)
        except Exception:
            pass

    needed = ["Open", "High", "Low", "Close"]
    if any(c not in data.columns for c in needed):
        return None
    data = data[needed].dropna()
    if data.empty:
        return None

    exit_price    = None
    exit_reason   = None
    exit_day      = None
    trigger_day   = None
    time_exit_date = (sig_date + timedelta(days=hold_days)).date()
    # Phase F.3 REVERSAL exit management state
    setup_type = signal.get("setup", "BREAKOUT")
    rev_trailing_active = False
    rev_dynamic_sl = sl

    for i in range(min(hold_days, len(data))):
        row  = data.iloc[i]
        try:
            o = float(row["Open"].item()) if hasattr(row["Open"], "item") else float(row["Open"])
            h = float(row["High"].item()) if hasattr(row["High"], "item") else float(row["High"])
            l = float(row["Low"].item())  if hasattr(row["Low"],  "item") else float(row["Low"])
            c = float(row["Close"].item()) if hasattr(row["Close"], "item") else float(row["Close"])
        except Exception:
            continue

        if trigger_day is None:
            if i >= MAX_TRIGGER_DAYS:
                return None
            if h >= entry:
                trigger_day = i
                if l <= sl:
                    return None
            else:
                continue

        days_in_trade = i - trigger_day

        # --- Phase F.3: REVERSAL Exit Management ---
        if setup_type == "REVERSAL":
            # Rule 1: Trailing-Stop ab Day 5 — bei +5% intraday wird Stop auf Breakeven gezogen
            if not rev_trailing_active and days_in_trade >= 5:
                if h >= entry * 1.05:
                    rev_dynamic_sl = max(rev_dynamic_sl, entry)
                    rev_trailing_active = True
            # Rule 2: Time-Stop bei Day 14 wenn keine Progress
            if days_in_trade >= 14:
                pnl_now = (c / entry - 1) * 100
                if pnl_now <= 0:
                    exit_price, exit_reason, exit_day = c, "Time Exit (REV-cut)", i + 1
                    break

        active_sl = rev_dynamic_sl if setup_type == "REVERSAL" else sl
        hit_tp = h >= tp
        hit_sl = l <= active_sl

        if hit_tp and hit_sl:
            exit_price  = tp if o >= entry else active_sl
            exit_reason = "Take Profit" if o >= entry else ("Trailing Stop" if rev_trailing_active else "Stop Loss")
            exit_day    = i + 1
            break
        elif hit_tp:
            exit_price, exit_reason, exit_day = tp, "Take Profit", i + 1
            break
        elif hit_sl:
            reason = "Trailing Stop" if rev_trailing_active else "Stop Loss"
            exit_price, exit_reason, exit_day = active_sl, reason, i + 1
            break

    # Trigger nie erreicht -> kein Trade
    if trigger_day is None:
        return None

    # Noch offen?
    if exit_price is None:
        if today < time_exit_date:
            return None   # Trade läuft noch
        # Time Exit: letzter verfügbarer Close innerhalb der Haltedauer
        # Handle timezone-aware index
        if data.index.tz is not None:
            cutoff = datetime.combine(time_exit_date, datetime.min.time()).replace(tzinfo=data.index.tz)
        else:
            cutoff = datetime.combine(time_exit_date, datetime.min.time())
        mask = data.index.normalize() <= cutoff
        subset = data[mask] if not data.empty else data
        if subset.empty:
            subset = data
        last_close = subset["Close"].iloc[-1]
        exit_price  = float(last_close.item()) if hasattr(last_close, "item") else float(last_close)
        exit_reason = "Time Exit"
        exit_day    = hold_days

    pnl_pct = (exit_price - entry) / entry * 100
    pnl_usd = TRADE_SIZE * pnl_pct / 100

    # Phase F.3 trailing-stop bei breakeven erzeugt regelmäßig 0%-Trades
    # (REVERSAL legacy). Diese polluten WR-Calculation (zählen als Loss).
    # Lösung: 0%-Trades nicht als Trade speichern — Position ist neutral exited.
    if abs(pnl_pct) < 0.01:
        return None

    return {
        "date":        signal["date"],
        "ticker":      ticker,
        "setup":       signal.get("setup", ""),
        "sector":      signal.get("sector", "Unknown"),
        "entry":       round(entry, 2),
        "stop":        round(sl, 2),
        "target":      round(tp, 2),
        "exit_price":  round(exit_price, 2),
        "exit_reason": exit_reason,
        "exit_day":    exit_day,
        "hold_days":   hold_days,
        "pnl_pct":     round(pnl_pct, 2),
        "pnl_usd":     round(pnl_usd, 2),
        "rr":          signal.get("rr", 0),
        "score":       signal.get("score", 0),
    }


# =============================================================
# CHART
# =============================================================
def build_chart(all_results, equity_curve):
    if len(all_results) < 1:
        print("Keine Ergebnisse für Chart.")
        return

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    text_color  = "#e0e0e0"
    grid_color  = "#2a2a3a"
    green       = "#26a65b"
    red         = "#e74c3c"
    blue        = "#4a9eff"

    wins  = [r for r in all_results if r["pnl_pct"] > 0]
    loses = [r for r in all_results if r["pnl_pct"] <= 0]
    win_rate = len(wins) / len(all_results) * 100 if all_results else 0
    avg_win  = sum(r["pnl_pct"] for r in wins)  / len(wins)  if wins  else 0
    avg_loss = sum(r["pnl_pct"] for r in loses) / len(loses) if loses else 0

    # --- 1. Equity Curve ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0f1117")
    ax1.plot(equity_curve, color=blue, linewidth=2, marker="o", markersize=4)
    ax1.fill_between(range(len(equity_curve)), equity_curve, alpha=0.15, color=blue)
    ax1.axhline(0, color=grid_color, linewidth=0.8)
    ax1.set_title("ApexScan – Equity Curve", color=text_color, fontsize=13, pad=10)
    ax1.set_xlabel("Closed Trades", color=text_color, fontsize=10)
    ax1.set_ylabel("Equity ($)", color=text_color, fontsize=10)
    ax1.tick_params(colors=text_color)
    ax1.grid(True, color=grid_color, linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(grid_color)

    final_eq = equity_curve[-1] if equity_curve else 0
    color_eq  = green if final_eq >= 0 else red
    ax1.annotate(
        f"  Final: ${final_eq:.2f}",
        xy=(len(equity_curve) - 1, final_eq),
        color=color_eq, fontsize=10, fontweight="bold"
    )

    # --- 2. PnL per Trade ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0f1117")
    pnls   = [r["pnl_pct"] for r in all_results]
    colors = [green if p > 0 else red for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, width=0.7)
    ax2.axhline(0, color=text_color, linewidth=0.6)
    ax2.set_title("PnL per Trade (%)", color=text_color, fontsize=11)
    ax2.set_xlabel("Trade #", color=text_color, fontsize=9)
    ax2.tick_params(colors=text_color)
    ax2.grid(True, axis="y", color=grid_color, linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor(grid_color)

    # --- 3. Stats ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0f1117")
    ax3.axis("off")
    tp_count = sum(1 for r in all_results if r["exit_reason"] == "Take Profit")
    sl_count = sum(1 for r in all_results if r["exit_reason"] == "Stop Loss")
    te_count = sum(1 for r in all_results if r["exit_reason"] == "Time Exit")
    total_pnl = sum(r["pnl_usd"] for r in all_results)

    stats = [
        ("Trades total",    str(len(all_results))),
        ("Win rate",        f"{win_rate:.1f}%"),
        ("Avg win",         f"+{avg_win:.2f}%"),
        ("Avg loss",        f"{avg_loss:.2f}%"),
        ("Take Profits",    str(tp_count)),
        ("Stop Losses",     str(sl_count)),
        ("Time Exits",      str(te_count)),
        ("Total PnL",       f"${total_pnl:.2f}"),
    ]

    y = 0.95
    ax3.text(0.5, 1.02, "Statistics", color=text_color, fontsize=11,
             ha="center", va="top", transform=ax3.transAxes, fontweight="bold")
    for label, value in stats:
        col = green if "+" in value or (value.startswith("$") and total_pnl >= 0 and label == "Total PnL") else (
              red if value.startswith("-") or (value.startswith("$") and total_pnl < 0) else text_color)
        ax3.text(0.05, y, label,  color="#888888",  fontsize=10, transform=ax3.transAxes)
        ax3.text(0.65, y, value,  color=col,        fontsize=10, transform=ax3.transAxes, fontweight="bold")
        y -= 0.115

    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart gespeichert: {CHART_FILE}")


# =============================================================
# CHART HELPER (accepts custom filename)
# =============================================================
def build_chart_to(all_results, equity_curve, chart_file):
    """Wrapper to build chart to a specific file."""
    global CHART_FILE
    _orig = CHART_FILE
    CHART_FILE = chart_file
    build_chart(all_results, equity_curve)
    CHART_FILE = _orig


# =============================================================
# SHARED EVALUATION LOGIC
# =============================================================
def run_evaluation(signals, saved, label="ALL"):
    """Evaluate a list of signals and return (all_results, equity_curve)."""
    today       = datetime.now(timezone.utc).date()
    new_results = []
    skipped     = 0

    for sig in signals:
        if already_saved(sig, saved):
            skipped += 1
            continue
        result = evaluate_trade(sig, today)
        if result is None:
            continue
        new_results.append(result)
        print(f"  [{label}] {sig['ticker']} | {result['exit_reason']:11} | {result['pnl_pct']:>+6.2f}%")

    all_results  = saved + new_results
    equity       = START_CAPITAL
    equity_curve = [equity]
    for r in all_results:
        equity += r["pnl_usd"]
        r["equity"] = round(equity, 2)
        equity_curve.append(equity)

    return all_results, equity_curve, new_results, skipped


def print_summary(all_results, new_results, skipped, label):
    wins     = [r for r in all_results if r["pnl_pct"] > 0]
    losses   = [r for r in all_results if r["pnl_pct"] <= 0]
    win_rate = len(wins) / len(all_results) * 100 if all_results else 0
    equity   = all_results[-1]["equity"] if all_results else 0

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Trades total : {len(all_results)}  (neu: {len(new_results)}, übersprungen: {skipped})")
    print(f"  Win rate     : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Final equity : ${equity:.2f}")
    print(f"{'='*55}")


# =============================================================
# MAIN
# =============================================================
def main():
    import subprocess, os
    if os.path.exists(".git"):
        try:
            subprocess.run(["git", "pull", "origin", "master", "--rebase"],
                           capture_output=True, check=True)
            print("Git: Synced with GitHub.")
        except subprocess.CalledProcessError:
            print("Git: Pull failed – continuing anyway.")

    # Market-Regime-Backup: wenn Scanner Donnerstag/Freitag ausgefallen ist,
    # uebernimmt Equity das Regime-Berechnen
    maybe_refresh_market_regime()

    signals = load_json(SIGNALS_FILE, [])

    if not signals:
        print(f"Keine Signale in {SIGNALS_FILE} gefunden.")
        return

    # Nur aktive Setups tracken
    # Phase H (2026-05-22): track ALL active Phase G/H setups. Was {BREAKOUT, REVERSAL} —
    # a stale filter that silently blocked STAGE_2/VCP/SQUEEZE/MR closed-trade data from
    # accumulating. REVERSAL kept so legacy open positions still close out cleanly.
    ACTIVE_SETUPS = {"BREAKOUT", "VCP", "SHORT_SQUEEZE", "STAGE_2", "MEAN_REVERSION", "REVERSAL"}
    signals = [s for s in signals if s.get("setup") in ACTIVE_SETUPS]
    print(f"Signale geladen: {len(signals)} (Setups: {', '.join(sorted(ACTIVE_SETUPS))})")
    print()

    # ---- Track 1: ALLE Signale ----
    saved_all = load_json(RESULTS_FILE, [])
    all_results, eq_curve, new_all, skipped_all = run_evaluation(signals, saved_all, "ALL")
    save_json(RESULTS_FILE, all_results)
    print_summary(all_results, new_all, skipped_all, "APEX EQUITY — ALLE SIGNALE")
    build_chart(all_results, eq_curve)
    print(f"Gespeichert: {RESULTS_FILE}  |  Chart: {CHART_FILE}")

    print()

    # ---- Track 2: TELEGRAM-PUSHED (replicates live: top-2 by score per scan-date among TG-quality) ----
    # Was: broad is_quality_signal filter (RR/upside/score threshold). That over-counted vs.
    # what TG actually pushed (which is top-2/day). Now: faithful replication.
    quality_signals  = telegram_pushed_signals(signals)
    quality_keys     = {(s["ticker"], s["date"]) for s in quality_signals}
    # Prune stale outcomes (old broad-quality signals no longer in TG-pushed set)
    saved_top2_raw   = load_json(RESULTS_FILE_TOP2, [])
    saved_top2       = [t for t in saved_top2_raw if (t["ticker"], t["date"]) in quality_keys]

    print(f"Telegram-Pushed (Top-2/Tag, TG-quality): {len(quality_signals)}")

    if quality_signals:
        top2_results, eq_curve_top2, new_top2, skipped_top2 = run_evaluation(
            quality_signals, saved_top2, "TOP2"
        )
        save_json(RESULTS_FILE_TOP2, top2_results)
        print_summary(top2_results, new_top2, skipped_top2, "APEX EQUITY — TELEGRAM-PUSHED")

        # Build top2 chart
        build_chart_to(top2_results, eq_curve_top2, CHART_FILE_TOP2)
        print(f"Gespeichert: {RESULTS_FILE_TOP2}  |  Chart: {CHART_FILE_TOP2}")
    else:
        print("  Keine Qualitätssignale zum Auswerten.")

    print()
    print("Letzte 10 Trades (alle):")
    for r in all_results[-10:]:
        print(
            f"  {r['date']} | {r['ticker']:6} | {r['setup']:10} | "
            f"{r['pnl_pct']:>+6.2f}% | ${r['pnl_usd']:>+7.2f} | "
            f"{r['exit_reason']:11} D+{r['exit_day']} | "
            f"Equity: ${r.get('equity', 0):.2f}"
        )

    # ---- Track 3: Aktive Pending/Open/Expired Snapshot ----
    # Schreibt Live-Status fuer JEDES nicht-geschlossene Signal in
    # apex_open_positions.json. Dashboard nutzt das als Single-Source.
    today = datetime.now().date()
    open_positions = compute_open_positions(signals, all_results, today)
    # NaN-Schutz: Browser-JSON-Parser akzeptiert kein NaN. Ersetze durch None.
    import math as _math
    def _clean_nan(obj):
        if isinstance(obj, dict):
            return {k: _clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_nan(x) for x in obj]
        if isinstance(obj, float) and _math.isnan(obj):
            return None
        return obj
    open_positions = _clean_nan(open_positions)
    save_json(OPEN_POSITIONS_FILE, open_positions)
    pending_n = sum(1 for p in open_positions if p["status"] == "pending")
    open_n    = sum(1 for p in open_positions if p["status"] == "open")
    expired_n = sum(1 for p in open_positions if p["status"] == "expired")
    print(f"Active-Snapshot: {pending_n} pending, {open_n} open, {expired_n} expired "
          f"-> {OPEN_POSITIONS_FILE}")


def compute_open_positions(signals, closed_results, today):
    """Berechnet aktiven Status (pending/open/expired) für jedes nicht-geschlossene Signal.
    Schreibt apex_open_positions.json. Dashboard nutzt das als Single-Source-of-Truth.

    Lifecycle:
      - Signal frisch (age <= MAX_TRIGGER_DAYS) und High < buy_above   -> pending
      - High >= buy_above irgendwann in den ersten 3d                  -> open (mit PnL)
      - Age > MAX_TRIGGER_DAYS und nie getriggert                      -> expired
      - Getriggert und TP/SL/Time-Exit erreicht                        -> nicht hier (in closed_results)
    """
    closed_keys = {(t["ticker"], t["date"]) for t in closed_results}
    # Cutoff: letzte 60 Tage (max Hold ueber alle Setups)
    cutoff = today - timedelta(days=60)
    candidates = []
    for s in signals:
        if (s["ticker"], s.get("date", "")) in closed_keys:
            continue
        try:
            sd = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if sd < cutoff:
            continue
        candidates.append(s)
    if not candidates:
        return []

    # Bulk-Download (60d Daily) fuer alle aktiven Tickers — eine yf-call statt N
    tickers = sorted({s["ticker"] for s in candidates})
    print(f"compute_open: {len(candidates)} candidates, batch-download {len(tickers)} tickers...")
    try:
        bulk = yf.download(tickers, period="60d", interval="1d",
                           auto_adjust=True, progress=False, threads=True,
                           group_by="ticker" if len(tickers) > 1 else None)
    except Exception as e:
        print(f"compute_open: bulk download failed {e}")
        return []

    def extract(t):
        try:
            if len(tickers) == 1:
                df = bulk
            elif hasattr(bulk.columns, "levels"):
                df = bulk[t] if t in bulk.columns.get_level_values(0) else None
            else:
                df = None
            if df is None or df.empty:
                return None
            # Flatten MultiIndex if any
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            return None

    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    result = []
    for s in candidates:
        df = extract(s["ticker"])
        if df is None:
            continue
        sd = datetime.strptime(s["date"], "%Y-%m-%d").date()
        age_days = (today - sd).days
        entry = float(s["buy_above"])
        # Nur Bars NACH dem Signal-Tag
        try:
            post = df[df.index.date > sd]
        except Exception:
            continue

        base = {
            "ticker":   s["ticker"],
            "date":     s["date"],
            "setup":    s["setup"],
            "sector":   s.get("sector", ""),
            "entry":    round(entry, 2),
            "stop":     round(float(s["stop"]), 2),
            "target":   round(float(s["target"]), 2),
            "score":    round(float(s.get("score", 0)), 1),
            "rr":       round(float(s.get("rr", 0)), 2),
            "upside_pct": round(float(s.get("upside_pct", 0)), 1),
            "days_since_signal": age_days,
            "updated_at": now_iso,
        }

        if len(post) == 0:
            # Noch kein Bar nach dem Signal (Signal heute, vor US-Open)
            result.append({**base, "status": "pending", "trigger_day": None,
                           "trigger_date": None, "current_price": None,
                           "current_high": None, "pnl_pct_unrealized": None})
            continue

        # Trigger suchen — bis MAX_TRIGGER_DAYS oder Ende der post-Bars
        trigger_idx = None
        for i in range(min(MAX_TRIGGER_DAYS, len(post))):
            try:
                hi = post["High"].iloc[i]
                hi = float(hi.item()) if hasattr(hi, "item") else float(hi)
            except Exception:
                continue
            if hi >= entry:
                trigger_idx = i
                break

        try:
            import math
            last_close_raw = post["Close"].iloc[-1]
            last_close = float(last_close_raw.item()) if hasattr(last_close_raw, "item") else float(last_close_raw)
            if math.isnan(last_close):
                last_close = None
        except Exception:
            last_close = None

        if trigger_idx is None:
            status = "pending" if age_days <= MAX_TRIGGER_DAYS else "expired"
            result.append({**base, "status": status, "trigger_day": None,
                           "trigger_date": None, "current_price": last_close,
                           "current_high": None, "pnl_pct_unrealized": None})
            continue

        # Triggered
        try:
            trigger_date = post.index[trigger_idx].date().isoformat()
        except Exception:
            trigger_date = None
        trigger_day_num = trigger_idx + 1   # D+1, D+2, ...

        # High seit Trigger (fuer Trail-Anzeige)
        try:
            import math
            highs_since = post["High"].iloc[trigger_idx:]
            current_high_val = float(highs_since.max().item()) if hasattr(highs_since.max(), "item") else float(highs_since.max())
            if math.isnan(current_high_val):
                current_high_val = None
        except Exception:
            current_high_val = None

        pnl_pct = ((last_close - entry) / entry * 100) if last_close is not None else None
        result.append({
            **base,
            "status": "open",
            "trigger_day":  trigger_day_num,
            "trigger_date": trigger_date,
            "current_price": round(last_close, 2) if last_close is not None else None,
            "current_high":  round(current_high_val, 2) if current_high_val is not None else None,
            "pnl_pct_unrealized": round(pnl_pct, 2) if pnl_pct is not None else None,
        })
    return result


def git_push():
    import subprocess, json as _json, os
    # Always write equity timestamp so dashboard shows last run time
    try:
        import pytz
        tz = pytz.timezone("Europe/Berlin")
        mf = "apex_market.json"
        market = {}
        if os.path.exists(mf):
            with open(mf, "r", encoding="utf-8") as f:
                market = _json.load(f)
        market["equity_updated"] = datetime.now(tz).strftime("%Y-%m-%d %H:%M")
        with open(mf, "w", encoding="utf-8") as f:
            _json.dump(market, f, indent=2)
    except Exception:
        pass

    print("\nPushe auf GitHub...")
    try:
        subprocess.run(
            ["git", "add",
             "apex_equity_results.json", "apex_equity_top2.json", "apex_market.json", "dashboard.html"],
            check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"Equity update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            capture_output=True
        )
        if result.returncode == 0:
            subprocess.run(["git", "push", "origin", "master"], check=True, capture_output=True)
            print("GitHub: Push erfolgreich.")
        else:
            print("GitHub: Keine Änderungen – kein Push nötig.")
    except subprocess.CalledProcessError as e:
        print(f"GitHub: Push fehlgeschlagen – {e}")


if __name__ == "__main__":
    main()
    git_push()