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
SIGNALS_FILE      = "apex_signals.json"
RESULTS_FILE      = "apex_equity_results.json"       # alle Signale
RESULTS_FILE_TOP2 = "apex_equity_top2.json"          # nur Top-2 Qualitätssignale
CHART_FILE        = "apex_equity_chart.png"
CHART_FILE_TOP2   = "apex_equity_top2_chart.png"
TRADE_SIZE        = 200.0
START_CAPITAL     = 0.0

# Qualitätsfilter — muss identisch zu ApexScan.py sein
TG_MIN_RR     = 2.0
TG_MIN_UPSIDE = 8.0
TG_MIN_SCORE  = 70.0

def is_quality_signal(signal):
    """Returns True if signal would have passed the Telegram quality filter."""
    return (
        float(signal.get("rr", 0))        >= TG_MIN_RR and
        float(signal.get("upside_pct", signal.get("upside", 0))) >= TG_MIN_UPSIDE and
        float(signal.get("score", 0))     >= TG_MIN_SCORE
    )

# Horizon -> maximale Haltezeit in Handelstagen
HORIZON_DAYS = {
    "1-3 weeks":   15,
    "2-6 weeks":   30,
    "3-8 weeks":   40,
    "4-12 weeks":  60,
}
DEFAULT_HOLD = 21


# =============================================================
# HELPERS
# =============================================================
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

    for i in range(min(hold_days, len(data))):
        row  = data.iloc[i]
        try:
            o = float(row["Open"].item()) if hasattr(row["Open"], "item") else float(row["Open"])
            h = float(row["High"].item()) if hasattr(row["High"], "item") else float(row["High"])
            l = float(row["Low"].item())  if hasattr(row["Low"],  "item") else float(row["Low"])
        except Exception:
            continue

        # Step 1: wait for buy_above to be triggered
        if trigger_day is None:
            if h >= entry:
                trigger_day = i
                # Same candle hit stop too -> unreliable gap, skip
                if l <= sl:
                    return None
            else:
                continue  # not triggered yet

        # Step 2: trade is live, track TP/SL
        hit_tp = h >= tp
        hit_sl = l <= sl

        if hit_tp and hit_sl:
            exit_price  = tp if o >= entry else sl
            exit_reason = "Take Profit" if o >= entry else "Stop Loss"
            exit_day    = i + 1
            break
        elif hit_tp:
            exit_price, exit_reason, exit_day = tp, "Take Profit", i + 1
            break
        elif hit_sl:
            exit_price, exit_reason, exit_day = sl, "Stop Loss", i + 1
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
    signals = load_json(SIGNALS_FILE, [])

    if not signals:
        print(f"Keine Signale in {SIGNALS_FILE} gefunden.")
        return

    # Nur aktive Setups tracken
    ACTIVE_SETUPS = {"BREAKOUT", "REVERSAL"}
    signals = [s for s in signals if s.get("setup") in ACTIVE_SETUPS]
    print(f"Signale geladen: {len(signals)} (nur BREAKOUT + REVERSAL)")
    print()

    # ---- Track 1: ALLE Signale ----
    saved_all = load_json(RESULTS_FILE, [])
    all_results, eq_curve, new_all, skipped_all = run_evaluation(signals, saved_all, "ALL")
    save_json(RESULTS_FILE, all_results)
    print_summary(all_results, new_all, skipped_all, "APEX EQUITY — ALLE SIGNALE")
    build_chart(all_results, eq_curve)
    print(f"Gespeichert: {RESULTS_FILE}  |  Chart: {CHART_FILE}")

    print()

    # ---- Track 2: NUR QUALITÄTSSIGNALE (Top-2 Filter) ----
    quality_signals = [s for s in signals if is_quality_signal(s)]
    saved_top2      = load_json(RESULTS_FILE_TOP2, [])

    print(f"Qualitätssignale (RR≥{TG_MIN_RR}, Upside≥{TG_MIN_UPSIDE}%, Score≥{TG_MIN_SCORE}): {len(quality_signals)}")

    if quality_signals:
        top2_results, eq_curve_top2, new_top2, skipped_top2 = run_evaluation(
            quality_signals, saved_top2, "TOP2"
        )
        save_json(RESULTS_FILE_TOP2, top2_results)
        print_summary(top2_results, new_top2, skipped_top2, "APEX EQUITY — QUALITÄTSSIGNALE")

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


def git_push():
    import subprocess
    print("\nPushe auf GitHub...")
    try:
        subprocess.run(
            ["git", "add",
             "apex_equity_results.json", "apex_equity_top2.json", "dashboard.html"],
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