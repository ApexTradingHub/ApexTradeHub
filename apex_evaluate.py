"""
ApexScan – Signal Evaluator
============================
Liest die neuesten Signale, reichert sie mit yfinance Daten an
und generiert einen fertigen Prompt für Claude.ai.

Aufruf: py apex_evaluate.py
"""

import json
import subprocess
import webbrowser
from datetime import datetime

SIGNALS_FILE = "apex_signals.json"
MARKET_FILE  = "apex_market_regime.json"

LOCALE_MONTHS = {
    1:"Januar", 2:"Februar", 3:"März", 4:"April",
    5:"Mai", 6:"Juni", 7:"Juli", 8:"August",
    9:"September", 10:"Oktober", 11:"November", 12:"Dezember"
}

def date_de(dt):
    return f"{dt.day}. {LOCALE_MONTHS[dt.month]} {dt.year}"


def load_latest_signals():
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8") as f:
            all_signals = json.load(f)
    except FileNotFoundError:
        return [], None

    if not all_signals:
        return [], None

    dates = sorted({s.get("date","") for s in all_signals}, reverse=True)
    latest_date = dates[0]
    signals = [s for s in all_signals if s.get("date") == latest_date
               and s.get("setup") in {"BREAKOUT", "REVERSAL"}]
    return signals, latest_date


def load_market_regime():
    try:
        with open(MARKET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def enrich_signal(ticker):
    """Holt Earnings, Analyst Konsens, 52w Range aus yfinance."""
    result = {
        "earnings_date": "?",
        "analyst_rating": "?",
        "analyst_target": "?",
        "week52_low": "?",
        "week52_high": "?",
    }
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Earnings Datum
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else None
                if dates is not None:
                    ed = dates.iloc[0] if hasattr(dates, 'iloc') else dates
                    if hasattr(ed, 'date'):
                        days_away = (ed.date() - datetime.now().date()).days
                        result["earnings_date"] = f"{date_de(ed)} (in {days_away} Tagen)"
                    else:
                        result["earnings_date"] = str(ed)
        except Exception:
            pass

        # Info
        try:
            info = t.info
            # Analyst rating
            rec = info.get("recommendationKey", "")
            n_analysts = info.get("numberOfAnalystOpinions", "")
            target = info.get("targetMeanPrice", "")
            if rec:
                result["analyst_rating"] = rec.upper()
            if target:
                result["analyst_target"] = f"${target:.2f}" if isinstance(target, float) else str(target)
            if n_analysts:
                result["analyst_rating"] += f" ({n_analysts} Analysten)"

            # 52w Range
            low52  = info.get("fiftyTwoWeekLow", "?")
            high52 = info.get("fiftyTwoWeekHigh", "?")
            result["week52_low"]  = f"${low52:.2f}"  if isinstance(low52,  float) else str(low52)
            result["week52_high"] = f"${high52:.2f}" if isinstance(high52, float) else str(high52)
        except Exception:
            pass

    except Exception as e:
        print(f"  yfinance Fehler für {ticker}: {e}")

    return result


def build_prompt(signals, scan_date, regime=None):
    today_str = date_de(datetime.now())
    scan_str  = date_de(datetime.strptime(scan_date, "%Y-%m-%d"))

    regime_line = ""
    if regime:
        regime_line = f"Marktregime: {regime.get('mode','?')} ({regime.get('summary','')})\n"

    lines = [
        f"Ich bin ein privater Trader. Mein technischer Scanner hat am {scan_str} folgende Signale gefunden.",
        f"Heute ist der {today_str}.",
        regime_line,
        "Bitte bewerte jedes Signal kurz auf Basis aktueller Nachrichten, Marktumfeld und fundamentaler Lage.",
        "Für jede Aktie:",
        "  - 1-2 Sätze was du weißt + mögliche Risiken",
        "  - Klare Einschätzung: KAUFEN / ABWARTEN / MEIDEN",
        "",
        "══════════════════════════════════════════════════",
    ]

    for i, s in enumerate(signals, 1):
        upside  = s.get("upside_pct", s.get("upside", "?"))
        ticker  = s.get("ticker", "?")
        extra   = s.get("_enriched", {})

        lines.append(f"{i}. {ticker} | {s.get('setup')} | {s.get('sector','?')}")
        lines.append(f"   Kurs: ${s.get('price')}  →  Ziel: ${s.get('target')}  (+{upside}%)")
        lines.append(f"   Einstieg ab: ${s.get('buy_above')}  |  Stop: ${s.get('stop')}")
        lines.append(f"   RR: {s.get('rr')}  |  RSI: {s.get('rsi')}  |  Horizont: {s.get('horizon','?')}")
        lines.append(f"   Perf 20d: {s.get('perf_20d', s.get('perf_20','?'))}%  |  60d: {s.get('perf_60d', s.get('perf_60','?'))}%  |  Score: {s.get('score','?')}")
        lines.append(f"   52w Range: {extra.get('week52_low','?')} – {extra.get('week52_high','?')}")
        lines.append(f"   Earnings: {extra.get('earnings_date','?')}")
        lines.append(f"   Analysten: {extra.get('analyst_rating','?')}  |  Ø Kursziel: {extra.get('analyst_target','?')}")
        lines.append("")

    lines += [
        "══════════════════════════════════════════════════",
        "Bitte halte dich kurz – pro Signal maximal 3 Zeilen.",
        "Schreibe am Ende welche 1-2 Signale du heute für am stärksten hältst und warum.",
    ]

    return "\n".join(lines)


def copy_to_clipboard(text):
    try:
        proc = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
        proc.communicate(input=text.encode('utf-16-le'))
        return True
    except Exception as e:
        print(f"Clipboard Fehler: {e}")
        return False


def main():
    signals, scan_date = load_latest_signals()

    if not signals:
        print("Keine BREAKOUT/REVERSAL Signale gefunden.")
        print("Stelle sicher dass ApexScan.py bereits gelaufen ist.")
        return

    regime = load_market_regime()

    print(f"Signale vom {scan_date}: {len(signals)} gefunden")
    print("Lade Zusatzdaten (Earnings, Analysten, 52w)...")
    print()

    # Anreichern
    for s in signals:
        print(f"  → {s['ticker']}...")
        s["_enriched"] = enrich_signal(s["ticker"])

    prompt = build_prompt(signals, scan_date, regime)

    print()
    print("=" * 55)
    print(prompt)
    print("=" * 55)
    print()

    if copy_to_clipboard(prompt):
        print("✓ Prompt in Zwischenablage kopiert!")
    else:
        filename = f"apex_prompt_{scan_date}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"✓ Prompt gespeichert: {filename}")

    print()
    answer = input("Claude.ai jetzt öffnen? (j/n): ").strip().lower()
    if answer == "j":
        webbrowser.open("https://claude.ai")
        print("✓ Claude.ai geöffnet – Strg+V und Enter!")
    else:
        print("Claude.ai öffnen und Strg+V drücken.")


if __name__ == "__main__":
    main()