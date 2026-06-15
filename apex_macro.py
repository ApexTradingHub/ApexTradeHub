"""
ApexMacro — FRED-Macro-Snapshot fuer Risk-Regime-Klassifikation.

Liest FRED API, schreibt apex_macro.json. Stand-alone, fasst Live-Code (ApexScan.py) NICHT an.
Phase 1: Snapshot-Datenerfassung. Phase 2 (spaeter): Join mit Trade-DB im Backtest.
Phase 3 (spaeter): Score-Penalty / Hard-Gate wenn Macro-Regime RISK_OFF.

Run lokal:
    $env:FRED_API_KEY = "dein_key"
    py apex_macro.py             # write apex_macro.json
    py apex_macro.py --print     # nur Konsole

Run im Cron (taeglich vor Scanner reicht):
    FRED_API_KEY=xxx python apex_macro.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_KEY = os.environ.get("FRED_API_KEY", "")
OUT_FILE = Path(__file__).parent / "apex_macro.json"
HISTORY_FILE = Path(__file__).parent / "apex_macro_history.json"

# 3-State-Label-Mapping für Telegram / Dashboards
REGIME_LABELS = {
    "RISK_ON":  ("🟢", "Good"),
    "ELEVATED": ("🟡", "Mid"),
    "RISK_OFF": ("🔴", "Bad"),
    "UNKNOWN":  ("⚪", "n/a"),
}

# FRED series we track. (id, label, decimals)
SERIES = [
    ("VIXCLS",       "vix",           2),  # VIX Close
    ("BAMLH0A0HYM2", "hy_oas",        2),  # HY Option-Adjusted Spread (%)
    ("T10Y2Y",       "yield_curve",   2),  # 10y - 2y Treasury (%)
    ("DFF",          "fed_funds",     2),  # Effective Fed Funds Rate (%)
    ("DTB3",         "tbill_3m",      2),  # 3M T-Bill (%)
]

# Risk-Regime-Thresholds (Phase 1 — bewusst konservativ, kann nach Backtest tuned werden)
# VIX 20 = elevated, 25 = stress, 30+ = panic. HY-OAS 3.5 = normal, 5 = stress, 7+ = crisis.
VIX_ELEVATED = 20.0
VIX_STRESS   = 25.0
HY_ELEVATED  = 3.5
HY_STRESS    = 5.0


def fetch_latest(series_id: str) -> dict:
    """Hole letzte Beobachtung (sort_order=desc, limit=2 fuer delta)."""
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY env var nicht gesetzt")
    params = {
        "series_id": series_id,
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,  # FRED hat manchmal '.' (kein Wert) — wir filtern
    }
    url = f"{FRED_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ApexNext/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    obs = data.get("observations", [])
    # filter '.' values (FRED-Konvention fuer "kein Wert")
    obs = [o for o in obs if o.get("value") not in (".", "", None)]
    if not obs:
        return {"value": None, "date": None, "prev": None, "delta": None}

    latest = obs[0]
    prev = obs[1] if len(obs) > 1 else None
    val = float(latest["value"])
    prev_val = float(prev["value"]) if prev else None
    delta = (val - prev_val) if prev_val is not None else None
    return {
        "value": val,
        "date": latest["date"],
        "prev": prev_val,
        "delta": delta,
    }


def classify_regime(vix: float | None, hy: float | None) -> tuple[str, list[str]]:
    """RISK_ON / ELEVATED / RISK_OFF basierend auf VIX + HY-OAS."""
    reasons = []
    if vix is None and hy is None:
        return "UNKNOWN", ["keine FRED-Daten"]

    vix_state = "ok"
    if vix is not None:
        if vix >= VIX_STRESS:
            vix_state = "stress"
            reasons.append(f"VIX {vix:.1f} ≥ {VIX_STRESS}")
        elif vix >= VIX_ELEVATED:
            vix_state = "elevated"
            reasons.append(f"VIX {vix:.1f} ≥ {VIX_ELEVATED}")

    hy_state = "ok"
    if hy is not None:
        if hy >= HY_STRESS:
            hy_state = "stress"
            reasons.append(f"HY-OAS {hy:.2f} ≥ {HY_STRESS}")
        elif hy >= HY_ELEVATED:
            hy_state = "elevated"
            reasons.append(f"HY-OAS {hy:.2f} ≥ {HY_ELEVATED}")

    # Worst-of regiert (Credit fuehrt Equity meistens)
    if "stress" in (vix_state, hy_state):
        return "RISK_OFF", reasons
    if "elevated" in (vix_state, hy_state):
        return "ELEVATED", reasons
    return "RISK_ON", reasons or ["VIX + HY-OAS unauffaellig"]


def build_snapshot() -> dict:
    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "series": {},
        "regime_macro": None,
        "reasons": [],
    }
    for series_id, label, decimals in SERIES:
        try:
            d = fetch_latest(series_id)
            out["series"][label] = {
                "id": series_id,
                "value": round(d["value"], decimals) if d["value"] is not None else None,
                "date": d["date"],
                "delta": round(d["delta"], decimals) if d["delta"] is not None else None,
            }
        except Exception as e:
            out["series"][label] = {"id": series_id, "value": None, "error": str(e)[:120]}
        time.sleep(0.15)  # FRED ist generoes, aber sei freundlich

    vix = out["series"].get("vix", {}).get("value")
    hy = out["series"].get("hy_oas", {}).get("value")
    regime, reasons = classify_regime(vix, hy)
    out["regime_macro"] = regime
    out["reasons"] = reasons
    return out


def render_human(snap: dict) -> str:
    lines = [f"# ApexMacro Snapshot — {snap['updated']}", ""]
    regime = snap.get("regime_macro", "UNKNOWN")
    icon, label = REGIME_LABELS.get(regime, REGIME_LABELS["UNKNOWN"])
    lines.append(f"**Regime: {icon} {label} ({regime})**")
    if snap["reasons"]:
        for r in snap["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append("| Series | Value | Date | Δ vs prev |")
    lines.append("|---|---:|---|---:|")
    for label, d in snap["series"].items():
        if "error" in d:
            lines.append(f"| {label} ({d['id']}) | ERROR | — | {d['error']} |")
            continue
        val = d.get("value")
        date = d.get("date") or "—"
        delta = d.get("delta")
        delta_s = f"{delta:+.2f}" if delta is not None else "—"
        val_s = f"{val}" if val is not None else "—"
        lines.append(f"| {label} ({d['id']}) | {val_s} | {date} | {delta_s} |")
    return "\n".join(lines)


def macro_telegram_line(snap: dict | None = None) -> str:
    """1-Zeilen-Header fuer Telegram. Gibt leer-String zurueck wenn keine Daten."""
    if snap is None:
        if not OUT_FILE.exists():
            return ""
        try:
            snap = json.loads(OUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return ""
    regime = snap.get("regime_macro", "UNKNOWN")
    icon, label = REGIME_LABELS.get(regime, REGIME_LABELS["UNKNOWN"])
    vix = snap.get("series", {}).get("vix", {})
    hy = snap.get("series", {}).get("hy_oas", {})
    vix_val = vix.get("value")
    vix_delta = vix.get("delta")
    hy_val = hy.get("value")
    parts = [f"{icon} Macro: <b>{label}</b>"]
    if vix_val is not None:
        d = f" ({vix_delta:+.1f})" if vix_delta is not None else ""
        parts.append(f"VIX {vix_val}{d}")
    if hy_val is not None:
        parts.append(f"HY {hy_val}")
    return " · ".join(parts)


def fetch_history(series_id: str, start: str) -> dict:
    """Vollstaendige tagesweise Serie als {YYYY-MM-DD: float}."""
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY env var nicht gesetzt")
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


def build_history(start: str = "2024-01-01") -> dict:
    """Pull 2y+ history for vix + hy_oas, write apex_macro_history.json."""
    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "start":   start,
        "series":  {},
    }
    for label, series_id in [("vix", "VIXCLS"), ("hy_oas", "BAMLH0A0HYM2"),
                             ("yield_curve", "T10Y2Y")]:
        try:
            out["series"][label] = fetch_history(series_id, start)
            print(f"  {label}: {len(out['series'][label])} obs")
        except Exception as e:
            out["series"][label] = {}
            print(f"  {label} ERROR: {e}")
        time.sleep(0.2)
    return out


def macro_at_date(target_date: str, history: dict | None = None,
                  max_back_days: int = 7) -> dict:
    """
    Lookup macro values at a given date (or before, fuer WE).
    Return: {vix, hy_oas, yield_curve, regime} + 'date_used' showing actual hit date.
    Used by apex_postmortem.py.
    """
    if history is None:
        if not HISTORY_FILE.exists():
            return {"vix": None, "hy_oas": None, "yield_curve": None,
                    "regime": "UNKNOWN", "date_used": None}
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    from datetime import datetime as _dt, timedelta as _td
    series = history.get("series", {})
    d = _dt.strptime(target_date, "%Y-%m-%d").date()

    def _lookup(label):
        s = series.get(label, {})
        for i in range(max_back_days):
            key = (d - _td(days=i)).isoformat()
            if key in s:
                return s[key], key
        return None, None

    vix, vdate = _lookup("vix")
    hy, _      = _lookup("hy_oas")
    yc, _      = _lookup("yield_curve")
    regime, _ = classify_regime(vix, hy)
    return {
        "vix":         round(vix, 2) if vix is not None else None,
        "hy_oas":      round(hy, 2) if hy is not None else None,
        "yield_curve": round(yc, 2) if yc is not None else None,
        "regime":      regime,
        "date_used":   vdate,
    }


def main():
    if not FRED_KEY:
        print("ERROR: FRED_API_KEY env var nicht gesetzt.", file=sys.stderr)
        sys.exit(1)

    # --backfill: pull 2y+ history fuer Postmortem-Lookups
    if "--backfill" in sys.argv:
        start = "2024-01-01"
        for a in sys.argv:
            if a.startswith("--start="):
                start = a.split("=", 1)[1]
        print(f"Backfill from {start}...")
        hist = build_history(start)
        HISTORY_FILE.write_text(json.dumps(hist), encoding="utf-8")
        total = sum(len(s) for s in hist["series"].values())
        print(f"OK: {HISTORY_FILE.name} written ({total} obs across {len(hist['series'])} series)")
        return

    snap = build_snapshot()

    if "--print" in sys.argv:
        print(render_human(snap))
        return

    OUT_FILE.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"OK: {OUT_FILE.name} written ({snap['regime_macro']})")
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print()
        print(render_human(snap))


if __name__ == "__main__":
    main()
