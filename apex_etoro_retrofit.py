# -*- coding: utf-8 -*-
"""Retro-Fix: falsch als 'eToro order_dropped' geschlossene Trades in eToro-History
nachschauen und korrekten Preis/Reason nachtragen.
Beispiel: NKTR wurde am 06.07 real bei eToro geoeffnet+geschlossen (TP+5%), Sync
labelte es aber als phantom weil sync-Feature erst 07.07 kam.

Usage (auf VM oder lokal, env-vars muessen gesetzt sein):
  python3 apex_etoro_retrofit.py --dry-run    # zeigt was korrigiert wuerde
  python3 apex_etoro_retrofit.py --apply       # schreibt Aenderungen
"""
import argparse, json, sys
from datetime import datetime, timezone, timedelta
from etoro_client import EToroClient, EToroError

STATE_FILE = "apex_positions.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Schreibt Aenderungen (Default: dry-run)")
    ap.add_argument("--min-date", default=None, help="minDate fuer History (default: 30d zurueck)")
    args = ap.parse_args()

    state = json.load(open(STATE_FILE, encoding="utf-8"))
    closed = state.get("closed", [])
    candidates = [p for p in closed if p.get("exit_reason") == "eToro order_dropped" and p.get("etoro_order_id")]
    print(f"Kandidaten (closed mit reason='eToro order_dropped' und etoro_order_id): {len(candidates)}")
    if not candidates:
        return

    c = EToroClient(dry_run=False)   # history ist read-only, dry_run egal
    md = args.min_date or (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"Fetching history since {md}...")
    hist = c.get_history(min_date=md, page_size=500)
    items = hist.get("items", hist) if isinstance(hist, dict) else hist
    if not isinstance(items, list):
        print(f"Unerwartete History-Response: {type(items)}"); sys.exit(1)
    print(f"History items: {len(items)}")
    by_order = {int(it.get("orderId", 0)): it for it in items if it.get("orderId")}

    changes = []
    for pos in candidates:
        oid = int(pos.get("etoro_order_id", 0))
        hit = by_order.get(oid)
        if not hit:
            print(f"  {pos.get('ticker'):6s} orderId {oid} -> NICHT in History (echter order_dropped)")
            continue
        close_rate = float(hit.get("closeRate") or 0)
        open_rate  = float(hit.get("openRate") or 0)
        sl_rate    = float(hit.get("stopLossRate") or 0)
        tp_rate    = float(hit.get("takeProfitRate") or 0)
        net_profit = float(hit.get("netProfit") or 0)
        if tp_rate and close_rate >= tp_rate * 0.999:  new_reason = "eToro TP"
        elif sl_rate and close_rate <= sl_rate * 1.001: new_reason = "eToro SL"
        else:                                           new_reason = "eToro closed"
        entry_actual = float(pos.get("entry_actual") or 0) or open_rate
        pnl_pct = ((close_rate / entry_actual) - 1) * 100 if entry_actual else 0
        pnl_usd = float(pos.get("size_usd", 50)) * pnl_pct / 100
        print(f"  {pos.get('ticker'):6s} orderId {oid} -> {new_reason} @ ${close_rate:.2f} "
              f"(open ${open_rate:.2f}, netP ${net_profit:+.2f}, pnl {pnl_pct:+.2f}%)")
        changes.append((pos, {
            "exit_reason":       new_reason,
            "exit_price":        close_rate,
            "etoro_open_rate":   open_rate,
            "etoro_close_rate":  close_rate,
            "etoro_position_id": hit.get("positionId"),
            "etoro_net_profit":  net_profit,
            "pnl_pct":           round(pnl_pct, 2),
            "pnl_usd":           round(pnl_usd, 2),
            "retrofit_at":       datetime.now(timezone.utc).isoformat(),
        }))

    if not args.apply:
        print(f"\nDRY-RUN: {len(changes)} Trade(s) waeren geaendert. Zum Schreiben: --apply")
        return

    for pos, upd in changes:
        pos.update(upd)
    # Stats neu rechnen (wins/losses/equity delta ist minimal — der Trader macht das eh im naechsten Run neu)
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nAPPLIED: {len(changes)} Trade(s) korrigiert -> {STATE_FILE}")

if __name__ == "__main__":
    main()
