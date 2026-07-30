# -*- coding: utf-8 -*-
"""Einmaliges Reconcile: pusht den aktuellen Paper-SL/TP JEDER offenen Live-Position zu eToro.
Noetig weil der update_sl_tp-Bug (404, jetzt gefixt) seit Live-Start ALLE Trailing-Pushes
verschluckte -> eToro haelt bei vielen Positionen noch den alten Initial-Stop, waehrend Paper
den Stop laengst nachgezogen hat. Laeuft auf der VM (Keys aus run_trader.sh sourcen).

  eval "$(grep '^export ' ~/run_trader.sh)" && python3 etoro_reconcile_stops.py
"""
import json, io
from etoro_client import EToroClient, EToroError

st = json.load(io.open("apex_positions.json", encoding="utf-8"))
c = EToroClient(); c.dry_run = False
print(f"[{c.env.upper()}] Reconcile — pushe Paper-SL/TP aller offenen Positionen zu eToro\n")

ok, skip, fail = 0, 0, 0
for p in st.get("open", []):
    pid = p.get("etoro_position_id")
    tk = p.get("ticker")
    if not pid:
        print(f"  SKIP {tk:8} kein position_id (nicht gesynct)"); skip += 1; continue
    stop, tgt = p.get("stop"), p.get("target")
    try:
        c.update_sl_tp(pid, stop_loss=stop, take_profit=tgt)
        print(f"  OK   {tk:8} SL->${stop}  TP->${tgt}  (pos {pid})"); ok += 1
    except EToroError as e:
        print(f"  FAIL {tk:8} [{e.status}] {getattr(e,'message','')[:80]}"); fail += 1
    except Exception as e:
        print(f"  FAIL {tk:8} {e}"); fail += 1

print(f"\nFertig: {ok} gepusht, {skip} skip, {fail} fail")
