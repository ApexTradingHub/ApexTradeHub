# -*- coding: utf-8 -*-
"""Einmalig: schliesst ALLE von uns getrackten offenen eToro-Positionen (Market-Close).

Anlass 2026-08-07: Der eToro-Spiegel wird abgeschaltet (TRADING_MODE=live -> paper), weil ein
zweiter AI-Trader auf demselben Demo-Konto getestet werden soll. Damit der neue Trader ein
LEERES Konto vorfindet und unsere Positionen nicht als seine eigenen behandelt, werden sie
vorher glattgestellt.

SICHERHEIT: schliesst NUR Positionen, die in apex_positions.json unter "open" stehen UND eine
etoro_position_id haben. Alles andere im Konto wird nicht angefasst.

Laeuft auf der VM (Keys aus run_trader.sh sourcen, Werte erscheinen nie im Output):

  cd ~/ApexTradeHub
  eval "$(grep '^export ' ~/run_trader.sh)"
  python3 etoro_close_all.py            # Trockenlauf: zeigt nur, was geschlossen wuerde
  python3 etoro_close_all.py --apply    # schliesst wirklich
"""
import io
import json
import sys

from etoro_client import EToroClient, EToroError

APPLY = "--apply" in sys.argv

st = json.load(io.open("apex_positions.json", encoding="utf-8"))
open_pos = [p for p in st.get("open", []) if p.get("etoro_position_id")]
no_id = [p for p in st.get("open", []) if not p.get("etoro_position_id")]

c = EToroClient()
c.dry_run = not APPLY
print(f"[{c.env.upper()}] {'SCHLIESSE' if APPLY else 'TROCKENLAUF —'} "
      f"{len(open_pos)} getrackte eToro-Positionen\n")

ok = fail = 0
for p in open_pos:
    tk = p.get("ticker")
    pid = p.get("etoro_position_id")
    iid = p.get("etoro_instrument_id")
    pnl = p.get("pnl_pct")
    line = (f"  {tk:8} pos {pid}  entry ${p.get('entry_actual')}  "
            f"pnl {float(pnl or 0):+.2f}%")
    if not APPLY:
        print(line + "   -> wuerde geschlossen")
        continue
    try:
        c.close_position(pid, instrument_id=iid)
        print(line + "   -> GESCHLOSSEN")
        ok += 1
    except EToroError as e:
        print(line + f"   -> FAIL [{e.status}] {getattr(e, 'message', '')[:70]}")
        fail += 1
    except Exception as e:
        print(line + f"   -> FAIL {e}")
        fail += 1

for p in no_id:
    print(f"  {p.get('ticker'):8} SKIP — keine etoro_position_id (nur Paper, nichts zu schliessen)")

if APPLY:
    print(f"\nFertig: {ok} geschlossen, {fail} fehlgeschlagen, {len(no_id)} skip")
    print("Danach im eToro-Portfolio gegenpruefen, dass das Konto leer ist.")
else:
    print(f"\nTrockenlauf — nichts gesendet. Mit --apply ausfuehren.")
