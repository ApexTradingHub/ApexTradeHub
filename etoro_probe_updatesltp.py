# -*- coding: utf-8 -*-
"""Prober fuer den korrekten Update-SL/TP-Endpoint (aktueller ist [404] RouteNotFound).
Testet Kandidaten-Pfade gegen SJMs echte Demo-Position. 404 = Route weg, 400 = Route da/Body
falsch, 200 = Treffer (setzt gleich SJMs SL korrekt). Laeuft auf der VM (liest Keys aus env).

  python3 etoro_probe_updatesltp.py
"""
import sys
from etoro_client import EToroClient, EToroError

# SJM: echte Demo-Position, gewuenschter Trailing-SL
POSITION_ID = 3567641111
SL = 120.18
TP = 129.98

c = EToroClient()
c.dry_run = False            # echt testen, nicht dry-run
env = c.env                  # "demo"
print(f"[{env.upper()}] dry_run={c.dry_run} — teste Update-SL/TP-Endpoints fuer Position {POSITION_ID}\n")

body_pascal = {"positionId": POSITION_ID, "StopLossRate": SL, "TakeProfitRate": TP}
body_id_path = {"StopLossRate": SL, "TakeProfitRate": TP}   # wenn id im Pfad

# (method, path, body, note)
candidates = [
    ("PATCH", f"/api/v1/trading/execution/{env}/positions", body_pascal, "Memory-dokumentiert (+/execution/)"),
    ("PATCH", f"/api/v1/trading/execution/{env}/positions/{POSITION_ID}", body_id_path, "execution + id im Pfad"),
    ("PATCH", f"/api/v2/trading/execution/{env}/positions", body_pascal, "v2 execution (wie orders)"),
    ("PUT",   f"/api/v1/trading/execution/{env}/positions", body_pascal, "PUT statt PATCH"),
    ("POST",  f"/api/v1/trading/execution/{env}/edit-positions", body_pascal, "edit-positions"),
    ("PATCH", f"/api/v1/trading/execution/{env}/positions/{POSITION_ID}", body_pascal, "execution+id+voller Body"),
]

hit = None
for method, path, body, note in candidates:
    try:
        r = c._request(method, path, body=body, write=True)
        print(f"  200 OK  {method} {path}   [{note}]")
        print(f"          response: {str(r)[:200]}")
        hit = (method, path, note); break
    except EToroError as e:
        print(f"  [{e.status}] {getattr(e,'code','?') or getattr(e,'errorCode','?')} {method} {path}   [{note}]")
        # 400/422 = Endpoint EXISTIERT (Body/Validation) -> auch wertvoll
        if e.status in (400, 422):
            print(f"          -> Route EXISTIERT (Status {e.status}): {getattr(e,'message','')[:160]}")
    except Exception as e:
        print(f"  ERR     {method} {path}: {e}")

print()
if hit:
    print(f"=== TREFFER: {hit[0]} {hit[1]}  ({hit[2]}) — SJM-SL auf {SL} gesetzt. ===")
else:
    print("=== Kein 200. Falls ein 400/422 dabei war: Route existiert, Body-Format anpassen. ===")
    print("    Sonst weitere Pfad-Varianten noetig (eToro-Docs / builders.etoro.com).")
