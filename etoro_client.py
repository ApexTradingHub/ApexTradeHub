# -*- coding: utf-8 -*-
"""eToro Public API Client — REST wrapper fuer Apex-Trader.

Auth-Mapping (bei eToro VERDREHT, aus Debug 2026-07-02 bestaetigt):
  x-api-key   = ETORO_API_KEY   = "Oeffentlicher Schluessel" (aus dem Portal)
  x-user-key  = ETORO_USER_KEY  = generierter API-Schluessel-Wert (nur 1x angezeigt)

Env-Vars (im run_trader.sh setzen, NICHT in ~/.bashrc — Cron sourct das nicht):
  ETORO_API_KEY     - dein "Oeffentlicher Schluessel"
  ETORO_USER_KEY    - dein generierter Schluessel-Wert
  ETORO_ENV         - "demo" (default) oder "live"
  ETORO_DRY_RUN     - "1" (default) = Write-Calls loggen aber nicht senden

CLI:
  py etoro_client.py test       - Auth-Test (Apple-Suche)
  py etoro_client.py balance    - virtuelles Guthaben abrufen
  py etoro_client.py positions  - offene Positionen listen
  py etoro_client.py quote AAPL - Live-Preis fuer Ticker
"""
import os
import sys
import json
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error

BASE_URL   = "https://public-api.etoro.com/api/v1"
BASE_URL_V2 = "https://public-api.etoro.com/api/v2"


class EToroError(Exception):
    def __init__(self, status, code, message, body=None):
        self.status = status
        self.code = code
        self.message = message
        self.body = body
        super().__init__(f"[{status}] {code}: {message}")


class EToroClient:
    def __init__(self, api_key=None, user_key=None, env=None, dry_run=None):
        self.api_key  = api_key  or os.environ.get("ETORO_API_KEY")
        self.user_key = user_key or os.environ.get("ETORO_USER_KEY")
        self.env      = (env or os.environ.get("ETORO_ENV", "demo")).lower()
        self.dry_run  = bool(int(os.environ.get("ETORO_DRY_RUN", "1"))) if dry_run is None else dry_run
        if not self.api_key or not self.user_key:
            raise RuntimeError("ETORO_API_KEY + ETORO_USER_KEY muessen gesetzt sein")
        if self.env not in ("demo", "live"):
            raise ValueError(f"ETORO_ENV must be demo|live, got {self.env}")

    # ---------- HTTP core ----------
    def _headers(self):
        return {
            "x-api-key":    self.api_key,
            "x-user-key":   self.user_key,
            "x-request-id": str(uuid.uuid4()),
            "Accept":       "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method, path, params=None, body=None, write=False):
        """Kern-Request. write=True -> respektiert Dry-Run."""
        if write and self.dry_run:
            print(f"[DRY-RUN] {method} {path}  params={params}  body={body}")
            return {"_dry_run": True, "method": method, "path": path, "body": body}
        if path.startswith("http"):
            url = path
        elif path.startswith("/api/"):
            url = "https://public-api.etoro.com" + path
        else:
            url = BASE_URL + path
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            body_raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
            try:
                j = json.loads(body_raw)
                raise EToroError(e.code, j.get("errorCode", "?"), j.get("errorMessage", body_raw), j)
            except (ValueError, AttributeError):
                raise EToroError(e.code, "?", body_raw, None)

    # ---------- Read-Only: known-working ----------
    def search_instrument(self, query):
        """Instrument-Suche. Getestet 2026-07-02 = OK. Nutzen wir fuer Ticker->instrumentId."""
        return self._request("GET", "/api/v1/market-data/search", params={"query": query})

    def resolve_ticker(self, ticker):
        """Ticker (AAPL) -> instrumentId (int). Erster Treffer der Symbol matcht."""
        res = self.search_instrument(ticker)
        # Antwortstruktur: siehe API-Portal Doku. Suche nach 'symbol' == ticker in Ergebnisliste.
        items = res.get("items") or res.get("results") or res.get("data") or (res if isinstance(res, list) else [])
        for it in items if isinstance(items, list) else []:
            sym = (it.get("symbol") or it.get("ticker") or "").upper()
            if sym == ticker.upper():
                return it.get("instrumentId") or it.get("id")
        # Fallback: erster Treffer
        if items and isinstance(items, list):
            it = items[0]
            return it.get("instrumentId") or it.get("id")
        return None

    # ---------- Read-Only: TODO endpoints (Pfade aus api-portal.etoro.com noch verifizieren) ----------
    def get_balance(self):
        """Virtuelles Guthaben (Demo) bzw. echtes (Live). Pfad TBD via API-Portal."""
        # Wahrscheinlich: /api/v1/user/portfolio  ODER  /api/v1/user/balance
        # Bei 404 hier: exakten Pfad aus api-portal.etoro.com -> Portfolio API kopieren.
        return self._request("GET", "/api/v1/user/portfolio")

    def get_positions(self):
        """Offene Positionen. Pfad TBD."""
        return self._request("GET", "/api/v1/user/positions")

    def get_quote(self, instrument_id):
        """Live-Preis (bid/ask). Pfad TBD."""
        return self._request("GET", f"/api/v1/market-data/instruments/{instrument_id}/quote")

    # ---------- Write: Trading (RESPEKTIERT DRY-RUN) ----------
    def open_position(self, instrument_id, size_usd, direction="BUY", stop_loss=None, take_profit=None):
        """Marktorder oeffnen. Pfad TBD (v2 unified order endpoint).
        Bei Dry-Run: loggen, nicht senden."""
        body = {
            "instrumentId": instrument_id,
            "amount":       float(size_usd),
            "direction":    direction,
            "leverage":     1,
        }
        if stop_loss   is not None: body["stopLoss"]   = float(stop_loss)
        if take_profit is not None: body["takeProfit"] = float(take_profit)
        # Demo vs Live: Doku sagt Trading-Endpoints haben /demo/ im Pfad wenn Demo-Key.
        path = "/api/v2/trading/execution/orders"
        return self._request("POST", path, body=body, write=True)

    def close_position(self, position_id):
        """Position schliessen. Pfad TBD."""
        return self._request("DELETE", f"/api/v1/trading/positions/{position_id}", write=True)

    def update_sl_tp(self, position_id, stop_loss=None, take_profit=None):
        """SL/TP nachziehen (fuer Trailing-Ladder)."""
        body = {}
        if stop_loss   is not None: body["stopLoss"]   = float(stop_loss)
        if take_profit is not None: body["takeProfit"] = float(take_profit)
        return self._request("PATCH", f"/api/v1/trading/positions/{position_id}", body=body, write=True)


# ---------- CLI ----------
def _cli():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].lower()
    c = EToroClient()
    print(f"[{c.env.upper()}] dry_run={c.dry_run}")

    if cmd == "test":
        r = c.search_instrument("Apple")
        n = len(r.get("items") or r.get("results") or r.get("data") or [])
        print(f"OK — search returned data (top-level keys: {list(r.keys())[:5]}, ~{n} items)")

    elif cmd == "balance":
        try:
            r = c.get_balance()
            print(json.dumps(r, indent=2)[:800])
        except EToroError as e:
            print(f"FEHLER {e.status}: {e.message}")
            print("Hinweis: falls 404 -> genauen Pfad aus api-portal.etoro.com -> User/Portfolio API kopieren")

    elif cmd == "positions":
        try:
            r = c.get_positions()
            print(json.dumps(r, indent=2)[:800])
        except EToroError as e:
            print(f"FEHLER {e.status}: {e.message}")

    elif cmd == "quote":
        if len(sys.argv) < 3:
            print("Usage: quote TICKER"); return
        tk = sys.argv[2].upper()
        iid = c.resolve_ticker(tk)
        print(f"{tk} -> instrumentId {iid}")
        if iid:
            try:
                print(json.dumps(c.get_quote(iid), indent=2)[:400])
            except EToroError as e:
                print(f"Quote-FEHLER {e.status}: {e.message}")

    elif cmd == "resolve":
        if len(sys.argv) < 3:
            print("Usage: resolve TICKER"); return
        tk = sys.argv[2].upper()
        print(f"{tk} -> instrumentId {c.resolve_ticker(tk)}")

    else:
        print(f"Unbekanntes Kommando: {cmd}")


if __name__ == "__main__":
    _cli()
