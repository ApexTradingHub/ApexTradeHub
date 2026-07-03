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
            # Cloudflare bannt Python-urllib default UA (Error 1010) — normaler UA reicht:
            "User-Agent":   "ApexTrader/1.0 (+https://github.com/ApexTradingHub)",
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
    def search_instrument(self, query, by_symbol=False):
        """Instrument-Suche. Zwei Modi:
        - by_symbol=True: internalSymbolFull=AAPL (exact) + fields=... -> liefert Symbole mit
        - by_symbol=False (default): query=Apple (fuzzy) -> Trending-Liste, meist nur IDs"""
        if by_symbol:
            return self._request("GET", "/api/v1/market-data/search", params={
                "internalSymbolFull": query,
                "fields": "instrumentId,internalSymbolFull,displayname,marketId,symbol",
            })
        return self._request("GET", "/api/v1/market-data/search", params={"query": query})

    def resolve_ticker(self, ticker):
        """Ticker (AAPL) -> instrumentId (int) via internalSymbolFull-Search."""
        res = self.search_instrument(ticker, by_symbol=True)
        items = res.get("items", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        tk = ticker.upper()
        def iid(it):
            v = it.get("instrumentId") or it.get("id")
            try: return int(v) if v is not None else None
            except (TypeError, ValueError): return None
        # Symbol-Match (jetzt HABEN wir die Felder)
        for it in items:
            i = iid(it)
            if i is None or i <= 0: continue
            for f in ("internalSymbolFull", "symbol", "symbolFull", "ticker"):
                if (it.get(f) or "").upper() == tk:
                    return i
        # Fallback: erster valider Treffer (bei exact-search sollte Item 0 stimmen)
        for it in items:
            i = iid(it)
            if i is not None and i > 0:
                return i
        return None

    def get_instruments_meta(self, instrument_ids):
        """Bulk-Metadata fuer eine Liste von instrumentIds. Liefert Namen/Symbole/Sektor."""
        if isinstance(instrument_ids, (int, str)):
            instrument_ids = [instrument_ids]
        ids_str = ",".join(str(i) for i in instrument_ids)
        return self._request("GET", "/api/v1/market-data/instruments", params={"instrumentIds": ids_str})

    def _trade_prefix(self):
        """/trading/info/demo oder /trading/info/live abhaengig von env."""
        return f"/api/v1/trading/info/{self.env}"

    def get_balance(self):
        """Portfolio-Metadaten: Waehrung, buying power, open P&L."""
        return self._request("GET", f"{self._trade_prefix()}/portfolio")

    def get_positions(self):
        """Offene Positionen + pending Orders aus der portfolio-Response.
        eToro fillt Orders erst am Market-Open (pre-open = 'Ausstehend noch nicht ausgefuehrt')."""
        r = self.get_balance()
        cp = r.get("clientPortfolio", r) if isinstance(r, dict) else {}
        pending = (cp.get("orders", []) + cp.get("stockOrders", []) +
                   cp.get("entryOrders", []) + cp.get("ordersForOpen", []))
        return {
            "positions":  cp.get("positions", []),
            "pending":    pending,
            "n_positions": len(cp.get("positions", [])),
            "n_pending":  len(pending),
            "credit":     cp.get("credit", 0),
            "bonusCredit": cp.get("bonusCredit", 0),
        }

    def get_instrument_details(self, instrument_id):
        """Instrument-Metadata (symbol, displayName, pipSize)."""
        return self._request("GET", f"/api/v1/market-data/instruments/{instrument_id}")

    def get_rates(self, instrument_ids):
        """Live snapshot rates (bid/ask/execution) fuer bis zu 100 IDs.
        Endpoint: /market-data/instruments/rates (bestaetigt via api-portal.etoro.com).
        Rate-Limit: 120 req/60s."""
        if isinstance(instrument_ids, (int, str)):
            instrument_ids = [instrument_ids]
        return self._request("GET", "/api/v1/market-data/instruments/rates",
                             params={"instrumentIds": ",".join(str(i) for i in instrument_ids)})

    # ---------- Write: Trading (RESPEKTIERT DRY-RUN) ----------
    def open_position(self, instrument_id, size_usd, direction="Buy", stop_loss=None, take_profit=None):
        """Marktorder oeffnen. eToro erwartet 'transaction' mit TitleCase-Werten:
        Buy | Sell | SellShort | BuyToCover."""
        # eToro v2 execution: PascalCase-Felder erforderlich (verifiziert 03-07 nach 2. Order
        # ohne SL/TP-Fill — camelCase 'stopLossRate' wurde ignoriert, PascalCase 'StopLossRate' greift).
        d = str(direction).strip().lower()
        is_buy = d in ("buy", "buytocover")
        tx = {"buy": "Buy", "sell": "Sell", "sellshort": "SellShort", "buytocover": "BuyToCover"}.get(d, direction)
        body = {
            "InstrumentID": instrument_id,
            "Amount":       float(size_usd),
            "IsBuy":        is_buy,
            "Leverage":     1,
            "transaction":  tx,   # camelCase legacy, akzeptiert
        }
        if stop_loss is not None:
            body["StopLossRate"] = float(stop_loss)
            body["StopLossType"] = "fixed"
            body["IsNoStopLoss"] = False
        else:
            body["IsNoStopLoss"] = True
        if take_profit is not None:
            body["TakeProfitRate"] = float(take_profit)
            body["IsNoTakeProfit"] = False
        else:
            body["IsNoTakeProfit"] = True
        body["IsTslEnabled"] = False
        # v2 unified order endpoint mit env-Prefix (demo|live).
        return self._request("POST", f"/api/v2/trading/execution/{self.env}/orders", body=body, write=True)

    def close_position(self, position_id):
        return self._request("DELETE", f"/api/v1/trading/execution/{self.env}/positions/{position_id}", write=True)

    def update_sl_tp(self, position_id, stop_loss=None, take_profit=None):
        """SL/TP nachziehen (fuer Trailing-Ladder). eToro-Felder: stopLossRate/takeProfitRate."""
        body = {}
        if stop_loss   is not None: body["stopLossRate"]   = float(stop_loss)
        if take_profit is not None: body["takeProfitRate"] = float(take_profit)
        return self._request("PATCH", f"/api/v1/trading/execution/{self.env}/positions/{position_id}", body=body, write=True)


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

    elif cmd == "quote":   # Live rates via /market-data/instruments/rates
        if len(sys.argv) < 3:
            print("Usage: quote TICKER"); return
        tk = sys.argv[2].upper()
        iid = c.resolve_ticker(tk)
        print(f"{tk} -> instrumentId {iid}")
        if iid:
            try:
                print(json.dumps(c.get_rates(iid), indent=2)[:600])
            except EToroError as e:
                print(f"Rates-FEHLER {e.status}: {e.message}")

    elif cmd == "resolve":
        if len(sys.argv) < 3:
            print("Usage: resolve TICKER"); return
        tk = sys.argv[2].upper()
        print(f"{tk} -> instrumentId {c.resolve_ticker(tk)}")

    elif cmd == "search":   # Debug: symbol-based search + meta lookup
        if len(sys.argv) < 3:
            print("Usage: search SYMBOL"); return
        r = c.search_instrument(sys.argv[2], by_symbol=True)
        items = r.get("items", []) if isinstance(r, dict) else []
        ids = [it.get("instrumentId") for it in items if it.get("instrumentId", 0) > 0]
        print(f"totalItems={r.get('totalItems')}  IDs: {ids}")
        if ids:
            try:
                meta = c.get_instruments_meta(ids[:8])
                m_items = meta.get("items", meta) if isinstance(meta, dict) else meta
                if isinstance(m_items, list):
                    for m in m_items[:8]:
                        print(f"  {m}")
                else:
                    print(f"  {meta}")
            except EToroError as e:
                print(f"  meta-lookup FEHLER {e.status}: {e.message}")

    elif cmd == "open":
        # py etoro_client.py open TICKER SIZE_USD [SL_PCT] [TP_PCT]
        # Beispiel: py etoro_client.py open AAPL 50 4 6  -> $50 AAPL, SL -4%, TP +6%
        # Respektiert ETORO_DRY_RUN=1 (default) — logged nur.
        if len(sys.argv) < 4:
            print("Usage: open TICKER SIZE_USD [SL_PCT] [TP_PCT]"); return
        tk = sys.argv[2].upper()
        size = float(sys.argv[3])
        sl_pct = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
        tp_pct = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
        iid = c.resolve_ticker(tk)
        if not iid:
            print(f"Ticker nicht aufgeloest: {tk}"); return
        # Preis via eToro-Rates (bid/ask). Ask fuer BUY, Bid fuer SELL.
        try:
            r = c.get_rates(iid)
            rates = r.get("rates", r.get("items", [])) if isinstance(r, dict) else (r if isinstance(r, list) else [])
            it = rates[0] if rates else {}
            price = it.get("ask") or it.get("lastExecution") or it.get("bid")
        except EToroError as e:
            print(f"Rates-FEHLER {e.status}: {e.message}"); return
        if not price:
            print(f"Kein Preis in Rates-Antwort: {r}"); return
        price = float(price)
        sl_abs = round(price * (1 - sl_pct/100), 2)
        tp_abs = round(price * (1 + tp_pct/100), 2)
        print(f"{tk} ({iid}) @ ${price:.2f} -> SL ${sl_abs} / TP ${tp_abs}")
        try:
            r = c.open_position(iid, size, "BUY", stop_loss=sl_abs, take_profit=tp_abs)
            print(f"Result: {json.dumps(r, indent=2)[:400]}")
        except EToroError as e:
            print(f"FEHLER {e.status}: {e.message}")

    else:
        print(f"Unbekanntes Kommando: {cmd}")


if __name__ == "__main__":
    _cli()
