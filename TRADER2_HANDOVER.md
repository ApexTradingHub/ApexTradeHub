# Memo: zweiter AI-Trader auf dem eToro-Demo-Konto

**Stand 2026-08-07.** Kontext für einen neuen Chat, der den zweiten Trader aufsetzt.

## Ausgangslage

Das eToro-Demo-Konto ist **frei**. Der bestehende ApexNext-Trader wurde von `live` auf
`paper` umgestellt und seine 6 offenen Positionen wurden glattgestellt — das Konto ist leer.
ApexNext läuft weiter, schickt aber nichts mehr an eToro.

## Was du NICHT anfassen darfst

| | |
|---|---|
| `~/ApexTradeHub/` | Verzeichnis des bestehenden Traders. Sein `run_trader.sh` macht dort `git checkout -- <state-files>` und `git pull --rebase` — es würde fremde Dateien zurücksetzen und überschreiben. **Eigenes Verzeichnis benutzen** (`~/trader2`). |
| `~/run_trader.sh` | Enthält die eToro-Keys von ApexNext im Klartext, liegt bewusst außerhalb des Repos. Nicht kopieren, nicht wiederverwenden. |
| Crontab-Zeile `*/5 13-21 * * 1-5` | Gehört ApexNext. Nicht ändern. |
| Globales `pip install` | **Der wichtigste Punkt, siehe unten.** |

## Die drei Regeln

**1. Eigene Python-Umgebung — nicht verhandelbar.**
```bash
cd ~/trader2 && python3 -m venv venv && source venv/bin/activate
```
Im `run.sh` dann `/home/ubuntu/trader2/venv/bin/python` benutzen, **nicht** `python3`.

*Warum so streng:* Ein globales Paket-Upgrade hat ApexNext schon einmal getroffen — ein
unpinntes `yfinance` zog eine neue `lxml`-Anforderung nach, der ImportError wurde von einem
`except: pass` verschluckt, und die komplette Earnings-Logik war **30 Tage lang stillschweigend
tot**. Niemand hat es gemerkt, weil nur ein Feature ausfiel, nicht das Programm.

**2. Cron versetzt takten.** ApexNext läuft zur Minute 0, 5, 10 … Für trader2:
```
2-57/5 13-21 * * 1-5 /home/ubuntu/trader2/run.sh >> /home/ubuntu/trader2.log 2>&1
```
Eigene Logdatei — nicht in `~/trader.log` schreiben.

**3. Eigene API-Keys** im eToro-Portal erzeugen, nicht die von ApexNext wiederverwenden.
Beide bleiben so unabhängig widerrufbar.

## VM-Budget (gemessen 2026-08-07)

```
verfügbar        601 MB   (nach Deaktivierung von fwupd, das 148 MB verschwendete)
ApexNext Spitze  138 MB / 32 Sekunden Laufzeit
Swap             1,9 GB frei, praktisch unbenutzt
Platte           38 GB frei
```
Reichlich Platz. Miss deinen Trader genauso:
`/usr/bin/time -v <befehl> 2>&1 | grep "Maximum resident"`.
Über ~400 MB Spitze wird es eng — dann lieber eine zweite Always-Free-VM.

## eToro-API: die Fallen, die Stunden gekostet haben

- **Header-Bezeichnungen sind verdreht.** `x-api-key` = der *„Öffentliche Schlüssel"* aus dem
  Portal. `x-user-key` = der *generierte* Schlüsselwert (wird nur **einmal** angezeigt).
- **`User-Agent`-Header ist Pflicht.** Ohne ihn blockt Cloudflare mit Error 1010, ohne Body.
- **Case-Inkonsistenz:** Order öffnen (`POST /api/v2/trading/execution/{env}/orders`) will
  **PascalCase** (`InstrumentID`, `StopLossRate`). SL/TP ändern
  (`PATCH /api/v2/trading/{env}/positions/{positionId}`) will **camelCase**
  (`stopLossRate`) und gibt **202**, nicht 200. Falscher Pfad → `[404] RouteNotFound`,
  still verschluckt.
- **Position schließen ist POST, nicht DELETE:**
  `POST /api/v1/trading/execution/{env}/market-close-orders/positions/{positionId}`
- `orderId` (aus dem Öffnen) ≠ `positionID` (aus dem Portfolio). Zum Schließen die **positionID**.
- Referenz-Implementierung mit allen Endpunkten: `~/ApexTradeHub/etoro_client.py` — lesen lohnt,
  kopieren in dein eigenes Verzeichnis ist ok.

## Zwei Dinge, die später wehtun würden

- **Nicht beide Systeme gleichzeitig auf `live`.** Wenn ApexNext zurückgeschaltet wird, während
  trader2 läuft, verwalten zwei Systeme dasselbe Konto und sehen gegenseitig fremde Positionen.
  Vorher klären, wem das Konto gehört.
- **Ein Stop garantiert die Auslösung, nicht den Preis.** Bei einer Übernacht-Lücke wird zum
  ersten Preis am Morgen verkauft — bei uns zuletzt 5 % unter dem Stop (PINS nach Earnings).
  Wer Positionen über Nacht hält, trägt Gap-Risiko, das kein Trailing abfängt.
