@echo off
title ApexScan
color 0A

echo.
echo  ========================================
echo   APEXSCAN - Trading Signal Scanner
echo  ========================================
echo.

cd /d "%~dp0"

echo  Synchronisiere mit GitHub...
git pull origin master

echo.
echo  ========================================
echo   Scanner wird gestartet...
echo  ========================================
echo.

py ApexScan.py

echo.
echo  ========================================
echo   Equity Tracker wird aktualisiert...
echo  ========================================
echo.

py apex_equity.py

echo.
echo  ========================================
echo   Dashboard wird auf GitHub gepusht...
echo  ========================================
echo.

git add apex_signals.json apex_equity_results.json apex_equity_top2.json apex_market.json dashboard.html
git commit -m "Scan update %date% %time%"
git push origin master

echo.
echo  Dashboard live unter:
echo  https://apextradinghub.github.io/ApexTradeHub/dashboard.html
echo.