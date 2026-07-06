@echo off
cd /d "%~dp0"
start http://localhost:8080/dashboard.html
py -m http.server 8080
