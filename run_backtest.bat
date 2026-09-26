@echo off
title AI Trading Backtest - Paper Mode
echo Running backtest on FREE Yahoo Finance data (no keys needed)...
echo.
python -m trading.backtest RELIANCE --days 365
pause
