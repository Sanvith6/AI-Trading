@echo off
title AI Trading Agent — NSE Dashboard
echo Starting AI Trading Agent Web Dashboard...
echo.
echo Model: %OPENROUTER_MODEL%
echo Trade Mode: %TRADE_MODE%
echo.
streamlit run trading/trading_app.py --server.port 8502
pause
