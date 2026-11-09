@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM Live US Trading Portfolio & Market Monitor
REM Displays real-time prices, unrealized P&L, stop-loss distance, and market clock
REM ============================================================================

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m trading.cli monitor %*
) else (
    py -3 -m trading.cli monitor %*
)
exit /b %ERRORLEVEL%
