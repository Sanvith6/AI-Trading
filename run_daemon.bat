@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM US Trading Continuous Daemon
REM Continuously loops during market hours, tracks stop-losses, and rebalances portfolio
REM Default interval: 300 seconds (5 minutes)
REM ============================================================================

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m trading.cli daemon %*
) else (
    py -3 -m trading.cli daemon %*
)
exit /b %ERRORLEVEL%
