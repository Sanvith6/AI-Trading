@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM Laya Specialist V2 vs Baseline Comparative Report Launcher
REM Displays side-by-side comparison of Laya shadow decisions vs baseline setups
REM ============================================================================

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m trading.cli laya-report %*
) else (
    py -3 -m trading.cli laya-report %*
)
exit /b %ERRORLEVEL%
