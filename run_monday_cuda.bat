@echo off
setlocal
cd /d "%~dp0"
REM ============================================================================
REM Monday US Trading Production Launcher (with CUDA / Native Python fallback)
REM ============================================================================

REM Optional WSL override: if USE_WSL=1, delegate into WSL
if "%USE_WSL%"=="1" (
    wsl bash -c "cd $(wslpath '%~dp0') && PYTHONPATH=. python3 -m trading.cli %*"
    exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -m trading.cli %*
) else (
    py -3 -m trading.cli %*
)
exit /b %ERRORLEVEL%
