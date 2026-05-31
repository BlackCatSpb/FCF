@echo off
title EVA Production Server
cd /d "%~dp0"

:: ============================================================
:: EVA — Production Launch Script
:: ============================================================
:: Usage: eva_api.bat [--port 8000] [--host 0.0.0.0] [--reload]
:: ============================================================

set PORT=8000
set HOST=0.0.0.0
set RELOAD=

:parse
if "%1"=="" goto :run
if "%1"=="--port" set PORT=%2& shift & shift & goto :parse
if "%1"=="--host" set HOST=%2& shift & shift & goto :parse
if "%1"=="--reload" set RELOAD=--reload& shift & goto :parse
if "%1"=="--help" goto :help
shift
goto :parse

:help
echo EVA Production Server
echo.
echo Usage: %0 [--port PORT] [--host HOST] [--reload]
echo.
echo Options:
echo   --port PORT    Port to listen on (default: 8000)
echo   --host HOST    Host to bind to (default: 0.0.0.0)
echo   --reload       Auto-reload on code changes (dev only)
echo   --help         Show this help
exit /b 0

:run
echo [EVA] Starting production server on %HOST%:%PORT% ...
echo [EVA] GPU: %CUDA_VISIBLE_DEVICES%
python -W ignore -m uvicorn eva_api:app --host %HOST% --port %PORT% %RELOAD% --log-level info --timeout-keep-alive 120
if %errorlevel% neq 0 (
    echo [EVA] Server crashed with code %errorlevel%
    echo [EVA] Restarting in 5 seconds...
    timeout /t 5 /nobreak >nul
    goto :run
)
