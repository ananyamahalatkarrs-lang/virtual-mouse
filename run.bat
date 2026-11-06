@echo off
title AI Virtual Mouse
cd /d "%~dp0"
echo ===================================================
echo             AI VIRTUAL MOUSE LAUNCHER
echo ===================================================
echo.
echo Starting webcam and gesture engine...
echo Press 'q' or 'ESC' in the window, or wave BYE to exit.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in %~dp0.venv
    echo Please ensure dependencies are installed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" virtual.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
