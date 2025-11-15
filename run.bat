@echo off
chcp 65001 >nul
echo ============================================================
echo Sentiment Analysis Pipeline - Quick Start
echo ============================================================
echo.

echo [*] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo [*] Creating directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "saved_models" mkdir saved_models
if not exist "artifacts" mkdir artifacts
if not exist "plots" mkdir plots
if not exist "mlruns" mkdir mlruns
echo [OK] Directories created
echo.

echo [*] Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo ============================================================
echo Running Pipeline (WITHOUT DVC)
echo ============================================================
echo.

python main.py --no-dvc

echo.
echo ============================================================
echo Pipeline Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. View results: mlflow ui
echo 2. Open browser: http://localhost:5000
echo.
pause