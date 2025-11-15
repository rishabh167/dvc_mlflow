# PowerShell script for Sentiment Analysis Pipeline
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Sentiment Analysis Pipeline - Quick Start" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[*] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create directories
Write-Host "[*] Creating directories..." -ForegroundColor Yellow
$dirs = @("data\raw", "data\processed", "saved_models", "artifacts", "plots", "mlruns")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "[OK] Directories created" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Dependencies installed" -ForegroundColor Green
Write-Host ""

# Run pipeline
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Running Pipeline (WITHOUT DVC)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

python main.py --no-dvc

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. View results: mlflow ui" -ForegroundColor White
Write-Host "2. Open browser: http://localhost:5000" -ForegroundColor White
Write-Host ""