param(
    [string]$PythonSelector = "3.11",
    [string]$EnvDir = ".venv311",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Invoke-CheckedCommand {
    param(
        [string]$Description,
        [scriptblock]$Command
    )

    Write-Host $Description -ForegroundColor Cyan
    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

try {
    & py "-$PythonSelector" --version | Out-Null
}
catch {
    Write-Host "Python $PythonSelector is not installed." -ForegroundColor Red
    Write-Host "Install Python 3.11.9 (Windows x64) first, then rerun this script." -ForegroundColor Yellow
    Write-Host "Official release page: https://www.python.org/downloads/release/python-3119/"
    exit 1
}

$envPath = Join-Path $projectRoot $EnvDir

if (Test-Path $envPath) {
    if (-not $ForceRecreate) {
        Write-Host "Environment '$EnvDir' already exists." -ForegroundColor Yellow
        Write-Host "Rerun with -ForceRecreate to rebuild it." -ForegroundColor Yellow
        exit 1
    }

    Remove-Item -LiteralPath $envPath -Recurse -Force
}

Invoke-CheckedCommand "Creating virtual environment in $EnvDir" {
    & py "-$PythonSelector" -m venv $EnvDir
}

$venvPython = Join-Path $projectRoot "$EnvDir\Scripts\python.exe"

Invoke-CheckedCommand "Upgrading packaging tools" {
    & $venvPython -m pip install --upgrade pip wheel "setuptools<81"
}

Invoke-CheckedCommand "Installing CPU-only PyTorch" {
    & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
}

Invoke-CheckedCommand "Installing computer vision dependencies" {
    & $venvPython -m pip install -r (Join-Path $projectRoot "requirements-py311.txt")
}

Invoke-CheckedCommand "Running import smoke test" {
    & $venvPython -c "import cv2, ultralytics, fer, mediapipe, transformers, torch, PIL, pandas, streamlit; print('All imports passed.')"
}

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate it with:" -ForegroundColor Green
Write-Host ".\$EnvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the pipeline with:" -ForegroundColor Green
Write-Host "python ai_pipeline.py"
