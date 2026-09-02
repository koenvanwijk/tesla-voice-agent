$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "Tesla Voice Agent - local backend" -ForegroundColor Cyan

# Prefer Python 3.11 because faster-whisper/CTranslate2 is well supported there.
$Python = $null
try {
  & py -3.11 --version *> $null
  if ($LASTEXITCODE -eq 0) { $Python = "py -3.11" }
} catch {}

if (-not $Python) {
  Write-Host "Python 3.11 is required. Install it from python.org, then run this script again." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path ".venv")) {
  Write-Host "Creating Python virtual environment..."
  Invoke-Expression "$Python -m venv .venv"
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r "backend\requirements.txt"

if (-not (Test-Path "backend\.env")) {
  Copy-Item "backend\.env.example" "backend\.env"
  Write-Host "Created backend\.env from example." -ForegroundColor Yellow
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Ollama is not installed or not on PATH. Install Ollama first, then rerun this script." -ForegroundColor Red
  Write-Host "https://ollama.com/download/windows"
  exit 1
}

$OllamaUp = $false
try {
  Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
  $OllamaUp = $true
} catch {}

if (-not $OllamaUp) {
  Write-Host "Starting Ollama..."
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
  Start-Sleep -Seconds 3
}

$Model = "qwen3:4b"
$EnvFile = Get-Content "backend\.env" -ErrorAction SilentlyContinue
$ModelLine = $EnvFile | Where-Object { $_ -match '^OLLAMA_MODEL=' } | Select-Object -First 1
if ($ModelLine) { $Model = ($ModelLine -replace '^OLLAMA_MODEL=', '').Trim() }

Write-Host "Ensuring Ollama model '$Model' is available..."
& ollama pull $Model

Write-Host ""
Write-Host "Backend + UI starting on http://127.0.0.1:8787" -ForegroundColor Green
Write-Host "In NVIDIA Brev expose/tunnel port 8787 and open that HTTPS URL in the Tesla browser." -ForegroundColor Green
Write-Host "Stop with Ctrl+C." -ForegroundColor DarkGray

& $VenvPython -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8787
