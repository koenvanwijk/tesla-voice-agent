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

$EnvFile = Get-Content "backend\.env" -ErrorAction SilentlyContinue

# Download the configured Piper voice if necessary.
$Voice = "nl_NL-pim-medium"
$VoiceLine = $EnvFile | Where-Object { $_ -match '^PIPER_VOICE=' } | Select-Object -First 1
if ($VoiceLine) { $Voice = ($VoiceLine -replace '^PIPER_VOICE=', '').Trim() }

$VoiceDirValue = "backend/voices"
$VoiceDirLine = $EnvFile | Where-Object { $_ -match '^PIPER_VOICE_DIR=' } | Select-Object -First 1
if ($VoiceDirLine) { $VoiceDirValue = ($VoiceDirLine -replace '^PIPER_VOICE_DIR=', '').Trim() }

if ([System.IO.Path]::IsPathRooted($VoiceDirValue)) {
  $VoiceDir = $VoiceDirValue
} else {
  $VoiceDir = Join-Path $RepoRoot $VoiceDirValue
}

New-Item -ItemType Directory -Path $VoiceDir -Force | Out-Null
$VoiceModel = Join-Path $VoiceDir "$Voice.onnx"
$VoiceConfig = Join-Path $VoiceDir "$Voice.onnx.json"

if (-not (Test-Path $VoiceModel) -or -not (Test-Path $VoiceConfig)) {
  Write-Host "Downloading Dutch Piper voice '$Voice'..."
  & $VenvPython -m piper.download_voices --download-dir $VoiceDir $Voice
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to download Piper voice '$Voice'." -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host "Piper voice '$Voice' is ready."
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
$ModelLine = $EnvFile | Where-Object { $_ -match '^OLLAMA_MODEL=' } | Select-Object -First 1
if ($ModelLine) { $Model = ($ModelLine -replace '^OLLAMA_MODEL=', '').Trim() }

Write-Host "Ensuring Ollama model '$Model' is available..."
& ollama pull $Model

Write-Host ""
Write-Host "Backend + UI starting on http://127.0.0.1:8787" -ForegroundColor Green
Write-Host "Local voice: Piper $Voice" -ForegroundColor Green
Write-Host "In NVIDIA Brev expose/tunnel port 8787 and open that HTTPS URL in the Tesla browser." -ForegroundColor Green
Write-Host "Stop with Ctrl+C." -ForegroundColor DarkGray

& $VenvPython -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8787
