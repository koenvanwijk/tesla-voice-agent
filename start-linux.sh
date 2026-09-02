#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

printf '\033[36mTesla Voice Agent - local Linux backend\033[0m\n'

PYTHON_BIN=""
for candidate in python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.10+ is required (3.11 recommended)."
  echo "On Ubuntu/Debian: sudo apt install python3.11 python3.11-venv python3-pip"
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating Python virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r backend/requirements.txt

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
  echo "Created backend/.env from example."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required for browser audio decoding."
  echo "Install it with: sudo apt install ffmpeg"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed or not on PATH."
  echo "Install it from https://ollama.com/download/linux and rerun this script."
  exit 1
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  nohup ollama serve > /tmp/tesla-voice-agent-ollama.log 2>&1 &
  for _ in {1..20}; do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

MODEL="$(grep -E '^OLLAMA_MODEL=' backend/.env | tail -n1 | cut -d= -f2- || true)"
MODEL="${MODEL:-qwen3:4b}"
echo "Ensuring Ollama model '$MODEL' is available..."
ollama pull "$MODEL"

VOICE="$(grep -E '^PIPER_VOICE=' backend/.env | tail -n1 | cut -d= -f2- || true)"
VOICE="${VOICE:-nl_NL-pim-medium}"
VOICE_DIR="$(grep -E '^PIPER_VOICE_DIR=' backend/.env | tail -n1 | cut -d= -f2- || true)"
VOICE_DIR="${VOICE_DIR:-backend/voices}"
mkdir -p "$VOICE_DIR"

echo "Ensuring Piper voice '$VOICE' is available..."
"$VENV_PYTHON" -m piper.download_voices --data-dir "$VOICE_DIR" "$VOICE"

cat <<EOF

Backend + UI starting on http://127.0.0.1:8787
Expose/tunnel port 8787 in NVIDIA Brev and open that HTTPS URL in the Tesla browser.
Stop with Ctrl+C.
EOF

exec "$VENV_PYTHON" -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port 8787
