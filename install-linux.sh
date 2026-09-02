#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/koenvanwijk/tesla-voice-agent.git"
DEFAULT_INSTALL_DIR="$HOME/tesla-voice-agent"
INSTALL_DIR="${TESLA_VOICE_AGENT_DIR:-$DEFAULT_INSTALL_DIR}"
SERVICE_NAME="tesla-voice-agent"
PORT="${TESLA_VOICE_AGENT_PORT:-8787}"

info() { printf '\033[36m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" != "Linux" ]]; then
  die "This installer supports Linux only."
fi

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
elif [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=""
else
  die "sudo is required to install system packages and the systemd service."
fi

if ! command -v apt-get >/dev/null 2>&1; then
  die "This installer currently targets Ubuntu/Debian systems with apt-get."
fi

info "Installing Ubuntu/Debian prerequisites..."
$SUDO apt-get update
$SUDO apt-get install -y python3 python3-venv python3-pip ffmpeg curl git ca-certificates

if ! python3 - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  die "Python 3.10+ is required."
fi

if [[ -d "$PWD/.git" && -f "$PWD/backend/requirements.txt" && -f "$PWD/start-linux.sh" ]]; then
  REPO_DIR="$PWD"
  info "Using current checkout: $REPO_DIR"
  git -C "$REPO_DIR" pull --ff-only
else
  REPO_DIR="$INSTALL_DIR"
  if [[ -d "$REPO_DIR/.git" ]]; then
    info "Updating existing checkout: $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only
  elif [[ -e "$REPO_DIR" ]]; then
    die "$REPO_DIR already exists but is not a Git checkout. Set TESLA_VOICE_AGENT_DIR to another path."
  else
    info "Cloning Tesla Voice Agent to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

cd "$REPO_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if systemctl list-unit-files ollama.service >/dev/null 2>&1; then
  $SUDO systemctl enable --now ollama.service
elif ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  warn "No ollama.service found; starting Ollama for this boot."
  nohup ollama serve > /tmp/tesla-voice-agent-ollama.log 2>&1 &
fi

info "Preparing Python environment..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
  info "Created backend/.env from backend/.env.example"
fi

for _ in {1..40}; do
  if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || die "Ollama did not become reachable on 127.0.0.1:11434."

MODEL="$(grep -E '^OLLAMA_MODEL=' backend/.env | tail -n1 | cut -d= -f2- || true)"
MODEL="${MODEL:-qwen3:4b}"
info "Ensuring Ollama model '$MODEL' is available..."
ollama pull "$MODEL"

VOICE="$(grep -E '^PIPER_VOICE=' backend/.env | tail -n1 | cut -d= -f2- || true)"
VOICE="${VOICE:-nl_NL-pim-medium}"
VOICE_DIR="$(grep -E '^PIPER_VOICE_DIR=' backend/.env | tail -n1 | cut -d= -f2- || true)"
VOICE_DIR="${VOICE_DIR:-backend/voices}"
if [[ "$VOICE_DIR" != /* ]]; then
  VOICE_DIR="$REPO_DIR/$VOICE_DIR"
fi
mkdir -p "$VOICE_DIR"
info "Ensuring Piper voice '$VOICE' is available..."
.venv/bin/python -m piper.download_voices --data-dir "$VOICE_DIR" "$VOICE"

TARGET_USER="${SUDO_USER:-${USER:-$(id -un)}}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[[ -n "$TARGET_HOME" ]] || TARGET_HOME="$HOME"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

info "Installing systemd service $SERVICE_NAME..."
TMP_SERVICE="$(mktemp)"
cat > "$TMP_SERVICE" <<EOF
[Unit]
Description=Tesla Voice Agent local backend and UI
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$REPO_DIR
Environment=HOME=$TARGET_HOME
ExecStart=$VENV_PYTHON -m uvicorn server:app --app-dir backend --host 0.0.0.0 --port $PORT
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
$SUDO install -m 0644 "$TMP_SERVICE" "$SERVICE_PATH"
rm -f "$TMP_SERVICE"
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now "$SERVICE_NAME.service"

info "Waiting for Tesla Voice Agent health endpoint..."
HEALTH=""
for _ in {1..40}; do
  if HEALTH="$(curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)"; then
    break
  fi
  sleep 0.5
done

if [[ -z "$HEALTH" ]]; then
  $SUDO systemctl --no-pager --full status "$SERVICE_NAME.service" || true
  die "Service started but the health endpoint on port $PORT did not respond."
fi

printf '\n\033[32mTesla Voice Agent is installed and running.\033[0m\n'
printf 'Local URL: http://127.0.0.1:%s\n' "$PORT"
printf 'Health: %s\n' "$HEALTH"
printf 'Service: sudo systemctl status %s\n' "$SERVICE_NAME"
printf 'Logs: sudo journalctl -u %s -f\n' "$SERVICE_NAME"
printf '\nExpose port %s with NVIDIA Brev Connect/Tunnel and open the resulting HTTPS URL in the Tesla browser.\n' "$PORT"
