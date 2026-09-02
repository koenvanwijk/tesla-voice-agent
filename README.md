# Tesla Voice Agent

Tesla-first hands-free voice assistant with a GitHub Pages frontend and a fully local AI backend on your own laptop.

No OpenAI API or other cloud AI API is required.

## Architecture

Recommended setup:

`Tesla browser -> NVIDIA Brev Secure Link/Tunnel -> laptop:8787 -> faster-whisper -> Ollama -> Piper TTS -> Tesla Web Audio`

- **Speech-to-text:** `faster-whisper`, local on the laptop
- **LLM:** Ollama, local on the laptop
- **Speech output:** Piper TTS, local on the laptop
- **Playback:** Web Audio in the Tesla browser
- **Ingress/auth:** NVIDIA Brev Secure Link/Tunnel
- **Frontend:** served by the laptop for normal use; also deployed to GitHub Pages for testing/launcher use

Serving frontend and API from the same Brev URL avoids cross-origin authentication problems.

## Windows quick start

Prerequisites:

1. Python 3.11
2. Ollama
3. NVIDIA Brev access to this laptop / registered compute

Clone the repository and run:

```powershell
git clone https://github.com/koenvanwijk/tesla-voice-agent.git
cd tesla-voice-agent
powershell -ExecutionPolicy Bypass -File .\start-windows.ps1
```

If you already cloned an earlier version:

```powershell
git pull
powershell -ExecutionPolicy Bypass -File .\start-windows.ps1
```

The script will:

1. create `.venv`
2. install the Python dependencies, including Piper
3. copy `backend/.env.example` to `backend/.env` on first run
4. download the configured Dutch Piper voice (`nl_NL-pim-medium` by default)
5. start Ollama if necessary
6. pull the configured Ollama model (`qwen3:4b` by default)
7. start the UI + API at `http://127.0.0.1:8787`

The first voice turn also downloads the configured Whisper model if it is not cached yet.

## Expose through NVIDIA Brev

Expose **port 8787** using the Secure Link/Tunnel functionality for your Brev-connected machine.

In the Brev console, add port `8787` under the machine's access/tunnel section and copy the generated HTTPS URL. Brev's tunnel layer performs browser authentication before exposing the HTTP application.

Open that HTTPS URL directly in the Tesla browser. This is the recommended production URL; do not use the raw laptop IP on the public internet.

## Tesla use

1. Open the Brev HTTPS URL.
2. Authenticate with Brev if prompted.
3. Press **START**.
4. Allow microphone access.
5. Talk normally.
6. Roughly 0.9 seconds of silence ends the turn automatically.
7. The laptop transcribes, answers and synthesizes the Dutch voice locally.
8. The Tesla plays the returned WAV through its already-activated Web Audio context and then starts listening again.

The client requests browser echo cancellation, noise suppression and auto gain control and adapts its voice threshold to ambient cabin noise.

## Configuration

Edit `backend/.env`:

```dotenv
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b

WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

PIPER_VOICE=nl_NL-pim-medium
PIPER_VOICE_DIR=backend/voices
PIPER_LENGTH_SCALE=0.95
PIPER_NOISE_SCALE=0.667
PIPER_NOISE_W_SCALE=0.8
PIPER_USE_CUDA=0

VOICE_AGENT_TOKEN=
SERVE_FRONTEND=1
```

`PIPER_LENGTH_SCALE` controls speaking speed: lower is faster. Piper runs on CPU by default so Ollama can keep the GPU for the LLM.

For an NVIDIA GPU with a compatible CTranslate2 installation you can experiment with:

```dotenv
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

The default CPU/int8 Whisper + CPU Piper configuration is intentionally portable and avoids GPU contention with Ollama.

### Optional second authentication layer

Brev authentication is sufficient for the recommended same-origin setup. If you want an additional application bearer token, set:

```dotenv
VOICE_AGENT_TOKEN=some-long-random-secret
```

Then enter the same token under **Instellingen** in the web UI.

## GitHub Pages

The repository contains a GitHub Actions workflow that deploys `frontend/` to GitHub Pages.

Expected URL:

`https://koenvanwijk.github.io/tesla-voice-agent/`

GitHub Pages is useful as a standalone frontend or test page. In that mode, open **Instellingen** and enter the backend URL. A Brev-authenticated tunnel can be awkward for cross-origin `fetch` because authentication redirects/cookies are involved, so the direct Brev URL remains the preferred Tesla setup.

If GitHub Pages has not yet been enabled for the repository, go to **Settings -> Pages -> Build and deployment -> Source: GitHub Actions** once.

## API

### Health

```http
GET /health
```

The response includes Ollama, Whisper and Piper state. A healthy local voice setup reports `piper_ready: true`.

### Voice turn

```http
POST /api/turn
Content-Type: multipart/form-data
X-Session-ID: <conversation-id>
```

Form field: `audio` (`webm` or `ogg` from MediaRecorder).

Example response (audio shortened here):

```json
{
  "session_id": "...",
  "transcript": "Wat staat er vandaag op de planning?",
  "reply": "...",
  "audio_b64": "UklGR...",
  "audio_mime": "audio/wav",
  "stt_ms": 530,
  "llm_ms": 410,
  "tts_ms": 115,
  "total_ms": 1060
}
```

## Why this stack?

This version deliberately uses a modular local pipeline rather than a cloud or end-to-end speech model:

- no AI cloud dependency
- Dutch STT and TTS both run locally
- easy to debug
- models are replaceable independently
- Ollama gives many local model choices
- faster-whisper is reliable for Dutch
- Piper has native Dutch (`nl_NL`) voices and is lightweight enough to run beside Ollama
- Brev gives a secure route to the laptop

A next latency upgrade can stream LLM tokens into Piper sentence-by-sentence instead of waiting for the entire answer before speech synthesis starts.
