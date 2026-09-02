# Tesla Voice Agent

Tesla-first hands-free voice assistant with a GitHub Pages frontend and a fully local AI backend on your own laptop.

No OpenAI API or other cloud AI API is required.

## Architecture

Recommended setup:

`Tesla browser -> NVIDIA Brev Secure Link/Tunnel -> laptop:8787 -> faster-whisper -> Ollama -> browser speech synthesis`

- **Speech-to-text:** `faster-whisper`, local on the laptop
- **LLM:** Ollama, local on the laptop
- **Speech output:** Web Speech synthesis in the Tesla browser
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

The script will:

1. create `.venv`
2. install the Python dependencies
3. copy `backend/.env.example` to `backend/.env` on first run
4. start Ollama if necessary
5. pull the configured Ollama model (`qwen3:4b` by default)
6. start the UI + API at `http://127.0.0.1:8787`

The first run also downloads the configured Whisper model.

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
7. The laptop transcribes and answers locally; the Tesla browser speaks the result.

The client requests browser echo cancellation, noise suppression and auto gain control and adapts its voice threshold to ambient cabin noise.

## Configuration

Edit `backend/.env`:

```dotenv
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
VOICE_AGENT_TOKEN=
SERVE_FRONTEND=1
```

For an NVIDIA GPU with a compatible CTranslate2 installation you can experiment with:

```dotenv
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

The default CPU/int8 configuration is intentionally portable.

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

### Voice turn

```http
POST /api/turn
Content-Type: multipart/form-data
X-Session-ID: <conversation-id>
```

Form field: `audio` (`webm` or `ogg` from MediaRecorder).

Example response:

```json
{
  "session_id": "...",
  "transcript": "Wat staat er vandaag op de planning?",
  "reply": "...",
  "stt_ms": 530,
  "llm_ms": 410,
  "total_ms": 945
}
```

## Why this stack?

This first version deliberately uses a pipeline rather than an end-to-end speech model:

- no AI cloud dependency
- easy to debug
- models are replaceable independently
- Ollama gives many local model choices
- faster-whisper is reliable for Dutch
- Brev gives a secure route to the laptop

Later upgrades can replace browser TTS with Piper/Kokoro, add streaming STT/TTS, or switch Ollama to a local vLLM/llama.cpp/OpenAI-compatible endpoint without changing the Tesla UI substantially.
