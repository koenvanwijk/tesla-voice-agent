# Tesla Voice Agent

Tesla-first hands-free voice assistant with a GitHub Pages frontend and a fully local AI backend on your own laptop.

No OpenAI API or other cloud AI API is required.

## Architecture

Recommended setup:

`Tesla browser -> NVIDIA Brev Secure Link/Tunnel -> Linux laptop:8787 -> faster-whisper -> Ollama streaming -> Piper TTS per sentence -> Tesla Web Audio queue`

- **Speech-to-text:** `faster-whisper`, local on the laptop
- **LLM:** Ollama, local on the laptop
- **Speech output:** Piper TTS, local on the laptop
- **Low-latency behavior:** Ollama tokens are grouped into speakable sentences; each completed sentence is synthesized and sent immediately instead of waiting for the whole answer
- **Playback:** queued Web Audio in the Tesla browser
- **Ingress/auth:** NVIDIA Brev Secure Link/Tunnel
- **Frontend:** served by the laptop for normal use; also deployable to GitHub Pages for testing/launcher use

Serving frontend and API from the same Brev URL avoids cross-origin authentication problems.

## Linux quick start

Designed primarily for Ubuntu/Debian-style Linux systems.

Prerequisites:

1. Python 3.10+ (3.11 recommended)
2. `python3-venv` / venv support
3. `ffmpeg`
4. Ollama
5. NVIDIA Brev access to this laptop / registered compute

Typical Ubuntu/Debian prerequisites:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg curl git
```

Install Ollama if it is not already installed, then clone and start:

```bash
git clone https://github.com/koenvanwijk/tesla-voice-agent.git
cd tesla-voice-agent
bash start-linux.sh
```

If you already cloned an earlier version:

```bash
cd tesla-voice-agent
git pull
bash start-linux.sh
```

The script will:

1. create `.venv`
2. install the Python dependencies, including Piper
3. copy `backend/.env.example` to `backend/.env` on first run
4. verify `ffmpeg`
5. start Ollama if necessary
6. pull the configured Ollama model (`qwen3:4b` by default)
7. download the configured Dutch Piper voice (`nl_NL-pim-medium` by default)
8. start the UI + API at `http://127.0.0.1:8787`

The first voice turn also downloads the configured Whisper model if it is not cached yet.

## Expose through NVIDIA Brev

Expose **port 8787** using the Secure Link/Tunnel functionality for your Brev-connected laptop.

Open the generated Brev HTTPS URL directly in the Tesla browser. This is the recommended production URL; do not expose the raw laptop port directly to the public internet.

## Tesla use

1. Open the Brev HTTPS URL.
2. Authenticate with Brev if prompted.
3. Press **START**.
4. Allow microphone access.
5. Talk normally.
6. Roughly 0.9 seconds of silence ends the turn automatically.
7. The laptop transcribes the turn locally.
8. Ollama starts generating the answer as a stream.
9. As soon as a complete sentence is available, Piper synthesizes that sentence locally and sends a WAV chunk to the Tesla.
10. The Tesla queues and plays those chunks while the backend continues producing later sentences.
11. When the response and audio queue are both finished, listening resumes automatically.

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

### NVIDIA GPU option

If the Linux laptop has a supported NVIDIA GPU and CTranslate2 CUDA support is working, you can switch Whisper to the GPU:

```dotenv
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

The default CPU/int8 Whisper + CPU Piper configuration is intentionally portable and leaves the GPU mostly available to Ollama.

### Optional second authentication layer

Brev authentication is sufficient for the recommended same-origin setup. If you want an additional application bearer token, set:

```dotenv
VOICE_AGENT_TOKEN=some-long-random-secret
```

Then enter the same token under **Instellingen** in the web UI.

## Windows fallback

A `start-windows.ps1` launcher remains in the repository, but Linux is now the primary supported host setup.

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

A healthy setup reports at least:

```json
{
  "ollama": true,
  "piper_ready": true,
  "streaming_tts": true
}
```

### Streaming voice turn

The Tesla UI uses:

```http
POST /api/stream-turn
Content-Type: multipart/form-data
X-Session-ID: <conversation-id>
```

Form field: `audio` (`webm` or `ogg` from MediaRecorder).

The response is `application/x-ndjson`. Events look like:

```json
{"type":"transcript","text":"Wat staat er vandaag op de planning?","stt_ms":530}
{"type":"text","delta":"Je hebt "}
{"type":"text","delta":"om tien uur een afspraak."}
{"type":"audio","audio_b64":"UklGR...","audio_mime":"audio/wav","tts_ms":110}
{"type":"done","reply":"Je hebt om tien uur een afspraak.","first_audio_ms":820,"total_ms":1150}
```

`first_audio_ms` is the useful latency metric for the in-car experience: time from the uploaded speech turn until the first synthesized audio chunk is ready.

### Non-streaming compatibility endpoint

`POST /api/turn` remains available and returns one complete base64-encoded WAV after the entire LLM answer has been generated.

## Why this stack?

- no AI cloud dependency
- Dutch STT and TTS both run locally
- first speech can start before the LLM has finished the whole answer
- easy to debug
- models are replaceable independently
- Ollama gives many local model choices
- faster-whisper is reliable for Dutch
- Piper has native Dutch (`nl_NL`) voices and is lightweight enough to run beside Ollama
- Brev gives a secure route to the laptop

A later upgrade can add true barge-in: keep listening while the agent speaks and immediately stop current audio/generation when you start talking.
