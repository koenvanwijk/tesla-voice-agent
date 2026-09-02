import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
import wave
from collections import defaultdict, deque
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
TOKEN = os.getenv("VOICE_AGENT_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
PIPER_VOICE = os.getenv("PIPER_VOICE", "nl_NL-pim-medium").strip()
_piper_voice_dir = Path(os.getenv("PIPER_VOICE_DIR", "backend/voices"))
PIPER_VOICE_DIR = _piper_voice_dir if _piper_voice_dir.is_absolute() else ROOT / _piper_voice_dir
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "0.95"))
PIPER_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
PIPER_NOISE_W_SCALE = float(os.getenv("PIPER_NOISE_W_SCALE", "0.8"))
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "0") == "1"
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(12 * 1024 * 1024)))
SERVE_FRONTEND = os.getenv("SERVE_FRONTEND", "1") != "0"

ALLOWED_ORIGINS = [
    value.strip()
    for value in os.getenv(
        "ALLOWED_ORIGINS",
        "https://koenvanwijk.github.io,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if value.strip()
]

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Je bent een snelle handsfree assistent in een auto. "
    "Antwoord standaard in het Nederlands. Wees beknopt en spreekbaar: meestal 1 tot 4 korte zinnen. "
    "Gebruik geen markdown-tabellen. Lees lange URLs, code en lange opsommingen niet onnodig voor. "
    "Als iets onzeker is, zeg dat kort en verzin geen actuele informatie.",
)

app = FastAPI(title="Tesla Voice Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Session-ID"],
)

_whisper_model = None
_piper_voice = None
_whisper_lock = asyncio.Lock()
_piper_lock = asyncio.Lock()
_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))


def require_token(authorization: str | None) -> None:
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid voice agent token")


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def get_piper_voice() -> PiperVoice:
    global _piper_voice
    if _piper_voice is None:
        model_path = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx"
        config_path = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            raise RuntimeError(
                f"Piper voice '{PIPER_VOICE}' is missing in {PIPER_VOICE_DIR}. "
                "Run start-windows.ps1 again to download it."
            )
        _piper_voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=PIPER_USE_CUDA)
    return _piper_voice


def transcribe_file(path: str) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(
        path,
        language="nl",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def synthesize_wav(text: str) -> bytes:
    voice = get_piper_voice()
    syn_config = SynthesisConfig(
        length_scale=PIPER_LENGTH_SCALE,
        noise_scale=PIPER_NOISE_SCALE,
        noise_w_scale=PIPER_NOISE_W_SCALE,
    )

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        configured = False
        for chunk in voice.synthesize(text, syn_config):
            if not configured:
                wav_file.setframerate(chunk.sample_rate)
                wav_file.setsampwidth(chunk.sample_width)
                wav_file.setnchannels(chunk.sample_channels)
                configured = True
            wav_file.writeframes(chunk.audio_int16_bytes)

        if not configured:
            raise RuntimeError("Piper produced no audio")

    return wav_io.getvalue()


async def ask_ollama(session_id: str, user_text: str) -> str:
    history = list(_histories[session_id])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.35, "num_predict": 220},
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach Ollama: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {response.status_code}: {response.text[:300]}",
        )

    data = response.json()
    reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Ollama returned an empty reply")

    _histories[session_id].append({"role": "user", "content": user_text})
    _histories[session_id].append({"role": "assistant", "content": reply})
    return reply


@app.get("/health")
async def health():
    ollama_ok = False
    installed_models = []
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
        if response.is_success:
            ollama_ok = True
            installed_models = [m.get("name") for m in response.json().get("models", [])]
    except Exception:
        pass

    piper_model = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx"
    piper_config = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx.json"

    return {
        "ok": True,
        "ollama": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "model_installed": any(
            name == OLLAMA_MODEL or (name or "").startswith(f"{OLLAMA_MODEL}:")
            for name in installed_models
        ),
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "piper_voice": PIPER_VOICE,
        "piper_ready": piper_model.exists() and piper_config.exists(),
        "piper_cuda": PIPER_USE_CUDA,
        "token_required": bool(TOKEN),
    }


@app.post("/api/turn")
async def turn(
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    require_token(authorization)
    started = time.perf_counter()
    session_id = (x_session_id or str(uuid.uuid4()))[:128]

    raw = await audio.read(MAX_AUDIO_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload too large")

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = Path(audio.filename).suffix[:12] or suffix

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        async with _whisper_lock:
            transcript = await asyncio.to_thread(transcribe_file, tmp_path)
        stt_ms = round((time.perf_counter() - t0) * 1000)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not transcript:
        return {
            "session_id": session_id,
            "transcript": "",
            "reply": "",
            "audio_b64": "",
            "audio_mime": "audio/wav",
            "stt_ms": stt_ms,
            "llm_ms": 0,
            "tts_ms": 0,
            "total_ms": round((time.perf_counter() - started) * 1000),
        }

    t1 = time.perf_counter()
    reply = await ask_ollama(session_id, transcript)
    llm_ms = round((time.perf_counter() - t1) * 1000)

    try:
        t2 = time.perf_counter()
        async with _piper_lock:
            wav_bytes = await asyncio.to_thread(synthesize_wav, reply)
        tts_ms = round((time.perf_counter() - t2) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Piper TTS failed: {exc}") from exc

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "audio_mime": "audio/wav",
        "stt_ms": stt_ms,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


if SERVE_FRONTEND and FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
