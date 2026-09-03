import asyncio
import base64
import io
import json
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
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
TOKEN = os.getenv("VOICE_AGENT_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
# Optional OpenAI-compatible backend (e.g. the DGX Spark deepseek endpoint).
# Set LLM_BACKEND=openai (or just OPENAI_BASE_URL) to route chat to /v1/chat/completions.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Extra JSON merged into the OpenAI chat payload, e.g. to disable Qwen thinking:
#   OPENAI_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}
OPENAI_EXTRA_BODY = os.getenv("OPENAI_EXTRA_BODY", "").strip()
USE_OPENAI = LLM_BACKEND in ("openai", "vllm", "llamacpp") or bool(OPENAI_BASE_URL)
LLM_MODEL = (OPENAI_MODEL or OLLAMA_MODEL) if USE_OPENAI else OLLAMA_MODEL
try:
    _EXTRA_BODY = json.loads(OPENAI_EXTRA_BODY) if OPENAI_EXTRA_BODY else {}
    if not isinstance(_EXTRA_BODY, dict):
        _EXTRA_BODY = {}
except json.JSONDecodeError:
    _EXTRA_BODY = {}
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


async def transcribe_upload(audio: UploadFile) -> tuple[str, int]:
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
        started = time.perf_counter()
        async with _whisper_lock:
            transcript = await asyncio.to_thread(transcribe_file, tmp_path)
        stt_ms = round((time.perf_counter() - started) * 1000)
        return transcript, stt_ms
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_messages(session_id: str, user_text: str) -> list[dict[str, str]]:
    history = list(_histories[session_id])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]


def remember_turn(session_id: str, user_text: str, reply: str) -> None:
    _histories[session_id].append({"role": "user", "content": user_text})
    _histories[session_id].append({"role": "assistant", "content": reply})


def _llm_chat_url() -> str:
    return f"{OPENAI_BASE_URL}/chat/completions" if USE_OPENAI else f"{OLLAMA_URL}/api/chat"


def _llm_models_url() -> str:
    return f"{OPENAI_BASE_URL}/models" if USE_OPENAI else f"{OLLAMA_URL}/api/tags"


def _llm_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"} if (USE_OPENAI and OPENAI_API_KEY) else {}


def _llm_payload(messages: list[dict[str, str]], stream: bool) -> dict:
    if USE_OPENAI:
        body = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": stream,
            "temperature": 0.35,
            "max_tokens": 220,
        }
        body.update(_EXTRA_BODY)
        return body
    return {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {"temperature": 0.35, "num_predict": 220},
    }


def _stream_delta(line: str) -> tuple[str, bool, str | None]:
    """Parse one streamed line into (delta_text, done, error).

    Handles Ollama JSONL ({"message":{"content":...},"done":...}) and OpenAI
    SSE ("data: {choices:[{delta:{content}}]}" ... "data: [DONE]").
    """
    if not line:
        return "", False, None
    if USE_OPENAI:
        if not line.startswith("data:"):
            return "", False, None
        body = line[len("data:"):].strip()
        if body == "[DONE]":
            return "", True, None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return "", False, None
        if data.get("error"):
            return "", False, str(data["error"])
        choice = (data.get("choices") or [{}])[0]
        delta = (choice.get("delta") or {}).get("content") or ""
        return delta, choice.get("finish_reason") is not None, None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return "", False, None
    if data.get("error"):
        return "", False, str(data["error"])
    delta = (data.get("message") or {}).get("content", "") or ""
    return delta, bool(data.get("done")), None


async def ask_ollama(session_id: str, user_text: str) -> str:
    payload = _llm_payload(build_messages(session_id, user_text), stream=False)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(_llm_chat_url(), json=payload, headers=_llm_headers())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach LLM: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned {response.status_code}: {response.text[:300]}",
        )

    data = response.json()
    if USE_OPENAI:
        reply = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    else:
        reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        raise HTTPException(status_code=502, detail="LLM returned an empty reply")

    remember_turn(session_id, user_text, reply)
    return reply


def pop_speakable_chunk(buffer: str, force: bool = False) -> tuple[str | None, str]:
    # Speak complete sentences immediately. For unusually long sentences, cut
    # near a comma/space so the first audio does not wait indefinitely.
    min_chars = 12
    for index, char in enumerate(buffer):
        if char in ".!?\n" and index + 1 >= min_chars:
            chunk = buffer[: index + 1].strip()
            rest = buffer[index + 1 :].lstrip()
            return (chunk or None), rest

    if len(buffer) >= 220:
        window = buffer[:220]
        cut = max(window.rfind(", "), window.rfind("; "), window.rfind(" "))
        if cut < 100:
            cut = 200
        chunk = buffer[:cut].strip()
        rest = buffer[cut:].lstrip()
        return (chunk or None), rest

    if force:
        chunk = buffer.strip()
        return (chunk or None), ""

    return None, buffer


def ndjson_event(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"


@app.get("/health")
async def health():
    ollama_ok = False
    installed_models = []
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(_llm_models_url(), headers=_llm_headers())
        if response.is_success:
            ollama_ok = True
            body = response.json()
            if USE_OPENAI:
                installed_models = [m.get("id") for m in body.get("data", [])]
            else:
                installed_models = [m.get("name") for m in body.get("models", [])]
    except Exception:
        pass

    piper_model = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx"
    piper_config = PIPER_VOICE_DIR / f"{PIPER_VOICE}.onnx.json"

    return {
        "ok": True,
        "ollama": ollama_ok,
        "llm_backend": "openai" if USE_OPENAI else "ollama",
        "llm_url": _llm_chat_url(),
        "ollama_model": LLM_MODEL,
        "model_installed": any(
            name == LLM_MODEL or (name or "").startswith(f"{LLM_MODEL}:")
            for name in installed_models
        ),
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "piper_voice": PIPER_VOICE,
        "piper_ready": piper_model.exists() and piper_config.exists(),
        "piper_cuda": PIPER_USE_CUDA,
        "streaming_tts": True,
        "token_required": bool(TOKEN),
    }


@app.post("/api/stream-turn")
async def stream_turn(
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    require_token(authorization)
    request_started = time.perf_counter()
    session_id = (x_session_id or str(uuid.uuid4()))[:128]
    transcript, stt_ms = await transcribe_upload(audio)

    if not transcript:
        async def empty_stream():
            yield ndjson_event({
                "type": "done",
                "session_id": session_id,
                "transcript": "",
                "reply": "",
                "stt_ms": stt_ms,
                "llm_ms": 0,
                "tts_ms": 0,
                "first_audio_ms": 0,
                "total_ms": round((time.perf_counter() - request_started) * 1000),
            })

        return StreamingResponse(
            empty_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    async def event_stream():
        reply_parts: list[str] = []
        speech_buffer = ""
        tts_ms_total = 0
        first_audio_ms = 0
        llm_started = time.perf_counter()

        yield ndjson_event({
            "type": "transcript",
            "session_id": session_id,
            "text": transcript,
            "stt_ms": stt_ms,
        })

        payload = _llm_payload(build_messages(session_id, transcript), stream=True)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
                async with client.stream("POST", _llm_chat_url(), json=payload, headers=_llm_headers()) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                        yield ndjson_event({
                            "type": "error",
                            "message": f"LLM returned {response.status_code}: {detail}",
                        })
                        return

                    async for line in response.aiter_lines():
                        delta, done, err = _stream_delta(line)
                        if err:
                            yield ndjson_event({"type": "error", "message": err})
                            return

                        if delta:
                            reply_parts.append(delta)
                            speech_buffer += delta
                            yield ndjson_event({"type": "text", "delta": delta})

                        while True:
                            sentence, speech_buffer = pop_speakable_chunk(speech_buffer)
                            if not sentence:
                                break
                            tts_started = time.perf_counter()
                            async with _piper_lock:
                                wav_bytes = await asyncio.to_thread(synthesize_wav, sentence)
                            chunk_tts_ms = round((time.perf_counter() - tts_started) * 1000)
                            tts_ms_total += chunk_tts_ms
                            if first_audio_ms == 0:
                                first_audio_ms = round((time.perf_counter() - request_started) * 1000)
                            yield ndjson_event({
                                "type": "audio",
                                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                                "audio_mime": "audio/wav",
                                "tts_ms": chunk_tts_ms,
                            })

                        if done:
                            break

            sentence, speech_buffer = pop_speakable_chunk(speech_buffer, force=True)
            if sentence:
                tts_started = time.perf_counter()
                async with _piper_lock:
                    wav_bytes = await asyncio.to_thread(synthesize_wav, sentence)
                chunk_tts_ms = round((time.perf_counter() - tts_started) * 1000)
                tts_ms_total += chunk_tts_ms
                if first_audio_ms == 0:
                    first_audio_ms = round((time.perf_counter() - request_started) * 1000)
                yield ndjson_event({
                    "type": "audio",
                    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    "audio_mime": "audio/wav",
                    "tts_ms": chunk_tts_ms,
                })

            reply = "".join(reply_parts).strip()
            if not reply:
                yield ndjson_event({"type": "error", "message": "Ollama returned an empty reply"})
                return

            remember_turn(session_id, transcript, reply)
            elapsed_since_llm = round((time.perf_counter() - llm_started) * 1000)
            llm_ms = max(0, elapsed_since_llm - tts_ms_total)

            yield ndjson_event({
                "type": "done",
                "session_id": session_id,
                "transcript": transcript,
                "reply": reply,
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms_total,
                "first_audio_ms": first_audio_ms,
                "total_ms": round((time.perf_counter() - request_started) * 1000),
            })
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            yield ndjson_event({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/turn")
async def turn(
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    # Non-streaming compatibility endpoint.
    require_token(authorization)
    started = time.perf_counter()
    session_id = (x_session_id or str(uuid.uuid4()))[:128]
    transcript, stt_ms = await transcribe_upload(audio)

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
