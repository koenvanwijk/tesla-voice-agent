import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import time
import uuid
import wave
from collections import defaultdict, deque
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
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

# --- Agent backends -------------------------------------------------------
# Besides the raw LLM, a voice turn can be routed to a coding-agent CLI
# (Claude Code or OpenClaw). Each agent returns a full text reply which is
# then streamed to Piper TTS just like an LLM answer.
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "120"))
CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "").strip()
OPENCLAW_BIN = os.getenv("OPENCLAW_BIN", "openclaw").strip()
# OpenClaw needs a newer Node than some system defaults; this bin dir (if set,
# else the newest ~/.nvm node >= v24) is prepended to PATH for agent calls.
AGENT_NODE_BIN_DIR = os.getenv("AGENT_NODE_BIN_DIR", "").strip()

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
_piper_voices: dict = {}
_whisper_lock = asyncio.Lock()
_piper_lock = asyncio.Lock()
_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_MESSAGES))
# Per voice-session continuity for the Claude agent: voice session id -> the
# Claude Code session id learned from the last reply (so the next turn resumes).
_claude_threads: dict[str, str] = {}


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


_VOICE_LABELS = {
    "nl_NL-pim-medium": "Pim — Nederlands (medium)",
    "nl_NL-ronnie-medium": "Ronnie — Nederlands (medium)",
}


def _discover_voices() -> dict[str, dict]:
    """Available TTS voices keyed by id. Piper voices are any .onnx (+json) in
    PIPER_VOICE_DIR; future engines (e.g. XTTS on the DGX) register here too."""
    voices: dict[str, dict] = {}
    if PIPER_VOICE_DIR.exists():
        for onnx in sorted(PIPER_VOICE_DIR.glob("*.onnx")):
            vid = onnx.stem
            if (PIPER_VOICE_DIR / f"{vid}.onnx.json").exists():
                voices[vid] = {"engine": "piper", "label": _VOICE_LABELS.get(vid, vid)}
    return voices


VOICES = _discover_voices()
DEFAULT_VOICE = PIPER_VOICE if PIPER_VOICE in VOICES else next(iter(VOICES), PIPER_VOICE)


def resolve_voice(requested: str | None) -> str:
    v = (requested or "").strip()
    return v if v in VOICES else DEFAULT_VOICE


def get_piper_voice(voice_id: str) -> PiperVoice:
    if voice_id not in _piper_voices:
        model_path = PIPER_VOICE_DIR / f"{voice_id}.onnx"
        config_path = PIPER_VOICE_DIR / f"{voice_id}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            raise RuntimeError(f"Piper-stem '{voice_id}' ontbreekt in {PIPER_VOICE_DIR}.")
        _piper_voices[voice_id] = PiperVoice.load(
            model_path, config_path=config_path, use_cuda=PIPER_USE_CUDA
        )
    return _piper_voices[voice_id]


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


def synthesize_wav(text: str, voice_id: str | None = None) -> bytes:
    voice = get_piper_voice(resolve_voice(voice_id))
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


async def resolve_turn_input(audio: UploadFile | None, text: str) -> tuple[str, int]:
    """Turn input from either a voice upload (STT) or typed text (stt_ms=0)."""
    if audio is not None and audio.filename:
        return await transcribe_upload(audio)
    return (text or "").strip(), 0


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


# --- Agent backends -------------------------------------------------------

def _detect_node_bin_dir() -> str:
    """Newest ~/.nvm node >= v24 bin dir, or AGENT_NODE_BIN_DIR if set."""
    if AGENT_NODE_BIN_DIR:
        return AGENT_NODE_BIN_DIR
    best: tuple[int, ...] | None = None
    best_dir = ""
    for bin_dir in Path.home().glob(".nvm/versions/node/v*/bin"):
        if not (bin_dir / "node").exists():
            continue
        try:
            parts = tuple(int(p) for p in bin_dir.parent.name[1:].split("."))
        except ValueError:
            continue
        if parts[0] >= 24 and (best is None or parts > best):
            best, best_dir = parts, str(bin_dir)
    return best_dir


NODE_BIN_DIR = _detect_node_bin_dir()

AGENT_REGISTRY: dict[str, dict] = {
    "llm": {"label": f"LLM ({LLM_MODEL})", "bin": None},
    "claude": {"label": "Claude", "bin": CLAUDE_BIN},
    "openclaw": {"label": "OpenClaw", "bin": OPENCLAW_BIN},
}


def _agent_path() -> str:
    path = os.environ.get("PATH", "")
    if NODE_BIN_DIR and NODE_BIN_DIR not in path.split(os.pathsep):
        path = NODE_BIN_DIR + os.pathsep + path
    return path


def agent_available(agent_id: str) -> bool:
    spec = AGENT_REGISTRY.get(agent_id)
    if not spec:
        return False
    if spec["bin"] is None:  # the built-in LLM is always available
        return True
    return shutil.which(spec["bin"], path=_agent_path()) is not None


def resolve_agent(requested: str | None) -> str:
    a = (requested or "").strip().lower()
    return a if a in AGENT_REGISTRY and agent_available(a) else "llm"


def available_agents() -> list[dict]:
    return [
        {"id": aid, "label": spec["label"], "available": agent_available(aid)}
        for aid, spec in AGENT_REGISTRY.items()
    ]


def _deep_find(obj, keys: tuple[str, ...]) -> str | None:
    """Depth-first search for the first non-empty string under any of `keys`."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k in keys:
                v = cur.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _claude_session_cwd(claude_session_id: str) -> str:
    """The cwd a Claude session was recorded under, so --resume finds it."""
    for jsonl in Path.home().glob(f".claude/projects/*/{claude_session_id}.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    rec = json.loads(line)
                    cwd = rec.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
        except (OSError, json.JSONDecodeError):
            continue
        break
    return str(Path.home())


def _build_agent_call(
    agent_id: str, session_id: str, user_text: str, conversation: str
) -> tuple[list[str], str, str]:
    """Return (argv, cwd, conversation_token) for a coding-agent turn."""
    conversation = (conversation or "").strip()
    if agent_id == "claude":
        cmd = [CLAUDE_BIN, "-p", user_text, "--output-format", "json",
               "--append-system-prompt", SYSTEM_PROMPT]
        if CLAUDE_MODEL:
            cmd += ["--model", CLAUDE_MODEL]
        resume_id = conversation or _claude_threads.get(session_id, "")
        cwd = str(Path.home())
        if resume_id:
            cmd += ["--resume", resume_id]
            cwd = _claude_session_cwd(resume_id)
        return cmd, cwd, resume_id
    if agent_id == "openclaw":
        cmd = [OPENCLAW_BIN, "agent", "-m", user_text, "--json"]
        if conversation.startswith("discord:"):
            target = conversation.split(":", 1)[1]
            # Use the channel's conversation context; do not --deliver (no post).
            cmd += ["--channel", "discord", "--to", target]
            token = conversation
        elif conversation.startswith("session:"):
            sid = conversation.split(":", 1)[1] or f"tva-{session_id}"
            cmd += ["--session-id", sid[:128]]
            token = f"session:{sid}"
        elif conversation:
            cmd += ["--session-id", conversation[:128]]
            token = f"session:{conversation}"
        else:
            sid = f"tva-{session_id}"
            cmd += ["--session-id", sid[:128]]
            token = f"session:{sid}"
        return cmd, str(Path.home()), token
    raise HTTPException(status_code=400, detail=f"Unknown agent '{agent_id}'")


def _parse_agent_reply(agent_id: str, stdout: str) -> tuple[str, str | None]:
    """Return (reply_text, conversation_id_from_output)."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        text = stdout.strip()  # some CLIs print plain text instead of JSON
        if text:
            return text, None
        raise HTTPException(status_code=502, detail=f"{agent_id} returned no output")
    if agent_id == "claude":
        reply = (data.get("result") if isinstance(data, dict) else "") or ""
        conv = data.get("session_id") if isinstance(data, dict) else None
    else:  # openclaw
        reply = _deep_find(data, ("finalAssistantVisibleText", "finalAssistantRawText")) or ""
        conv = None
    reply = reply.strip()
    if not reply:
        raise HTTPException(status_code=502, detail=f"{agent_id} returned an empty reply")
    return reply, conv


async def run_cli_agent(
    agent_id: str, session_id: str, user_text: str, conversation: str = ""
) -> tuple[str, str]:
    """Run one agent turn. Returns (reply, conversation_token) for continuity."""
    cmd, cwd, token = _build_agent_call(agent_id, session_id, user_text, conversation)
    env = {**os.environ, "PATH": _agent_path()}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=AGENT_TIMEOUT
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=502, detail=f"Agent '{agent_id}' not installed: {exc}") from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Agent '{agent_id}' timed out after {AGENT_TIMEOUT:.0f}s") from exc

    if proc.returncode != 0:
        err = (stderr_b or stdout_b).decode("utf-8", errors="replace").strip()[:300]
        raise HTTPException(status_code=502, detail=f"Agent '{agent_id}' failed ({proc.returncode}): {err}")

    reply, out_conv = _parse_agent_reply(agent_id, stdout_b.decode("utf-8", errors="replace"))
    if agent_id == "claude":
        token = out_conv or token
        if token:
            _claude_threads[session_id] = token  # remember for the next turn
    remember_turn(session_id, user_text, reply)
    return reply, token


async def _run_capture(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    env = {**os.environ, "PATH": _agent_path()}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env, cwd=str(Path.home()),
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (FileNotFoundError, asyncio.TimeoutError):
        return 124, "", "unavailable"
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _claude_session_label(jsonl: Path) -> str:
    try:
        with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 40:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("message")
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = " ".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        text = ""
                    text = " ".join(text.split()).strip()
                    if text:
                        return text[:80]
    except OSError:
        pass
    return ""


def list_claude_sessions(limit: int = 25) -> list[dict]:
    root = Path.home() / ".claude" / "projects"
    if not root.exists():
        return []
    files = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for jsonl in files[:limit]:
        out.append({
            "id": jsonl.stem,
            "label": _claude_session_label(jsonl) or jsonl.stem[:8],
            "kind": "session",
            "updated": int(jsonl.stat().st_mtime * 1000),
        })
    return out


async def list_openclaw_conversations(limit: int = 25) -> list[dict]:
    out: list[dict] = []
    rc, so, _ = await _run_capture(
        [OPENCLAW_BIN, "sessions", "--json", "--limit", str(limit)]
    )
    if rc == 0:
        try:
            for s in (json.loads(so) or {}).get("sessions", []):
                sid = s.get("sessionId") or s.get("key")
                if not sid:
                    continue
                label = " · ".join(x for x in (s.get("model"), s.get("key")) if x)
                out.append({
                    "id": f"session:{sid}",
                    "label": label[:80] or str(sid),
                    "kind": "session",
                    "updated": int(s.get("updatedAt") or 0),
                })
        except json.JSONDecodeError:
            pass
    rc, so, _ = await _run_capture(
        [OPENCLAW_BIN, "directory", "groups", "list", "--channel", "discord", "--json"]
    )
    if rc == 0:
        try:
            groups = json.loads(so)
            for g in groups if isinstance(groups, list) else []:
                raw = g.get("raw") or {}
                if raw.get("type") != 0:  # only text channels are messageable
                    continue
                cid = raw.get("id") or g.get("id", "").split(":", 1)[-1]
                out.append({
                    "id": f"discord:{cid}",
                    "label": g.get("handle") or ("#" + str(g.get("name", cid))),
                    "kind": "discord",
                    "updated": 0,
                })
        except json.JSONDecodeError:
            pass
    return out


async def agent_event_stream(agent_id, session_id, transcript, voice_id, request_started, stt_ms, conversation=""):
    """NDJSON events for a coding-agent turn: full reply, then progressive TTS."""
    agent_started = time.perf_counter()
    try:
        reply, conversation = await run_cli_agent(agent_id, session_id, transcript, conversation)
    except HTTPException as exc:
        yield ndjson_event({"type": "error", "message": str(exc.detail)})
        return
    agent_ms = round((time.perf_counter() - agent_started) * 1000)
    yield ndjson_event({"type": "text", "delta": reply})

    tts_ms_total = 0
    first_audio_ms = 0

    async def speak(sentence: str) -> str:
        nonlocal tts_ms_total, first_audio_ms
        tts_started = time.perf_counter()
        async with _piper_lock:
            wav_bytes = await asyncio.to_thread(synthesize_wav, sentence, voice_id)
        chunk_ms = round((time.perf_counter() - tts_started) * 1000)
        tts_ms_total += chunk_ms
        if first_audio_ms == 0:
            first_audio_ms = round((time.perf_counter() - request_started) * 1000)
        return ndjson_event({
            "type": "audio",
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "audio_mime": "audio/wav",
            "tts_ms": chunk_ms,
        })

    buffer = reply
    while True:
        sentence, buffer = pop_speakable_chunk(buffer)
        if not sentence:
            break
        yield await speak(sentence)
    sentence, _ = pop_speakable_chunk(buffer, force=True)
    if sentence:
        yield await speak(sentence)

    yield ndjson_event({
        "type": "done",
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "conversation": conversation,
        "stt_ms": stt_ms,
        "llm_ms": agent_ms,
        "tts_ms": tts_ms_total,
        "first_audio_ms": first_audio_ms,
        "total_ms": round((time.perf_counter() - request_started) * 1000),
    })


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
        "voices": [{"id": vid, "label": spec["label"], "engine": spec["engine"]} for vid, spec in VOICES.items()],
        "voice_default": DEFAULT_VOICE,
        "agents": available_agents(),
        "agent_default": "llm",
        "piper_ready": piper_model.exists() and piper_config.exists(),
        "piper_cuda": PIPER_USE_CUDA,
        "streaming_tts": True,
        "token_required": bool(TOKEN),
    }


@app.get("/voices")
async def voices():
    return {
        "default": DEFAULT_VOICE,
        "voices": [
            {"id": vid, "label": spec["label"], "engine": spec["engine"]}
            for vid, spec in VOICES.items()
        ],
    }


@app.get("/agents")
async def agents():
    return {"default": "llm", "agents": available_agents()}


@app.get("/conversations")
async def conversations(agent: str = "llm"):
    agent_id = resolve_agent(agent)
    if agent_id == "claude":
        return {"agent": "claude", "conversations": list_claude_sessions()}
    if agent_id == "openclaw":
        return {"agent": "openclaw", "conversations": await list_openclaw_conversations()}
    return {"agent": agent_id, "conversations": []}


@app.post("/api/stream-turn")
async def stream_turn(
    audio: UploadFile | None = File(default=None),
    voice: str = Form(default=""),
    agent: str = Form(default=""),
    conversation: str = Form(default=""),
    text: str = Form(default=""),
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    require_token(authorization)
    request_started = time.perf_counter()
    session_id = (x_session_id or str(uuid.uuid4()))[:128]
    voice_id = resolve_voice(voice)
    agent_id = resolve_agent(agent)
    transcript, stt_ms = await resolve_turn_input(audio, text)

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

        if agent_id != "llm":
            async for ev in agent_event_stream(
                agent_id, session_id, transcript, voice_id, request_started, stt_ms, conversation
            ):
                yield ev
            return

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
                                wav_bytes = await asyncio.to_thread(synthesize_wav, sentence, voice_id)
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
                    wav_bytes = await asyncio.to_thread(synthesize_wav, sentence, voice_id)
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
    audio: UploadFile | None = File(default=None),
    voice: str = Form(default=""),
    agent: str = Form(default=""),
    conversation: str = Form(default=""),
    text: str = Form(default=""),
    authorization: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    # Non-streaming compatibility endpoint.
    require_token(authorization)
    started = time.perf_counter()
    session_id = (x_session_id or str(uuid.uuid4()))[:128]
    voice_id = resolve_voice(voice)
    agent_id = resolve_agent(agent)
    transcript, stt_ms = await resolve_turn_input(audio, text)

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
    conversation_out = ""
    if agent_id == "llm":
        reply = await ask_ollama(session_id, transcript)
    else:
        reply, conversation_out = await run_cli_agent(agent_id, session_id, transcript, conversation)
    llm_ms = round((time.perf_counter() - t1) * 1000)

    try:
        t2 = time.perf_counter()
        async with _piper_lock:
            wav_bytes = await asyncio.to_thread(synthesize_wav, reply, voice_id)
        tts_ms = round((time.perf_counter() - t2) * 1000)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Piper TTS failed: {exc}") from exc

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply,
        "conversation": conversation_out,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "audio_mime": "audio/wav",
        "stt_ms": stt_ms,
        "llm_ms": llm_ms,
        "tts_ms": tts_ms,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


if SERVE_FRONTEND and FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
