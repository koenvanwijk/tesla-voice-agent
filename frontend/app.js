(() => {
  const mainButton = document.getElementById('mainButton');
  const statusEl = document.getElementById('status');
  const indicator = document.getElementById('indicator');
  const meterFill = document.getElementById('meterFill');
  const conversation = document.getElementById('conversation');
  const backendUrlInput = document.getElementById('backendUrl');
  const backendTokenInput = document.getElementById('backendToken');
  const saveSettingsButton = document.getElementById('saveSettings');
  const testBackendButton = document.getElementById('testBackend');
  const healthResult = document.getElementById('healthResult');

  let stream = null;
  let audioContext = null;
  let analyser = null;
  let analyserData = null;
  let mediaRecorder = null;
  let currentAudioSource = null;
  let chunks = [];
  let audioQueue = [];
  let audioDrainRunning = false;
  let turnStreamDone = false;
  let running = false;
  let speaking = false;
  let processing = false;
  let speechStart = 0;
  let lastLoudAt = 0;
  let loudSince = 0;
  let noiseFloor = 0.012;
  let rafId = null;
  let sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());

  const cfg = {
    triggerMs: 160,
    silenceMs: 900,
    minUtteranceMs: 400,
    maxUtteranceMs: 15000,
  };

  function defaultBackendUrl() {
    if (!location.hostname.endsWith('github.io')) return location.origin;
    return localStorage.getItem('tva-backend-url') || '';
  }

  backendUrlInput.value = defaultBackendUrl();
  backendTokenInput.value = localStorage.getItem('tva-backend-token') || '';

  function setStatus(text, mode = 'idle') {
    statusEl.textContent = text;
    indicator.className = `indicator ${mode}`;
  }

  function addMessage(role, text, meta = '') {
    if (!text) return;
    const row = document.createElement('article');
    row.className = `message ${role}`;
    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = role === 'user' ? 'Jij' : 'Agent';
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = text;
    row.append(who, body);
    if (meta) {
      const small = document.createElement('div');
      small.className = 'meta';
      small.textContent = meta;
      row.append(small);
    }
    conversation.prepend(row);
  }

  function backendUrl(path) {
    const base = backendUrlInput.value.trim().replace(/\/$/, '');
    if (!base) throw new Error('Vul eerst een backend URL in bij Instellingen.');
    return `${base}${path}`;
  }

  function authHeaders() {
    const token = backendTokenInput.value.trim();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  function saveSettings() {
    localStorage.setItem('tva-backend-url', backendUrlInput.value.trim());
    localStorage.setItem('tva-backend-token', backendTokenInput.value.trim());
    healthResult.textContent = 'Opgeslagen.';
  }

  async function testBackend() {
    healthResult.textContent = 'Testen…';
    try {
      const response = await fetch(backendUrl('/health'), { headers: authHeaders() });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      healthResult.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      healthResult.textContent = `Fout: ${error.message}`;
    }
  }

  function rmsLevel() {
    analyser.getFloatTimeDomainData(analyserData);
    let sum = 0;
    for (let i = 0; i < analyserData.length; i += 1) {
      const v = analyserData[i];
      sum += v * v;
    }
    return Math.sqrt(sum / analyserData.length);
  }

  function recorderMimeType() {
    const options = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
    ];
    return options.find(type => MediaRecorder.isTypeSupported(type)) || '';
  }

  function beginRecording(now) {
    if (!running || processing || speaking || mediaRecorder) return;
    chunks = [];
    const mimeType = recorderMimeType();
    mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    mediaRecorder.ondataavailable = event => {
      if (event.data && event.data.size) chunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      const recorder = mediaRecorder;
      mediaRecorder = null;
      const blob = new Blob(chunks, { type: recorder?.mimeType || 'audio/webm' });
      chunks = [];
      if (!running) return;
      if (blob.size > 800) await submitTurn(blob);
      else resumeListening();
    };
    mediaRecorder.start(120);
    speechStart = now;
    lastLoudAt = now;
    loudSince = 0;
    setStatus('Luistert…', 'listening');
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  }

  function resumeListening() {
    processing = false;
    speaking = false;
    currentAudioSource = null;
    audioQueue = [];
    audioDrainRunning = false;
    turnStreamDone = false;
    speechStart = 0;
    lastLoudAt = 0;
    loudSince = 0;
    if (running) setStatus('Luistert', 'listening');
  }

  function base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  async function playAudioChunk(audioB64) {
    if (!audioContext) throw new Error('Audio-context is niet actief.');
    if (audioContext.state !== 'running') await audioContext.resume();

    const wavBuffer = base64ToArrayBuffer(audioB64);
    const decoded = await audioContext.decodeAudioData(wavBuffer.slice(0));

    await new Promise((resolve, reject) => {
      const source = audioContext.createBufferSource();
      currentAudioSource = source;
      source.buffer = decoded;
      source.connect(audioContext.destination);
      source.onended = () => {
        if (currentAudioSource === source) currentAudioSource = null;
        resolve();
      };
      try {
        source.start(0);
      } catch (error) {
        reject(error);
      }
    });
  }

  function finishTurnIfReady() {
    if (!processing || !turnStreamDone || audioDrainRunning || audioQueue.length) return;
    resumeListening();
  }

  async function drainAudioQueue() {
    if (audioDrainRunning || !processing) return;
    audioDrainRunning = true;

    try {
      while (audioQueue.length && processing && running) {
        speaking = true;
        setStatus('Praat…', 'speaking');
        const audioB64 = audioQueue.shift();
        await playAudioChunk(audioB64);
      }
    } catch (error) {
      failTurn(error);
      return;
    } finally {
      audioDrainRunning = false;
    }

    speaking = false;
    if (processing && !turnStreamDone) {
      setStatus('Denkt verder…', 'processing');
    }
    finishTurnIfReady();
  }

  function enqueueAudio(audioB64) {
    if (!audioB64 || !processing) return;
    audioQueue.push(audioB64);
    void drainAudioQueue();
  }

  function failTurn(error) {
    const message = error instanceof Error ? error.message : String(error);
    audioQueue = [];
    turnStreamDone = true;
    processing = false;
    speaking = false;

    if (currentAudioSource) {
      try {
        currentAudioSource.stop();
      } catch {}
      currentAudioSource = null;
    }

    addMessage('assistant', `Fout: ${message}`);
    setStatus(`Fout: ${message}`, 'error');
  }

  async function submitTurn(blob) {
    processing = true;
    speaking = false;
    turnStreamDone = false;
    audioQueue = [];
    setStatus('Denkt…', 'processing');

    const form = new FormData();
    const ext = blob.type.includes('ogg') ? 'ogg' : 'webm';
    form.append('audio', blob, `turn.${ext}`);

    let transcriptAdded = false;
    let replyText = '';
    let sawDone = false;

    try {
      const response = await fetch(backendUrl('/api/stream-turn'), {
        method: 'POST',
        headers: {
          ...authHeaders(),
          'X-Session-ID': sessionId,
        },
        body: form,
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch {}
        throw new Error(detail);
      }

      if (!response.body) throw new Error('Streaming response wordt niet ondersteund.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';

      const handleEvent = event => {
        if (event.type === 'transcript') {
          if (event.session_id) sessionId = event.session_id;
          if (event.text && !transcriptAdded) {
            addMessage('user', event.text, `spraak ${event.stt_ms} ms`);
            transcriptAdded = true;
          }
          return;
        }

        if (event.type === 'text') {
          replyText += event.delta || '';
          return;
        }

        if (event.type === 'audio') {
          enqueueAudio(event.audio_b64);
          return;
        }

        if (event.type === 'error') {
          throw new Error(event.message || 'Onbekende streamingfout');
        }

        if (event.type === 'done') {
          sawDone = true;
          if (event.session_id) sessionId = event.session_id;

          if (!event.transcript) {
            turnStreamDone = true;
            setStatus('Niets verstaan — luister opnieuw', 'listening');
            finishTurnIfReady();
            return;
          }

          const reply = event.reply || replyText.trim();
          addMessage(
            'assistant',
            reply,
            `LLM ${event.llm_ms ?? 0} ms · TTS ${event.tts_ms ?? 0} ms · eerste audio ${event.first_audio_ms ?? 0} ms · totaal ${event.total_ms ?? 0} ms`,
          );
          turnStreamDone = true;
          finishTurnIfReady();
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        pending += decoder.decode(value, { stream: true });
        const lines = pending.split('\n');
        pending = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          handleEvent(JSON.parse(line));
        }
      }

      pending += decoder.decode();
      if (pending.trim()) handleEvent(JSON.parse(pending));

      if (!sawDone && processing) {
        throw new Error('De streamingverbinding stopte vóór het antwoord klaar was.');
      }
    } catch (error) {
      failTurn(error);
    }
  }

  function monitor() {
    if (!running || !analyser) return;
    const now = performance.now();
    const rms = rmsLevel();
    const threshold = Math.max(0.024, noiseFloor * 2.6);
    const normalized = Math.min(1, rms / Math.max(threshold * 1.8, 0.04));
    meterFill.style.width = `${Math.round(normalized * 100)}%`;

    if (!processing && !speaking) {
      if (!mediaRecorder) {
        if (rms < threshold * 0.75) {
          noiseFloor = noiseFloor * 0.985 + rms * 0.015;
        }
        if (rms > threshold) {
          if (!loudSince) loudSince = now;
          if (now - loudSince >= cfg.triggerMs) beginRecording(now);
        } else {
          loudSince = 0;
        }
      } else {
        if (rms > threshold * 0.85) lastLoudAt = now;
        const utteranceMs = now - speechStart;
        const quietMs = now - lastLoudAt;
        if (
          (utteranceMs >= cfg.minUtteranceMs && quietMs >= cfg.silenceMs) ||
          utteranceMs >= cfg.maxUtteranceMs
        ) {
          stopRecording();
        }
      }
    }

    rafId = requestAnimationFrame(monitor);
  }

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Deze browser geeft geen microfoontoegang via getUserMedia.');
    if (!window.MediaRecorder) throw new Error('MediaRecorder wordt niet ondersteund door deze browser.');

    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    await audioContext.resume();

    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.35;
    analyserData = new Float32Array(analyser.fftSize);
    source.connect(analyser);

    running = true;
    mainButton.textContent = 'STOP';
    mainButton.classList.add('active');
    setStatus('Luistert', 'listening');
    monitor();
  }

  async function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    audioQueue = [];
    turnStreamDone = true;

    if (currentAudioSource) {
      try {
        currentAudioSource.stop();
      } catch {}
      currentAudioSource = null;
    }

    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    mediaRecorder = null;
    stream?.getTracks().forEach(track => track.stop());
    stream = null;

    if (audioContext) await audioContext.close().catch(() => {});
    audioContext = null;
    analyser = null;
    speaking = false;
    processing = false;
    audioDrainRunning = false;
    meterFill.style.width = '0%';
    mainButton.textContent = 'START';
    mainButton.classList.remove('active');
    setStatus('Gestopt', 'idle');
  }

  mainButton.addEventListener('click', async () => {
    try {
      if (running) await stop();
      else await start();
    } catch (error) {
      setStatus(`Fout: ${error.message}`, 'error');
    }
  });

  saveSettingsButton.addEventListener('click', saveSettings);
  testBackendButton.addEventListener('click', testBackend);

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  }
})();
