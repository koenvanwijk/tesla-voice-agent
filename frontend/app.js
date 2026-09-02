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
  let chunks = [];
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
    speechStart = 0;
    lastLoudAt = 0;
    loudSince = 0;
    if (running) setStatus('Luistert', 'listening');
  }

  async function submitTurn(blob) {
    processing = true;
    setStatus('Denkt…', 'processing');
    const form = new FormData();
    const ext = blob.type.includes('ogg') ? 'ogg' : 'webm';
    form.append('audio', blob, `turn.${ext}`);

    try {
      const response = await fetch(backendUrl('/api/turn'), {
        method: 'POST',
        headers: {
          ...authHeaders(),
          'X-Session-ID': sessionId,
        },
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      if (data.session_id) sessionId = data.session_id;
      if (!data.transcript) {
        setStatus('Niets verstaan — luister opnieuw', 'listening');
        processing = false;
        return;
      }
      addMessage('user', data.transcript, `spraak ${data.stt_ms} ms`);
      addMessage('assistant', data.reply, `LLM ${data.llm_ms} ms · totaal ${data.total_ms} ms`);
      await speak(data.reply);
    } catch (error) {
      addMessage('assistant', `Fout: ${error.message}`);
      setStatus(`Fout: ${error.message}`, 'error');
      processing = false;
    }
  }

  function chooseDutchVoice() {
    if (!('speechSynthesis' in window)) return null;
    const voices = speechSynthesis.getVoices();
    return voices.find(v => /^nl(-|_)/i.test(v.lang)) || voices.find(v => /^nl/i.test(v.lang)) || null;
  }

  function speak(text) {
    return new Promise(resolve => {
      if (!text || !('speechSynthesis' in window)) {
        resumeListening();
        resolve();
        return;
      }
      speaking = true;
      setStatus('Praat…', 'speaking');
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'nl-NL';
      const voice = chooseDutchVoice();
      if (voice) utterance.voice = voice;
      utterance.rate = 1.02;
      utterance.onend = () => {
        resumeListening();
        resolve();
      };
      utterance.onerror = () => {
        resumeListening();
        resolve();
      };
      speechSynthesis.speak(utterance);
    });
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
    speechSynthesis?.cancel?.();
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    mediaRecorder = null;
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
    if (audioContext) await audioContext.close().catch(() => {});
    audioContext = null;
    analyser = null;
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
