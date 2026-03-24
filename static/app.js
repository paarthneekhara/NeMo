(function () {
  const runBtn = document.getElementById("runBtn");
  const uploadAudioBtn = document.getElementById("uploadAudioBtn");
  const contextAudioFileEl = document.getElementById("contextAudioFile");
  const contextStatusEl = document.getElementById("contextStatus");
  const contextSelectEl = document.getElementById("contextSelect");
  const useContextAudioEl = document.getElementById("useContextAudio");
  const cfgScaleEl = document.getElementById("cfgScale");
  const temperatureEl = document.getElementById("temperature");
  const topkEl = document.getElementById("topk");
  const useCfgEl = document.getElementById("useCfg");
  const tokensPerSecondEl = document.getElementById("tokensPerSecond");
  const promptEl = document.getElementById("prompt");
  const partialTextEl = document.getElementById("partialText");
  const statusEl = document.getElementById("status");
  const streamViz = document.getElementById("streamViz");
  const audioStatsEl = document.getElementById("audioStats");
  const outputClipsEl = document.getElementById("outputClips");
  const backendLoaderEl = document.getElementById("backendLoader");
  const backendLoaderTextEl = document.getElementById("backendLoaderText");

  let ws = null;
  let isConnected = false;
  let running = false;
  let resetInFlight = false;
  let audioCtx = null;
  let analyser = null;
  let gainNode = null;
  let workletNode = null;
  let sourceSampleRate = 24000;
  let vizRaf = 0;
  let receivedChunks = 0;
  let playedChunks = 0;
  let lastRunChunkBuffers = [];
  let queuedSamples = 0;
  let uploadedContextAudioId = null;
  let reconnectTimerId = null;
  let reconnectAttempt = 0;
  let autoResetTimerId = null;
  let pendingAutoReset = false;
  let pendingAutoResetAfterRun = false;
  let ttfaMs = null;
  let sawGenerationFinished = false;
  let currentRunClipFinalized = false;

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setBackendLoader(isVisible, text) {
    if (!backendLoaderEl) return;
    backendLoaderEl.classList.toggle("hidden", !isVisible);
    if (text && backendLoaderTextEl) {
      backendLoaderTextEl.textContent = text;
    }
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/stream`;
  }

  function queuedSeconds() {
    if (!audioCtx || audioCtx.sampleRate <= 0) return 0;
    return Math.max(0, queuedSamples / audioCtx.sampleRate);
  }

  function formatTtfaMs() {
    return Number.isFinite(ttfaMs) ? ttfaMs.toFixed(1) : "n/a";
  }

  function refreshAudioStats() {
    audioStatsEl.textContent = `chunks_rx=${receivedChunks} chunks_played=${playedChunks} queued_sec=${queuedSeconds().toFixed(2)} ttfa_ms=${formatTtfaMs()}`;
  }

  function clearPlaybackQueue() {
    queuedSamples = 0;
    if (workletNode) {
      workletNode.port.postMessage({ type: "clear" });
    }
    refreshAudioStats();
  }

  function getInferenceOptions() {
    return {
      use_cfg: Boolean(useCfgEl.checked),
      cfg_scale: Number(cfgScaleEl.value || 2.5),
      temperature: Number(temperatureEl.value || 0.5),
      topk: Number(topkEl.value || 80),
    };
  }

  function getContextPayload() {
    return {
      context_text_index: Number(contextSelectEl.value || 0),
      use_context_audio: Boolean(useContextAudioEl.checked),
      context_audio_id: uploadedContextAudioId,
    };
  }

  function isSocketOpen() {
    return Boolean(ws && ws.readyState === WebSocket.OPEN);
  }

  function refreshControls() {
    const disabledBecauseBusy = running || resetInFlight || !isConnected;
    runBtn.disabled = disabledBecauseBusy;
    uploadAudioBtn.disabled = !isConnected || running || resetInFlight;
    contextSelectEl.disabled = !isConnected || running || resetInFlight;
    useContextAudioEl.disabled = !isConnected || running || resetInFlight;
    contextAudioFileEl.disabled = !isConnected || running || resetInFlight;
  }

  function clearRunMetrics() {
    ttfaMs = null;
    refreshAudioStats();
  }

  function mergeArrayBuffers(buffers) {
    const totalBytes = buffers.reduce((acc, buf) => acc + buf.byteLength, 0);
    const merged = new Uint8Array(totalBytes);
    let offset = 0;
    for (const buf of buffers) {
      const view = new Uint8Array(buf);
      merged.set(view, offset);
      offset += view.byteLength;
    }
    return merged;
  }

  async function persistFinishedClip(chunkBuffers, sampleRate) {
    const body = mergeArrayBuffers(chunkBuffers);
    const resp = await fetch("/api/save_pcm_clip", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Sample-Rate": String(sampleRate),
      },
      body,
    });
    const payload = await resp.json();
    if (!resp.ok || !payload.clip_url) {
      throw new Error(payload.error || `HTTP ${resp.status}`);
    }
    return payload.clip_url;
  }

  async function appendFinishedClip() {
    if (!outputClipsEl || !lastRunChunkBuffers.length) return;
    let clipUrl = "";
    try {
      clipUrl = await persistFinishedClip(lastRunChunkBuffers, sourceSampleRate);
    } catch (err) {
      console.error("Failed to persist finished clip", { error: String(err), sampleRate: sourceSampleRate });
      setStatus("Failed to persist finished clip (see console).");
      return;
    }
    const row = document.createElement("div");
    row.className = "clipRow";
    const title = document.createElement("div");
    title.className = "clipTitle";
    title.textContent = `Run ${outputClipsEl.children.length + 1} • chunks=${lastRunChunkBuffers.length} • ttfa_ms=${formatTtfaMs()}`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = clipUrl;
    audio.addEventListener("error", () => {
      console.error("Failed to decode finished clip", {
        sampleRate: sourceSampleRate,
        chunks: lastRunChunkBuffers.length,
        blobUrl: clipUrl,
      });
      setStatus("Failed to decode finished clip (see console).");
    });
    row.appendChild(title);
    row.appendChild(audio);
    outputClipsEl.prepend(row);
  }

  async function fetchContextTexts() {
    try {
      const resp = await fetch("/api/context_texts");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const items = Array.isArray(data.items) ? data.items : [];
      const prevSelectedValue = contextSelectEl.value;
      contextSelectEl.innerHTML = "";
      items.forEach((t, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        const trimmed = t.length > 120 ? `${t.slice(0, 120)}...` : t;
        opt.textContent = `${idx}: ${trimmed}`;
        contextSelectEl.appendChild(opt);
      });
      if (items.length > 0) {
        const wanted = Number(prevSelectedValue);
        if (Number.isInteger(wanted) && wanted >= 0 && wanted < items.length) {
          contextSelectEl.value = String(wanted);
        }
      }
    } catch (_err) {
      setStatus("Failed to load context text list.");
    }
  }

  function maybeRunQueuedAutoReset() {
    if (!pendingAutoReset) return;
    pendingAutoReset = false;
    requestAutoReset("Queued context update");
  }

  function sendResetNow(reason) {
    if (!isSocketOpen()) return;
    if (running) {
      pendingAutoResetAfterRun = true;
      setStatus(`${reason}: queued until current run finishes.`);
      return;
    }
    if (resetInFlight) {
      pendingAutoReset = true;
      return;
    }
    resetInFlight = true;
    refreshControls();
    setBackendLoader(true, "Applying backend context/speaker...");
    setStatus(`${reason}: applying backend state...`);
    const payload = {
      type: "reset",
      inference: getInferenceOptions(),
      ...getContextPayload(),
    };
    ws.send(JSON.stringify(payload));
  }

  function requestAutoReset(reason) {
    if (!isSocketOpen()) return;
    if (autoResetTimerId) {
      clearTimeout(autoResetTimerId);
      autoResetTimerId = null;
    }
    autoResetTimerId = setTimeout(() => {
      autoResetTimerId = null;
      sendResetNow(reason);
    }, 200);
  }

  function scheduleReconnect() {
    if (reconnectTimerId) return;
    const delay = Math.min(5000, 1000 * (2 ** reconnectAttempt));
    reconnectAttempt += 1;
    setBackendLoader(false);
    setStatus(`Disconnected. Reconnecting in ${delay}ms...`);
    reconnectTimerId = setTimeout(() => {
      reconnectTimerId = null;
      connectWebSocket();
    }, delay);
  }

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setStatus("Connecting...");
    const socket = new WebSocket(wsUrl());
    ws = socket;
    socket.binaryType = "arraybuffer";

    socket.onopen = async () => {
      if (ws !== socket) return;
      isConnected = true;
      reconnectAttempt = 0;
      setBackendLoader(false);
      setStatus("Connected.");
      refreshControls();
      await fetchContextTexts();
      requestAutoReset("Initial state sync");
      refreshAudioStats();
    };

    socket.onmessage = async (event) => {
      if (ws !== socket) return;
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "partial_text") {
            partialTextEl.value = msg.text || "";
          } else if (msg.type === "status") {
            if (typeof msg.sample_rate === "number" && msg.sample_rate > 0) {
              sourceSampleRate = msg.sample_rate;
            }
            if (typeof msg.ttfa_ms_backend === "number" && Number.isFinite(msg.ttfa_ms_backend)) {
              ttfaMs = msg.ttfa_ms_backend;
              refreshAudioStats();
            }
            if (msg.state === "reset_started") {
              resetInFlight = true;
              setBackendLoader(true, "Applying backend context/speaker...");
              setStatus("Applying backend context/speaker...");
            } else if (msg.state === "reset_done" || msg.state === "reset") {
              resetInFlight = false;
              setBackendLoader(false);
              setStatus("Context/speaker applied.");
              maybeRunQueuedAutoReset();
            } else {
              setStatus(JSON.stringify(msg));
            }
            if (msg.state === "generation_finished") {
              sawGenerationFinished = true;
            }
            if (msg.state === "done" || msg.state === "error") {
              running = false;
              if (msg.state === "done" && !currentRunClipFinalized) {
                await appendFinishedClip();
                currentRunClipFinalized = true;
              }
              if (pendingAutoResetAfterRun) {
                pendingAutoResetAfterRun = false;
                requestAutoReset("Post-run context update");
              }
            }
            refreshControls();
          }
        } catch (_err) {
          setStatus(`Invalid JSON from server: ${event.data}`);
        }
        return;
      }

      receivedChunks += 1;
      if (event.data instanceof ArrayBuffer) {
        lastRunChunkBuffers.push(event.data.slice(0));
        pushPcmChunk(event.data);
      }
    };

    socket.onclose = () => {
      if (ws !== socket) return;
      isConnected = false;
      // If socket closes immediately after generation_finished, still keep clip.
      if (running && sawGenerationFinished && !currentRunClipFinalized && lastRunChunkBuffers.length > 0) {
        void appendFinishedClip();
        currentRunClipFinalized = true;
      }
      running = false;
      resetInFlight = false;
      pendingAutoReset = false;
      pendingAutoResetAfterRun = false;
      if (autoResetTimerId) {
        clearTimeout(autoResetTimerId);
        autoResetTimerId = null;
      }
      setBackendLoader(false);
      refreshControls();
      scheduleReconnect();
    };

    socket.onerror = () => {
      if (ws !== socket) return;
      setStatus("WebSocket error.");
    };
  }

  async function uploadReferenceAudio() {
    const file = contextAudioFileEl.files && contextAudioFileEl.files[0];
    if (!file) {
      contextStatusEl.textContent = "Choose an audio file first.";
      return;
    }
    const form = new FormData();
    form.append("file", file);
    uploadAudioBtn.disabled = true;
    contextStatusEl.textContent = "Uploading...";
    try {
      const resp = await fetch("/api/upload_reference_audio", { method: "POST", body: form });
      const data = await resp.json();
      if (!resp.ok || !data.audio_id) {
        throw new Error(data.error || "Upload failed");
      }
      uploadedContextAudioId = data.audio_id;
      contextStatusEl.textContent = `Uploaded: ${file.name} (id=${uploadedContextAudioId})`;
      requestAutoReset("Uploaded reference audio");
    } catch (err) {
      contextStatusEl.textContent = `Upload error: ${String(err)}`;
    } finally {
      refreshControls();
    }
  }

  function startVisualizer() {
    if (!audioCtx || !analyser || !streamViz) return;
    const ctx2d = streamViz.getContext("2d");
    if (!ctx2d) return;
    const bins = new Uint8Array(analyser.fftSize);
    const w = streamViz.width;
    const h = streamViz.height;
    const draw = () => {
      analyser.getByteTimeDomainData(bins);
      ctx2d.fillStyle = "#121720";
      ctx2d.fillRect(0, 0, w, h);
      ctx2d.lineWidth = 2;
      ctx2d.strokeStyle = "#45d483";
      ctx2d.beginPath();
      for (let i = 0; i < bins.length; i++) {
        const x = (i / (bins.length - 1)) * w;
        const y = (bins[i] / 255.0) * h;
        if (i === 0) ctx2d.moveTo(x, y);
        else ctx2d.lineTo(x, y);
      }
      ctx2d.stroke();
      vizRaf = requestAnimationFrame(draw);
    };
    if (vizRaf) cancelAnimationFrame(vizRaf);
    draw();
  }

  async function ensureAudioGraph() {
    const needRecreate = audioCtx && audioCtx.sampleRate !== sourceSampleRate;
    if (needRecreate) {
      if (vizRaf) cancelAnimationFrame(vizRaf);
      try {
        await audioCtx.close();
      } catch (_err) {
        // ignore close errors
      }
      audioCtx = null;
      analyser = null;
      gainNode = null;
      workletNode = null;
    }

    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC({ sampleRate: sourceSampleRate });
      await audioCtx.audioWorklet.addModule("/static/pcm_player_worklet.js");

      workletNode = new AudioWorkletNode(audioCtx, "pcm-player-processor", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      gainNode = audioCtx.createGain();
      gainNode.gain.value = 1.0;
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;

      workletNode.connect(gainNode);
      gainNode.connect(analyser);
      analyser.connect(audioCtx.destination);

      workletNode.port.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "played_chunk") {
          playedChunks += 1;
          const n = Number(msg.samples || 0);
          if (n > 0) queuedSamples = Math.max(0, queuedSamples - n);
          refreshAudioStats();
        }
      };

      startVisualizer();
    }
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
  }

  function int16ToFloat32(arrayBuffer) {
    const i16 = new Int16Array(arrayBuffer);
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) {
      f32[i] = i16[i] / 32768.0;
    }
    return f32;
  }

  function pushPcmChunk(arrayBuffer) {
    if (!workletNode) return;
    const f32 = int16ToFloat32(arrayBuffer);
    if (!f32.length) return;
    queuedSamples += f32.length;
    workletNode.port.postMessage({ type: "append", samples: f32 }, [f32.buffer]);
    refreshAudioStats();
  }

  async function streamDummyText(promptText) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setStatus("WebSocket not connected.");
      return;
    }
    await ensureAudioGraph();
    ws.send(JSON.stringify({ type: "start", inference: getInferenceOptions() }));
    partialTextEl.value = "";

    const words = promptText.trim().split(/\s+/).filter(Boolean);
    const wordsPerSecond = Math.max(1, Number(tokensPerSecondEl.value || 25));
    const delayMs = Math.max(1, Math.floor(1000 / wordsPerSecond));

    running = true;
    refreshControls();
    for (let i = 0; i < words.length; i++) {
      if (!running || !ws || ws.readyState !== WebSocket.OPEN) break;
      const token = i === 0 ? words[i] : ` ${words[i]}`;
      ws.send(JSON.stringify({ type: "text_chunk", text: token }));
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "eos" }));
    }
  }

  async function fetchLlmText(promptText) {
    const resp = await fetch("/api/llm_generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText }),
    });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.error || `HTTP ${resp.status}`);
    }
    return {
      text: String(payload.text || ""),
      model: String(payload.model || "unknown"),
    };
  }

  runBtn.addEventListener("click", async () => {
    if (running) return;
    receivedChunks = 0;
    playedChunks = 0;
    queuedSamples = 0;
    lastRunChunkBuffers = [];
    sawGenerationFinished = false;
    currentRunClipFinalized = false;
    clearRunMetrics();
    clearPlaybackQueue();
    await ensureAudioGraph();
    const promptText = (promptEl.value || "").trim();
    if (!promptText) {
      setStatus("Prompt is empty.");
      return;
    }
    try {
      running = true;
      refreshControls();
      setStatus("Calling OpenAI...");
      const llm = await fetchLlmText(promptText);
      running = false;
      setStatus(`OpenAI (${llm.model}) responded. Streaming to TTS...`);
      await streamDummyText(llm.text);
    } catch (err) {
      running = false;
      refreshControls();
      setStatus(`OpenAI error: ${String(err)}`);
    }
  });

  contextSelectEl.addEventListener("change", () => {
    requestAutoReset("Context text changed");
  });

  useContextAudioEl.addEventListener("change", () => {
    requestAutoReset("Reference audio toggle changed");
  });

  uploadAudioBtn.addEventListener("click", async () => {
    await uploadReferenceAudio();
  });

  contextAudioFileEl.addEventListener("change", () => {
    const file = contextAudioFileEl.files && contextAudioFileEl.files[0];
    if (!file) return;
    contextStatusEl.textContent = `Ready to upload: ${file.name}`;
  });

  refreshControls();
  refreshAudioStats();
  connectWebSocket();
})();
