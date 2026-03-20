(function () {
  const connectBtn = document.getElementById("connectBtn");
  const runBtn = document.getElementById("runBtn");
  const resetBtn = document.getElementById("resetBtn");
  const replayBtn = document.getElementById("replayBtn");
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

  let ws = null;
  let running = false;
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

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/stream`;
  }

  function queuedSeconds() {
    if (!audioCtx || audioCtx.sampleRate <= 0) return 0;
    return Math.max(0, queuedSamples / audioCtx.sampleRate);
  }

  function refreshAudioStats() {
    audioStatsEl.textContent = `chunks_rx=${receivedChunks} chunks_played=${playedChunks} queued_sec=${queuedSeconds().toFixed(2)}`;
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

  async function fetchContextTexts() {
    try {
      const resp = await fetch("/api/context_texts");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const items = Array.isArray(data.items) ? data.items : [];
      contextSelectEl.innerHTML = "";
      items.forEach((t, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        const trimmed = t.length > 120 ? `${t.slice(0, 120)}...` : t;
        opt.textContent = `${idx}: ${trimmed}`;
        contextSelectEl.appendChild(opt);
      });
    } catch (_err) {
      setStatus("Failed to load context text list.");
    }
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
    } catch (err) {
      contextStatusEl.textContent = `Upload error: ${String(err)}`;
    } finally {
      uploadAudioBtn.disabled = false;
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

  async function replayLastAudio() {
    if (!lastRunChunkBuffers.length) {
      setStatus("No previous audio to replay yet.");
      return;
    }
    await ensureAudioGraph();
    clearPlaybackQueue();
    playedChunks = 0;
    receivedChunks = lastRunChunkBuffers.length;
    refreshAudioStats();
    for (const chunk of lastRunChunkBuffers) {
      pushPcmChunk(chunk.slice(0));
    }
    setStatus(`Replaying ${lastRunChunkBuffers.length} chunk(s).`);
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
    runBtn.disabled = true;
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

  connectBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      setStatus("Already connected.");
      return;
    }
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      setStatus("Connected.");
      runBtn.disabled = false;
      resetBtn.disabled = false;
      uploadAudioBtn.disabled = false;
      replayBtn.disabled = true;
      await fetchContextTexts();
      refreshAudioStats();
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "partial_text") {
            partialTextEl.value = msg.text || "";
          } else if (msg.type === "status") {
            if (typeof msg.sample_rate === "number" && msg.sample_rate > 0) {
              sourceSampleRate = msg.sample_rate;
            }
            setStatus(JSON.stringify(msg));
            if (msg.state === "done" || msg.state === "error") {
              running = false;
              runBtn.disabled = false;
              replayBtn.disabled = !(msg.state === "done" && lastRunChunkBuffers.length > 0);
            }
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

    ws.onclose = () => {
      running = false;
      runBtn.disabled = true;
      resetBtn.disabled = true;
      uploadAudioBtn.disabled = true;
      replayBtn.disabled = true;
      setStatus("Disconnected.");
    };

    ws.onerror = () => {
      setStatus("WebSocket error.");
    };
  });

  runBtn.addEventListener("click", async () => {
    if (running) return;
    receivedChunks = 0;
    playedChunks = 0;
    queuedSamples = 0;
    lastRunChunkBuffers = [];
    replayBtn.disabled = true;
    clearPlaybackQueue();
    await ensureAudioGraph();
    await streamDummyText(promptEl.value || "");
  });

  resetBtn.addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    running = false;
    clearPlaybackQueue();
    const payload = {
      type: "reset",
      inference: getInferenceOptions(),
      ...getContextPayload(),
    };
    ws.send(JSON.stringify(payload));
    setStatus("Reset requested.");
  });

  replayBtn.addEventListener("click", async () => {
    await replayLastAudio();
  });

  uploadAudioBtn.addEventListener("click", async () => {
    await uploadReferenceAudio();
  });

  contextAudioFileEl.addEventListener("change", () => {
    const file = contextAudioFileEl.files && contextAudioFileEl.files[0];
    if (!file) return;
    contextStatusEl.textContent = `Ready to upload: ${file.name}`;
  });
})();
