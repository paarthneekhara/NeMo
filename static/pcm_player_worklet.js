class PcmPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.currentChunkSamples = 0;

    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "append" && msg.samples) {
        const chunk = new Float32Array(msg.samples);
        if (chunk.length > 0) {
          this.queue.push(chunk);
        }
      } else if (msg.type === "clear") {
        this.queue = [];
        this.current = null;
        this.offset = 0;
        this.currentChunkSamples = 0;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const ch0 = output[0];
    ch0.fill(0.0);

    let writePos = 0;
    while (writePos < ch0.length) {
      if (!this.current) {
        if (this.queue.length === 0) {
          break;
        }
        this.current = this.queue.shift();
        this.offset = 0;
        this.currentChunkSamples = this.current.length;
      }

      const remaining = this.current.length - this.offset;
      const n = Math.min(remaining, ch0.length - writePos);
      ch0.set(this.current.subarray(this.offset, this.offset + n), writePos);
      this.offset += n;
      writePos += n;

      if (this.offset >= this.current.length) {
        this.port.postMessage({ type: "played_chunk", samples: this.currentChunkSamples });
        this.current = null;
        this.offset = 0;
        this.currentChunkSamples = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-player-processor", PcmPlayerProcessor);
