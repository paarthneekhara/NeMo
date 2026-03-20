"""Minimal streaming demo for EasyMagpieTTS with async codec decoding."""

import csv
import contextlib
import io
import os
import sys
import threading
import time
import types
import wave
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

os.environ["OMP_NUM_THREADS"] = "2"

# Avoid megatron unified_memory JIT path for this inference script.
_um = types.ModuleType("megatron.core.inference.unified_memory")
_um.has_unified_memory = False
_um.create_unified_mempool = None
sys.modules["megatron.core.inference.unified_memory"] = _um

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.utils import logging


# ----------------------------
# Editable defaults (no CLI)
# ----------------------------
MODEL_PATH = "/datap/misc/EasyMagpieAssets/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay.nemo"
CODEC_MODEL_PATH = "/datap/misc/EasyMagpieAssets/25fps_spectral_codec_with_bandwidth_extension.nemo"
PHONEME_TOKENIZER_PATH = "/datap/misc/EasyMagpieAssets/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json"
CONTEXT_TEXTS_PATH = "/datap/misc/EasyMagpieAssets/unique_text_contexts.txt"
DEFAULT_CONTEXT_TEXT_INDEX = 0

# Set to a valid path to use context audio instead of text context.
CONTEXT_AUDIO_PATH: Optional[str] = None
CONTEXT_AUDIO_DURATION_SEC = 5.0
LANGUAGE = "en"
# Directory where streamed chunk wav files are saved.
# If relative, it is resolved from the current working directory.
STREAM_SAVE_DIR = "/datap/misc/EasyMagpieAssets/easymagpie_stream_outputs"

USE_CFG = True
CFG_SCALE = 2.5
USE_LOCAL_TRANSFORMER = True
TEMPERATURE = 0.5
TOPK = 80
MAX_DECODER_STEPS = 330
PHONEME_INPUT_TYPE = "pred"
PHONEME_SAMPLING_METHOD = "argmax"

WORDS_PER_SECOND = 100.0
DECODE_EVERY_AUDIO_FRAMES = 20
DECODE_POLL_INTERVAL_SEC = 0.01
MAIN_LOOP_IDLE_SLEEP_SEC = 0.002
STREAM_MIN_CHUNK_SEC = 0.12
WARMUP_ENABLED = True
WARMUP_TEXT = "This is a startup warmup run for local transformer and generation path."
WARMUP_MAX_DECODER_STEPS = 80

# Keep generation and decoding on separate GPUs when available.
GENERATION_GPU_INDEX = 0
DECODE_GPU_INDEX = 1
# Precision mode for generation compute: "bf16" (AMP), "fp16" (AMP), or "fp32".
MODEL_PRECISION = "bf16"
# Local transformer backend: "torch" or "trt".
LOCAL_TRANSFORMER_BACKEND = "torch"

DEFAULT_QUESTION = "What is the main idea behind this demo?"
DUMMY_RESPONSE_TEMPLATE = (
    "Sure. This is a streaming speech synthesis demo where text arrives gradually, "
    "tokens are consumed incrementally, and generated codec frames are decoded in the "
    "background for low-latency playback. "
)


def _apply_inference_overrides(model_cfg, codec_model_path: str, phoneme_tokenizer_path: str):
    with open_dict(model_cfg):
        model_cfg.target = "nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel"
        model_cfg.codecmodel_path = codec_model_path
        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        model_cfg.run_val_inference = False
        model_cfg.use_utmos = False
        model_cfg.use_meta_init_for_decoder = True
        if getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = phoneme_tokenizer_path
    return model_cfg


def resolve_devices() -> Tuple[torch.device, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    num_gpus = torch.cuda.device_count()
    generation_idx = max(0, min(GENERATION_GPU_INDEX, num_gpus - 1))

    if DECODE_GPU_INDEX < num_gpus:
        decode_idx = DECODE_GPU_INDEX
    else:
        decode_idx = generation_idx

    return torch.device(f"cuda:{generation_idx}"), torch.device(f"cuda:{decode_idx}")


def resolve_autocast_dtype(precision: str, generation_device: torch.device) -> Optional[torch.dtype]:
    p = precision.lower().strip()
    if p == "bf16":
        if generation_device.type == "cuda" and not torch.cuda.is_bf16_supported():
            logging.warning("bf16 requested but not supported on this GPU; disabling AMP (fp32).")
            return None
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    if p == "fp32":
        return None
    raise ValueError(f"Unsupported MODEL_PRECISION='{precision}'. Use one of: bf16, fp16, fp32.")


def autocast_context(device: torch.device, autocast_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and autocast_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return contextlib.nullcontext()


def load_model(generation_device: torch.device) -> EasyMagpieTTSInferenceModel:
    logging.info(f"Loading model from .nemo: {MODEL_PATH}")
    model_cfg = EasyMagpieTTSInferenceModel.restore_from(MODEL_PATH, return_config=True)
    model_cfg = _apply_inference_overrides(model_cfg, CODEC_MODEL_PATH, PHONEME_TOKENIZER_PATH)
    model = EasyMagpieTTSInferenceModel.restore_from(
        MODEL_PATH,
        override_config_path=model_cfg,
        map_location=torch.device("cpu"),
    )
    model.use_kv_cache_for_inference = True
    # Keep model weights in fp32 and use AMP autocast during inference calls.
    model.eval().to(device=generation_device).float()
    return model


def build_decode_codec_helper(model: EasyMagpieTTSInferenceModel, decode_device: torch.device) -> CodecHelper:
    codec_model = AudioCodecModel.restore_from(
        CODEC_MODEL_PATH,
        strict=False,
        map_location=torch.device("cpu"),
    )
    if hasattr(codec_model, "discriminator"):
        del codec_model.discriminator
    codec_model.freeze()
    codec_model = codec_model.to(decode_device).eval()

    codec_converter = None
    if model._codec_converter is not None:  # noqa: SLF001
        vq_new = deepcopy(model._codec_converter.vector_quantizer_new).to(decode_device).eval()  # noqa: SLF001
        codec_converter = VectorQuantizerIndexConverter(
            vector_quantizer_original=codec_model.vector_quantizer,
            vector_quantizer_new=vq_new,
        ).to(decode_device)
        codec_converter.eval()

    return CodecHelper(codec_model=codec_model, codec_converter=codec_converter)


def load_context_texts(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Context text file not found: {path}")

    # Support TSV with `text_context` or plain one-line-per-entry txt files.
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        if "\t" in first_line and "text_context" in first_line:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("text_context") or "").strip()
                if text:
                    texts.append(text)
        else:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)

    if not texts:
        raise ValueError(f"No usable context texts found in {path}")
    return texts


def build_context_inputs(
    model: EasyMagpieTTSInferenceModel, context_text: str, context_audio_path: Optional[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    text_ids = model.tokenizer.encode(context_text, tokenizer_name=model.text_conditioning_tokenizer_name)
    context_text_tokens = torch.tensor([text_ids], dtype=torch.long, device=device)
    context_text_lens = torch.tensor([len(text_ids)], dtype=torch.long, device=device)

    if context_audio_path:
        context_audio = model._load_audio_for_inference(context_audio_path, model.sample_rate)
        context_audio = model._adjust_audio_to_duration_for_inference(
            context_audio,
            model.sample_rate,
            CONTEXT_AUDIO_DURATION_SEC,
            model.codec_model_samples_per_frame,
        )
        context_audio = context_audio.to(device=device, dtype=model_dtype)
        context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
        with torch.inference_mode():
            context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(context_audio, context_audio_lens)
    else:
        context_audio_codes = torch.zeros(
            1,
            model.data_num_audio_codebooks,
            0,
            dtype=torch.long,
            device=device,
        )
        context_audio_codes_lens = torch.zeros(1, dtype=torch.long, device=device)

    return context_audio_codes, context_audio_codes_lens, context_text_tokens, context_text_lens


def create_base_streaming_state(
    model: EasyMagpieTTSInferenceModel,
    context_text: str,
    generation_device: torch.device,
    autocast_dtype: Optional[torch.dtype],
):
    context_audio_codes, context_audio_codes_lens, context_text_tokens, context_text_lens = build_context_inputs(
        model, context_text, CONTEXT_AUDIO_PATH
    )
    with torch.inference_mode():
        with autocast_context(generation_device, autocast_dtype):
            state = model.streaming_init(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_lens,
                use_cfg=USE_CFG,
                cfg_scale=CFG_SCALE,
                use_local_transformer=USE_LOCAL_TRANSFORMER,
                temperature=TEMPERATURE,
                topk=TOPK,
                phoneme_input_type=PHONEME_INPUT_TYPE,
                phoneme_sampling_method=PHONEME_SAMPLING_METHOD,
                use_inference_mode=True,
            )
    return state


def clone_streaming_state(state):
    # For this demo, deepcopy is sufficient and keeps logic minimal.
    return deepcopy(state)


def run_startup_warmup(
    model: EasyMagpieTTSInferenceModel,
    base_state,
    generation_device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> None:
    if not WARMUP_ENABLED:
        print("[WARMUP] Disabled (WARMUP_ENABLED=False).", flush=True)
        return

    print("[WARMUP] Starting warmup run before launching UI...", flush=True)
    t0 = time.perf_counter()

    state = clone_streaming_state(base_state)
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
    token_ids = model.tokenizer.encode(WARMUP_TEXT, tokenizer_name=main_tokenizer_name)
    token_ids.append(model.eos_id)

    device = next(model.parameters()).device
    pending_token_ids: Deque[int] = deque(token_ids)
    emitted_audio_frames = 0
    steps = 0

    print(
        f"[WARMUP] text_tokens={len(token_ids)}, max_steps={WARMUP_MAX_DECODER_STEPS}, "
        f"amp_dtype={autocast_dtype}, generation_device={generation_device}",
        flush=True,
    )
    with torch.inference_mode():
        while not bool(state.finished.all()) and steps < WARMUP_MAX_DECODER_STEPS:
            if pending_token_ids:
                tok = pending_token_ids.popleft()
                text_tokens = torch.tensor([tok], dtype=torch.long, device=device)
            else:
                text_tokens = None

            step_t0 = time.perf_counter()
            if autocast_dtype is None:
                state, audio_codes, _phoneme_tokens = model.streaming_step(
                    state=state,
                    text_tokens=text_tokens,
                    use_inference_mode=True,
                )
            else:
                with autocast_context(generation_device, autocast_dtype):
                    state, audio_codes, _phoneme_tokens = model.streaming_step(
                        state=state,
                        text_tokens=text_tokens,
                        use_inference_mode=True,
                    )
            step_ms = (time.perf_counter() - step_t0) * 1000.0
            steps += 1

            if audio_codes is not None:
                emitted_audio_frames += int(audio_codes.size(-1))

            if steps == 1 or steps % 10 == 0 or len(pending_token_ids) == 0:
                print(
                    f"[WARMUP] step={steps:03d} step_ms={step_ms:7.2f} "
                    f"pending_text_tokens={len(pending_token_ids):03d} "
                    f"emitted_audio_frames={emitted_audio_frames:04d} "
                    f"finished={bool(state.finished.all())}",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(
        f"[WARMUP] Completed in {elapsed:.2f}s (steps={steps}, "
        f"emitted_audio_frames={emitted_audio_frames}, finished={bool(state.finished.all())}).",
        flush=True,
    )


def _to_int16_pcm(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return np.zeros((0,), dtype=np.int16)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


def _write_wav_int16(path: str, sample_rate: int, pcm: np.ndarray) -> None:
    pcm_i16 = np.asarray(pcm, dtype=np.int16).reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _write_wav_bytes_int16(path: str, sample_rate: int, pcm_bytes: bytes) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _wav_bytes_from_int16(sample_rate: int, pcm: np.ndarray) -> bytes:
    pcm_i16 = np.asarray(pcm, dtype=np.int16).reshape(-1)
    with io.BytesIO() as bio:
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_i16.tobytes())
        return bio.getvalue()


def _decode_new_audio_chunk(
    model: EasyMagpieTTSInferenceModel,
    decode_codec_helper: CodecHelper,
    all_audio_codes: torch.Tensor,
    last_emitted_sample_idx: int,
) -> Tuple[np.ndarray, int]:
    pred_codes_lens = torch.tensor([all_audio_codes.size(-1)], dtype=torch.long, device=all_audio_codes.device)
    pred_codes, pred_codes_lens = model._prepare_codes_for_decode(all_audio_codes, pred_codes_lens)  # noqa: SLF001
    audio, audio_len, _decoded_codes = decode_codec_helper.codes_to_audio(pred_codes, pred_codes_lens)
    full_wav = audio[0, : audio_len[0]].detach().float().cpu().numpy()
    if full_wav.shape[0] <= last_emitted_sample_idx:
        return np.zeros((0,), dtype=np.float32), last_emitted_sample_idx
    chunk = full_wav[last_emitted_sample_idx:]
    return chunk, full_wav.shape[0]


@dataclass
class DecodeSharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: bool = False
    accumulated_audio_codes: Optional[torch.Tensor] = None
    generated_audio_frames: int = 0
    last_decode_frame_mark: int = 0
    last_emitted_sample_idx: int = 0
    decoded_chunks: Deque[np.ndarray] = field(default_factory=deque)


def decode_worker(model: EasyMagpieTTSInferenceModel, decode_codec_helper: CodecHelper, shared: DecodeSharedState):
    while True:
        with shared.lock:
            should_stop = shared.stop
            codes_snapshot = shared.accumulated_audio_codes
            frame_count_snapshot = shared.generated_audio_frames
            should_decode = (
                codes_snapshot is not None
                and frame_count_snapshot > shared.last_decode_frame_mark
                and (
                    frame_count_snapshot - shared.last_decode_frame_mark >= DECODE_EVERY_AUDIO_FRAMES
                    or should_stop
                )
            )
            last_emitted_sample_idx = shared.last_emitted_sample_idx

        if should_decode and codes_snapshot is not None:
            chunk_f32, new_sample_idx = _decode_new_audio_chunk(
                model,
                decode_codec_helper,
                codes_snapshot,
                last_emitted_sample_idx,
            )
            with shared.lock:
                shared.last_decode_frame_mark = frame_count_snapshot
                shared.last_emitted_sample_idx = new_sample_idx
                if chunk_f32.size > 0:
                    shared.decoded_chunks.append(_to_int16_pcm(chunk_f32))

        with shared.lock:
            done = should_stop and (
                shared.accumulated_audio_codes is None or shared.last_decode_frame_mark >= shared.generated_audio_frames
            )
        if done:
            return
        time.sleep(DECODE_POLL_INTERVAL_SEC)


def build_dummy_response(question: str) -> str:
    q = (question or "").strip()
    if not q:
        q = DEFAULT_QUESTION
    return f"{DUMMY_RESPONSE_TEMPLATE} You asked: {q}."


def make_app(
    model: EasyMagpieTTSInferenceModel,
    base_state,
    decode_device: torch.device,
    decode_codec_helper: CodecHelper,
    autocast_dtype: Optional[torch.dtype],
):
    state_lock = threading.Lock()
    current_state = {"state": clone_streaming_state(base_state)}

    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
    device = next(model.parameters()).device
    sr = int(model.output_sample_rate)

    def reset_to_base() -> str:
        with state_lock:
            current_state["state"] = clone_streaming_state(base_state)
        return "Streaming state reset to post-init base state."

    def run_demo(question: str):
        with state_lock:
            state = current_state["state"]

        response_text = build_dummy_response(question)
        words = response_text.split()
        word_dt = 1.0 / max(1e-6, WORDS_PER_SECOND)
        next_word_time = time.perf_counter()

        text_buffer = ""
        partial_text = ""
        sent_token_ids: List[int] = []
        pending_token_ids: Deque[int] = deque()
        llm_word_idx = 0
        llm_done = False

        run_tag = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(os.path.abspath(STREAM_SAVE_DIR), f"easymagpie_stream_{run_tag}")
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Saving streamed chunks to: {run_dir}")
        emitted_chunk_count = 0
        cumulative_pcm_bytes = bytearray()
        pending_emit_i16 = np.zeros((0,), dtype=np.int16)
        min_emit_samples = max(1, int(sr * STREAM_MIN_CHUNK_SEC))

        def _save_and_package_chunk(chunk: np.ndarray, force_emit: bool = False) -> Optional[bytes]:
            nonlocal emitted_chunk_count, cumulative_pcm_bytes, pending_emit_i16
            chunk_i16 = np.asarray(chunk, dtype=np.int16).reshape(-1)
            if chunk_i16.size == 0:
                if not force_emit:
                    return None
            if pending_emit_i16.size == 0:
                pending_emit_i16 = chunk_i16
            elif chunk_i16.size > 0:
                pending_emit_i16 = np.concatenate([pending_emit_i16, chunk_i16])

            if pending_emit_i16.size < min_emit_samples and not force_emit:
                return None

            emit_i16 = pending_emit_i16
            pending_emit_i16 = np.zeros((0,), dtype=np.int16)

            chunk_wav_path = os.path.join(run_dir, f"chunk_{emitted_chunk_count:06d}.wav")
            _write_wav_int16(path=chunk_wav_path, sample_rate=sr, pcm=emit_i16)

            cumulative_pcm_bytes.extend(emit_i16.tobytes())
            cumulative_wav_path = os.path.join(run_dir, f"cumulative_{emitted_chunk_count:06d}.wav")
            _write_wav_bytes_int16(path=cumulative_wav_path, sample_rate=sr, pcm_bytes=bytes(cumulative_pcm_bytes))

            emitted_chunk_count += 1
            return _wav_bytes_from_int16(sample_rate=sr, pcm=emit_i16)

        shared = DecodeSharedState()
        worker = threading.Thread(target=decode_worker, args=(model, decode_codec_helper, shared), daemon=True)
        worker.start()

        steps = 0
        with torch.inference_mode():
            while not bool(state.finished.all()) and steps < MAX_DECODER_STEPS:
                now = time.perf_counter()
                while not llm_done and now >= next_word_time:
                    word = words[llm_word_idx]
                    if text_buffer:
                        text_buffer += " "
                        partial_text += " "
                    text_buffer += word
                    partial_text += word

                    token_ids = model.tokenizer.encode(text_buffer, tokenizer_name=main_tokenizer_name)
                    new_ids = token_ids[len(sent_token_ids) :]
                    for tid in new_ids:
                        pending_token_ids.append(tid)
                    sent_token_ids.extend(new_ids)

                    llm_word_idx += 1
                    next_word_time += word_dt
                    if llm_word_idx >= len(words):
                        llm_done = True
                        pending_token_ids.append(model.eos_id)
                    now = time.perf_counter()

                if pending_token_ids:
                    tok = pending_token_ids.popleft()
                    text_tokens = torch.tensor([tok], dtype=torch.long, device=device)
                elif llm_done:
                    text_tokens = None
                else:
                    # Keep UI responsive while waiting for next LLM word.
                    emitted = False
                    with shared.lock:
                        pending_chunks = list(shared.decoded_chunks)
                        shared.decoded_chunks.clear()
                    for chunk in pending_chunks:
                        payload = _save_and_package_chunk(chunk)
                        if payload is not None:
                            emitted = True
                            yield partial_text, payload
                    if not emitted:
                        time.sleep(MAIN_LOOP_IDLE_SLEEP_SEC)
                    continue

                if autocast_dtype is None:
                    state, audio_codes, _phoneme_tokens = model.streaming_step(
                        state=state,
                        text_tokens=text_tokens,
                        use_inference_mode=True,
                    )
                else:
                    with autocast_context(device, autocast_dtype):
                        state, audio_codes, _phoneme_tokens = model.streaming_step(
                            state=state,
                            text_tokens=text_tokens,
                            use_inference_mode=True,
                        )
                steps += 1

                if audio_codes is not None:
                    new_codes = audio_codes.detach().to(device=decode_device, dtype=torch.long, non_blocking=True)
                    with shared.lock:
                        if shared.accumulated_audio_codes is None:
                            shared.accumulated_audio_codes = new_codes
                        else:
                            shared.accumulated_audio_codes = torch.cat(
                                [shared.accumulated_audio_codes, new_codes], dim=-1
                            )
                        shared.generated_audio_frames = int(shared.accumulated_audio_codes.size(-1))

                with shared.lock:
                    pending_chunks = list(shared.decoded_chunks)
                    shared.decoded_chunks.clear()
                for chunk in pending_chunks:
                    payload = _save_and_package_chunk(chunk)
                    if payload is not None:
                        yield partial_text, payload

            with shared.lock:
                shared.stop = True
            worker.join()

            with shared.lock:
                pending_chunks = list(shared.decoded_chunks)
                shared.decoded_chunks.clear()
            for chunk in pending_chunks:
                payload = _save_and_package_chunk(chunk)
                if payload is not None:
                    yield partial_text, payload

            payload = _save_and_package_chunk(np.zeros((0,), dtype=np.int16), force_emit=True)
            if payload is not None:
                yield partial_text, payload

        with state_lock:
            current_state["state"] = state

    with gr.Blocks(title="EasyMagpie Minimal Streaming Demo") as app:
        gr.Markdown("## EasyMagpie Minimal Streaming Demo")
        question = gr.Textbox(label="Question", value=DEFAULT_QUESTION, lines=2)
        with gr.Row():
            run_btn = gr.Button("Run")
            reset_btn = gr.Button("Reset to Base State")
        status = gr.Textbox(label="Status", value="Ready", interactive=False)
        partial = gr.Textbox(label="Streaming LLM Text", lines=4)
        audio = gr.Audio(label="Streaming Audio", streaming=True, autoplay=True, format="wav")

        run_btn.click(fn=run_demo, inputs=[question], outputs=[partial, audio])
        reset_btn.click(fn=reset_to_base, inputs=None, outputs=[status])

    return app


def main():
    t0 = time.perf_counter()
    os.environ["EASYMAGPIE_LT_BACKEND"] = LOCAL_TRANSFORMER_BACKEND
    generation_device, decode_device = resolve_devices()
    autocast_dtype = resolve_autocast_dtype(precision=MODEL_PRECISION, generation_device=generation_device)
    model = load_model(generation_device=generation_device)
    decode_codec_helper = build_decode_codec_helper(model=model, decode_device=decode_device)
    context_texts = load_context_texts(CONTEXT_TEXTS_PATH)
    context_idx = max(0, min(DEFAULT_CONTEXT_TEXT_INDEX, len(context_texts) - 1))
    context_text = context_texts[context_idx]
    if CONTEXT_AUDIO_PATH:
        use_language_tag = bool(getattr(model, "add_language_to_context_text", False))
        context_text = f"[{LANGUAGE.upper()}]" if use_language_tag else "[NO TEXT CONTEXT]"

    base_state = create_base_streaming_state(
        model=model,
        context_text=context_text,
        generation_device=generation_device,
        autocast_dtype=autocast_dtype,
    )
    run_startup_warmup(
        model=model,
        base_state=base_state,
        generation_device=generation_device,
        autocast_dtype=autocast_dtype,
    )
    print(f"[PROFILE] Setup + streaming_init time: {time.perf_counter() - t0:.2f}s")
    print(f"[INFO] Generation device: {generation_device}", flush=True)
    print(f"[INFO] Decode device: {decode_device}", flush=True)
    print(f"[INFO] Generation weights dtype: {next(model.parameters()).dtype}", flush=True)
    print(f"[INFO] Generation AMP dtype: {autocast_dtype}", flush=True)
    print(f"[INFO] Local transformer backend: {LOCAL_TRANSFORMER_BACKEND}", flush=True)
    print(f"[INFO] Context text index: {context_idx}", flush=True)
    if CONTEXT_AUDIO_PATH:
        print(f"[INFO] Using context audio: {CONTEXT_AUDIO_PATH}", flush=True)

    app = make_app(
        model,
        base_state,
        decode_device=decode_device,
        decode_codec_helper=decode_codec_helper,
        autocast_dtype=autocast_dtype,
    )
    app.queue().launch(server_name="0.0.0.0", server_port=7861, share=False)


if __name__ == "__main__":
    main()
