#!/usr/bin/env python3
"""Minimal Flask WebSocket streaming demo for EasyMagpieTTS.

Frontend responsibilities:
- Open websocket
- Simulate dummy LLM by sending text chunks + eos

Backend responsibilities:
- Run EasyMagpie streaming_step as text tokens arrive
- Decode audio codes asynchronously
- Push partial text/status as JSON and audio chunks as raw PCM messages
"""

import contextlib
import csv
import json
import os
import queue
import sys
import threading
import time
import types
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_sock import Sock
from omegaconf import open_dict
from werkzeug.utils import secure_filename

# Avoid megatron unified_memory JIT path for this inference script.
_um = types.ModuleType("megatron.core.inference.unified_memory")
_um.has_unified_memory = False
_um.create_unified_mempool = None
sys.modules["megatron.core.inference.unified_memory"] = _um

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.utils import logging


os.environ["OMP_NUM_THREADS"] = "2"

# ----------------------------
# Editable defaults (no CLI)
# ----------------------------
MODEL_PATH = "/datap/misc/EasyMagpieAssets/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay.nemo"
CODEC_MODEL_PATH = "/datap/misc/EasyMagpieAssets/25fps_spectral_codec_with_bandwidth_extension.nemo"
PHONEME_TOKENIZER_PATH = "/datap/misc/EasyMagpieAssets/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json"
CONTEXT_TEXTS_PATH = "/datap/misc/EasyMagpieAssets/unique_text_contexts.txt"
DEFAULT_CONTEXT_TEXT_INDEX = 0

CONTEXT_AUDIO_PATH: Optional[str] = None
CONTEXT_AUDIO_DURATION_SEC = 5.0
LANGUAGE = "en"
UPLOAD_DIR = os.path.abspath("./tmp/easymagpie_uploads")

USE_CFG = True
CFG_SCALE = 2.5
USE_LOCAL_TRANSFORMER = True
TEMPERATURE = 0.5
TOPK = 80
MAX_DECODER_STEPS = 500
PHONEME_INPUT_TYPE = "pred"
PHONEME_SAMPLING_METHOD = "argmax"

DECODE_EVERY_AUDIO_FRAMES = 6
DECODE_POLL_INTERVAL_SEC = 0.01
MAIN_LOOP_IDLE_SLEEP_SEC = 0.002
WARMUP_ENABLED = True
WARMUP_TEXT = "This is a startup warmup run for local transformer and generation path."
WARMUP_MAX_DECODER_STEPS = 80

GENERATION_GPU_INDEX = 0
DECODE_GPU_INDEX = 1
MODEL_PRECISION = "bf16"  # bf16 | fp16 | fp32
LOCAL_TRANSFORMER_BACKEND = "trt"  # torch | trt

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7862


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
    decode_idx = DECODE_GPU_INDEX if DECODE_GPU_INDEX < num_gpus else generation_idx
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
        context_audio = model._load_audio_for_inference(context_audio_path, model.sample_rate)  # noqa: SLF001
        context_audio = model._adjust_audio_to_duration_for_inference(  # noqa: SLF001
            context_audio,
            model.sample_rate,
            CONTEXT_AUDIO_DURATION_SEC,
            model.codec_model_samples_per_frame,
        )
        context_audio = context_audio.to(device=device, dtype=model_dtype)
        context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
        with torch.inference_mode():
            context_audio_codes, context_audio_codes_lens = model._codec_helper.audio_to_codes(context_audio, context_audio_lens)  # noqa: SLF001
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
    context_audio_path: Optional[str],
    inference_options: dict,
    generation_device: torch.device,
    autocast_dtype: Optional[torch.dtype],
):
    context_audio_codes, context_audio_codes_lens, context_text_tokens, context_text_lens = build_context_inputs(
        model, context_text, context_audio_path
    )
    with torch.inference_mode():
        with autocast_context(generation_device, autocast_dtype):
            state = model.streaming_init(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_lens,
                use_cfg=bool(inference_options["use_cfg"]),
                cfg_scale=float(inference_options["cfg_scale"]),
                use_local_transformer=bool(inference_options["use_local_transformer"]),
                temperature=float(inference_options["temperature"]),
                topk=int(inference_options["topk"]),
                phoneme_input_type=PHONEME_INPUT_TYPE,
                phoneme_sampling_method=PHONEME_SAMPLING_METHOD,
                use_inference_mode=True,
            )
    return state


def clone_streaming_state(state):
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

    print("[WARMUP] Starting warmup run before launching server...", flush=True)
    t0 = time.perf_counter()
    state = clone_streaming_state(base_state)
    main_tokenizer_name = list(model.cfg.text_tokenizers.keys())[0]
    token_ids = model.tokenizer.encode(WARMUP_TEXT, tokenizer_name=main_tokenizer_name)
    token_ids.append(model.eos_id)
    pending_token_ids: Deque[int] = deque(token_ids)
    device = next(model.parameters()).device
    steps = 0

    with torch.inference_mode():
        while not bool(state.finished.all()) and steps < WARMUP_MAX_DECODER_STEPS:
            text_tokens = (
                torch.tensor([pending_token_ids.popleft()], dtype=torch.long, device=device) if pending_token_ids else None
            )
            if autocast_dtype is None:
                state, _audio_codes, _phoneme_tokens = model.streaming_step(
                    state=state,
                    text_tokens=text_tokens,
                    use_inference_mode=True,
                )
            else:
                with autocast_context(generation_device, autocast_dtype):
                    state, _audio_codes, _phoneme_tokens = model.streaming_step(
                        state=state,
                        text_tokens=text_tokens,
                        use_inference_mode=True,
                    )
            steps += 1

    print(f"[WARMUP] Completed in {time.perf_counter() - t0:.2f}s.", flush=True)


def _to_int16_pcm(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return np.zeros((0,), dtype=np.int16)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


def _pcm_bytes_from_int16(pcm: np.ndarray) -> bytes:
    pcm_i16 = np.asarray(pcm, dtype=np.int16).reshape(-1)
    return pcm_i16.tobytes()


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
                model=model,
                decode_codec_helper=decode_codec_helper,
                all_audio_codes=codes_snapshot,
                last_emitted_sample_idx=last_emitted_sample_idx,
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


class EasyMagpieWsServer:
    def __init__(self):
        os.environ["EASYMAGPIE_LT_BACKEND"] = LOCAL_TRANSFORMER_BACKEND
        self.generation_device, self.decode_device = resolve_devices()
        self.autocast_dtype = resolve_autocast_dtype(MODEL_PRECISION, self.generation_device)
        self.model = load_model(self.generation_device)
        self.decode_codec_helper = build_decode_codec_helper(self.model, self.decode_device)
        self.context_texts = load_context_texts(CONTEXT_TEXTS_PATH)
        context_idx = max(0, min(DEFAULT_CONTEXT_TEXT_INDEX, len(self.context_texts) - 1))
        self.default_context_text = self.context_texts[context_idx]
        self.default_context_audio_path = CONTEXT_AUDIO_PATH
        self.default_inference_options = {
            "use_cfg": bool(USE_CFG),
            "cfg_scale": float(CFG_SCALE),
            "temperature": float(TEMPERATURE),
            "topk": int(TOPK),
            "use_local_transformer": bool(USE_LOCAL_TRANSFORMER),
        }
        if self.default_context_audio_path:
            use_language_tag = bool(getattr(self.model, "add_language_to_context_text", False))
            self.default_context_text = f"[{LANGUAGE.upper()}]" if use_language_tag else "[NO TEXT CONTEXT]"

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.upload_dir = UPLOAD_DIR

        self.base_state = create_base_streaming_state(
            model=self.model,
            context_text=self.default_context_text,
            context_audio_path=self.default_context_audio_path,
            inference_options=self.default_inference_options,
            generation_device=self.generation_device,
            autocast_dtype=self.autocast_dtype,
        )
        run_startup_warmup(
            model=self.model,
            base_state=self.base_state,
            generation_device=self.generation_device,
            autocast_dtype=self.autocast_dtype,
        )
        self.main_tokenizer_name = list(self.model.cfg.text_tokenizers.keys())[0]
        self.device = next(self.model.parameters()).device
        self.sr = int(self.model.output_sample_rate)
        self.state_lock = threading.Lock()

        print(f"[INFO] Generation device: {self.generation_device}", flush=True)
        print(f"[INFO] Decode device: {self.decode_device}", flush=True)
        print(f"[INFO] Generation AMP dtype: {self.autocast_dtype}", flush=True)
        print(f"[INFO] Local transformer backend: {LOCAL_TRANSFORMER_BACKEND}", flush=True)

    def _send_json(self, ws, payload: dict):
        ws.send(json.dumps(payload))

    @staticmethod
    def _normalize_inference_options(base: dict, override: Optional[dict]) -> dict:
        out = dict(base)
        if not isinstance(override, dict):
            return out
        if "use_cfg" in override:
            out["use_cfg"] = bool(override["use_cfg"])
        if "cfg_scale" in override:
            out["cfg_scale"] = float(override["cfg_scale"])
        if "temperature" in override:
            out["temperature"] = float(override["temperature"])
        if "topk" in override:
            out["topk"] = int(override["topk"])
        if "use_local_transformer" in override:
            out["use_local_transformer"] = bool(override["use_local_transformer"])
        out["cfg_scale"] = max(0.0, out["cfg_scale"])
        out["temperature"] = max(0.01, out["temperature"])
        out["topk"] = max(1, out["topk"])
        return out

    def _resolve_context_audio_path(self, audio_id: Optional[str]) -> Optional[str]:
        if not audio_id:
            return None
        safe_name = os.path.basename(audio_id)
        path = os.path.join(self.upload_dir, safe_name)
        return path if os.path.exists(path) else None

    def _build_state_for_context(self, context_text: str, context_audio_path: Optional[str], inference_options: dict):
        return create_base_streaming_state(
            model=self.model,
            context_text=context_text,
            context_audio_path=context_audio_path,
            inference_options=inference_options,
            generation_device=self.generation_device,
            autocast_dtype=self.autocast_dtype,
        )

    def _handle_connection(self, ws):
        events: "queue.Queue[dict]" = queue.Queue()

        def receiver():
            try:
                while True:
                    msg = ws.receive()
                    if msg is None:
                        events.put({"type": "disconnect"})
                        return
                    if isinstance(msg, bytes):
                        continue
                    try:
                        parsed = json.loads(msg)
                    except json.JSONDecodeError:
                        events.put({"type": "invalid", "error": "Invalid JSON payload"})
                        continue
                    events.put(parsed)
            except Exception as exc:  # noqa: BLE001
                events.put({"type": "disconnect", "error": str(exc)})

        recv_thread = threading.Thread(target=receiver, daemon=True)
        recv_thread.start()

        with self.state_lock:
            session_base_state = clone_streaming_state(self.base_state)
            state = clone_streaming_state(session_base_state)
        session_context_text = self.default_context_text
        session_context_audio_path = self.default_context_audio_path
        session_inference_options = dict(self.default_inference_options)

        shared = DecodeSharedState()
        decode_thread = threading.Thread(
            target=decode_worker, args=(self.model, self.decode_codec_helper, shared), daemon=True
        )
        decode_thread.start()

        text_buffer = ""
        partial_text = ""
        sent_token_ids: List[int] = []
        pending_token_ids: Deque[int] = deque()
        llm_done = False
        started = False
        steps = 0
        should_stop = False

        self._send_json(
            ws,
            {
                "type": "status",
                "state": "connected",
                "sample_rate": self.sr,
                "audio_format": "pcm_s16le",
                "channels": 1,
            },
        )

        try:
            with torch.inference_mode():
                while not should_stop and steps < MAX_DECODER_STEPS:
                    while True:
                        try:
                            event = events.get_nowait()
                        except queue.Empty:
                            break

                        etype = event.get("type")
                        if etype == "disconnect":
                            should_stop = True
                            break
                        if etype == "invalid":
                            self._send_json(ws, {"type": "status", "state": "error", "detail": event.get("error", "")})
                            continue
                        if etype == "start":
                            started = True
                            text_buffer = ""
                            partial_text = ""
                            sent_token_ids.clear()
                            pending_token_ids.clear()
                            llm_done = False
                            steps = 0
                            session_inference_options = self._normalize_inference_options(
                                session_inference_options, event.get("inference")
                            )
                            with self.state_lock:
                                session_base_state = self._build_state_for_context(
                                    context_text=session_context_text,
                                    context_audio_path=session_context_audio_path,
                                    inference_options=session_inference_options,
                                )
                                state = clone_streaming_state(session_base_state)
                            self._send_json(ws, {"type": "status", "state": "running"})
                            continue
                        if etype == "reset":
                            text_buffer = ""
                            partial_text = ""
                            sent_token_ids.clear()
                            pending_token_ids.clear()
                            llm_done = False
                            steps = 0
                            requested_idx = event.get("context_text_index")
                            if isinstance(requested_idx, int) and 0 <= requested_idx < len(self.context_texts):
                                session_context_text = self.context_texts[requested_idx]

                            requested_audio_id = event.get("context_audio_id")
                            use_context_audio = bool(event.get("use_context_audio", False))
                            resolved_audio = self._resolve_context_audio_path(requested_audio_id)
                            session_context_audio_path = resolved_audio if use_context_audio and resolved_audio else None

                            session_inference_options = self._normalize_inference_options(
                                session_inference_options, event.get("inference")
                            )
                            with self.state_lock:
                                session_base_state = self._build_state_for_context(
                                    context_text=session_context_text,
                                    context_audio_path=session_context_audio_path,
                                    inference_options=session_inference_options,
                                )
                                state = clone_streaming_state(session_base_state)
                            self._send_json(
                                ws,
                                {
                                    "type": "status",
                                    "state": "reset",
                                    "context_audio_active": bool(session_context_audio_path),
                                },
                            )
                            continue
                        if etype == "text_chunk":
                            if not started:
                                continue
                            chunk = event.get("text", "")
                            if chunk:
                                text_buffer += chunk
                                partial_text += chunk
                                token_ids = self.model.tokenizer.encode(
                                    text_buffer, tokenizer_name=self.main_tokenizer_name
                                )
                                new_ids = token_ids[len(sent_token_ids) :]
                                for tid in new_ids:
                                    pending_token_ids.append(tid)
                                sent_token_ids.extend(new_ids)
                                self._send_json(ws, {"type": "partial_text", "text": partial_text})
                            continue
                        if etype == "eos":
                            if started and not llm_done:
                                llm_done = True
                                pending_token_ids.append(self.model.eos_id)
                            continue

                    if should_stop:
                        break

                    if started and not bool(state.finished.all()):
                        if pending_token_ids:
                            tok = pending_token_ids.popleft()
                            text_tokens = torch.tensor([tok], dtype=torch.long, device=self.device)
                        elif llm_done:
                            text_tokens = None
                        else:
                            text_tokens = None

                        if text_tokens is not None or llm_done:
                            if self.autocast_dtype is None:
                                state, audio_codes, _phoneme_tokens = self.model.streaming_step(
                                    state=state,
                                    text_tokens=text_tokens,
                                    use_inference_mode=True,
                                )
                            else:
                                with autocast_context(self.generation_device, self.autocast_dtype):
                                    state, audio_codes, _phoneme_tokens = self.model.streaming_step(
                                        state=state,
                                        text_tokens=text_tokens,
                                        use_inference_mode=True,
                                    )
                            steps += 1

                            if audio_codes is not None:
                                new_codes = audio_codes.detach().to(
                                    device=self.decode_device, dtype=torch.long, non_blocking=True
                                )
                                with shared.lock:
                                    if shared.accumulated_audio_codes is None:
                                        shared.accumulated_audio_codes = new_codes
                                    else:
                                        shared.accumulated_audio_codes = torch.cat(
                                            [shared.accumulated_audio_codes, new_codes], dim=-1
                                        )
                                    shared.generated_audio_frames = int(shared.accumulated_audio_codes.size(-1))

                            if bool(state.finished.all()):
                                self._send_json(ws, {"type": "status", "state": "generation_finished"})

                    with shared.lock:
                        pending_chunks = list(shared.decoded_chunks)
                        shared.decoded_chunks.clear()
                    for chunk in pending_chunks:
                        if chunk.size > 0:
                            ws.send(_pcm_bytes_from_int16(chunk))

                    if started and llm_done and bool(state.finished.all()):
                        break

                    time.sleep(MAIN_LOOP_IDLE_SLEEP_SEC)

            with shared.lock:
                shared.stop = True
            decode_thread.join()

            with shared.lock:
                pending_chunks = list(shared.decoded_chunks)
                shared.decoded_chunks.clear()
            for chunk in pending_chunks:
                if chunk.size > 0:
                    ws.send(_pcm_bytes_from_int16(chunk))

            self._send_json(ws, {"type": "status", "state": "done"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(ws, {"type": "status", "state": "error", "detail": str(exc)})

    def make_flask_app(self) -> Flask:
        app = Flask(__name__, static_folder="static")
        sock = Sock(app)

        @app.get("/")
        def index():
            return send_from_directory(app.static_folder, "index.html")

        @app.get("/api/context_texts")
        def context_texts():
            return jsonify({"items": self.context_texts})

        @app.post("/api/upload_reference_audio")
        def upload_reference_audio():
            if "file" not in request.files:
                return jsonify({"error": "Missing file field"}), 400
            file = request.files["file"]
            if not file or not file.filename:
                return jsonify({"error": "Empty upload"}), 400
            safe_name = secure_filename(file.filename)
            if not safe_name:
                return jsonify({"error": "Invalid filename"}), 400
            stamped_name = f"{int(time.time() * 1000)}_{safe_name}"
            dst = os.path.join(self.upload_dir, stamped_name)
            file.save(dst)
            return jsonify({"audio_id": stamped_name})

        @sock.route("/ws/stream")
        def stream(ws):
            self._handle_connection(ws)

        return app


def main():
    server = EasyMagpieWsServer()
    app = server.make_flask_app()
    print(f"[INFO] Starting Flask+WebSocket server at http://{SERVER_HOST}:{SERVER_PORT}", flush=True)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
