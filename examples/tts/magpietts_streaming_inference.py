# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MagpieTTS Streaming Inference Test Script.

This script tests the streaming TTS inference functionality for batch_size=1.
It loads a model, processes context audio/text, and generates audio by feeding
text tokens one at a time through the streaming interface.

Example usage:
    # From checkpoint
    python examples/tts/magpietts_streaming_inference.py \
        --hparams_file /path/to/hparams.yaml \
        --checkpoint_file /path/to/model.ckpt \
        --codecmodel_path /path/to/codec.nemo \
        --context_audio /path/to/context.wav \
        --text "Hello, this is a test of streaming TTS inference." \
        --output_path /path/to/output.wav

    # From .nemo file
    python examples/tts/magpietts_streaming_inference.py \
        --nemo_file /path/to/model.nemo \
        --codecmodel_path /path/to/codec.nemo \
        --context_audio /path/to/context.wav \
        --text "Hello, this is a test of streaming TTS inference." \
        --output_path /path/to/output.wav
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models import EasyMagpieTTSModel
from nemo.utils import logging


def load_model(
    hparams_file: Optional[str],
    checkpoint_file: Optional[str],
    nemo_file: Optional[str],
    codecmodel_path: str,
    device: str = "cuda",
) -> EasyMagpieTTSModel:
    """
    Load an EasyMagpieTTSModel from checkpoint or .nemo file.

    Args:
        hparams_file: Path to hparams.yaml (required with checkpoint_file).
        checkpoint_file: Path to .ckpt file (required with hparams_file).
        nemo_file: Path to .nemo file (alternative to hparams + checkpoint).
        codecmodel_path: Path to the audio codec model.
        device: Device to load model on.

    Returns:
        Loaded model ready for inference.
    """
    if hparams_file is not None and checkpoint_file is not None:
        # Load from hparams + checkpoint
        logging.info(f"Loading model from checkpoint: {checkpoint_file}")
        model_cfg = OmegaConf.load(hparams_file)

        # Handle different config structures
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        with open_dict(model_cfg):
            # Override codec model path
            model_cfg.codecmodel_path = codecmodel_path

            # Disable training datasets
            model_cfg.train_ds = None
            model_cfg.validation_ds = None

        model = EasyMagpieTTSModel(cfg=model_cfg)

        # Load weights
        ckpt = torch.load(checkpoint_file, weights_only=False)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)

    elif nemo_file is not None:
        # Load from .nemo file
        logging.info(f"Loading model from NeMo archive: {nemo_file}")
        model_cfg = EasyMagpieTTSModel.restore_from(nemo_file, return_config=True)

        with open_dict(model_cfg):
            model_cfg.codecmodel_path = codecmodel_path
            model_cfg.train_ds = None
            model_cfg.validation_ds = None

        model = EasyMagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)

    else:
        raise ValueError("Must provide either (hparams_file + checkpoint_file) or nemo_file")

    model.to(device)
    model.eval()
    logging.info("Model loaded and ready for streaming inference.")

    return model


def load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file.
        target_sample_rate: Target sample rate.

    Returns:
        Audio tensor of shape (1, num_samples).
    """
    audio, sr = sf.read(audio_path, dtype='float32')

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)

    return torch.from_numpy(audio).unsqueeze(0)  # (1, num_samples)


def get_num_audio_samples_to_slice(duration: float, sample_rate: int, codec_model_samples_per_frame: int) -> int:
    """
    Get the precise number of audio samples to slice for a given duration.

    This ensures the audio is sliced to an exact number of samples that aligns
    with the codec model's frame size.

    Args:
        duration: Target duration in seconds.
        sample_rate: Sample rate of the audio.
        codec_model_samples_per_frame: Number of samples per codec frame (codec downsample factor).

    Returns:
        Number of audio samples aligned to codec frame boundaries.
    """
    num_codec_frames = int(duration * sample_rate / codec_model_samples_per_frame)
    num_audio_samples = num_codec_frames * codec_model_samples_per_frame
    return num_audio_samples


def adjust_audio_to_duration(
    audio: torch.Tensor,
    sample_rate: int,
    target_duration: float,
    codec_model_samples_per_frame: int,
) -> torch.Tensor:
    """
    Adjust audio to exactly target_duration seconds, aligned to codec frame boundaries.

    If audio is longer than target_duration, take the first target_duration seconds.
    If audio is shorter, repeat it until it reaches target_duration seconds.
    The resulting length is aligned to codec frame boundaries.

    Args:
        audio: Audio tensor of shape (1, num_samples).
        sample_rate: Sample rate of the audio.
        target_duration: Target duration in seconds.
        codec_model_samples_per_frame: Number of samples per codec frame (codec downsample factor).

    Returns:
        Audio tensor of shape (1, target_num_samples) where
        target_num_samples is aligned to codec frame boundaries.
    """
    target_num_samples = get_num_audio_samples_to_slice(target_duration, sample_rate, codec_model_samples_per_frame)
    current_num_samples = audio.size(1)

    if current_num_samples >= target_num_samples:
        # Audio is longer than target - take the first target_duration seconds
        audio = audio[:, :target_num_samples]
    else:
        # Audio is shorter - repeat until we have enough samples
        num_repeats = int(np.ceil(target_num_samples / current_num_samples))
        audio_repeated = audio.repeat(1, num_repeats)
        audio = audio_repeated[:, :target_num_samples]

    return audio


def run_streaming_inference(
    model: EasyMagpieTTSModel,
    context_audio: torch.Tensor,
    context_audio_lens: torch.Tensor,
    context_text: str,
    text: str,
    inference_mode: Optional[str] = None,
    use_cfg: bool = False,
    cfg_scale: float = 1.5,
    use_local_transformer: bool = False,
    temperature: float = 0.7,
    topk: int = 80,
    max_steps: int = 500,
    verbose: bool = True,
) -> tuple:
    """
    Run streaming TTS inference.

    Args:
        model: The loaded EasyMagpieTTSModel.
        context_audio: Context audio tensor (1, num_samples).
        context_audio_lens: Length of context audio (1,).
        context_text: Context text for speaker conditioning.
        text: Main text to synthesize.
        inference_mode: Inference mode name (e.g., "streaming_4_8").
        use_cfg: Whether to use classifier-free guidance.
        cfg_scale: CFG scale factor.
        use_local_transformer: Whether to use local transformer.
        temperature: Sampling temperature.
        topk: Top-k sampling parameter.
        max_steps: Maximum generation steps.
        verbose: Whether to print progress.

    Returns:
        Tuple of (audio, audio_len, codes, codes_len, timing_info).
    """
    device = next(model.parameters()).device

    # Encode context audio to codes
    context_audio = context_audio.to(device)
    context_audio_lens = context_audio_lens.to(device)

    with torch.inference_mode():
        context_audio_codes, context_audio_codes_lens = model.audio_to_codes(
            context_audio, context_audio_lens
        )

    # Tokenize context text
    # Use the text conditioning tokenizer
    tokenizer_name = model.text_conditioning_tokenizer_name
    context_text_tokens = model.tokenizer.encode(context_text, tokenizer_name=tokenizer_name)
    context_text_tokens = torch.tensor([context_text_tokens], dtype=torch.long, device=device)
    context_text_tokens_lens = torch.tensor([context_text_tokens.size(1)], dtype=torch.long, device=device)

    # Tokenize main text
    # Get the appropriate tokenizer name for main text
    if hasattr(model.tokenizer, 'tokenizers') and 'english_phoneme' in model.tokenizer.tokenizers:
        main_tokenizer_name = 'english_phoneme'
    else:
        main_tokenizer_name = tokenizer_name

    text_tokens = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name)
    text_tokens = text_tokens + [model.eos_id]
    text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

    # Get streaming delays for logging
    mode_name = inference_mode or model.default_inference_mode
    training_mode = model.mode_name_to_mode.get(mode_name, model.training_modes[0])
    phoneme_delay = training_mode.streaming_phonemes_delay
    speech_delay = training_mode.streaming_speech_delay

    if verbose:
        logging.info(f"Context audio codes shape: {context_audio_codes.shape}")
        logging.info(f"Context text tokens: {context_text_tokens.shape}")
        logging.info(f"Main text tokens: {text_tokens.shape} ({len(text_tokens)} tokens)")
        logging.info(f"Using inference mode: {mode_name}")
        logging.info(f"Phoneme delay: {phoneme_delay}, Speech delay: {speech_delay}")
        logging.info("Phases: Prompt (0 to phoneme_delay) -> Phoneme-only (phoneme_delay to speech_delay) -> Audio")

    # Initialize streaming state
    start_time = time.time()

    state = model.streaming_init(
        context_audio_codes=context_audio_codes,
        context_audio_codes_lens=context_audio_codes_lens,
        context_text_tokens=context_text_tokens,
        context_text_tokens_lens=context_text_tokens_lens,
        inference_mode=inference_mode,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
        use_local_transformer=use_local_transformer,
        temperature=temperature,
        topk=topk,
    )

    init_time = time.time() - start_time
    if verbose:
        logging.info(f"Streaming init completed in {init_time:.3f}s")

    # Feed text tokens one at a time
    generation_start = time.time()
    num_audio_frames = 0
    num_phoneme_frames = 0
    prompt_phase_tokens = 0
    phoneme_only_phase_tokens = 0

    for i, token in enumerate(text_tokens):
        state, audio_codes, phoneme_tokens = model.streaming_step(
            state, text_token=token.unsqueeze(0)
        )

        # Track which phase we're in
        if audio_codes is None and phoneme_tokens is None:
            prompt_phase_tokens += 1
        elif audio_codes is None and phoneme_tokens is not None:
            phoneme_only_phase_tokens += 1
            num_phoneme_frames += 1
        else:
            if audio_codes is not None:
                num_audio_frames += 1
            if phoneme_tokens is not None:
                num_phoneme_frames += 1

        if verbose and (i + 1) % 10 == 0:
            phase = "prompt" if audio_codes is None and phoneme_tokens is None else (
                "phoneme-only" if audio_codes is None else "audio"
            )
            logging.info(
                f"Processed {i + 1}/{len(text_tokens)} text tokens (phase: {phase}), "
                f"audio frames: {num_audio_frames}, phoneme frames: {num_phoneme_frames}"
            )

        if state.finished:
            if verbose:
                logging.info(f"EOS detected at text token {i + 1}")
            break

    # Continue generating until finished (text has ended)
    continuation_steps = 0
    while not state.finished and continuation_steps < max_steps:
        state, audio_codes, phoneme_tokens = model.streaming_step(state, text_token=None)

        if audio_codes is not None:
            num_audio_frames += 1
        if phoneme_tokens is not None:
            num_phoneme_frames += 1

        continuation_steps += 1

        if verbose and continuation_steps % 20 == 0:
            logging.info(
                f"Continuation step {continuation_steps}, "
                f"audio frames: {num_audio_frames}, phoneme frames: {num_phoneme_frames}"
            )

    generation_time = time.time() - generation_start

    if verbose:
        logging.info(f"Generation completed in {generation_time:.3f}s")
        logging.info(f"Prompt phase tokens: {prompt_phase_tokens}")
        logging.info(f"Phoneme-only phase tokens: {phoneme_only_phase_tokens}")
        logging.info(f"Audio frames generated: {num_audio_frames}")
        logging.info(f"Phoneme frames generated: {num_phoneme_frames}")
        logging.info(f"Continuation steps: {continuation_steps}")

    # Finalize and get complete audio
    audio, audio_len, codes, codes_len = model.streaming_finalize(state)

    total_time = time.time() - start_time

    timing_info = {
        'init_time': init_time,
        'generation_time': generation_time,
        'total_time': total_time,
        'num_text_tokens': len(text_tokens),
        'prompt_phase_tokens': prompt_phase_tokens,
        'phoneme_only_phase_tokens': phoneme_only_phase_tokens,
        'num_audio_frames': num_audio_frames,
        'num_phoneme_frames': num_phoneme_frames,
        'continuation_steps': continuation_steps,
    }

    return audio, audio_len, codes, codes_len, timing_info


def main():
    parser = argparse.ArgumentParser(
        description="MagpieTTS Streaming Inference Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model loading arguments
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument(
        '--hparams_file',
        type=str,
        default=None,
        help='Path to hparams.yaml file',
    )
    model_group.add_argument(
        '--checkpoint_file',
        type=str,
        default=None,
        help='Path to .ckpt checkpoint file',
    )
    model_group.add_argument(
        '--nemo_file',
        type=str,
        default=None,
        help='Path to .nemo model file',
    )
    model_group.add_argument(
        '--codecmodel_path',
        type=str,
        required=True,
        help='Path to audio codec model (.nemo)',
    )

    # Input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--context_audio',
        type=str,
        required=True,
        help='Path to context audio file for speaker cloning',
    )
    input_group.add_argument(
        '--context_text',
        type=str,
        default="[NO TEXT CONTEXT]",
        help='Context text for speaker conditioning (default: "[NO TEXT CONTEXT]")',
    )
    input_group.add_argument(
        '--context_duration',
        type=float,
        default=5.0,
        help='Target duration for context audio in seconds. If audio is longer, '
             'first N seconds are used. If shorter, audio is repeated. (default: 5.0)',
    )
    input_group.add_argument(
        '--text',
        type=str,
        required=True,
        help='Text to synthesize',
    )

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_path',
        type=str,
        default='streaming_output.wav',
        help='Path for output audio file',
    )

    # Inference arguments
    infer_group = parser.add_argument_group('Inference Parameters')
    infer_group.add_argument(
        '--inference_mode',
        type=str,
        default=None,
        help='Inference mode name (e.g., "streaming_4_8"). Uses model default if not specified.',
    )
    infer_group.add_argument(
        '--use_cfg',
        action='store_true',
        help='Enable classifier-free guidance',
    )
    infer_group.add_argument(
        '--cfg_scale',
        type=float,
        default=1.5,
        help='CFG scale factor (higher = stronger conditioning)',
    )
    infer_group.add_argument(
        '--use_local_transformer',
        action='store_true',
        help='Use local transformer for inference',
    )
    infer_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature',
    )
    infer_group.add_argument(
        '--topk',
        type=int,
        default=80,
        help='Top-k sampling parameter',
    )
    infer_group.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='Maximum generation steps after text ends',
    )
    infer_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on',
    )
    infer_group.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information',
    )

    args = parser.parse_args()

    # Validate arguments
    has_ckpt_mode = args.hparams_file is not None and args.checkpoint_file is not None
    has_nemo_mode = args.nemo_file is not None

    if not (has_ckpt_mode or has_nemo_mode):
        parser.error("Must provide either (--hparams_file and --checkpoint_file) or --nemo_file")

    # Load model
    model = load_model(
        hparams_file=args.hparams_file,
        checkpoint_file=args.checkpoint_file,
        nemo_file=args.nemo_file,
        codecmodel_path=args.codecmodel_path,
        device=args.device,
    )

    model = model.float()

    # Load context audio
    logging.info(f"Loading context audio from: {args.context_audio}")
    context_audio = load_audio(args.context_audio, model.sample_rate)
    original_duration = context_audio.size(1) / model.sample_rate
    logging.info(f"Original context audio duration: {original_duration:.2f}s")

    # Adjust context audio to target duration (aligned to codec frame boundaries)
    context_audio = adjust_audio_to_duration(
        context_audio,
        sample_rate=model.sample_rate,
        target_duration=args.context_duration,
        codec_model_samples_per_frame=model.codec_model_samples_per_frame,
    )
    context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long)
    adjusted_duration = context_audio.size(1) / model.sample_rate

    logging.info(f"Adjusted context audio to {adjusted_duration:.2f}s (target: {args.context_duration}s)")
    logging.info(f"Context audio shape: {context_audio.shape}, sample_rate={model.sample_rate}")
    logging.info(f"Context text: {args.context_text}")
    logging.info(f"Text to synthesize: {args.text}")

    # Run streaming inference
    audio, audio_len, codes, codes_len, timing_info = run_streaming_inference(
        model=model,
        context_audio=context_audio,
        context_audio_lens=context_audio_lens,
        context_text=args.context_text,
        text=args.text,
        inference_mode=args.inference_mode,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        use_local_transformer=args.use_local_transformer,
        temperature=args.temperature,
        topk=args.topk,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Save output
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_np = audio[0, :audio_len[0].item()].cpu().numpy()
    sf.write(args.output_path, audio_np, model.sample_rate)

    logging.info(f"Output saved to: {args.output_path}")
    logging.info(f"Audio duration: {audio_len[0].item() / model.sample_rate:.2f}s")
    logging.info(f"Generated codes shape: {codes.shape}")

    # Print timing summary
    logging.info("\n=== Timing Summary ===")
    logging.info(f"Init time: {timing_info['init_time']:.3f}s")
    logging.info(f"Generation time: {timing_info['generation_time']:.3f}s")
    logging.info(f"Total time: {timing_info['total_time']:.3f}s")
    logging.info(f"Text tokens processed: {timing_info['num_text_tokens']}")
    logging.info(f"  - Prompt phase tokens: {timing_info['prompt_phase_tokens']}")
    logging.info(f"  - Phoneme-only phase tokens: {timing_info['phoneme_only_phase_tokens']}")
    logging.info(f"Audio frames generated: {timing_info['num_audio_frames']}")
    logging.info(f"Phoneme frames generated: {timing_info['num_phoneme_frames']}")
    logging.info(f"Continuation steps: {timing_info['continuation_steps']}")

    # Calculate RTF
    audio_duration = audio_len[0].item() / model.sample_rate
    rtf = audio_duration / timing_info['total_time']
    logging.info(f"Real-time factor (RTF): {rtf:.2f}x")


if __name__ == "__main__":
    main()
