# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Evaluation script for Duplex EARTTS models.

Args:
    config-path (str): Path to the directory containing the YAML configuration file.
    config-name (str): Name of the YAML configuration file.
    checkpoint_path (str): Path to the Duplex EARTTS checkpoint file.

Usage:
    python duplex_eartts_inf.py \
        --config-path=conf/ \
        --config-name=duplex_eartts.yaml \
        ++checkpoint_path=duplex_eartts_results/duplex_eartts/model.ckpt \
        ++datasets_json_path=/path/to/evalset_config.json \
        ++out_dir=duplex_eartts_results/duplex_eartts/audio_samples/dummy_dataset
"""

import os
import soundfile as sf
from nemo.collections.audio.parts.utils.resampling import resample
import torch
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexEARTTSDataset

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.speechlm2.parts.metrics.asr_cer_wer import Intelligibility

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

import json

def read_jsonl_batches_old(file_path, batch_size, drop_last=False):
    """
    Reads a JSONL file and yields batches of size batch_size.

    Args:
        file_path (str): Path to the JSONL file
        batch_size (int): Number of samples per batch
        drop_last (bool): If True, drop the last incomplete batch

    Yields:
        List[dict]: A batch of samples
    """
    batch = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

            batch.append(sample)

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch and not drop_last:
        yield batch


def read_jsonl_batches(
    file_path,
    batch_size,
    drop_last=False,
    max_batches=None,   # <-- DEBUG OPTION
):
    """
    Reads a JSONL file and yields batches of size batch_size.

    Args:
        file_path (str): Path to the JSONL file
        batch_size (int): Number of samples per batch
        drop_last (bool): If True, drop the last incomplete batch
        max_batches (int or None): If set, only yield this many batches (debug mode)

    Yields:
        List[dict]: A batch of samples
    """
    batch = []
    num_batches = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx}: {e}")

            batch.append(sample)

            if len(batch) == batch_size:
                yield batch
                batch = []
                num_batches += 1

                # --- DEBUG STOP ---
                if max_batches is not None and num_batches >= max_batches:
                    return

    if batch and not drop_last:
        yield batch

import os
import torch
import librosa
from torch.nn.utils.rnn import pad_sequence

def collate_and_tokenize_custom(
    batch,
    model,
    extra_duration_thrshould=1.3,
    sample_rate=22050,
    root_path=None
):
    test_sentences = [s["text"] for s in batch]

    # --- TEXT TOKENIZATION ---
    tokenized = [
        torch.as_tensor(
            [model.tokenizer.bos] + model.tokenizer.text_to_ids(text),
            dtype=torch.long,
            device=model.device
        )
        for text in test_sentences
    ]

    input_ids = pad_sequence(
        tokenized,
        batch_first=True,
        padding_value=model.text_pad_id
    )

    # --- AUDIO LOADING ---
    audio_list = []
    audio_lengths = []
    target_num_frames = []

    for s in batch:
        audio_path = s["context_audio_filepath"]
        if root_path is not None:
            audio_path = os.path.join(root_path, audio_path)

        wav, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        wav = torch.as_tensor(wav, dtype=torch.float32)
        audio_list.append(wav)
        audio_lengths.append(len(wav))

        # target duration
        tdur_audio_path = s["audio_filepath"]
        if root_path is not None:
            tdur_audio_path = os.path.join(root_path, tdur_audio_path)

        wav_dur, sr_ = librosa.load(tdur_audio_path, sr=sample_rate, mono=True)
        tdur = wav_dur.shape[0] // model.target_samples_per_frame 
        target_num_frames.append(tdur * extra_duration_thrshould)

    max_audio_len = max(audio_lengths)
    B = len(audio_lengths)

    padded_audio = torch.zeros(
        (B, max_audio_len),
        dtype=torch.float32
    )

    for i, wav in enumerate(audio_list):
        padded_audio[i, :len(wav)] = wav

    padded_audio = padded_audio.to(model.device)

    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    # Expand text length to match speech
    B, L = input_ids.shape
    target_len = int(max(target_num_frames))

    padded_input_ids = torch.full(
        (B, target_len),
        fill_value=model.text_pad_id,
        dtype=input_ids.dtype,
        device=input_ids.device
    )
    padded_input_ids[:, :L] = input_ids

    return {
        "input_ids": padded_input_ids,
        "raw_text": test_sentences,
        "context_audio": padded_audio,
        "context_audio_lengths": audio_lengths,
        "target_audio_paths": [s["audio_filepath"] for s in batch],
        "target_num_frames": target_num_frames,
    }

@hydra_runner(config_path="conf", config_name="duplex_eartts")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.get("checkpoint_path", None):
        model = DuplexEARTTS.load_from_checkpoint(
            cfg.checkpoint_path,
            cfg=OmegaConf.to_container(cfg, resolve=True),
        ).eval()
    else:
        raise ValueError("For evaluation, you must provide `cfg.checkpoint_path`.")

    intelligibility = Intelligibility("stt_en_fastconformer_transducer_large", reuse_asr_hyps=False).reset()

    for batch_id, batch in enumerate(read_jsonl_batches(cfg.datasets_json_path, cfg.batch_size, max_batches=None)):
        inputs = collate_and_tokenize_custom(batch, model, extra_duration_thrshould=1.3, sample_rate=model.target_sample_rate, root_path=cfg.audio_dir)
        
        model.set_init_inputs(
            speaker_audio=inputs["context_audio"],
            speaker_audio_lens=inputs["context_audio_lengths"],
            system_prompt=cfg.get("inference_system_prompt", None)
        )
        init_inputs = model.get_init_inputs(B=inputs["input_ids"].size(0))

        audio, audio_len = model.offline_inference(
            next_subword_ids=inputs["input_ids"],
            formatter="custom",
            init_inputs=init_inputs,
        )

        # wav_dur = int(inputs["target_num_frames"][i] * model.target_samples_per_frame)
        # reset audio len to the actual size removing extra long audio padding
        audio_len = (torch.tensor(inputs["target_num_frames"]) * model.target_samples_per_frame).int()
 
        # resample audio to the asr sampling rate
        metric_audio_pred = resample(audio, model.target_sample_rate, 16000)
        metric_audio_pred_lens = (audio_len / model.target_sample_rate * 16000).to(torch.long)

        intelligibility.update(
            name="dataset",
            refs=inputs["raw_text"],
            pred_audio=metric_audio_pred,
            pred_audio_lens=metric_audio_pred_lens,
            asr_hyps=None,
        )

        # save audio to cfg.out_dir
        os.makedirs(cfg.out_dir, exist_ok=True)

        audio = audio.detach().cpu().float()
        audio_len = audio_len.cpu()

        for i in range(audio.size(0)):
            wav = audio[i, : audio_len[i]].numpy()
            #wav = audio[i, : wav_dur].numpy() # use precomputed estimated duration to avoid longer audios
            # Use original target audio filename
            target_path = inputs["target_audio_paths"][i]
            base_name = os.path.basename(target_path)
            out_path = os.path.join(cfg.out_dir, base_name)

            sf.write(
                out_path,
                wav,
                samplerate=model.target_sample_rate,
            )

            print(f"Saved: {out_path}")

    cer_wer = intelligibility.compute()
    for k, m in cer_wer.items():
        print(k, m)

if __name__ == "__main__":
    inference()
