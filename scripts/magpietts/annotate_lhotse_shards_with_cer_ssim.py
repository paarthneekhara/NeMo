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
This script annotates Lhotse shard cuts with CER and SSIM metrics.

CER is computed by transcribing target audio with Whisper and comparing against ground truth text.
SSIM is computed as cosine similarity between speaker embeddings (TitaNet) of target and context audio.

Metrics are stored as supervision-level custom fields:
  cut.supervisions[0].custom["annotated_cer"]
  cut.supervisions[0].custom["annotated_ssim"]
  cut.supervisions[0].custom["gt_transcript_for_cer"]
  cut.supervisions[0].custom["asr_transcript_for_cer"]

Both the Whisper transcript and the ground truth text are normalized (via nemo_text_processing
Normalizer when available, then process_text_for_cer) before CER computation.

Output is new cuts.NNNNNN.jsonl.gz files preserving all existing fields (recordings, codes, etc.).

Example of input shards:
    $ tree ${CUTS_DIR}
    ${CUTS_DIR}/
        cuts.000000.jsonl.gz
        cuts.000001.jsonl.gz
        ...

    $ tree ${TARGET_AUDIO_DIR}
    ${TARGET_AUDIO_DIR}/
        recording.000000.tar
        recording.000001.tar
        ...

    $ tree ${CONTEXT_AUDIO_DIR}
    ${CONTEXT_AUDIO_DIR}/
        recording.000000.tar
        recording.000001.tar
        ...

Each srun task runs as an independent single-GPU process (no DDP). Rank and world size
are auto-detected from SLURM environment variables (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID).

Example usage:
    python -u ${CODE_DIR}/scripts/magpietts/annotate_lhotse_shards_with_cer_ssim.py \
        --cuts-dir ${CUTS_DIR} \
        --target-audio-dir ${TARGET_AUDIO_DIR} \
        --context-audio-dir ${CONTEXT_AUDIO_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --batch-size ${BATCH_SIZE} \
        --log-level "INFO"

Expected output:
    $ tree ${OUTPUT_DIR}
    ${OUTPUT_DIR}/
        cuts.000000.jsonl.gz
        cuts.000001.jsonl.gz
        ...
"""

# Pin each srun task to its own GPU before any CUDA initialization happens.
# Must run before importing torch, lightning, or NeMo, which may init the CUDA driver.
import os as _os

if "SLURM_LOCALID" in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = _os.environ["SLURM_LOCALID"]
elif "LOCAL_RANK" in _os.environ:
    _os.environ["CUDA_VISIBLE_DEVICES"] = _os.environ["LOCAL_RANK"]

import argparse
import glob
import gzip
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import math
import time

import lightning.pytorch as pl
import numpy as np
import torch
from lhotse import CutSet
from lhotse.dataset import IterableDatasetWrapper, SimpleCutSampler
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.parts.utils.helpers import process_text_for_cer

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


def collate_audio_vectors(
    audio_list: List[torch.Tensor], audio_lens_list: List[int], padding_value: Union[float, int]
) -> torch.Tensor:
    """Collate variable-length 1-D audio tensors into a zero-padded batch."""
    assert all(len(t.shape) == 1 for t in audio_list)
    assert len(audio_list) == len(audio_lens_list)
    result = audio_list[0].new_ones(len(audio_lens_list), max(audio_lens_list)) * padding_value
    for i, t in enumerate(audio_list):
        result[i, : t.shape[0]] = t
    return result


class AudioPairCERSSIMDataset(Dataset):
    """
    Lhotse Dataset that loads target (and optionally context) audio at 16 kHz,
    along with ground truth text and language metadata for CER/SSIM computation.
    When context audio is unavailable (text-context shards), SSIM will be hardcoded to 1.0.
    """

    SAMPLE_RATE = 16000
    MAX_AUDIO_SAMPLES = 30 * 16000  # 30s at 16kHz (Whisper's native window)

    def __getitem__(self, cuts: CutSet) -> Optional[Dict[str, Any]]:
        target_audios_list = []
        target_lengths_list = []
        context_audios_list = []
        context_lengths_list = []
        ground_truth_texts_list = []
        languages_list = []
        target_cut_ids_list = []
        shard_indices_list = []
        has_context_audio = True

        for cut in cuts:
            if not cut.has_custom("shard_origin"):
                raise ValueError(f"Cut {cut.id} is missing required key 'shard_origin'.")

            origin_path = cut.custom["shard_origin"]
            match = re.search(r"cuts\.(\d+)\.jsonl\.gz$", origin_path)
            if match is None:
                raise ValueError(f"Could not parse shard index from shard_origin: {origin_path}")
            shard_idx_origin = int(match.group(1))

            target_audio = torch.from_numpy(
                cut.recording.resample(self.SAMPLE_RATE).load_audio().squeeze(0)
            )
            if target_audio.shape[0] > self.MAX_AUDIO_SAMPLES:
                target_audio = target_audio[: self.MAX_AUDIO_SAMPLES]
            target_audios_list.append(target_audio)
            target_lengths_list.append(target_audio.shape[0])

            if cut.has_custom("context_recording"):
                context_audio = torch.from_numpy(
                    cut.context_recording.resample(self.SAMPLE_RATE).load_audio().squeeze(0)
                )
                if context_audio.shape[0] > self.MAX_AUDIO_SAMPLES:
                    context_audio = context_audio[: self.MAX_AUDIO_SAMPLES]
                context_audios_list.append(context_audio)
                context_lengths_list.append(context_audio.shape[0])
            else:
                has_context_audio = False

            if cut.supervisions and len(cut.supervisions) > 0:
                sup = cut.supervisions[0]
                if sup.has_custom("normalized_text"):
                    gt_text = sup.normalized_text
                else:
                    gt_text = sup.text or ""
            else:
                gt_text = ""
            ground_truth_texts_list.append(gt_text)

            if cut.has_custom("lang"):
                language = cut.lang
            elif cut.supervisions and cut.supervisions[0].language is not None:
                language = cut.supervisions[0].language.lower()
            else:
                language = "en"
            languages_list.append(language)

            target_cut_ids_list.append(cut.id)
            shard_indices_list.append(shard_idx_origin)

        if not target_audios_list:
            raise ValueError("AudioPairCERSSIMDataset.__getitem__ received an empty CutSet.")

        target_audio_padded = collate_audio_vectors(target_audios_list, target_lengths_list, padding_value=0.0)

        batch = {
            "target_audios_16khz": target_audio_padded,
            "target_audio_lens_16khz": torch.LongTensor(target_lengths_list),
            "has_context_audio": has_context_audio,
            "ground_truth_texts": ground_truth_texts_list,
            "languages": languages_list,
            "target_cut_id": target_cut_ids_list,
            "shard_idx_origin": shard_indices_list,
        }

        if has_context_audio:
            context_audio_padded = collate_audio_vectors(context_audios_list, context_lengths_list, padding_value=0.0)
            batch["context_audios_16khz"] = context_audio_padded
            batch["context_audio_lens_16khz"] = torch.LongTensor(context_lengths_list)

        return batch


class CERSSIMExtractor(pl.LightningModule):
    """
    LightningModule that computes CER (via Whisper) and SSIM (via TitaNet) for
    each cut. Distributes shards across ranks via predict_dataloader.
    """

    def __init__(
        self,
        cuts_dir: str,
        target_audio_dir: str,
        batch_size: int,
        context_audio_dir: Optional[str] = None,
        whisper_model_name: str = "openai/whisper-large-v3",
        log_every_n_batches: int = 10,
        override_language: Optional[str] = None,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        num_shards: Optional[int] = None,
    ):
        super().__init__()
        self.cuts_dir = Path(cuts_dir)
        self.target_audio_dir = Path(target_audio_dir)
        self.context_audio_dir = Path(context_audio_dir) if context_audio_dir else None
        self.batch_size = batch_size
        self.whisper_model_name = whisper_model_name
        self.log_every_n_batches = log_every_n_batches
        self.override_language = override_language
        self.shard_rank = shard_rank
        self.shard_world_size = shard_world_size
        self.num_shards = num_shards
        self._batch_counter = 0
        self._total_batches_estimate = 0
        self._total_cuts_estimate = 0
        self._processed_cuts = 0
        self._start_time: Optional[float] = None

        logging.info(f"Loading Whisper model: {self.whisper_model_name}")
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False
        logging.info("Whisper model loaded.")

        logging.info("Loading TitaNet speaker verification model: titanet_large")
        import nemo.collections.asr as nemo_asr

        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        self.speaker_model.freeze()
        logging.info("TitaNet model loaded.")

        self._normalizer_cache: Dict[str, Optional[Any]] = {}
        self._rank_dataloaders: Optional[List[DataLoader]] = None

    def _get_cached_normalizer(self, lang_key: Optional[str]):
        if not PYNINI_AVAILABLE:
            return None
        lang_key = lang_key if lang_key else "en"
        if lang_key not in self._normalizer_cache:
            logging.info(f"Creating normalizer for language: {lang_key}")
            try:
                self._normalizer_cache[lang_key] = Normalizer(input_case="cased", lang=lang_key)
            except Exception as e:
                logging.warning(f"Failed to create normalizer for language: {lang_key}. Error: {e}")
                self._normalizer_cache[lang_key] = None
        return self._normalizer_cache[lang_key]

    def predict_dataloader(self) -> List[DataLoader]:
        if self._rank_dataloaders is not None:
            return self._rank_dataloaders

        current_global_rank = self.shard_rank
        world_size = self.shard_world_size

        logging.info(f"[Rank {current_global_rank}/{world_size}] Creating assigned subset of dataloaders...")

        cuts_shard_pattern = str(self.cuts_dir / "cuts.*.jsonl.gz")
        all_cuts_shard_paths = sorted(glob.glob(cuts_shard_pattern))

        if not all_cuts_shard_paths:
            raise FileNotFoundError(
                f"[Rank {current_global_rank}/{world_size}] No input cut shards found: {cuts_shard_pattern}"
            )

        num_total_shards = len(all_cuts_shard_paths)

        first_idx = int(re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[0]).group(1))
        last_idx = int(re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[-1]).group(1))
        if first_idx != 0:
            raise ValueError(f"Expected first shard index to be 0, but found {first_idx}")
        if last_idx != num_total_shards - 1:
            raise ValueError(f"Expected last shard index to be {num_total_shards - 1}, but found {last_idx}")

        if self.num_shards is not None and self.num_shards < num_total_shards:
            logging.info(
                f"[Rank {current_global_rank}/{world_size}] Limiting to first {self.num_shards} shards "
                f"(0..{self.num_shards - 1}) out of {num_total_shards} found."
            )
            num_total_shards = self.num_shards

        logging.info(
            f"[Rank {current_global_rank}/{world_size}] Using {num_total_shards} total shards (0..{num_total_shards - 1})."
        )

        is_distributed = world_size > 1
        assigned_shard_indices = []

        if num_total_shards > 0:
            if not is_distributed:
                assigned_shard_indices = list(range(num_total_shards))
            else:
                num_per_rank_base = num_total_shards // world_size
                num_with_extra = num_total_shards % world_size

                if current_global_rank < num_with_extra:
                    start = current_global_rank * (num_per_rank_base + 1)
                    count = num_per_rank_base + 1
                else:
                    start = num_with_extra + current_global_rank * num_per_rank_base
                    count = num_per_rank_base

                assigned_shard_indices = list(range(start, start + count))
                logging.info(
                    f"[Rank {current_global_rank}/{world_size}] Assigned shard indices "
                    f"{start} through {start + count - 1} ({count} shards)"
                )

        if not assigned_shard_indices:
            logging.info(f"[Rank {current_global_rank}/{world_size}] No shards assigned. Returning empty list.")
            self._rank_dataloaders = []
            return []

        dataloaders = []
        total_cuts = 0
        total_batches = 0
        for shard_idx in tqdm(
            assigned_shard_indices,
            desc=f"[Rank {current_global_rank}/{world_size}] Creating DataLoaders",
        ):
            cuts_path = str(self.cuts_dir / f"cuts.{shard_idx:06d}.jsonl.gz")
            fields = {
                "cuts": [cuts_path],
                "recording": [str(self.target_audio_dir / f"recording.{shard_idx:06d}.tar")],
            }
            if self.context_audio_dir is not None:
                fields["context_recording"] = [str(self.context_audio_dir / f"recording.{shard_idx:06d}.tar")]
            if not all(Path(v[0]).is_file() for v in fields.values()):
                raise FileNotFoundError(
                    f"[Rank {current_global_rank}/{world_size}] Missing files for shard {shard_idx}: {fields}"
                )

            with gzip.open(cuts_path, 'rt') as f:
                num_cuts_in_shard = sum(1 for _ in f)
            total_cuts += num_cuts_in_shard
            total_batches += math.ceil(num_cuts_in_shard / self.batch_size)

            shard_cutset = CutSet.from_shar(fields=fields)
            sampler = SimpleCutSampler(
                shard_cutset, max_cuts=self.batch_size, shuffle=False, drop_last=False, rank=0, world_size=1
            )
            dataset = AudioPairCERSSIMDataset()
            iterable_dataset = IterableDatasetWrapper(dataset=dataset, sampler=sampler)
            dl = DataLoader(dataset=iterable_dataset, batch_size=None, num_workers=1, pin_memory=True)
            dataloaders.append(dl)

        self._total_cuts_estimate = total_cuts
        self._total_batches_estimate = total_batches
        logging.info(
            f"[Rank {current_global_rank}/{world_size}] Created {len(dataloaders)} DataLoaders. "
            f"Total cuts: {total_cuts}, estimated batches: {total_batches}."
        )
        self._rank_dataloaders = dataloaders
        return self._rank_dataloaders

    def _transcribe_batch_with_whisper(
        self,
        audio_arrays: List[np.ndarray],
        languages: List[str],
    ) -> List[str]:
        """Transcribe a list of 16 kHz numpy audio arrays with Whisper, grouped by language."""
        if not audio_arrays:
            return []

        grouped_indices = defaultdict(list)
        for idx, lang in enumerate(languages):
            grouped_indices[lang].append(idx)

        transcripts = [""] * len(audio_arrays)
        for lang, indices in grouped_indices.items():
            forced_decoder_ids = (
                self.whisper_processor.get_decoder_prompt_ids(language=lang, task="transcribe") if lang else None
            )
            speech_arrays = [audio_arrays[idx] for idx in indices]
            inputs = self.whisper_processor(
                speech_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
            batch_transcripts = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            for idx, text in zip(indices, batch_transcripts):
                transcripts[idx] = text

        return transcripts

    def _compute_speaker_embeddings(
        self,
        audio_padded: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Extract speaker embeddings from padded audio batch using TitaNet."""
        audio_padded = audio_padded.to(self.device)
        audio_lens = audio_lens.to(self.device)
        with torch.no_grad():
            _, embeddings = self.speaker_model.forward(
                input_signal=audio_padded, input_signal_length=audio_lens
            )
        return embeddings

    def forward(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        target_audios = batch["target_audios_16khz"]
        target_lens = batch["target_audio_lens_16khz"]
        has_context_audio = batch["has_context_audio"]
        context_audios = batch.get("context_audios_16khz", None)
        context_lens = batch.get("context_audio_lens_16khz", None)
        gt_texts = batch["ground_truth_texts"]
        languages = batch["languages"]
        if self.override_language:
            languages = [self.override_language] * len(languages)
        cut_ids = batch["target_cut_id"]
        shard_indices = batch["shard_idx_origin"]

        batch_size = target_audios.shape[0]

        # --- Whisper transcription ---
        self.whisper_model.to(self.device)
        audio_arrays = [target_audios[i, : target_lens[i]].cpu().numpy() for i in range(batch_size)]
        try:
            pred_transcripts = self._transcribe_batch_with_whisper(audio_arrays, languages)
        except Exception as e:
            logging.warning(f"[Rank {self.shard_rank}] Batched Whisper failed, falling back to per-item: {e}")
            pred_transcripts = []
            for i in range(batch_size):
                try:
                    t = self._transcribe_batch_with_whisper([audio_arrays[i]], [languages[i]])
                    pred_transcripts.append(t[0])
                except Exception as inner_e:
                    logging.warning(f"[Rank {self.shard_rank}] Whisper failed for cut {cut_ids[i]}: {inner_e}")
                    pred_transcripts.append(None)

        # --- Normalize transcripts and compute CER ---
        cer_values = []
        gt_transcripts_for_cer = []
        asr_transcripts_for_cer = []
        for i in range(batch_size):
            if pred_transcripts[i] is None or gt_texts[i] == "":
                cer_values.append(1.0)
                gt_transcripts_for_cer.append("")
                asr_transcripts_for_cer.append("")
                continue

            normalizer = self._get_cached_normalizer(languages[i])

            pred_text = pred_transcripts[i]
            gt_text = gt_texts[i]
            if normalizer is not None:
                try:
                    pred_text = normalizer.normalize(pred_text)
                except Exception:
                    pass
                try:
                    gt_text = normalizer.normalize(gt_text)
                except Exception:
                    pass

            pred_norm = process_text_for_cer(pred_text)
            gt_norm = process_text_for_cer(gt_text)

            if languages[i] in ("zh", "chinese"):
                pred_norm = pred_norm.replace(" ", "")
                gt_norm = gt_norm.replace(" ", "")

            asr_transcripts_for_cer.append(pred_norm)
            gt_transcripts_for_cer.append(gt_norm)

            if gt_norm == "":
                cer_values.append(1.0)
                continue
            cer = min(word_error_rate([pred_norm], [gt_norm], use_cer=True), 1.0)
            cer_values.append(float(cer))

        # --- Speaker similarity ---
        if has_context_audio and context_audios is not None:
            try:
                target_embeddings = self._compute_speaker_embeddings(target_audios, target_lens)
                context_embeddings = self._compute_speaker_embeddings(context_audios, context_lens)
            except Exception as e:
                logging.warning(f"[Rank {self.shard_rank}] Speaker embedding extraction failed: {e}")
                target_embeddings = None
                context_embeddings = None

            ssim_values = []
            for i in range(batch_size):
                if target_embeddings is None or context_embeddings is None:
                    ssim_values.append(-1.0)
                    continue
                t_emb = target_embeddings[i].cpu().float().numpy()
                c_emb = context_embeddings[i].cpu().float().numpy()
                norm_product = np.linalg.norm(t_emb) * np.linalg.norm(c_emb)
                if norm_product < 1e-8:
                    ssim_values.append(0.0)
                else:
                    ssim_values.append(float(np.dot(t_emb, c_emb) / norm_product))
        else:
            ssim_values = [1.0] * batch_size

        # --- Assemble results ---
        results = []
        for i in range(batch_size):
            results.append({
                "target_cut_id": cut_ids[i],
                "shard_idx": shard_indices[i],
                "annotated_cer": cer_values[i],
                "annotated_ssim": ssim_values[i],
                "gt_transcript_for_cer": gt_transcripts_for_cer[i],
                "asr_transcript_for_cer": asr_transcripts_for_cer[i],
                "annotated_language": languages[i],
            })

        # --- Progress tracking and periodic logging ---
        self._batch_counter += 1
        self._processed_cuts += batch_size
        if self._start_time is None:
            self._start_time = time.time()

        if self._batch_counter % self.log_every_n_batches == 0:
            elapsed = time.time() - self._start_time
            cuts_per_sec = self._processed_cuts / elapsed if elapsed > 0 else 0
            pct = (self._batch_counter / self._total_batches_estimate * 100) if self._total_batches_estimate > 0 else 0
            remaining_batches = max(self._total_batches_estimate - self._batch_counter, 0)
            secs_per_batch = elapsed / self._batch_counter if self._batch_counter > 0 else 0
            eta_secs = remaining_batches * secs_per_batch
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            sample_idx = 0
            logging.info(
                f"[Rank {self.shard_rank}] Batch {self._batch_counter}/{self._total_batches_estimate} "
                f"({pct:.1f}%) | Cuts {self._processed_cuts}/{self._total_cuts_estimate} | "
                f"{cuts_per_sec:.1f} cuts/s | Elapsed {elapsed_str} | ETA {eta_str}\n"
                f"  Shard {shard_indices[sample_idx]} | Cut {cut_ids[sample_idx]}\n"
                f"  GT (normalized):  {gt_transcripts_for_cer[sample_idx]}\n"
                f"  ASR (normalized): {asr_transcripts_for_cer[sample_idx]}\n"
                f"  CER: {cer_values[sample_idx]:.4f} | SSIM: {ssim_values[sample_idx]:.4f}"
            )

        return results

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> List[Dict[str, Any]]:
        try:
            return self.forward(batch)
        except Exception as e:
            cut_ids = batch.get("target_cut_id", [])
            shard_indices = batch.get("shard_idx_origin", [])
            languages = batch.get("languages", [])
            if self.override_language:
                languages = [self.override_language] * len(cut_ids)
            logging.error(
                f"[Rank {self.shard_rank}] forward() failed for batch_idx {batch_idx} "
                f"(cuts: {cut_ids}): {e}. Returning sentinel values."
            )
            results = []
            for i in range(len(cut_ids)):
                results.append({
                    "target_cut_id": cut_ids[i],
                    "shard_idx": shard_indices[i] if i < len(shard_indices) else -1,
                    "annotated_cer": 1.0,
                    "annotated_ssim": -1.0,
                    "gt_transcript_for_cer": "",
                    "asr_transcript_for_cer": "",
                    "annotated_language": languages[i] if i < len(languages) else "unknown",
                })
            return results


class CERSSIMPredictionWriter(BasePredictionWriter):
    """
    Accumulates per-cut CER/SSIM results and writes annotated cuts.jsonl.gz
    files when a shard is fully processed.
    """

    def __init__(self, cuts_dir: str, output_dir: str, shard_rank: int = 0, shard_world_size: int = 1):
        super().__init__(write_interval="batch")
        self.cuts_dir = Path(cuts_dir)
        self.output_dir = Path(output_dir)
        self.rank: int = shard_rank
        self.world_size: int = shard_world_size
        # shard_idx -> {cut_id -> {"annotated_cer": float, "annotated_ssim": float, ...}}
        self.shard_results: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.last_processed_shard_idx: int = -1

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"[Rank {self.rank}/{self.world_size}] CERSSIMPredictionWriter setup complete.")

    def _write_shard(self, shard_idx: int) -> None:
        """Load original cuts, annotate with CER/SSIM, write new jsonl.gz."""
        results = self.shard_results.pop(shard_idx, {})
        if not results:
            logging.warning(f"[Rank {self.rank}] No results for shard {shard_idx}, skipping write.")
            return

        input_path = self.cuts_dir / f"cuts.{shard_idx:06d}.jsonl.gz"
        output_path = self.output_dir / f"cuts.{shard_idx:06d}.jsonl.gz"

        annotated_count = 0
        total_count = 0

        with gzip.open(str(input_path), 'rt') as f_in, gzip.open(str(output_path), 'wt') as f_out:
            for line in f_in:
                total_count += 1
                cut_data = json.loads(line)
                cut_id = cut_data.get("id", "")

                if cut_id in results:
                    metrics = results[cut_id]
                    if "supervisions" in cut_data and len(cut_data["supervisions"]) > 0:
                        sup = cut_data["supervisions"][0]
                        if "custom" not in sup or sup["custom"] is None:
                            sup["custom"] = {}
                        sup["custom"]["annotated_cer"] = metrics["annotated_cer"]
                        sup["custom"]["annotated_ssim"] = metrics["annotated_ssim"]
                        sup["custom"]["gt_transcript_for_cer"] = metrics["gt_transcript_for_cer"]
                        sup["custom"]["asr_transcript_for_cer"] = metrics["asr_transcript_for_cer"]
                        sup["custom"]["annotated_language"] = metrics["annotated_language"]

                        if metrics["annotated_language"] in ("zh", "chinese"):
                            if sup.get("text"):
                                sup["text"] = sup["text"].replace(" ", "")
                            if sup.get("custom", {}).get("normalized_text"):
                                sup["custom"]["normalized_text"] = sup["custom"]["normalized_text"].replace(" ", "")

                        annotated_count += 1
                    else:
                        logging.warning(
                            f"[Rank {self.rank}] Cut {cut_id} has no supervisions, cannot store metrics."
                        )

                f_out.write(json.dumps(cut_data) + "\n")

        logging.info(
            f"[Rank {self.rank}] Wrote shard {shard_idx}: {annotated_count}/{total_count} cuts annotated -> {output_path}"
        )

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        predictions: Optional[List[Dict[str, Any]]],
        batch_indices: Optional[List[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not predictions:
            logging.error(
                f"[Rank {self.rank}] Empty predictions for batch_idx {batch_idx}, dataloader_idx {dataloader_idx}."
            )
            return

        current_shard_idx = predictions[0]["shard_idx"]

        if current_shard_idx != self.last_processed_shard_idx and self.last_processed_shard_idx != -1:
            logging.info(
                f"[Rank {self.rank}] Shard changed {self.last_processed_shard_idx} -> {current_shard_idx}. "
                f"Writing shard {self.last_processed_shard_idx}."
            )
            self._write_shard(self.last_processed_shard_idx)

        self.last_processed_shard_idx = current_shard_idx

        for pred in predictions:
            cut_id = pred["target_cut_id"]
            shard_idx = pred["shard_idx"]
            self.shard_results[shard_idx][cut_id] = {
                "annotated_cer": pred["annotated_cer"],
                "annotated_ssim": pred["annotated_ssim"],
                "gt_transcript_for_cer": pred["gt_transcript_for_cer"],
                "asr_transcript_for_cer": pred["asr_transcript_for_cer"],
                "annotated_language": pred["annotated_language"],
            }

    def teardown(self, trainer: Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self.last_processed_shard_idx != -1 and self.last_processed_shard_idx in self.shard_results:
            logging.info(f"[Rank {self.rank}] Writing final shard {self.last_processed_shard_idx}.")
            self._write_shard(self.last_processed_shard_idx)

        remaining = list(self.shard_results.keys())
        if remaining:
            logging.warning(f"[Rank {self.rank}] Writing {len(remaining)} remaining shards: {remaining}")
            for shard_idx in remaining:
                self._write_shard(shard_idx)

        logging.info(f"[Rank {self.rank}/{self.world_size}] CERSSIMPredictionWriter teardown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Lhotse shard cuts with CER (Whisper) and SSIM (TitaNet) metrics."
    )
    parser.add_argument(
        "--cuts-dir", type=str, required=True, help="Directory containing input cuts/cuts.*.jsonl.gz shards."
    )
    parser.add_argument(
        "--target-audio-dir", type=str, required=True, help="Directory containing target_audio/recording.*.tar shards."
    )
    parser.add_argument(
        "--context-audio-dir", type=str, default=None,
        help="Directory containing context_audio/recording.*.tar shards. "
             "If not provided, SSIM is hardcoded to 1.0 (text-context mode).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save annotated cuts.*.jsonl.gz.")
    parser.add_argument(
        "--whisper-model-name", type=str, default="openai/whisper-large-v3",
        help="HuggingFace Whisper model name for ASR.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU.")
    parser.add_argument(
        "--log-every-n-batches", type=int, default=10,
        help="Log a sample (GT text, predicted text, CER, SSIM) every N batches per rank.",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Override language for Whisper transcription and text normalization "
             "(e.g. 'it', 'de', 'es'). If not set, language is read from each cut.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Process only the first N shards (0..N-1). If not set, all shards are processed.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    log_level_val = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = '%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level_val, format=log_format)

    # Detect rank/world_size from SLURM environment.
    # CUDA_VISIBLE_DEVICES is already pinned at module top (before torch import)
    # so each srun task runs as an independent single-GPU process with no DDP.
    shard_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    shard_world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))

    logging.info(
        f"Running as independent single-GPU process: shard_rank={shard_rank}, "
        f"shard_world_size={shard_world_size}, local_rank={local_rank}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )

    extractor = CERSSIMExtractor(
        cuts_dir=args.cuts_dir,
        target_audio_dir=args.target_audio_dir,
        batch_size=args.batch_size,
        context_audio_dir=args.context_audio_dir,
        whisper_model_name=args.whisper_model_name,
        log_every_n_batches=args.log_every_n_batches,
        override_language=args.language,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        num_shards=args.num_shards,
    )

    pred_writer = CERSSIMPredictionWriter(
        cuts_dir=args.cuts_dir,
        output_dir=args.output_dir,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="auto",
        callbacks=[pred_writer],
        use_distributed_sampler=False,
    )

    logging.info(f"Starting CER/SSIM annotation. Shard rank {shard_rank}/{shard_world_size}.")
    trainer.predict(extractor, return_predictions=False)
    logging.info("CER/SSIM annotation finished.")


if __name__ == "__main__":
    import torch.multiprocessing

    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
