# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from pathlib import Path
from typing import List, Optional

import librosa
import torch
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.parts.utils.tts_dataset_utils import beta_binomial_prior_distribution, stack_tensors
from nemo.utils import logging
from nemo.utils.decorators import experimental


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors

def normalize_volume_torch(audio, volume_level: float = 0.95):
    """Apply peak normalization to the input audio.
    """
    if not (0.0 <= volume_level <= 1.0):
        raise ValueError(f"Volume must be in range [0.0, 1.0], received {volume_level}")

    if audio.size == 0:
        return audio

    max_sample = torch.max(torch.abs(audio))
    if max_sample == 0:
        return audio

    return volume_level * (audio / torch.max(torch.abs(audio)))

def build_lhotse_dataloader(dataset, data_cfg, is_eval=False):
    """Buld dataloader given an input dataset."""
    return get_lhotse_dataloader_from_config(
        data_cfg,
        global_rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        dataset=dataset,
    )


@experimental
class T5TTSLhotseDataset(torch.utils.data.Dataset):
    """
    Class for processing and loading text to speech training examples.

    Args:
        sample_rate: Sample rate to load audio as. If the audio is stored at a different sample rate, then it will
            be resampled.
        text_tokenizer: Tokenizer to apply to the text field.
        speaker_path: Optional, path to JSON file with speaker indices, for multi-speaker training. Can be created with
            scripts.dataset_processing.tts.create_speaker_map.py
        featurizers: Optional, list of featurizers to load feature data from. Should be the same config provided
            when running scripts.dataset_processing.tts.compute_features.py before training.
        feature_processors: Optional, list of feature processors to run on training examples.
        align_prior_hop_length: Optional int, hop length of audio features.
            If provided alignment prior will be calculated and included in batch output. Must match hop length
            of audio features used for training.
        min_duration: Optional float, if provided audio files in the training manifest shorter than 'min_duration'
            will be ignored.
        max_duration: Optional float, if provided audio files in the training manifest longer than 'max_duration'
            will be ignored.
        volume_norm: Whether to apply volume normalization to loaded audio.
    """
    def __init__(
        self,
        sample_rate: int,
        align_prior_hop_length: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        volume_norm: bool = True,
        codec_model_downsample_factor: int = None,
        bos_id: int = None,
        eos_id: int = None,
        audio_bos_id: int = None,
        audio_eos_id: int = None,
        prior_scaling_factor: float = None,
        load_cached_codes_if_available: bool = True,
        dataset_type: str = 'train',
        tokenizer_config=None,
        load_16khz_audio: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.text_tokenizer = None
        self.align_prior_hop_length = align_prior_hop_length
        self.volume_norm = volume_norm

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.audio_bos_id = audio_bos_id
        self.audio_eos_id = audio_eos_id
        self.codec_model_downsample_factor = codec_model_downsample_factor
        self.include_align_prior = prior_scaling_factor is not None
        self.prior_scaling_factor = prior_scaling_factor
        self.load_cached_codes_if_available = load_cached_codes_if_available
        self.dataset_type = dataset_type
        self.tokenizer_config = tokenizer_config
        self.load_16khz_audio = load_16khz_audio

    def __getitem__(self, cuts):
        cuts = cuts.sort_by_duration()

        logging.debug(f"Len: {len(cuts)}")

        # load audios and text
        num_codec_frames = []
        align_priors = []
        context_audios = []
        context_audios_lens = []
        target_audios = []
        target_audios_lens = []
        target_audios_16khz = []
        target_audios_16khz_lens = []
        context_text_tokens = []
        context_text_tokens_lens = []
        target_text_tokens = []
        target_text_tokens_lens = []

        for i, cut in enumerate(cuts):
            # load target/answer audio
            answer_audio = torch.FloatTensor(cut.target_audio.resample(self.sample_rate).load_audio()).squeeze(0)
            if self.volume_norm:
                answer_audio = normalize_volume_torch(answer_audio)

            answer_audio = torch.nn.functional.pad(
                answer_audio,
                (0, self.codec_model_downsample_factor - (answer_audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            ).unsqueeze(0)

            answer_audio_len = answer_audio.shape[1]
            target_audios.append(answer_audio)
            target_audios_lens.append(answer_audio_len)
            num_frames = int(answer_audio_len / self.codec_model_downsample_factor) + 1 # +1 for EOS
            num_codec_frames.append(num_frames)

            # load context audio
            context_audio = torch.FloatTensor(cut.resample(self.sample_rate).load_audio()).squeeze(0)
            if self.volume_norm:
                context_audio = normalize_volume_torch(context_audio)

            context_audio = torch.nn.functional.pad(
                context_audio,
                (0, self.codec_model_downsample_factor - (context_audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            ).unsqueeze(0)
            context_audios_len = context_audio.shape[1]
            context_audios.append(context_audio)
            context_audios_lens.append(context_audios_len)

            # load context text
            if cut.supervisions[0].speaker == "user":
                context_text = cut.supervisions[0].text
                # check if the text is not empty
                if context_text.replace(" ", ""):
                    context_text = self.text_tokenizer(context_text)
                    context_text = context_text + [self.eos_id]
                else:
                    context_text = [self.eos_id]
                
                context_text = torch.tensor(context_text, dtype=torch.int32)
                context_text_len = context_text.shape[0]
                context_text_tokens.append(context_text)
                context_text_tokens_lens.append(context_text_len)
            else:
                raise Exception("First speaker should be user")

            if cut.supervisions[1].speaker == "agent":
                target_text = cut.supervisions[1].text
                # check if the text is not empty
                if target_text.replace(" ", ""):
                    target_text = self.text_tokenizer(target_text)
                    target_text = target_text + [self.eos_id]
                else:
                    target_text = [self.eos_id]

                target_text = torch.tensor(target_text, dtype=torch.int32)
                target_text_len = target_text.shape[0]
                target_text_tokens.append(target_text)
                target_text_tokens_lens.append(target_text_len)
            else:
                raise Exception("Second speaker should be agent")

            if self.include_align_prior:
                # align_prior = self.beta_binomial_interpolator(spec_len, text_len)
                align_prior = beta_binomial_prior_distribution(phoneme_count=target_text_len, mel_count=num_frames, scaling_factor=self.prior_scaling_factor)
                align_prior = torch.tensor(align_prior, dtype=torch.float32)
                align_priors.append(align_prior)

            if self.load_16khz_audio:
                target_audio_16khz = librosa.resample(answer_audio.squeeze(0).numpy(), orig_sr=self.sample_rate, target_sr=16000)
                target_audio_16khz = torch.FloatTensor(target_audio_16khz).unsqueeze(0)
                target_audio_16khz_len = target_audio_16khz.shape[1]
                target_audios_16khz.append(target_audio_16khz)
                target_audios_16khz_lens.append(target_audio_16khz_len)

        # collate target/agent audios
        target_audios = collate_vectors(
            [a.squeeze(0) for a in target_audios], max_length=max(target_audios_lens), padding_value=0.0
        ).float()
        target_audios_lens = torch.IntTensor(target_audios_lens)
        num_codec_frames = torch.IntTensor(num_codec_frames)

        # collate context/user audios
        context_audios = collate_vectors(
            [a.squeeze(0) for a in context_audios], max_length=max(context_audios_lens), padding_value=0.0
        ).float()
        context_audios_lens = torch.IntTensor(context_audios_lens)

        # collate context/user text
        context_text_tokens = collate_vectors(context_text_tokens, max_length=max(context_text_tokens_lens), padding_value=self.text_tokenizer.pad)
        context_text_tokens_lens = torch.IntTensor(context_text_tokens_lens)

        # collate target/agent text
        target_text_tokens = collate_vectors(target_text_tokens, max_length=max(target_text_tokens_lens), padding_value=self.text_tokenizer.pad)
        target_text_tokens_lens = torch.IntTensor(target_text_tokens_lens)

        # collate align prior
        if self.include_align_prior:
            spec_max_len = max([prior.shape[0] for prior in align_priors])
            text_max_len = max([prior.shape[1] for prior in align_priors])
            align_priors = stack_tensors(align_priors, max_lens=[text_max_len, spec_max_len],)

        # collate 16khz target/agent audio
        if self.load_16khz_audio:
            target_audios_16khz = collate_vectors(
                [a.squeeze(0) for a in target_audios_16khz], max_length=max(target_audios_16khz_lens), padding_value=0.0
            ).float()
            target_audios_16khz_lens = torch.IntTensor(target_audios_16khz_lens)

        """
        # ToDo: implement audio codes load on lhotse dataset
        if self.load_cached_codes_if_available and 'target_audio_codes_path' in data.manifest_entry:
            audio_codes_path = data.manifest_entry['target_audio_codes_path']
            audio_codes = torch.load(audio_codes_path).long() # (C, T)
            spec_len = audio_codes.shape[1] + 1 # +1 for EOS
            auidio_bos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_bos_id, dtype=audio_codes.dtype)
            audio_eos_tensor = torch.full((audio_codes.shape[0], 1), self.audio_eos_id, dtype=audio_codes.dtype)
            audio_codes = torch.cat([auidio_bos_tensor, audio_codes, audio_eos_tensor], dim=1)
            audio_codes_len = audio_codes.shape[1]
            example['audio_codes'] = audio_codes
            example['audio_codes_len'] = audio_codes_len
            example['audio_filepath'] = audio_codes_path
        else:
            # Only load audio if codes are not available
            audio_array, _, audio_filepath_rel = load_audio(
                manifest_entry=data.manifest_entry,
                audio_dir=data.audio_dir,
                sample_rate=self.sample_rate,
                volume_norm=self.volume_norm,
            )
            audio = torch.tensor(audio_array, dtype=torch.float32)
            # Pad audio to be multiple of downsample factor
            audio = torch.nn.functional.pad(
                audio,
                (0, self.codec_model_downsample_factor - (audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            )
            audio_len = audio.shape[0]
            example['audio_filepath'] = audio_filepath_rel
            example['audio'] = audio
            example['audio_len'] = audio_len
            spec_len = int(audio_len / self.codec_model_downsample_factor) + 1 # +1 for EOS
        
        if self.load_cached_codes_if_available and 'context_audio_codes_path' in data.manifest_entry:
            context_audio_codes_path = data.manifest_entry['context_audio_codes_path']
            context_audio_codes = torch.load(context_audio_codes_path).long()
            context_bos_tensor = torch.full((context_audio_codes.shape[0], 1), self.audio_bos_id, dtype=context_audio_codes.dtype)
            context_eos_tensor = torch.full((context_audio_codes.shape[0], 1), self.audio_eos_id, dtype=context_audio_codes.dtype)
            context_audio_codes = torch.cat([context_bos_tensor, context_audio_codes, context_eos_tensor], dim=1)
            context_audio_codes_len = context_audio_codes.shape[1]
            example['context_audio_codes'] = context_audio_codes
            example['context_audio_codes_len'] = context_audio_codes_len
        elif 'context_audio_filepath' in data.manifest_entry:
            context_audio_filepath = os.path.join(data.audio_dir, data.manifest_entry['context_audio_filepath'])
            context_duration = data.manifest_entry['context_audio_duration']
            context_audio_array = _read_audio(audio_filepath=context_audio_filepath, sample_rate=self.sample_rate, offset=0, duration=context_duration)
            context_audio_array = context_audio_array.samples
            context_audio = torch.tensor(context_audio_array, dtype=torch.float32)
            # Pad audio to be multiple of downsample factor
            context_audio = torch.nn.functional.pad(
                context_audio,
                (0, self.codec_model_downsample_factor - (context_audio.shape[0] % self.codec_model_downsample_factor)),
                value=0
            )
            context_audio_len = context_audio.shape[0]
            example['context_audio'] = context_audio
            example['context_audio_len'] = context_audio_len

        """
        batch_dict = {
            # "dataset_names": dataset_names,
            # "audio_filepaths": audio_filepath_list,
            "sample_ids": list(cuts.ids),
            "text": target_text_tokens,
            "text_lens": target_text_tokens_lens,
            "context_text": context_text_tokens,
            "context_text_lens": context_text_tokens_lens,
            'audio': target_audios,
            'audio_lens': target_audios_lens,
            # 'audio_codes': batch_audio_codes
            # 'audio_codes_lens': batch_audio_codes_len
            'context_audio': context_audios,
            'context_audio_lens': context_audios_lens,
            # 'context_audio_codes': batch_context_audio_codes
            # 'context_audio_codes_lens': batch_context_audio_codes_len
        }

        if self.include_align_prior:
            batch_dict["align_prior_matrix"] = align_priors
        
        if self.load_16khz_audio:
            batch_dict['audio_16khz'] = target_audios_16khz
            batch_dict['audio_lens_16khz'] = target_audios_16khz_lens

        return batch_dict


    def collate_fn(self, batch: List[dict]):
        return batch
