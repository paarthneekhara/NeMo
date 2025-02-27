# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from pathlib import Path
from typing import List

import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.fastpitch_codec_loss import AudioTokenLoss, MaskedSoftmax
from nemo.collections.tts.modules.fastpitch_codec_modules import FastPitchCodecModule
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
)

from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    BoolType,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenIndex
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging, model_utils
from nemo.utils.decorators import experimental


@experimental
class FastPitchCodecModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = self._create_tokenizer(cfg.text_tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        aligner = instantiate(cfg.alignment_module)
        
        vocab_size = len(self.text_tokenizer.tokens)
        pad_token = self.text_tokenizer.pad
        text_encoder = instantiate(cfg.text_encoder, n_embed=vocab_size, padding_idx=pad_token)
        audio_decoder = instantiate(cfg.audio_decoder)
        duration_decoder = instantiate(cfg.duration_decoder)
        speaking_rate_predictor = instantiate(cfg.speaking_rate_predictor)

        self.vector_quantizer = instantiate(cfg.vector_quantizer)
        self.fsq_scale = (self.vector_quantizer.fsqs[0].num_levels[0][0] // 2).item()
        self.num_codebooks = cfg.get("num_codebooks")

        max_token_duration = cfg.get("max_token_duration")
        speaking_rate_factor = cfg.get("speaking_rate_factor")

        self.audio_parallel_prob = cfg.get("audio_parallel_prob", 0.1)
        self.audio_denoise_prob = cfg.get("audio_denoise_prob", 0.0)
        self.audio_infill_min = cfg.get("audio_infill_min", 0.0)
        self.audio_infill_max = cfg.get("audio_infill_max", 1.0)

        self.duration_parallel_prob = cfg.get("duration_parallel_prob", 0.1)
        self.duration_denoise_prob = cfg.get("duration_denoise_prob", 0.0)
        self.duration_infill_min = cfg.get("duration_infill_min", 0.0)
        self.duration_infill_max = cfg.get("duration_infill_max", 1.0)
        self.decoder_duration_noise_percent = cfg.get("decoder_duration_noise_percent", 0.0)

        self.fastpitch = FastPitchCodecModule(
            aligner_module=aligner,
            vector_quantizer_module=self.vector_quantizer,
            text_encoder_module=text_encoder,
            duration_decoder_module=duration_decoder,
            audio_decoder_module=audio_decoder,
            speaking_rate_predictor_module=speaking_rate_predictor,
            max_token_duration=max_token_duration,
            speaking_rate_factor=speaking_rate_factor
        )
        self.context_encoder = instantiate(cfg.context_encoder)
        self.context_aligner_encoder = instantiate(cfg.context_aligner_encoder)

        self.context_min_len = cfg.get("context_min_len", 128)
        self.context_max_len = cfg.get("context_max_len", 256)
        self.space_token = cfg.get("space_token", 102)

        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        audio_token_label_smoothing = cfg.get("audio_token_label_smoothing", 0.0)
        self.duration_loss_scale = cfg.get("duration_loss_scale", 1.0)
        duration_label_smoothing = cfg.get("duration_label_smoothing", 0.0)
        self.speaking_rate_loss_scale = cfg.get("speaking_rate_loss_scale", 1.0)

        self.aligner_bin_loss_scale = cfg.get("aligner_bin_loss_scale", 0.01)
        self.aligner_ctc_loss_scale = cfg.get("aligner_ctc_loss_scale", 0.01)

        self.bin_loss_start_epoch = cfg.get("bin_loss_start_epoch", 0)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 100)

        self.audio_token_loss_fn = AudioTokenLoss(
            num_codebooks=self.num_codebooks, label_smoothing=audio_token_label_smoothing
        )
        self.duration_loss_fn = MaskedSoftmax(label_smoothing=duration_label_smoothing)
        self.speaking_rate_loss_fn = torch.nn.L1Loss()
        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.log_config = cfg.get("log_config", None)

    def _create_tokenizer(self, tokenizer_config):
        if "g2p" in tokenizer_config:
            if "phoneme_dict" in tokenizer_config.g2p:
                tokenizer_config.g2p.phoneme_dict = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', tokenizer_config.g2p.phoneme_dict,
                )

            if "heteronyms" in tokenizer_config.g2p:
                tokenizer_config.g2p.heteronyms = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', tokenizer_config.g2p.heteronyms,
                )

        text_tokenizer = instantiate(tokenizer_config)
        return text_tokenizer

    def parse(self, str_input: str) -> torch.tensor:
        if self.training:
            logging.warning("parse() is meant to be called in eval mode.")

        if not hasattr(self.text_tokenizer, "set_phone_prob"):
            text_tokens = self.text_tokenizer.encode(str_input)
        else:
            with self.text_tokenizer.set_phone_prob(prob=1.0):
                text_tokens = self.text_tokenizer.encode(str_input)

        token_tensor = torch.tensor(text_tokens).unsqueeze_(0).long().to(self.device)
        return token_tensor

    def create_infill_mask(self, input_lens, parallel_prob, denoise_prob, infill_min, infill_max):
        batch_size = input_lens.shape[0]
        len_mask = get_mask_from_lengths(input_lens)
        max_len = len_mask.shape[1]

        # [batch_size]
        do_infill = torch.rand(size=[batch_size], device=input_lens.device) >= parallel_prob
        do_denoise = torch.rand(size=[batch_size], device=input_lens.device) < denoise_prob

        infill_percent = torch.rand(size=[batch_size], device=input_lens.device)
        infill_percent = infill_min + (infill_max - infill_min) * infill_percent
        infill_percent = torch.where(do_denoise, torch.ones_like(infill_percent), infill_percent)
        infill_percent = do_infill.int() * infill_percent
        infill_len = (infill_percent * input_lens.float())
        infill_rank = torch.clamp_min(infill_len - 1, 0).long()
        infill_rank = rearrange(infill_rank, 'B -> B 1')

        # [batch_size, time]
        infill_vals = torch.rand(size=len_mask.shape, device=input_lens.device)
        infill_vals = infill_vals * len_mask
        infill_topk = torch.topk(infill_vals, k=max_len, dim=1, sorted=True).values
        infill_min_val = torch.gather(infill_topk, index=infill_rank, dim=1)
        infill_mask = infill_vals >= infill_min_val

        infill_mask = infill_mask * len_mask
        infill_loss_mask = ~infill_mask * len_mask

        return infill_mask, infill_loss_mask

    def sample_context_audio(self, audio_tokens, audio_codes, audio_lens, text, text_lens, durs, sample):
        # [B, T_text]
        cum_ends = torch.cumsum(durs, dim=1).long()
        space_ends = torch.where(text == self.space_token, cum_ends, torch.zeros_like(cum_ends))
        space_ends_invert = torch.where(text == self.space_token, cum_ends, audio_tokens.shape[2] * torch.ones_like(cum_ends))
        # [B]
        min_space = space_ends_invert.topk(k=2, dim=1, largest=False).values[:, 1]
        max_space = space_ends.topk(k=2, dim=1).values[:, 1]
        batch_size = audio_codes.shape[0]
        max_lens = torch.clamp_max(audio_lens // 2, max=self.context_max_len)
        min_lens = torch.clamp_max(max_lens, max=self.context_min_len)

        text_list = []
        dur_list = []
        text_len_list = []

        context_list = []
        context_len_list = []

        audio_token_list = []
        audio_code_list = []
        audio_token_len_list = []

        for i in range(batch_size):
            if sample:
                end_index = torch.randint(low=min_lens[i], high= max_lens[i] + 1, size=[]).item()
            else:
                end_index = max_lens[i].item()

            min_space_i = min_space[i].item()
            max_space_i = max_space[i].item()
            space_ends_i = space_ends[i]
            valid_space = torch.logical_and(space_ends_i >= min_space_i, space_ends_i <= max_space_i)
            valid_space = torch.logical_and(valid_space, space_ends_i <= end_index)
            valid_space = torch.logical_or(valid_space, space_ends_i == min_space_i)
            space_ends_i = torch.where(valid_space, space_ends_i, torch.zeros_like(space_ends_i))
            end_indices = space_ends_i.topk(k=1)
            end_index = end_indices.values[0].item()
            text_start_index = end_indices.indices[0].item() + 1

            audio_len_input_i = audio_lens[i].item()
            audio_code_len_i = audio_len_input_i - end_index
            context_i = audio_codes[i, :, :end_index]
            audio_tokens_i = audio_tokens[i, :, end_index:audio_len_input_i]
            audio_codes_i = audio_codes[i, :, end_index:audio_len_input_i]

            text_len_input_i = text_lens[i].item()
            text_len_i = text_len_input_i - text_start_index
            text_i = text[i, text_start_index:text_len_input_i]
            dur_i = durs[i, text_start_index:text_len_input_i]

            text_list.append(text_i)
            dur_list.append(dur_i)
            text_len_list.append(text_len_i)
            context_list.append(context_i)
            context_len_list.append(end_index)
            audio_token_list.append(audio_tokens_i)
            audio_code_list.append(audio_codes_i)
            audio_token_len_list.append(audio_code_len_i)

        max_text_len = max(text_len_list)
        out_text_lens = torch.tensor(text_len_list, dtype=torch.int32, device=audio_codes.device)
        out_text = stack_tensors(tensors=text_list, max_lens=[max_text_len]).to(audio_codes.device)
        out_durs = stack_tensors(tensors=dur_list, max_lens=[max_text_len]).to(audio_codes.device)

        max_audio_len = max(audio_token_len_list)
        out_audio_lens = torch.tensor(audio_token_len_list, dtype=torch.int32, device=audio_codes.device)
        out_audio_tokens = stack_tensors(tensors=audio_token_list, max_lens=[max_audio_len]).to(audio_codes.device)
        out_audio_codes = stack_tensors(tensors=audio_code_list, max_lens=[max_audio_len]).to(audio_codes.device)

        max_context_len = max(context_len_list)
        context_lens = torch.tensor(context_len_list, dtype=torch.int32, device=audio_codes.device)
        context_codes = stack_tensors(tensors=context_list, max_lens=[max_context_len]).to(audio_codes.device)

        return out_text, out_durs, out_text_lens, out_audio_tokens, out_audio_codes, out_audio_lens, \
               context_codes, context_lens

    def sample_context_audio_end(self, audio_tokens, audio_codes, audio_lens, text, durs):
        # [B, T_text]
        cum_ends = torch.cumsum(durs, dim=1).long()
        space_ends = torch.where(text == self.space_token, cum_ends, torch.zeros_like(cum_ends))
        space_ends_invert = torch.where(text == self.space_token, cum_ends,
                                        audio_tokens.shape[2] * torch.ones_like(cum_ends))
        # [B]
        min_space = space_ends_invert.topk(k=2, dim=1, largest=False).values[:, 1]
        max_space = space_ends.topk(k=2, dim=1).values[:, 1]
        batch_size = audio_codes.shape[0]
        max_lens = torch.clamp_max(audio_lens // 2, max=self.context_max_len)
        min_lens = torch.clamp_max(max_lens, max=self.context_min_len)

        text_list = []
        dur_list = []
        text_len_list = []

        context_list = []
        context_len_list = []

        audio_token_list = []
        audio_code_list = []
        audio_token_len_list = []

        for i in range(batch_size):
            audio_len_input_i = audio_lens[i].item()
            rand_len = torch.randint(low=min_lens[i], high=max_lens[i] + 1, size=[]).item()
            start_index = audio_len_input_i - rand_len

            min_space_i = min_space[i].item()
            max_space_i = max_space[i].item()
            space_ends_i = space_ends[i]
            valid_space = torch.logical_and(space_ends_i >= min_space_i, space_ends_i <= max_space_i)
            valid_space = torch.logical_and(valid_space, space_ends_i >= start_index)
            valid_space = torch.logical_or(valid_space, space_ends_i == max_space_i)
            space_ends_i = torch.where(valid_space, space_ends_i, audio_tokens.shape[2] * torch.ones_like(space_ends_i))
            start_indices = space_ends_i.topk(k=1, largest=False)
            start_index = start_indices.values[0].item()
            text_end_index = start_indices.indices[0].item() + 1

            context_len_i = audio_len_input_i - start_index
            context_i = audio_codes[i, :, start_index:audio_len_input_i]
            audio_tokens_i = audio_tokens[i, :, :start_index]
            audio_codes_i = audio_codes[i, :, :start_index]

            text_i = text[i, :text_end_index]
            dur_i = durs[i, :text_end_index]

            text_list.append(text_i)
            dur_list.append(dur_i)
            text_len_list.append(text_end_index)
            context_list.append(context_i)
            context_len_list.append(context_len_i)
            audio_token_list.append(audio_tokens_i)
            audio_code_list.append(audio_codes_i)
            audio_token_len_list.append(start_index)

        max_text_len = max(text_len_list)
        out_text_lens = torch.tensor(text_len_list, dtype=torch.int32, device=audio_codes.device)
        out_text = stack_tensors(tensors=text_list, max_lens=[max_text_len]).to(audio_codes.device)
        out_durs = stack_tensors(tensors=dur_list, max_lens=[max_text_len]).to(audio_codes.device)

        max_audio_len = max(audio_token_len_list)
        out_audio_lens = torch.tensor(audio_token_len_list, dtype=torch.int32, device=audio_codes.device)
        out_audio_tokens = stack_tensors(tensors=audio_token_list, max_lens=[max_audio_len]).to(audio_codes.device)
        out_audio_codes = stack_tensors(tensors=audio_code_list, max_lens=[max_audio_len]).to(audio_codes.device)

        max_context_len = max(context_len_list)
        context_lens = torch.tensor(context_len_list, dtype=torch.int32, device=audio_codes.device)
        context_codes = stack_tensors(tensors=context_list, max_lens=[max_context_len]).to(audio_codes.device)

        return out_text, out_durs, out_text_lens, out_audio_tokens, out_audio_codes, out_audio_lens, \
               context_codes, context_lens

    def get_context(self, audio_tokens, audio_lens, text, text_lens, attn_prior):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_lens).detach()

        context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_lens)
        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, _, _ = self.fastpitch.get_alignments(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
            context_emb=context_aligner_emb,
            attn_prior=attn_prior,
        )

        _, _, _, _, _, _, \
        context_codes, context_lens = self.sample_context_audio(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            sample=False
        )
        context_emb, context, context_lens = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    def get_context_audio(self, audio_tokens, audio_lens):
        batch_size = audio_tokens.shape[0]
        context_lens = torch.clamp_max(audio_lens // 2, max=self.context_max_len)
        context_token_list = []
        for i in range(batch_size):
            context_len_i = context_lens[i]
            context_tokens_i = audio_tokens[i, :, :context_len_i]
            context_token_list.append(context_tokens_i)

        max_context_len = max(context_lens)
        context_tokens = stack_tensors(tensors=context_token_list, max_lens=[max_context_len]).to(audio_tokens.device)

        context_tokens_rearrange = rearrange(context_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        context_codes = self.vector_quantizer.decode(indices=context_tokens_rearrange, input_len=audio_lens).detach()

        context_emb, context, context_lens = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    def _add_decoder_duration_noise(self, durs, text_lens):
        text_mask = get_mask_from_lengths(text_lens)
        max_text_len = text_mask.shape[1]

        indices = torch.arange(max_text_len, device=durs.device) + 1
        noise_mask = torch.where(
            rearrange(indices, 'T -> 1 T') == rearrange(text_lens, 'B -> B 1'),
            torch.zeros_like(text_mask),
            text_mask
        )

        end_indices = torch.cumsum(durs, dim=1)
        end_indices = end_indices * text_mask

        add_noise = torch.rand(size=durs.shape, device=durs.device) < self.decoder_duration_noise_percent
        noise_mask = noise_mask * add_noise
        shift_backward = torch.rand(size=durs.shape, device=durs.device) <= 0.5
        shift_backward = shift_backward * noise_mask
        shift_forward = ~shift_backward * noise_mask

        min_end_indices = torch.nn.functional.pad(end_indices[:, :-1], pad=[1, 0]) + 1
        end_indices_noise = end_indices - shift_backward.int()
        end_indices_noise = torch.maximum(end_indices_noise, min_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        max_end_indices = torch.nn.functional.pad(end_indices_noise[:, 1:] - 1, pad=[0, 1])
        end_indices_noise = end_indices_noise + shift_forward.int()
        end_indices_noise = torch.minimum(end_indices_noise, max_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        end_indices_noise = end_indices_noise.int()

        durs_noise = end_indices_noise - torch.nn.functional.pad(end_indices_noise[:, :-1], pad=[1, 0])
        durs_noise = durs_noise * text_mask

        return durs_noise


    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "attn_prior": NeuralType(('B', 'T_audio', 'T_text'), ProbsType(), optional=True),
            "sample_context_start": NeuralType((), BoolType(), optional=True),
            "sample_context_end": NeuralType((), BoolType(), optional=True),
            "add_duration_noise": NeuralType((), BoolType(), optional=True),
        },
        output_types={
            "audio_token_sample": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "audio_loss_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_loss_mask": NeuralType(('B', 'T_text'), MaskType()),
            "text_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "speaking_rate_pred": NeuralType(tuple('B'), FloatType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        }
    )
    def forward(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
        attn_prior=None,
        sample_context_start=False,
        sample_context_end=False,
        add_duration_noise=False,
    ):
        assert not (sample_context_start and sample_context_end)
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens).detach()

        context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_token_lens)
        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, align_hard, align_soft, align_logits = self.fastpitch.get_alignments(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            context_emb=context_aligner_emb,
            attn_prior=attn_prior,
        )

        if sample_context_start:
            text_sample, dur_sample, text_sample_lens, audio_token_sample, audio_codes_sample, audio_token_sample_lens, \
            context_codes, context_lens = self.sample_context_audio(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
                sample=True
            )
        elif sample_context_end:
            text_sample, dur_sample, text_sample_lens, audio_token_sample, audio_codes_sample, audio_token_sample_lens, \
            context_codes, context_lens = self.sample_context_audio_end(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                durs=durs
            )
        else:
            text_sample, dur_sample, text_sample_lens, audio_token_sample, audio_codes_sample, audio_token_sample_lens, \
            context_codes, context_lens = self.sample_context_audio(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
                sample=False
            )

        context_emb, context, context_lens = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )

        if sample_context_start or sample_context_end:
            audio_maskin, audio_loss_mask = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                parallel_prob=self.audio_parallel_prob,
                denoise_prob=self.audio_denoise_prob,
                infill_min=self.audio_infill_min,
                infill_max=self.audio_infill_max,
            )
            dur_maskin, dur_loss_mask = self.create_infill_mask(
                input_lens=text_sample_lens,
                parallel_prob=self.duration_parallel_prob,
                denoise_prob=self.duration_denoise_prob,
                infill_min=self.duration_infill_min,
                infill_max=self.duration_infill_max,
            )
        else:
            audio_maskin = torch.zeros(
                [audio_codes_sample.shape[0], audio_codes_sample.shape[2]], dtype=torch.bool, device=audio_tokens.device
            )
            dur_maskin = torch.zeros_like(text_sample, dtype=torch.bool)
            audio_loss_mask = None
            dur_loss_mask = None

        speaking_rate = self.fastpitch.get_speaking_rate(text_lens=text_lens, audio_lens=audio_token_lens)
        dur_indices = self.fastpitch.duration_to_index(durs=dur_sample, lengths=text_sample_lens)

        if add_duration_noise:
            dur_sample = self._add_decoder_duration_noise(durs=dur_sample, text_lens=text_sample_lens)

        audio_tokens_pred, audio_token_logits, dur_indices_pred, dur_logits, speaking_rate_pred = self.fastpitch(
            text=text_sample,
            text_lens=text_sample_lens,
            context_emb=context_emb,
            context=context,
            context_lens=context_lens,
            speaking_rate=speaking_rate,
            audio_codes=audio_codes_sample,
            audio_lens=audio_token_sample_lens,
            audio_maskin=audio_maskin,
            durs=dur_sample,
            dur_indices=dur_indices,
            dur_maskin=dur_maskin,
        )

        return (
            audio_token_sample,
            audio_token_sample_lens,
            audio_tokens_pred,
            audio_token_logits,
            audio_loss_mask,
            dur_indices,
            dur_loss_mask,
            text_sample_lens,
            dur_indices_pred,
            dur_logits,
            speaking_rate,
            speaking_rate_pred,
            align_hard,
            align_soft,
            align_logits,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "num_audio_iters": NeuralType((), IntType()),
            "attn_prior": NeuralType(('B', 'T_audio', 'T_text'), ProbsType(), optional=True)
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
        }
    )
    def infer_gta(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
        num_audio_iters=1,
        attn_prior=None
    ):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T').detach()
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens)

        context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_token_lens)
        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, align_hard, align_soft, align_logits = self.fastpitch.get_alignments(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            context_emb=context_aligner_emb,
            attn_prior=attn_prior,
        )

        text_sample, dur_sample, text_sample_lens, audio_token_sample, audio_codes_sample, audio_token_sample_lens, \
        context_codes, context_lens = self.sample_context_audio(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            sample=False
        )
        context_emb, context, context_lens = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )

        audio_tokens_pred = self.fastpitch.infer_gta(
            text=text,
            text_lens=text_lens,
            context_emb=context_emb,
            context=context,
            context_lens=context_lens,
            audio_lens=audio_token_lens,
            durs=durs,
            num_audio_iters=num_audio_iters,
        )
        return audio_tokens_pred, align_soft

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)

        if batch_idx % 2 == 0:
            sample_context_start = True
            sample_context_end = False
        else:
            sample_context_start = False
            sample_context_end = True

        (
            audio_token_sample,
            audio_token_sample_lens,
            _,
            audio_token_logits,
            audio_loss_mask,
            dur_indices,
            dur_loss_mask,
            text_sample_lens,
            _,
            dur_logits,
            speaking_rate,
            speaking_rate_pred,
            align_hard,
            align_soft,
            align_logits
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
            attn_prior=attn_prior,
            sample_context_start=sample_context_start,
            sample_context_end=sample_context_end,
            add_duration_noise=True
        )

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_token_sample, mask=audio_loss_mask
        )
        train_audio_token_loss = self.audio_token_loss_scale * audio_token_loss

        duration_loss = self.duration_loss_fn(logits=dur_logits, target_index=dur_indices.detach(), mask=dur_loss_mask)
        train_dur_loss = self.duration_loss_scale * duration_loss

        speaking_rate_loss = self.speaking_rate_loss_fn(input=speaking_rate_pred, target=speaking_rate.detach())
        train_speaking_rate_loss = self.speaking_rate_loss_scale * speaking_rate_loss

        ctc_loss = self.forward_sum_loss_fn(attn_logprob=align_logits, in_lens=text_lens, out_lens=audio_token_lens)
        train_ctc_loss = self.aligner_ctc_loss_scale * ctc_loss

        bin_loss = self.bin_loss_fn(hard_attention=align_hard, soft_attention=align_soft)
        if self.current_epoch < self.bin_loss_start_epoch:
            bin_loss_weight = 0.0
        elif self.current_epoch > self.bin_loss_warmup_epochs:
            bin_loss_weight = 1.0
        else:
            bin_loss_weight = (self.current_epoch - self.bin_loss_start_epoch) / (self.bin_loss_warmup_epochs - self.bin_loss_start_epoch)
        train_bin_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_loss

        loss = train_audio_token_loss + train_dur_loss + train_speaking_rate_loss + train_ctc_loss + train_bin_loss

        metrics = {
            "t_audio_token_loss": audio_token_loss,
            "t_duration_loss": duration_loss,
            "t_speaking_rate_loss": speaking_rate_loss,
            "t_ctc_loss": ctc_loss,
            "t_bin_loss": bin_loss,
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", audio_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        attn_prior = batch_dict.get("align_prior_matrix", None)

        (
            audio_token_sample,
            audio_token_sample_lens,
            audio_tokens_pred,
            audio_token_logits,
            _,
            dur_indices,
            _,
            text_sample_lens,
            dur_indices_pred,
            dur_logits,
            speaking_rate,
            speaking_rate_pred,
            align_hard,
            align_soft,
            align_logits
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
            attn_prior=attn_prior
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)
        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_token_sample, mask=audio_mask
        )
        audio_token_correct = mask_sequence_tensor(audio_token_sample == audio_tokens_pred, audio_token_sample_lens).sum()
        audio_token_accuracy = audio_token_correct / audio_token_sample_lens.sum() / self.num_codebooks

        text_mask = get_mask_from_lengths(text_sample_lens)
        duration_loss = self.duration_loss_fn(
            logits=dur_logits,
            target_index=dur_indices,
            mask=text_mask
        )
        dur_token_correct = mask_sequence_tensor(dur_indices == dur_indices_pred, text_sample_lens).sum()
        dur_token_accuracy = dur_token_correct / text_sample_lens.sum()

        speaking_rate_loss = self.speaking_rate_loss_fn(input=speaking_rate_pred, target=speaking_rate)

        metrics = {
            "val_loss": audio_token_loss,
            "val_audio_token_loss": audio_token_loss,
            "val_audio_token_accuracy": audio_token_accuracy,
            "val_duration_loss": duration_loss,
            "val_dur_token_accuracy": dur_token_accuracy,
            "val_speaking_rate_loss": speaking_rate_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True
        )

        sampler = dataset.get_sampler(dataloader_params.batch_size, world_size=self.trainer.world_size)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **dataloader_params
        )

    def _setup_test_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=False
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def configure_callbacks(self):
        if not self.log_config:
            return []

        data_loader = self._setup_test_dataloader(
            dataset_config=self.log_config.dataset, dataloader_params=self.log_config.dataloader_params
        )
        generators = instantiate(self.log_config.generators)
        log_dir = Path(self.log_config.log_dir) if self.log_config.log_dir else None
        log_callback = LoggingCallback(
            generators=generators,
            data_loader=data_loader,
            log_epochs=self.log_config.log_epochs,
            epoch_frequency=self.log_config.epoch_frequency,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=self.log_config.log_tensorboard,
            log_wandb=self.log_config.log_wandb,
        )

        return [log_callback]

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
