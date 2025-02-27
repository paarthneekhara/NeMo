

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

import math
import torch
from einops import rearrange

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.modules.common import DropoutWithoutScaling
from nemo.collections.tts.modules.transformer import PositionalEmbedding
from nemo.collections.tts.parts.utils.helpers import (
    binarize_attention_parallel,
    get_mask_from_lengths,
    regulate_len
)
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
    VoidType
)
from nemo.core.neural_types.neural_type import NeuralType


def get_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


class Conv1d(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation=None):
        super().__init__()
        padding = get_padding(kernel_size=kernel_size, stride=stride)
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if activation is None:
            self.activation = None
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, mask):
        out = self.conv(inputs)
        if self.activation:
            out = self.activation(out)
        out = out * rearrange(mask, 'B T -> B 1 T')
        return out


class PreNet(NeuralModule):

    def __init__(self, input_dim, output_dim, bottleneck_dim, dropout_rate):
        super(PreNet, self).__init__()
        self.dropout = DropoutWithoutScaling(dropout_rate=dropout_rate)
        self.bottleneck_layer = torch.nn.Linear(input_dim, bottleneck_dim)
        self.hidden_layer = torch.nn.Linear(bottleneck_dim, output_dim)
        self.output_layer = torch.nn.Linear(output_dim, output_dim)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, inputs, mask):
        out = self.bottleneck_layer(inputs)
        out = self.dropout(out)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        out = out * rearrange(mask, 'B T -> B T 1')
        return out


class ContextEncoder(NeuralModule):

    def __init__(
        self,
        input_dim,
        encoder,
        rnn_layers,
        rnn_dim,
    ):
        super(ContextEncoder, self).__init__()
        d_model = encoder.d_model
        self.pre_conv = Conv1d(
            in_channels=input_dim,
            out_channels=d_model
        )
        self.encoder = encoder
        self.rnn = torch.nn.LSTM(input_size=d_model, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True)
        self.emb_layer = torch.nn.Linear(in_features=rnn_dim, out_features=d_model)

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)
        context = rearrange(context, 'B D T -> B T D')
        context = self.encoder(inputs=context, mask=mask)

        out = torch.nn.utils.rnn.pack_padded_sequence(context, audio_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, padded_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # [B, D]
        out = out[torch.arange(len(padded_lens)), (padded_lens - 1), :]
        context_emb = self.emb_layer(out)

        context = rearrange(context, 'B T D -> B D T')

        return context_emb, context, audio_lens


class ContextUtteranceEncoder(NeuralModule):

    def __init__(
        self,
        input_dim,
        context_emb_dim,
        filters,
        rnn_layers,
        rnn_dim,
    ):
        super(ContextUtteranceEncoder, self).__init__()
        self.conv1 = Conv1d(
            in_channels=input_dim,
            out_channels=filters,
            activation="lrelu"
        )
        self.conv2 = Conv1d(
            in_channels=filters,
            out_channels=filters,
            activation="lrelu"
        )
        self.rnn = torch.nn.LSTM(input_size=filters, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True)
        self.emb_layer = torch.nn.Linear(in_features=rnn_dim, out_features=context_emb_dim)

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)

        out = self.conv1(inputs=audio_codes, mask=mask)
        out = self.conv2(inputs=out, mask=mask)

        out = rearrange(out, 'B D T -> B T D')
        out = torch.nn.utils.rnn.pack_padded_sequence(out, audio_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, padded_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # [B, D]
        out = out[torch.arange(len(padded_lens)), (padded_lens - 1), :]
        context_emb = self.emb_layer(out)
        return context_emb


class SpeakingRatePredictor(NeuralModule):

    def __init__(self, context_dim):
        super(SpeakingRatePredictor, self).__init__()
        self.hidden_layer = torch.nn.Linear(context_dim, context_dim)
        self.speaking_rate_layer = torch.nn.Linear(context_dim, 1)

    @property
    def input_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation())
        }

    @property
    def output_types(self):
        return {
            "speaking_rate": NeuralType(tuple('B'), FloatType())
        }

    @typecheck()
    def forward(self, context_emb):
        out = self.hidden_layer(context_emb)
        out = self.speaking_rate_layer(out)
        out = rearrange(out, 'B 1 -> B')
        return out


class DurationDecoder(NeuralModule):

    def __init__(self, fft, num_duration=50, emb_dim=128, dropout_rate=0.1):
        super(DurationDecoder, self).__init__()
        d_model = fft.d_model
        self.fft = fft

        self.dur_emb = torch.nn.Embedding(num_duration, emb_dim, _weight=torch.zeros([num_duration, emb_dim]))
        self.dur_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))
        self.dur_cond_layer = torch.nn.Linear(emb_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.duration_layer = torch.nn.Linear(d_model, num_duration)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "dur_maskin": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'C', 'T_text'), LogitsType())
        }

    @typecheck()
    def forward(self, inputs, dur_indices, text_lens, dur_maskin):
        text_mask = get_mask_from_lengths(text_lens)
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        dur_emb = self.dur_emb(dur_indices.detach())
        dur_res = self.dur_cond_layer(dur_emb)
        dur_res = self.dropout(dur_res)
        dur_res = dur_res * rearrange(dur_maskin, 'B T -> B T 1')

        masked_mask = ~dur_maskin * text_mask
        mask_res = self.dur_mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + dur_res + mask_res

        # [B, T, D]
        dec_input = dec_input * rearrange(text_mask, 'B T -> B T 1')
        dec_out = self.fft(inputs=dec_input, mask=text_mask)

        # [B, T, num_codes]
        dur_logits = self.duration_layer(dec_out)
        dur_logits = dur_logits * text_mask_3d
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        # [B, T]
        dur_indices_pred = dur_logits.max(dim=1).indices
        dur_indices_pred = dur_indices_pred * text_mask

        return dur_indices_pred, dur_logits


class AudioDecoder(NeuralModule):

    def __init__(self, pre_net, fft, num_codebooks, codebook_size, codebook_dim):
        super(AudioDecoder, self).__init__()
        self.hidden_dim = fft.d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.hidden_dim]))
        self.fft = fft
        self.pre_net = pre_net

        self.audio_token_layer = torch.nn.Linear(self.hidden_dim, self.num_logits)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
            "audio_maskin": NeuralType(('B', 'T_audio'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, context, context_mask, audio_codes, audio_maskin):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        audio_res = self.pre_net(inputs=audio_codes, mask=audio_maskin)

        masked_mask = ~audio_maskin * audio_mask
        mask_res = self.audio_mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + audio_res + mask_res

        dec_input = dec_input * audio_mask_3d
        dec_out = self.fft(inputs=dec_input, audio_mask=audio_mask, context=context, context_mask=context_mask)

        # [batch_size, audio_len, num_codebook * codebook_size]
        audio_logits = self.audio_token_layer(dec_out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        audio_tokens = audio_logits.max(dim=3).indices
        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits


class FastPitchCodecModule(NeuralModule):
    def __init__(
        self,
        aligner_module: NeuralModule,
        vector_quantizer_module: NeuralModule,
        text_encoder_module: NeuralModule,
        audio_decoder_module: NeuralModule,
        duration_decoder_module: NeuralModule,
        speaking_rate_predictor_module: NeuralModule,
        min_token_duration: int = 1,
        max_token_duration: int = 50,
        speaking_rate_factor: float = 5.0,
        speaking_rate_dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.text_hidden_dim = text_encoder_module.d_model
        self.audio_hidden_dim = audio_decoder_module.hidden_dim
        self.min_token_duration = min_token_duration
        self.max_token_duration = max_token_duration
        self.speaking_rate_factor = speaking_rate_factor

        if self.text_hidden_dim != self.audio_hidden_dim:
            self.audio_input_layer = torch.nn.Linear(self.text_hidden_dim, self.audio_hidden_dim)
        else:
            self.audio_input_layer = None

        self.positional_embedding = PositionalEmbedding(self.audio_hidden_dim)

        self.aligner = aligner_module
        self.vector_quantizer = vector_quantizer_module

        self.text_encoder = text_encoder_module
        self.audio_decoder = audio_decoder_module
        self.duration_decoder = duration_decoder_module
        self.speaking_rate_predictor = speaking_rate_predictor_module

        self.speaking_rate_cond_layer = torch.nn.Linear(1, self.text_hidden_dim)
        self.speaking_rate_dropout = torch.nn.Dropout(speaking_rate_dropout_rate)

    def _condition_on_speaking_rate(self, inputs, speaking_rate, mask):
        speaking_rate = rearrange(speaking_rate, 'B -> B 1 1').detach()
        # [B, T, hidden_dim]
        sr_res = self.speaking_rate_cond_layer(speaking_rate)
        sr_res = self.speaking_rate_dropout(sr_res)

        out = inputs + sr_res

        out = out * rearrange(mask, 'B T -> B T 1')
        return out

    def _get_audio_input(self, text_enc, durs):
        audio_input = text_enc

        if self.audio_input_layer is not None:
            audio_input = self.audio_input_layer(audio_input)

        audio_input, audio_lens = regulate_len(durs, audio_input, pace=1.0)
        audio_mask = get_mask_from_lengths(audio_lens)

        max_audio_len = audio_mask.shape[1]
        pos_seq = torch.arange(max_audio_len, device=text_enc.device).to(text_enc.dtype)
        pos_emb = self.positional_embedding(pos_seq)

        audio_input = audio_input + pos_emb
        audio_input = audio_input * rearrange(audio_mask, 'B T -> B T 1')

        return audio_input, audio_lens

    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "num_iters": NeuralType((), IntType()),
        },
        output_types={
            "audio_tokens": NeuralType(('B', 'C', 'T_token'), TokenIndex())
        }
    )
    def _audio_token_infer(self, inputs, audio_lens, context, context_mask, num_iters):
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)

        batch_size = inputs.shape[0]
        num_tokens = inputs.shape[1]
        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        pad_frames = (num_iters - 1) - ((num_tokens - 1) % num_iters)
        padded_length = num_tokens + pad_frames

        # [B, T]
        audio_maskin = torch.zeros_like(audio_mask, dtype=torch.bool)
        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.audio_decoder.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1], self.audio_decoder.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(num_iters):
            # [B, C, T], [B, C, W, T]
            audio_tokens_i, audio_logits = self.audio_decoder(
                inputs=inputs,
                audio_mask=audio_mask,
                context=context,
                context_mask=context_mask,
                audio_codes=audio_codes,
                audio_maskin=audio_maskin,
            )
            audio_tokens_i = rearrange(audio_tokens_i, 'B C T -> B T C')
            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            audio_codes_i = self.vector_quantizer.decode(indices=audio_tokens_rearrange_i, input_len=audio_lens)
            audio_codes_i = rearrange(audio_codes_i, 'B D T -> B T D')

            if i == 0:
                top_i = torch.clamp_max(index_shift, max=num_tokens - 1)
            else:
                # [B, C, T]
                logits = audio_logits.max(dim=2).values
                # [B, T]
                logits = logits.sum(dim=1)

                logits = torch.where(audio_maskin, -100.0, logits)
                logits = torch.where(~audio_mask, -100.0, logits)
                logits = torch.nn.functional.pad(logits, pad=[0, pad_frames], value=-100.0)
                # [B, T // num_iters, num_iters]
                logit_shape = (batch_size, padded_length // num_iters, num_iters)
                logits = torch.reshape(logits, logit_shape)
                # [B, T // num_iters, 1]
                top_i = torch.topk(logits, k=1, dim=2).indices
                top_i = rearrange(top_i, 'B T 1 -> B T')
                # [B, T // num_iters]
                top_i = top_i + index_shift
                top_i = torch.clamp_max(top_i, max=num_tokens - 1)

            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(audio_mask, maskin_i, False)
            maskin_i = torch.where(audio_maskin, False, maskin_i)
            maskin_3d_i = rearrange(maskin_i, 'B T -> B T 1')

            audio_tokens = torch.where(maskin_3d_i, audio_tokens_i, audio_tokens)
            audio_codes = torch.where(maskin_3d_i, audio_codes_i, audio_codes)
            audio_maskin = torch.logical_or(audio_maskin, maskin_i)

        audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
        audio_tokens = torch.where(audio_maskin_3d, audio_tokens, audio_tokens_i)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens


    @typecheck(
        input_types={
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "num_iters": NeuralType((), IntType()),
        },
        output_types={
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "durs_indices": NeuralType(('B', 'T_text'), TokenIndex()),
        }
    )
    def _duration_infer(self, inputs, text_lens, num_iters):
        # [B, T]
        text_mask = get_mask_from_lengths(text_lens)

        batch_size = inputs.shape[0]
        num_tokens = inputs.shape[1]
        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        pad_frames = (num_iters - 1) - ((num_tokens - 1) % num_iters)
        padded_length = num_tokens + pad_frames

        dur_maskin = torch.zeros_like(text_mask, dtype=torch.bool)
        durs = torch.zeros_like(text_mask, dtype=torch.int)
        dur_indices = torch.zeros_like(text_mask, dtype=torch.int)

        for i in range(num_iters):
            dur_indices_i, dur_logits = self.duration_decoder(
                inputs=inputs, dur_indices=dur_indices, text_lens=text_lens, dur_maskin=dur_maskin
            )
            durs_i = self.index_to_duration(dur_indices=dur_indices_i, mask=text_mask)

            if i == 0:
                top_i = torch.clamp_max(index_shift, max=num_tokens - 1)
            else:
                # [B, T]
                logits = dur_logits.max(dim=1).values

                logits = torch.where(dur_maskin, -100.0, logits)
                logits = torch.where(~text_mask, -100.0, logits)
                logits = torch.nn.functional.pad(logits, pad=[0, pad_frames], value=-100.0)
                # [B, T // num_iters, num_iters]
                logit_shape = (batch_size, padded_length // num_iters, num_iters)
                logits = torch.reshape(logits, logit_shape)
                # [B, T // num_iters, 1]
                top_i = torch.topk(logits, k=1, dim=2).indices
                top_i = rearrange(top_i, 'B T 1 -> B T')
                # [B, T // num_iters]
                top_i = top_i + index_shift
                top_i = torch.clamp_max(top_i, max=num_tokens - 1)

            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(text_mask, maskin_i, False)
            maskin_i = torch.where(dur_maskin, False, maskin_i)

            durs = torch.where(maskin_i, durs_i, durs)
            dur_indices = torch.where(maskin_i, dur_indices_i, dur_indices)
            dur_maskin = torch.logical_or(dur_maskin, maskin_i)

        durs = torch.where(dur_maskin, durs, durs_i)
        dur_indices = torch.where(dur_maskin, dur_indices, dur_indices_i)

        return durs, dur_indices

    @typecheck(
        input_types={
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "lengths": NeuralType(tuple('B'), LengthsType())
        },
        output_types={
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex())
        }
    )
    def duration_to_index(self, durs, lengths):
        durs = torch.clamp(durs.float(), min=self.min_token_duration, max=self.max_token_duration)
        dur_indices = durs - self.min_token_duration
        dur_indices = mask_sequence_tensor(tensor=dur_indices, lengths=lengths)
        dur_indices = dur_indices.int()
        return dur_indices

    def index_to_duration(self, dur_indices, mask):
        # [B, T]
        durs = dur_indices + self.min_token_duration
        durs = durs * mask
        return durs

    @typecheck(
        input_types={
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_lens": NeuralType(tuple('B'), LengthsType())
        },
        output_types={
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
        }
    )
    def get_speaking_rate(self, text_lens, audio_lens):
        speaking_rate = self.speaking_rate_factor * text_lens.float() / audio_lens.float()
        return speaking_rate

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'D', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "attn_prior": NeuralType(('B', 'T_audio', 'T_text'), ProbsType()),
        },
        output_types={
            "durations": NeuralType(('B', 'T_text'), TokenDurationType()),
            "attn_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType())
        }
    )
    def get_alignments(self, text, text_lens, audio_codes, audio_lens, context_emb, attn_prior=None):
        text_mask = get_mask_from_lengths(text_lens)
        text_mask = rearrange(text_mask, "B T_text -> B T_text 1")
        # Aligner requires an inverted mask
        aligner_text_mask = text_mask == 0
        # [batch_size, 1, hidden_dim]
        context_emb = rearrange(context_emb, 'B D -> B 1 D')

        # [batch_size, text_len, hidden_dim]
        text_emb = self.text_encoder.word_emb(text)
        text_emb = rearrange(text_emb, "B T_text D -> B D T_text")

        # [batch_size, 1, audio_len, text_len]
        attn_soft, attn_logprob = self.aligner(
            queries=audio_codes, keys=text_emb, mask=aligner_text_mask, attn_prior=attn_prior, conditioning=context_emb
        )
        attn_hard = binarize_attention_parallel(attn=attn_soft, in_lens=text_lens, out_lens=audio_lens)

        durations = attn_hard.sum(2)
        durations = rearrange(durations, 'B 1 T_text -> B T_text')
        return durations, attn_hard, attn_soft, attn_logprob

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_maskin": NeuralType(('B', 'T'), MaskType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_maskin": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "speaking_rate_pred": NeuralType(tuple('B'), FloatType()),
        }

    @typecheck()
    def forward(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        speaking_rate,
        audio_codes,
        audio_lens,
        audio_maskin,
        durs,
        dur_indices,
        dur_maskin,
    ):
        audio_mask = get_mask_from_lengths(audio_lens)
        # [batch_size, text_len]
        text_mask = get_mask_from_lengths(text_lens)
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')
        speaking_rate_pred = self.speaking_rate_predictor(context_emb=context_emb)
        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(text=text, text_mask=text_mask, context_emb=context_emb)
        text_enc_sr_cond = self._condition_on_speaking_rate(
            inputs=text_enc, speaking_rate=speaking_rate, mask=text_mask
        )

        dur_indices_pred, dur_logits = self.duration_decoder(
            inputs=text_enc_sr_cond, dur_indices=dur_indices, text_lens=text_lens, dur_maskin=dur_maskin
        )

        audio_codes = rearrange(audio_codes, 'B C T -> B T C')
        audio_input, _ = self._get_audio_input(text_enc=text_enc, durs=durs)
        audio_tokens_pred, audio_logits = self.audio_decoder(
            inputs=audio_input,
            audio_mask=audio_mask,
            context=context,
            context_mask=context_mask,
            audio_codes=audio_codes,
            audio_maskin=audio_maskin
        )

        return audio_tokens_pred, audio_logits, dur_indices_pred, dur_logits, speaking_rate_pred

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "num_audio_iters": NeuralType((), IntType()),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_token'), TokenIndex())
        }
    )
    def infer_gta(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        audio_lens,
        durs,
        num_audio_iters=1,
    ):
        audio_mask = get_mask_from_lengths(audio_lens)
        # [batch_size, text_len]
        text_mask = get_mask_from_lengths(text_lens)
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')
        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(text=text, text_mask=text_mask, context_emb=context_emb)

        audio_input, _ = self._get_audio_input(text_enc=text_enc, durs=durs)
        audio_tokens_pred = self._audio_token_infer(
            inputs=audio_input,
            audio_lens=audio_lens,
            context=context,
            context_mask=context_mask,
            num_iters=num_audio_iters,
        )

        return audio_tokens_pred

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "num_audio_iters": NeuralType((), IntType()),
            "num_duration_iters": NeuralType((), IntType()),
            "speaking_rate": NeuralType(tuple('B'), FloatType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType())
        }
    )
    def infer(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        num_audio_iters=1,
        num_duration_iters=1,
        speaking_rate=None
    ):
        # [batch_size, text_len]
        text_mask = get_mask_from_lengths(text_lens)
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')

        if speaking_rate is None:
            speaking_rate = self.speaking_rate_predictor(context_emb=context_emb)

        # [batch_size, text_len, hidden_dim]
        text_enc = self.text_encoder(text=text, text_mask=text_mask, context_emb=context_emb)
        text_enc_sr_cond = self._condition_on_speaking_rate(
            inputs=text_enc, speaking_rate=speaking_rate, mask=text_mask
        )

        durs, dur_indices = self._duration_infer(
            inputs=text_enc_sr_cond,
            text_lens=text_lens,
            num_iters=num_duration_iters,
        )

        audio_input, audio_lens = self._get_audio_input(text_enc=text_enc, durs=durs)
        audio_tokens_pred = self._audio_token_infer(
            inputs=audio_input,
            audio_lens=audio_lens,
            context=context,
            context_mask=context_mask,
            num_iters=num_audio_iters,
        )

        return audio_tokens_pred, audio_lens