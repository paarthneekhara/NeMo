# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from nemo.collections.tts.modules.submodules import ConditionalInput, ConditionalLayerNorm, LinearNorm
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
)
from nemo.core.classes import NeuralModule, adapter_mixins, typecheck
from nemo.core.neural_types.elements import EncodedRepresentation, LengthsType, MaskType, TokenDurationType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        #        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0))

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].repeat(bsz, 1, 1)
        else:
            return pos_emb[None, :, :]


class PositionwiseConvFF(nn.Module):
    def __init__(
        self, d_model, d_inner, kernel_size, dropout, pre_lnorm, condition_dim=None, condition_types=None
    ):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size[0], 1, (kernel_size[0] // 2)),
            #nn.ReLU(),
            nn.LeakyReLU(0.01),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size[1], 1, (kernel_size[1] // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = ConditionalLayerNorm(
            hidden_dim=d_model, condition_dim=condition_dim, condition_types=condition_types
        )
        self.pre_lnorm = pre_lnorm

    def forward(self, inp, conditioning=None):
        return self._forward(inp, conditioning)

    def _forward(self, inp, conditioning=None):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.layer_norm(inp, conditioning).to(inp.dtype)
            core_out = core_out.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out, conditioning).to(inp.dtype)

        return output


class CausalPositionwiseConvFF(nn.Module):
    def __init__(
        self, d_model, d_inner, kernel_size, dropout, condition_dim=None, condition_types=None
    ):
        super(CausalPositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = nn.Dropout(dropout)
        self.pad_len = kernel_size // 2

        self.input_conv = nn.Conv1d(in_channels=d_model, out_channels=d_inner, kernel_size=kernel_size)
        self.activation = nn.LeakyReLU(0.01)
        self.hidden_conv = nn.Conv1d(in_channels=d_inner, out_channels=d_model, kernel_size=kernel_size)

        self.layer_norm = ConditionalLayerNorm(
            hidden_dim=d_model, condition_dim=condition_dim, condition_types=condition_types
        )

    def forward(self, inp, conditioning=None):
        return self._forward(inp, conditioning)

    def _forward(self, inp, conditioning=None):
        # positionwise feed-forward
        core_out = inp.transpose(1, 2)
        core_out = F.pad(core_out, (self.pad_len, 0))
        core_out = self.input_conv(core_out)
        core_out = self.activation(core_out)
        core_out = F.pad(core_out, (self.pad_len, 0))
        core_out = self.hidden_conv(core_out)
        core_out = core_out.transpose(1, 2)

        # residual connection + layer normalization
        output = self.layer_norm(inp + core_out, conditioning).to(inp.dtype)

        return output


class LinearFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(LinearFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.input_layer = nn.Linear(d_model, d_inner)
        self.activation = nn.LeakyReLU(0.01)
        self.out_layer = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, conditioning=None):
        return self._forward(inp, conditioning)

    def _forward(self, inp, conditioning=None):
        res = self.input_layer(inp)
        res = self.activation(res)
        res = self.out_layer(res)
        res = self.dropout(res)

        output = self.layer_norm(inp + res)
        output = output.to(inp.dtype)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        pre_lnorm,
        dropatt=0.1,
        condition_dim=None,
        condition_types=None,
        causal=False
    ):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm
        self.causal = causal

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = ConditionalLayerNorm(
            hidden_dim=d_model, condition_dim=condition_dim, condition_types=condition_types
        )

    def forward(self, inp, mask=None, conditioning=None):
        return self._forward(inp, mask, conditioning)

    def _forward(self, inp, mask=None, conditioning=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp, conditioning)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)

        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if mask is not None:
            # [B, 1, T]
            attn_mask = rearrange(~mask, 'B T -> B 1 T')
            # [B * n_head, T, T]
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)

            if self.causal:
                # [B, T, T]
                attn_mask = ~torch.tril(~attn_mask)

            attn_mask = attn_mask.to(attn_score.dtype)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out, conditioning)

        output = output * rearrange(mask, 'B T -> B T 1')

        return output


class CrossAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_encoded,
        d_head,
        dropout,
        dropatt,
        pre_lnorm,
    ):
        super(CrossAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.query_layer = nn.Linear(d_model, n_head * d_head)
        self.key_layer = nn.Linear(d_encoded, n_head * d_head)
        self.value_layer = nn.Linear(d_encoded, n_head * d_head)
        self.dropout = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.out_layer = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, inputs, encoded, mask, attn_mask):
        # Inputs: [B, T1, D]
        # Encoded: [B, T2, D]
        # Mask: [B, T1]
        # Attn Mask: [B, T1, T2]
        batch_size = inputs.size(0)
        input_max_len = inputs.size(1)
        encoded_max_len = encoded.size(1)

        n_head, d_head = self.n_head, self.d_head

        if self.pre_lnorm:
            # layer normalization
            inputs = self.layer_norm(inputs)

        # [B, T, num_head * head_dim]
        head_q = self.query_layer(inputs)
        head_k = self.key_layer(encoded)
        head_v = self.value_layer(encoded)

        # [B, T, num_head, head_dim]
        head_q = head_q.view(batch_size, input_max_len, n_head, d_head)
        head_k = head_k.view(batch_size, encoded_max_len, n_head, d_head)
        head_v = head_v.view(batch_size, encoded_max_len, n_head, d_head)

        head_q = rearrange(head_q, 'B T H D -> H B T D')
        head_k = rearrange(head_k, 'B T H D -> H B T D')
        head_v = rearrange(head_v, 'B T H D -> H B T D')

        # [num_head * B, T1, head_dim]
        q = head_q.reshape(-1, input_max_len, d_head)
        # [num_head * B, T2, head_dim]
        k = head_k.reshape(-1, encoded_max_len, d_head)
        v = head_v.reshape(-1, encoded_max_len, d_head)

        # [num_head * B, head_Dim, T2]
        k = rearrange(k, 'H T D -> H D T')

        # [num_head * B, T1, T2]
        attn_score = torch.bmm(q, k)
        attn_score.mul_(self.scale)

        # [n_head * B, T1, T2]
        attn_mask = attn_mask.repeat(n_head, 1, 1)
        attn_score.masked_fill_(~attn_mask, -float('inf'))

        # [num_head * B, T1, T2]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = torch.nan_to_num(attn_prob, nan=0.0)
        attn_prob = self.dropatt(attn_prob)
        # [num_head * B, T1, head_dim]
        attn_vec = torch.bmm(attn_prob, v)

        # [num_head, B, T1, head_dim]
        attn_vec = attn_vec.view(n_head, batch_size, input_max_len, d_head)
        attn_vec = rearrange(attn_vec, 'H B T D -> B T H D')
        # [B, T1, num_head * head_dim]
        attn_vec = attn_vec.contiguous().view(batch_size, input_max_len, n_head * d_head)

        # [B, T1, D]
        res = self.out_layer(attn_vec)
        res = self.dropout(res)
        output = inputs + res

        if not self.pre_lnorm:
            output = self.layer_norm(output)

        output = output * rearrange(mask, 'B T -> B T 1')

        return output


class TransformerLayer(nn.Module, adapter_mixins.AdapterModuleMixin):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        condition_dim=None,
        condition_types=None,
        causal=False,
        pre_lnorm=False,
    ):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=dropatt,
            condition_dim=condition_dim,
            condition_types=condition_types,
            causal=causal,
            pre_lnorm=pre_lnorm,
        )
        if kernel_size == 1:
            self.pos_ff = LinearFF(d_model, d_inner, dropout)
        elif causal:
            self.pos_ff = CausalPositionwiseConvFF(
                d_model,
                d_inner,
                kernel_size,
                dropout,
                condition_dim=condition_dim,
                condition_types=condition_types
            )
        else:
            self.pos_ff = PositionwiseConvFF(
                d_model,
                d_inner,
                kernel_size,
                dropout,
                pre_lnorm=pre_lnorm,
                condition_dim=condition_dim,
                condition_types=condition_types
            )

    def forward(self, dec_inp, mask=None, conditioning=None):
        output = self.dec_attn(dec_inp, mask=mask, conditioning=conditioning)
        output = self.pos_ff(output, conditioning)
        output = output * rearrange(mask, 'B T -> B T 1')

        if self.is_adapter_available():
            output = self.forward_enabled_adapters(output)
            output = output * mask

        return output


class FFTransformerDecoder(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        condition_dim=None,
        condition_types=None,
        causal=False
    ):
        super(FFTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()
        self.cond_input = ConditionalInput(
            hidden_dim=d_model, condition_dim=condition_dim, condition_types=condition_types
        )

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head,
                    d_model,
                    d_head,
                    d_inner,
                    kernel_size,
                    dropout,
                    dropatt=dropatt,
                    pre_lnorm=pre_lnorm,
                    condition_dim=condition_dim,
                    condition_types=condition_types,
                    causal=causal
                )
            )

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "seq_lens": NeuralType(('B'), LengthsType()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T', 'D'), MaskType()),
        }

    @typecheck()
    def forward(self, input, seq_lens, conditioning=None):
        mask = mask_from_lens(seq_lens)
        return self._forward(input, mask, conditioning)

    def _forward(self, inp, mask, conditioning):
        max_len = inp.size(1)
        mask_3d = rearrange(mask, 'B T -> B T 1')
        pos_seq = torch.arange(max_len, device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask_3d
        inp = inp + pos_emb
        inp = self.cond_input(inp, conditioning)
        out = self.drop(inp)

        for layer in self.layers:
            out = layer(out, mask=mask, conditioning=conditioning)

        # out = self.drop(out)
        return out, mask


class FFTransformerEncoder(FFTransformerDecoder):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        n_embed=None,
        d_embed=None,
        padding_idx=0,
        condition_dim=None,
        condition_types=None,
    ):
        super(FFTransformerEncoder, self).__init__(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            dropemb,
            pre_lnorm,
            condition_dim,
            condition_types,
        )

        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(n_embed, d_embed or d_model, padding_idx=self.padding_idx)

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T'), TokenIndex()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }

    def forward(self, input, conditioning=0):

        return self._forward(self.word_emb(input), (input != self.padding_idx).unsqueeze(2), conditioning)  # (B, L, 1)


class FFTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=1,
        n_layers=6,
        n_head=1,
        d_head=64,
        d_inner=1024,
        kernel_size=3,
        dropout=0.1,
        dropatt=0.1,
        dropemb=0.0,
        causal=False
    ):
        super(FFTransformer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.d_head = d_head

        self.pos_emb = PositionalEmbedding(self.in_dim)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                TransformerLayer(
                    n_head, in_dim, d_head, d_inner, kernel_size, dropout, dropatt=dropatt, causal=causal
                )
            )

        self.dense = LinearNorm(in_dim, out_dim)

    def forward(self, dec_inp, in_lens):
        # B, C, T --> B, T, C
        inp = dec_inp.transpose(1, 2)
        mask = get_mask_from_lengths(in_lens)
        mask_3d = rearrange(mask, 'B T -> B T 1')

        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask_3d

        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        out = self.dense(out).transpose(1, 2)
        return out


class FFTransformerFlat(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout=0.1,
        dropatt=0.1,
        causal=False,
    ):
        super(FFTransformerFlat, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.layer_norm = torch.nn.LayerNorm(self.d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(
                n_head,
                d_model,
                d_head,
                d_inner,
                kernel_size,
                dropout,
                dropatt=dropatt,
                pre_lnorm=True,
                condition_dim=None,
                condition_types=None,
                causal=causal
            )
            for _ in range(n_layer)
        ])

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T',), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, inputs, mask):
        out = inputs
        for layer in self.layers:
            out = layer(out, mask=mask)

        out = self.layer_norm(out)
        out = out * rearrange(mask, 'B T -> B T 1')

        return out


class TransformerCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        n_head_self,
        n_head_cross,
        d_model,
        d_encoded,
        d_head,
        d_inner,
        dropout,
        dropout_self_att,
        dropout_cross_att,
        kernel_size,
        causal,
        pre_lnorm
    ):
        super(TransformerCrossAttentionLayer, self).__init__()

        self.self_attn = MultiHeadAttn(
            n_head=n_head_self,
            d_model=d_model,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropout_self_att,
            causal=causal,
            pre_lnorm=pre_lnorm,
        )
        self.cross_attn = CrossAttn(
            n_head=n_head_cross,
            d_model=d_model,
            d_encoded=d_encoded,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropout_cross_att,
            pre_lnorm=pre_lnorm,
        )

        if kernel_size == 1:
            self.pos_ff = LinearFF(d_model, d_inner, dropout)
        elif causal:
            self.pos_ff = CausalPositionwiseConvFF(
                d_model,
                d_inner,
                kernel_size,
                dropout,
            )
        else:
            self.pos_ff = PositionwiseConvFF(
                d_model,
                d_inner,
                kernel_size,
                dropout,
                pre_lnorm=pre_lnorm,
            )

    def forward(self, inputs, encoded, mask, attn_mask):
        mask_3d = rearrange(mask, 'B T -> B T 1')
        output = self.self_attn(inp=inputs, mask=mask)
        output = self.cross_attn(inputs=output, encoded=encoded, mask=mask, attn_mask=attn_mask)
        output = self.pos_ff(output)
        output = output * mask_3d
        return output


class ContextTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout=0.1,
        dropatt=0.1,
    ):
        super(ContextTransformer, self).__init__()
        self.d_model = d_model

        self.pos_emb = PositionalEmbedding(self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                n_head,
                d_model,
                d_head,
                d_inner,
                kernel_size,
                dropout,
                dropatt=dropatt,
                pre_lnorm=True,
                condition_dim=None,
                condition_types=None,
            )
            for _ in range(n_layer)
        ])


    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, inputs, mask):
        mask_3d = rearrange(mask, 'B T -> B T 1')

        pos_seq = torch.arange(inputs.size(1), device=inputs.device).to(inputs.dtype)
        pos_emb = self.pos_emb(pos_seq)

        out = inputs + pos_emb
        out = out * mask_3d

        for layer in self.transformer_layers:
            out = layer(out, mask=mask)

        out = self.layer_norm(out)
        out = out * mask_3d

        return out


class TextTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head,
        n_embed,
        d_model,
        d_context,
        d_head,
        d_inner,
        kernel_size,
        padding_idx,
        causal=False,
        dropout=0.1,
        dropout_att=0.1,
    ):
        super(TextTransformer, self).__init__()
        self.d_model = d_model

        self.word_emb = nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.context_cond_layer = torch.nn.Linear(d_context, self.d_model)
        self.text_layers = nn.ModuleList([
            TransformerLayer(
                n_head,
                d_model,
                d_head,
                d_inner,
                kernel_size,
                dropout,
                dropatt=dropout_att,
                condition_dim=None,
                condition_types=None,
                causal=causal,
                pre_lnorm=True,
            )
            for _ in range(n_layer)
        ])

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation())
        }

    @typecheck()
    def forward(self, text, text_mask, context_emb):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')
        text_emb = self.word_emb(text)

        pos_seq = torch.arange(text_emb.size(1), device=text_emb.device).to(text_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)

        out = text_emb + pos_emb
        out = out * text_mask_3d

        for layer in self.text_layers:
            out = layer(out, mask=text_mask)
        out = self.layer_norm(out)

        context_emb = rearrange(context_emb, 'B D -> B 1 D')
        context_res = self.context_cond_layer(context_emb)
        out = out + context_res
        out = out * text_mask_3d

        return out


class AudioTransformer(NeuralModule):
    def __init__(
        self,
        n_layer,
        n_head_self,
        n_head_context,
        d_model,
        d_context,
        d_head,
        d_inner,
        kernel_size=None,
        causal=False,
        dropout=0.1,
        dropout_self_att=0.1,
        dropout_context_att=0.1,
        pre_lnorm=False,
    ):
        super(AudioTransformer, self).__init__()
        self.d_model = d_model
        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                n_head_self=n_head_self,
                n_head_cross=n_head_context,
                d_model=d_model,
                d_encoded=d_context,
                d_head=d_head,
                d_inner=d_inner,
                kernel_size=kernel_size,
                dropout=dropout,
                dropout_self_att=dropout_self_att,
                dropout_cross_att=dropout_context_att,
                causal=causal,
                pre_lnorm=pre_lnorm,
            )
            for _ in range(n_layer)
        ])

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_input'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),

        }

    def forward(self, inputs, audio_mask, context, context_mask):
        audio_mask_3d = rearrange(audio_mask, 'B T_audio -> B T_audio 1')

        max_context_len = context.shape[1]
        # [B, T_text, T_context]
        context_attn_mask = audio_mask_3d.repeat([1, 1, max_context_len])
        context_attn_mask = context_attn_mask * rearrange(context_mask, 'B T_context -> B 1 T_context')

        out = inputs
        for layer in self.transformer_layers:
            out = layer(
                inputs=out,
                mask=audio_mask,
                encoded=context,
                attn_mask=context_attn_mask
            )

        out = self.layer_norm(out)
        out = out * audio_mask_3d

        return out