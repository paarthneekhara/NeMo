# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import io
import json
import os
import random
from itertools import filterfalse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
# from encodec import EncodecModel
from tqdm.auto import tqdm

from nemo.collections.asr.data.audio_to_text import _TarredInstructionTuningDataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import T5Sentinel
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import beta_binomial_prior_distribution, general_padding, get_base_dir
from nemo.utils import logging

from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate

__all__ = ['T5SpeechLMTarredDataset', 'GPTSpeechLMTarredDataset']

@dataclass
class G2PConfig:
    _target_: str = "nemo.collections.tts.g2p.models.en_us_arpabet.EnglishG2p"
    phoneme_dict: str = "scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
    heteronyms: str = "scripts/tts_dataset_files/heteronyms-052722"
    phoneme_probability: float = 0.5

@dataclass
class TextTokenizer:
    _target_: str = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.EnglishPhonemesTokenizer"
    punct: bool = True
    stresses: bool = True
    chars: bool = True
    apostrophe: bool = True
    pad_with_space: bool = True
    add_blank_at: bool = True
    g2p: G2PConfig = G2PConfig()

@dataclass
class TextTokenizerConfig:
    text_tokenizer: TextTokenizer = TextTokenizer()

def _get_default_text_tokenizer_conf():
    text_tokenizer: TextTokenizerConfig = TextTokenizerConfig()
    return OmegaConf.create(OmegaConf.to_yaml(text_tokenizer))

def pad_text_to_speech_dims(text_tensor, pad_id):
    token_len = text_tensor.shape[0]
    empty_padding = torch.ones((7, token_len), dtype=text_tensor.dtype, device=text_tensor.device) * pad_id
    return torch.cat((text_tensor.unsqueeze(0), empty_padding), dim=0)

tokenizer_config = _get_default_text_tokenizer_conf()
phoneme_tokenizer = instantiate(tokenizer_config).text_tokenizer

class T5SpeechLMTarredDataset(_TarredInstructionTuningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained T5 SpeechLM models.
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        sample_rate: int,
        shuffle_n: int = 0,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        decoder_starts_with_pad: bool = False,
        add_eos_to_decoder_output: bool = True,
        add_sentinel_to_input: bool = True,
        ul2_prompt_token: str = None,
        segment_max_duration: Optional[int] = None,
        trim: bool = False,
        trim_ref: Optional[float] = None,
        trim_top_db: Optional[int] = None,
        trim_frame_length: Optional[int] = None,
        trim_hop_length: Optional[int] = None,
        pad_multiple: int = 1,
        pitch_augment: bool = False,
        speech_offset: Optional[int] = None,
        train_task: Optional[str] = None,
        seq_pattern: Optional[str] = "parallel",
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
        decoder_only_model: bool = False,
        use_phoneme_tokenizer: Optional[bool] = False,
        lm_vocab_size: Optional[int] = None,
        use_attention_prior: Optional[bool] = False,
        attention_prior_scaling_factor: Optional[float] = 1.0,
        cross_attention_epsilon: Optional[float] = 0.0,
        num_speech_codebooks: Optional[int] = 8,
        **kwargs,
    ):
        """
        Only speech parameters are explained here.
        segment_max_duration: Optional[int] = None, - Speech max segment duration
        trim: bool = False, - speech parameter
        trim_ref: Optional[float] = None, - speech parameter
        trim_top_db: Optional[int] = None, - speech parameter
        trim_frame_length: Optional[int] = None, - speech parameter
        trim_hop_length: Optional[int] = None, - speech parameter
        pad_multiple: int = 1, - speech parameter
        pitch_augment: bool = False, - speech parameter
        speech_offset: Optional[int] = None, - if speech tokens then add this offset to the token indices to distinguish between text and speech tokens.
        **kwargs,
        """
        # These two variables need to be set before calling super().__init__() because the parent class calls `load_data()` which requires these attributes.
        self.decoder_starts_with_pad = decoder_starts_with_pad
        self.add_eos_to_decoder_output = add_eos_to_decoder_output
        self.add_sentinel_to_input = add_sentinel_to_input
        self.ul2_prompt_token = ul2_prompt_token
        # Speech related variables
        # self.encodec_model = EncodecModel.encodec_model_24khz()
        # self.encodec_model.set_target_bandwidth(6.0)
        self.base_data_dir = None
        self.segment_max_duration = segment_max_duration
        self.sample_rate = sample_rate
        # self.featurizer = WaveformFeaturizer(sample_rate=self.sample_rate)
        self.pad_multiple = pad_multiple
        self.pitch_augment = pitch_augment
        self.trim = trim
        self.trim_ref = trim_ref if trim_ref is not None else np.max
        self.trim_top_db = trim_top_db if trim_top_db is not None else 60
        self.trim_frame_length = trim_frame_length if trim_frame_length is not None else 2048
        self.trim_hop_length = trim_hop_length if trim_hop_length is not None else 512
        self.speech_offset = speech_offset if speech_offset is not None else 3
        self.seq_pattern = seq_pattern
        self.min_duration = kwargs.get('min_duration', 0.1)
        self.max_duration = kwargs.get('max_duration', 20)
        self.use_attention_prior = use_attention_prior
        self.attention_prior_scaling_factor = attention_prior_scaling_factor
        self.cross_attention_epsilon = cross_attention_epsilon  # value of prior for context tokens (b/w 0 and 1)
        assert self.cross_attention_epsilon >= 0.0 and self.cross_attention_epsilon <= 1.0

        self.train_task = train_task

        # Initialized super part
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.use_phoneme_tokenizer = use_phoneme_tokenizer
        self.examples = []
        self.lm_vocab_size = tokenizer.vocab_size if lm_vocab_size is None else lm_vocab_size
        self.num_speech_codebooks = num_speech_codebooks

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        self.context_length = kwargs.pop('context_length', None) #only used in gpt dataset atm

        logging.info("Loading and tokenizing dataset ... ")

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            sample_rate=sample_rate,
            shuffle_n=shuffle_n,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            max_seq_length=max_seq_length,
            shard_strategy=shard_strategy,
            shard_manifests=shard_manifests,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
            decoder_only_model=decoder_only_model,
            use_phoneme_tokenizer=use_phoneme_tokenizer,
        )

        self.encodec, self.ref_encodec = None, None

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def pad_taskname_ids(self, taskname_ids):
        # Pad taskname_ids to be the same length for the prompt encoder
        if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
            max_taskname_length = max(len(ids) for ids in taskname_ids)
            taskname_ids = [ids + [self.pad_token_id] * (max_taskname_length - len(ids)) for ids in taskname_ids]
            taskname_ids = torch.tensor(taskname_ids)

        # Task ids are just used for a look up embeddings for prompt-table
        elif self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT:
            taskname_ids = torch.tensor(taskname_ids)

        return taskname_ids

    def _build_sample(self, tup):
        audio_filename, self.encodec, self.ref_encodec, offset_id = tup

        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]
        doc = {}
        doc['context'] = manifest_entry.context
        doc['context_type'] = manifest_entry.context_type
        doc['context_duration'] = manifest_entry.context_duration
        doc['answer'] = manifest_entry.answer
        doc['answer_type'] = manifest_entry.answer_type
        doc['answer_duration'] = manifest_entry.answer_duration
        doc['question'] = manifest_entry.question
        doc['question_type'] = manifest_entry.question_type

        taskname = "squad"
        prompt_template = self.task_templates[taskname]["prompt_template"]
        prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
        total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        truncation_field = self.task_templates[taskname]['truncate_field']
        answer_field = self.task_templates[taskname]["answer_field"]

        input_example = prompt_template

        question_in_manifest = manifest_entry.question

        # Format the input example according to the template
        # Get context, question and answer codes in a dict.
        input_dict = self._insert_data_in_template(input_example, prompt_template_fields, doc, answer_field)
        context_tokens = input_dict['context']
        question_tokens = input_dict['question']

        # Logic to prune context
        # In case of TTS task, the entire reference speech is not required, so we randomly select a portion
        # of the reference audio.
        # In case of Next token prediction, We want context[:T] to go in the encoder and context[T+1:] to be
        # predicted by the decoder.
        start_token_index = 0
        end_token_index = -1
        if "Text to speech this" in question_in_manifest:
            total_context_len = context_tokens[0].size()[1]
            reduced_len = min(
                400,
                int(total_context_len * 0.2)
                if total_context_len > 600
                else int(total_context_len * random.uniform(0.2, 0.5)),
            )
            start_token_index = random.randint(
                0, total_context_len - reduced_len
            )  # start index can be greater than 440
            context_tokens[0] = context_tokens[0][
                :, start_token_index : min(start_token_index + 440, start_token_index + reduced_len)
            ]
        elif "Next token prediction" in question_in_manifest:
            total_context_len = context_tokens[0].size()[1]
            end_token_index = int(total_context_len * random.uniform(0.01, 0.2))
            context_tokens[0] = context_tokens[0][:, :end_token_index]

        # Get virtual tokens
        virtual_tokens = self._insert_virtual_token_placeholders(input_example.split(' ')[0], virtual_token_splits)

        # a trick to align with the data format in t5 pretraining
        # new
        virtual_tokens = self.tokenizer.text_to_ids(virtual_tokens)
        if self.add_sentinel_to_input:
            question_tokens = question_tokens + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

        # Add BOS/EOS to the input of encoder if desired, adds EOS by default
        if self.ul2_prompt_token is not None:
            ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
            assert len(ul2_prompt_token_id) == 1
            context_tokens = ul2_prompt_token_id + context_tokens
        if self.add_bos:
            context_tokens = [self.tokenizer.bos_id] + context_tokens
        if self.add_eos:
            question_tokens = question_tokens + [self.tokenizer.eos_id]

        # Try to truncate input text to fit into the max sequence length
        if self._get_len(context_tokens, question_tokens, virtual_tokens) > self.max_seq_length:
            context_tokens, question_tokens, virtual_tokens = self._truncate_input_speech(
                context_tokens, question_tokens, virtual_tokens
            )

        virtual_tokens, virtual_tokens_len = self.list_to_tensor(virtual_tokens)
        context_tokens, context_tokens_len = self.list_to_tensor(context_tokens)
        question_tokens, question_tokens_len = self.list_to_tensor(question_tokens)

        if doc["question_type"] != "SPEECH" and doc["context_type"] == "SPEECH":
            question_tokens = pad_text_to_speech_dims(question_tokens, self.tokenizer.pad_id)
        if doc["context_type"] != "SPEECH" and doc["question_type"] == "SPEECH":
            context_tokens = pad_text_to_speech_dims(context_tokens, self.tokenizer.pad_id)
        context_tokens = context_tokens.to(question_tokens.device)
        context_and_question_tokens = torch.cat([context_tokens, question_tokens], dim=1)

        # get answer ids
        if answer_field in doc.keys():  # training and validation
            answer_ids = self._get_tokens(doc, answer_field, doc[answer_field])
            if end_token_index > -1:
                answer_ids[0] = answer_ids[0][:, end_token_index:]

            if self.decoder_starts_with_pad:
                answer_text_ids = [self.tokenizer.pad_id]
            else:
                answer_text_ids = [self.tokenizer.bos_id]
            # a trick to align with the data format in t5 pretraining
            # if self.add_sentinel_to_input:
            #     answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            answer_text_ids += answer_ids

            if self.add_eos_to_decoder_output:
                answer_text_ids += [self.tokenizer.eos_id]
            else:
                answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

        # Skip example if the final length doesn't fit length requirements even after truncation
        if (
            self.min_seq_length
            <= self._get_element_len(context_and_question_tokens) + self._get_element_len(virtual_tokens)
            <= self.max_seq_length
            and self.min_seq_length <= self._get_element_len(answer_text_ids) <= self.max_seq_length
        ):
            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                taskname_id = self.tokenizer.text_to_ids(taskname)
            elif (
                self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT
            ):  # TODO (@adithyare) this class and GPTPromptLearningDataset should be merged.
                taskname_id = -1
            else:
                raise ValueError("Invalid virtual prompt source specified")

            dec_input = None
            dec_labels = None

            if answer_field in doc.keys():  # training and validation
                dec_input = answer_text_ids[:-1]
                dec_labels = answer_text_ids[1:]

            dec_input, dec_input_len = self.list_to_tensor(dec_input, True)
            dec_labels, dec_labels_len = self.list_to_tensor(dec_labels, True)
            is_speech = True if doc["answer_type"] == "SPEECH" else False
            if is_speech:
                assert dec_input.dim() == 2 and dec_labels.dim() == 2
                if self.seq_pattern == "delay_parallel":
                    num_codebooks = dec_input.shape[0]
                    dec_input_padded = torch.cat(
                        [
                            torch.zeros_like(dec_input[:, 0:num_codebooks]),
                            dec_input,
                            torch.zeros_like(dec_input[:, 0:num_codebooks]),
                        ],
                        dim=1,
                    )
                    dec_labels_padded = torch.cat(
                        [
                            torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                            dec_labels,
                            torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                        ],
                        dim=1,
                    )
                    dec_input_new = []
                    dec_labels_new = []
                    for _c in range(self.num_speech_codebooks):
                        st = num_codebooks - _c
                        et_decoder_input = dec_input_padded.shape[1] - _c
                        et_decoder_labels = dec_labels_padded.shape[1] - _c
                        dec_input_new.append(dec_input_padded[_c, st:et_decoder_input])
                        dec_labels_new.append(dec_labels_padded[_c, st:et_decoder_labels])
                    dec_input = torch.stack(dec_input_new, dim=0)
                    dec_labels = torch.stack(dec_labels_new, dim=0)
                    dec_input_len = torch.tensor(dec_input.shape[1]).long()
                    dec_labels_len = torch.tensor(dec_labels.shape[1]).long()

            enc_len = context_tokens_len + question_tokens_len + virtual_tokens_len
            # TODO: Remove hardcoding
            num_question_offset = 4  # For "Text to Speech this"

            cross_attention_prior = torch.zeros(dec_labels_len, enc_len) + self.cross_attention_epsilon
            if self.use_attention_prior:
                cross_attention_question_prior = torch.from_numpy(
                    beta_binomial_prior_distribution(
                        question_tokens_len.item() - num_question_offset,
                        dec_labels_len.item(),
                        scaling_factor=self.attention_prior_scaling_factor,
                    )
                )
                cross_attention_prior[
                    :, virtual_tokens_len + context_tokens_len + num_question_offset :
                ] = cross_attention_question_prior

            return (
                taskname_id,
                virtual_tokens,
                virtual_tokens_len,
                context_and_question_tokens,
                context_tokens_len + question_tokens_len,
                dec_input,
                dec_input_len,
                dec_labels,
                dec_labels_len,
                is_speech,
                cross_attention_prior,
            )

    def _truncate_input_speech(self, context_tokens, question_tokens, virtual_tokens):
        total_len = self._get_len(context_tokens, question_tokens, virtual_tokens)
        context_len = self._get_element_len(context_tokens)
        truncation_length = total_len - self.max_seq_length + 1
        context_tokens[0] = context_tokens[0][:, min(truncation_length, context_len) :]
        return context_tokens, question_tokens, virtual_tokens

    def list_to_tensor(self, element, fill=False):
        """
        Convert list to tensor. The list might contain integers, 2D-tensors (speech tokens) and combination of two.
        If all of them are ints, simply convert to tensor
        If combination of 2D-tensor and ints. Convert int to the dimension of the tensor.
        example: [2, 4, 5] -> torch.tensor([2, 4, 5])
        example: [2, torch.tensor([[4, 5, 6], [6, 7, 8]])] -> torch.tensor( [[-1, 4, 5, 6], [2, 6, 7, 8]] )
        """
        ret, ln = None, None
        if element is None:
            return ret, ln

        max_len = max([1 if isinstance(item, int) else len(item) for item in element])
        if max_len == 1:
            ret = torch.as_tensor(element).long()
            ln = torch.tensor(ret.size()[0]).long()
        else:
            ret = []
            for e in element:
                if isinstance(e, int):
                    tmp = torch.full((8, 1), e if fill else -1)
                    tmp[7] = e
                else:
                    tmp = e
                ret.append(tmp)
            ret = torch.cat(ret, dim=1)
            ln = torch.tensor(ret.size()[1]).long()
        return ret, ln

    def _get_text_tokens(self, text):
        input_ids = self.tokenizer.text_to_ids(text)
        return input_ids

    def _get_phoneme_tokens(self, text):
        input_ids = phoneme_tokenizer.encode(text)
        input_ids_adjusted = [_id + self.lm_vocab_size for _id in input_ids]
        return input_ids_adjusted

    def _pad_wav_to_multiple(self, wav):
        if self.pad_multiple > 1:
            if wav.shape[0] % self.pad_multiple != 0:
                wav = torch.cat(
                    [wav, torch.zeros(self.pad_multiple - wav.shape[0] % self.pad_multiple, dtype=torch.float)]
                )
        return wav

    def _get_element_len(self, element):
        length = 0
        if isinstance(element, list):
            for e in element:
                if isinstance(e, int):
                    length += 1
                else:
                    if e.dim() > 1:
                        length += e.size()[1]
                    else:
                        length += e.size()[0]
        else:
            if element.dim() > 1:
                length += element.size()[1]
            else:
                length += element.size()[0]
        return length

    def _get_len(self, context_tokens, question_tokens, virtual_tokens):
        length = 0
        length += self._get_element_len(context_tokens)
        length += self._get_element_len(question_tokens)
        length += self._get_element_len(virtual_tokens)
        return length

    def _get_speech_tokens(self, field):

        # Convert to codes
        codec_codes, codec_codes_length = None, None  # Codes

        if self.train_task == 'tts':
            if field == 'context':
                self.ref_encodec = torch.load(io.BytesIO(self.ref_encodec), map_location="cpu").long()
                codec_codes = self.ref_encodec
            elif field == 'answer':
                self.encodec = torch.load(io.BytesIO(self.encodec), map_location="cpu").long()
                codec_codes = self.encodec
        elif self.train_task == 'asr':
            if field == 'context':
                self.ref_encodec = torch.load(io.BytesIO(self.ref_encodec), map_location="cpu").long()
                codec_codes = self.ref_encodec

        # codec_codes_length = torch.tensor(codec_codes.shape[1]).long()

        # Convert codes to codes corresponding to megatron embedding layer
        codec_codes[0] = (codec_codes[0] + self.speech_offset).long()

        return codec_codes

    def _get_tokens(self, doc, field, field_data):
        if f"{field}_type" not in doc.keys():
            field_tokens = self._get_text_tokens(field_data.strip(" "))  # list of ids
        elif doc[f"{field}_type"] == 'TEXT':
            _text = field_data.strip(" ")
            if self.use_phoneme_tokenizer:
                instruction_tokens = self._get_text_tokens("Phoneme TTS")
                field_tokens = self._get_phoneme_tokens(_text.replace("Text to speech this ", ""))
                field_tokens = instruction_tokens + field_tokens
            else:
                field_tokens = self._get_text_tokens(_text)  # list of ids
        elif doc[f"{field}_type"] == 'SPEECH':
            dur = -1
            if f"{field}_duration" in doc:
                dur = doc[f"{field}_duration"]
            field_tokens = self._get_speech_tokens(field)  # list of ids
            if not isinstance(field_tokens, list):
                field_tokens = [field_tokens]
        elif doc[f"{field}_type"] == 'TOKENS':
            # Do nothing; already tokenized
            field_tokens = field_data
        else:
            raise Exception(f"{field}_type not recognized")
        return field_tokens

    def _insert_data_in_template(self, input_example, prompt_template_fields, doc, answer_field):
        """ Format the input example according to the template """
        out_dict = {}
        for field in prompt_template_fields:
            # discard the last one, {label} / {answer}
            # Or if some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            if field == answer_field or field not in doc.keys():
                continue
                #  out_dict[field] = ""

            elif field in doc.keys():
                field_data = doc[field]
                if f"{field}_type" not in doc.keys():
                    doc[f"{field}_type"] = "TEXT"
                    raise Exception(f"{field}_type does not exist in doc")
                else:
                    out_dict[field] = self._get_tokens(doc, field, field_data)
        return out_dict

    def get_position_ids(self, virtual_token, context_and_qquestion):
        enc_input = []
        enc_input.append(virtual_token)
        if context_and_qquestion.dim() > 2:
            enc_input.append(context_and_qquestion[:, 0, :])
        else:
            enc_input.append(context_and_qquestion)

        enc_input = torch.cat(enc_input, dim=1)

        return build_position_ids(enc_input).contiguous()

    def collate_fn(self, batch):
        """ Prepares enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids for global batch """

        data_dict = self.pad_batch_and_build_loss_mask(batch)

        position_ids = self.get_position_ids(data_dict['virtual_tokens'], data_dict['context_and_question_tokens'])

        return (
            data_dict['virtual_tokens'],
            data_dict['context_and_question_tokens'],
            data_dict['enc_mask'],
            data_dict['dec_input'],
            data_dict['dec_input_mask'],
            data_dict['dec_labels'],
            data_dict['dec_labels_mask'],
            position_ids,
            data_dict['taskname_id'],
            data_dict['speech_mask'],
            data_dict['context_and_question_tokens_lens'],
            data_dict['cross_attention_prior'],
        )

    def pad_batch_and_build_loss_mask(self, batch):
        """ Pad enc_input, dec_input, labels in batch to max batch length while building loss_mask, enc_mask, and dec_mask """
        (
            taskname_ids,
            _,
            virtual_tokens_len,
            _,
            context_and_question_tokens_len,
            _,
            dec_input_len,
            _,
            dec_labels_len,
            _,
            _,
        ) = zip(*batch)

        taskname_ids = self.pad_taskname_ids(taskname_ids)

        max_virtual_tokens_len = max(virtual_tokens_len).item() if virtual_tokens_len is not None else 0
        if isinstance(virtual_tokens_len, tuple):
            virtual_tokens_len = torch.stack(virtual_tokens_len)
        virtual_mask = get_mask_from_lengths(virtual_tokens_len)

        max_context_and_question_tokens_len = (
            max(context_and_question_tokens_len).item() if context_and_question_tokens_len is not None else 0
        )
        if isinstance(context_and_question_tokens_len, tuple):
            context_and_question_tokens_len = torch.stack(context_and_question_tokens_len)
        context_and_question_mask = get_mask_from_lengths(context_and_question_tokens_len)

        max_dec_input_len = max(dec_input_len).item() if dec_input_len is not None else 0
        max_dec_labels_len = max(dec_labels_len).item() if dec_labels_len is not None else 0
        enc_mask = torch.cat([virtual_mask, context_and_question_mask], dim=1)

        (
            virtual_tokens_list,
            context_question_tokens_list,
            dec_input_list,
            dec_input_mask_list,
            dec_labels_list,
            dec_labels_mask_list,
            speech_mask_list,
            cross_attention_prior_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i, sample_tuple in enumerate(batch):
            (
                _,
                virtual_token,
                virtual_token_len,
                context_and_question_token,
                context_and_question_token_len,
                dec_input,
                dec_input_len,
                dec_label,
                dec_label_len,
                is_speech,
                cross_attention_prior,
            ) = sample_tuple

            virtual_tokens_list.append(
                general_padding(
                    virtual_token, virtual_token_len.item(), max_virtual_tokens_len, pad_value=self.tokenizer.pad_id
                )
            )

            context_tokens_padded = general_padding(
                context_and_question_token,
                context_and_question_token_len.item(),
                max_context_and_question_tokens_len,
                pad_value=self.tokenizer.pad_id,
            )
            if len(context_tokens_padded.shape) < 2:
                context_tokens_padded = pad_text_to_speech_dims(context_tokens_padded, self.tokenizer.pad_id)
            context_question_tokens_list.append(context_tokens_padded)

            if max_dec_input_len > 0:
                dec_input_padded = general_padding(
                    dec_input, dec_input_len.item(), max_dec_input_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_input_padded.shape) < 2:
                    dec_input_padded = pad_text_to_speech_dims(dec_input_padded, self.tokenizer.pad_id)
                dec_input_list.append(dec_input_padded)
                dec_mask = (
                    torch.as_tensor(([1] * dec_input_len) + ([0] * (max_dec_input_len - dec_input_len)))
                    .long()
                    .contiguous()
                )
                dec_input_mask_list.append(dec_mask)
                speech_mask = dec_mask if is_speech else torch.zeros(dec_mask.shape)
                speech_mask_list.append(speech_mask)

            if max_dec_labels_len > 0:
                loss_mask = (
                    torch.as_tensor(([1] * dec_label_len) + ([0] * (max_dec_labels_len - dec_label_len)))
                    .long()
                    .contiguous()
                )
                dec_label_padded = general_padding(
                    dec_label, dec_label_len.item(), max_dec_labels_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_label_padded.shape) < 2:
                    dec_label_padded = pad_text_to_speech_dims(dec_label_padded, self.tokenizer.pad_id)
                dec_labels_list.append(dec_label_padded)
                dec_labels_mask_list.append(loss_mask)

                _p0 = max_dec_labels_len - dec_label_len
                _p1 = (
                    max_virtual_tokens_len
                    + max_context_and_question_tokens_len
                    - context_and_question_token_len
                    - virtual_token_len
                )

                cross_attention_prior_padded = torch.nn.functional.pad(
                    cross_attention_prior, pad=(0, _p1, 0, _p0), mode="constant", value=1,
                )
                cross_attention_prior_list.append(cross_attention_prior_padded)

        data_dict = {
            "taskname_id": taskname_ids,
            "virtual_tokens": torch.stack(virtual_tokens_list),
            "context_and_question_tokens": torch.stack(context_question_tokens_list),
            "enc_mask": enc_mask,
            "dec_input": torch.stack(dec_input_list) if len(dec_input_list) > 0 else None,
            "dec_input_mask": torch.stack(dec_input_mask_list) if len(dec_input_mask_list) > 0 else None,
            "dec_labels": torch.stack(dec_labels_list) if len(dec_labels_list) > 0 else None,
            "dec_labels_mask": torch.stack(dec_labels_mask_list) if len(dec_labels_mask_list) > 0 else None,
            "speech_mask": torch.stack(speech_mask_list) if len(speech_mask_list) > 0 else None,
            "context_and_question_tokens_lens": context_and_question_tokens_len,
            "cross_attention_prior": torch.stack(cross_attention_prior_list)
            if len(cross_attention_prior_list) > 0
            else None,
        }

        return data_dict

class GPTSpeechLMTarredDataset(T5SpeechLMTarredDataset):
    """ No support for cross attention here yet"""
    def _build_sample(self, tup):
        audio_filename, self.encodec, self.ref_encodec, offset_id = tup

        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]
        doc = {}
        doc['context'] = manifest_entry.context
        doc['context_type'] = manifest_entry.context_type
        doc['context_duration'] = manifest_entry.context_duration
        doc['answer'] = manifest_entry.answer
        doc['answer_type'] = manifest_entry.answer_type
        doc['answer_duration'] = manifest_entry.answer_duration
        doc['question'] = manifest_entry.question
        doc['question_type'] = manifest_entry.question_type


        taskname = "squad"
        prompt_template = self.task_templates[taskname]["prompt_template"]
        prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
        # total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        # truncation_field = self.task_templates[taskname]['truncate_field']
        answer_field = self.task_templates[taskname]["answer_field"]

        input_example = prompt_template

        # question_in_manifest = manifest_entry.question

        # Format the input example according to the template
        # Get context, question and answer codes in a dict.
        input_dict = self._insert_data_in_template(input_example, prompt_template_fields, doc, answer_field)
        context_tokens = input_dict['context']
        question_tokens = input_dict['question']

        # Logic to prune context
        # In case of TTS task, the entire reference speech is not required, so we randomly select a portion
        # of the reference audio.
        # In case of Next token prediction, We want context[:T] to go in the encoder and context[T+1:] to be
        # predicted by the decoder.
        start_token_index = 0
        end_token_index = -1

        total_context_len = context_tokens[0].size()[1]
        context_3s = 3*75
        if total_context_len > context_3s:
            start_token_index = random.randint(0, total_context_len - context_3s)
            # logging.debug(f"start_token_index: {start_token_index}")
        end_token_index = start_token_index + min(context_3s, total_context_len)
        # logging.debug(f"end_token_index: {end_token_index}")
        context_tokens[0] = context_tokens[0][
            :, start_token_index : end_token_index
        ]
        # if self.context_length is not None:
        #     start_token_index = random.randint(0, total_context_len - math.ceil(self.context_length * 75))
        #     context_tokens[0] = context_tokens[0][
        #         :, start_token_index : start_token_index + math.floor(self.context_length * 75)
        #     ]
        # else:
        #     reduced_len = min(
        #         400,
        #         int(total_context_len * 0.2)
        #         if total_context_len > 600
        #         else int(total_context_len * random.uniform(0.2, 0.5)),
        #     )
        #     start_token_index = random.randint(
        #         0, total_context_len - reduced_len
        #     )  # start index can be greater than 440
        #     context_tokens[0] = context_tokens[0][
        #         :, start_token_index : min(start_token_index + 440, start_token_index + reduced_len)
        #     ]
        # if "Text to speech this" in question_in_manifest:
        # elif "Next token prediction" in question_in_manifest:
        #     total_context_len = context_tokens[0].size()[1]
        #     end_token_index = int(total_context_len * random.uniform(0.01, 0.2))
        #     context_tokens[0] = context_tokens[0][:, :end_token_index]

        # Get virtual tokens
        virtual_tokens = self._insert_virtual_token_placeholders(input_example.split(' ')[0], virtual_token_splits)

        # a trick to align with the data format in t5 pretraining
        # new
        virtual_tokens = self.tokenizer.text_to_ids(virtual_tokens)
        if self.add_sentinel_to_input:
            question_tokens = question_tokens + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

        # Add BOS/EOS to the input of encoder if desired, adds EOS by default
        if self.ul2_prompt_token is not None:
            ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
            assert len(ul2_prompt_token_id) == 1
            context_tokens = ul2_prompt_token_id + context_tokens
        if self.add_bos:
            context_tokens = [self.tokenizer.bos_id] + context_tokens
        if self.add_eos:
            # question_tokens = question_tokens + [self.tokenizer.eos_id]
            question_tokens = [self.tokenizer.pad_id] + question_tokens + [self.tokenizer.pad_id]

        # # Try to truncate input text to fit into the max sequence length
        # if self._get_len(context_tokens, question_tokens, virtual_tokens) > self.max_seq_length:
        #     context_tokens, question_tokens, virtual_tokens = self._truncate_input_speech(
        #         context_tokens, question_tokens, virtual_tokens
        #     )

        virtual_tokens, virtual_tokens_len = self.list_to_tensor(virtual_tokens)
        context_tokens, context_tokens_len = self.list_to_tensor(context_tokens)
        question_tokens, question_tokens_len = self.list_to_tensor(question_tokens)

        if doc["question_type"] != "SPEECH" and doc["context_type"] == "SPEECH":
            question_tokens = pad_text_to_speech_dims(question_tokens, self.tokenizer.pad_id)
        if doc["context_type"] != "SPEECH" and doc["question_type"] == "SPEECH":
            context_tokens = pad_text_to_speech_dims(context_tokens, self.tokenizer.pad_id)
        context_and_question_tokens = torch.cat([context_tokens, question_tokens], dim=1)

        # get answer ids
        if answer_field in doc.keys():  # training and validation
            answer_ids = self._get_tokens(doc, answer_field, doc[answer_field])
            # if end_token_index > -1:
            #     answer_ids[0] = answer_ids[0][:, end_token_index:]

            # if self.decoder_starts_with_pad:
            #     answer_text_ids = [self.tokenizer.pad_id]
            # else:
            #     answer_text_ids = [self.tokenizer.bos_id]
            # # a trick to align with the data format in t5 pretraining
            # # if self.add_sentinel_to_input:
            # #     answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            answer_text_ids = answer_ids

            if self.add_eos_to_decoder_output:
                answer_text_ids += [self.tokenizer.eos_id]
            else:
                answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

        # Skip example if the final length doesn't fit length requirements even after truncation
        input_ids = answer_text_ids
        input_ids, input_ids_len = self.list_to_tensor(input_ids, True)
        input_len = self._get_element_len(context_and_question_tokens) + self._get_element_len(answer_text_ids) - 1
        # debug = False
        if input_len > self.max_seq_length:
            # debug = True
            # logging.info(f"Overflow. input_len:{input_len}. self.max_seq_length:{self.max_seq_length}. overflow_len:{self.max_seq_length - input_len}.")
            overflow_len = self.max_seq_length - input_len
            # truncate context if context after truncation is at least 1s
            # else truncate answer as final option
            if context_tokens_len - overflow_len > 75:
                # logging.info(f"Cutting context. context_tokens:{context_tokens.shape}. context_tokens_len:{context_tokens_len}.")
                context_tokens = context_tokens[:, :context_tokens_len - overflow_len]
                context_tokens_len = context_tokens_len - overflow_len
                # logging.info(f"Cut context. context_tokens:{context_tokens.shape}. context_tokens_len:{context_tokens_len}.")
            else:
                # logging.info(f"Cutting answer. input_ids:{input_ids.shape}. input_ids_len:{input_ids_len}.")
                input_ids = input_ids[:,:input_ids_len - overflow_len]
                input_ids_len = input_ids_len - overflow_len
                # logging.info(f"Cut answer. input_ids:{input_ids.shape}. input_ids_len:{input_ids_len}.")

        is_speech = True if doc["answer_type"] == "SPEECH" else False
        if is_speech:
            assert input_ids.dim() == 2
            if self.seq_pattern == "delay_parallel":
                num_codebooks = input_ids.shape[0]
                dec_input_padded = torch.cat(
                    [
                        torch.zeros_like(input_ids[:, 0:num_codebooks]),
                        input_ids,
                        torch.zeros_like(input_ids[:, 0:num_codebooks]),
                    ],
                    dim=1,
                )
                dec_input_new = []
                for _c in range(self.num_speech_codebooks):
                    st = num_codebooks - _c
                    et_decoder_input = dec_input_padded.shape[1] - _c
                    dec_input_new.append(dec_input_padded[_c, st:et_decoder_input])
                try:
                    input_ids = torch.stack(dec_input_new, dim=0)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                input_ids_len = torch.tensor(input_ids.shape[1]).long()


        # import ipdb
        # if debug:
        #     logging.info(
        #         f"Return from getitem: \ncontext_tokens:{context_tokens.shape}\ncontext_tokens_len:{context_tokens_len}\n"
        #         f"question_tokens:{question_tokens.shape}\nquestion_tokens_len:{question_tokens_len}\ninput_ids:{input_ids.shape}\ninput_ids_len:{input_ids_len}"
        #     )
        #     ipdb.set_trace()

        return (
            context_tokens,
            context_tokens_len,
            question_tokens,
            question_tokens_len,
            input_ids,
            input_ids_len,
        )

    def collate_fn(self, batch):
        (
            _,
            context_tokens_len,
            _,
            question_tokens_len,
            _,
            input_ids_len,
        ) = zip(*batch)

        # max_virtual_tokens_len = max(virtual_tokens_len).item() if virtual_tokens_len is not None else 0
        # if isinstance(virtual_tokens_len, tuple):
        #     virtual_tokens_len = torch.stack(virtual_tokens_len)
        # virtual_mask = get_mask_from_lengths(virtual_tokens_len)

        decoder_input_len = torch.stack(context_tokens_len) + torch.stack(question_tokens_len) + torch.stack(input_ids_len)
        max_decoder_input_len = (
            max(decoder_input_len).item() if decoder_input_len is not None else 0
        )
        # if isinstance(decoder_input_len, tuple):
        #     decoder_input_len = torch.stack(decoder_input_len)

        decoder_mask = get_mask_from_lengths(decoder_input_len-1)
        speech_mask = get_mask_from_lengths(decoder_input_len-1)
        context_question_mask = torch.ones(speech_mask.shape)
        (
            decoder_input_list,
            decoder_labels_list,
        ) = (
            [],
            [],
        )
        for i, sample_tuple in enumerate(batch):
            (
                context_tokens,
                context_tokens_len,
                question_tokens,
                question_tokens_len,
                input_ids,
                input_ids_len,
            ) = sample_tuple

            context_tokens_input = context_tokens.clone().contiguous().detach()
            for l in range(1, context_tokens_input.shape[0]):
                context_tokens_input[l] += self.speech_offset + 1024 * l  # TODO: fix hardcode
            input_ids_shifted = input_ids.clone().contiguous().detach()
            for l in range(1, input_ids_shifted.shape[0]):
                input_ids_shifted[l] += self.speech_offset + 1024 * l  # TODO: fix hardcode

            complete_input = torch.cat([context_tokens_input, question_tokens, input_ids_shifted], dim=1)
            complete_input_padded  = general_padding(
                complete_input,
                decoder_input_len[i].item(),
                max_decoder_input_len,
                pad_value=self.tokenizer.pad_id,
            )
            complete_output = torch.cat([context_tokens, question_tokens, input_ids], dim=1)
            complete_output_padded  = general_padding(
                complete_output,
                decoder_input_len[i].item(),
                max_decoder_input_len,
                pad_value=self.tokenizer.pad_id,
            )
            # if len(decoder_input_padded.shape) < 2:
            #     decoder_input_padded = pad_text_to_speech_dims(decoder_input_padded, self.tokenizer.pad_id)
            decoder_labels = complete_output_padded[:, 1:].contiguous()
            decoder_input = complete_input_padded[:, :-1].contiguous()

            decoder_input_list.append(decoder_input)
            decoder_labels_list.append(decoder_labels)

            decoder_mask[i, :context_tokens_len+question_tokens_len-1] = 0  # Mask out context and question
            speech_mask[i, context_tokens_len:context_tokens_len+question_tokens_len] = 0  # Mask out context and question
            context_question_mask[i, :context_tokens_len+question_tokens_len] = 0

        # Using causal attention mask for whole input
        batch_size = len(decoder_input_list)
        attention_mask = torch.tril(torch.ones((batch_size, max_decoder_input_len-1, max_decoder_input_len-1))).view(
            batch_size, 1, max_decoder_input_len-1, max_decoder_input_len-1
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5


        decoder_input = torch.stack(decoder_input_list)
        position_ids = build_position_ids(decoder_input)
        data_dict = {
            "tokens": decoder_input,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": torch.stack(decoder_labels_list),
            "speech_mask": speech_mask,  # For TTS, can just be loss_mask since answer will always be speech
            "loss_mask": decoder_mask,  # Mask out context and question and padding
            "attention_prior": None,
            "context_question_mask": context_question_mask,
        }

        return data_dict