# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

import collections
import copy
import os
import random
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelWithLMHead

from nemo.collections.nlp.data.dialogue import DialogueGPTClassificationDataset, DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.design_data_processor import DialogueDesignDataProcessor
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueClassificationMetrics
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging

__all__ = ['DialogueGPTClassificationModel']


class DialogueGPTClassificationModel(NLPModel):
    def __init__(
        self, cfg: DictConfig, trainer: Trainer = None,
    ):

        self.cfg = cfg
        self.eval_mode = cfg.dataset.eval_mode
        self.data_prepared = False
        self.epoch_number = 0
        super().__init__(cfg=cfg, trainer=trainer, no_lm_init=True)

        if self.cfg.library == "huggingface":
            self.language_model = AutoModelWithLMHead.from_pretrained(cfg.language_model.pretrained_model_name)
            self.language_model.resize_token_embeddings(len(self.tokenizer.tokenizer))
        elif self.cfg.library == "megatron":
            self.prompt_learning = self.cfg.prompt_learning
            if self.prompt_learning:
                # removing tokenizer cfg as this triggers tokenizer construction which is not helpful here as we have a separate tokenizer
                new_cfg = copy.copy(cfg)
                del new_cfg.tokenizer
                self.language_model = MegatronGPTPromptLearningModel(new_cfg, trainer)
            else:
                self.language_model = MegatronGPTModel.restore_from(cfg.language_model.lm_checkpoint, trainer=trainer)

        all_labels = list(
            self._train_dl.dataset.all_possible_labels.union(
                self._validation_dl.dataset.all_possible_labels, self._test_dl.dataset.all_possible_labels
            )
        )
        self.label_to_ids = collections.defaultdict(int)

        for i in range(len(all_labels)):
            self.label_to_ids[all_labels[i]] = i

        self.all_existing_labels = set(self.label_to_ids.keys())

        self.token_to_words = {}
        self.classification_report = ClassificationReport(
            num_classes=len(self.label_to_ids) + 1, mode='micro', label_ids=self.label_to_ids, dist_sync_on_step=True
        )

    def training_step(self, batch, batch_idx):

        (
            input_ids,
            attn_masks,
            labels,
            candidate_input_ids,
            candidate_attn_masks,
            template_length,
            utterance_length,
            correct_candidate,
        ) = batch
        # construct training samples as generating " Answer: yes/no" after "<utterance> <label_type>: <candidate_label>"
        if self.eval_mode == "binary_score":
            new_input_ids = []
            new_attn_masks = []
            for i in range(candidate_input_ids.size(0)):
                # in some datasets like assistant, there might be 60+ possible intents with 1 correct intent
                # therefore we might not want to use all possible intents as negative samples
                # instead use {binary_score_subsample_ratio} negative samples for every positive sample
                if self.cfg.dataset.binary_score_subsample:
                    new_input_ids.append(candidate_input_ids[i, 2 * correct_candidate[i].item(), :])
                    new_attn_masks.append(candidate_attn_masks[i, 2 * correct_candidate[i].item(), :])
                    possible_negatives = []
                    for j in range(0, candidate_input_ids.size(1), 2):
                        if j > 0 and torch.equal(candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]):
                            break
                        if j != 2 * correct_candidate[i].item():
                            possible_negatives.append(j)
                    negative_samples = random.choices(
                        possible_negatives, k=int(self.cfg.dataset.binary_score_subsample_ratio)
                    )
                    for negative_sample in negative_samples:
                        new_input_ids.append(candidate_input_ids[i, negative_sample, :])
                        new_attn_masks.append(candidate_attn_masks[i, negative_sample, :])

                else:
                    for j in range(0, candidate_input_ids.size(1), 2):
                        if j > 0 and torch.equal(candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]):
                            break
                        new_input_ids.append(candidate_input_ids[i, j, :])
                        new_attn_masks.append(candidate_attn_masks[i, j, :])
            input_ids = torch.stack(new_input_ids)
            attn_masks = torch.stack(new_attn_masks)
            labels = self.get_binary_score_labels(input_ids)

        loss = self(input_ids, attn_masks, labels, inference=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch)

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, mode='val')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, mode='test')

    def eval_epoch_end(self, outputs, mode='val'):

        generated_field = []
        ground_truth_field = []
        inputs = []
        for output in outputs:
            generated_field += output["generated_field"]
            ground_truth_field += output["ground_truth_field"]
            inputs += output["input"]

        with_slots = self.cfg.dataset.target_template == "with_slots"

        generated_labels, generated_slots = DialogueClassificationMetrics.split_label_and_slots(
            generated_field, with_slots=with_slots
        )
        ground_truth_labels, ground_truth_slots = DialogueClassificationMetrics.split_label_and_slots(
            ground_truth_field, with_slots=with_slots
        )

        os.makedirs(self.cfg.dataset.dialogues_example_dir, exist_ok=True)
        filename = os.path.join(
            self.cfg.dataset.dialogues_example_dir, f"{mode}_predictions_epoch{self.epoch_number}.jsonl"
        )

        DialogueClassificationMetrics.save_predictions(
            filename,
            generated_labels,
            generated_slots,
            ground_truth_labels,
            ground_truth_slots,
            generated_field,
            ground_truth_field,
            inputs,
        )

        label_acc = np.mean([int(generated_labels[i] == ground_truth_labels[i]) for i in range(len(generated_labels))])

        generated_field_ids = torch.tensor([self.label_to_ids[label] for label in generated_labels], dtype=int).to(
            self.classification_report.device
        )

        ground_truth_field_ids = torch.tensor(
            [self.label_to_ids[label] for label in ground_truth_labels], dtype=int
        ).to(self.classification_report.device)

        tp, fn, fp, _ = self.classification_report(generated_field_ids, ground_truth_field_ids)

        precision, recall, f1, report = self.classification_report.compute()
        self.classification_report.reset()

        (
            slot_precision,
            slot_recall,
            slot_f1,
            slot_joint_goal_accuracy,
        ) = DialogueClassificationMetrics.get_slot_filling_metrics(generated_slots, ground_truth_slots)

        logging.info(report)

        self.log('{}_precision'.format(self.cfg.dataset.field), precision)
        self.log('{}_f1'.format(self.cfg.dataset.field), f1)
        self.log('{}_recall'.format(self.cfg.dataset.field), recall)
        self.log('{}_{}_accuracy'.format(mode, self.cfg.dataset.field), label_acc * 100)
        self.log('slot_precision', slot_precision)
        self.log('slot_recall', slot_recall)
        self.log('slot_f1', slot_f1)
        self.log('slot_joint_goal_accuracy', slot_joint_goal_accuracy)

        if mode == 'val':
            self.epoch_number += 1
            if self.cfg.save_model:
                filename = '{}/epoch-{}-model.bin'.format(self.cfg.dataset.dialogues_example_dir, self.epoch_number)
                torch.save(self.language_model.state_dict(), filename)

    def test_step(self, batch, batch_idx):
        return self.eval_step_helper(batch=batch, mode='test')

    # for inference only
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # return self(batch)
        raise NotImplementedError()

    def forward(self, input_ids, attention_mask, labels, inference=True):

        if self.cfg.library == "huggingface":
            output = self.language_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output['loss']

        elif self.cfg.library == "megatron":
            num_prompt_tokens = (
                len(self.language_model.pseudo_token_ids) if hasattr(self.language_model, 'pseudo_token_ids') else 0
            )

            position_ids = torch.arange(
                start=0, end=num_prompt_tokens + input_ids.size(1), dtype=torch.long, device=input_ids.device,
            )

            position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

            prompt_ids = torch.tensor([0] * input_ids.size(0)) if self.prompt_learning else None

            attn_mask_add_on = torch.ones((attention_mask.size(0), num_prompt_tokens), device=attention_mask.device)
            full_attention_mask = torch.cat([attn_mask_add_on, attention_mask], axis=-1)
            full_attention_mask_expand = torch.tril(
                full_attention_mask.unsqueeze(2).tile(full_attention_mask.size(1))
            ).unsqueeze(1)

            attn_mask = full_attention_mask_expand > 0

            prompt_token_labels = torch.full(
                size=(input_ids.size(0), num_prompt_tokens),
                fill_value=self.tokenizer.tokenizer.pad_token_id,
                dtype=torch.long,
            )

            if self.prompt_learning:
                prompt_token_labels.data = torch.LongTensor(
                    np.tile(np.array(self.language_model.pseudo_token_ids), (input_ids.size(0), 1))
                )

            prompt_token_labels = prompt_token_labels.to(input_ids.device)

            input_ids_new = torch.cat([torch.zeros_like(prompt_token_labels), input_ids], axis=1)
            make_up_last_column_input_ids = (
                torch.ones_like(input_ids_new[:, -1:]) * self.tokenizer.tokenizer.pad_token_id
            )
            left_shifted_input_ids = torch.cat([input_ids_new[:, 1:], make_up_last_column_input_ids], axis=-1)
            if self.prompt_learning:
                unmasked_unreduced_loss = self.language_model(
                    input_ids_new,
                    position_ids,
                    attn_mask,
                    labels=left_shifted_input_ids,
                    taskname_ids=prompt_ids,
                    inference=inference,
                )
            else:
                unmasked_unreduced_loss = self.language_model(
                    input_ids, position_ids, attn_mask, labels=left_shifted_input_ids
                )

            if isinstance(unmasked_unreduced_loss, tuple):
                unmasked_unreduced_loss = unmasked_unreduced_loss[0]

            labels = torch.cat([prompt_token_labels, labels], axis=1)
            make_up_last_column_labels = torch.ones_like(labels[:, -1:]) * self.tokenizer.tokenizer.pad_token_id
            new_labels = torch.cat([labels[:, 1:], make_up_last_column_labels], axis=-1)
            filler = torch.zeros_like(new_labels)
            labels_mask_0 = torch.where(new_labels != -100, new_labels, filler)
            labels_mask = labels_mask_0 > 0

            loss = self.mask_and_reduce_loss(labels_mask, unmasked_unreduced_loss)
        return loss

    def mask_and_reduce_loss(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss

    def decode(self, tokens):
        if tokens not in self.token_to_words:
            self.token_to_words[tokens] = self.tokenizer.tokenizer.decode(tokens)
        return self.token_to_words[tokens]

    def binary_score_candidates(
        self,
        candidate_input_ids,
        candidate_attn_masks,
        utterance_length,
        labels,
        template_length,
        correct_candidate,
        minus_negative=True,
    ):
        best_candidate_input_ids = []

        for i in range(candidate_input_ids.size(0)):
            best_j = 0

            lowest_loss = float("inf")

            for j in range(0, candidate_input_ids.size(1), 2):

                if j > 0 and torch.equal(candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]):
                    break

                start_yes = j if j // 2 == correct_candidate[i].item() else j + 1

                cand_loss = self(
                    candidate_input_ids[i, start_yes : start_yes + 1, :],
                    candidate_attn_masks[i, start_yes : start_yes + 1, :],
                    self.get_binary_score_labels(candidate_input_ids[i, start_yes : start_yes + 1, :]),
                )

                considered_loss = cand_loss.item()

                if minus_negative:
                    start_no = j + 1 if j // 2 == correct_candidate[i].item() else j

                    negative_cand_loss = self(
                        candidate_input_ids[i, start_no : start_no + 1, :],
                        candidate_attn_masks[i, start_no : start_no + 1, :],
                        self.get_binary_score_labels(candidate_input_ids[i, start_no : start_no + 1, :]),
                    )
                    considered_loss -= negative_cand_loss.item()

                if considered_loss < lowest_loss:
                    best_j = start_yes
                    lowest_loss = considered_loss

            best_candidate_input_ids.append(candidate_input_ids[i, best_j, :])

        candidate_tokens = torch.stack(best_candidate_input_ids)
        generated_field, ground_truth_field = self.process_into_structured_fields(
            candidate_tokens, labels, template_length=template_length
        )
        return generated_field, ground_truth_field

    def get_binary_score_labels(self, input_ids):
        # mask out every token except the last token for yes/no/true/false
        labels = torch.zeros_like(input_ids)

        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if input_ids.data[0, j] == self.tokenizer.tokenizer.pad_token_id:
                    stop_point = j
                    break
            last_point = stop_point - 1
            labels.data[i, last_point] = input_ids[i, last_point]

        return labels

    def rank_candidates(
        self, candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length, minus_prior=True
    ):
        best_candidate_input_ids = []

        for i in range(candidate_input_ids.size(0)):
            best_j = 0

            lowest_loss = float("inf")

            utterance_end = utterance_length[i].item()
            for j in range(candidate_input_ids.size(1)):

                if 0 < j < candidate_input_ids.size(1) and torch.equal(
                    candidate_input_ids[i, j, :], candidate_input_ids[i, 0, :]
                ):
                    break

                cand_loss = self(
                    candidate_input_ids[i, j : j + 1, :],
                    candidate_attn_masks[i, j : j + 1, :],
                    candidate_input_ids[i, j : j + 1, :],
                )

                considered_loss = cand_loss.item()

                if minus_prior:
                    utterance_free_cand_loss = self(
                        candidate_input_ids[i, j : j + 1, utterance_end:],
                        candidate_attn_masks[i, j : j + 1, utterance_end:],
                        candidate_input_ids[i, j : j + 1, utterance_end:],
                    )
                    considered_loss -= utterance_free_cand_loss.item()

                if considered_loss < lowest_loss:
                    best_j = j
                    lowest_loss = considered_loss

            best_candidate_input_ids.append(candidate_input_ids[i, best_j, :])

        candidate_tokens = torch.stack(best_candidate_input_ids)
        generated_field, ground_truth_field = self.process_into_structured_fields(
            candidate_tokens, labels, template_length=template_length
        )
        return generated_field, ground_truth_field

    def prepare_megatron_generation(self, labels, input_ids, template_length):
        """
        # adapted from MegatronGPTModel._bucketize_gpt_inference 
        """
        batch_size = labels.size(0)
        prompt_tags = [self.prompt_tags[0]] * batch_size if self.prompt_learning else None
        batch_tokens = input_ids.tolist()

        # unpad tokens
        lens = template_length
        indxs = [index for index in range(batch_size)]
        for lenn, index in zip(lens, indxs):
            batch_tokens[index] = batch_tokens[index][:lenn]

        # chunk tokens by same length
        pre_buckets, lens = [], list(set(lens.tolist()))
        for lenn in lens:
            pre_buckets.append([(tokens, index) for index, tokens in enumerate(batch_tokens) if len(tokens) == lenn])

        buckets, positions, bucket_prompt_tags = [], [], []

        # get buckets and prompts initial positions
        for bucket in pre_buckets:
            buckets.append(torch.tensor([item[0] for item in bucket]).to(device=labels.device))
            positions.append([item[1] for item in bucket])

            # bucket prompt tags identically to their corresponding examples
            if prompt_tags:
                bucket_prompt_tags.append([prompt_tags[item[1]] for item in bucket])

        # Flatten position list
        positions = [item for sublist in positions for item in sublist]

        # Flatten buckets and bucket_prompt_tags # temp fix for megatron complete issue. However, this is also slower than bucketized inference
        buckets = [item.unsqueeze(0) for sublist in buckets for item in sublist]
        bucket_prompt_tags = [[item] for sublist in bucket_prompt_tags for item in sublist]

        request = {"tokens": buckets, "prompt_tags": bucket_prompt_tags}

        return positions, request

    def post_process_megatron_generation(self, outputs):
        text_outputs = [output[0] for output in outputs]
        generated_tokens = self.tokenizer.tokenizer(text_outputs, padding=True, return_tensors="pt").data["input_ids"]
        return generated_tokens

    def generate_candidates(self, labels, template_length, input_ids, attn_masks):

        tokens_to_generate = self.cfg.tokens_to_generate

        if self.cfg.library == "huggingface":
            generated_tokens = []
            max_length = 0
            for i in range(input_ids.size(0)):
                param_dict = {
                    "input_ids": input_ids[i : i + 1, : template_length[i]],
                    "attention_masks": attn_masks[i : i + 1, : template_length[i]],
                    "max_length": template_length[i] + tokens_to_generate,
                    "pad_token_id": self.tokenizer.tokenizer.pad_token_id,
                }
                generated_tokens.append(self.language_model.generate(**param_dict))
                max_length = max(max_length, generated_tokens[-1].size(1))

            # pad each generated to ensure they are of same length in dim 1, therefore stack-able
            generated_tokens = [
                torch.cat(
                    [i, torch.ones((1, max_length - i.size(1))).to(i.device) * self.tokenizer.tokenizer.pad_token_id],
                    axis=-1,
                )
                for i in generated_tokens
            ]
            generated_tokens = torch.cat(generated_tokens, axis=0)

        elif self.cfg.library == "megatron":
            positions, request = self.prepare_megatron_generation(labels, input_ids, template_length)
            outputs = self.language_model.complete(request, positions, tokens_to_generate)
            generated_tokens = self.post_process_megatron_generation(outputs)

        generated_field, ground_truth_field = self.process_into_structured_fields(
            generated_tokens, labels, template_length=template_length
        )

        return generated_field, ground_truth_field

    def eval_step_helper(self, batch, mode='val'):
        (
            input_ids,
            attn_masks,
            labels,
            candidate_input_ids,
            candidate_attn_masks,
            template_length,
            utterance_length,
            correct_candidate,
        ) = batch

        loss = self(input_ids, attn_masks, labels)
        self.log("{}_loss".format(mode), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # ranking using perplexity of candidates following the "<utterance> <label_type>:"
        if self.eval_mode == "ranking":
            generated_field, ground_truth_field = self.rank_candidates(
                candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length
            )
        # autoregressively generate candidates (possibly with constraint)
        elif self.eval_mode == "generation":
            generated_field, ground_truth_field = self.generate_candidates(
                labels, template_length, input_ids, attn_masks
            )
        # comparing likelihood based on the perplexity of generating " Answer: yes" after "<utterance> <label_type>: <candidate_label>"
        # (optionally, the difference of that with " Answer: no" using the flag minus_negative=True)
        elif self.eval_mode == "binary_score":
            generated_field, ground_truth_field = self.binary_score_candidates(
                candidate_input_ids, candidate_attn_masks, utterance_length, labels, template_length, correct_candidate
            )

        else:
            raise ValueError(
                "{} is not among supported options (ranking, generation, binary_score)".format(self.eval_mode)
            )

        return {
            'loss': loss,
            'input': self.tokenizer.tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            'generated_field': generated_field,
            'ground_truth_field': ground_truth_field,
        }

    def process_into_structured_fields(self, generated_tokens, labels, template_length=None):

        generated_field = []

        for i in range(generated_tokens.size(0)):
            start_point = 0 if template_length is None else template_length[i].item()
            stop_point = generated_tokens.size(1)

            for j in range(start_point, stop_point):
                if generated_tokens.data[i, j] == self.tokenizer.tokenizer.pad_token_id:
                    stop_point = j
                    break

            # this is to account for the tokens ' Answer: ' + 'yes'/'no'/'true'/'false'
            if self.eval_mode == "binary_score":
                stop_point -= 3

            one_generated_field = self.decode(generated_tokens[i, start_point:stop_point]).strip()
            generated_field.append(one_generated_field)

        ground_truth_field = self.process_ground_truth_field(labels)

        return generated_field, ground_truth_field

    def process_ground_truth_field(self, labels):
        ground_truth_field = []

        for i in range(labels.size(0)):
            correct_label = tuple(
                [j for j in labels.data[i] if j != self.tokenizer.tokenizer.pad_token_id and j != -100]
            )
            ground_truth_field.append(self.decode(correct_label).strip())

        return ground_truth_field

    def prepare_data(self):
        """
        Preprocessed schema and dialogues and caches this
        """
        if self.data_prepared:
            return

        if self._cfg.dataset.task == 'sgd':
            self.dialogues_processor = DialogueSGDDataProcessor(
                data_dir=self._cfg.dataset.data_dir,
                dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
                tokenizer=self.tokenizer,
                cfg=self._cfg.dataset,
            )
        elif self._cfg.dataset.task in ['assistant', "zero_shot"]:
            self.dialogues_processor = DialogueAssistantDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset
            )
        elif self._cfg.dataset.task == 'design':
            self.dialogues_processor = DialogueDesignDataProcessor(
                data_dir=self._cfg.dataset.data_dir, tokenizer=self.tokenizer, cfg=self._cfg.dataset,
            )
        else:
            raise ValueError("Only sgd, assistant, zero_shot, design supported for Dialogue GPT Classification Model")

        self.data_prepared = True

    def setup(self, stage=None):
        super().setup()
        if self.cfg.library == "megatron" and self.prompt_learning:
            self.language_model.init_new_prompts()

    def update_data_dirs(self, data_dir: str, dialogues_example_dir: str):
        """
        Update data directories

        Args:
            data_dir: path to data directory
            dialogues_example_dir: path to preprocessed dialogues example directory, if not exists will be created.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} is not found")
        self._cfg.dataset.data_dir = data_dir
        self._cfg.dataset.dialogues_example_dir = dialogues_example_dir
        logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
        logging.info(f'Setting model.dataset.dialogues_example_dir to {dialogues_example_dir}.')

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, split=train_data_config.ds_item)

    def setup_multiple_validation_data(self, val_data_config: Optional[DictConfig] = None):
        return self.setup_validation_data(val_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split=val_data_config.ds_item)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self.setup_test_data(test_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, split=test_data_config.ds_item)

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        dataset = DialogueGPTClassificationDataset(
            dataset_split=split,
            dialogues_processor=self.dialogues_processor,
            tokenizer=self.dialogues_processor._tokenizer,
            cfg=dataset_cfg,
        )

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
