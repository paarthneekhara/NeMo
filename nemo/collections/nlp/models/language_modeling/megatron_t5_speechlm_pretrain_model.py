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

import itertools
from typing import Any, List

import torch
from encodec import EncodecModel
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.t5_speechlm_dataset import T5SpeechLMDataset
from nemo.collections.nlp.data.language_modeling.megatron.t5_speechlm_indexed_dataset import (
    build_train_valid_test_datasets,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_speechlm_model import MegatronSpeechLMBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_finetune_model import MegatronT5FinetuneModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import MegatronTokenLevelHead
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
    init_method_normal,
)
from nemo.collections.nlp.modules.common.speech_residual_networks import SimplestModule
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False
try:
    from pynvml.smi import nvidia_smi
except:
    pass

import pprint

__all__ = ['MegatronT5SpeechLMModel']


class MegatronT5SpeechLMModel(MegatronSpeechLMBaseModel):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron T5 model.

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained T5 model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in
    initalization are specified by the user in the config file. The model can
    be prompt-tuned for multiple tasks at once. Virtual prompts are stored in a
    prompt table and can be added or deleted without disrupting virtual prompts
    for other tasks.

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After p-tuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a
    new virtual prompt via p-tuning, they do not need to retrain on all previous
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # torch.autograd.set_detect_anomaly(True)
        super().__init__(cfg, trainer)
        self.model_type = ModelType.encoder_and_decoder
        speech_codebook_size = cfg.data.get('speech_codebook_size', 1024)
        speech_offset = cfg.data.get('speech_offset', 30000)
        speech_head_type = cfg.get('speech_head_type', 'token_level')  # token_level, linear

        self.speech_offset = speech_offset
        self.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.speech_offset = speech_offset
        self.frozen_model.enc_dec_model.speech_codebook_size = speech_codebook_size
        self.frozen_model.enc_dec_model.cross_entropy_type = cfg.get('cross_entropy_type', 'regular')
        self.frozen_model.enc_dec_model.seq_pattern = cfg.get('seq_pattern', 'parallel')
        self.frozen_model.enc_dec_model.speech_head_type = speech_head_type

        # Parallel output is used only for vocab parallel cross entropy.
        self.frozen_model.enc_dec_model.parallel_output = (
            self.frozen_model.enc_dec_model.cross_entropy_type == 'vocab_parallel'
        )
        # Need to explicitly set this since it is already initialiazed
        self.frozen_model.enc_dec_model.tokens_head.parallel_output = self.frozen_model.enc_dec_model.parallel_output

        list_of_speech_heads = []
        list_of_speech_tokens_embeddings = []
        for _ in range(7):
            _speech_head_embedding = tensor_parallel.VocabParallelEmbedding(
                speech_codebook_size, embedding_dim=self.word_embeddings.embedding_dim
            )
            _speech_head_embedding.weight.data.fill_(0)
            _speech_head_embedding.shared = True
            list_of_speech_tokens_embeddings.append(_speech_head_embedding)
            if speech_head_type == 'token_level':
                list_of_speech_heads.append(MegatronTokenLevelHead(_speech_head_embedding.weight.size(0), False))
            elif speech_head_type == 'linear':
                # Linear layer that maps from hidden size to speech codebook size
                hidden_size = self.frozen_model.enc_dec_model.decoder_cfg.hidden_size
                init_method_std = self.frozen_model.enc_dec_model.decoder_cfg.init_method_std
                # Changing to ColumnParallelLinear instead of Linear to support 3b Tensor Parallelism
                _speech_head = tensor_parallel.ColumnParallelLinear(
                    input_size=hidden_size,
                    output_size=speech_codebook_size,
                    bias=True,
                    gather_output=not self.frozen_model.enc_dec_model.parallel_output,
                    init_method=init_method_normal(init_method_std),
                    use_cpu_initialization=False,
                    params_dtype=self.frozen_model.enc_dec_model.dtype,
                )
                list_of_speech_heads.append(_speech_head)

        self.frozen_model.enc_dec_model.speech_tokens_heads = torch.nn.ModuleList(list_of_speech_heads)
        self.frozen_model.enc_dec_model.speech_tokens_embeddings = torch.nn.ModuleList(
            list_of_speech_tokens_embeddings
        )

        if speech_head_type == 'token_level':
            self.frozen_model.enc_dec_model.speech_residual_model_1 = SimplestModule(
                self.frozen_model.enc_dec_model.decoder_cfg.hidden_size, speech_offset + speech_codebook_size
            )
            self.frozen_model.enc_dec_model.speech_residual_model_2 = SimplestModule(
                self.frozen_model.enc_dec_model.decoder_cfg.hidden_size, speech_codebook_size
            )

        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model.set_target_bandwidth(6.0)
        encodec_model.cuda()
        encodec_model.eval()

        self.additional_models = {'encodec': encodec_model}

    def first_stage_of_pipeline(self):
        if self.frozen_model.enc_dec_model.pre_process and parallel_state.get_pipeline_model_parallel_rank() == 0:
            return True
        return False

    def setup_optimizer_param_groups(self):
        """
            Used to create param groups for the optimizer.
            As an example, this can be used to specify per-layer learning rates:

            optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                        ], lr=1e-2, momentum=0.9)

            See https://pytorch.org/docs/stable/optim.html for more information.
            By default, ModelPT will use self.parameters().
            Override this method to add custom param groups.
            In the config file, add 'optim_param_groups' to support different LRs
            for different components (unspecified params will use the default LR):

            model:
                optim_param_groups:
                    encoder:
                        lr: 1e-4
                        momentum: 0.8
                    decoder:
                        lr: 1e-3
                optim:
                    lr: 3e-3
                    momentum: 0.9
        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        known_groups = []
        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group)
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if n.startswith(group):
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            param_groups = [{"params": self.parameters()}]

        self._optimizer_param_groups = param_groups

    def forward(
        self, enc_input, enc_mask, dec_input, dec_mask, position_ids, labels=None, speech_mask=None, inference=False,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        if self.first_stage_of_pipeline():
            # Get embeddings for text tokens and insert virtual token embeddings
            # input_embeds = self.embed_input(input_ids, taskname_ids, inference)
            # import ipdb; ipdb.set_trace()
            input_embeds = self.get_embeddings(enc_input)
            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output, debug_tensors = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
                speech_mask=speech_mask,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output, debug_tensors = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                    speech_mask=speech_mask,
                )

        return output, encoder_input, debug_tensors

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True)
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            if hasattr(t5_cfg, 'encoder') and hasattr(t5_cfg, 'decoder'):
                t5_cfg.encoder.masked_softmax_fusion = False
                t5_cfg.decoder.masked_softmax_fusion = False
            else:
                t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_o2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)
            t5_cfg.precision = trainer.precision
            t5_cfg.tokenizer.num_sentinel_tokens = 39184 - 29056  # cfg.num_speech_tokens 39168
            t5_cfg.seq_length = cfg.get('model_seq_length', 2048)
            t5_cfg.max_position_embeddings = cfg.get('model_seq_length', 2048)

        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
        print(f"self.frozen_model {self.frozen_model}")
        # import ipdb; ipdb.set_trace()

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        seq_length = self.cfg.data.seq_length
        dec_seq_length = seq_length
        tensor_shape = [seq_length, get_micro_batch_size(), self.hidden_size]
        
        # TODO: Pipeline parallelism is not supported yet (follow GPT implementation for it)
        data_iter = dataloader_iter

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            tensor_shape=tensor_shape,
            decoder_seq_length=dec_seq_length,
            dtype=self.autocast_dtype,
            grad_scaler=self.trainer.precision_plugin.scaler.scale if self.cfg.precision == 16 else None,
            sequence_parallel=self.cfg.get('sequence_parallel', False),
            enable_autocast=self.enable_autocast,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def convert_tokens_to_range(self, tokens, apply_offset_correction=True, token_type="encoder"):
        # convert tokens to range [0, 1024]
        output_tokens = tokens.clone()
        if apply_offset_correction:
            output_tokens[0] = output_tokens[0] - self.speech_offset
        output_tokens = torch.clamp(output_tokens, min=0, max=1023)
        if self.cfg.seq_pattern == "delay_parallel" and token_type == "decoder":
            output_tokens_new = []
            for _c in range(output_tokens.shape[0]):
                si = _c
                ei = _c + output_tokens.shape[1] - 8
                output_tokens_new.append(output_tokens[_c, si:ei])
            output_tokens_new = torch.stack(output_tokens_new)
            output_tokens = output_tokens_new

        return output_tokens

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            for key in batch:
                batch[key] = batch[key].cuda()

            enc_input = batch['enc_input']
            enc_mask = batch['enc_mask']
            dec_input = batch['dec_input']
            dec_input_mask = batch['dec_mask']
            labels = batch['labels']
            loss_mask = batch['loss_mask']
            position_ids = batch['position_ids']
            speech_mask = batch['speech_mask']

            output_tensor, _, debug_tensors = model(
                enc_input,
                enc_mask,
                dec_input,
                dec_input_mask,
                position_ids,
                labels=labels,
                speech_mask=speech_mask,
                inference=False,
            )
            output_tensor = output_tensor.contiguous()

            if self.trainer.global_step % 100 == 0:
                try:
                    # Print GPU utilization
                    nvsmi = nvidia_smi.getInstance()
                    gpu_utilization = nvsmi.DeviceQuery('utilization.gpu, memory.used, memory.total')
                    pprint.pprint(gpu_utilization)
                except:
                    pass
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        # Encodec does not work with fp16, so we disable autocast for logging audio
                        if speech_mask[0].sum() != 0:
                            if self.cfg.seq_pattern in ["delay_parallel", "parallel"]:
                                enc_input_example = self.convert_tokens_to_range(enc_input[0])
                                dec_input_example = self.convert_tokens_to_range(dec_input[0], token_type="decoder")

                                enc_wav = self.additional_models['encodec'].decode([[enc_input_example[None], None]])[
                                    0, 0
                                ]
                                self.logger.experiment.add_audio("Enc Input", enc_wav, self.global_step, 24000)

                                dec_wav = self.additional_models['encodec'].decode([[dec_input_example[None], None]])[
                                    0, 0
                                ]
                                self.logger.experiment.add_audio("Dec Input", dec_wav, self.global_step, 24000)

                                token_logits = debug_tensors[0]
                                speech_logits_list = debug_tensors[1]
                                if self.frozen_model.enc_dec_model.parallel_output:
                                    # Gather from tensor parallel region
                                    token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                                        token_logits
                                    )
                                    for _i in range(len(speech_logits_list)):
                                        speech_logits_list[
                                            _i
                                        ] = tensor_parallel.gather_from_tensor_model_parallel_region(
                                            speech_logits_list[_i]
                                        )

                                speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t, b, 1024, 7)
                                token_logits_example = token_logits[:, 0, :] * 1
                                speech_logits_example = speech_logits[:, 0, :, :] * 1
                                first_layer_tokens = token_logits_example.argmax(dim=1) - 30000
                                other_layer_tokens = []
                                for _i in range(speech_logits_example.shape[2]):
                                    other_layer_tokens.append(speech_logits_example[:, :, _i].argmax(dim=1))

                                all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens)  # (8, t)
                                all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                                all_layer_tokens = self.convert_tokens_to_range(
                                    all_layer_tokens, apply_offset_correction=False, token_type="decoder"
                                )
                                predicted_wav = self.additional_models['encodec'].decode(
                                    [[all_layer_tokens[None], None]]
                                )[0, 0]
                                self.logger.experiment.add_audio("Pred Wav", predicted_wav, self.global_step, 24000)

                            elif self.cfg.seq_pattern == "flatten":
                                enc_input_example = enc_input[0][0]
                                # Add a dummy token to enc_input_example in the beginning
                                enc_input_example = torch.cat([torch.tensor([0]).cuda(), enc_input_example])[:-1]
                                dec_input_example = dec_input[0][0]

                                token_logits = debug_tensors[0]
                                speech_logits = debug_tensors[1]
                                token_logits_example = token_logits[:, 0, :] * 1
                                predicted_tokens = token_logits_example.argmax(dim=1)
                                predicted_tokens = torch.cat([torch.tensor([0]).cuda(), predicted_tokens])[:-1]

                                labels_example = labels[0][0]
                                labels_example = torch.cat([torch.tensor([0]).cuda(), labels_example])[:-1]

                                all_layer_tokens_predicted = []
                                all_layer_tokens_encinput = []
                                all_layer_tokens_decinput = []
                                all_layer_tokens_labels = []
                                for _c in range(8):
                                    # 0th layer tokens are indices 0, 8, 16, 24, 32, 40, 48, 56
                                    # 1st layer tokens are indices 1, 9, 17, 25, 33, 41, 49, 57
                                    layer_tokens_predicted = predicted_tokens[_c::8]
                                    layer_tokens_encinput = enc_input_example[_c::8]
                                    layer_tokens_decinput = dec_input_example[_c::8]
                                    layer_tokens_labels = labels_example[_c::8]

                                    layer_tokens_predicted = (
                                        layer_tokens_predicted - self.speech_offset - (_c * self.speech_codebook_size)
                                    )
                                    layer_tokens_encinput = (
                                        layer_tokens_encinput - self.speech_offset - (_c * self.speech_codebook_size)
                                    )
                                    layer_tokens_decinput = (
                                        layer_tokens_decinput - self.speech_offset - (_c * self.speech_codebook_size)
                                    )
                                    layer_tokens_labels = (
                                        layer_tokens_labels - self.speech_offset - (_c * self.speech_codebook_size)
                                    )

                                    all_layer_tokens_predicted.append(layer_tokens_predicted)
                                    all_layer_tokens_encinput.append(layer_tokens_encinput)
                                    all_layer_tokens_decinput.append(layer_tokens_decinput)
                                    all_layer_tokens_labels.append(layer_tokens_labels)

                                all_layer_tokens_predicted = torch.stack(all_layer_tokens_predicted)
                                all_layer_tokens_encinput = torch.stack(all_layer_tokens_encinput)
                                all_layer_tokens_decinput = torch.stack(all_layer_tokens_decinput)
                                all_layer_tokens_labels = torch.stack(all_layer_tokens_labels)

                                all_layer_tokens_predicted = torch.clip(all_layer_tokens_predicted, 0, 1023)
                                all_layer_tokens_encinput = torch.clip(all_layer_tokens_encinput, 0, 1023)
                                all_layer_tokens_labels = torch.clip(all_layer_tokens_labels, 0, 1023)

                                enc_wav = self.additional_models['encodec'].decode(
                                    [[all_layer_tokens_encinput[None], None]]
                                )[0, 0]
                                self.logger.experiment.add_audio("Enc Input", enc_wav, self.global_step, 24000)

                                dec_wav = self.additional_models['encodec'].decode(
                                    [[all_layer_tokens_decinput[None], None]]
                                )[0, 0]
                                self.logger.experiment.add_audio("Dec Input", dec_wav, self.global_step, 24000)

                                predicted_wav = self.additional_models['encodec'].decode(
                                    [[all_layer_tokens_predicted[None], None]]
                                )[0, 0]
                                self.logger.experiment.add_audio("Pred Wav", predicted_wav, self.global_step, 24000)

                                labels_wav = self.additional_models['encodec'].decode(
                                    [[all_layer_tokens_labels[None], None]]
                                )[0, 0]
                                self.logger.experiment.add_audio("Labels Wav", labels_wav, self.global_step, 24000)
                            else:
                                raise NotImplementedError(f"seq_pattern {self.cfg.seq_pattern} not implemented")

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When using pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.frozen_model.enc_dec_model.set_input_tensor(input_tensor)

    def on_train_epoch_start(self) -> None:
        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        mbs = self.cfg.get('validation_micro_batch_size', self.cfg.micro_batch_size)
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_validation_epoch_start()

    def training_step(self, dataloader_iter, batch_idx):
        self._optimizer.zero_grad()
        loss_mean = self.fwd_bwd_step(dataloader_iter, batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        print(f'global_step {self.trainer.global_step}')
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        consumed_samples = self.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples', consumed_samples, prog_bar=True, rank_zero_only=True, batch_size=1,
        )
        return loss_mean

    def get_predictions(self, input_ids, enc_mask, encoder_input, labels):
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5FinetuneModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5FinetuneModel.ids_to_text(labels, self.tokenizer)
        input_text = MegatronT5FinetuneModel.ids_to_text(input_ids, self.tokenizer)
        return {
            'predicted_token_ids': preds_text,
            'labels': labels_text,
            'enc_inputs': input_text,
        }

    def get_embeddings(self, tokens):
        '''
        Does the speech embeddings here
        '''
        out = None
        assert tokens.dim() == 3
        for i in range(tokens.size()[1]):
            if i == 0:
                # Embed first layer using word embeddings
                out = self.word_embeddings(tokens[:, i, :])
            else:
                # Embed other layers using speech embeddings
                cur = self.frozen_model.enc_dec_model.speech_tokens_embeddings[i - 1](tokens[:, i, :])
                # do not add embeddings of zero tokens of other channels (except the first channel)
                include_channel_flag = (torch.sum(tokens[:, i, :], dim=1) > 0).float()
                cur = cur * include_channel_flag.unsqueeze(1).unsqueeze(2)
                out = out + cur

        return out

    def validation_step(self, batch, batch_idx, inference=False):
        # virtual_tokens, context_tokens, question_tokens, enc_mask, dec_input, dec_input_mask, labels, loss_mask, position_ids, taskname_ids, speech_mask = batch
        # enc_input_ids, dec_input_ids, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch
        enc_input_ids = batch['enc_input']
        # does not use dataloader_iter due to device placement issues arising from PTL
        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(enc_input_ids.size(0), gbs)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)
        metrics = {'loss': loss_mean}

        with torch.no_grad():
            labels_original = batch['labels'].clone()
            loss_mask = batch['loss_mask']
            speech_mask = batch['speech_mask']
            output_tensor, encoder_input, output_logits = self.forward(
                enc_input_ids,
                batch['enc_mask'],
                batch['dec_input'],
                batch['dec_mask'],
                batch['position_ids'],
                labels=batch['labels'],
                speech_mask=batch['speech_mask'],
                inference=True,
            )

            first_layer_logits, speech_logits_list = output_logits  # first_layer_logits: (t,bs,vocab_size)
            if self.frozen_model.enc_dec_model.parallel_output:
                # Gather from tensor parallel region
                first_layer_logits = tensor_parallel.gather_from_tensor_model_parallel_region(first_layer_logits)
                for _i in range(len(speech_logits_list)):
                    speech_logits_list[_i] = tensor_parallel.gather_from_tensor_model_parallel_region(
                        speech_logits_list[_i]
                    )
            speech_logits = torch.stack(speech_logits_list, dim=-1)  # (t,bs,1024,7)
            first_layer_preds = first_layer_logits.argmax(dim=2)  # (t,bs)
            first_layer_preds = first_layer_preds.transpose(0, 1)  # (bs,t)
            labels_first_layer = labels_original[:, 0, :]  # (bs,t)
            correct_predictions = first_layer_preds == labels_first_layer  # (bs,t)
            correct_predictions_speech = correct_predictions * loss_mask * speech_mask  # (bs,t)
            total_correct_predictions_speech = torch.sum(correct_predictions_speech)
            total_predictions_speech = torch.sum(loss_mask * speech_mask)
            total_predictions_speech = max(total_predictions_speech, 1)  # to avoid divide by zero
            first_layer_accuracy = total_correct_predictions_speech / total_predictions_speech

            correct_predictions_text = correct_predictions * loss_mask * (1 - speech_mask)  # (bs,t)
            total_correct_predictions_text = torch.sum(correct_predictions_text)
            total_predictions_text = torch.sum(loss_mask * (1 - speech_mask))
            total_predictions_text = max(total_predictions_text, 1)  # to avoid divide by zero
            first_layer_accuracy_text = total_correct_predictions_text / total_predictions_text

            first_layer_loss = torch.nn.functional.cross_entropy(
                first_layer_logits.permute(1, 2, 0), labels_first_layer, reduction='none'
            )  # (bs,t)
            first_layer_loss_speech = torch.sum(first_layer_loss * loss_mask * speech_mask) / total_predictions_speech
            first_layer_loss_text = (
                torch.sum(first_layer_loss * loss_mask * (1 - speech_mask)) / total_predictions_text
            )

            metrics['first_layer_accuracy'] = first_layer_accuracy
            metrics['first_layer_loss'] = first_layer_loss_speech
            metrics['text_loss'] = first_layer_loss_text
            metrics['text_accuracy'] = first_layer_accuracy_text

            speech_loss = first_layer_loss
            for i in range(7):
                speech_logits_i = speech_logits[:, :, :, i]
                speech_preds_i = speech_logits_i.argmax(dim=2)  # (t,bs)
                speech_preds_i = speech_preds_i.transpose(0, 1)  # (bs,t)
                labels_i = labels_original[:, i + 1, :]  # (bs,t)
                correct_predictions_i = speech_preds_i == labels_i  # (bs,t)
                correct_predictions_i = correct_predictions_i * loss_mask * speech_mask  # (bs,t)
                total_correct_predictions_i = torch.sum(correct_predictions_i)
                total_predictions_i = torch.sum(loss_mask * speech_mask)
                total_predictions_i = max(total_predictions_i, 1)
                speech_accuracy_i = total_correct_predictions_i / total_predictions_i
                loss_i = torch.nn.functional.cross_entropy(
                    speech_logits_i.permute(1, 2, 0), labels_i, reduction='none'
                )  # (bs,t)
                loss_i = torch.sum(loss_i * loss_mask * speech_mask) / total_predictions_i
                metrics[f'speech_accuracy_{i+1}'] = speech_accuracy_i
                metrics[f'speech_loss_{i+1}'] = loss_i
                speech_loss += loss_i
            metrics['speech_loss_total'] = speech_loss

        self.train(mode=mode)
        self.frozen_model.train()
        return metrics

    def validation_epoch_end(self, outputs):
        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([i['loss'] for i in outputs]).mean()
            else:
                averaged_loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            logging.info(f'Validation loss: {averaged_loss}')

        else:
            averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
            logging.info(f'Validation loss: {averaged_loss}')
            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)

            averaged_first_layer_accuracy = torch.stack([item['first_layer_accuracy'] for item in outputs]).mean()
            logging.info(f'Validation first_layer_accuracy: {averaged_first_layer_accuracy}')
            self.log(
                'val_first_layer_accuracy_speech',
                averaged_first_layer_accuracy,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )

            averaged_first_layer_loss = torch.stack([item['first_layer_loss'] for item in outputs]).mean()
            logging.info(f'Validation first_layer_loss: {averaged_first_layer_loss}')
            self.log(
                'val_first_layer_loss_speech',
                averaged_first_layer_loss,
                prog_bar=True,
                rank_zero_only=True,
                batch_size=1,
            )

            averaged_speech_loss_total = torch.stack([item['speech_loss_total'] for item in outputs]).mean()
            logging.info(f'Validation speech_loss_total: {averaged_speech_loss_total}')
            self.log(
                'val_speech_loss_total', averaged_speech_loss_total, prog_bar=True, rank_zero_only=True, batch_size=1
            )

            averaged_text_loss = torch.stack([item['text_loss'] for item in outputs]).mean()
            logging.info(f'Validation text_loss: {averaged_text_loss}')
            self.log('val_text_loss', averaged_text_loss, prog_bar=True, rank_zero_only=True, batch_size=1)

            averaged_text_accuracy = torch.stack([item['text_accuracy'] for item in outputs]).mean()
            logging.info(f'Validation text_accuracy: {averaged_text_accuracy}')
            self.log('val_text_accuracy', averaged_text_accuracy, prog_bar=True, rank_zero_only=True, batch_size=1)

            for i in range(1, 8):
                averaged_speech_accuracy = torch.stack([item[f'speech_accuracy_{i}'] for item in outputs]).mean()
                averaged_speech_loss = torch.stack([item[f'speech_loss_{i}'] for item in outputs]).mean()
                logging.info(f'Validation speech_accuracy_{i}: {averaged_speech_accuracy}')
                logging.info(f'Validation speech_loss_{i}: {averaged_speech_loss}')
                self.log(
                    f'val_speech_accuracy_{i}',
                    averaged_speech_accuracy,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                )
                self.log(
                    f'val_speech_loss_{i}', averaged_speech_loss, prog_bar=True, rank_zero_only=True, batch_size=1
                )

        if self.cfg.get("report_validation_metric", False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]

            all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in outputs]))
            all_inputs = list(itertools.chain(*[item['enc_inputs'] for item in outputs]))

            assert len(all_preds) == len(all_labels)
            assert len(all_preds) == len(all_inputs)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                val_metric_dict = self.validation_metric.get_score(
                    [i[2] for i in gather_results_dedup], [i[1] for i in gather_results_dedup],
                )

                for metric, val in val_metric_dict.items():
                    logging.info(f'Validation {metric}: {val}')
                val_metric = list(val_metric_dict.items())[0][1]
                metric_name = list(val_metric_dict.items())[0][0]
            else:
                val_metric = torch.tensor(0.0).cuda()
                metric_name = ''

            self.log(f'val_{metric_name}', val_metric, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    rampup_batch_size=None,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            # TODO: look at this
            consumed_samples = self.compute_consumed_samples(0)
            # consumed_samples = 0
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )

            drop_last = True
            if not self.cfg.data.get('validation_drop_last', True):
                logging.info(f'Drop last in validation dataset is set to False')
                drop_last = False
            pad_samples_to_global_batch_size = False
            if self.cfg.data.get('pad_samples_to_global_batch_size', False):
                logging.info('pad_samples_to_global_batch_size set to True')
                pad_samples_to_global_batch_size = True

            self._validation_dl = self.build_pretraining_data_loader(
                self._validation_ds, consumed_samples, "validation", drop_last, pad_samples_to_global_batch_size
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def setup(self, stage=None):

        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict' and self.first_stage_of_pipeline():
            return

        # self.setup_test_data()
        if stage == 'test':
            self.setup_test_data(self.cfg.data)
            return

        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        # print("BUILDING TRAIN VALID TEST DATASETS DONE")
        # import ipdb; ipdb.set_trace()

    def build_train_valid_test_datasets(self):
        logging.info('Building T5 datasets.')

        global_batch_size = self.cfg.global_batch_size

        # TODO: remove hardcoding
        max_train_steps = self.trainer.max_steps
        eval_iters = 100
        test_iters = 100
        print("MAX TRAIN STEPS: ", max_train_steps)

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.get('skip_warmup', True),
            tokenizer=self.tokenizer,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building T5 datasets.')
        # import ipdb; ipdb.set_trace()

        return self._train_ds, self._validation_ds, self._test_ds

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        input_ids, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        batch_size, seq_length = input_ids.shape
        if self.first_stage_of_pipeline():
            input_embeds = self.embed_input(input_ids, taskname_ids, use_cached_reps=True)

            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds

        else:
            encoder_input = torch.zeros((batch_size, seq_length, self.hidden_size), dtype=self.autocast_dtype).cuda()

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5FinetuneModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5FinetuneModel.ids_to_text(input_ids, self.tokenizer)

        if labels is not None:
            labels_text = MegatronT5FinetuneModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:

        gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        all_preds = list(itertools.chain(*[item['preds_text'] for item in outputs[0]]))
        all_labels = list(itertools.chain(*[item['labels_text'] for item in outputs[0]]))
        all_inputs = list(itertools.chain(*[item['input_text'] for item in outputs[0]]))

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_inputs)

        # Gather inputs, predictions, and ground truths from all workers
        torch.distributed.all_gather_object(
            gather_results,
            [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
            group=parallel_state.get_data_parallel_group(),
        )

        # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
        if parallel_state.get_data_parallel_rank() == 0:
            gather_results_dedup = list(set(itertools.chain(*gather_results)))

            input_prediction_pair = []
            correct = 0
            for (input, pred, label) in gather_results_dedup:
                input_prediction_pair.append((input, pred))
                if label:
                    if pred == label:
                        correct += 1

            acc = correct / len(gather_results_dedup) if all_labels[0] else None
            logging.info(f'Prediction results: {acc}')
            logging.info(f'Test finish')
