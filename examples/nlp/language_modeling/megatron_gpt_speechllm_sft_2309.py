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

import os

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronSpeechGPTModel, MegatronSpeechGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        gpt_cfg.micro_batch_size = cfg.model.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        # gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        # gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        gpt_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        gpt_cfg.rampup_batch_size = cfg.model.rampup_batch_size  # Missing from older checkpoints?
        gpt_cfg.override_vocab_size = cfg.model.override_vocab_size  # Missing from older checkpoints?
        gpt_cfg.output_size = cfg.model.output_size  # Missing from older checkpoints?
        gpt_cfg.use_flash_attention = cfg.model.use_flash_attention  # Missing from older checkpoints?
        gpt_cfg.embedding_scale = cfg.model.get("embedding_scale", 1.0)  # Missing from older checkpoints?
        gpt_cfg.speech_loss_scale = cfg.model.get("speech_loss_scale", 1.0)  # Missing from older checkpoints?
        gpt_cfg.text_size = cfg.model.get("text_size", 256000)  # Missing from older checkpoints?
        gpt_cfg.task_templates = cfg.model.task_templates
        gpt_cfg.seq_pattern = cfg.model.seq_pattern
        gpt_cfg.use_speech_mask_for_embedding = cfg.model.use_speech_mask_for_embedding
        gpt_cfg.return_all_selfattention_probs = cfg.model.return_all_selfattention_probs
        gpt_cfg.train_check_interval = cfg.model.train_check_interval
        gpt_cfg.train_check_interval = cfg.model.train_check_interval
        gpt_cfg.attn_prior_end_step = cfg.model.attn_prior_end_step
        gpt_cfg.attn_prior_scaledown_start_step = cfg.model.attn_prior_scaledown_start_step
        gpt_cfg.attn_prior_starting_strength = cfg.model.attn_prior_starting_strength
        gpt_cfg.use_attention_prior = cfg.model.get("use_attention_prior", False)
        gpt_cfg.share_embeddings_and_output_weights = cfg.model.get("share_embeddings_and_output_weights", False)
        gpt_cfg.temperature = cfg.model.get("temperature", 0.7)
        gpt_cfg.top_k = cfg.model.get("top_k", 60)
        # gpt_cfg.bias_activation_fusion = cfg.model.bias_activation_fusion  # Missing from older checkpoints?
        # gpt_cfg.bias_dropout_add_fusion = cfg.model.bias_dropout_add_fusion  # Missing from older checkpoints?
        sft_cls = MegatronSpeechGPTModel
        gpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


def load_from_nemo(cls, cfg, trainer, gpt_cfg, modify_confg_fn, restore=True):
    gpt_cfg = modify_confg_fn(gpt_cfg, cfg, add_cfg_to_tree=False)
    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
    if restore:
        model = cls.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=gpt_cfg,
            save_restore_connector=save_restore_connector,
        )
    else:
        gpt_cfg.tokenizer.model = cfg.tokenizer_model
        gpt_cfg.position_embedding_type = cfg.get("override_position_embedding_type", "rope")
        model = cls(gpt_cfg, trainer)
    return model

# def load_from_ckpt(cls, cfg, trainer, gpt_cfg, modify_confg_fn):
#     gpt_cfg = modify_confg_fn(gpt_cfg, cfg, add_cfg_to_tree=False)
#     save_restore_connector = NLPSaveRestoreConnector()
#     if os.path.isdir(cfg.model.restore_from_path):
#         save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
#     model = cls.restore_from(
#         restore_path=cfg.model.restore_from_path,
#         trainer=trainer,
#         override_config_path=gpt_cfg,
#         save_restore_connector=save_restore_connector,
#     )
#     return model


@hydra_runner(config_path="conf", config_name="megatron_gpt_speechllm_sft_config_2309")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if "restore_from_path" in cfg.model:
        logging.info("Restore from {cfg.model.restore_from_path}")
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
        gpt_cfg = MegatronSpeechGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
            map_location="cpu",
        )
        model = load_from_nemo(MegatronSpeechGPTSFTModel, cfg, trainer, gpt_cfg, modify_confg_fn=_modify_config, restore=cfg.restore)
    else:
        model = MegatronSpeechGPTSFTModel(cfg.model, trainer)
    mode = cfg.get("mode", "training")
    if mode == "training":
        if cfg.get('init_from_ptl_ckpt', None) is not None:
            print("Initializing from PTL checkpoint")
            model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
        trainer.fit(model)
    elif mode == "inference":
        # Pass +mode="inference" +init_from_ptl_ckpt="/path/to/model.ckpt"
        model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
        trainer.test(model)


if __name__ == '__main__':
    main()
