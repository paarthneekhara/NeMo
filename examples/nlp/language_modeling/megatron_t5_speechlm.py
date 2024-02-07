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

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_model import MegatronT5SpeechLMModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# mp.set_start_method("spawn", force=True)


"""
This is an example of how to ptune/prompt-tune a pretrained T5 model.
Be sure to use a .nemo T5 model with this code. If you've downloaded
a model from NGC or are otherwise using a MegatronLM model, please use
either megatron_ckpt_to_nemo.py or megatron_lm_ckpt_to_nemo.py found
within this examples directory to convert your model to .nemo format.
"""


@hydra_runner(config_path="conf", config_name="megatron_t5_prompt_learning.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # MegatronTrainerBuilder compat checks
    if "gradient_as_bucket_view" not in cfg.model:
        with open_dict(cfg):
            cfg.model.gradient_as_bucket_view=False

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # load existing or init new soft prompt T5 model
    if cfg.model.get("restore_path", None) is not None:
        logging.info(f"cfg.model.restore_path {cfg.model.restore_path}")
        model = MegatronT5SpeechLMModel.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )

    else:
        logging.info(f"cfg.model.restore_path is None")
        model = MegatronT5SpeechLMModel(cfg.model, trainer=trainer)
        model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    trainer.fit(model)


if __name__ == '__main__':
    main()
