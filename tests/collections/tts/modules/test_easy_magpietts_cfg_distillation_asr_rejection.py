# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest
import torch

from nemo.collections.tts.modules.easy_magpietts_cfg_distillation import (
    ASRRejectionParams,
    TeacherOutput,
    apply_asr_rejection,
)

ROUTER = {
    "language_routes": {"en": "nemo_en", "de": "whisper"},
    "backends": {"nemo_en": {"type": "nemo_process"}, "whisper": {"type": "whisper"}},
}


def _teacher_output(batch_size, rejected=None, sample_weights=None):
    return TeacherOutput(
        codes=torch.zeros(batch_size, 8, 10, dtype=torch.long),
        logits=torch.zeros(batch_size, 1),
        lens=torch.full((batch_size,), 10, dtype=torch.long),
        sample_weights=sample_weights,
        rejected=rejected,
        codes_lt=None,
        logits_lt=None,
    )


class TestASRRejectionParams:
    @pytest.mark.unit
    def test_disabled_without_config(self):
        assert ASRRejectionParams.create_from_config(None, 0.01) is None

    @pytest.mark.unit
    def test_defaults_to_zero_cer_and_model_rejection_weight(self):
        params = ASRRejectionParams.create_from_config(dict(ROUTER), 0.01)
        assert params.max_cer == 0.0
        assert params.rejection_weight == 0.01
        assert params.threshold_for("en", "anything") == 0.0

    @pytest.mark.unit
    def test_dataset_language_and_threshold_selection(self):
        cfg = {**ROUTER, "max_cer": None, "max_cer_by_lang": {"en": 0.0, "de": None}, "datasets": ["synthRepeat2611"]}
        params = ASRRejectionParams.create_from_config(cfg, 0.01)
        assert params.threshold_for("en", "synthRepeat2611") == 0.0
        assert params.threshold_for("en", "libritts") is None
        assert params.threshold_for("de", "synthRepeat2611") is None
        assert params.threshold_for("hi", "synthRepeat2611") is None

    @pytest.mark.unit
    def test_per_language_override(self):
        cfg = {**ROUTER, "max_cer": 0.02, "max_cer_by_lang": {"de": 0.05}}
        params = ASRRejectionParams.create_from_config(cfg, None)
        assert params.threshold_for("en", "x") == 0.02
        assert params.threshold_for("de", "x") == 0.05

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg",
        [
            {"max_cer": 0.0},
            {**ROUTER, "max_cer": 1.5},
            {**ROUTER, "max_cer": None},
            {**ROUTER, "max_cer_by_lang": {"hi": 0.0}},
            {**ROUTER, "rejection_weight": 2.0},
        ],
    )
    def test_invalid_config(self, cfg):
        with pytest.raises(ValueError):
            ASRRejectionParams.create_from_config(cfg, 0.01)


class TestApplyASRRejection:
    @pytest.mark.unit
    def test_rejects_items_above_threshold(self):
        batch = {"dataset_names": ["a", "b", "c", "d"], "languages": ["en"] * 4, "raw_texts": ["x"] * 4}
        rejected = torch.tensor([True, False, False, False])
        out = apply_asr_rejection(
            teacher_output=_teacher_output(4, rejected=rejected),
            batch=batch,
            item_indices=[1, 2, 3],
            cers=[0.0, 0.1, 0.03],
            thresholds=[0.0, 0.0, 0.05],
            transcripts=["x", "y", "z"],
            rejection_weight=0.0,
        )
        assert out.rejected.tolist() == [True, False, True, False]
        assert out.sample_weights.tolist() == [1.0, 1.0, 0.0, 1.0]

    @pytest.mark.unit
    def test_keeps_existing_weights_and_skips_without_rejections(self):
        weights = torch.tensor([0.01, 1.0])
        teacher_output = _teacher_output(2, rejected=torch.tensor([True, False]), sample_weights=weights)
        out = apply_asr_rejection(
            teacher_output=teacher_output,
            batch={},
            item_indices=[1],
            cers=[0.0],
            thresholds=[0.0],
            transcripts=["x"],
            rejection_weight=0.01,
        )
        assert out.sample_weights is weights
        assert out.rejected.tolist() == [True, False]
