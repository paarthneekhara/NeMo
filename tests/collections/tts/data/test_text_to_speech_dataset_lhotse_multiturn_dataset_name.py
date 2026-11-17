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

from types import SimpleNamespace

import pytest
from lhotse import CutSet
from lhotse.testing.dummies import dummy_cut, dummy_supervision

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset


def _dataset():
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.dataset_type = "val"
    dataset._training_step = SimpleNamespace(value=0)
    dataset.challenging_text_start_step = 10**9
    dataset.output_roles = {"agent"}
    return dataset


def _cut(idx, speaker, task="tts"):
    cut = dummy_cut(idx, supervisions=[dummy_supervision(idx)])
    cut.supervisions[0].speaker = speaker
    cut.task = task
    return cut


@pytest.mark.unit
def test_dataset_name_survives_speaker_role_rewrite():
    dataset = _dataset()
    cuts = CutSet(
        [
            _cut(0, "| Language:en Dataset:synthRepeat2611 Speaker:hifitts_8051 |"),
            _cut(1, "not a speaker tag"),
        ]
    )
    cuts, _ = dataset._prepare_cuts(cuts)
    cuts = list(cuts)
    assert [c.supervisions[0].speaker for c in cuts] == ["agent", "agent"]
    assert [dataset._get_dataset_name(c) for c in cuts] == ["synthRepeat2611", "unknown"]


@pytest.mark.unit
def test_non_tts_cut_keeps_speaker_tag_lookup():
    dataset = _dataset()
    cuts, _ = dataset._prepare_cuts(CutSet([_cut(0, "| Language:de Dataset:mdesta_de Speaker:1 |", task="s2s")]))
    cut = next(iter(cuts))
    assert cut.supervisions[0].speaker.startswith("| Language:de")
    assert dataset._get_dataset_name(cut) == "mdesta_de"
