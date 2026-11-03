# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for the per-entry ``keep_supervision_custom`` filter of Lhotse input configs."""
from unittest.mock import patch

from lhotse import CutSet
from lhotse.testing.dummies import dummy_cut, dummy_supervision
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.cutset import parse_group, read_cutset_from_config


def _cuts(emotions):
    cuts = []
    for idx, emotion in enumerate(emotions):
        sup = dummy_supervision(idx)
        sup.custom = None if emotion is None else {"emotion": emotion, "speaker_id": f"spk{idx % 2}"}
        cuts.append(dummy_cut(idx, supervisions=[sup]))
    return CutSet.from_cuts(cuts)


def _parser_returning(cuts):
    return lambda cfg: (cuts, True)


EMOTIONS = ["other", "angry", "neutral", "Generic", "Happy_Intense_Correlated", None]
PROPAGATE = {"reweight_temperature": None}


def test_keep_supervision_custom_keeps_listed_values():
    cfg = OmegaConf.create(
        {"type": "lhotse_shar", "shar_path": "/fake", "keep_supervision_custom": {"emotion": ["other", "neutral", "Generic"]}}
    )
    with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=_parser_returning(_cuts(EMOTIONS))):
        cuts, _ = parse_group(cfg, PROPAGATE)
    assert [c.supervisions[0].custom["emotion"] for c in cuts] == ["other", "neutral", "Generic"]


def test_keep_supervision_custom_requires_every_field():
    cfg = OmegaConf.create(
        {
            "type": "lhotse_shar",
            "shar_path": "/fake",
            "keep_supervision_custom": {"emotion": ["other", "neutral"], "speaker_id": ["spk0"]},
        }
    )
    with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=_parser_returning(_cuts(EMOTIONS))):
        cuts, _ = parse_group(cfg, PROPAGATE)
    assert [c.supervisions[0].custom["emotion"] for c in cuts] == ["other", "neutral"]


def test_without_keep_supervision_custom_nothing_is_dropped():
    cfg = OmegaConf.create({"type": "lhotse_shar", "shar_path": "/fake"})
    with patch("nemo.collections.common.data.lhotse.cutset.get_parser_fn", return_value=_parser_returning(_cuts(EMOTIONS))):
        cuts, _ = parse_group(cfg, PROPAGATE)
    assert len(list(cuts)) == len(EMOTIONS)


def test_filter_applies_only_to_its_entry():
    config = OmegaConf.create(
        {
            "input_cfg": [
                {"type": "lhotse_shar", "shar_path": "/all", "weight": 1.0},
                {
                    "type": "lhotse_shar",
                    "shar_path": "/subset",
                    "weight": 1.0,
                    "keep_supervision_custom": {"emotion": ["other"]},
                },
            ],
            "shuffle": False,
            "shard_seed": 0,
            "force_finite": True,
        }
    )
    sources = {"/all": _cuts(EMOTIONS), "/subset": _cuts(EMOTIONS).modify_ids(lambda cid: f"subset-{cid}")}
    with patch(
        "nemo.collections.common.data.lhotse.cutset.get_parser_fn",
        return_value=lambda cfg: (sources[cfg.shar_path], True),
    ):
        cuts, _ = read_cutset_from_config(config)
    cuts = list(cuts)
    ids = [c.id for c in cuts]
    assert len([i for i in ids if not i.startswith("subset-")]) == len(EMOTIONS)
    subset = [c for c in cuts if c.id.startswith("subset-")]
    assert [c.supervisions[0].custom["emotion"] for c in subset] == ["other"]
