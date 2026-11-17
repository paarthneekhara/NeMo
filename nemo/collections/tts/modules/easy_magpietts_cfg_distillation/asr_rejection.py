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

"""ASR-based rejection of teacher rollouts for EasyMagpieTTS CFG distillation."""

import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from nemo.collections.tts.modules.easy_magpietts_cfg_distillation.distillation_types import TeacherOutput
from nemo.utils import logging


@dataclass
class ASRRejectionParams:
    """Settings of the ASR rejection rule.

    A teacher rollout is checked when its dataset is listed in `datasets` (or `datasets` is `None`), its
    language has an ASR route in `router_cfg.language_routes`, and its CER threshold is not `None`. The
    threshold is `max_cer_by_dataset[dataset]` when the dataset is listed there, else
    `max_cer_by_lang[language]` when the language is listed there, else `max_cer`. A checked rollout whose
    CER against the reference text exceeds the threshold is rejected.

    Attributes:
        max_cer: Default CER threshold, or `None` to check only the languages in `max_cer_by_lang`.
        max_cer_by_lang: Per-language thresholds; a `None` value turns the rule off for that language.
        max_cer_by_dataset: Per-dataset thresholds, which take precedence over the language thresholds; a
            `None` value turns the rule off for that dataset.
        datasets: Dataset names (the `Dataset:` field of the speaker tag) to check, or `None` for all.
        rejection_weight: Loss weight of rejected rollouts, or `None` to keep their weight.
        normalize_transcript: Whether to apply the text normalizer of the rollout language to the ASR
            transcript before computing CER, so that written forms such as digits match the spoken-form
            reference text.
        strip_diacritics_languages: Languages whose transcript and reference lose their nonspacing marks
            (and the Arabic tatweel) before computing CER, e.g. Arabic, whose references may be vocalized
            while ASR transcripts are not.
        router_cfg: `RewardASRRouter` configuration (`language_routes` and `backends`).
    """

    max_cer: Optional[float]
    max_cer_by_lang: dict[str, Optional[float]]
    max_cer_by_dataset: dict[str, Optional[float]]
    datasets: Optional[frozenset[str]]
    rejection_weight: Optional[float]
    normalize_transcript: bool
    strip_diacritics_languages: frozenset[str]
    router_cfg: dict[str, Any]

    @classmethod
    def create_from_config(
        cls,
        cfg: Optional[Mapping[str, Any]],
        default_rejection_weight: Optional[float],
    ) -> Optional["ASRRejectionParams"]:
        """Create the rule settings from the `asr_rejection` model configuration.

        Args:
            cfg: `asr_rejection` configuration as a plain mapping, or `None` to disable the rule.
            default_rejection_weight: Loss weight used when `cfg` sets no `rejection_weight`.

        Returns:
            Rule settings, or `None` when the rule is disabled.

        Raises:
            ValueError: If a threshold or weight is out of range, no threshold is set, or a thresholded
                language has no ASR route.
        """
        if cfg is None:
            return None

        router_cfg = {k: cfg[k] for k in ("language_routes", "backends") if k in cfg}
        routes = dict(router_cfg.get("language_routes") or {})
        if not routes:
            raise ValueError("`asr_rejection.language_routes` must map at least one language to an ASR backend.")

        max_cer = cfg.get("max_cer", 0.0)
        max_cer_by_lang = dict(cfg.get("max_cer_by_lang") or {})
        max_cer_by_dataset = dict(cfg.get("max_cer_by_dataset") or {})
        thresholds = [
            ("max_cer", max_cer),
            *((f"max_cer_by_lang.{k}", v) for k, v in max_cer_by_lang.items()),
            *((f"max_cer_by_dataset.{k}", v) for k, v in max_cer_by_dataset.items()),
        ]
        for name, value in thresholds:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"`asr_rejection.{name}` must be in the range [0, 1] or `None`.")

        if all(value is None for _, value in thresholds):
            raise ValueError(
                "`asr_rejection` sets no CER threshold; set `max_cer`, `max_cer_by_lang` or `max_cer_by_dataset`."
            )

        unrouted = sorted(k for k, v in max_cer_by_lang.items() if v is not None and k not in routes)
        if unrouted:
            raise ValueError(
                f"`asr_rejection.max_cer_by_lang` sets thresholds for languages without an ASR route: {unrouted}."
            )

        rejection_weight = cfg.get("rejection_weight", default_rejection_weight)
        if rejection_weight is not None and not (0.0 <= rejection_weight <= 1.0):
            raise ValueError("`asr_rejection.rejection_weight` must be in the range [0, 1] or `None`.")

        datasets = cfg.get("datasets")

        return cls(
            max_cer=max_cer,
            max_cer_by_lang=max_cer_by_lang,
            max_cer_by_dataset=max_cer_by_dataset,
            datasets=frozenset(datasets) if datasets is not None else None,
            rejection_weight=rejection_weight,
            normalize_transcript=bool(cfg.get("normalize_transcript", False)),
            strip_diacritics_languages=frozenset(cfg.get("strip_diacritics_languages") or ()),
            router_cfg=router_cfg,
        )

    def threshold_for(self, language: str, dataset_name: str) -> Optional[float]:
        """Return the CER threshold for one rollout, or `None` when the rollout is not checked."""
        if self.datasets is not None and dataset_name not in self.datasets:
            return None
        if language not in self.router_cfg["language_routes"]:
            return None
        if dataset_name in self.max_cer_by_dataset:
            return self.max_cer_by_dataset[dataset_name]
        return self.max_cer_by_lang.get(language, self.max_cer)


def strip_diacritics(text: str) -> str:
    """Remove nonspacing marks and the Arabic tatweel from NFC-normalized text."""
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn" and ch != "\u0640")


def apply_asr_rejection(
    teacher_output: TeacherOutput,
    batch: dict[str, Any],
    item_indices: list[int],
    cers: list[float],
    thresholds: list[float],
    transcripts: list[str],
    rejection_weight: Optional[float],
) -> TeacherOutput:
    """Reject checked teacher rollouts whose CER exceeds their threshold.

    Args:
        teacher_output: Finalized teacher-rollout predictions and metadata.
        batch: Batch containing optional `dataset_names`, `languages` and `raw_texts`.
        item_indices: Batch indices of the checked rollouts.
        cers: CER of each checked rollout.
        thresholds: CER threshold of each checked rollout.
        transcripts: Normalized ASR transcript of each checked rollout, used for logging.
        rejection_weight: Optional loss weight assigned to rejected items.

    Returns:
        Teacher output updated with ASR rejection flags and sample weights.
    """
    rejected_indices = [i for i, cer, t in zip(item_indices, cers, thresholds) if cer > t]
    if not rejected_indices:
        return teacher_output

    device = teacher_output.lens.device
    should_reject = torch.zeros_like(teacher_output.lens, dtype=torch.bool)
    should_reject[rejected_indices] = True

    if teacher_output.rejected is None:
        teacher_output.rejected = should_reject
    else:
        teacher_output.rejected = teacher_output.rejected | should_reject

    if rejection_weight is not None:
        if teacher_output.sample_weights is None:
            sample_weights = torch.ones_like(teacher_output.lens, dtype=torch.float, device=device)
        else:
            sample_weights = teacher_output.sample_weights.clone()
        teacher_output.sample_weights = torch.where(
            condition=should_reject,
            input=torch.full_like(sample_weights, rejection_weight),
            other=sample_weights,
        )

    dataset_names = batch.get("dataset_names")
    languages = batch.get("languages")
    texts = batch.get("raw_texts")
    by_index = {i: (cer, t, hyp) for i, cer, t, hyp in zip(item_indices, cers, thresholds, transcripts)}

    for item_idx in rejected_indices:
        cer, threshold, hyp = by_index[item_idx]
        dataset_name = dataset_names[item_idx] if dataset_names is not None else "Unknown"
        language = languages[item_idx] if languages is not None else "Unknown"
        text = texts[item_idx] if texts is not None else ""
        logging.info(
            f"Item {item_idx} rejected by ASR (CER {cer:.3f} > {threshold}, dataset: {dataset_name}, "
            f"language: {language}); text={text[:160]!r} asr={hyp[:160]!r}."
        )
    return teacher_output
