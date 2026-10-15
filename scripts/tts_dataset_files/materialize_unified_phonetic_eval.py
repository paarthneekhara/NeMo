#!/usr/bin/env python3
"""Create evaluation manifests with full NeMo phonetic text for configured languages."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    normalize_phonemizer_language,
    phonemize_text_with_tokenizer,
)
from nemo.core.classes.common import safe_instantiate


def resolve_from_base(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", type=Path, required=True)
    parser.add_argument("--datasets-base-path", type=Path, required=True)
    parser.add_argument("--phonemizer-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_config = json.loads(args.input_config.read_text(encoding="utf-8"))
    phonemizer_config = OmegaConf.load(args.phonemizer_config)
    phonemizer_configs = phonemizer_config.get("phonemizers", {})
    full_only_languages = {
        normalize_phonemizer_language(language)
        for language in phonemizer_config.get("full_only_languages", [])
    }
    phonemizers = {}
    counts: Counter[str] = Counter()
    samples = defaultdict(list)
    output_config = {}
    manifests_dir = args.output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, metadata in dataset_config.items():
        language = normalize_phonemizer_language(metadata.get("language"))
        input_manifest = resolve_from_base(metadata["manifest_path"], args.datasets_base_path)
        output_manifest = manifests_dir / f"{dataset_name}.jsonl"
        phonemizer = None
        if language in phonemizer_configs:
            if language not in phonemizers:
                phonemizers[language] = safe_instantiate(phonemizer_configs[language])
            phonemizer = phonemizers[language]

        with input_manifest.open(encoding="utf-8") as input_stream, output_manifest.open(
            "w", encoding="utf-8"
        ) as output_stream:
            for line_number, line in enumerate(input_stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                original_text = record.get("original_text") or record.get("text")
                if not isinstance(original_text, str) or not original_text.strip():
                    raise ValueError(f"Missing text in {input_manifest}:{line_number}")
                record["original_text"] = original_text

                if phonemizer is not None:
                    source_text = record.get("normalized_text") or original_text
                    phonetic_text, _ = phonemize_text_with_tokenizer(
                        text=source_text,
                        phonemizer=phonemizer,
                        language=language,
                        full_only=language in full_only_languages,
                    )
                    if not phonetic_text or phonetic_text == source_text:
                        counts[f"{language}:unchanged"] += 1
                    else:
                        record["text"] = phonetic_text
                        counts[f"{language}:phonemized"] += 1
                        if len(samples[language]) < 3:
                            samples[language].append(
                                {
                                    "dataset": dataset_name,
                                    "original_text": original_text,
                                    "phonetic_text": phonetic_text,
                                }
                            )
                else:
                    counts[f"{language}:regular"] += 1

                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        derived_metadata = dict(metadata)
        derived_metadata["manifest_path"] = str(output_manifest)
        output_config[dataset_name] = derived_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "all_full_nemo_phonetic.json"
    config_path.write_text(json.dumps(output_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "materialization_summary.json").write_text(
        json.dumps(
            {
                "input_config": str(args.input_config),
                "phonemizer_config": str(args.phonemizer_config),
                "counts": dict(sorted(counts.items())),
                "samples": dict(sorted(samples.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(config_path)
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
