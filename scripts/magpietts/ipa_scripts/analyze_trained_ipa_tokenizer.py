#!/usr/bin/env python3
"""Measure tokens per second for an existing IPA tokenizer by language."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

import analyze_ipa_tokenization as analysis


def parse_languages(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return [lang for lang in available if lang != "ja"]

    languages = [lang.strip() for lang in value.split(",") if lang.strip()]
    unknown = sorted(set(languages) - set(available))
    if unknown:
        raise ValueError(f"Unknown languages: {unknown}; available: {available}")
    return languages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure aggregate IPA BPE tokens per second for each language."
    )
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--config", required=True, help="JSON mapping languages to cuts directories")
    parser.add_argument(
        "--test_langs",
        default="all",
        help="Comma-separated languages or 'all'; 'all' excludes Japanese",
    )
    parser.add_argument(
        "--samples_per_lang",
        type=int,
        default=1000,
        help="Number of sampled cuts per language (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path at which to save detailed JSON results",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as config_file:
        cuts_dirs = json.load(config_file)

    languages = parse_languages(args.test_langs, list(cuts_dirs))
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # The v2605 config contains the exact IPA cuts directories.
    analysis.get_ipa_dir = lambda cuts_dir: cuts_dir

    results = []
    for lang in languages:
        pairs = analysis.sample_text_pairs(
            lang,
            cuts_dirs,
            num_samples=args.samples_per_lang,
            seed=args.seed,
        )
        total_duration = sum(pair.duration for pair in pairs)
        total_tokens = sum(len(tokenizer.encode(pair.ipa_text).ids) for pair in pairs)
        tokens_per_second = total_tokens / total_duration if total_duration else 0.0
        results.append(
            {
                "language": lang,
                "samples": len(pairs),
                "duration_seconds": total_duration,
                "tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
            }
        )

    print("language\tsamples\tduration_seconds\ttokens\ttokens_per_second")
    for result in results:
        print(
            f"{result['language']}\t{result['samples']}\t"
            f"{result['duration_seconds']:.2f}\t{result['tokens']}\t"
            f"{result['tokens_per_second']:.4f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
