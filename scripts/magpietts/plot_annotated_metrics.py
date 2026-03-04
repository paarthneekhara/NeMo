#!/usr/bin/env python3
"""
Plot histograms and summary statistics for annotated CER/SSIM metrics
from Lhotse shard cuts produced by annotate_lhotse_shards_with_cer_ssim.py.

Usage:
    python plot_annotated_metrics.py \
        --cuts-dir /path/to/cuts_with_ipa_annotated \
        --output-dir /path/to/annotated_metrics

    # Optionally filter by a CER threshold to see the "good" vs "bad" split:
    python plot_annotated_metrics.py \
        --cuts-dir /path/to/cuts_with_ipa_annotated \
        --output-dir /path/to/annotated_metrics \
        --cer-threshold 0.5
"""

import argparse
import glob
import gzip
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(cuts_dir: str):
    """Read all cuts.*.jsonl.gz and extract annotated_cer / annotated_ssim."""
    pattern = os.path.join(cuts_dir, "cuts.*.jsonl.gz")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No cut shards found matching {pattern}")

    cer_values = []
    ssim_values = []
    total = 0
    skipped = 0

    for path in shard_paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                total += 1
                cut = json.loads(line)
                custom = (
                    cut.get("supervisions", [{}])[0].get("custom", {})
                    if cut.get("supervisions")
                    else {}
                )
                cer = custom.get("annotated_cer")
                ssim = custom.get("annotated_ssim")
                if cer is None or ssim is None:
                    skipped += 1
                    continue
                if cer < 0 or ssim < 0:
                    skipped += 1
                    continue
                cer_values.append(cer)
                ssim_values.append(ssim)

    print(f"Loaded {len(cer_values)} annotated cuts from {len(shard_paths)} shards "
          f"({skipped} skipped out of {total} total).")
    return np.array(cer_values), np.array(ssim_values)


def plot_histogram(values, title, xlabel, filepath, bins=50, vline=None, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    mean_val = np.mean(values)
    median_val = np.median(values)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.4f}")
    ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.5, label=f"Median = {median_val:.4f}")
    if vline is not None:
        ax.axvline(vline, color="green", linestyle="-.", linewidth=1.5, label=f"Threshold = {vline:.2f}")

    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Plot CER/SSIM histograms from annotated Lhotse shards.")
    parser.add_argument("--cuts-dir", type=str, required=True,
                        help="Directory with annotated cuts.*.jsonl.gz files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save plots and summary.")
    parser.add_argument("--cer-threshold", type=float, default=None,
                        help="Optional CER threshold to mark on histogram and report split.")
    parser.add_argument("--ssim-threshold", type=float, default=None,
                        help="Optional SSIM threshold to mark on histogram and report split.")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of histogram bins.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cer, ssim = load_metrics(args.cuts_dir)
    if len(cer) == 0:
        print("No valid annotated cuts found. Exiting.")
        return

    # --- Histograms ---
    plot_histogram(cer, "CER Distribution", "Character Error Rate (CER)",
                   str(out / "cer_histogram.png"), bins=args.bins,
                   vline=args.cer_threshold, color="steelblue")

    plot_histogram(ssim, "Speaker Similarity (SSIM) Distribution", "SSIM (cosine similarity)",
                   str(out / "ssim_histogram.png"), bins=args.bins,
                   vline=args.ssim_threshold, color="darkorange")

    # --- Summary stats ---
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    lines = []
    lines.append(f"Dataset: {args.cuts_dir}")
    lines.append(f"Total annotated cuts: {len(cer)}")
    lines.append("")
    lines.append("=== CER Statistics ===")
    lines.append(f"  Mean:   {np.mean(cer):.4f}")
    lines.append(f"  Median: {np.median(cer):.4f}")
    lines.append(f"  Std:    {np.std(cer):.4f}")
    lines.append(f"  Min:    {np.min(cer):.4f}")
    lines.append(f"  Max:    {np.max(cer):.4f}")
    for p in percentiles:
        lines.append(f"  P{p:02d}:    {np.percentile(cer, p):.4f}")
    if args.cer_threshold is not None:
        below = np.sum(cer <= args.cer_threshold)
        lines.append(f"  CER <= {args.cer_threshold:.2f}: {below}/{len(cer)} ({below/len(cer)*100:.1f}%)")

    lines.append("")
    lines.append("=== SSIM Statistics ===")
    lines.append(f"  Mean:   {np.mean(ssim):.4f}")
    lines.append(f"  Median: {np.median(ssim):.4f}")
    lines.append(f"  Std:    {np.std(ssim):.4f}")
    lines.append(f"  Min:    {np.min(ssim):.4f}")
    lines.append(f"  Max:    {np.max(ssim):.4f}")
    for p in percentiles:
        lines.append(f"  P{p:02d}:    {np.percentile(ssim, p):.4f}")
    if args.ssim_threshold is not None:
        above = np.sum(ssim >= args.ssim_threshold)
        lines.append(f"  SSIM >= {args.ssim_threshold:.2f}: {above}/{len(ssim)} ({above/len(ssim)*100:.1f}%)")

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = out / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
