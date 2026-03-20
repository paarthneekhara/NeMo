#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import wave


def combine_chunks(input_dir: Path, output_name: str = "chunks_combined.wav") -> Path:
    chunk_paths = sorted(input_dir.glob("chunk_*.wav"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.wav files found in {input_dir}")

    output_path = input_dir / output_name

    with wave.open(str(chunk_paths[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for chunk_path in chunk_paths[1:]:
        with wave.open(str(chunk_path), "rb") as chunk:
            if (
                chunk.getnchannels() != params.nchannels
                or chunk.getsampwidth() != params.sampwidth
                or chunk.getframerate() != params.framerate
                or chunk.getcomptype() != params.comptype
                or chunk.getcompname() != params.compname
            ):
                raise ValueError(f"Chunk format mismatch: {chunk_path}")
            frames.append(chunk.readframes(chunk.getnframes()))

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate chunk_*.wav files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing chunk WAVs")
    parser.add_argument(
        "--output",
        default="chunks_combined.wav",
        help="Output WAV filename (default: chunks_combined.wav)",
    )
    args = parser.parse_args()

    output_path = combine_chunks(args.input_dir, args.output)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
