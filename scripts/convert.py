"""CLI tool to convert any audio file to target persona voice."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timbre_lite.inference.runner import VoiceConverter


def main() -> None:
    """CLI entrypoint for voice conversion."""
    parser = argparse.ArgumentParser(
        description="Convert input voice audio to target persona."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to source audio file (WAV, MP3, M4A, etc.).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output converted 24kHz WAV file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained Stage 2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Process audio chunk-by-chunk using causal streaming engine.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )

    args = parser.parse_args()

    print(f"Loading VoiceConverter on device: {args.device}...")
    converter = VoiceConverter(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    mode_str = "streaming (320-sample causal)" if args.streaming else "batch sequence"
    print(f"Converting '{args.input}' -> '{args.output}' via {mode_str}...")
    out_path = converter.convert_file(
        input_path=args.input,
        output_path=args.output,
        streaming=args.streaming,
    )
    print(f"Conversion completed successfully: {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
