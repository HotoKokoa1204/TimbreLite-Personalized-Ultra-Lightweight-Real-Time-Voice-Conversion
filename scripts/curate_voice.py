"""CLI tool to curate audio recordings into clean human speech datasets.

Automatically removes mechanical keyboard clicks, mouse snaps, and background noise
using dual acoustic (Crest Factor) screening and neural Whisper verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

from timbre_lite.data.curation import CurationConfig, curate_source_speaker_audio
from timbre_lite.data.speech_filter import SpeechVerifier

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma"}


def clean_manifest(
    manifest_path: Path,
    output_path: Path | None = None,
    mode: str = "hybrid",
    max_crest: float = 12.0,
) -> Path:
    """Filter an existing manifest file to remove non-speech/click entries."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Auditing existing manifest: {manifest_path} ({len(items)} items)...")

    verifier = SpeechVerifier(mode=mode, max_crest_factor=max_crest)

    audio_chunks = []
    sample_rates = []
    for item in items:
        p = Path(item.get("audio_16k") or item.get("audio_24k"))
        wav, sr = sf.read(p)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        audio_chunks.append(wav)
        sample_rates.append(sr)

    verifications = verifier.verify_batch(
        audio_chunks, sample_rates[0] if sample_rates else 16000
    )

    clean_items = []
    rejected_clicks = 0
    rejected_noise = 0

    for item, v in zip(items, verifications):
        if v.is_speech:
            clean_items.append(item)
        elif "crest" in v.reason or "click" in v.reason:
            rejected_clicks += 1
        else:
            rejected_noise += 1

    dest = output_path or manifest_path.with_name(manifest_path.stem + "_clean.json")
    dest.write_text(
        json.dumps(clean_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_sec = sum(x["duration_sec"] for x in clean_items)
    print("\n" + "=" * 60)
    print("MANIFEST CLEANING REPORT:")
    print(f"Original items: {len(items)}")
    print(f"Verified speech: {len(clean_items)} ({total_sec / 60:.2f} min)")
    print(f"Rejected clicks: {rejected_clicks}")
    print(f"Rejected noise / silence: {rejected_noise}")
    print(f"Clean manifest saved to: {dest}")
    print("=" * 60 + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate audio recordings into clean human speech datasets."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/my_voice/錄製.m4a",
        help="Path to raw audio file or directory containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/my_voice/processed",
        help="Destination directory for processed 24k/16k audio and manifests.",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "acoustic", "whisper", "none"],
        default="acoustic",
        help="Speech verification mode (hybrid=acoustic+whisper, acoustic=fast CPU).",
    )
    parser.add_argument(
        "--max-crest",
        type=float,
        default=11.5,
        help="Maximum crest factor threshold (mechanical clicks typically > 15).",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=0.008,
        help="Minimum RMS energy threshold to reject room noise and quiet pauses.",
    )
    parser.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=0.15,
        help="Minimum fraction of voiced frames required to reject clicks.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new samples to existing manifests instead of overwriting.",
    )
    parser.add_argument(
        "--clean-manifest",
        type=str,
        default=None,
        help="Clean an existing manifest file instead of curating from raw audio.",
    )

    args = parser.parse_args()

    # Handle cleaning existing manifest directly
    if args.clean_manifest:
        clean_manifest(
            Path(args.clean_manifest), mode=args.mode, max_crest=args.max_crest
        )
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Collect audio files
    if input_path.is_dir():
        audio_files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in AUDIO_EXTS]
        )
        if not audio_files:
            print(f"Error: No supported audio files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(audio_files)} audio files in {input_path}")
    else:
        audio_files = [input_path]

    filter_enabled = args.mode != "none"
    cfg = CurationConfig(
        filter_speech=filter_enabled,
        speech_filter_mode=args.mode if filter_enabled else "acoustic",
        max_crest_factor=args.max_crest,
        min_rms=args.min_rms,
        min_voiced_ratio=args.min_voiced_ratio,
    )

    print(
        f"\nCurating audio using mode: '{args.mode}' "
        f"(max_crest={args.max_crest}, min_rms={args.min_rms}, "
        f"min_voiced={args.min_voiced_ratio})..."
    )
    output_dir = Path(args.output_dir)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i + 1}/{len(audio_files)}] Processing: {audio_file.name}...")
        should_append = args.append or (i > 0)
        train_records, val_records = curate_source_speaker_audio(
            source_audio_path=audio_file,
            output_dir=output_dir,
            config=cfg,
            append=should_append,
        )

    all_items = train_records + val_records
    total_speech = len(all_items)
    train_dur = sum(r["duration_sec"] for r in train_records)
    val_dur = sum(r["duration_sec"] for r in val_records)
    total_dur = train_dur + val_dur

    print("\n" + "=" * 65)
    print("AUDIO CURATION COMPLETED SUCCESSFULLY!")
    print(f" - Clean Speech Segments : {total_speech}")
    print(f" - Total Speech Duration : {total_dur / 60:.2f} min ({total_dur:.1f}s)")
    print(
        f" - Training Split        : {len(train_records)} utts ({train_dur / 60:.2f}m)"
    )
    print(f" - Validation Split      : {len(val_records)} utts ({val_dur / 60:.2f}m)")
    print(f" - Manifest Location     : {output_dir / 'train_manifest.json'}")
    print("=" * 65)
    print("\nNext step: Extract teacher features or train Stage 1 Cleanser directly:")
    manifest_target = output_dir / "train_manifest.json"
    print(
        f"python -m timbre_lite.training.train_stage1 "
        f"--train-manifest {manifest_target} --epochs 100\n"
    )


if __name__ == "__main__":
    main()
