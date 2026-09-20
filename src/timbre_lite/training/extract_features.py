"""Offline ContentVec teacher feature extraction and caching utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from timbre_lite.data.loader import load_audio
from timbre_lite.distillation.alignment import align_teacher_features
from timbre_lite.distillation.teacher import ContentVecTeacher


def extract_and_cache_features(
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device | str | None = None,
    teacher: ContentVecTeacher | None = None,
) -> Path:
    """Extract and cache 75Hz ContentVec representations for all manifest items.

    Reads 16kHz audio from the given manifest, extracts 50Hz representations via
    ContentVec, temporally aligns them to 75Hz (hop = 320 at 24kHz), and saves
    each tensor as a standalone .pt file. Writes an updated manifest containing
    'teacher_features_path' for fast zero-overhead training.

    Args:
        manifest_path: Path to input train or val manifest JSON.
        output_dir: Target directory to store extracted .pt tensors.
        device: Target torch compute device (defaults to CUDA if available).
        teacher: Optional pre-instantiated ContentVecTeacher module.

    Returns:
        Path to the updated manifest JSON file.
    """
    manifest_p = Path(manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_p, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    if teacher is None:
        teacher = ContentVecTeacher(device=device)

    updated_records: list[dict[str, Any]] = []

    for record in tqdm(records, desc=f"Extracting features ({manifest_p.stem})"):
        sample_id = str(record["id"])
        audio_16k_path = str(record["audio_16k"])
        num_samples_24k = int(record["num_samples_24k"])
        target_frames_75hz = num_samples_24k // 320

        feat_path = out_dir / f"{sample_id}_contentvec.pt"

        if not feat_path.is_file():
            audio_16k, _ = load_audio(audio_16k_path, target_sr=16000)
            with torch.no_grad():
                feat_50hz = teacher.extract_features(audio_16k)
                feat_75hz = align_teacher_features(
                    feat_50hz, target_frames=target_frames_75hz
                )
                # Store shape (768, num_frames)
                tensor_to_save = feat_75hz.squeeze(0).cpu()
                torch.save(tensor_to_save, feat_path)

        rec_copy = dict(record)
        rec_copy["teacher_features_path"] = str(feat_path)
        updated_records.append(rec_copy)

    updated_manifest_path = out_dir / f"{manifest_p.stem}_with_features.json"
    with open(updated_manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, indent=2, ensure_ascii=False)

    return updated_manifest_path


def main() -> None:
    """CLI entrypoint for offline feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and cache offline ContentVec features."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input train or val manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features/my_voice",
        help="Directory to save extracted .pt tensors and updated manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for extraction.",
    )

    args = parser.parse_args()
    updated_path = extract_and_cache_features(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"Features extracted and saved. Updated manifest: {updated_path}")


if __name__ == "__main__":
    main()
