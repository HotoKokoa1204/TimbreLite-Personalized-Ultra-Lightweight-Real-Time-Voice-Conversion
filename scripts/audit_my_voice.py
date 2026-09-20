"""Audit and characterize user gaming audio dataset.

Analyzes acoustic characteristics and runs batched Whisper-tiny ASR
to classify segments into Speech, Transients/Keyboard clicks, and Low-energy noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def compute_audio_features(wav: np.ndarray, sr: int) -> dict[str, float]:
    """Compute acoustic metrics: RMS, peak, crest factor, zero crossing rate."""
    rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
    peak = float(np.max(np.abs(wav)))
    crest = float(peak / (rms + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav)))) / 2.0)
    return {
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "zcr": zcr,
    }


def main() -> None:
    train_manifest_path = Path("data/my_voice/processed/train_manifest.json")
    val_manifest_path = Path("data/my_voice/processed/val_manifest.json")

    manifest = []
    if train_manifest_path.exists():
        manifest.extend(json.loads(train_manifest_path.read_text("utf-8")))
    if val_manifest_path.exists():
        manifest.extend(json.loads(val_manifest_path.read_text("utf-8")))

    print(f"Total utterances to audit: {len(manifest)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper-tiny on {device}...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(
        device
    )
    model.eval()

    results: list[dict[str, Any]] = []

    # Batch process for high throughput
    batch_size = 16
    for i in tqdm(range(0, len(manifest), batch_size), desc="Auditing audio"):
        batch_items = manifest[i : i + batch_size]
        batch_waves = []
        batch_feats = []

        for item in batch_items:
            wav_path = Path(item["audio_16k"])
            if not wav_path.exists():
                wav_path = Path(item["audio_24k"])
            wav, sr = sf.read(wav_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            feats = compute_audio_features(wav, sr)
            batch_feats.append(feats)
            batch_waves.append(wav)

        inputs = processor(
            batch_waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_features.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                inputs,
                language="zh",
                task="transcribe",
                max_new_tokens=64,
            )

        transcripts = processor.batch_decode(gen_ids, skip_special_tokens=True)

        for item, feats, text in zip(batch_items, batch_feats, transcripts):
            text_clean = text.strip()
            # Heuristic speech classification
            # Common Whisper hallucinations on pure noise/keyboard clicks:
            hallucinations = [
                "See you next time",
                "Subtitles by",
                "Thank you",
                "thank you",
                "watching",
                "MBC",
                "Amara.org",
                "you",
                "You",
                "Bye",
                "bye",
            ]
            is_hallucination = any(h in text_clean for h in hallucinations)

            # Keyboard click heuristic: high crest (> 8.0) and
            # very short or hallucinated
            is_click = feats["crest"] > 9.0 and (
                len(text_clean) == 0 or is_hallucination
            )

            category = "speech"
            if len(text_clean) == 0:
                category = "silence_or_ambient"
            elif is_click:
                category = "keyboard_mouse_click"
            elif is_hallucination:
                category = "noise_or_hallucination"

            results.append(
                {
                    "id": item["id"],
                    "audio_24k": item["audio_24k"],
                    "duration_sec": item["duration_sec"],
                    "category": category,
                    "transcript": text_clean,
                    "rms": round(feats["rms"], 4),
                    "crest": round(feats["crest"], 2),
                    "zcr": round(feats["zcr"], 3),
                }
            )

    # Summarize stats
    categories = {}
    total_duration = sum(r["duration_sec"] for r in results)
    category_durations = {}

    for r in results:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1
        category_durations[cat] = category_durations.get(cat, 0.0) + r["duration_sec"]

    out_file = Path("data/my_voice/audit_summary.json")
    out_file.write_text(
        json.dumps(
            {
                "total_utterances": len(results),
                "total_duration_sec": total_duration,
                "categories_count": categories,
                "categories_duration_sec": category_durations,
                "utterances": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY:")
    print(f"Total Utterances: {len(results)}")
    print(f"Total Segment Duration: {total_duration / 60:.2f} minutes")
    for cat, count in categories.items():
        dur = category_durations[cat]
        pct = dur / total_duration * 100
        print(f" - {cat}: {count} ({dur / 60:.2f}m, {pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
