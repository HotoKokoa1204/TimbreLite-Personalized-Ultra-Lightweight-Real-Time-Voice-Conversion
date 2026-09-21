import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import pipeline


def main() -> None:
    """Transcribe all 50 source validation utterances and cache transcripts."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("PREPARING SOURCE VALIDATION TRANSCRIPTS FOR CER/WER EVALUATION")
    print("=" * 70)

    val_manifest_p = Path("data/my_voice/processed/val_manifest.json")
    assert val_manifest_p.exists(), f"Manifest {val_manifest_p} not found"

    with open(val_manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 50, f"Expected 50 val utterances, got {len(manifest)}"

    out_file = Path("data/my_voice/processed/val_transcripts.json")
    if out_file.exists():
        cached = json.loads(out_file.read_text(encoding="utf-8"))
        if len(cached) == 50 and all(v.get("transcript") for v in cached.values()):
            print(f"Transcripts already cached ({len(cached)} items) in {out_file}.")
            return

    print("Loading Whisper ASR pipeline ('openai/whisper-base')...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device="cpu",
    )

    transcripts: dict[str, dict[str, str]] = {}
    for i, item in enumerate(manifest):
        utt_id = item["id"]
        audio_path = item.get("audio_16k") or item["audio_24k"]
        print(f"[{i+1}/50] Transcribing {utt_id} ({audio_path})...")

        wav, sr = sf.read(str(audio_path))
        result = asr(
            {"raw": wav.astype(np.float32), "sampling_rate": sr},
            generate_kwargs={
                "language": "zh",
                "task": "transcribe",
                "no_repeat_ngram_size": 3,
            },
        )
        text = str(result["text"]).strip()
        print(f"       -> '{text}'")

        transcripts[utt_id] = {
            "id": utt_id,
            "audio_24k": item["audio_24k"],
            "transcript": text,
        }

    # Strict assertion: all 50 entries must have valid transcript
    assert len(transcripts) == 50, f"Expected 50 transcripts, got {len(transcripts)}"
    assert all(len(v["transcript"]) > 0 for v in transcripts.values()), (
        "Found empty transcript in validation set"
    )

    out_file.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSuccessfully saved 50 validated source transcripts to {out_file}")


if __name__ == "__main__":
    main()
