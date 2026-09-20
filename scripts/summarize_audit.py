"""Print summary of audit results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    path = Path("data/my_voice/audit_summary.json")
    if not path.exists():
        print("Audit summary not found")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    utts = data["utterances"]

    print(f"Total utterances: {len(utts)}")
    print(f"Total duration: {data['total_duration_sec'] / 60:.2f} min\n")

    # Group by types of transcripts
    short_sounds = []
    repetitive_or_hallucinated = []
    clear_gaming_speech = []

    for u in utts:
        text = u["transcript"].strip()

        # Check repetitive character loop (Whisper hallucination on noise/keyboard)
        is_repetitive = len(text) > 5 and len(set(text.replace(" ", ""))) <= 3
        is_hallucinated = any(
            w in text.lower()
            for w in ["thank you", "watching", "subtitles", "mbc", "see you"]
        )

        if len(text) <= 2:
            short_sounds.append(u)
        elif is_repetitive or is_hallucinated:
            repetitive_or_hallucinated.append(u)
        else:
            clear_gaming_speech.append(u)

    dur_speech = sum(u["duration_sec"] for u in clear_gaming_speech)
    dur_short = sum(u["duration_sec"] for u in short_sounds)
    dur_noise = sum(u["duration_sec"] for u in repetitive_or_hallucinated)

    print(f"1. Speech: {len(clear_gaming_speech)} ({dur_speech / 60:.2f}m)")
    print(f"2. Short vocalizations: {len(short_sounds)} ({dur_short / 60:.2f}m)")
    print(f"3. Noise: {len(repetitive_or_hallucinated)} ({dur_noise / 60:.2f}m)")

    print("\n--- Example Clear Gaming Callouts ---")
    for u in clear_gaming_speech[::15][:12]:
        print(f"  [{u['id']}] ({u['duration_sec']}s): {u['transcript']}")

    print("\n--- Example Short Vocalizations / Sounds ---")
    for u in short_sounds[:6]:
        print(f"  [{u['id']}] ({u['duration_sec']}s): '{u['transcript']}'")

    print("\n--- Example Suspected Noise / Hallucination ---")
    for u in repetitive_or_hallucinated[:6]:
        c_val = u["crest"]
        t_val = u["transcript"]
        print(f"  [{u['id']}] ({u['duration_sec']}s, crest={c_val}): '{t_val}'")


if __name__ == "__main__":
    main()
