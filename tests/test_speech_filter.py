"""Unit tests for pure human speech verification and click rejection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from timbre_lite.data.curation import (
    CurationConfig,
    curate_source_speaker_audio,
)
from timbre_lite.data.loader import save_wav
from timbre_lite.data.speech_filter import SpeechVerifier


def test_acoustic_verification_speech_vs_click() -> None:
    """Verify that speech-like signals pass while sharp impulse clicks are rejected."""
    verifier = SpeechVerifier(mode="acoustic", max_crest_factor=12.0, min_rms=0.005)
    fs = 16000

    # 1. Speech-like harmonic wave (sinusoid with smooth envelope, low crest ~1.4 - 3.0)
    t = np.linspace(0, 1.0, fs, endpoint=False)
    speech_wave = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    is_speech, conf, reason, crest, rms = verifier.verify_acoustic(speech_wave, fs)
    assert is_speech is True
    assert crest < 5.0
    assert reason == "passed_acoustic_check"

    # 2. Impulse click (near-silence with sharp mechanical spike, crest > 20.0)
    click_wave = np.random.normal(0, 0.001, fs).astype(np.float32)
    click_wave[fs // 2] = 0.9  # Sharp transient
    click_wave[fs // 2 + 1] = -0.8

    is_speech_click, conf_click, reason_click, crest_click, _ = (
        verifier.verify_acoustic(click_wave, fs)
    )
    assert is_speech_click is False
    assert crest_click > 15.0
    assert "high_crest_click" in reason_click

    # 3. Near silence (below min_rms)
    silent_wave = np.zeros(fs, dtype=np.float32)
    is_speech_silent, _, reason_silent, _, _ = verifier.verify_acoustic(silent_wave, fs)
    assert is_speech_silent is False
    assert reason_silent == "below_min_rms"


def test_hallucination_and_repetition_detection() -> None:
    """Verify detection of Whisper hallucination patterns and repetition loops."""
    verifier = SpeechVerifier(mode="acoustic")

    # Valid speech transcripts
    bad1, _ = verifier.is_hallucination_or_repetition("那時候我剛不要引起了")
    assert bad1 is False

    bad2, _ = verifier.is_hallucination_or_repetition("差不多差不多了")
    assert bad2 is False

    # Common noise hallucinations
    bad3, reason3 = verifier.is_hallucination_or_repetition("See you next time!")
    assert bad3 is True
    assert "hallucination" in reason3

    bad4, reason4 = verifier.is_hallucination_or_repetition("Subtitles by Amara.org")
    assert bad4 is True
    assert "hallucination" in reason4

    # Repetition loops from non-speech noise
    bad5, reason5 = verifier.is_hallucination_or_repetition(
        "小小小小小小小小小小小小小小小小小小小小小小小小"
    )
    assert bad5 is True
    assert "repetition" in reason5

    bad6, reason6 = verifier.is_hallucination_or_repetition("pppppppppppppppppppp")
    assert bad6 is True
    assert "repetition" in reason6


def test_verify_batch_acoustic_mode() -> None:
    """Verify batch acoustic verification produces aligned results."""
    verifier = SpeechVerifier(mode="acoustic", max_crest_factor=10.0)
    fs = 16000

    t = np.linspace(0, 1.0, fs, endpoint=False)
    wave1 = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    click_wave = np.zeros(fs, dtype=np.float32)
    click_wave[100] = 0.8
    click_wave += np.random.normal(0, 0.001, fs).astype(np.float32)

    results = verifier.verify_batch([wave1, click_wave], fs)
    assert len(results) == 2
    assert results[0].is_speech is True
    assert results[1].is_speech is False


def test_curation_pipeline_with_speech_filter() -> None:
    """Verify that curate_source_speaker_audio filters out non-speech segments."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_audio_path = tmp_path / "synthetic_session.wav"
        out_dir = tmp_path / "processed"

        fs = 48000
        # Create 10 seconds:
        # 0.0 - 1.0s: silence
        # 1.0 - 4.5s: speech burst 1 (harmonic audio, 3.5s)
        # 4.5 - 5.5s: silence
        # 5.5 - 8.5s: series of clicks simulating keyboard clatter (isolated spikes)
        # 8.5 - 10.0s: silence
        total_len = fs * 10
        audio = np.zeros(total_len, dtype=np.float32)

        # Segment 1: continuous tone
        t1 = np.arange(int(fs * 3.5)) / fs
        speech_part = 0.4 * np.sin(2 * np.pi * 350 * t1).astype(np.float32)
        audio[fs * 1 : fs * 1 + len(speech_part)] = speech_part

        # Segment 2: impulsive spikes
        click_start = int(fs * 5.5)
        for offset in range(0, int(fs * 2.5), int(fs * 0.3)):
            audio[click_start + offset] = 0.8
            audio[click_start + offset + 1] = -0.7

        save_wav(raw_audio_path, torch.from_numpy(audio), fs)

        # Run curation with acoustic filtering
        cfg = CurationConfig(
            min_utterance_sec=2.0,
            max_utterance_sec=8.0,
            silence_split_sec=0.4,
            filter_speech=True,
            speech_filter_mode="acoustic",
            max_crest_factor=10.0,
        )

        train_records, val_records = curate_source_speaker_audio(
            raw_audio_path, out_dir, config=cfg
        )

        # Only the harmonic speech segment should survive; click segment rejected!
        total_recs = len(train_records) + len(val_records)
        assert total_recs == 1
