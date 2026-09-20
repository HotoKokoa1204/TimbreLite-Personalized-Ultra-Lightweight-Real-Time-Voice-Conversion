"""Unit tests for audio curation, filtering, VAD slicing, and normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from timbre_lite.data.curation import (
    AudioSegmenter,
    CurationConfig,
    curate_source_speaker_audio,
)
from timbre_lite.data.loader import load_audio, save_wav


def test_audio_loader_and_saver() -> None:
    """Verify audio save and load roundtrip with soundfile / PyAV."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "test_roundtrip.wav"
        sr = 24000
        duration_sec = 1.0
        t = torch.linspace(0, duration_sec, int(sr * duration_sec))
        orig_audio = 0.5 * torch.sin(2 * np.pi * 440 * t)

        save_wav(wav_path, orig_audio, sr)
        assert wav_path.is_file()

        loaded_audio, loaded_sr = load_audio(wav_path, target_sr=sr)
        assert loaded_sr == sr
        assert loaded_audio.shape == orig_audio.shape
        # Verify waveform similarity
        max_diff = float(torch.max(torch.abs(loaded_audio - orig_audio)))
        assert max_diff < 1e-3


def test_highpass_filter_attenuation() -> None:
    """Verify 4th-order 60Hz Butterworth filter attenuates sub-audible frequencies."""
    config = CurationConfig(hpf_cutoff_hz=60.0)
    segmenter = AudioSegmenter(config)
    fs = 48000
    t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)

    # 20 Hz low rumble vs 500 Hz speech component
    low_rumble = 0.5 * np.sin(2 * np.pi * 20 * t).astype(np.float32)
    mid_speech = 0.5 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

    filtered_low = segmenter.apply_highpass_filter(low_rumble, fs)
    filtered_mid = segmenter.apply_highpass_filter(mid_speech, fs)

    # Low frequency energy should be attenuated by >= 20 dB (factor > 10x)
    low_ratio = float(np.std(filtered_low) / (np.std(low_rumble) + 1e-8))
    mid_ratio = float(np.std(filtered_mid) / (np.std(mid_speech) + 1e-8))

    assert low_ratio < 0.1, f"Low rumble not sufficiently attenuated: {low_ratio}"
    assert mid_ratio > 0.95, f"Mid tone improperly attenuated: {mid_ratio}"


def test_transient_gate_suppression() -> None:
    """Verify transient click gate attenuates sharp spikes in quiet intervals."""
    segmenter = AudioSegmenter()
    fs = 48000
    audio = np.zeros(fs * 2, dtype=np.float32)
    # Add a quiet noise floor
    audio += np.random.normal(0, 0.001, len(audio)).astype(np.float32)
    # Add an isolated sharp click (simulate keyboard click)
    click_pos = int(fs * 1.0)
    audio[click_pos] = 0.8
    audio[click_pos + 1] = -0.7

    gated = segmenter.apply_transient_gate(audio, fs)
    # The click magnitude should be drastically reduced
    assert abs(gated[click_pos]) < 0.2
    assert abs(gated[click_pos + 1]) < 0.2


def test_vad_slicing_bounds_and_silence_split() -> None:
    """Verify utterances are bounded [2.0s, 8.0s] and split on silences > 400ms."""
    config = CurationConfig(
        min_utterance_sec=2.0,
        max_utterance_sec=8.0,
        silence_split_sec=0.4,
    )
    segmenter = AudioSegmenter(config)
    fs = 48000
    # Construct an audio signal:
    # 0.0 - 1.0s: silence
    # 1.0 - 4.5s: speech burst 1 (3.5s)
    # 4.5 - 5.1s: silence (0.6s > 0.4s split threshold)
    # 5.1 - 8.3s: speech burst 2 (3.2s)
    # 8.3 - 10.0s: silence
    total_len = fs * 10
    audio = np.zeros(total_len, dtype=np.float32)
    t1 = np.arange(int(fs * 3.5)) / fs
    t2 = np.arange(int(fs * 3.2)) / fs
    s1 = int(fs * 1.0)
    audio[s1 : s1 + len(t1)] = 0.1 * np.sin(2 * np.pi * 300 * t1)
    s2 = int(fs * 5.1)
    audio[s2 : s2 + len(t2)] = 0.1 * np.sin(2 * np.pi * 400 * t2)

    segments = segmenter.detect_speech_segments(audio, fs)
    assert len(segments) == 2

    for start, end in segments:
        dur = (end - start) / fs
        assert 2.0 <= dur <= 8.0, f"Segment duration {dur}s outside [2.0s, 8.0s]"


def test_loudness_normalization_target_rms() -> None:
    """Verify loudness normalization scales audio to -20 dBFS with peak limiting."""
    config = CurationConfig(target_loudness_dbfs=-20.0)
    segmenter = AudioSegmenter(config)
    fs = 48000
    t = np.arange(fs * 3) / fs
    # Soft audio with RMS around 0.01
    soft_audio = 0.014 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    normed = segmenter.normalize_loudness(soft_audio)
    rms = float(np.sqrt(np.mean(normed**2)))
    # -20 dBFS target is 10^(-20/20) = 0.1
    assert abs(rms - 0.1) < 0.01
    assert np.max(np.abs(normed)) <= 0.99


def test_dual_rate_process_utterance() -> None:
    """Verify dual-rate processor exports synchronized 24kHz and 16kHz tensors."""
    config = CurationConfig(native_sr=48000, codec_sr=24000, teacher_sr=16000)
    segmenter = AudioSegmenter(config)
    fs = 48000
    dur_sec = 3.0
    t = np.arange(int(fs * dur_sec)) / fs
    utterance = 0.05 * np.sin(2 * np.pi * 250 * t).astype(np.float32)

    sample = segmenter.process_utterance(utterance, idx=1, prefix="test")
    assert sample.id == "test_0001"
    assert len(sample.audio_24k) == 24000 * 3
    assert len(sample.audio_16k) == 16000 * 3
    assert abs(sample.duration_sec - 3.0) < 0.01


def test_curate_source_speaker_manifest_split() -> None:
    """Verify complete curation pipeline writes WAVs and adheres to 90/10 split."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_audio_path = tmp_path / "raw_mic.wav"
        out_dir = tmp_path / "curated"

        # Create 15 seconds synthetic audio with multiple speech bursts
        fs = 48000
        audio = np.zeros(fs * 18, dtype=np.float32)
        # 3 speech bursts of 3s each separated by 1s silence
        for b in range(3):
            start = fs * (b * 4 + 1)
            end = start + fs * 3
            t = np.arange(fs * 3) / fs
            audio[start:end] = 0.08 * np.sin(2 * np.pi * 350 * t)

        save_wav(raw_audio_path, torch.from_numpy(audio), fs)

        train_rec, val_rec = curate_source_speaker_audio(
            source_audio_path=raw_audio_path,
            output_dir=out_dir,
            config=CurationConfig(native_sr=fs, train_ratio=0.66, seed=123),
        )

        assert len(train_rec) >= 1
        assert len(val_rec) >= 1
        assert len(train_rec) + len(val_rec) == 3

        # Check files exist
        for r in train_rec + val_rec:
            assert Path(str(r["audio_24k"])).is_file()
            assert Path(str(r["audio_16k"])).is_file()
            assert 24000 * 2 <= int(str(r["num_samples_24k"])) <= 24000 * 8
            assert 16000 * 2 <= int(str(r["num_samples_16k"])) <= 16000 * 8
