"""Dual-rate audio curation and segmentation pipeline for Source Speaker recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import scipy.signal as signal
import torch
import torchaudio.functional as taf
from tqdm import tqdm

from timbre_lite.data.loader import load_audio, save_wav


@dataclass
class CurationConfig:
    """Configuration parameters for audio curation and segmentation.

    Attributes:
        native_sr: Native input sampling rate (Hz).
        codec_sr: Causal neural codec sampling rate (Hz).
        teacher_sr: Phonetic teacher model sampling rate (Hz).
        hpf_cutoff_hz: High-pass filter cutoff frequency (Hz).
        min_utterance_sec: Minimum speech segment duration (seconds).
        max_utterance_sec: Maximum speech segment duration (seconds).
        silence_split_sec: Silence duration triggering an utterance split (seconds).
        target_loudness_dbfs: Target integrated/RMS loudness in dBFS.
        transient_gate_threshold: Amplitude derivative multiplier for click gating.
        train_ratio: Fraction of utterances allocated to the training split.
        seed: Deterministic random seed for train/val split.
    """

    native_sr: int = 48000
    codec_sr: int = 24000
    teacher_sr: int = 16000
    hpf_cutoff_hz: float = 60.0
    min_utterance_sec: float = 2.0
    max_utterance_sec: float = 8.0
    silence_split_sec: float = 0.4
    target_loudness_dbfs: float = -20.0
    transient_gate_threshold: float = 6.0
    train_ratio: float = 0.9
    seed: int = 42
    filter_speech: bool = True
    speech_filter_mode: str = "hybrid"
    max_crest_factor: float = 12.0


@dataclass
class DualRateSample:
    """A curated speech utterance exported at dual sampling rates.

    Attributes:
        id: Unique utterance identifier.
        audio_24k: 1D tensor at 24kHz for EnCodec streaming.
        audio_16k: 1D tensor at 16kHz for ContentVec teacher extraction.
        duration_sec: Utterance duration in seconds.
    """

    id: str
    audio_24k: torch.Tensor
    audio_16k: torch.Tensor
    duration_sec: float


class AudioSegmenter:
    """Preprocesses raw microphone recordings into clean dual-rate speech segments."""

    def __init__(self, config: CurationConfig | None = None) -> None:
        """Initialize AudioSegmenter.

        Args:
            config: Optional CurationConfig instance.
        """
        self.config = config or CurationConfig()

    def apply_highpass_filter(self, audio: np.ndarray, fs: int) -> np.ndarray:
        """Apply 4th-order Butterworth high-pass filter to eliminate sub-audible rumble.

        Args:
            audio: 1D numpy array of audio samples.
            fs: Audio sampling rate in Hz.

        Returns:
            Filtered 1D numpy array with float32 dtype.
        """
        sos = signal.butter(
            4, self.config.hpf_cutoff_hz, btype="highpass", fs=fs, output="sos"
        )
        filtered = signal.sosfilt(sos, audio)
        return cast(np.ndarray, filtered.astype(np.float32))

    def apply_transient_gate(self, audio: np.ndarray, fs: int) -> np.ndarray:
        """Suppress non-speech transient clicks (keyboard snaps, mouse clicks).

        Args:
            audio: 1D numpy array of audio samples.
            fs: Audio sampling rate in Hz.

        Returns:
            Transient-suppressed 1D numpy array.
        """
        frame_size = int(fs * 0.01)  # 10ms frame
        if len(audio) < frame_size * 2:
            return audio

        # Calculate local frame-based energy
        num_frames = len(audio) // frame_size
        padded_len = num_frames * frame_size
        truncated = audio[:padded_len].reshape(-1, frame_size)
        rms = np.sqrt(np.mean(truncated**2, axis=-1, keepdims=True)) + 1e-8

        # Noise floor estimated as 25th percentile of RMS
        noise_floor = np.percentile(rms, 25)
        speech_thresh = max(noise_floor * 2.5, 0.005)

        clean = audio.copy()
        for i in range(num_frames):
            frame_rms = rms[i, 0]
            idx_start = i * frame_size
            idx_end = idx_start + frame_size
            frame = clean[idx_start:idx_end]
            peak = np.max(np.abs(frame))
            crest = peak / (frame_rms + 1e-8)

            is_quiet = (
                (i > 0 and rms[i - 1, 0] < speech_thresh)
                or (i < num_frames - 1 and rms[i + 1, 0] < speech_thresh)
                or frame_rms < speech_thresh
            )
            if crest > 6.0 and is_quiet:
                outliers = np.abs(frame) > (5.0 * (noise_floor + 1e-8))
                if np.any(outliers):
                    frame[outliers] *= 0.1
                    clean[idx_start:idx_end] = frame

        return clean

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio RMS loudness to target dBFS with peak limiting.

        Args:
            audio: 1D numpy array of audio samples.

        Returns:
            Loudness-normalized 1D numpy array.
        """
        current_rms = np.sqrt(np.mean(audio**2)) + 1e-9
        target_rms = 10.0 ** (self.config.target_loudness_dbfs / 20.0)

        gain = target_rms / current_rms
        scaled = audio * gain

        # Peak limiter to avoid clipping
        max_val = np.max(np.abs(scaled))
        if max_val > 0.99:
            scaled = scaled * (0.99 / max_val)

        return cast(np.ndarray, scaled.astype(np.float32))

    def detect_speech_segments(
        self, audio: np.ndarray, fs: int
    ) -> list[tuple[int, int]]:
        """Identify speech intervals bounded between min and max duration.

        Args:
            audio: 1D numpy array of audio samples.
            fs: Sampling rate in Hz.

        Returns:
            List of (start_sample, end_sample) tuples within [min_sec, max_sec].
        """
        frame_sec = 0.01  # 10ms frames
        frame_len = int(fs * frame_sec)
        num_frames = len(audio) // frame_len
        if num_frames == 0:
            return []

        frames = audio[: num_frames * frame_len].reshape(-1, frame_len)
        rms = np.sqrt(np.mean(frames**2, axis=-1))

        # Dynamic VAD energy threshold based on ambient noise floor
        noise_floor = np.percentile(rms, 20)
        vad_thresh = max(noise_floor * 2.2, 0.003)

        is_speech = rms > vad_thresh
        silence_frames_limit = int(self.config.silence_split_sec / frame_sec)

        raw_segments: list[tuple[int, int]] = []
        in_segment = False
        seg_start = 0
        silence_counter = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_segment:
                    in_segment = True
                    seg_start = max(0, i - 5)  # 50ms pre-pad
                silence_counter = 0
            elif in_segment:
                silence_counter += 1
                if silence_counter >= silence_frames_limit or i == len(is_speech) - 1:
                    in_segment = False
                    seg_end = min(num_frames, i - silence_counter + 5)  # 50ms post-pad
                    start_samp = seg_start * frame_len
                    end_samp = seg_end * frame_len
                    raw_segments.append((start_samp, end_samp))

        # Filter, merge, and split into [min_utterance_sec, max_utterance_sec]
        min_samples = int(self.config.min_utterance_sec * fs)
        max_samples = int(self.config.max_utterance_sec * fs)

        bounded_segments: list[tuple[int, int]] = []
        i = 0
        while i < len(raw_segments):
            start, end = raw_segments[i]
            dur = end - start

            # If segment is too short, try merging with next segment
            while dur < min_samples and i + 1 < len(raw_segments):
                next_start, next_end = raw_segments[i + 1]
                gap = next_start - end
                if gap < int(fs * 0.8) and (next_end - start) <= max_samples:
                    end = next_end
                    dur = end - start
                    i += 1
                else:
                    break

            if dur < min_samples:
                # Discard isolated micro-noise/breaths shorter than minimum
                i += 1
                continue

            if dur <= max_samples:
                bounded_segments.append((start, end))
            else:
                # Subdivide long utterances (> max_samples) at local minimums
                curr_start = start
                while (end - curr_start) > max_samples:
                    target_split = curr_start + int(fs * 5.0)  # Default ~5s
                    search_start = curr_start + min_samples
                    search_end = min(end, curr_start + max_samples)

                    # Search for energy minimum inside search window
                    search_frames_start = search_start // frame_len
                    search_frames_end = search_end // frame_len
                    if search_frames_end > search_frames_start:
                        local_rms = rms[search_frames_start:search_frames_end]
                        min_idx = int(np.argmin(local_rms))
                        best_split = (search_frames_start + min_idx) * frame_len
                    else:
                        best_split = target_split

                    bounded_segments.append((curr_start, best_split))
                    curr_start = best_split

                if (end - curr_start) >= min_samples:
                    bounded_segments.append((curr_start, end))

            i += 1

        return bounded_segments

    def process_utterance(
        self,
        utterance: np.ndarray,
        idx: int,
        prefix: str = "user_utt",
    ) -> DualRateSample:
        """Standardize an utterance into dual-rate format with normalized loudness.

        Args:
            utterance: 1D numpy array of audio at native sampling rate.
            idx: Utterance integer index.
            prefix: Identifier prefix string.

        Returns:
            DualRateSample container holding 24kHz and 16kHz tensors.
        """
        # 1. Loudness normalization to -20 dBFS
        normed = self.normalize_loudness(utterance)
        t_native = torch.from_numpy(normed).to(dtype=torch.float32)

        # 2. Resample to 24kHz (EnCodec) and 16kHz (ContentVec)
        t_24k = taf.resample(
            t_native,
            orig_freq=self.config.native_sr,
            new_freq=self.config.codec_sr,
        )
        t_16k = taf.resample(
            t_native,
            orig_freq=self.config.native_sr,
            new_freq=self.config.teacher_sr,
        )

        duration = len(t_24k) / self.config.codec_sr
        sample_id = f"{prefix}_{idx:04d}"

        return DualRateSample(
            id=sample_id,
            audio_24k=t_24k,
            audio_16k=t_16k,
            duration_sec=duration,
        )


def curate_source_speaker_audio(
    source_audio_path: str | Path,
    output_dir: str | Path,
    config: CurationConfig | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Execute complete curation pipeline on Source Speaker recording.

    Decodes audio, applies high-pass filter, click suppression, VAD segmentation,
    loudness normalization, dual-rate export, and train/val manifest generation.

    Args:
        source_audio_path: Path to raw recording file (e.g. data/my_voice/錄製.m4a).
        output_dir: Base output directory for processed files and manifests.
        config: Optional CurationConfig override.

    Returns:
        Tuple of (train_manifest_records, val_manifest_records).
    """
    cfg = config or CurationConfig()
    segmenter = AudioSegmenter(cfg)
    out_base = Path(output_dir)
    dir_24k = out_base / "24k"
    dir_16k = out_base / "16k"
    dir_24k.mkdir(parents=True, exist_ok=True)
    dir_16k.mkdir(parents=True, exist_ok=True)

    # 1. Load native audio (typically 48kHz mono float32)
    raw_audio, native_sr = load_audio(source_audio_path, target_sr=cfg.native_sr)
    audio_np = raw_audio.numpy()

    # 2. High-pass filter (>60Hz) and transient click gate
    hpf_audio = segmenter.apply_highpass_filter(audio_np, cfg.native_sr)
    gated_audio = segmenter.apply_transient_gate(hpf_audio, cfg.native_sr)

    # 3. VAD segmentation bounded strictly between 2.0s and 8.0s
    segments = segmenter.detect_speech_segments(gated_audio, cfg.native_sr)

    # 4. Filter non-speech, keyboard clicks, and noise if enabled
    valid_segments: list[tuple[int, int]] = []
    if cfg.filter_speech and segments:
        from timbre_lite.data.speech_filter import SpeechVerifier

        verifier = SpeechVerifier(
            mode=cfg.speech_filter_mode,
            max_crest_factor=cfg.max_crest_factor,
        )
        candidate_chunks = [gated_audio[s:e] for s, e in segments]
        verifications = verifier.verify_batch(candidate_chunks, cfg.native_sr)
        valid_segments = [seg for seg, v in zip(segments, verifications) if v.is_speech]
        rejected_count = len(segments) - len(valid_segments)
        print(
            f"Speech verification: {len(valid_segments)}/{len(segments)} "
            f"utterances verified ({rejected_count} rejected as clicks/noise)."
        )
    else:
        valid_segments = segments

    samples: list[DualRateSample] = []
    for idx, (start, end) in enumerate(
        tqdm(valid_segments, desc="Curating utterances"), start=1
    ):
        chunk = gated_audio[start:end]
        sample = segmenter.process_utterance(chunk, idx, prefix="user_utt")
        samples.append(sample)

        # Save dual-rate WAV files
        save_wav(dir_24k / f"{sample.id}.wav", sample.audio_24k, cfg.codec_sr)
        save_wav(dir_16k / f"{sample.id}.wav", sample.audio_16k, cfg.teacher_sr)

    # 4. Generate deterministic 90% train / 10% val manifest
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    split_point = int(len(samples) * cfg.train_ratio)
    train_idx = set(indices[:split_point])

    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []

    for i, s in enumerate(samples):
        record: dict[str, object] = {
            "id": s.id,
            "audio_24k": str(dir_24k / f"{s.id}.wav"),
            "audio_16k": str(dir_16k / f"{s.id}.wav"),
            "duration_sec": round(s.duration_sec, 3),
            "num_samples_24k": len(s.audio_24k),
            "num_samples_16k": len(s.audio_16k),
        }
        if i in train_idx:
            train_records.append(record)
        else:
            val_records.append(record)

    with open(out_base / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)

    with open(out_base / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)

    return train_records, val_records


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/my_voice/錄製.m4a"
    dest = sys.argv[2] if len(sys.argv) > 2 else "data/my_voice/processed"
    print(f"Starting source speaker audio curation from {src} to {dest}...")
    train_r, val_r = curate_source_speaker_audio(src, dest)
    print(
        f"Curation complete! Total: {len(train_r) + len(val_r)} utterances "
        f"({len(train_r)} train, {len(val_r)} val)."
    )
