"""Pure human voice activity and acoustic verification pipeline.

Discriminates between genuine human speech and non-speech acoustic artifacts
such as mechanical keyboard clicks, mouse snaps, breath bursts, and background noise.
Supports ultra-fast acoustic feature screening and neural Whisper-tiny verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchaudio.functional as taf


@dataclass
class SpeechVerificationResult:
    """Result of speech verification for an audio segment.

    Attributes:
        is_speech: True if the segment is verified as genuine human speech.
        confidence: Verification confidence score between 0.0 and 1.0.
        reason: Explanation of classification result.
        crest_factor: Peak-to-RMS ratio of the waveform.
        rms: Root-mean-square amplitude of the waveform.
        transcript: Recognized speech transcript (empty if acoustic-only).
    """

    is_speech: bool
    confidence: float
    reason: str
    crest_factor: float
    rms: float
    transcript: str = ""


class SpeechVerifier:
    """Verifies whether audio segments contain genuine human speech.

    Combines ultra-fast acoustic metrics (crest factor, RMS, zero-crossing rate)
    with optional neural ASR validation (Whisper-tiny) to reject mechanical
    keyboard clatter, mouse clicks, and ambient noise.
    """

    # Common Whisper hallucination strings triggered by non-speech noise
    HALLUCINATION_PATTERNS: tuple[str, ...] = (
        "see you next time",
        "subtitles by",
        "thank you",
        "watching",
        "mbc",
        "amara.org",
        "bye",
        "you",
        "...",
    )

    def __init__(
        self,
        mode: str = "hybrid",
        max_crest_factor: float = 12.0,
        min_rms: float = 0.003,
        min_voiced_ratio: float = 0.0,
        whisper_model_name: str = "openai/whisper-tiny",
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize SpeechVerifier.

        Args:
            mode: Verification mode: 'acoustic', 'whisper', or 'hybrid'.
            max_crest_factor: Maximum allowed peak/RMS ratio (clicks usually > 15).
            min_rms: Minimum required RMS energy to avoid near-silence.
            min_voiced_ratio: Min fraction of voiced frames required (0.0 to disable).
            whisper_model_name: HuggingFace model identifier for Whisper.
            device: Target compute device for neural verification.
        """
        self.mode = mode
        self.max_crest_factor = max_crest_factor
        self.min_rms = min_rms
        self.min_voiced_ratio = min_voiced_ratio
        self.whisper_model_name = whisper_model_name
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._whisper_model: Any = None
        self._whisper_processor: Any = None

    def _ensure_whisper_loaded(self) -> tuple[Any, Any]:
        """Lazy-load Whisper model and processor on target device."""
        if self._whisper_model is None or self._whisper_processor is None:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            self._whisper_processor = WhisperProcessor.from_pretrained(
                self.whisper_model_name
            )
            self._whisper_model = (
                WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)
                .to(self.device)
                .eval()
            )
        return self._whisper_model, self._whisper_processor

    def verify_acoustic(
        self, audio: np.ndarray, fs: int
    ) -> tuple[bool, float, str, float, float]:
        """Verify audio segment based strictly on fast physical acoustic metrics.

        Args:
            audio: 1D numpy array of audio samples.
            fs: Sampling rate in Hz.

        Returns:
            Tuple of (is_speech, confidence, reason, crest_factor, rms).
        """
        if len(audio) == 0:
            return False, 0.0, "empty_buffer", 0.0, 0.0

        rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
        peak = float(np.max(np.abs(audio)))
        crest = float(peak / (rms + 1e-9))

        if rms < self.min_rms:
            return False, 0.1, "below_min_rms", crest, rms

        # Sharp impulsive clicks (mechanical switches, mouse clicks)
        if crest > self.max_crest_factor:
            return (
                False,
                0.15,
                f"high_crest_click (crest={crest:.1f} > {self.max_crest_factor})",
                crest,
                rms,
            )

        if self.min_voiced_ratio > 0.0:
            from timbre_lite.modules.f0 import CausalF0Tracker, F0TrackerConfig

            if (
                not hasattr(self, "_f0_tracker")
                or self._f0_tracker is None
                or getattr(self._f0_tracker, "sr", None) != fs
            ):
                tracker_cfg = F0TrackerConfig(sample_rate=fs, f_min=65.0, f_max=380.0)
                self._f0_tracker = CausalF0Tracker(tracker_cfg)

            _, _, _, vuv = self._f0_tracker.process_utterance(audio)
            voiced_ratio = float(vuv.mean())
            if voiced_ratio < self.min_voiced_ratio:
                reason = (
                    f"low_voicing (voiced_ratio={voiced_ratio:.2f} "
                    f"< {self.min_voiced_ratio:.2f})"
                )
                return (
                    False,
                    0.20,
                    reason,
                    crest,
                    rms,
                )

        # Voiced speech check: standard voice crest is typically 3.0 to 9.0
        confidence = float(
            np.clip(1.0 - (crest / (self.max_crest_factor * 1.5)), 0.5, 0.95)
        )
        return True, confidence, "passed_acoustic_check", crest, rms

    def is_hallucination_or_repetition(self, text: str) -> tuple[bool, str]:
        """Detect whether ASR transcript is a hallucination or repetitive loop.

        Args:
            text: Decoded transcript string.

        Returns:
            Tuple of (is_bad, reason_description).
        """
        clean_text = text.strip()
        if not clean_text:
            return True, "empty_transcript"

        lower_text = clean_text.lower()
        for pattern in self.HALLUCINATION_PATTERNS:
            if pattern in lower_text:
                return True, f"hallucination_pattern_'{pattern}'"

        # Character repetition loop detection (e.g. '小小小小小小...' or 'pppppp')
        chars = [c for c in clean_text if not c.isspace()]
        if len(chars) >= 5:
            unique_chars = len(set(chars))
            diversity = unique_chars / len(chars)
            if diversity < 0.25 and unique_chars <= 3:
                return True, f"repetition_loop_diversity_{diversity:.2f}"

        return False, "valid_speech"

    def verify_batch(
        self,
        audio_list: list[np.ndarray],
        fs: int,
    ) -> list[SpeechVerificationResult]:
        """Verify multiple audio segments with optional neural batching.

        Args:
            audio_list: List of 1D numpy arrays.
            fs: Audio sampling rate in Hz.

        Returns:
            List of SpeechVerificationResult objects corresponding to audio_list.
        """
        results: list[SpeechVerificationResult] = []
        if not audio_list:
            return results

        # 1. Acoustic preliminary screening
        acoustic_passes: list[int] = []
        for idx, wav in enumerate(audio_list):
            is_ok, conf, reason, crest, rms = self.verify_acoustic(wav, fs)
            if self.mode == "acoustic":
                results.append(
                    SpeechVerificationResult(
                        is_speech=is_ok,
                        confidence=conf,
                        reason=reason,
                        crest_factor=crest,
                        rms=rms,
                        transcript="",
                    )
                )
            else:
                # Placeholder, will be updated by Whisper if passed
                results.append(
                    SpeechVerificationResult(
                        is_speech=is_ok,
                        confidence=conf,
                        reason=reason,
                        crest_factor=crest,
                        rms=rms,
                        transcript="",
                    )
                )
                if is_ok or self.mode == "whisper":
                    acoustic_passes.append(idx)

        if self.mode == "acoustic" or not acoustic_passes:
            return results

        # 2. Neural verification on surviving candidate segments
        model, processor = self._ensure_whisper_loaded()

        # Prepare 16kHz resampled waveforms for Whisper
        whisper_inputs: list[np.ndarray] = []
        for idx in acoustic_passes:
            wav = audio_list[idx]
            if fs != 16000:
                t_wav = torch.from_numpy(wav).float()
                t_16k = taf.resample(t_wav, orig_freq=fs, new_freq=16000)
                wav_16k = t_16k.numpy()
            else:
                wav_16k = wav
            whisper_inputs.append(wav_16k)

        # Batch inference in chunks of 16
        batch_size = 16
        for b_start in range(0, len(acoustic_passes), batch_size):
            b_indices = acoustic_passes[b_start : b_start + batch_size]
            b_waves = whisper_inputs[b_start : b_start + batch_size]

            inputs = processor(
                b_waves,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).input_features.to(self.device)

            with torch.no_grad():
                gen_ids = model.generate(
                    inputs,
                    language="zh",
                    task="transcribe",
                    max_new_tokens=64,
                )

            transcripts = processor.batch_decode(gen_ids, skip_special_tokens=True)

            for original_idx, text in zip(b_indices, transcripts):
                text_clean = text.strip()
                is_bad, bad_reason = self.is_hallucination_or_repetition(text_clean)

                res = results[original_idx]
                if is_bad:
                    results[original_idx] = SpeechVerificationResult(
                        is_speech=False,
                        confidence=0.2,
                        reason=bad_reason,
                        crest_factor=res.crest_factor,
                        rms=res.rms,
                        transcript=text_clean,
                    )
                else:
                    results[original_idx] = SpeechVerificationResult(
                        is_speech=True,
                        confidence=0.95,
                        reason="verified_speech",
                        crest_factor=res.crest_factor,
                        rms=res.rms,
                        transcript=text_clean,
                    )

        return results
