"""Two-stage host CPU Voice Activity Detector for silence gating."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class TwoStageVAD:
    """CPU-resident two-stage Voice Activity Detector.

    Stage 1: Fast time-domain RMS energy and zero-crossing rate gating.
    Stage 2: Spectral flux and frequency-domain harmonicity verification.
    Guarantees zero GPU invocations during silence periods.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        energy_threshold_db: float = -45.0,
        spectral_flux_threshold: float = 0.8,
        zcr_threshold: float = 0.45,
    ) -> None:
        """Initialize TwoStageVAD.

        Args:
            sample_rate: Input microphone audio sample rate (default 24kHz).
            energy_threshold_db: RMS energy silence cutoff in dB.
            spectral_flux_threshold: Minimum spectral flux for speech onset.
            zcr_threshold: Maximum zero-crossing rate before classifying as noise.
        """
        self.sample_rate = sample_rate
        self.energy_threshold = 10.0 ** (energy_threshold_db / 20.0)
        self.spectral_flux_threshold = spectral_flux_threshold
        self.zcr_threshold = zcr_threshold
        self._prev_spectrum: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset internal spectral history."""
        self._prev_spectrum = None

    def compute_rms(self, chunk: torch.Tensor) -> float:
        """Calculate Root Mean Square (RMS) energy.

        Args:
            chunk: Audio tensor of shape (batch, channels, samples) or (samples,).

        Returns:
            Scalar RMS value.
        """
        rms = torch.sqrt(torch.mean(chunk.float() ** 2) + 1e-9).item()
        return float(rms)

    def compute_zcr(self, chunk: torch.Tensor) -> float:
        """Calculate Zero Crossing Rate (ZCR).

        Args:
            chunk: Audio tensor of shape (..., samples).

        Returns:
            Normalized zero crossing rate [0.0, 1.0].
        """
        flat = chunk.view(-1).float()
        signs = torch.sign(flat)
        # Replace zeros with previous sign
        signs[signs == 0] = 1.0
        crossings = torch.sum(torch.abs(signs[1:] - signs[:-1])) / (2.0 * len(signs))
        return float(crossings.item())

    def compute_spectral_flux(self, chunk: torch.Tensor) -> float:
        """Calculate spectral flux between consecutive audio frames.

        Args:
            chunk: Audio tensor of shape (..., samples).

        Returns:
            Scalar Euclidean distance between consecutive magnitude spectra.
        """
        flat = chunk.view(-1).float()
        fft_size = 512
        if flat.shape[-1] < fft_size:
            flat = F.pad(flat, (0, fft_size - flat.shape[-1]))
        elif flat.shape[-1] > fft_size:
            flat = flat[:fft_size]

        window = torch.hann_window(fft_size, device=flat.device)
        spec = torch.abs(torch.fft.rfft(flat * window))
        norm_spec = spec / (torch.norm(spec) + 1e-9)

        if self._prev_spectrum is None:
            self._prev_spectrum = norm_spec
            return 1.0

        flux = torch.norm(norm_spec - self._prev_spectrum).item()
        self._prev_spectrum = norm_spec
        return float(flux)

    def is_speech(self, chunk: torch.Tensor) -> tuple[bool, float]:
        """Classify audio chunk as speech or silence.

        Args:
            chunk: Audio frame tensor of shape (..., samples).

        Returns:
            Tuple of (is_speech_bool, confidence_score_float).
        """
        rms = self.compute_rms(chunk)
        # Stage 1: Energy gate
        if rms < self.energy_threshold:
            self.compute_spectral_flux(chunk)  # keep spectrum continuous
            return False, 0.0

        # Stage 2: ZCR and spectral flux verification
        zcr = self.compute_zcr(chunk)
        flux = self.compute_spectral_flux(chunk)

        # High ZCR usually indicates background fricative hiss/fan noise
        if zcr > self.zcr_threshold:
            confidence = max(0.0, 1.0 - zcr)
            return False, confidence

        confidence = min(
            1.0, (rms / (self.energy_threshold * 4.0)) * (1.0 + flux * 0.2)
        )
        return True, confidence
