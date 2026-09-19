"""Formal specification and contract for streaming neural audio codecs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CodecContract:
    """Architectural contract and parameters for a streaming neural codec.

    Attributes:
        name: Canonical identifier of the codec model.
        sample_rate: Native audio sample rate in Hz (e.g. 24000 or 16000).
        frame_samples: Number of audio samples consumed per streaming frame hop.
        frame_ms: Duration of one frame in milliseconds.
        latent_dim: Number of channels in continuous latent representation.
        is_causal: Whether architecture enforces zero-lookahead causality.
        quantizer_type: Type of vector quantizer (e.g. 'RSVQ', 'RVQ', 'none').
        weights_available: Whether checkpoint weights are publicly accessible.
    """

    name: str
    sample_rate: int
    frame_samples: int
    frame_ms: float
    latent_dim: int
    is_causal: bool
    quantizer_type: str
    weights_available: bool

    def validate_audio_chunk_shape(self, num_samples: int) -> bool:
        """Validate whether an incoming audio chunk matches the frame contract.

        Args:
            num_samples: Number of samples in the input chunk.

        Returns:
            True if num_samples equals frame_samples, False otherwise.
        """
        return num_samples == self.frame_samples
