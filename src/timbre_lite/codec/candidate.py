"""Streaming neural audio codec candidates and unified streaming interfaces."""

from __future__ import annotations

import abc
import typing as tp

import torch
import torch.nn as nn

from timbre_lite.codec.contract import CodecContract
from timbre_lite.codec.state import CodecState


class StreamingCodec(nn.Module, abc.ABC):
    """Abstract base class establishing the uniform causal streaming interface."""

    def __init__(self) -> None:
        """Initialize base streaming codec module."""
        super().__init__()

    @property
    @abc.abstractmethod
    def contract(self) -> CodecContract:
        """Get the architectural specification and contract for this codec."""
        ...

    @abc.abstractmethod
    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> CodecState:
        """Initialize clean streaming state containers.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device for state tensors.

        Returns:
            A clean CodecState container.
        """
        ...

    @abc.abstractmethod
    def encode_chunk(
        self, audio_chunk: torch.Tensor, state: CodecState
    ) -> tuple[torch.Tensor, CodecState]:
        """Encode a single streaming audio chunk into continuous latent space.

        Args:
            audio_chunk: Raw audio tensor of shape (batch, 1, frame_samples).
            state: Previous streaming state.

        Returns:
            Tuple of (latent_chunk, next_state), where latent_chunk has shape
            (batch, latent_dim, 1).
        """
        ...

    @abc.abstractmethod
    def decode_chunk(
        self,
        latent_chunk: torch.Tensor,
        state: CodecState,
        bypass_quantizer: bool = True,
    ) -> tuple[torch.Tensor, CodecState]:
        """Decode a single continuous latent frame back into audio waveform.

        Args:
            latent_chunk: Latent representation of shape (batch, latent_dim, 1).
            state: Previous streaming state.
            bypass_quantizer: If True, feeds continuous latent to decoder.
                If False, routes latent through vector quantizer first.

        Returns:
            Tuple of (audio_chunk, next_state), where audio_chunk has shape
            (batch, 1, frame_samples).
        """
        ...

    @abc.abstractmethod
    def forward_full(
        self, audio: torch.Tensor, bypass_quantizer: bool = True
    ) -> torch.Tensor:
        """Perform full utterance forward pass in batch mode.

        Args:
            audio: Full waveform tensor of shape (batch, 1, total_samples).
            bypass_quantizer: Whether to bypass vector quantizer.

        Returns:
            Reconstructed waveform tensor of shape (batch, 1, total_samples).
        """
        ...


class StreamCodec2Candidate(StreamingCodec):
    """Candidate A: StreamCodec2 (arXiv:2509.13670) 16 kHz / 20 ms causal codec.

    Note:
        Official pretrained weights are not publicly available on open hubs
        as of September 2026. This candidate defines the contract and flags
        fallback status during Phase 0 audit.
    """

    def __init__(self) -> None:
        """Initialize StreamCodec2 contract and placeholder network."""
        super().__init__()
        self._contract = CodecContract(
            name="StreamCodec2",
            sample_rate=16000,
            frame_samples=320,
            frame_ms=20.0,
            latent_dim=200,
            is_causal=True,
            quantizer_type="RSVQ",
            weights_available=False,
        )

    @property
    def contract(self) -> CodecContract:
        """Return the StreamCodec2 architectural contract."""
        return self._contract

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> CodecState:
        """Initialize clean streaming state containers."""
        return CodecState()

    def encode_chunk(
        self, audio_chunk: torch.Tensor, state: CodecState
    ) -> tuple[torch.Tensor, CodecState]:
        """Encode audio chunk."""
        raise NotImplementedError(
            "StreamCodec2 pretrained weights are unavailable in open source. "
            "Please select Candidate B (EnCodec 24kHz Causal)."
        )

    def decode_chunk(
        self,
        latent_chunk: torch.Tensor,
        state: CodecState,
        bypass_quantizer: bool = True,
    ) -> tuple[torch.Tensor, CodecState]:
        """Decode latent chunk."""
        raise NotImplementedError(
            "StreamCodec2 pretrained weights are unavailable in open source. "
            "Please select Candidate B (EnCodec 24kHz Causal)."
        )

    def forward_full(
        self, audio: torch.Tensor, bypass_quantizer: bool = True
    ) -> torch.Tensor:
        """Full forward pass."""
        raise NotImplementedError(
            "StreamCodec2 pretrained weights are unavailable in open source."
        )


class EnCodec24kCandidate(StreamingCodec):
    """Candidate B: Meta EnCodec 24 kHz Causal Mono neural audio codec.

    Operates with native 24 kHz sample rate, 320-sample hops (13.33 ms), and
    128-dimensional continuous latent space. Pretrained weights are officially
    distributed by Meta Research.
    """

    def __init__(self, buffer_chunks: int = 12) -> None:
        """Initialize EnCodec 24kHz model with streaming buffer configuration.

        Args:
            buffer_chunks: Number of historical chunks to retain in the streaming
                encoder receptive field buffer. 12 chunks = 160 ms history.
        """
        super().__init__()
        import encodec

        self.model = encodec.model.EncodecModel.encodec_model_24khz()
        self.model.eval()

        self.buffer_chunks = buffer_chunks
        self.buffer_samples = buffer_chunks * 320

        self._contract = CodecContract(
            name="EnCodec_24kHz_Causal",
            sample_rate=24000,
            frame_samples=320,
            frame_ms=13.333333333333334,
            latent_dim=128,
            is_causal=True,
            quantizer_type="RVQ",
            weights_available=True,
        )

    @property
    def contract(self) -> CodecContract:
        """Return the EnCodec 24kHz architectural contract."""
        return self._contract

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> CodecState:
        """Initialize clean streaming state containers for EnCodec streaming.

        Args:
            batch_size: Number of audio streams in the batch.
            device: Target torch device.

        Returns:
            Fresh CodecState with zero-initialized history buffers.
        """
        dev = device if device is not None else next(self.parameters()).device
        state = CodecState()
        state.encoder_conv_states["audio_history"] = torch.zeros(
            batch_size, 1, self.buffer_samples, device=dev
        )
        return state

    def encode_chunk(
        self, audio_chunk: torch.Tensor, state: CodecState
    ) -> tuple[torch.Tensor, CodecState]:
        """Encode 320-sample audio chunk into 128-d continuous latent vector.

        Args:
            audio_chunk: Raw audio tensor of shape (batch, 1, 320).
            state: Current CodecState containing audio history.

        Returns:
            Tuple of (z_chunk, next_state) with z_chunk shape (batch, 128, 1).
        """
        assert audio_chunk.dim() == 3 and audio_chunk.shape[-1] == 320, (
            f"Expected chunk shape (batch, 1, 320), got {audio_chunk.shape}"
        )
        device = audio_chunk.device
        if (
            "audio_history" not in state.encoder_conv_states
            or state.encoder_conv_states["audio_history"].shape[0]
            != audio_chunk.shape[0]
        ):
            state.encoder_conv_states["audio_history"] = torch.zeros(
                audio_chunk.shape[0], 1, self.buffer_samples, device=device
            )

        history = state.encoder_conv_states["audio_history"]
        window = torch.cat([history[:, :, 320:], audio_chunk], dim=-1)

        with torch.no_grad():
            z_window = self.model.encoder(window)
            z_chunk = z_window[:, :, -1:]

        next_state = state.clone()
        next_state.encoder_conv_states["audio_history"] = window
        next_state.frame_index += 1

        return z_chunk, next_state

    def decode_chunk(
        self,
        latent_chunk: torch.Tensor,
        state: CodecState,
        bypass_quantizer: bool = True,
    ) -> tuple[torch.Tensor, CodecState]:
        """Decode 128-d continuous latent frame back into 320 audio samples.

        Args:
            latent_chunk: Continuous latent of shape (batch, 128, 1).
            state: Current CodecState.
            bypass_quantizer: Whether to bypass the RVQ quantizer.

        Returns:
            Tuple of (audio_chunk, next_state) with audio_chunk (batch, 1, 320).
        """
        assert latent_chunk.dim() == 3 and latent_chunk.shape[1] == 128, (
            f"Expected latent shape (batch, 128, 1), got {latent_chunk.shape}"
        )

        with torch.no_grad():
            if bypass_quantizer:
                audio_chunk = self.model.decoder(latent_chunk)
            else:
                codes = self.model.quantizer.encode(
                    latent_chunk, self.model.frame_rate, self.model.bandwidth
                )
                emb = self.model.quantizer.decode(codes)
                audio_chunk = self.model.decoder(emb)

        next_state = state.clone()
        return audio_chunk, next_state

    def forward_full(
        self, audio: torch.Tensor, bypass_quantizer: bool = True
    ) -> torch.Tensor:
        """Batch forward pass of full utterance.

        Args:
            audio: Waveform tensor of shape (batch, 1, total_samples).
            bypass_quantizer: Whether to bypass vector quantizer.

        Returns:
            Reconstructed waveform of identical sample length.
        """
        with torch.no_grad():
            z = self.model.encoder(audio)
            if bypass_quantizer:
                out = self.model.decoder(z)
            else:
                frames = self.model.encode(audio)
                out = self.model.decode(frames)
        return tp.cast(torch.Tensor, out[:, :, : audio.shape[-1]])
