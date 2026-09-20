"""Sub-250K Personalized Adapter and Full Streaming Transformation Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.codec.state import CodecState
from timbre_lite.modules.causal_layers import CausalDilatedResidualBlock
from timbre_lite.modules.cleanser import CleanserState, ContentCleanser
from timbre_lite.modules.prosody import InGraphProsodyHead, ProsodyState


class FusionMode(str, Enum):
    """Supported dual-stream fusion strategies."""

    ADDITIVE = "additive"
    FILM = "film"


class DualStreamFusion(nn.Module):
    """Fuses cleansed phonetic content with in-graph prosody representations."""

    def __init__(
        self,
        content_dim: int = 64,
        prosody_dim: int = 16,
        mode: FusionMode = FusionMode.ADDITIVE,
    ) -> None:
        """Initialize DualStreamFusion layer.

        Args:
            content_dim: Cleansed content dimension (64 or 32).
            prosody_dim: In-graph prosody dimension (16 or 32).
            mode: Fusion strategy (ADDITIVE or FILM).
        """
        super().__init__()
        self.content_dim = content_dim
        self.prosody_dim = prosody_dim
        self.mode = mode

        if mode == FusionMode.ADDITIVE:
            self.prosody_proj = nn.Conv1d(prosody_dim, content_dim, kernel_size=1)
        else:
            self.gamma = nn.Conv1d(prosody_dim, content_dim, kernel_size=1)
            self.beta = nn.Conv1d(prosody_dim, content_dim, kernel_size=1)

    def forward(self, content: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        """Fuse content and prosody streams.

        Args:
            content: Cleansed phonetic tensor of shape (batch, content_dim, time).
            prosody: Prosody tensor of shape (batch, prosody_dim, time).

        Returns:
            Fused conditioning representation of shape (batch, content_dim, time).
        """
        if self.mode == FusionMode.ADDITIVE:
            proj_p: torch.Tensor = self.prosody_proj(prosody)
            fused: torch.Tensor = content + proj_p
            return fused
        else:
            g: torch.Tensor = self.gamma(prosody)
            b: torch.Tensor = self.beta(prosody)
            film_fused: torch.Tensor = (1.0 + g) * content + b
            return film_fused


@dataclass
class AdapterState:
    """Persistent streaming state for PersonalizedAdapter.

    Attributes:
        tcn_states: Convolutional receptive field buffers.
        gru_state: Recurrent GRU hidden state tensor.
        frame_index: Processed frame counter.
    """

    tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    gru_state: torch.Tensor | None = None
    frame_index: int = 0

    def reset(self) -> None:
        """Reset internal adapter states."""
        self.tcn_states.clear()
        self.gru_state = None
        self.frame_index = 0

    def apply_decay(self, decay_factor: float) -> None:
        """Apply lazy exponential soft decay to recurrent GRU hidden state.

        Args:
            decay_factor: Multiplicative scalar alpha in [0.0, 1.0].
        """
        if self.gru_state is not None:
            self.gru_state = self.gru_state * decay_factor


class PersonalizedAdapter(nn.Module):
    """Sub-250K parameter Personalized Adapter generating target vocal timbre.

    Transforms fused content and prosody representations into the continuous
    latent space of the target persona ($C_{latent} = 128$) using a causal TCN
    and recurrent GRU.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 128,
        hidden_dim: int = 64,
        tcn_layers: int = 4,
        gru_hidden: int = 64,
    ) -> None:
        """Initialize PersonalizedAdapter.

        Args:
            in_dim: Input fused representation dimension (64 or 32).
            out_dim: Output continuous latent dimension (128).
            hidden_dim: Intermediate feature channel dimension.
            tcn_layers: Number of dilated causal residual blocks.
            gru_hidden: Hidden dimension of causal GRU.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden

        self.skip_proj = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        tcn_blocks: list[CausalDilatedResidualBlock] = []
        for i in range(tcn_layers):
            tcn_blocks.append(
                CausalDilatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=3,
                    dilation=2**i,
                )
            )
        self.tcn_blocks: list[CausalDilatedResidualBlock] = tcn_blocks
        self.tcn = nn.ModuleList(tcn_blocks)

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.out_proj = nn.Conv1d(gru_hidden, out_dim, kernel_size=1)

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters in this module.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> AdapterState:
        """Initialize clean streaming state.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device.

        Returns:
            Clean AdapterState container.
        """
        dev = device if device is not None else next(self.parameters()).device
        tcn_states = [
            block.init_state(batch_size=batch_size, device=dev)
            for block in self.tcn_blocks
        ]
        gru_state = torch.zeros(1, batch_size, self.gru_hidden, device=dev)
        return AdapterState(tcn_states=tcn_states, gru_state=gru_state, frame_index=0)

    def forward_sequence(self, u_seq: torch.Tensor) -> torch.Tensor:
        """Process full temporal sequence of fused conditioning representations.

        Args:
            u_seq: Fused conditioning tensor of shape (batch, in_dim, time).

        Returns:
            Target persona latent tensor of shape (batch, out_dim, time).
        """
        skip: torch.Tensor = self.skip_proj(u_seq)
        h: torch.Tensor = self.in_proj(u_seq)
        for block in self.tcn_blocks:
            h = block.forward_sequence(h)

        # GRU expects (batch, time, channels)
        h_perm = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h_perm)
        h = gru_out.permute(0, 2, 1)

        delta: torch.Tensor = self.out_proj(h)
        out: torch.Tensor = skip + delta
        return out

    def forward_chunk(
        self, u_chunk: torch.Tensor, state: AdapterState | None = None
    ) -> tuple[torch.Tensor, AdapterState]:
        """Streaming chunk forward pass maintaining temporal receptive field.

        Args:
            u_chunk: Single fused conditioning frame of shape (batch, in_dim, 1).
            state: Previous streaming state.

        Returns:
            Tuple of (out_chunk, next_state) where out_chunk has shape
            (batch, out_dim, 1).
        """
        if state is None:
            state = self.init_state(batch_size=u_chunk.shape[0], device=u_chunk.device)

        skip_chunk: torch.Tensor = self.skip_proj(u_chunk)
        h: torch.Tensor = self.in_proj(u_chunk)
        next_tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.tcn_blocks):
            block_state = state.tcn_states[i] if i < len(state.tcn_states) else None
            h, next_s = block.forward_chunk(h, state=block_state)
            next_tcn_states.append(next_s)

        # Recurrent GRU step
        h_perm = h.permute(0, 2, 1)
        prev_gru_h = state.gru_state
        if prev_gru_h is None or prev_gru_h.shape[1] != u_chunk.shape[0]:
            prev_gru_h = torch.zeros(
                1, u_chunk.shape[0], self.gru_hidden, device=u_chunk.device
            )

        gru_out, next_gru_h = self.gru(h_perm, prev_gru_h)
        h = gru_out.permute(0, 2, 1)

        delta_chunk: torch.Tensor = self.out_proj(h)
        out_chunk: torch.Tensor = skip_chunk + delta_chunk
        next_state = AdapterState(
            tcn_states=next_tcn_states,
            gru_state=next_gru_h,
            frame_index=state.frame_index + 1,
        )
        return out_chunk, next_state


@dataclass
class PipelineStreamingState:
    """Encapsulates full pipeline streaming state across all modules.

    Attributes:
        codec_state: EnCodec streaming buffer state.
        cleanser_state: ContentCleanser receptive field state.
        prosody_state: InGraphProsodyHead receptive field state.
        adapter_state: PersonalizedAdapter TCN + GRU state.
    """

    codec_state: CodecState
    cleanser_state: CleanserState
    prosody_state: ProsodyState
    adapter_state: AdapterState

    def reset(self) -> None:
        """Reset all internal pipeline streaming states."""
        self.codec_state.reset()
        self.cleanser_state.reset()
        self.prosody_state.reset()
        self.adapter_state.reset()

    def apply_decay(self, decay_factor: float) -> None:
        """Apply lazy decay to adapter recurrent state upon speech resumption.

        Args:
            decay_factor: Multiplicative scalar alpha in [0.0, 1.0].
        """
        self.adapter_state.apply_decay(decay_factor)


class FullPersonalizedPipeline(nn.Module):
    """Unified end-to-end streaming Voice Conversion Pipeline.

    Fuses EnCodec 24kHz -> Content Cleanser + In-Graph Prosody -> DualStreamFusion
    -> Sub-250K Personalized Adapter -> EnCodec Decoder.
    """

    def __init__(
        self,
        codec: EnCodec24kCandidate,
        cleanser: ContentCleanser,
        prosody_head: InGraphProsodyHead,
        fusion: DualStreamFusion,
        adapter: PersonalizedAdapter,
    ) -> None:
        """Initialize FullPersonalizedPipeline.

        Args:
            codec: Verified causal 24kHz codec candidate.
            cleanser: Speaker-invariant content cleanser.
            prosody_head: In-graph prosody extraction head.
            fusion: Dual-stream conditioning fusion layer.
            adapter: Personalized adapter model.
        """
        super().__init__()
        self.codec = codec
        self.cleanser = cleanser
        self.prosody_head = prosody_head
        self.fusion = fusion
        self.adapter = adapter

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters across all personal conversion modules.

        Returns:
            Total trainable parameter count (strictly under 250K).
        """
        trainable = (
            self.cleanser.count_parameters()
            + self.prosody_head.count_parameters()
            + sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
            + self.adapter.count_parameters()
        )
        return trainable

    def init_streaming_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> PipelineStreamingState:
        """Initialize composite streaming state container.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device.

        Returns:
            Clean PipelineStreamingState.
        """
        dev = device if device is not None else next(self.parameters()).device
        return PipelineStreamingState(
            codec_state=self.codec.init_state(),
            cleanser_state=self.cleanser.init_state(batch_size=batch_size, device=dev),
            prosody_state=self.prosody_head.init_state(
                batch_size=batch_size, device=dev
            ),
            adapter_state=self.adapter.init_state(batch_size=batch_size, device=dev),
        )

    def forward_sequence(
        self, z_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process full temporal sequence of continuous codec latents.

        Args:
            z_seq: Input continuous latent tensor of shape (batch, 128, time).

        Returns:
            Tuple of (z_adapted, c_seq, p_seq) where z_adapted has shape
            (batch, 128, time).
        """
        c_seq = self.cleanser.forward_sequence(z_seq)
        p_seq = self.prosody_head.forward_sequence(z_seq)
        u_seq = self.fusion(c_seq, p_seq)
        z_adapted = self.adapter.forward_sequence(u_seq)
        return z_adapted, c_seq, p_seq

    def step_audio_chunk(
        self,
        audio_chunk: torch.Tensor,
        state: PipelineStreamingState | None = None,
    ) -> tuple[torch.Tensor, PipelineStreamingState]:
        """Execute single end-to-end streaming step (320 samples -> 320 samples).

        Args:
            audio_chunk: Raw PCM audio chunk (batch, 1, 320).
            state: Pipeline streaming state container.

        Returns:
            Tuple of (converted_audio_chunk, next_state) where converted chunk
            has shape (batch, 1, 320).
        """
        if state is None:
            state = self.init_streaming_state(
                batch_size=audio_chunk.shape[0], device=audio_chunk.device
            )

        # 1. Causal EnCodec continuous encoder (320 -> 128)
        z_chunk, next_codec_state = self.codec.encode_chunk(
            audio_chunk, state.codec_state
        )

        # 2. Dual-stream: Content Cleanser & In-Graph Prosody
        c_chunk, next_cleanser_state = self.cleanser.forward_chunk(
            z_chunk, state.cleanser_state
        )
        p_chunk, next_prosody_state = self.prosody_head.forward_chunk(
            z_chunk, state.prosody_state
        )

        # 3. Dual-stream fusion
        u_chunk = self.fusion(c_chunk, p_chunk)

        # 4. Personalized Adapter (64 -> 128)
        z_target, next_adapter_state = self.adapter.forward_chunk(
            u_chunk, state.adapter_state
        )

        # 5. Causal EnCodec continuous decoder (128 -> 320)
        y_out, final_codec_state = self.codec.decode_chunk(
            z_target, next_codec_state, bypass_quantizer=True
        )

        next_state = PipelineStreamingState(
            codec_state=final_codec_state,
            cleanser_state=next_cleanser_state,
            prosody_state=next_prosody_state,
            adapter_state=next_adapter_state,
        )
        return y_out, next_state
