"""In-Graph causal prosody extraction head for zero-overhead inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from timbre_lite.modules.causal_layers import CausalDilatedResidualBlock


@dataclass
class ProsodyState:
    """Persistent streaming state for InGraphProsodyHead.

    Attributes:
        tcn_states: Convolutional receptive field buffers.
        frame_index: Processed frame counter.
    """

    tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    frame_index: int = 0

    def reset(self) -> None:
        """Reset internal prosody state."""
        self.tcn_states.clear()
        self.frame_index = 0


class InGraphProsodyHead(nn.Module):
    """Causal prosody head extracting pitch, voicing, and energy directly from latents.

    Operates entirely within the neural execution graph without relying on
    external CPU pitch trackers (CREPE/RMVPE/YIN) during real-time streaming inference.
    """

    def __init__(
        self,
        in_dim: int = 128,
        prosody_dim: int = 16,
        hidden_dim: int = 32,
        tcn_layers: int = 2,
    ) -> None:
        """Initialize InGraphProsodyHead.

        Args:
            in_dim: Dimension of input codec latent space (128).
            prosody_dim: Output prosody feature dimension (16 or 32).
            hidden_dim: Intermediate feature channel dimension.
            tcn_layers: Number of dilated causal residual blocks.
        """
        super().__init__()
        self.in_dim = in_dim
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim

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

        # Multi-task prosody projection:
        # Features 0..prosody_dim-3: general prosodic embedding
        # Feature -2: continuous log-F0 estimate
        # Feature -1: voiced / unvoiced probability logit
        self.out_proj = nn.Conv1d(hidden_dim, prosody_dim, kernel_size=1)

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters in this module.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> ProsodyState:
        """Initialize clean streaming state.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device.

        Returns:
            Clean ProsodyState container.
        """
        dev = device if device is not None else next(self.parameters()).device
        tcn_states = [
            block.init_state(batch_size=batch_size, device=dev)
            for block in self.tcn_blocks
        ]
        return ProsodyState(tcn_states=tcn_states, frame_index=0)

    def forward_sequence(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Extract prosody feature sequence from full temporal latent sequence.

        Args:
            z_seq: Input continuous latent tensor of shape (batch, in_dim, time).

        Returns:
            Prosody tensor of shape (batch, prosody_dim, time).
        """
        h: torch.Tensor = self.in_proj(z_seq)
        for block in self.tcn_blocks:
            h = block.forward_sequence(h)
        out: torch.Tensor = self.out_proj(h)
        return out

    def forward_chunk(
        self, z_chunk: torch.Tensor, state: ProsodyState | None = None
    ) -> tuple[torch.Tensor, ProsodyState]:
        """Streaming chunk prosody extraction.

        Args:
            z_chunk: Single latent chunk of shape (batch, in_dim, 1).
            state: Previous streaming state.

        Returns:
            Tuple of (prosody_chunk, next_state) where prosody_chunk has shape
            (batch, prosody_dim, 1).
        """
        if state is None:
            state = self.init_state(batch_size=z_chunk.shape[0], device=z_chunk.device)

        h: torch.Tensor = self.in_proj(z_chunk)
        next_tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.tcn_blocks):
            block_state = state.tcn_states[i] if i < len(state.tcn_states) else None
            h, next_s = block.forward_chunk(h, state=block_state)
            next_tcn_states.append(next_s)

        out_chunk: torch.Tensor = self.out_proj(h)
        next_state = ProsodyState(
            tcn_states=next_tcn_states,
            frame_index=state.frame_index + 1,
        )
        return out_chunk, next_state

    def get_pitch_and_voicing(
        self, prosody: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract explicit pitch and voicing probability from prosody representation.

        Args:
            prosody: Prosody tensor of shape (batch, prosody_dim, time).

        Returns:
            Tuple of (f0_continuous, vuv_prob) with shapes (batch, 1, time).
        """
        log_f0 = prosody[:, -2:-1, :]
        vuv_logit = prosody[:, -1:, :]
        vuv_prob = torch.sigmoid(vuv_logit)
        return log_f0, vuv_prob
