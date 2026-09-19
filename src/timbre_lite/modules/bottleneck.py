"""Stateful causal bottleneck neural pipeline for streaming latent transformation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from timbre_lite.modules.causal_layers import CausalDilatedResidualBlock


@dataclass
class BottleneckState:
    """Encapsulates persistent streaming state for the causal bottleneck module.

    Attributes:
        tcn_states: Receptive field buffers for each dilated causal TCN block.
        gru_state: Recurrent hidden state tensor for the causal GRU layer.
        frame_index: Sequential index of the currently processed latent frame.
    """

    tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    gru_state: torch.Tensor | None = None
    frame_index: int = 0

    def reset(self) -> None:
        """Reset internal bottleneck states to initial zero buffers."""
        self.tcn_states.clear()
        self.gru_state = None
        self.frame_index = 0

    def clone(self) -> BottleneckState:
        """Deep clone state container.

        Returns:
            Cloned BottleneckState instance.
        """
        cloned_tcn = [(s[0].clone(), s[1].clone()) for s in self.tcn_states]
        cloned_gru = self.gru_state.clone() if self.gru_state is not None else None
        return BottleneckState(
            tcn_states=cloned_tcn,
            gru_state=cloned_gru,
            frame_index=self.frame_index,
        )


class CausalBottleneck(nn.Module):
    """Compact causal neural bottleneck module operating on continuous codec latents.

    Transforms 128-d continuous latents through an ultra-compact bottleneck
    (e.g. 64-d or 32-d) combining dilated causal TCNs and a causal GRU to maintain
    temporal timbre stability across frames with strict streaming equivalence.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        bottleneck_dim: int = 64,
        tcn_layers: int = 4,
        gru_hidden: int = 64,
        kernel_size: int = 3,
        residual: bool = True,
    ) -> None:
        """Initialize stateful causal bottleneck network.

        Args:
            latent_dim: Dimension of input and output codec latent space (128).
            bottleneck_dim: Compressed intermediate feature dimension (64 or 32).
            tcn_layers: Number of dilated causal residual blocks.
            gru_hidden: Hidden size for recurrent causal GRU.
            kernel_size: Convolution kernel size for causal TCN layers.
            residual: Whether to add input latent as a residual skip connection.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.tcn_layers = tcn_layers
        self.gru_hidden = gru_hidden

        self.in_proj = nn.Conv1d(latent_dim, bottleneck_dim, kernel_size=1)

        tcn_blocks: list[CausalDilatedResidualBlock] = []
        for i in range(tcn_layers):
            dilation = 2**i
            tcn_blocks.append(
                CausalDilatedResidualBlock(
                    channels=bottleneck_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
        self.tcn_blocks: list[CausalDilatedResidualBlock] = tcn_blocks
        self.tcn = nn.ModuleList(tcn_blocks)

        self.gru = nn.GRU(
            input_size=bottleneck_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.out_proj = nn.Conv1d(bottleneck_dim, latent_dim, kernel_size=1)
        self.residual = residual
        if residual:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters in this module.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> BottleneckState:
        """Initialize clean streaming state container.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device.

        Returns:
            Clean BottleneckState.
        """
        dev = device if device is not None else next(self.parameters()).device
        tcn_states = [
            block.init_state(batch_size=batch_size, device=dev)
            for block in self.tcn_blocks
        ]
        gru_state = torch.zeros(1, batch_size, self.gru_hidden, device=dev)
        return BottleneckState(
            tcn_states=tcn_states, gru_state=gru_state, frame_index=0
        )

    def forward_sequence(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward pass across temporal latent sequence.

        Args:
            z_seq: Continuous latent tensor of shape (batch, latent_dim, time).

        Returns:
            Reconstructed latent tensor of shape (batch, latent_dim, time).
        """
        h: torch.Tensor = self.in_proj(z_seq)

        for block in self.tcn_blocks:
            h = block.forward_sequence(h)

        # GRU expects (batch, time, channels)
        h_perm = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h_perm)
        h = gru_out.permute(0, 2, 1)

        out: torch.Tensor = self.out_proj(h)
        if self.residual:
            out = z_seq + out
        return out

    def forward_chunk(
        self, z_chunk: torch.Tensor, state: BottleneckState | None = None
    ) -> tuple[torch.Tensor, BottleneckState]:
        """Streaming chunk forward pass maintaining persistent internal states.

        Args:
            z_chunk: Single latent frame of shape (batch, latent_dim, 1).
            state: Previous BottleneckState.

        Returns:
            Tuple of (out_chunk, next_state), where out_chunk has shape
            (batch, latent_dim, 1).
        """
        if state is None:
            state = self.init_state(batch_size=z_chunk.shape[0], device=z_chunk.device)

        h: torch.Tensor = self.in_proj(z_chunk)

        next_tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.tcn_blocks):
            block_state = state.tcn_states[i] if i < len(state.tcn_states) else None
            h, next_s = block.forward_chunk(h, state=block_state)
            next_tcn_states.append(next_s)

        # GRU step: input shape (batch, 1, channels)
        h_perm = h.permute(0, 2, 1)
        prev_gru_h = state.gru_state
        if prev_gru_h is None or prev_gru_h.shape[1] != z_chunk.shape[0]:
            prev_gru_h = torch.zeros(
                1, z_chunk.shape[0], self.gru_hidden, device=z_chunk.device
            )

        gru_out, next_gru_h = self.gru(h_perm, prev_gru_h)
        h = gru_out.permute(0, 2, 1)

        out_chunk: torch.Tensor = self.out_proj(h)
        if self.residual:
            out_chunk = z_chunk + out_chunk

        next_state = BottleneckState(
            tcn_states=next_tcn_states,
            gru_state=next_gru_h,
            frame_index=state.frame_index + 1,
        )
        return out_chunk, next_state
