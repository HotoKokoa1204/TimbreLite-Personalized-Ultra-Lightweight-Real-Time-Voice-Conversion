"""Causal convolutional layers with exact streaming state preservation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class CausalConv1d(nn.Module):
    """1D Causal Convolution enforcing zero lookahead via asymmetric left-padding.

    Supports both batched sequence forward passes and stateful single-chunk
    streaming invocations with exact numerical equivalence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize causal 1D convolution layer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Temporal convolution kernel size.
            dilation: Spacing between kernel elements.
            bias: Whether to add a learnable bias to the output.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_len = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        """Initialize zero-filled receptive field buffer for streaming.

        Args:
            batch_size: Number of audio streams in batch.
            device: Target torch device.

        Returns:
            Tensor of shape (batch_size, in_channels, pad_len).
        """
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(batch_size, self.in_channels, self.pad_len, device=dev)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Process full sequence with causal left-padding.

        Args:
            x: Input tensor of shape (batch, in_channels, time).

        Returns:
            Output tensor of shape (batch, out_channels, time).
        """
        if self.pad_len > 0:
            x_padded = F.pad(x, (self.pad_len, 0))
        else:
            x_padded = x
        out: torch.Tensor = self.conv(x_padded)
        return out

    def forward_chunk(
        self, x_chunk: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process single streaming chunk maintaining receptive field state.

        Args:
            x_chunk: Chunk tensor of shape (batch, in_channels, chunk_len).
            state: Previous receptive field tensor of shape
                (batch, in_channels, pad_len).

        Returns:
            Tuple of (out_chunk, next_state), where out_chunk has shape
            (batch, out_channels, chunk_len) and next_state has shape
            (batch, in_channels, pad_len).
        """
        if self.pad_len == 0:
            return self.conv(x_chunk), x_chunk

        if state is None:
            state = self.init_state(batch_size=x_chunk.shape[0], device=x_chunk.device)

        x_padded = torch.cat([state, x_chunk], dim=-1)
        next_state = x_padded[:, :, -self.pad_len :]
        out_chunk = self.conv(x_padded)
        return out_chunk, next_state


class CausalDilatedResidualBlock(nn.Module):
    """Residual block composed of dilated causal convolutions and ELU activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        """Initialize causal dilated residual block.

        Args:
            channels: Number of input and output feature channels.
            kernel_size: Temporal convolution kernel size.
            dilation: Dilation factor for the first causal convolution.
        """
        super().__init__()
        self.channels = channels
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation
        )
        self.act = nn.ELU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size=1, dilation=1)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize clean streaming states for internal convolutions.

        Args:
            batch_size: Number of parallel streams.
            device: Target torch device.

        Returns:
            Tuple of (state_conv1, state_conv2).
        """
        return (
            self.conv1.init_state(batch_size=batch_size, device=device),
            self.conv2.init_state(batch_size=batch_size, device=device),
        )

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass across full temporal sequence.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor of shape (batch, channels, time).
        """
        residual = x
        h = self.conv1.forward_sequence(x)
        h = self.act(h)
        h = self.conv2.forward_sequence(h)
        return residual + h

    def forward_chunk(
        self,
        x_chunk: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Streaming chunk forward pass maintaining convolutional states.

        Args:
            x_chunk: Streaming chunk tensor of shape (batch, channels, chunk_len).
            state: Tuple of (state_conv1, state_conv2).

        Returns:
            Tuple of (out_chunk, next_state).
        """
        if state is None:
            state = self.init_state(batch_size=x_chunk.shape[0], device=x_chunk.device)

        state1, state2 = state
        residual = x_chunk

        h, next_state1 = self.conv1.forward_chunk(x_chunk, state=state1)
        h = self.act(h)
        h, next_state2 = self.conv2.forward_chunk(h, state=state2)

        out_chunk = residual + h
        return out_chunk, (next_state1, next_state2)
