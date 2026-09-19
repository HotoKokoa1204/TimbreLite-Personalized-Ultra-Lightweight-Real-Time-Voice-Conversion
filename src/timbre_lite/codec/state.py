"""State container for streaming neural audio codecs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CodecState:
    """Encapsulates streaming internal memory states across chunk boundaries.

    Attributes:
        encoder_conv_states: Receptive field buffers for encoder causal conv.
        encoder_lstm_states: Hidden and cell states (h, c) for encoder recurrent.
        decoder_conv_states: Receptive field buffers for decoder causal conv.
        decoder_lstm_states: Hidden and cell states (h, c) for decoder recurrent.
        overlap_buffer: Tail overlap buffer for transposed convolution synthesis.
        frame_index: Sequential index of currently processed streaming frame.
    """

    encoder_conv_states: dict[str, torch.Tensor] = field(default_factory=dict)
    encoder_lstm_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    decoder_conv_states: dict[str, torch.Tensor] = field(default_factory=dict)
    decoder_lstm_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    overlap_buffer: torch.Tensor | None = None
    frame_index: int = 0

    def reset(self) -> None:
        """Reset all internal streaming buffers to initial clean state."""
        self.encoder_conv_states.clear()
        self.encoder_lstm_states.clear()
        self.decoder_conv_states.clear()
        self.decoder_lstm_states.clear()
        self.overlap_buffer = None
        self.frame_index = 0

    def clone(self) -> CodecState:
        """Create a deep clone of the current state container.

        Returns:
            A new CodecState instance with cloned tensors.
        """
        cloned = CodecState(
            encoder_conv_states={
                k: v.clone() for k, v in self.encoder_conv_states.items()
            },
            encoder_lstm_states={
                k: (v[0].clone(), v[1].clone())
                for k, v in self.encoder_lstm_states.items()
            },
            decoder_conv_states={
                k: v.clone() for k, v in self.decoder_conv_states.items()
            },
            decoder_lstm_states={
                k: (v[0].clone(), v[1].clone())
                for k, v in self.decoder_lstm_states.items()
            },
            overlap_buffer=(
                self.overlap_buffer.clone() if self.overlap_buffer is not None else None
            ),
            frame_index=self.frame_index,
        )
        return cloned
