"""Configurable lookback buffer for onset consonant preservation."""

from __future__ import annotations

from collections import deque
from enum import Enum

import torch


class LookbackMode(str, Enum):
    """Operation mode for lookback buffer upon speech onset."""

    WARM_UP = "warm_up"  # Mode A: Pre-warms state FIFO without audio latency
    BURST_REPLAY = "burst_replay"  # Mode B: Replays initial plosive consonants


class LookbackBuffer:
    """Circular lookback buffer holding 20-40ms of recent audio frames.

    Preserves initial voiceless plosives and plosive attacks ('P', 'T', 'K')
    that occur before VAD classification confidence threshold is crossed.
    """

    def __init__(
        self,
        capacity_chunks: int = 2,
        mode: LookbackMode = LookbackMode.WARM_UP,
    ) -> None:
        """Initialize LookbackBuffer.

        Args:
            capacity_chunks: Number of historical audio chunks to retain (2 = 26.67ms).
            mode: Onset handling strategy (WARM_UP or BURST_REPLAY).
        """
        self.capacity_chunks = capacity_chunks
        self.mode = mode
        self._buffer: deque[torch.Tensor] = deque(maxlen=capacity_chunks)

    def push(self, chunk: torch.Tensor) -> None:
        """Add new chunk into circular lookback history.

        Args:
            chunk: Audio chunk tensor.
        """
        self._buffer.append(chunk.detach().clone())

    def get_chunks(self) -> list[torch.Tensor]:
        """Retrieve all buffered lookback chunks in chronological order.

        Returns:
            List of historical audio chunk tensors.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Clear all buffered historical chunks."""
        self._buffer.clear()

    @property
    def current_size(self) -> int:
        """Return number of currently buffered chunks.

        Returns:
            Integer chunk count.
        """
        return len(self._buffer)
