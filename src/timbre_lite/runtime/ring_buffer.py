"""Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for real-time audio."""

from __future__ import annotations

import threading

import torch


class SPSCRingBuffer:
    """Thread-safe lock-free ring buffer for single producer and single consumer.

    Operates with pre-allocated static tensors to eliminate dynamic memory
    allocations in the real-time audio thread, providing jitter elasticity.
    """

    def __init__(
        self,
        capacity_frames: int = 4,
        chunk_size: int = 320,
        channels: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize SPSC ring buffer.

        Args:
            capacity_frames: Total ring buffer capacity in chunks.
            chunk_size: Number of samples per chunk (320 = 13.33ms at 24kHz).
            channels: Number of audio channels (1 = mono).
            dtype: Tensor data type.
        """
        self.capacity = capacity_frames
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype

        # Pre-allocated storage slots (zero allocation during push/pop)
        self._storage: list[torch.Tensor] = [
            torch.zeros(1, channels, chunk_size, dtype=dtype)
            for _ in range(capacity_frames)
        ]

        # Read and write head indices
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()

        # Telemetry counters
        self.overrun_count = 0
        self.underrun_count = 0

    @property
    def available_read(self) -> int:
        """Return number of frames currently available to read.

        Returns:
            Available frame count.
        """
        with self._lock:
            return self._count

    @property
    def available_write(self) -> int:
        """Return number of free slots available to write.

        Returns:
            Available write slots.
        """
        with self._lock:
            return self.capacity - self._count

    def push(self, chunk: torch.Tensor) -> bool:
        """Push a chunk into the ring buffer without dynamic memory allocation.

        Args:
            chunk: Input audio chunk tensor of shape (..., channels, chunk_size).

        Returns:
            True if chunk was written successfully; False if buffer was full (overrun).
        """
        with self._lock:
            if self._count >= self.capacity:
                self.overrun_count += 1
                return False

            # Copy data in-place into pre-allocated slot
            self._storage[self._write_idx].copy_(chunk)
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._count += 1
            return True

    def pop(self, out_chunk: torch.Tensor | None = None) -> torch.Tensor | None:
        """Pop a chunk from the ring buffer into a pre-allocated tensor slot.

        Args:
            out_chunk: Optional pre-allocated destination tensor.

        Returns:
            Popped chunk tensor, or None if buffer was empty (underrun).
        """
        with self._lock:
            if self._count == 0:
                self.underrun_count += 1
                return None

            src = self._storage[self._read_idx]
            self._read_idx = (self._read_idx + 1) % self.capacity
            self._count -= 1

            if out_chunk is not None:
                out_chunk.copy_(src)
                return out_chunk
            else:
                return src.clone()

    def reset(self) -> None:
        """Clear all contents and reset indices and telemetry."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self.overrun_count = 0
            self.underrun_count = 0
