"""WASAPI audio engine runtime abstraction for low-period audio pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from timbre_lite.runtime.ring_buffer import SPSCRingBuffer


class EngineStatus(str, Enum):
    """Lifecycle status of the audio engine."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class WASAPIConfig:
    """Configuration parameters for Windows WASAPI audio engine abstraction.

    Attributes:
        sample_rate: Audio sample rate in Hz (default 24000).
        chunk_size: Frame period in samples (default 320 = 13.33ms).
        channels: Channel count (1 = mono).
        elasticity_frames: SPSC ring buffer elasticity queue depth (1-3).
    """

    sample_rate: int = 24000
    chunk_size: int = 320
    channels: int = 1
    elasticity_frames: int = 2  # 2 frames = 26.67ms balanced elasticity


class WASAPIEngine:
    """Audio engine runtime abstraction modeling WASAPI low-period pipelines.

    Simulates Windows MMCSS Pro Audio real-time thread scheduling protocol
    with pre-allocated ring buffers and bounded latency elasticity. Note: This
    is a Python runtime abstraction; direct Windows COM (IAudioClient3 / IMMDevice)
    bindings are scheduled for the native C++ deployment phase.
    """

    def __init__(self, config: WASAPIConfig | None = None) -> None:
        """Initialize WASAPIEngine.

        Args:
            config: Audio engine configuration parameters.
        """
        self.config = config if config is not None else WASAPIConfig()
        self.status: EngineStatus = EngineStatus.STOPPED

        # Lock-free SPSC ring buffers for input capture and output render
        self.input_buffer = SPSCRingBuffer(
            capacity_frames=self.config.elasticity_frames + 2,
            chunk_size=self.config.chunk_size,
            channels=self.config.channels,
        )
        self.output_buffer = SPSCRingBuffer(
            capacity_frames=self.config.elasticity_frames + 2,
            chunk_size=self.config.chunk_size,
            channels=self.config.channels,
        )

        # Pre-allocated scratch slot for audio callback
        self._scratch_slot = torch.zeros(
            1, self.config.channels, self.config.chunk_size
        )

    def start(self) -> None:
        """Start real-time audio engine processing."""
        self.status = EngineStatus.RUNNING
        self.input_buffer.reset()
        self.output_buffer.reset()

    def stop(self) -> None:
        """Stop real-time audio engine processing."""
        self.status = EngineStatus.STOPPED

    def on_microphone_chunk(self, chunk: torch.Tensor) -> bool:
        """Audio capture callback called by WASAPI capture endpoint.

        Args:
            chunk: Input PCM audio chunk tensor.

        Returns:
            True if chunk was enqueued; False on buffer overrun.
        """
        if self.status != EngineStatus.RUNNING:
            return False
        return self.input_buffer.push(chunk)

    def on_render_chunk(self) -> torch.Tensor | None:
        """Audio playback callback called by WASAPI render endpoint.

        Returns:
            PCM chunk to render, or None on buffer underrun.
        """
        if self.status != EngineStatus.RUNNING:
            return None
        return self.output_buffer.pop(out_chunk=self._scratch_slot)

    @property
    def total_underruns(self) -> int:
        """Return total output render underruns observed.

        Returns:
            Integer underrun count.
        """
        return self.output_buffer.underrun_count

    @property
    def total_overruns(self) -> int:
        """Return total input capture overruns observed.

        Returns:
            Integer overrun count.
        """
        return self.input_buffer.overrun_count
