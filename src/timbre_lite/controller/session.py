"""Host Session Controller with zero-GPU silence gating and lazy state decay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from timbre_lite.controller.buffer import LookbackBuffer, LookbackMode
from timbre_lite.controller.vad import TwoStageVAD


class SessionState(str, Enum):
    """Lifecycle states for the host voice session."""

    IDLE = "idle"  # Silent, 0% GPU workload
    WARMING_UP = "warming_up"  # Onset handling (Mode A/B lookback)
    ACTIVE = "active"  # Active speech transformation
    HANGOVER = "hangover"  # Intra-phrase micro-pause (80-120ms)


class SessionAction(str, Enum):
    """Action directive for the streaming pipeline."""

    IDLE_SKIP = "idle_skip"  # Do not launch GPU inference
    WARMUP_FEED = "warmup_feed"  # Warm up internal FIFO buffers
    BURST_REPLAY = "burst_replay"  # Burst replay plosive onset chunks
    PROCESS_NORMAL = "process_normal"  # Regular active chunk inference


@dataclass
class SessionDecision:
    """Decision output produced by the Session Controller per frame.

    Attributes:
        action: Directive indicating how the frame should be processed.
        chunks_to_process: List of audio chunks to feed to the neural pipeline.
        decay_factor: Host-computed lazy decay factor alpha for recurrent state.
        state: Current session lifecycle state.
        vad_confidence: VAD confidence score in [0.0, 1.0].
    """

    action: SessionAction
    chunks_to_process: list[torch.Tensor]
    decay_factor: float
    state: SessionState
    vad_confidence: float


class SessionController:
    """Host Session Controller orchestrating silence gating and lazy decay.

    Ensures 0% GPU compute during silence while preserving plosive onsets and
    maintaining vocal timbre across intra-phrase micro-pauses.
    """

    def __init__(
        self,
        vad: TwoStageVAD | None = None,
        lookback_buffer: LookbackBuffer | None = None,
        hangover_frames: int = 8,  # 8 * 13.33ms = 106.7ms
        decay_lambda: float = 2.0,  # Decay rate per second
        frame_duration: float = 0.013333,
    ) -> None:
        """Initialize SessionController.

        Args:
            vad: Two-stage CPU Voice Activity Detector instance.
            lookback_buffer: Circular lookback buffer instance.
            hangover_frames: Number of frames to hold hangover (80-120ms).
            decay_lambda: Soft decay rate parameter lambda.
            frame_duration: Duration of single audio frame in seconds.
        """
        self.vad = vad if vad is not None else TwoStageVAD()
        self.lookback_buffer = (
            lookback_buffer
            if lookback_buffer is not None
            else LookbackBuffer(capacity_chunks=2, mode=LookbackMode.WARM_UP)
        )
        self.hangover_frames = hangover_frames
        self.decay_lambda = decay_lambda
        self.frame_duration = frame_duration

        self.state = SessionState.IDLE
        self._hangover_count = 0
        self._last_speech_time = -1000.0
        self._current_time = 0.0

    def reset(self) -> None:
        """Reset internal session controller state."""
        self.state = SessionState.IDLE
        self._hangover_count = 0
        self._last_speech_time = -1000.0
        self._current_time = 0.0
        self.vad.reset()
        self.lookback_buffer.clear()

    def compute_lazy_decay(self, elapsed_silence: float) -> float:
        """Compute exponential soft decay factor on host CPU.

        Calculates:
            alpha = exp(-lambda * max(0.0, elapsed_silence - hangover_duration))

        Args:
            elapsed_silence: Total silence duration in seconds.

        Returns:
            Scalar alpha in (0.0, 1.0].
        """
        hangover_duration = self.hangover_frames * self.frame_duration
        excess_silence = max(0.0, elapsed_silence - hangover_duration)
        alpha = math.exp(-self.decay_lambda * excess_silence)
        return float(min(1.0, max(0.0, alpha)))

    def process_frame(
        self, chunk: torch.Tensor, timestamp: float | None = None
    ) -> SessionDecision:
        """Process incoming audio frame through host session state machine.

        Args:
            chunk: Microphone audio chunk of shape (..., 320).
            timestamp: Optional wall-clock timestamp in seconds.

        Returns:
            SessionDecision with execution directives and lazy decay factor.
        """
        if timestamp is not None:
            self._current_time = timestamp
        else:
            self._current_time += self.frame_duration

        is_speech, confidence = self.vad.is_speech(chunk)

        # Case 1: Active speech detected
        if is_speech:
            self._hangover_count = self.hangover_frames
            elapsed_silence = self._current_time - self._last_speech_time
            self._last_speech_time = self._current_time

            if self.state == SessionState.IDLE:
                # Transition: IDLE -> ACTIVE (Onset triggered)
                decay_factor = self.compute_lazy_decay(elapsed_silence)
                lookback_chunks = self.lookback_buffer.get_chunks()
                self.lookback_buffer.push(chunk)

                if self.lookback_buffer.mode == LookbackMode.WARM_UP:
                    self.state = SessionState.ACTIVE
                    # Pre-warm with lookback history, then process current chunk
                    return SessionDecision(
                        action=SessionAction.WARMUP_FEED,
                        chunks_to_process=lookback_chunks + [chunk],
                        decay_factor=decay_factor,
                        state=self.state,
                        vad_confidence=confidence,
                    )
                else:
                    self.state = SessionState.ACTIVE
                    # Burst replay lookback history
                    return SessionDecision(
                        action=SessionAction.BURST_REPLAY,
                        chunks_to_process=lookback_chunks + [chunk],
                        decay_factor=decay_factor,
                        state=self.state,
                        vad_confidence=confidence,
                    )

            else:
                # Already in ACTIVE or HANGOVER -> Continues active speech
                self.state = SessionState.ACTIVE
                self.lookback_buffer.push(chunk)
                return SessionDecision(
                    action=SessionAction.PROCESS_NORMAL,
                    chunks_to_process=[chunk],
                    decay_factor=1.0,
                    state=self.state,
                    vad_confidence=confidence,
                )

        # Case 2: Silence detected by VAD
        else:
            if self.state == SessionState.ACTIVE:
                # Transition: ACTIVE -> HANGOVER
                self.state = SessionState.HANGOVER
                self._hangover_count = self.hangover_frames - 1
                self.lookback_buffer.push(chunk)
                return SessionDecision(
                    action=SessionAction.PROCESS_NORMAL,
                    chunks_to_process=[chunk],
                    decay_factor=1.0,
                    state=self.state,
                    vad_confidence=confidence,
                )

            elif self.state == SessionState.HANGOVER:
                hangover_duration = self.hangover_frames * self.frame_duration
                elapsed_since_speech = self._current_time - self._last_speech_time
                if (
                    self._hangover_count > 0
                    and elapsed_since_speech <= hangover_duration
                ):
                    self._hangover_count -= 1
                    self.lookback_buffer.push(chunk)
                    return SessionDecision(
                        action=SessionAction.PROCESS_NORMAL,
                        chunks_to_process=[chunk],
                        decay_factor=1.0,
                        state=self.state,
                        vad_confidence=confidence,
                    )
                else:
                    # Hangover expired -> Transition to IDLE
                    self.state = SessionState.IDLE
                    self._hangover_count = 0
                    self.lookback_buffer.push(chunk)
                    return SessionDecision(
                        action=SessionAction.IDLE_SKIP,
                        chunks_to_process=[],
                        decay_factor=1.0,
                        state=self.state,
                        vad_confidence=confidence,
                    )

            else:
                # Persistent IDLE: 0% GPU workload, no CUDA launches
                self.state = SessionState.IDLE
                self.lookback_buffer.push(chunk)
                return SessionDecision(
                    action=SessionAction.IDLE_SKIP,
                    chunks_to_process=[],
                    decay_factor=1.0,
                    state=self.state,
                    vad_confidence=confidence,
                )
