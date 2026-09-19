"""Automated test suite for Sub-Issue #13: Host Session Controller."""

from __future__ import annotations

import math

import torch

from timbre_lite.controller.buffer import LookbackBuffer, LookbackMode
from timbre_lite.controller.session import (
    SessionAction,
    SessionController,
    SessionState,
)
from timbre_lite.controller.vad import TwoStageVAD


def generate_voiced_speech(duration_sec: float = 0.5) -> torch.Tensor:
    """Generate synthetic voiced speech signal with harmonics."""
    sr = 24000
    samples = int(sr * duration_sec)
    t = torch.linspace(0, duration_sec, samples).unsqueeze(0).unsqueeze(0)
    f0 = 160.0
    phase = 2 * math.pi * f0 * t
    harmonics = (
        0.6 * torch.sin(phase)
        + 0.3 * torch.sin(2 * phase)
        + 0.15 * torch.sin(3 * phase)
    )
    return harmonics * 0.4


def test_two_stage_vad_speech_vs_silence() -> None:
    """Test 1: Verify TwoStageVAD discriminates voiced speech from silence/noise."""
    vad = TwoStageVAD(sample_rate=24000, energy_threshold_db=-45.0)

    # 1. Complete silence
    silence = torch.zeros(1, 1, 320)
    is_speech, conf = vad.is_speech(silence)
    assert not is_speech
    assert conf == 0.0

    # 2. Quiet room noise (-60 dB)
    quiet_noise = torch.randn(1, 1, 320) * 0.0005
    is_speech, _ = vad.is_speech(quiet_noise)
    assert not is_speech

    # 3. Voiced speech chunk
    voiced = generate_voiced_speech(0.05)[:, :, :320]
    is_speech, conf = vad.is_speech(voiced)
    assert is_speech
    assert conf > 0.3


def test_lookback_buffer_circular_retention() -> None:
    """Test 2: Verify LookbackBuffer retains exactly the latest N frames in order."""
    buffer = LookbackBuffer(capacity_chunks=2)
    assert buffer.current_size == 0

    c0 = torch.tensor([[[0.0]]])
    c1 = torch.tensor([[[1.0]]])
    c2 = torch.tensor([[[2.0]]])

    buffer.push(c0)
    assert buffer.current_size == 1
    buffer.push(c1)
    assert buffer.current_size == 2

    chunks = buffer.get_chunks()
    assert torch.equal(chunks[0], c0)
    assert torch.equal(chunks[1], c1)

    # Push 3rd chunk, c0 should be evicted
    buffer.push(c2)
    assert buffer.current_size == 2
    chunks = buffer.get_chunks()
    assert torch.equal(chunks[0], c1)
    assert torch.equal(chunks[1], c2)


def test_zero_gpu_silence_gating() -> None:
    """Test 3: Verify 0% GPU workload during silence (IDLE_SKIP directives)."""
    controller = SessionController()
    silence = torch.zeros(1, 1, 320)

    # Feed 10 silent frames
    for _ in range(10):
        decision = controller.process_frame(silence)
        assert decision.action == SessionAction.IDLE_SKIP
        assert len(decision.chunks_to_process) == 0
        assert decision.state == SessionState.IDLE


def test_onset_lookback_warmup_and_burst_modes() -> None:
    """Test 4: Verify speech onset triggers lookback delivery in Modes A and B."""
    voiced = generate_voiced_speech(0.05)[:, :, :320]
    noise = torch.randn(1, 1, 320) * 0.0001

    # Mode A: WARM_UP
    ctrl_warmup = SessionController(
        lookback_buffer=LookbackBuffer(capacity_chunks=2, mode=LookbackMode.WARM_UP)
    )
    ctrl_warmup.process_frame(noise)
    ctrl_warmup.process_frame(noise)
    decision = ctrl_warmup.process_frame(voiced)
    assert decision.action == SessionAction.WARMUP_FEED
    assert len(decision.chunks_to_process) == 3  # 2 lookback + 1 current
    assert decision.state == SessionState.ACTIVE

    # Mode B: BURST_REPLAY
    ctrl_burst = SessionController(
        lookback_buffer=LookbackBuffer(
            capacity_chunks=2, mode=LookbackMode.BURST_REPLAY
        )
    )
    ctrl_burst.process_frame(noise)
    ctrl_burst.process_frame(noise)
    decision = ctrl_burst.process_frame(voiced)
    assert decision.action == SessionAction.BURST_REPLAY
    assert len(decision.chunks_to_process) == 3  # 2 lookback + 1 current
    assert decision.state == SessionState.ACTIVE


def test_hangover_bridges_intra_phrase_micropause() -> None:
    """Test 5: Verify hangover preserves active timbre across 100ms micro-pause."""
    controller = SessionController(hangover_frames=8)  # 8 * 13.33ms = 106.7ms
    voiced = generate_voiced_speech(0.05)[:, :, :320]
    silence = torch.zeros(1, 1, 320)

    # 1. First phrase callout ("Enemy B!")
    decision = controller.process_frame(voiced)
    assert decision.state == SessionState.ACTIVE

    # 2. 80ms micro-pause (6 frames of silence)
    for _ in range(6):
        decision = controller.process_frame(silence)
        assert decision.state == SessionState.HANGOVER
        assert decision.action == SessionAction.PROCESS_NORMAL
        assert decision.decay_factor == 1.0  # Timbre state fully intact!

    # 3. Second phrase callout ("Two of them!")
    decision = controller.process_frame(voiced)
    assert decision.state == SessionState.ACTIVE
    assert decision.action == SessionAction.PROCESS_NORMAL
    assert decision.decay_factor == 1.0  # Zero reset artifact!


def test_prolonged_silence_lazy_decay() -> None:
    """Test 6: Verify prolonged silence triggers smooth lazy decay on resumption."""
    controller = SessionController(hangover_frames=8, decay_lambda=2.0)
    voiced = generate_voiced_speech(0.05)[:, :, :320]
    silence = torch.zeros(1, 1, 320)

    # Active speech at t = 0.0s
    controller.process_frame(voiced, timestamp=0.0)

    # Prolonged silence: 2.0 seconds later
    # Hangover expires after ~0.106s, silence continues in IDLE
    controller.process_frame(silence, timestamp=0.5)
    decision_idle = controller.process_frame(silence, timestamp=1.5)
    assert decision_idle.state == SessionState.IDLE
    assert decision_idle.action == SessionAction.IDLE_SKIP

    # Resumption at t = 2.0s
    decision_resume = controller.process_frame(voiced, timestamp=2.0)
    assert decision_resume.state == SessionState.ACTIVE
    # elapsed silence = 2.0 - 0.0 = 2.0s. Excess silence = 2.0 - 0.106 = ~1.89s
    # decay = exp(-2.0 * 1.89) = exp(-3.78) ~ 0.0227
    assert 0.01 <= decision_resume.decay_factor <= 0.05
