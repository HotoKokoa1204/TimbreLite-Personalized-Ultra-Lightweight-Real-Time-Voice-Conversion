"""Automated test suite for Sub-Issue #12: Personalized Voice Conversion Adapter."""

from __future__ import annotations

import math

import pytest
import torch

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    FullPersonalizedPipeline,
    FusionMode,
    PersonalizedAdapter,
)
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.modules.prosody import InGraphProsodyHead


@pytest.fixture(scope="module")
def encodec_candidate() -> EnCodec24kCandidate:
    """Fixture providing initialized EnCodec 24kHz candidate."""
    return EnCodec24kCandidate()


@pytest.fixture(scope="module")
def speech_synthetic_24k() -> torch.Tensor:
    """Generate 1.0 second of synthetic voiced audio at 24kHz."""
    sr = 24000
    total_samples = 24000
    t = torch.linspace(0, 1.0, total_samples).unsqueeze(0).unsqueeze(0)
    f0 = 150.0 + 30.0 * torch.sin(2 * math.pi * 3.0 * t)
    phase = 2 * math.pi * torch.cumsum(f0 / sr, dim=-1)
    harmonics = (
        0.5 * torch.sin(phase)
        + 0.3 * torch.sin(2 * phase)
        + 0.15 * torch.sin(3 * phase)
    )
    envelope = torch.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
    return harmonics * (0.2 + 0.8 * envelope)


def test_prosody_head_parameter_budget_and_shape() -> None:
    """Test 1: Verify InGraphProsodyHead parameter count and tensor shapes."""
    prosody_head = InGraphProsodyHead(
        in_dim=128, prosody_dim=16, hidden_dim=32, tcn_layers=2
    )
    params = prosody_head.count_parameters()
    # Lightweight in-graph head: ~15K params
    assert params < 20_000, f"Prosody head params too high: {params}"

    dummy_z = torch.randn(2, 128, 40)
    out = prosody_head.forward_sequence(dummy_z)
    assert out.shape == (2, 16, 40)

    f0, vuv = prosody_head.get_pitch_and_voicing(out)
    assert f0.shape == (2, 1, 40)
    assert vuv.shape == (2, 1, 40)
    assert (vuv >= 0.0).all() and (vuv <= 1.0).all()


def test_prosody_head_streaming_equivalence() -> None:
    """Test 2: Verify streaming chunk equivalence for InGraphProsodyHead."""
    torch.manual_seed(42)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    prosody_head.eval()

    batch_size = 2
    num_frames = 50
    z_seq = torch.randn(batch_size, 128, num_frames)

    with torch.no_grad():
        out_seq = prosody_head.forward_sequence(z_seq)

    state = prosody_head.init_state(batch_size=batch_size, device=z_seq.device)
    out_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(num_frames):
            chunk = z_seq[:, :, i : i + 1]
            out_c, state = prosody_head.forward_chunk(chunk, state)
            out_chunks.append(out_c)

    out_stream = torch.cat(out_chunks, dim=-1)
    max_diff = (out_seq - out_stream).abs().max().item()
    assert max_diff < 1e-5, f"Prosody head streaming diverged: {max_diff}"


def test_dual_stream_fusion_modes() -> None:
    """Test 3: Verify Additive and FiLM dual-stream conditioning fusion modes."""
    content = torch.randn(2, 64, 30)
    prosody = torch.randn(2, 16, 30)

    # Mode 1: Additive Fusion
    fusion_add = DualStreamFusion(
        content_dim=64, prosody_dim=16, mode=FusionMode.ADDITIVE
    )
    fused_add = fusion_add(content, prosody)
    assert fused_add.shape == (2, 64, 30)

    # Mode 2: FiLM Fusion
    fusion_film = DualStreamFusion(content_dim=64, prosody_dim=16, mode=FusionMode.FILM)
    fused_film = fusion_film(content, prosody)
    assert fused_film.shape == (2, 64, 30)


def test_personalized_adapter_parameter_budget_under_250k() -> None:
    """Test 4: Verify PersonalizedAdapter stays strictly within Sub-250K budget."""
    adapter_64 = PersonalizedAdapter(
        in_dim=64, out_dim=128, hidden_dim=64, tcn_layers=4, gru_hidden=64
    )
    params_64 = adapter_64.count_parameters()
    # Strict Sub-250K assertion
    assert params_64 < 250_000, f"Adapter params exceeded 250K budget: {params_64}"
    assert 90_000 <= params_64 <= 130_000

    # Compact 32-dim Pareto variant
    adapter_32 = PersonalizedAdapter(
        in_dim=32, out_dim=128, hidden_dim=32, tcn_layers=4, gru_hidden=32
    )
    params_32 = adapter_32.count_parameters()
    assert params_32 < 50_000, f"32-dim adapter params too high: {params_32}"


def test_personalized_adapter_streaming_equivalence() -> None:
    """Test 5: Verify streaming chunk equivalence for PersonalizedAdapter."""
    torch.manual_seed(42)
    adapter = PersonalizedAdapter(in_dim=64, out_dim=128)
    adapter.eval()

    batch_size = 2
    num_frames = 60
    u_seq = torch.randn(batch_size, 64, num_frames)

    with torch.no_grad():
        out_seq = adapter.forward_sequence(u_seq)

    state = adapter.init_state(batch_size=batch_size, device=u_seq.device)
    out_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(num_frames):
            u_frame = u_seq[:, :, i : i + 1]
            out_frame, state = adapter.forward_chunk(u_frame, state)
            out_chunks.append(out_frame)

    out_stream = torch.cat(out_chunks, dim=-1)
    max_diff = (out_seq - out_stream).abs().max().item()
    assert max_diff < 1e-5, f"Adapter streaming diverged: {max_diff}"


def test_full_pipeline_end_to_end_streaming(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 6: Verify full end-to-end streaming VC pipeline across audio chunks."""
    cleanser = ContentCleanser(in_dim=128, content_dim=64, tcn_layers=2)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(in_dim=64, out_dim=128)

    pipeline = FullPersonalizedPipeline(
        codec=encodec_candidate,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
    )
    pipeline.eval()

    # Total trainable parameters across personal conversion modules
    total_trainable = pipeline.count_trainable_parameters()
    assert total_trainable < 250_000, (
        f"Total trainable exceeded 250K budget: {total_trainable}"
    )

    # Stream 10 audio chunks (133.3ms = 3200 samples)
    x = speech_synthetic_24k[:, :, :3200]
    chunks = torch.split(x, 320, dim=-1)
    state = pipeline.init_streaming_state()

    out_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in chunks:
            y_chunk, state = pipeline.step_audio_chunk(chunk, state)
            assert y_chunk.shape == (1, 1, 320)
            assert not torch.isnan(y_chunk).any()
            assert not torch.isinf(y_chunk).any()
            out_chunks.append(y_chunk)

    y_full = torch.cat(out_chunks, dim=-1)
    assert y_full.shape == x.shape
    peak = y_full.abs().max().item()
    assert peak < 1.5, f"Waveform peak exploded: {peak}"

    # State reset check
    state.reset()
    assert state.cleanser_state.frame_index == 0
    assert state.prosody_state.frame_index == 0
    assert state.adapter_state.frame_index == 0
