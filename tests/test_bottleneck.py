"""Automated test suite for Sub-Issue #10: Stateful Causal Bottleneck Pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.modules.bottleneck import CausalBottleneck


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


def test_parameter_count_under_budget() -> None:
    """Test 1: Verify bottleneck parameter count is strictly under 150K target."""
    # Standard 64-dim bottleneck
    bottleneck_64 = CausalBottleneck(
        latent_dim=128, bottleneck_dim=64, tcn_layers=4, gru_hidden=64
    )
    params_64 = bottleneck_64.count_parameters()
    assert params_64 < 150_000, f"Parameters exceeded budget: {params_64}"
    # Assert expected ~107K
    assert 90_000 <= params_64 <= 125_000

    # Ultra-compact 32-dim variant for Pareto benchmarks
    bottleneck_32 = CausalBottleneck(
        latent_dim=128, bottleneck_dim=32, tcn_layers=4, gru_hidden=32
    )
    params_32 = bottleneck_32.count_parameters()
    assert params_32 < 50_000, f"32-dim parameters exceeded budget: {params_32}"


def test_streaming_chunk_vs_sequence_equivalence() -> None:
    """Test 2: Streaming chunk-by-chunk matches batched sequence forward pass."""
    torch.manual_seed(42)
    bottleneck = CausalBottleneck(
        latent_dim=128, bottleneck_dim=64, tcn_layers=4, gru_hidden=64
    )
    bottleneck.eval()

    # 60 frames = 800ms of audio latent sequence
    batch_size = 2
    num_frames = 60
    z_seq = torch.randn(batch_size, 128, num_frames)

    # 1. Batched causal forward pass
    with torch.no_grad():
        out_seq = bottleneck.forward_sequence(z_seq)

    # 2. Sequential frame-by-frame streaming pass
    state = bottleneck.init_state(batch_size=batch_size, device=z_seq.device)
    out_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_frames):
            z_frame = z_seq[:, :, i : i + 1]
            out_frame, state = bottleneck.forward_chunk(z_frame, state)
            out_chunks.append(out_frame)

    out_stream = torch.cat(out_chunks, dim=-1)

    # 3. Invariant check: max diff must be under numerical tolerance
    max_abs_diff = (out_seq - out_stream).abs().max().item()
    mean_abs_diff = (out_seq - out_stream).abs().mean().item()

    assert max_abs_diff < 1e-5, (
        f"Streaming diverged from batch! Max diff: {max_abs_diff}"
    )
    assert mean_abs_diff < 1e-6, f"Mean diff too high: {mean_abs_diff}"


def test_state_persistence_and_reset() -> None:
    """Test 3: Verify state advances frame counter and resets properly."""
    bottleneck = CausalBottleneck(latent_dim=128, bottleneck_dim=64)
    state = bottleneck.init_state(batch_size=1)
    assert state.frame_index == 0

    z_frame = torch.randn(1, 128, 1)
    with torch.no_grad():
        _, state = bottleneck.forward_chunk(z_frame, state)
        assert state.frame_index == 1
        _, state = bottleneck.forward_chunk(z_frame, state)
        assert state.frame_index == 2

    # Reset
    state.reset()
    assert state.frame_index == 0
    assert len(state.tcn_states) == 0
    assert state.gru_state is None


def test_end_to_end_codec_bottleneck_passthrough(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 4: End-to-end audio reconstruction through EnCodec + CausalBottleneck."""
    x = speech_synthetic_24k[:, :, :9600]  # 400ms = 30 chunks
    bottleneck = CausalBottleneck(
        latent_dim=128, bottleneck_dim=64, tcn_layers=4, gru_hidden=64
    )
    bottleneck.eval()

    with torch.no_grad():
        z = encodec_candidate.model.encoder(x)
        z_hat = bottleneck.forward_sequence(z)
        y_out = encodec_candidate.model.decoder(z_hat)[:, :, : x.shape[-1]]

    assert y_out.shape == x.shape
    assert not torch.isnan(y_out).any(), "NaN in bottleneck reconstructed audio"
    assert not torch.isinf(y_out).any(), "Inf in bottleneck reconstructed audio"

    peak = y_out.abs().max().item()
    assert peak < 1.5, f"Audio peak exploded: {peak}"

    energy_in = torch.mean(x**2).item()
    energy_out = torch.mean(y_out**2).item()
    assert energy_out > 0.01 * energy_in, "Reconstructed audio collapsed"
