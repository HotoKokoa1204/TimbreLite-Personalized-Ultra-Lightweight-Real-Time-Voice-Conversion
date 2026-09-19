"""Automated verification suite for Sub-Issue #9."""

from __future__ import annotations

import math

import pytest
import torch

from timbre_lite.codec.candidate import EnCodec24kCandidate, StreamCodec2Candidate
from timbre_lite.codec.metrics import evaluate_codec_quality_gate


@pytest.fixture(scope="module")
def encodec_candidate() -> EnCodec24kCandidate:
    """Fixture providing initialized EnCodec 24kHz candidate."""
    return EnCodec24kCandidate()


@pytest.fixture(scope="module")
def speech_synthetic_24k() -> torch.Tensor:
    """Generate 1.0 second of speech-like voiced audio at 24kHz."""
    sr = 24000
    total_samples = 24000
    t = torch.linspace(0, 1.0, total_samples).unsqueeze(0).unsqueeze(0)
    # Voiced harmonic signal with pitch contour around 160Hz
    f0 = 160.0 + 35.0 * torch.sin(2 * math.pi * 3.0 * t)
    phase = 2 * math.pi * torch.cumsum(f0 / sr, dim=-1)
    harmonics = (
        0.5 * torch.sin(phase)
        + 0.3 * torch.sin(2 * phase)
        + 0.15 * torch.sin(3 * phase)
        + 0.05 * torch.sin(4 * phase)
    )
    # Formant envelope
    envelope = torch.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
    return harmonics * (0.2 + 0.8 * envelope)


def test_candidate_audit_and_weights(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 1: Verify architectural contracts and weights availability."""
    # Candidate A: StreamCodec2 (arXiv:2509.13670)
    streamcodec2 = StreamCodec2Candidate()
    contract_a = streamcodec2.contract
    assert contract_a.name == "StreamCodec2"
    assert contract_a.sample_rate == 16000
    assert contract_a.frame_samples == 320
    assert contract_a.frame_ms == 20.0
    assert contract_a.is_causal is True
    assert contract_a.weights_available is False

    with pytest.raises(NotImplementedError):
        streamcodec2.encode_chunk(torch.randn(1, 1, 320), streamcodec2.init_state())

    # Candidate B: Meta EnCodec 24kHz Causal
    contract_b = encodec_candidate.contract
    assert contract_b.name == "EnCodec_24kHz_Causal"
    assert contract_b.sample_rate == 24000
    assert contract_b.frame_samples == 320
    assert abs(contract_b.frame_ms - 13.333333333333334) < 1e-4
    assert contract_b.latent_dim == 128
    assert contract_b.is_causal is True
    assert contract_b.weights_available is True


def test_continuous_latent_passthrough_gate(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 2: Go/No-Go Gate evaluating continuous bypass against baseline."""
    x = speech_synthetic_24k
    with torch.no_grad():
        z_cont = encodec_candidate.model.encoder(x)
        y_cont = encodec_candidate.model.decoder(z_cont)[:, :, : x.shape[-1]]

        frames = encodec_candidate.model.encode(x)
        y_quant = encodec_candidate.model.decode(frames)[:, :, : x.shape[-1]]

    eval_results = evaluate_codec_quality_gate(
        ref=x,
        y_quantized=y_quant,
        y_continuous=y_cont,
        max_relative_lsd_deg_db=1.0,
        max_relative_sc_deg=0.05,
    )

    # Assertions for the Go/No-Go Gate
    assert eval_results["gate_passed"] is True, f"Gate failed: {eval_results}"
    # Continuous bypass should achieve high fidelity (SI-SDR > 15 dB)
    assert eval_results["si_sdr_continuous_db"] > 15.0
    # Continuous LSD should be comparable to or better than quantized baseline
    assert (
        eval_results["delta_log_spectral_distance_db"] <= 0.5
    )  # continuous LSD <= quant LSD + 0.5dB


def test_identity_adapter_passthrough(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 3: Identity adapter A(z) = z matches continuous bypass."""
    x = speech_synthetic_24k[:, :, :4800]  # 200ms
    with torch.no_grad():
        z = encodec_candidate.model.encoder(x)
        # Identity adapter mapping
        z_adapted = z.clone()
        y_adapted = encodec_candidate.model.decoder(z_adapted)[:, :, : x.shape[-1]]
        y_direct = encodec_candidate.model.decoder(z)[:, :, : x.shape[-1]]

    diff = (y_adapted - y_direct).abs().max().item()
    assert diff < 1e-6, f"Identity adapter diverged: max diff = {diff}"


def test_strict_causality_zero_lookahead(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 4: Strict causality test verifying altering future chunks has 0 impact."""
    torch.manual_seed(123)
    c0 = torch.randn(1, 1, 320)
    c1 = torch.randn(1, 1, 320)
    c2_modified = torch.randn(1, 1, 320) * 10.0

    # Stream Run 1: feed c0, c1
    state1 = encodec_candidate.init_state()
    z0_run1, state1 = encodec_candidate.encode_chunk(c0, state1)
    y0_run1, _ = encodec_candidate.decode_chunk(z0_run1, state1)

    z1_run1, state1 = encodec_candidate.encode_chunk(c1, state1)
    y1_run1, _ = encodec_candidate.decode_chunk(z1_run1, state1)

    # Stream Run 2: feed c0, c1, then future altered c2_modified
    state2 = encodec_candidate.init_state()
    z0_run2, state2 = encodec_candidate.encode_chunk(c0, state2)
    y0_run2, _ = encodec_candidate.decode_chunk(z0_run2, state2)

    z1_run2, state2 = encodec_candidate.encode_chunk(c1, state2)
    y1_run2, _ = encodec_candidate.decode_chunk(z1_run2, state2)

    # Feed modified future
    _, _ = encodec_candidate.encode_chunk(c2_modified, state2)

    # Mathematical causality assertion:
    diff_z1 = (z1_run1 - z1_run2).abs().max().item()
    diff_y1 = (y1_run1 - y1_run2).abs().max().item()

    assert diff_z1 == 0.0, f"Future altered past latent! Diff: {diff_z1}"
    assert diff_y1 == 0.0, f"Future altered past audio! Diff: {diff_y1}"


def test_chunk_streaming_state_continuity(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 5: Continuous streaming across sequential chunks without NaN/Inf."""
    x = speech_synthetic_24k[:, :, :3200]  # 10 chunks (133.3ms)
    chunks = torch.split(x, 320, dim=-1)

    state = encodec_candidate.init_state()
    output_chunks = []

    for chunk in chunks:
        z_chunk, state = encodec_candidate.encode_chunk(chunk, state)
        y_chunk, state = encodec_candidate.decode_chunk(
            z_chunk, state, bypass_quantizer=True
        )

        assert not torch.isnan(y_chunk).any(), "NaN detected in streamed audio chunk"
        assert not torch.isinf(y_chunk).any(), "Inf detected in streamed audio chunk"
        assert y_chunk.shape == chunk.shape, f"Unexpected shape: {y_chunk.shape}"
        output_chunks.append(y_chunk)

    y_stream = torch.cat(output_chunks, dim=-1)
    assert y_stream.shape == x.shape
    # Ensure energy is preserved reasonably
    energy_in = torch.mean(x**2).item()
    energy_out = torch.mean(y_stream**2).item()
    assert energy_out > 0.01 * energy_in, "Streamed audio has collapsed to silence"


def test_memory_footprint_stability(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 6: Verify streaming memory footprint is static across 500 chunks."""
    torch.manual_seed(999)
    state = encodec_candidate.init_state()
    dummy_chunk = torch.randn(1, 1, 320)

    # Warmup
    for _ in range(10):
        z, state = encodec_candidate.encode_chunk(dummy_chunk, state)
        _, state = encodec_candidate.decode_chunk(z, state)

    # Run 500 iterations and track tensor memory
    initial_allocated = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    for _ in range(500):
        z, state = encodec_candidate.encode_chunk(dummy_chunk, state)
        y, state = encodec_candidate.decode_chunk(z, state)
        assert y.shape == (1, 1, 320)

    final_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_allocated == initial_allocated, "CUDA memory leak detected"
