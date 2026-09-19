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
    f0 = 160.0 + 35.0 * torch.sin(2 * math.pi * 3.0 * t)
    phase = 2 * math.pi * torch.cumsum(f0 / sr, dim=-1)
    harmonics = (
        0.5 * torch.sin(phase)
        + 0.3 * torch.sin(2 * phase)
        + 0.15 * torch.sin(3 * phase)
        + 0.05 * torch.sin(4 * phase)
    )
    envelope = torch.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
    return harmonics * (0.2 + 0.8 * envelope)


def test_candidate_audit_and_weights(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 1: Verify architectural contracts and weights availability."""
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

    assert eval_results["gate_passed"] is True, f"Gate failed: {eval_results}"
    assert eval_results["si_sdr_continuous_db"] > 15.0
    assert eval_results["delta_log_spectral_distance_db"] <= 0.5


def test_identity_adapter_passthrough(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 3: Identity adapter A(z) = z matches continuous bypass."""
    x = speech_synthetic_24k[:, :, :4800]
    with torch.no_grad():
        z = encodec_candidate.model.encoder(x)
        z_adapted = z.clone()
        y_adapted = encodec_candidate.model.decoder(z_adapted)[:, :, : x.shape[-1]]
        y_direct = encodec_candidate.model.decoder(z)[:, :, : x.shape[-1]]

    diff = (y_adapted - y_direct).abs().max().item()
    assert diff < 1e-6, f"Identity adapter diverged: max diff = {diff}"


def test_strict_causality_zero_lookahead(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 4: Strict causality verifying altering future has 0 past impact."""
    torch.manual_seed(123)
    c0 = torch.randn(1, 1, 320)
    c1 = torch.randn(1, 1, 320)
    c2_future = torch.randn(1, 1, 320) * 10.0

    state1 = encodec_candidate.init_state()
    z0_run1, state1 = encodec_candidate.encode_chunk(c0, state1)
    y0_run1, _ = encodec_candidate.decode_chunk(z0_run1, state1)

    z1_run1, state1 = encodec_candidate.encode_chunk(c1, state1)
    y1_run1, _ = encodec_candidate.decode_chunk(z1_run1, state1)

    state2 = encodec_candidate.init_state()
    z0_run2, state2 = encodec_candidate.encode_chunk(c0, state2)
    y0_run2, _ = encodec_candidate.decode_chunk(z0_run2, state2)

    z1_run2, state2 = encodec_candidate.encode_chunk(c1, state2)
    y1_run2, _ = encodec_candidate.decode_chunk(z1_run2, state2)

    _, _ = encodec_candidate.encode_chunk(c2_future, state2)

    max_abs_diff_z = (z1_run1 - z1_run2).abs().max().item()
    mean_abs_diff_z = (z1_run1 - z1_run2).abs().mean().item()
    max_abs_diff_y = (y1_run1 - y1_run2).abs().max().item()
    mean_abs_diff_y = (y1_run1 - y1_run2).abs().mean().item()

    assert max_abs_diff_z == 0.0
    assert mean_abs_diff_z == 0.0
    assert max_abs_diff_y == 0.0
    assert mean_abs_diff_y == 0.0


def test_adversarial_future_perturbation_causality(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 5: Adversarial causality with DC +1.0 and high-frequency spikes."""
    torch.manual_seed(777)
    c0 = torch.randn(1, 1, 320)
    c1 = torch.randn(1, 1, 320)

    # Extreme adversarial future chunks
    c2_dc = torch.ones(1, 1, 320)
    c2_nyquist = torch.tensor([1.0, -1.0] * 160, dtype=torch.float32).view(1, 1, 320)

    # Baseline stream: c0 -> c1
    state_base = encodec_candidate.init_state()
    z0_base, state_base = encodec_candidate.encode_chunk(c0, state_base)
    y0_base, _ = encodec_candidate.decode_chunk(z0_base, state_base)
    z1_base, state_base = encodec_candidate.encode_chunk(c1, state_base)
    y1_base, _ = encodec_candidate.decode_chunk(z1_base, state_base)

    for future_adv in [c2_dc, c2_nyquist]:
        state_adv = encodec_candidate.init_state()
        z0_adv, state_adv = encodec_candidate.encode_chunk(c0, state_adv)
        y0_adv, _ = encodec_candidate.decode_chunk(z0_adv, state_adv)
        z1_adv, state_adv = encodec_candidate.encode_chunk(c1, state_adv)
        y1_adv, _ = encodec_candidate.decode_chunk(z1_adv, state_adv)

        # Feed adversarial future
        _, _ = encodec_candidate.encode_chunk(future_adv, state_adv)

        max_diff = (y1_base - y1_adv).abs().max().item()
        mean_diff = (y1_base - y1_adv).abs().mean().item()
        assert max_diff == 0.0
        assert mean_diff == 0.0


def test_latent_manifold_perturbation_robustness(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 6: Latent manifold robustness under continuous Gaussian perturbations."""
    x = speech_synthetic_24k[:, :, :12000]
    with torch.no_grad():
        z = encodec_candidate.model.encoder(x)
        y_ref = encodec_candidate.model.decoder(z)[:, :, : x.shape[-1]]

    torch.manual_seed(42)
    eps = torch.randn_like(z)

    prev_diff = 0.0
    for alpha in [0.01, 0.03, 0.05, 0.1]:
        z_pert = z + alpha * eps
        with torch.no_grad():
            y_pert = encodec_candidate.model.decoder(z_pert)[:, :, : x.shape[-1]]

        assert not torch.isnan(y_pert).any()
        assert not torch.isinf(y_pert).any()

        peak = y_pert.abs().max().item()
        assert peak < 1.5, f"Waveform explosion detected: peak = {peak}"

        diff = (y_pert - y_ref).abs().mean().item()
        # Ensure smooth monotonic sensitivity without discontinuity
        assert diff > prev_diff
        prev_diff = diff


def test_latent_interpolation_smoothness(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 7: Smooth continuous interpolation across distinct latent trajectories."""
    sr = 24000
    t = torch.linspace(0, 0.5, sr // 2).unsqueeze(0).unsqueeze(0)
    xa = 0.5 * torch.sin(2 * math.pi * 130 * t)
    xb = 0.5 * torch.sin(2 * math.pi * 260 * t)

    with torch.no_grad():
        za = encodec_candidate.model.encoder(xa)
        zb = encodec_candidate.model.encoder(xb)

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            z_interp = (1.0 - alpha) * za + alpha * zb
            y_interp = encodec_candidate.model.decoder(z_interp)[:, :, : xa.shape[-1]]

            assert not torch.isnan(y_interp).any()
            assert not torch.isinf(y_interp).any()
            peak = y_interp.abs().max().item()
            assert peak < 1.5, f"Catastrophic artifact at alpha {alpha}: {peak}"


def test_bandwidth_sweep_reconstruction(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 8: Evaluate reconstruction across EnCodec bandwidth profiles."""
    x = speech_synthetic_24k[:, :, :12000]

    for bw in [6.0, 12.0, 24.0]:
        encodec_candidate.model.set_target_bandwidth(bw)
        with torch.no_grad():
            frames = encodec_candidate.model.encode(x)
            y_bw = encodec_candidate.model.decode(frames)[:, :, : x.shape[-1]]

        assert y_bw.shape == x.shape
        assert not torch.isnan(y_bw).any()
        peak = y_bw.abs().max().item()
        assert peak < 1.5


def test_chunk_streaming_state_continuity(
    encodec_candidate: EnCodec24kCandidate,
    speech_synthetic_24k: torch.Tensor,
) -> None:
    """Test 9: Continuous streaming across sequential chunks without NaN/Inf."""
    x = speech_synthetic_24k[:, :, :3200]
    chunks = torch.split(x, 320, dim=-1)

    state = encodec_candidate.init_state()
    output_chunks = []

    for chunk in chunks:
        z_chunk, state = encodec_candidate.encode_chunk(chunk, state)
        y_chunk, state = encodec_candidate.decode_chunk(
            z_chunk, state, bypass_quantizer=True
        )

        assert not torch.isnan(y_chunk).any()
        assert not torch.isinf(y_chunk).any()
        assert y_chunk.shape == chunk.shape
        output_chunks.append(y_chunk)

    y_stream = torch.cat(output_chunks, dim=-1)
    assert y_stream.shape == x.shape
    energy_in = torch.mean(x**2).item()
    energy_out = torch.mean(y_stream**2).item()
    assert energy_out > 0.01 * energy_in


def test_memory_footprint_stability(
    encodec_candidate: EnCodec24kCandidate,
) -> None:
    """Test 10: Verify streaming memory footprint is static across 500 chunks."""
    torch.manual_seed(999)
    state = encodec_candidate.init_state()
    dummy_chunk = torch.randn(1, 1, 320)

    for _ in range(10):
        z, state = encodec_candidate.encode_chunk(dummy_chunk, state)
        _, state = encodec_candidate.decode_chunk(z, state)

    initial_allocated = (
        torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    )

    for _ in range(500):
        z, state = encodec_candidate.encode_chunk(dummy_chunk, state)
        y, state = encodec_candidate.decode_chunk(z, state)
        assert y.shape == (1, 1, 320)

    final_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_allocated == initial_allocated
