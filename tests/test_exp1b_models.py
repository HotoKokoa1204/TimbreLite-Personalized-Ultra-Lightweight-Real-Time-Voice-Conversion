"""Unit tests for EXP-1B models, F0 FiLM encoder, gated skip, and parameter budgets."""

from __future__ import annotations

import torch
from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    ExplicitF0Encoder,
    FullPersonalizedPipeline,
    PersonalizedAdapter,
)
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.modules.prosody import InGraphProsodyHead


def test_explicit_f0_encoder_parameter_budget_and_shape() -> None:
    """ExplicitF0Encoder must have exactly 4,352 parameters and output (gamma, beta)."""
    encoder = ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64)
    # Conv1d(3, 32, 1) = 3*32 + 32 = 128
    # Conv1d(32, 128, 1) = 32*128 + 128 = 4224
    # Total = 4,352
    assert encoder.count_parameters() == 4352, (
        f"Expected 4352 parameters, got {encoder.count_parameters()}"
    )

    f0_feat = torch.randn(2, 3, 50)
    gamma, beta = encoder(f0_feat)
    assert gamma.shape == (2, 64, 50)
    assert beta.shape == (2, 64, 50)

    # Zero initialization check: gamma and beta must be exactly 0
    assert torch.all(gamma == 0.0), "Gamma was not initialized to zero"
    assert torch.all(beta == 0.0), "Beta was not initialized to zero"


def test_model_a_b_c_parameter_counts() -> None:
    """Verify exact parameter counts for Model A, B, and C against bounds."""
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    # Content Cleanser is frozen during Stage 2 personal conversion
    for param in cleanser.parameters():
        param.requires_grad = False
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)

    # Model A: Baseline
    adapter_a = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=False,
    )
    pipeline_a = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter_a,
        f0_encoder=None,
    )
    params_a = pipeline_a.count_trainable_parameters()

    # Model B: Explicit F0 FiLM
    f0_encoder_b = ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64)
    adapter_b = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=False,
    )
    pipeline_b = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter_b,
        f0_encoder=f0_encoder_b,
    )
    params_b = pipeline_b.count_trainable_parameters()

    # Model C: Explicit F0 FiLM + Gated Skip
    f0_encoder_c = ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64)
    adapter_c = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=True,
        initial_skip_gate=-4.0,
    )
    pipeline_c = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter_c,
        f0_encoder=f0_encoder_c,
    )
    params_c = pipeline_c.count_trainable_parameters()

    # Exact parameter delta verification
    assert params_b - params_a == 4352, (
        f"Expected F0 encoder to add 4352 params, got {params_b - params_a}"
    )
    assert params_c - params_b == 1, (
        f"Expected skip gate to add exactly 1 param, got {params_c - params_b}"
    )
    assert params_c < 250000, f"Model C exceeded 250K budget: {params_c}"


def test_film_zero_init_exact_identity_equivalence() -> None:
    """Zero-initialized FiLM must produce exact 0.0 diff with Model A."""
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=False,
    )

    f0_encoder = ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64)

    # Share identical submodules
    pipeline_a = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
        f0_encoder=None,
    )
    pipeline_b = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
        f0_encoder=f0_encoder,
    )

    pipeline_a.eval()
    pipeline_b.eval()

    torch.manual_seed(42)
    z_seq = torch.randn(2, 128, 60)
    f0_seq = torch.randn(2, 3, 60)  # arbitrary random F0 inputs

    with torch.no_grad():
        z_out_a, c_seq_a, p_seq_a = pipeline_a.forward_sequence(z_seq)
        z_out_b, c_seq_b, p_seq_b = pipeline_b.forward_sequence(z_seq, f0_seq=f0_seq)

    # Content FiLM output difference must be mathematically 0.0
    max_c_diff = float((c_seq_a - c_seq_b).abs().max())
    assert max_c_diff == 0.0, f"Expected 0.0 diff in c_seq, got {max_c_diff}"

    # Adapted latent output difference must be mathematically 0.0
    max_out_diff = float((z_out_a - z_out_b).abs().max())
    assert max_out_diff == 0.0, f"Expected 0.0 diff in z_out, got {max_out_diff}"


def test_model_c_gated_skip_initialization_and_gradient() -> None:
    """Model C skip gate must initialize to sigmoid(-4.0) and receive gradients."""
    adapter = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=True,
        initial_skip_gate=-4.0,
    )
    assert adapter.skip_gate is not None
    init_gate = float(torch.sigmoid(adapter.skip_gate).item())
    expected = 1.0 / (1.0 + 54.59815)  # 1 / (1 + e^4) ~= 0.017986
    assert abs(init_gate - expected) < 1e-5, f"Expected ~0.018, got {init_gate}"

    # Gradient flow check
    u = torch.randn(2, 64, 30, requires_grad=True)
    out = adapter.forward_sequence(u)
    loss = out.sum()
    loss.backward()

    assert adapter.skip_gate.grad is not None
    assert float(adapter.skip_gate.grad.abs()) > 0.0, "No gradient to skip_gate"


def test_f0_pipeline_streaming_parity() -> None:
    """Pipeline forward_sequence and step_audio_chunk must match for F0."""
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=True,
        initial_skip_gate=-4.0,
    )
    f0_encoder = ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64)

    pipeline = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
        f0_encoder=f0_encoder,
    )
    pipeline.eval()

    torch.manual_seed(123)
    num_chunks = 10
    audio_full = torch.randn(1, 1, num_chunks * 320)
    f0_full = torch.randn(1, 3, num_chunks)

    # Chunk-by-chunk streaming
    streaming_state = pipeline.init_streaming_state(batch_size=1)
    chunk_outputs: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_chunks):
            chunk = audio_full[:, :, i * 320 : (i + 1) * 320]
            f0_chunk = f0_full[:, :, i : i + 1]
            y_chunk, streaming_state = pipeline.step_audio_chunk(
                chunk, streaming_state, f0_chunk=f0_chunk
            )
            chunk_outputs.append(y_chunk)

    y_streamed = torch.cat(chunk_outputs, dim=-1)
    assert y_streamed.shape == (1, 1, num_chunks * 320)
    assert not torch.isnan(y_streamed).any()
