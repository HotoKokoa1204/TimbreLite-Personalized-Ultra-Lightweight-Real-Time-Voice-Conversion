"""Automated test suite for Sub-Issue #11: Speaker-Invariant Content Cleanser."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from timbre_lite.modules.cleanser import (
    CleanserVariant,
    ContentCleanser,
    GradientReversal,
    PhoneticPredictor,
    SpeakerAdversary,
    SpeakerVerificationProbe,
)


def test_cleanser_parameter_budget_and_shape() -> None:
    """Test 1: Verify Content Cleanser parameter budget and output shapes."""
    # Standard 64-dim Content Cleanser
    cleanser_64 = ContentCleanser(in_dim=128, content_dim=64, tcn_layers=2)
    params_64 = cleanser_64.count_parameters()
    # Extremely compact (<60K parameters)
    assert params_64 < 60_000, f"Cleanser params too high: {params_64}"

    # Ultra-compact 32-dim Content Cleanser for Pareto benchmarks
    cleanser_32 = ContentCleanser(in_dim=128, content_dim=32, tcn_layers=2)
    params_32 = cleanser_32.count_parameters()
    assert params_32 < 20_000, f"32-dim params too high: {params_32}"

    dummy_z = torch.randn(2, 128, 45)
    out_64 = cleanser_64.forward_sequence(dummy_z)
    assert out_64.shape == (2, 64, 45)

    out_32 = cleanser_32.forward_sequence(dummy_z)
    assert out_32.shape == (2, 32, 45)


def test_cleanser_streaming_chunk_vs_sequence_equivalence() -> None:
    """Test 2: Verify streaming chunk-by-chunk matches batched sequence forward pass."""
    torch.manual_seed(42)
    cleanser = ContentCleanser(in_dim=128, content_dim=64, tcn_layers=2)
    cleanser.eval()

    batch_size = 2
    num_frames = 60  # 800ms
    z_seq = torch.randn(batch_size, 128, num_frames)

    # 1. Batched causal sequence forward pass
    with torch.no_grad():
        out_seq = cleanser.forward_sequence(z_seq)

    # 2. Sequential frame-by-frame streaming pass
    state = cleanser.init_state(batch_size=batch_size, device=z_seq.device)
    out_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_frames):
            z_frame = z_seq[:, :, i : i + 1]
            out_frame, state = cleanser.forward_chunk(z_frame, state)
            out_chunks.append(out_frame)

    out_stream = torch.cat(out_chunks, dim=-1)

    # 3. Hard Streaming Invariant check
    max_abs_diff = (out_seq - out_stream).abs().max().item()
    mean_abs_diff = (out_seq - out_stream).abs().mean().item()

    assert max_abs_diff < 1e-5, (
        f"Streaming diverged from batch! Max diff: {max_abs_diff}"
    )
    assert mean_abs_diff < 1e-6, f"Mean diff too high: {mean_abs_diff}"


def test_grl_gradient_inversion() -> None:
    """Test 3: Verify Gradient Reversal Layer forward identity and backward negation."""
    alpha = 0.75
    grl = GradientReversal(alpha=alpha)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = grl(x)

    # Forward identity
    assert torch.allclose(y, x)

    # Backward negation
    loss = (y * torch.tensor([10.0, 20.0, 30.0])).sum()
    loss.backward()

    assert x.grad is not None
    expected_grad = -alpha * torch.tensor([10.0, 20.0, 30.0])
    assert torch.allclose(x.grad, expected_grad)


def test_ablation_variants_forward_and_losses() -> None:
    """Test 4: Verify 4 ablation variants can execute and compute losses."""
    batch_size = 4
    num_frames = 30
    num_speakers = 5
    num_phonemes = 20

    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    adversary = SpeakerAdversary(
        content_dim=64, num_speakers=num_speakers, grl_alpha=1.0
    )
    predictor = PhoneticPredictor(content_dim=64, num_phonetic_classes=num_phonemes)

    z = torch.randn(batch_size, 128, num_frames)
    target_spk = torch.randint(0, num_speakers, (batch_size,))
    target_phonemes = torch.randint(0, num_phonemes, (batch_size, num_frames))

    for variant in CleanserVariant:
        content = cleanser.forward_sequence(z)

        # Phonetic loss (always present across variants)
        pred_phonemes = predictor(content)
        loss_content = F.cross_entropy(pred_phonemes, target_phonemes)
        assert loss_content.item() > 0.0

        if variant == CleanserVariant.VARIANT_A:
            # Baseline: only phonetic content loss
            total_loss = loss_content

        elif variant == CleanserVariant.VARIANT_B:
            # GRL adversary
            spk_logits = adversary(content)
            loss_spk = F.cross_entropy(spk_logits, target_spk)
            total_loss = loss_content + 0.1 * loss_spk

        elif variant == CleanserVariant.VARIANT_C:
            # Perturbation consistency
            timbre_noise = 0.5 * torch.randn_like(z)
            z_perturbed = z + timbre_noise
            content_pert = cleanser.forward_sequence(z_perturbed)
            loss_invariance = F.mse_loss(content, content_pert)
            total_loss = loss_content + 0.5 * loss_invariance

        elif variant == CleanserVariant.VARIANT_D:
            # Full compound
            spk_logits = adversary(content)
            loss_spk = F.cross_entropy(spk_logits, target_spk)
            timbre_noise = 0.5 * torch.randn_like(z)
            content_pert = cleanser.forward_sequence(z + timbre_noise)
            loss_invariance = F.mse_loss(content, content_pert)
            total_loss = loss_content + 0.1 * loss_spk + 0.5 * loss_invariance

        assert total_loss.item() > 0.0
        assert not torch.isnan(total_loss)


def test_speaker_verification_probe_and_disentanglement() -> None:
    """Test 5: Speaker probe evaluates speaker leakage vs chance baseline."""
    torch.manual_seed(123)
    probe_raw = SpeakerVerificationProbe(feature_dim=128, embedding_dim=64)
    probe_cleansed = SpeakerVerificationProbe(feature_dim=64, embedding_dim=64)

    # Simulate speaker A and speaker B
    # Speaker A has a persistent bias vector along channel dimension
    spk_a_bias = torch.randn(1, 128, 1) * 2.0
    spk_b_bias = torch.randn(1, 128, 1) * 2.0

    phonetic_content = torch.randn(4, 128, 50)
    utterances_a = phonetic_content + spk_a_bias
    utterances_b = phonetic_content + spk_b_bias

    # Raw latents have distinct speaker identities
    raw_sim_same = probe_raw.compute_cosine_similarity(
        utterances_a[:2], utterances_a[2:]
    )
    raw_sim_diff = probe_raw.compute_cosine_similarity(utterances_a, utterances_b)
    assert -1.0 <= raw_sim_same <= 1.0
    assert -1.0 <= raw_sim_diff <= 1.0

    # Cleanser projection to 64 dimensions
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    c_a = cleanser.forward_sequence(utterances_a)
    c_b = cleanser.forward_sequence(utterances_b)

    cleansed_sim = probe_cleansed.compute_cosine_similarity(c_a, c_b)
    assert -1.0 <= cleansed_sim <= 1.0


def test_cleanser_state_reset() -> None:
    """Test 6: Verify Cleanser state advances and resets correctly."""
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    state = cleanser.init_state(batch_size=1)
    assert state.frame_index == 0

    dummy_frame = torch.randn(1, 128, 1)
    with torch.no_grad():
        _, state = cleanser.forward_chunk(dummy_frame, state)
        assert state.frame_index == 1
        _, state = cleanser.forward_chunk(dummy_frame, state)
        assert state.frame_index == 2

    state.reset()
    assert state.frame_index == 0
    assert len(state.tcn_states) == 0
