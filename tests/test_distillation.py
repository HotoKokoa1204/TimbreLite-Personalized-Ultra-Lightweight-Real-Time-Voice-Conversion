"""Unit tests for offline ContentVec distillation, temporal alignment, and loss."""

from __future__ import annotations

import pytest
import torch

from timbre_lite.distillation.alignment import align_teacher_features
from timbre_lite.distillation.loss import DistillationLoss, DistillationProjectionHead
from timbre_lite.distillation.teacher import ContentVecTeacher


def test_align_teacher_features_2d_and_3d() -> None:
    """Verify 1D linear interpolation aligns 50Hz features to target 75Hz frames."""
    # 2 seconds at 50Hz = 100 frames, channels = 768
    channels = 768
    time_50hz = 100
    target_75hz = 150

    # Test 3D (batch, channels, time)
    feat_3d = torch.randn(2, channels, time_50hz)
    aligned_3d = align_teacher_features(feat_3d, target_frames=target_75hz)
    assert aligned_3d.shape == (2, channels, target_75hz)

    # Test 2D (channels, time)
    feat_2d = torch.randn(channels, time_50hz)
    aligned_2d = align_teacher_features(feat_2d, target_frames=target_75hz)
    assert aligned_2d.shape == (channels, target_75hz)


def test_align_teacher_features_identity_and_errors() -> None:
    """Verify identity mapping when frames match and error on invalid inputs."""
    feat = torch.randn(1, 768, 80)
    aligned = align_teacher_features(feat, target_frames=80)
    assert aligned.shape == feat.shape
    assert torch.equal(aligned, feat)

    with pytest.raises(ValueError, match="target_frames must be positive"):
        align_teacher_features(feat, target_frames=0)

    with pytest.raises(ValueError, match="Expected 2D or 3D tensor"):
        align_teacher_features(torch.randn(1, 2, 3, 4), target_frames=10)


def test_distillation_projection_head_forward_and_grads() -> None:
    """Verify projection head maps from 64 to 768 with valid gradients."""
    head = DistillationProjectionHead(in_dim=64, out_dim=768, hidden_dim=256)
    x = torch.randn(2, 64, 45, requires_grad=True)
    out = head(x)
    assert out.shape == (2, 768, 45)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_distillation_loss_calculation() -> None:
    """Verify DistillationLoss handles perfect alignment, orthogonal targets, masks."""
    loss_fn = DistillationLoss(lambda_mse=1.0)
    batch = 2
    dim = 768
    time_frames = 50

    target = torch.randn(batch, dim, time_frames)
    # Case 1: Identical prediction
    loss_ident, m_ident = loss_fn(target, target)
    assert loss_ident.item() < 1e-5
    assert m_ident["loss_distill_cos"] < 1e-5
    assert m_ident["loss_distill_mse"] < 1e-5

    # Case 2: Inverted prediction (cosine similarity = -1 -> cos_loss = 2.0)
    inverted = -target
    loss_inv, m_inv = loss_fn(inverted, target)
    assert abs(m_inv["loss_distill_cos"] - 2.0) < 1e-4

    # Case 3: Masking out half of the frames
    mask = torch.ones(batch, time_frames)
    mask[:, 25:] = 0.0  # Zero out second half
    loss_masked, _ = loss_fn(inverted, target, mask=mask)
    assert loss_masked > 0.0


def test_contentvec_teacher_cpu_extraction() -> None:
    """Verify ContentVecTeacher extraction runs correctly on CPU device."""
    teacher = ContentVecTeacher(device="cpu")
    assert teacher.device == torch.device("cpu")

    # Pass 1 second dummy 16kHz audio
    audio = torch.randn(16000)
    features = teacher.extract_features(audio)
    assert features.ndim == 3
    assert features.shape[0] == 1
    assert features.shape[1] == 768
    # 1 second at 16kHz with 20ms hop -> ~49-50 frames
    assert 48 <= features.shape[-1] <= 52
