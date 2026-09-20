"""Distillation projection head and loss objectives for student phonetic learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class DistillationProjectionHead(nn.Module):
    """Projection head mapping student features to teacher dimension.

    Maps 64-dimensional Content Cleanser representations to the 768-dimensional
    ContentVec phonetic manifold during training. This module is strictly
    discarded during live streaming inference.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize DistillationProjectionHead.

        Args:
            in_dim: Student content dimension (64).
            out_dim: Teacher phonetic feature dimension (768).
            hidden_dim: Intermediate projection dimension (256).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=1),
            nn.GroupNorm(1, out_dim),
        )

    def forward(self, student_content: torch.Tensor) -> torch.Tensor:
        """Project student content into teacher representation space.

        Args:
            student_content: Tensor of shape (batch, in_dim, time).

        Returns:
            Projected representation of shape (batch, out_dim, time).
        """
        out: torch.Tensor = self.net(student_content)
        return out


class DistillationLoss(nn.Module):
    """Combined Cosine Similarity and MSE Distillation Loss.

    Guides the student model to replicate the teacher's phonetic trajectory:
        L_distill = (1 - cos_sim(student, teacher)) + lambda_mse * MSE
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        """Initialize DistillationLoss.

        Args:
            lambda_mse: Weight scalar for the Mean Squared Error term.
            eps: Small epsilon to prevent division by zero.
        """
        super().__init__()
        self.lambda_mse = lambda_mse
        self.eps = eps

    def forward(
        self,
        student_proj: torch.Tensor,
        teacher_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate combined distillation loss.

        Args:
            student_proj: Projected student tensor of shape (batch, channels, time).
            teacher_target: Target teacher tensor of shape (batch, channels, time).
            mask: Optional boolean or float mask of shape (batch, time).

        Returns:
            Tuple of (total_loss_tensor, loss_metrics_dict).
        """
        if student_proj.shape != teacher_target.shape:
            # If temporal lengths differ slightly due to rounding, truncate to shorter
            min_len = min(student_proj.shape[-1], teacher_target.shape[-1])
            student_proj = student_proj[..., :min_len]
            teacher_target = teacher_target[..., :min_len]
            if mask is not None:
                mask = mask[..., :min_len]

        # Cosine similarity along channels dimension (dim=1)
        cos_sim = F.cosine_similarity(student_proj, teacher_target, dim=1, eps=self.eps)
        # cos_sim has shape (batch, time)
        loss_cos = 1.0 - cos_sim

        # MSE loss
        loss_mse = F.mse_loss(student_proj, teacher_target, reduction="none").mean(
            dim=1
        )

        if mask is not None:
            if mask.ndim == 3 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_frames = mask.sum().clamp_min(1.0)
            loss_cos = (loss_cos * mask).sum() / valid_frames
            loss_mse = (loss_mse * mask).sum() / valid_frames
        else:
            loss_cos = loss_cos.mean()
            loss_mse = loss_mse.mean()

        total_loss = loss_cos + (self.lambda_mse * loss_mse)

        metrics = {
            "loss_distill_total": float(total_loss.detach().cpu()),
            "loss_distill_cos": float(loss_cos.detach().cpu()),
            "loss_distill_mse": float(loss_mse.detach().cpu()),
        }

        return total_loss, metrics
