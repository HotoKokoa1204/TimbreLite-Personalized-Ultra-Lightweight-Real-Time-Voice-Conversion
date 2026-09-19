"""Speaker-invariant Content Cleanser module with adversarial GRL learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.autograd import Function

from timbre_lite.modules.causal_layers import CausalDilatedResidualBlock


class GradientReversalFunction(Function):
    """Gradient Reversal Layer (GRL) autograd function.

    Multiplies the gradient by -lambda during backpropagation while acting
    as identity during the forward pass.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward identity mapping.

        Args:
            ctx: Autograd context object.
            x: Input tensor.
            alpha: Scaling factor for reversed gradient.

        Returns:
            Identical input tensor.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, None]:
        """Backward pass negating and scaling the incoming gradient.

        Args:
            ctx: Autograd context object.
            grad_outputs: Upstream gradient tensors.

        Returns:
            Tuple of (reversed_grad, None).
        """
        grad_output: torch.Tensor = grad_outputs[0]
        alpha: float = float(ctx.alpha)
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    """Module wrapper for GradientReversalFunction."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize GRL module.

        Args:
            alpha: Scaling coefficient for reversed gradient.
        """
        super().__init__()
        self.alpha = alpha

    def set_alpha(self, alpha: float) -> None:
        """Dynamically update GRL scaling coefficient.

        Args:
            alpha: New gradient scaling factor.
        """
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRL.

        Args:
            x: Input tensor.

        Returns:
            Tensor with inverted backward gradient hook.
        """
        out: torch.Tensor = GradientReversalFunction.apply(x, self.alpha)
        return out


class CleanserVariant(str, Enum):
    """Ablation variants for Content Cleanser training."""

    VARIANT_A = "variant_a_base"
    VARIANT_B = "variant_b_grl"
    VARIANT_C = "variant_c_perturbation"
    VARIANT_D = "variant_d_full_compound"


@dataclass
class CleanserState:
    """Persistent streaming state for the Content Cleanser module.

    Attributes:
        tcn_states: Convolutional receptive field buffers.
        frame_index: Processed frame counter.
    """

    tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    frame_index: int = 0

    def reset(self) -> None:
        """Reset internal streaming state."""
        self.tcn_states.clear()
        self.frame_index = 0


class ContentCleanser(nn.Module):
    """Causal Content Cleanser mapping codec latents to speaker-cleansed forms.

    Purges source speaker timbre identity from the continuous latent while
    preserving phonetic and linguistic information for downstream personalized
    voice conversion.
    """

    def __init__(
        self,
        in_dim: int = 128,
        content_dim: int = 64,
        tcn_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        """Initialize Content Cleanser module.

        Args:
            in_dim: Input continuous codec latent dimension (128).
            content_dim: Cleansed phonetic representation dimension (64 or 32).
            tcn_layers: Number of dilated causal residual layers.
            kernel_size: Causal convolution kernel size.
        """
        super().__init__()
        self.in_dim = in_dim
        self.content_dim = content_dim
        self.tcn_layers = tcn_layers

        self.in_proj = nn.Conv1d(in_dim, content_dim, kernel_size=1)

        tcn_blocks: list[CausalDilatedResidualBlock] = []
        for i in range(tcn_layers):
            dilation = 2**i
            tcn_blocks.append(
                CausalDilatedResidualBlock(
                    channels=content_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
        self.tcn_blocks: list[CausalDilatedResidualBlock] = tcn_blocks
        self.tcn = nn.ModuleList(tcn_blocks)
        self.out_proj = nn.Conv1d(content_dim, content_dim, kernel_size=1)

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters in this module.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> CleanserState:
        """Initialize clean streaming state.

        Args:
            batch_size: Number of parallel audio streams.
            device: Target torch device.

        Returns:
            Clean CleanserState container.
        """
        dev = device if device is not None else next(self.parameters()).device
        tcn_states = [
            block.init_state(batch_size=batch_size, device=dev)
            for block in self.tcn_blocks
        ]
        return CleanserState(tcn_states=tcn_states, frame_index=0)

    def forward_sequence(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Process full temporal sequence of continuous codec latents.

        Args:
            z_seq: Input continuous latent tensor of shape (batch, in_dim, time).

        Returns:
            Cleansed phonetic content tensor of shape (batch, content_dim, time).
        """
        h: torch.Tensor = self.in_proj(z_seq)
        for block in self.tcn_blocks:
            h = block.forward_sequence(h)
        out: torch.Tensor = self.out_proj(h)
        return out

    def forward_chunk(
        self, z_chunk: torch.Tensor, state: CleanserState | None = None
    ) -> tuple[torch.Tensor, CleanserState]:
        """Streaming chunk forward pass maintaining receptive field history.

        Args:
            z_chunk: Single latent chunk of shape (batch, in_dim, 1).
            state: Previous streaming state.

        Returns:
            Tuple of (out_chunk, next_state) where out_chunk has shape
            (batch, content_dim, 1).
        """
        if state is None:
            state = self.init_state(batch_size=z_chunk.shape[0], device=z_chunk.device)

        h: torch.Tensor = self.in_proj(z_chunk)
        next_tcn_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.tcn_blocks):
            block_state = state.tcn_states[i] if i < len(state.tcn_states) else None
            h, next_s = block.forward_chunk(h, state=block_state)
            next_tcn_states.append(next_s)

        out_chunk: torch.Tensor = self.out_proj(h)
        next_state = CleanserState(
            tcn_states=next_tcn_states,
            frame_index=state.frame_index + 1,
        )
        return out_chunk, next_state


class SpeakerAdversary(nn.Module):
    """GRL-equipped speaker classifier adversary for multi-speaker disentanglement.

    Used during Stage 1 training to penalize speaker identity retention in the
    cleansed content representation.
    """

    def __init__(
        self,
        content_dim: int = 64,
        num_speakers: int = 10,
        hidden_dim: int = 128,
        grl_alpha: float = 1.0,
    ) -> None:
        """Initialize speaker adversary.

        Args:
            content_dim: Dimension of cleansed phonetic representation.
            num_speakers: Total number of distinct training speakers.
            hidden_dim: Classification hidden layer dimension.
            grl_alpha: Initial GRL coefficient.
        """
        super().__init__()
        self.grl = GradientReversal(alpha=grl_alpha)
        self.classifier = nn.Sequential(
            nn.Conv1d(content_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, num_speakers),
        )

    def set_grl_alpha(self, alpha: float) -> None:
        """Update GRL alpha coefficient.

        Args:
            alpha: New scaling factor.
        """
        self.grl.set_alpha(alpha)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """Compute speaker classification logits with gradient reversal.

        Args:
            content: Cleansed content tensor of shape (batch, content_dim, time).

        Returns:
            Speaker classification logits of shape (batch, num_speakers).
        """
        reversed_content = self.grl(content)
        logits: torch.Tensor = self.classifier(reversed_content)
        return logits


class PhoneticPredictor(nn.Module):
    """Auxiliary phonetic classification head to ensure zero phonetic collapse."""

    def __init__(
        self,
        content_dim: int = 64,
        num_phonetic_classes: int = 42,
    ) -> None:
        """Initialize phonetic predictor.

        Args:
            content_dim: Dimension of cleansed representation.
            num_phonetic_classes: Number of phonetic targets (e.g. 42 phonemes).
        """
        super().__init__()
        self.head = nn.Conv1d(content_dim, num_phonetic_classes, kernel_size=1)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """Predict phonetic frame logits.

        Args:
            content: Content tensor of shape (batch, content_dim, time).

        Returns:
            Phonetic logits of shape (batch, num_phonetic_classes, time).
        """
        logits: torch.Tensor = self.head(content)
        return logits


class SpeakerVerificationProbe(nn.Module):
    """Objective probe measuring source speaker leakage in representations."""

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 64,
    ) -> None:
        """Initialize speaker verification probe.

        Args:
            feature_dim: Input feature dimension (e.g. 128 for raw, 64 for cleansed).
            embedding_dim: Projected speaker embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(feature_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized speaker embedding vector.

        Args:
            x: Feature tensor of shape (batch, feature_dim, time).

        Returns:
            Normalized speaker embedding tensor of shape (batch, embedding_dim).
        """
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=-1)

    def compute_cosine_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Calculate pairwise cosine similarity between two utterance sets.

        Args:
            x1: First feature tensor of shape (batch, feature_dim, time).
            x2: Second feature tensor of shape (batch, feature_dim, time).

        Returns:
            Mean cosine similarity scalar.
        """
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        sim: torch.Tensor = (emb1 * emb2).sum(dim=-1)
        return float(sim.mean().item())
