"""Trainer pipelines for Content Cleanser distillation and Adapter tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.distillation.loss import DistillationLoss, DistillationProjectionHead
from timbre_lite.modules.adapter import FullPersonalizedPipeline
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.training.losses import LatentLoss, MultiScaleSTFTLoss


class DistillationTrainer:
    """Stage 1: Trains Content Cleanser via offline ContentVec teacher distillation.

    Optimizes ContentCleanser and DistillationProjectionHead to predict 768-d
    phonetic features from continuous EnCodec latents while keeping EnCodec frozen.
    """

    def __init__(
        self,
        cleanser: ContentCleanser,
        proj_head: DistillationProjectionHead,
        codec: EnCodec24kCandidate,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize DistillationTrainer.

        Args:
            cleanser: Trainable ContentCleanser module (128 -> 64).
            proj_head: Trainable DistillationProjectionHead (64 -> 768).
            codec: Frozen EnCodec 24kHz candidate model.
            lr: Learning rate for AdamW optimizer.
            weight_decay: Weight decay regularizer.
            device: Target torch compute device.
        """
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.cleanser = cleanser.to(self.device)
        self.proj_head = proj_head.to(self.device)
        self.codec = codec
        self.codec.model.to(self.device)
        self.codec.model.eval()

        self.loss_fn = DistillationLoss().to(self.device)
        self.optimizer = AdamW(
            list(self.cleanser.parameters()) + list(self.proj_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(
        self,
        audio_24k: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform a single distillation forward-backward optimization step.

        Args:
            audio_24k: Input batch of shape (batch, num_samples).
            teacher_features: Target teacher features of shape (batch, 768, num_frames).
            mask: Optional frame valid mask tensor of shape (batch, num_frames).

        Returns:
            Dictionary of step loss metrics.
        """
        self.cleanser.train()
        self.proj_head.train()
        self.optimizer.zero_grad(set_to_none=True)

        audio_24k = audio_24k.to(self.device)
        teacher_features = teacher_features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # 1. Encode into continuous latents with frozen EnCodec encoder
        codec_model: Any = self.codec.model
        with torch.no_grad():
            if audio_24k.ndim == 2:
                audio_input = audio_24k.unsqueeze(1)
            else:
                audio_input = audio_24k
            z = codec_model.encoder(audio_input)  # (batch, 128, time)

        # 2. Cleanser forward
        cleansed = self.cleanser.forward_sequence(z)  # (batch, 64, time)

        # 3. Project to teacher dimension
        proj_student = self.proj_head(cleansed)  # (batch, 768, time)

        # 4. Compute distillation loss
        loss, metrics = self.loss_fn(proj_student, teacher_features, mask=mask)

        # 5. Backpropagation (OOM-safe)
        try:
            loss.backward()
        except RuntimeError as exc:
            if "CUDNN" in str(exc) or "out of memory" in str(exc).lower():
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                return {k: 0.0 for k in metrics}
            raise

        nn.utils.clip_grad_norm_(
            list(self.cleanser.parameters()) + list(self.proj_head.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {k: float(v) for k, v in metrics.items()}


class AdapterTrainer:
    """Stage 2: Target-domain timbre manifold auto-reconstruction trainer.

    Trains the Personalized Adapter, Prosody Head, and Fusion modules using
    target speaker audio (e.g. Hu Tao) via auto-encoding reconstruction:
        y_target -> frozen EnCodec Encoder -> z_target -> Cleanser (bottleneck)
                 -> Adapter + Prosody + Fusion -> z_adapted
                 -> frozen EnCodec Decoder -> y_recon

    Because the frozen Content Cleanser strips source speaker identity and enforces
    a phonetic/content bottleneck, training on target audio forces the downstream
    adapter and fusion layers to reconstruct speech on the target timbre manifold
    conditioned solely on phonetic and pitch representations.

    Note:
        This is an auto-encoding reconstruction objective on the target speaker's
        distribution, NOT a cross-speaker source-to-target paired conversion training.
    """

    def __init__(
        self,
        pipeline: FullPersonalizedPipeline,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize AdapterTrainer.

        Args:
            pipeline: FullPersonalizedPipeline module containing adapter & heads.
            lr: Learning rate for AdamW optimizer.
            weight_decay: Weight decay regularizer.
            device: Target torch device.
        """
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.pipeline = pipeline.to(self.device)
        self.stft_loss = MultiScaleSTFTLoss().to(self.device)
        self.latent_loss = LatentLoss().to(self.device)

        # Train adapter, prosody head, fusion, and optional F0 encoder
        trainable = (
            list(self.pipeline.adapter.parameters())
            + list(self.pipeline.prosody_head.parameters())
            + list(self.pipeline.fusion.parameters())
        )
        if self.pipeline.f0_encoder is not None:
            trainable += list(self.pipeline.f0_encoder.parameters())

        self.optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def train_step(
        self,
        target_audio_24k: torch.Tensor,
        mask: torch.Tensor | None = None,
        f0_3d: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one optimization step on target persona data.

        Args:
            target_audio_24k: Ground-truth target audio of shape (batch, num_samples).
            mask: Optional frame valid mask.
            f0_3d: Optional 3D normalized F0 tensor of shape (batch, 3, num_frames).

        Returns:
            Dictionary of step loss metrics. Returns zeros on OOM skip.
        """
        self.pipeline.train()
        self.optimizer.zero_grad(set_to_none=True)

        audio = target_audio_24k.to(self.device)
        if audio.ndim == 2:
            audio_in = audio.unsqueeze(1)
        else:
            audio_in = audio

        f0_in = f0_3d.to(self.device) if f0_3d is not None else None

        codec_model: Any = self.pipeline.codec.model
        # 1. Encode target audio with frozen EnCodec encoder
        with torch.no_grad():
            z_target = codec_model.encoder(audio_in)

        # 2. Pipeline transformation sequence
        z_adapted, _, _ = self.pipeline.forward_sequence(z_target, f0_seq=f0_in)

        # 3. Decode adapted latent back to audio
        y_recon = codec_model.decoder(z_adapted).squeeze(1)

        # 4. Losses
        loss_stft = self.stft_loss(y_recon, audio)
        loss_lat = self.latent_loss(z_adapted, z_target, mask=mask)
        total_loss = loss_stft + (2.0 * loss_lat)

        # 5. OOM-safe backward: if cuDNN workspace allocation fails mid-backward,
        #    free the fragmented cache and skip the batch cleanly instead of crashing.
        try:
            total_loss.backward()
        except RuntimeError as exc:
            if "CUDNN" in str(exc) or "out of memory" in str(exc).lower():
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                return {
                    "loss_adapter_total": 0.0,
                    "loss_adapter_stft": 0.0,
                    "loss_adapter_latent": 0.0,
                }
            raise

        trainable = (
            list(self.pipeline.adapter.parameters())
            + list(self.pipeline.prosody_head.parameters())
            + list(self.pipeline.fusion.parameters())
        )
        nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_adapter_total": float(total_loss.detach().cpu()),
            "loss_adapter_stft": float(loss_stft.detach().cpu()),
            "loss_adapter_latent": float(loss_lat.detach().cpu()),
        }
