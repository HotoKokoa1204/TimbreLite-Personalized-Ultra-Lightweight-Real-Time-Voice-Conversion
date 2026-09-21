"""Loss objectives for audio waveform synthesis and latent representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class STFTLoss(nn.Module):
    """Single-scale STFT loss combining spectral convergence and log magnitude."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 120,
        win_length: int = 600,
    ) -> None:
        """Initialize single-scale STFT loss.

        Args:
            n_fft: FFT window size.
            hop_length: Hop length between analysis frames.
            win_length: Analysis window length.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate spectral convergence and log STFT magnitude loss.

        Args:
            x: Reconstructed audio tensor of shape (batch, samples).
            y: Ground-truth audio tensor of shape (batch, samples).

        Returns:
            Tuple of (spectral_convergence_loss, log_stft_magnitude_loss).
        """
        window: torch.Tensor = getattr(self, "window")
        if window.device != x.device:
            window = window.to(device=x.device)

        # Min length matching
        min_len = min(x.shape[-1], y.shape[-1])
        x = x[..., :min_len]
        y = y[..., :min_len]

        x_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        y_stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        x_mag = torch.abs(x_stft) + 1e-7
        y_mag = torch.abs(y_stft) + 1e-7

        # Spectral convergence
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (
            torch.norm(y_mag, p="fro") + 1e-7
        )

        # Log STFT magnitude loss
        log_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))

        return sc_loss, log_loss


class MultiScaleSTFTLoss(nn.Module):
    """Multi-Scale STFT Loss measuring audio fidelity across frequency bands."""

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (50, 120, 240),
        win_lengths: tuple[int, ...] = (240, 600, 1200),
    ) -> None:
        """Initialize multi-scale STFT loss.

        Args:
            fft_sizes: Tuple of FFT window sizes.
            hop_sizes: Tuple of hop lengths.
            win_lengths: Tuple of window lengths.
        """
        super().__init__()
        self.losses = nn.ModuleList(
            [
                STFTLoss(n_fft=f, hop_length=h, win_length=w)
                for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate average multi-scale spectral loss.

        Args:
            x: Reconstructed audio tensor of shape (batch, samples).
            y: Target audio tensor of shape (batch, samples).

        Returns:
            Scalar loss tensor.
        """
        total_sc = torch.tensor(0.0, device=x.device)
        total_log = torch.tensor(0.0, device=x.device)

        for loss_fn in self.losses:
            sc, log_mag = loss_fn(x, y)
            total_sc = total_sc + sc
            total_log = total_log + log_mag

        return (total_sc + total_log) / len(self.losses)


class LatentLoss(nn.Module):
    """Reconstruction loss for continuous latent representations."""

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0) -> None:
        """Initialize LatentLoss.

        Args:
            l1_weight: Weight scalar for L1 loss.
            l2_weight: Weight scalar for MSE / L2 loss.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        pred_z: torch.Tensor,
        target_z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined L1 and L2 reconstruction loss.

        Args:
            pred_z: Predicted latent tensor (batch, channels, time).
            target_z: Target latent tensor (batch, channels, time).
            mask: Optional loss mask tensor.

        Returns:
            Scalar loss tensor.
        """
        min_len = min(pred_z.shape[-1], target_z.shape[-1])
        pred_z = pred_z[..., :min_len]
        target_z = target_z[..., :min_len]

        l1 = F.l1_loss(pred_z, target_z, reduction="none")
        l2 = F.mse_loss(pred_z, target_z, reduction="none")

        if mask is not None:
            mask = mask[..., :min_len].unsqueeze(1)
            valid = mask.sum().clamp_min(1.0) * pred_z.shape[1]
            loss_l1 = (l1 * mask).sum() / valid
            loss_l2 = (l2 * mask).sum() / valid
        else:
            loss_l1 = l1.mean()
            loss_l2 = l2.mean()

        return (self.l1_weight * loss_l1) + (self.l2_weight * loss_l2)
